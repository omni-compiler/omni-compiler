package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import exc.openmp.OMPpragma;
import java.util.*;
import xcodeml.util.XmOption;

public class XMPtranslateLocalPragma {
  private XMPglobalDecl		_globalDecl;
  private boolean		_all_profile = false;
  private boolean		_selective_profile = false;
  private boolean		doScalasca = false;
  private boolean		doTlog = false;
  private XobjectDef		currentDef;
  private XMPgenSym             tmpSym;

  public XMPtranslateLocalPragma(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
    tmpSym = new XMPgenSym();
  }

  public void translate(FuncDefBlock def) {
    FunctionBlock fb = def.getBlock();
    currentDef = def.getDef();

    // first, check static_desc
    BlockIterator i = new topdownBlockIterator(fb);
    for (i.init(); !i.end(); i.next()){
      Block b = i.getBlock();
      if (b.Opcode() == Xcode.XMP_PRAGMA){
	String pragmaName = ((PragmaBlock)b).getPragma();
	if (XMPpragma.valueOf(pragmaName) == XMPpragma.STATIC_DESC){
	  analyzeStaticDesc((PragmaBlock)b);
	  b.remove();
	}
      }
    }

    // first, skip tasks
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() ==  Xcode.XMP_PRAGMA) {
        PragmaBlock pb = (PragmaBlock)b;

        try {
	  translatePragma(pb);
        } catch (XMPexception e) {
          XMP.error(pb.getLineNo(), e.getMessage());
        }
      }
    }

    // next, remove tasks
    for (i.init(); !i.end(); i.next()){
      Block b = i.getBlock();
      if (b.Opcode() == Xcode.XMP_PRAGMA){
	String pragmaName = ((PragmaBlock)b).getPragma();
	if (XMPpragma.valueOf(pragmaName) == XMPpragma.TASKS){
	  b.replace(Bcons.COMPOUND(b.getBody()));
	}
      }
    }

    def.finalizeBlock();
  }

  private void translatePragma(PragmaBlock pb) throws XMPexception {
    String pragmaName = pb.getPragma();

    switch (XMPpragma.valueOf(pragmaName)) {
      case NODES:
        { translateNodes(pb);			break; }
      case TEMPLATE:
        { translateTemplate(pb);		break; }
      case DISTRIBUTE:
        { translateDistribute(pb);		break; }
      case ALIGN:
        { translateAlign(pb);			break; }
      case SHADOW:
        { translateShadow(pb);			break; }
      case STATIC_DESC:
	{ /* do nothing */                      break; }
      case TASK:
        { translateTask(pb);			break; }
      case TASKS:
        { translateTasks(pb);			break; }
      case LOOP:
        { translateLoop(pb,true);		break; }
      case REFLECT:
        { translateReflect(pb);			break; }
      case REDUCE_SHADOW:
        { translateReduceShadow(pb);		break; }
      case BARRIER:
        { translateBarrier(pb);			break; }
      case REDUCTION:
        { translateReduction(pb);		break; }
      case BCAST:
        { translateBcast(pb);			break; }
      case GMOVE:
        { translateGmove(pb);			break; }
      case ARRAY:
	{ translateArray(pb);                   break; }
      case POST:
        { translatePost(pb);                    break; }
      case WAIT:
        { translateWait(pb);                    break; }
      case LOCK:
        { translateLockUnlock(pb, "_XMP_lock_");   break; }
      case UNLOCK:
        { translateLockUnlock(pb, "_XMP_unlock_"); break; }
      case LOCAL_ALIAS:
        { translateLocalAlias(pb);		break; }
      case WAIT_ASYNC:
	{ translateWaitAsync(pb);               break; }
      case TEMPLATE_FIX:
	{ translateTemplateFix(pb);             break; }
      case REFLECT_INIT:
        { translateReflectInit(pb);             break; }
      case REFLECT_DO:
      { translateReflectDo(pb);                 break; }
      case GPU_REPLICATE:
        { translateGpuData(pb);			break; }
      case GPU_REPLICATE_SYNC:
        { translateGpuSync(pb);			break; }
      case GPU_REFLECT:
        { translateGpuReflect(pb);		break; }
      case GPU_BARRIER:
        { break; }
      case GPU_LOOP:
        { translateGpuLoop(pb);			break; }
      default:
        throw new XMPexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }

  private void addProfileFunctions(Xobject profileClause, Block funcCallBlock, String directiveName, PragmaBlock pb) throws XMPexception{
    
    boolean isProfile = false;
    if(profileClause != null)
      if(profileClause.Opcode() == Xcode.VAR && profileClause.getName() == "profile")
        isProfile = true;
    isProfile = isProfile && _selective_profile;
    
    if(_all_profile || isProfile){
      if(doScalasca == true){
        String lowerName = directiveName.toLowerCase();
        XobjList profileFuncArgs = Xcons.List(Xcons.StringConstant("#xmp " + lowerName + ":" + pb.getLineNo()));
        funcCallBlock.insert(createScalascaStartProfileCall(profileFuncArgs));
        funcCallBlock.add(createScalascaEndProfileCall(profileFuncArgs));
      }
      else if(doTlog == true){
        String upperName = directiveName.toUpperCase();
        funcCallBlock.insert(createTlogMacroInvoke("_XMP_M_TLOG_" + upperName + "_IN", null));
        funcCallBlock.add(createTlogMacroInvoke("_XMP_M_TLOG_" + upperName + "_OUT", null));
      }
    }
    else if(isProfile && doTlog == false){
      funcCallBlock.insert(createScalascaProfileOffCall(null));
      funcCallBlock.add(createScalascaProfileOnfCall(null));
    }
  }
  
  private void translateNodes(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);

    XobjList nodesDecl     = (XobjList)pb.getClauses();
    XobjList nodesNameList = (XobjList)nodesDecl.getArg(0);
    XobjList nodesDeclCopy = (XobjList)nodesDecl.copy();

    Iterator<Xobject> iter = nodesNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      nodesDeclCopy.setArg(0, x);
      XMPnodes.translateNodes(nodesDeclCopy, _globalDecl, true, pb);
    }
  }

  private void translateTemplate(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);

    XobjList templateDecl = (XobjList)pb.getClauses();
    XobjList templateNameList = (XobjList)templateDecl.getArg(0);
    XobjList templateDeclCopy = (XobjList)templateDecl.copy();

    Iterator<Xobject> iter = templateNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      templateDeclCopy.setArg(0, x);
      XMPtemplate.translateTemplate(templateDeclCopy, _globalDecl, true, pb);
    }
  }

  private void translateTemplateFix(PragmaBlock pb) throws XMPexception 
  {
    XobjList templateDecl = (XobjList)pb.getClauses();
    XobjList templateNameList = (XobjList)templateDecl.getArg(1);
    XobjList templateDeclCopy = (XobjList)templateDecl.copy();

    Iterator<Xobject> iter = templateNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      templateDeclCopy.setArg(1, x);
      XMPtemplate.translateTemplateFix(templateDeclCopy, _globalDecl, pb);
    }
  }

  private void translateReflectInit(PragmaBlock pb) throws XMPexception 
  {
    XobjList funcArgs     = (XobjList)pb.getClauses().getArg(0);
    XobjList widthList    = (XobjList)pb.getClauses().getArg(1);
    XobjList acc_or_host1 = (XobjList)pb.getClauses().getArg(2);
    XobjList acc_or_host2 = (XobjList)pb.getClauses().getArg(3);
    BlockList funcBody    = Bcons.emptyBody();

    boolean isHost = false;
    boolean isAcc  = false;

    if(acc_or_host1.Nargs() == 0 && acc_or_host2.Nargs() == 0){
      isHost = true;
    }
    else{
      if(acc_or_host1.Nargs() != 0){
        if(acc_or_host1.getArg(0).getName() == "acc"){
          isAcc = true;
        }
        else if(acc_or_host1.getArg(0).getName() == "host"){
          isHost = true;
        }
      }
      if(acc_or_host2.Nargs() != 0){
        if(acc_or_host2.getArg(0).getName() == "acc"){
          isAcc = true;
        }
        else if(acc_or_host2.getArg(0).getName() == "host"){
          isHost = true;
        }
      }
    }

    if(isHost){
      XMP.fatal("reflect_init for host has been not developed yet.");
    }

    Ident funcIdAcc = _globalDecl.declExternFunc("_XMP_reflect_init_acc");

    // if(widthList.Nargs() != 0){
    //   XMP.fatal("width clause in reflect_init has been not developed yet.");
    // }

    XobjList args = Xcons.List();
    args.add(Xcons.String("USE_DEVICE"));
    for(int i=0;i<funcArgs.Nargs();i++){
      Xobject array = funcArgs.getArg(i);
      String arrayName = array.getString();
      Ident arrayId = _globalDecl.findVarIdent(XMP.ADDR_PREFIX_ + arrayName);
      args.add(Xcons.List(arrayId.Ref()));

      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, pb);
      if(alignedArray == null){
        XMP.fatal(arrayName + " is not aligned.");
      }
      if(!alignedArray.hasShadow()){
        XMP.fatal(arrayName + " is not shadowed.");
      }

      for (int j = 0; j < widthList.Nargs(); j++){
	  XobjList width = (XobjList)widthList.getArg(j);

	  // Here the stride means the periodic flag.
	  // check wheter the shadow is full.
	  if (width.getArg(2).getInt() == 1 && alignedArray.getShadowAt(j).getType() == XMPshadow.SHADOW_FULL){
	      throw new XMPexception("Periodic reflect cannot be specified for a dimension with full shadow.");
	  }

	  Ident funcId = _globalDecl.declExternFunc("_XMP_set_reflect_acc__");
	  XobjList setFuncArgs = Xcons.List(alignedArray.getDescId().Ref(), Xcons.IntConstant(j),
					    width.getArg(0), width.getArg(1), width.getArg(2));
	  funcBody.add(Bcons.Statement(funcId.Call(setFuncArgs)));
      }
      
      Ident arrayDesc = _globalDecl.findVarIdent(XMP.DESC_PREFIX_ + arrayName);
      funcBody.add(Bcons.Statement(funcIdAcc.Call(Xcons.List(array, arrayDesc.Ref()))));
    }
   
    Block funcCallBlock = Bcons.PRAGMA(Xcode.ACC_PRAGMA, "HOST_DATA", (Xobject)Xcons.List(args), funcBody);

    pb.replace(funcCallBlock);
  }

  private void translateReflectDo(PragmaBlock pb) throws XMPexception
  {
    XobjList funcArgs     = (XobjList)pb.getClauses().getArg(0);
    XobjList acc_or_host1 = (XobjList)pb.getClauses().getArg(1);
    XobjList acc_or_host2 = (XobjList)pb.getClauses().getArg(2);
    boolean isHost = false;
    boolean isAcc  = false;

    if(acc_or_host1.Nargs() == 0 && acc_or_host2.Nargs() == 0){
      isHost = true;
    }
    else{
      if(acc_or_host1.Nargs() != 0){
        if(acc_or_host1.getArg(0).getName() == "acc"){
          isAcc = true;
        }
        else if(acc_or_host1.getArg(0).getName() == "host"){
          isHost = true;
        }
      }
      if(acc_or_host2.Nargs() != 0){
        if(acc_or_host2.getArg(0).getName() == "acc"){
          isAcc = true;
        }
        else if(acc_or_host2.getArg(0).getName() == "host"){
          isHost = true;
        }
      }
    }

    if(isHost){
      XMP.fatal("reflect_do for host has been not developed yet.");
    }

    Ident funcIdAcc = _globalDecl.declExternFunc("_XMP_reflect_do_acc");
    Block funcBody  = Bcons.emptyBlock();
    
    for(int i=0;i<funcArgs.Nargs();i++){
      Xobject array = funcArgs.getArg(i);
      String arrayName = array.getString();

      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, pb);
      if(alignedArray == null){
        XMP.fatal(arrayName + " is not aligned.");
      }
      if(!alignedArray.hasShadow()){
        XMP.fatal(arrayName + " is not shadowed.");
      }
      Ident arrayDesc = _globalDecl.findVarIdent(XMP.DESC_PREFIX_ + arrayName);
      funcBody.add(funcIdAcc.Call(Xcons.List(arrayDesc.Ref())));
    }

    pb.replace(funcBody);
  }
  
  private void translateDistribute(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);

    XobjList distributeDecl = (XobjList)pb.getClauses();
    XobjList distributeNameList = (XobjList)distributeDecl.getArg(0);
    XobjList distributeDeclCopy = (XobjList)distributeDecl.copy();

    Iterator<Xobject> iter = distributeNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      distributeDeclCopy.setArg(0, x);
      XMPtemplate.translateDistribute(distributeDeclCopy, _globalDecl, true, pb);
    }
  }

  private void translateAlign(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);

    XobjList alignDecl             = (XobjList)pb.getClauses();
    XobjList alignNameList         = (XobjList)alignDecl.getArg(0);
    XobjList alignSubscriptList    = (XobjList)alignDecl.getArg(3);
    XobjList alignSubscriptVarList = (XobjList)alignSubscriptList.left();
    
    String kind_bracket = alignSubscriptList.getTail().getString();
    boolean isSquare    = kind_bracket.equals("SQUARE");
    alignSubscriptList.removeLastArgs(); // Remove information of ROUND or SQUARE
    if(isSquare) alignSubscriptVarList.reverse();

    XobjList alignDeclCopy = (XobjList)alignDecl.copy();
    Iterator<Xobject> iter = alignNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      alignDeclCopy.setArg(0, x);
      XMPalignedArray.translateAlign(alignDeclCopy, _globalDecl, true, pb);
    }
  }

  private void translateLockUnlock(PragmaBlock pb, String funcNamePrefix) throws XMPexception {
    XobjList lockDecl   = (XobjList)pb.getClauses();
    XobjList lockObjVar = (XobjList)lockDecl.getArg(0);
    String coarrayName  = XMPutil.getXobjSymbolName(lockObjVar.getArg(0));
    XMPcoarray coarray  = _globalDecl.getXMPcoarray(coarrayName);
    if(coarray == null)
      throw new XMPexception("Variable in #pragma xmp lock() must be coarray");
    
    // When lockDecl.Nargs() is 1,
    // The specified lock object does not have a codimension.
    // e.g.) #pragma xmp lock(lock_obj)
    int imageDims = 0;
    if(lockDecl.Nargs() != 1)
      imageDims = lockDecl.getArg(1).Nargs();
    
    if(imageDims != 0)
      if(lockDecl.getArg(1).Nargs() != coarray.getImageDim())
        throw new XMPexception("Invalid number of dimensions of '" + coarrayName + "'");

    // Set descriptor of lock object
    XobjList funcArgs = Xcons.List();
    funcArgs.add(Xcons.SymbolRef(coarray.getDescId()));

    // Set offset
    //  e.g. xmp_lock_t lockobj[a][b][c]:[*]; #pragma xmp lock(lockobj[3][2][1]:[x]) -> offset = 3*b*c + 2*c + 1
    int arrayDims = lockObjVar.Nargs() - 1;
    Xobject offset = null;
    for(int i=0;i<arrayDims;i++){
      Xobject tmp = lockObjVar.getArg(i+1);
      for(int j=i+1;j<arrayDims;j++){
        tmp = Xcons.binaryOp(Xcode.MUL_EXPR, tmp, Xcons.IntConstant((int)coarray.getSizeAt(j)));
      }
      if(offset == null)
        offset = tmp;
      else
        offset = Xcons.binaryOp(Xcode.PLUS_EXPR, offset, tmp);
    }
    if(offset == null)
      funcArgs.add(Xcons.IntConstant(0));
    else
      funcArgs.add(offset);

    if(imageDims != 0)
      for(int i=0;i<lockDecl.getArg(1).Nargs();i++)
        funcArgs.add(lockDecl.getArg(1).getArg(i));

    String funcName = funcNamePrefix + String.valueOf(imageDims);
    pb.replace(_globalDecl.createFuncCallBlock(funcName, funcArgs));
  }

  private void translatePost(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);
    XobjList args = null;
    XobjList postDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);

    XobjList onRef = (XobjList)postDecl.getArg(0);
    String nodeName = onRef.getArg(0).getString();

    XMPnodes nodeObj = _globalDecl.getXMPnodes(nodeName, pb);
    if(nodeObj == null){
      throw new XMPexception("cannot find '" + nodeName + "' nodes");
    }
    args = Xcons.List(nodeObj.getDescId().Ref());
    
    XobjList nodeList   = (XobjList)onRef.getArg(1);
    String kind_bracket = nodeList.getTail().getString();
    boolean isSquare    = kind_bracket.equals("SQUARE");

    nodeList.removeLastArgs(); // Remove information of ROUND or SQUARE
    if(isSquare) nodeList.reverse();

    if(nodeObj.getDim() != nodeList.Nargs()){
      throw new XMPexception("Error. Dimension of node is different.");
    }

    String funcName = "_XMP_post_" + String.valueOf(nodeObj.getDim());

    for(int i=0;i<nodeObj.getDim();i++){
      Xobject v = nodeList.getArg(i).getArg(0);
      if(isSquare)
        v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, Xcons.IntConstant(1));

      args.add(v);
    }

    Xobject tag = postDecl.getArg(1);
    args.add(tag);

    pb.replace(_globalDecl.createFuncCallBlock(funcName, args));
  }

  private void translateWait(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);
    XobjList waitDecl = (XobjList)pb.getClauses();
    int numOfArgs = waitDecl.Nargs();

    // no arguments
    if(numOfArgs == 0){
      pb.replace(_globalDecl.createFuncCallBlock("_XMP_wait_noargs", null));
      return;
    }

    // only node
    XobjList onRef      = (XobjList)waitDecl.getArg(0);
    String nodeName     = onRef.getArg(0).getString();
    XobjList nodeList   = (XobjList)onRef.getArg(1);
    XMPnodes nodeObj    = _globalDecl.getXMPnodes(nodeName, pb);
    String funcName     = null;
    XobjList args       = Xcons.List(nodeObj.getDescId().Ref());
    String kind_bracket = nodeList.getTail().getString();
    boolean isSquare    = kind_bracket.equals("SQUARE");
    nodeList.removeLastArgs(); // Remove information of ROUND or SQUARE
    if(isSquare) nodeList.reverse();

    if(nodeObj == null){
      throw new XMPexception("cannot find '" + nodeName + "' nodes");
    }
    if(nodeObj.getDim() != nodeList.Nargs()){
      throw new XMPexception("Error. Dimension of node is different.");
    }

    for(int i=0;i<nodeObj.getDim();i++){
      Xobject v = nodeList.getArg(i);
      if(isSquare)
        v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, Xcons.IntConstant(1));
      
      args.add(v);
    }

    if(numOfArgs == 1){
      funcName = "_XMP_wait_node_" + String.valueOf(nodeObj.getDim());
      pb.replace(_globalDecl.createFuncCallBlock(funcName, args));
      return;
    }
    
    // node and tag
    if(numOfArgs == 2){ // node and tag
      funcName = "_XMP_wait_" + String.valueOf(nodeObj.getDim());
      Xobject tag = waitDecl.getArg(1);
      args.add(tag);
      pb.replace(_globalDecl.createFuncCallBlock(funcName, args));
      return;
    }
  }

  private void translateLocalAlias(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);
    XMPalignedArray.translateLocalAlias((XobjList)pb.getClauses(), _globalDecl, true, pb);
  }

  private void translateWaitAsync(PragmaBlock pb) throws XMPexception {

    BlockList ret_body = Bcons.emptyBody();

    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(pb.getClauses().getArg(1), _globalDecl, pb);
    Xobject on_ref_arg;
    if (on_ref != null){
      ret_body.add(on_ref.buildConstructor(_globalDecl));
      on_ref_arg = on_ref.getDescId().Ref();
    }
    else on_ref_arg = Xcons.Cnull();

    Ident f = _globalDecl.declExternFunc("xmpc_wait_async");

    XobjList funcArgs = (XobjList)pb.getClauses().getArg(0);
    for (Xobject i: funcArgs){
      ret_body.add(f.Call(Xcons.List(i, on_ref_arg)));
    }

    pb.replace(Bcons.COMPOUND(ret_body));
    
    // Ident funcId = _globalDecl.declExternFunc("_XMP_wait_async__");
    // XobjList funcArgs = (XobjList)pb.getClauses().getArg(0);
    // BlockList funcBody = Bcons.emptyBody();
    // for(Xobject i: funcArgs){
    //   funcBody.add(Bcons.Statement(funcId.Call(Xcons.List(i))));
    // }

    // Block funcCallBlock = Bcons.COMPOUND(funcBody);

    // // the following code comes from translateBcast.
    // XobjList onRef = (XobjList)pb.getClauses().getArg(1);
    // if(onRef != null && onRef.getArgs() != null){
    //   XMPquadruplet<String, Boolean, XobjList, XMPobject> execOnRefArgs = createExecOnRefArgs(onRef, pb);
    //   String execFuncSurfix = execOnRefArgs.getFirst();
    //   boolean splitComm = execOnRefArgs.getSecond().booleanValue();
    //   XobjList execFuncArgs = execOnRefArgs.getThird();
    //   if (splitComm) {
    //     BlockList waitAsyncBody = Bcons.blockList(funcCallBlock);
    //     funcCallBlock = createCommTaskBlock(waitAsyncBody, execFuncSurfix, execFuncArgs);
    //   }
    // }

    // BlockList bl = funcCallBlock.getBody();
    // for(Xobject i: funcArgs){
    //   bl.add(_globalDecl.declExternFunc("xmpc_end_async").Call(Xcons.List(i)));
    // }

    // pb.replace(funcCallBlock);
  }

  private void translateShadow(PragmaBlock pb) throws XMPexception {
    checkDeclPragmaLocation(pb);

    XobjList shadowDecl = (XobjList)pb.getClauses();
    XobjList shadowNameList = (XobjList)shadowDecl.getArg(0);
    XobjList shadowDeclCopy = (XobjList)shadowDecl.copy();

    Iterator<Xobject> iter = shadowNameList.iterator();
    while (iter.hasNext()) {
      Xobject x = iter.next();
      shadowDeclCopy.setArg(0, x);
      XMPshadow.translateShadow(shadowDeclCopy, _globalDecl, true, pb);
    }
  }

  private void translateReflect(PragmaBlock pb) throws XMPexception {
    XobjList accOrHost = (XobjList)pb.getClauses().getArg(3);
    boolean isACC = accOrHost.hasIdent("acc");
    Block reflectFuncCallBlock = XMPshadow.translateReflect(pb, _globalDecl, isACC);

    Xobject profileClause = pb.getClauses().getArg(4);
    addProfileFunctions(profileClause, reflectFuncCallBlock, "reflect", pb);
  }

  private void translateReduceShadow(PragmaBlock pb) throws XMPexception {
    XobjList accOrHost = (XobjList)pb.getClauses().getArg(3);
    boolean isACC = accOrHost.hasIdent("acc");
    Block reflectFuncCallBlock = XMPshadow.translateReduceShadow(pb, _globalDecl, isACC);

    Xobject profileClause = pb.getClauses().getArg(4);
    addProfileFunctions(profileClause, reflectFuncCallBlock, "reflect", pb);
  }

  static Block encloseWithAccHostDataDirective(Block block, XobjList useDeviceClauseArgs) throws XMPexception {
    if(useDeviceClauseArgs == null || useDeviceClauseArgs.isEmpty()){
      throw new XMPexception("empty argument for use_device clause");
    }
    return Bcons.PRAGMA(
            Xcode.ACC_PRAGMA,
            "HOST_DATA",
            Xcons.List(Xcons.List(Xcons.String("USE_DEVICE"), useDeviceClauseArgs)),
            Bcons.blockList(block));
  }

  private void translateGpuData(PragmaBlock pb) throws XMPexception {
    XMPgpuData.translateGpuData(pb, _globalDecl);
  }

  private void translateGpuSync(PragmaBlock pb) throws XMPexception {
    XMPgpuData.translateGpuSync(pb, _globalDecl);
  }

  private void translateGpuReflect(PragmaBlock pb) throws XMPexception {
    if (!XmOption.isXcalableMPGPU()) {
      XMP.warning("use -enable-gpu option to enable 'acc relfect' directive");
      translateReflect(pb);
    }
    else {
      XMPshadow.translateGpuReflect(pb, _globalDecl);
    }
  }

  private void translateGpuLoop(PragmaBlock pb) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();
    BlockList loopBody = pb.getBody();

    if (!XmOption.isXcalableMPGPU()) {
      XMP.warning("use -enable-gpu option to use 'acc loop' directive");
      pb.replace(Bcons.COMPOUND(loopBody));
      return;
    }

    // get block to schedule
    CforBlock schedBaseBlock = getOutermostLoopBlock(loopBody);

    // analyze loop
    XobjList loopIterList = (XobjList)loopDecl.getArg(0);
    if (loopIterList == null || loopIterList.Nargs() == 0) {
      loopIterList = Xcons.List(Xcons.String(schedBaseBlock.getInductionVar().getName()));
      loopDecl.setArg(0, loopIterList);
      translateFollowingLoop(pb, schedBaseBlock);
    } else {
      translateMultipleLoop(pb, schedBaseBlock);
    }

    // translate
    // FIXME implement reduction
    Block newLoopBlock = translateGpuClause(pb, null, schedBaseBlock);
    schedBaseBlock.replace(newLoopBlock);

    // replace pragma
    Block loopFuncCallBlock = Bcons.COMPOUND(loopBody);
    pb.replace(loopFuncCallBlock);
  }

  private void analyzeFollowingGpuLoop(PragmaBlock pb, CforBlock schedBaseBlock) throws XMPexception {
    // schedule loop
    analyzeGpuLoop(pb, schedBaseBlock, schedBaseBlock);
  }

  private void analyzeMultipleGpuLoop(PragmaBlock pb, CforBlock schedBaseBlock) throws XMPexception {
    // start translation
    XobjList loopDecl = (XobjList)pb.getClauses();
    BlockList loopBody = pb.getBody();

    // iterate index variable list
    XobjList loopVarList = (XobjList)loopDecl.getArg(0);
    Vector<CforBlock> loopVector = new Vector<CforBlock>(XMPutil.countElmts(loopVarList));
    for (XobjArgs i = loopVarList.getArgs(); i != null; i = i.nextArgs()) {
      loopVector.add(findLoopBlock(loopBody, i.getArg().getString()));
    }

    // schedule loop
    Iterator<CforBlock> it = loopVector.iterator();
    while (it.hasNext()) {
      CforBlock forBlock = it.next();
      analyzeGpuLoop(pb, forBlock, schedBaseBlock);
    }
  }

  private void analyzeGpuLoop(PragmaBlock pb, CforBlock forBlock, CforBlock schedBaseBlock) throws XMPexception {
    Xobject loopIndex = forBlock.getInductionVar();
    String loopIndexName = loopIndex.getSym();

    Ident initId = declIdentWithBlock(schedBaseBlock,
                                      "_XMP_loop_init_" + loopIndexName, Xtype.intType);
    Ident condId = declIdentWithBlock(schedBaseBlock,
                                      "_XMP_loop_cond_" + loopIndexName, Xtype.intType);
    Ident stepId = declIdentWithBlock(schedBaseBlock,
                                      "_XMP_loop_step_" + loopIndexName, Xtype.intType);

    schedBaseBlock.insert(Xcons.Set(initId.Ref(), forBlock.getLowerBound()));
    schedBaseBlock.insert(Xcons.Set(condId.Ref(), forBlock.getUpperBound()));
    schedBaseBlock.insert(Xcons.Set(stepId.Ref(), forBlock.getStep()));

    XMPutil.putLoopIter(schedBaseBlock, loopIndexName, Xcons.List(initId, condId, stepId));
  }

  // private void translateTask(PragmaBlock pb) throws XMPexception {
  //   // start translation
  //   XobjList taskDecl = (XobjList)pb.getClauses();
  //   XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
  //   BlockList taskBody = pb.getBody();

  //   // check if enclosed by TASKS
  //   Block parentBlock = pb.getParentBlock();
  //   boolean tasksFlag = false;
  //   if (parentBlock != null && parentBlock.Opcode() == Xcode.XMP_PRAGMA){
  //     String pragmaName = ((PragmaBlock)parentBlock).getPragma();
  //     if (XMPpragma.valueOf(pragmaName) == XMPpragma.TASKS){
  // 	tasksFlag = true;
  //     }
  //   }

  //   boolean nocomm_flag = (((XobjInt)taskDecl.getArg(1)).getInt() == 1);

  //   // create function arguments
  //   XobjList onRef = (XobjList)taskDecl.getArg(0);
  //   XMPquadruplet<String, Boolean, XobjList, XMPobject> execOnRefArgs = createExecOnRefArgs(onRef, pb);
  //   String execFuncSuffix = execOnRefArgs.getFirst();
  //   if (nocomm_flag) execFuncSuffix = execFuncSuffix + "_nocomm";
  //   XobjList execFuncArgs = execOnRefArgs.getThird();
  //   XMPobject onRefObject = execOnRefArgs.getForth();

  //   // setup task finalizer
  //   if (!nocomm_flag){
  //     Ident finFuncId = _globalDecl.declExternFunc("_XMP_end_task");
  //     setupFinalizer(taskBody, finFuncId, null);
  //   }

  //   // create function call
  //   BlockList taskFuncCallBlockList = Bcons.emptyBody();

  //   Ident taskDescId = null;
  //   if(!nocomm_flag){
  //     if(tasksFlag == true){
  // 	taskDescId = parentBlock.getBody().declLocalIdent(tmpSym.getStr("_XMP_TASK_desc"),
  // 							  Xtype.voidPtrType, StorageClass.AUTO,
  // 							  Xcons.Cast(Xtype.voidPtrType,
  // 								     Xcons.IntConstant(0)));
  //     }
  //     else {
  // 	taskDescId = taskFuncCallBlockList.declLocalIdent(tmpSym.getStr("_XMP_TASK_desc"),
  // 							  Xtype.voidPtrType, StorageClass.AUTO,
  // 							  Xcons.Cast(Xtype.voidPtrType,
  // 								     Xcons.IntConstant(0)));
  //     }
  //     execFuncArgs.cons(taskDescId.getAddr());
  //   }

  //   Ident execFuncId = _globalDecl.declExternFunc("_XMP_exec_task_" + execFuncSuffix, Xtype.intType);

  //   Block taskFuncCallBlock;
  //   if (tasksFlag == true){
  //     Ident flag = parentBlock.getBody().declLocalIdent(tmpSym.getStr("_XMP_is_member"),
  // 							Xtype.intType);
  //     parentBlock.getBody().insert(Xcons.Set(flag.Ref(), execFuncId.Call(execFuncArgs)));
  //     taskFuncCallBlock = Bcons.IF(BasicBlock.Cond(flag.Ref()), taskBody, null);
  //   }
  //   else {
  //     taskFuncCallBlock = Bcons.IF(BasicBlock.Cond(execFuncId.Call(execFuncArgs)),
  // 				   taskBody, null);
  //   }

  //   taskFuncCallBlockList.add(taskFuncCallBlock);
  //   pb.replace(Bcons.COMPOUND(taskFuncCallBlockList));

  //   if (!nocomm_flag){
  //     XobjList arg = Xcons.List(Xcode.POINTER_REF, taskDescId.Ref());
  //     taskFuncCallBlockList.add(_globalDecl.createFuncCallBlock("_XMP_exec_task_NODES_FINALIZE", arg));
  //   }

  //   Xobject profileClause = taskDecl.getArg(3);
  //   addProfileFunctions(profileClause, taskFuncCallBlock, "task", pb);
  // }

  private void translateTask(PragmaBlock pb) throws XMPexception {

    BlockList ret_body = Bcons.emptyBody();

    Block parentBlock = pb.getParentBlock();
    boolean tasksFlag = false;
    if (parentBlock != null && parentBlock.Opcode() == Xcode.XMP_PRAGMA){
      String pragmaName = ((PragmaBlock)parentBlock).getPragma();
      if (XMPpragma.valueOf(pragmaName) == XMPpragma.TASKS){
	tasksFlag = true;
      }
    }

    PragmaBlock declBlock = tasksFlag ? (PragmaBlock)parentBlock : pb;
    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(pb.getClauses().getArg(0), _globalDecl, declBlock);

    // when '*' = the executing node set is specified
    if (on_ref == null){
      pb.replace(Bcons.COMPOUND(pb.getBody()));
      return;
    }

    boolean nocomm_flag = (((XobjInt)pb.getClauses().getArg(1)).getInt() == 1);

    Block b = on_ref.buildConstructor(_globalDecl);
    BasicBlock bb = b.getBasicBlock();

    Ident taskNodesDescId = declBlock.getParentBlock().getBody().declLocalIdent(tmpSym.getStr("_XMP_TASK_NODES"),
										Xtype.voidPtrType);

    Ident f;
    if (!nocomm_flag){
      f = _globalDecl.declExternFunc("xmpc_create_task_nodes", Xtype.voidType);
      bb.add(f.callSubroutine(Xcons.List(taskNodesDescId.getAddr(), on_ref.getDescId().Ref())));
    }

    Ident g1 = _globalDecl.declExternFunc("xmpc_finalize_task_nodes", Xtype.voidType);
    Ident g2 = _globalDecl.declExternFunc("_XMP_ref_dealloc", Xtype.voidType);

    if (tasksFlag){
      parentBlock.insert(b);
      parentBlock.add(g1.Call(Xcons.List(taskNodesDescId.Ref())));
      parentBlock.add(g2.Call(Xcons.List(on_ref.getDescId().Ref())));
    }
    else {
      ret_body.add(b);
    }

    Xobject cond = null;
    if (!nocomm_flag){
      f = _globalDecl.declExternFunc("xmpc_test_task_on_nodes", Xtype.intType);
      cond = f.Call(Xcons.List(taskNodesDescId.Ref(), on_ref.getDescId().Ref()));
    }
    else {
      f = _globalDecl.declExternFunc("xmpc_test_task_nocomm", Xtype.intType);
      cond = f.Call(Xcons.List(on_ref.getDescId().Ref()));
    }

    ret_body.add(Bcons.IF(cond,Bcons.COMPOUND(pb.getBody()), null));
      
    if (!nocomm_flag){
      f = _globalDecl.declExternFunc("xmpc_end_task", Xtype.voidType);
      pb.getBody().add(f.Call(Xcons.List()));
    }

    if (!tasksFlag){
      if (!nocomm_flag) ret_body.add(g1.Call(Xcons.List(taskNodesDescId.Ref())));
      ret_body.add(g2.Call(Xcons.List(on_ref.getDescId().Ref())));
    }

    pb.replace(Bcons.COMPOUND(ret_body));

  }

  private void translateTasks(PragmaBlock pb) {
    // do nothing here
  }

  private void translateLoop(PragmaBlock pb, boolean isFromTranslateLoop) throws XMPexception {
    XobjList loopDecl  = (XobjList)pb.getClauses();
    BlockList loopBody = pb.getBody();

    Block newBlock = null;
    XobjList expandOpt = (XobjList)loopDecl.getArg(5);

    if (expandOpt == null || expandOpt.hasNullArg() || expandOpt.isEmptyList()){
      ;
    }
    else if (expandOpt.getArg(0).getInt() == LOOP_MARGIN){
      newBlock = divideMarginLoop(pb);
    }
    else if (expandOpt.getArg(0).getInt() == LOOP_PEEL_AND_WAIT){
      newBlock = peelLoop(pb);
    }
      
    if (newBlock != null){
      pb.replace(newBlock);
      Block next;
      for (Block b = newBlock.getBody().getHead(); b != null; b = next){
	next = b.getNext();
	if (b instanceof PragmaBlock)
	  translatePragma((PragmaBlock)b);
      }
      return;
    }
    
    // get block to schedule
    CforBlock schedBaseBlock = getOutermostLoopBlock(loopBody);

    // schedule loop
    XobjList loopIterList  = (XobjList)loopDecl.getArg(0);
    XobjList onRef         = (XobjList)loopDecl.getArg(1);
    XobjList onRefIterList = (XobjList)onRef.getArg(1);
    String kind_bracket    = onRefIterList.getTail().getString();
    boolean isSquare       = kind_bracket.equals("SQUARE");

    onRefIterList.removeLastArgs(); // Remove information of ROUND or SQUARE
    if(isFromTranslateLoop){
      loopIterList.removeLastArgs();  // Remove information of ROUND or SQUARE
      if(isSquare) onRefIterList.reverse();
    }
    
    if (loopIterList == null || loopIterList.Nargs() == 0) {
      loopIterList = Xcons.List(Xcons.String(schedBaseBlock.getInductionVar().getName()));
      loopDecl.setArg(0, loopIterList);
      translateFollowingLoop(pb, schedBaseBlock);
    }
    else {
      translateMultipleLoop(pb, schedBaseBlock);
    }

    // translate reduction clause
    XobjList reductionRefList = (XobjList)loopDecl.getArg(2);
    if (reductionRefList != null && reductionRefList.Nargs() > 0) {
      XobjList schedVarList = null;
      if (loopDecl.getArg(0) == null) {
        schedVarList = Xcons.List(Xcons.String(schedBaseBlock.getInductionVar().getSym()));
      }
      else {
        schedVarList = (XobjList)loopDecl.getArg(0).copy();
      }

      BlockList reductionBody = createReductionClauseBody(pb, reductionRefList, schedBaseBlock);
      // schedBaseBlock.add(createReductionClauseBlock(pb, reductionBody, schedVarList)); #418
      schedBaseBlock.getParentBlock().add(createReductionClauseBlock(pb, reductionBody, schedVarList));
    }

    // translate multicore clause
    XobjList multicoreClause = (XobjList)loopDecl.getArg(3);
    if (multicoreClause != null && multicoreClause.Nargs() > 0) {
      String devName = multicoreClause.getArg(0).getString();

      if (devName.equals("acc")) {
        if (XmOption.isXcalableMPGPU()) {
          Block newLoopBlock = translateGpuClause(pb, reductionRefList, schedBaseBlock);
          schedBaseBlock.replace(newLoopBlock);
        }
        else {
          XMP.warning("use '-enable-gpu' compiler option to use gpu clause");
        }
      }
      else if (devName.equals("threads")) {
        if (XmOption.isXcalableMPthreads()) {
          XobjList devArgs = (XobjList)multicoreClause.getArg(1);
          Block newLoopBlock = translateThreadsClauseToOMPpragma(devArgs, reductionRefList, schedBaseBlock, loopIterList);
          schedBaseBlock.replace(newLoopBlock);
        }
        else {
          XMP.warning("use '-enable-threads' compiler option to use 'threads' clause");
        }
      }
      else {
        throw new XMPexception("unknown clause in loop directive");
      }
    }

    // for XMPT
    if (XmOption.isXmptEnabled()){

      Ident xmptFuncId;
      XobjList xmptArgs;

      // event_loop_begin
      xmptFuncId = _globalDecl.declExternFunc("_XMP_loop_begin");
      xmptArgs = Xcons.List();

      String onRefObjName = onRef.getArg(0).getString();
      XMPobject onRefObj = _globalDecl.getXMPobject(onRefObjName, pb);
      xmptArgs.add(onRefObj.getDescId().Ref());

      Ident xmptDataId = pb.getParentBlock().getBody().declLocalIdent(tmpSym.getStr("xmpt_data"), Xtype.voidPtrType);
      xmptArgs.add(xmptDataId.getAddr());

      for (XobjArgs i = onRef.getArg(1).getArgs(); i != null; i = i.nextArgs()){
	//xmptArgs.add(i.getArg());
	xmptArgs.add(Xcons.IntConstant(0));
	xmptArgs.add(Xcons.IntConstant(0));
	xmptArgs.add(Xcons.IntConstant(0));
      }
      
      //schedBaseBlock.insert(xmptFuncId.Call(xmptArgs));
      loopBody.insert(xmptFuncId.Call(xmptArgs));

      // event_loop_end
      xmptFuncId = _globalDecl.declExternFunc("_XMP_loop_end");
      xmptArgs = Xcons.List(xmptDataId.getAddr());
      loopBody.add(xmptFuncId.Call(xmptArgs));
    }
	      
    // rewrite array refs in loop
    topdownXobjectIterator iter = new topdownXobjectIterator(getLoopBody(schedBaseBlock).toXobject());
    for (iter.init(); !iter.end(); iter.next()) {
      XMPrewriteExpr.rewriteArrayRefInLoop(iter.getXobject(), _globalDecl, schedBaseBlock, loopBody);
    }

    // replace pragma
    Block loopFuncCallBlock = Bcons.COMPOUND(loopBody);
    pb.replace(loopFuncCallBlock);

    // add function calls for profiling
    Xobject profileClause = null;
    if (loopDecl.Nargs() > 4)
      profileClause = loopDecl.getArg(4);

    addProfileFunctions(profileClause, loopFuncCallBlock, "loop", pb);
  }

  // XXX only supports C language
  private Block translateGpuClause(PragmaBlock pb, XobjList reductionRefList,
                                   CforBlock loopBlock) throws XMPexception {
    Ident funcId = _globalDecl.declExternIdent(_globalDecl.genSym(XMP.GPU_FUNC_PREFIX),
                                               Xtype.Function(Xtype.voidType));

    XobjList funcArgs = setupGPUparallelFunc(funcId, loopBlock, pb);

    return _globalDecl.createFuncCallBlock(funcId.getName(), funcArgs);
  }

  private XobjList setupGPUparallelFunc(Ident funcId, CforBlock loopBlock, PragmaBlock pb) throws XMPexception {
    XobjList loopVarList = (XobjList)((XobjList)(pb.getClauses())).getArg(0);
    XobjList newLoopVarList = setupGpuLoopBlock(Xcons.List(loopBlock.findVarIdent(loopBlock.getInductionVar().getSym())),
                                                loopVarList, loopBlock.getBody().getHead());

    // get params
    XobjList paramIdList = getGPUfuncParams(loopBlock, pb);

    // setup & decompile GPU function body
    XMPgpuDecompiler.decompile(funcId, paramIdList, getXMPalignedArrays(loopBlock),
                               loopBlock, newLoopVarList, pb, _globalDecl.getEnv());

    // generate func args
    XobjList funcArgs = Xcons.List();
    for (XobjArgs i = paramIdList.getArgs(); i != null; i = i.nextArgs()) {
      Ident paramId = (Ident)i.getArg();
      String paramName = paramId.getName();
      XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(paramId.getName(), loopBlock);
      if (gpuData == null) {
        if (paramId.Type().isArray()) {
          throw new XMPexception("array '" + paramName + "' should be declared as a gpuData");
        } else {
          funcArgs.add(paramId.Ref());
        }
      } else {
        XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
        funcArgs.add(gpuData.getDeviceAddrId().Ref());
      }
    }

    return funcArgs;
  }

  private XobjList setupGpuLoopBlock(XobjList newLoopVarList, XobjList loopVarList, Block b) throws XMPexception {
    switch (b.Opcode()) {
      case FOR_STATEMENT:
        {
          CforBlock loopBlock = (CforBlock)b;
          if (!loopBlock.isCanonical()) {
            loopBlock.Canonicalize();
          }

          Block bodyBlock = b.getBody().getHead();
          String loopVarName = loopBlock.getInductionVar().getSym();
          if (XMPutil.hasElmt(loopVarList, loopVarName)) {
            newLoopVarList.insert(b.findVarIdent(loopVarName));
            b.getParentBlock().setBody(loopBlock.getBody());
          }

          return setupGpuLoopBlock(newLoopVarList, loopVarList, bodyBlock);
        }
      case COMPOUND_STATEMENT:
        return setupGpuLoopBlock(newLoopVarList, loopVarList, b.getBody().getHead());
      default:
        return newLoopVarList;
    }
  }

  private ArrayList<XMPalignedArray> getXMPalignedArrays(CforBlock loopBlock) throws XMPexception {
    ArrayList<XMPalignedArray> alignedArrayList = new ArrayList<XMPalignedArray>();
    XobjList arrayNameList = Xcons.List();

    BasicBlockExprIterator iter = new BasicBlockExprIterator(loopBlock.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
          case ARRAY_REF:
            {
              String varName = x.getArg(0).getSym();
              if (!XMPutil.hasElmt(arrayNameList, varName)) {
                arrayNameList.add(Xcons.String(varName));

                XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(varName, loopBlock);
                if (gpuData == null) {
                  throw new XMPexception("array '" + varName + "' shoud be declared as a gpuData");
                } else {
                  XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
                  if (alignedArray != null) {
                    alignedArrayList.add(alignedArray);
                  }
                }
              }
            } break;
          default:
        }
      }
    }

    return alignedArrayList;
  }

  private XobjList getGPUfuncParams(CforBlock loopBlock, PragmaBlock pb) throws XMPexception {
    XobjList params = Xcons.List();
    XobjList loopVars = Xcons.List();

    BasicBlockExprIterator iter = new BasicBlockExprIterator(loopBlock.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
          case VAR:
            {
              String varName = x.getName();
              if (!(XMPutil.hasIdent(params, varName) ||
                    XMPutil.hasElmt(loopVars, varName))) {
                XobjList loopIter = XMPutil.getLoopIter(loopBlock, varName);
                if (loopIter == null) {
                  Ident id = loopBlock.findVarIdent(varName);
                  if (id != null) {
                    params.add(Ident.Param(varName, id.Type()));
                  }
                } else {
                  loopVars.add(Xcons.String(varName));
                }
              }
            } break;
          case ARRAY_REF:
            {
              String varName = x.getArg(0).getSym();
              if (!XMPutil.hasIdent(params, varName)) {
                XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(varName, loopBlock);
                if (gpuData == null) {
                  throw new XMPexception("array '" + varName + "' shoud be declared as a gpuData");
                } else {
                  XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
                  if (alignedArray == null) {
                    Ident id = loopBlock.findVarIdent(varName);
                    params.add(Ident.Param(varName, id.Type()));
                  } else {
                    Xtype alignedArrayParamType = null;
                    if (alignedArray.realloc()) {
                      alignedArrayParamType = alignedArray.getAddrId().Type();
                    } else {
                      alignedArrayParamType = alignedArray.getArrayId().Type();
                    }

                    params.add(Ident.Param(varName, alignedArrayParamType));
                    params.add(Ident.Param(XMP.GPU_DEVICE_DESC_PREFIX_ + varName, Xtype.voidPtrType));
                  }
                }
              }
            } break;
          default:
        }
      }
    }

    // FIXME consider the order
    XobjList loopIndexList = (XobjList)pb.getClauses().getArg(0);
    for (Xobject loopIndex : loopIndexList) {
      XobjList loopIter = XMPutil.getLoopIter(loopBlock, loopIndex.getName()); 

      Ident initId = (Ident)loopIter.getArg(0);
      params.add(Ident.Param(initId.getName(), initId.Type()));

      Ident condId = (Ident)loopIter.getArg(1);
      params.add(Ident.Param(condId.getName(), condId.Type()));

      Ident stepId = (Ident)loopIter.getArg(2);
      params.add(Ident.Param(stepId.getName(), stepId.Type()));
    }

    return params;
  }

  private Block createOMPpragmaBlock(OMPpragma pragma, Xobject args, Block body) {
    return Bcons.PRAGMA(Xcode.OMP_PRAGMA, pragma.toString(), args, Bcons.blockList(body));
  }

  private Block translateThreadsClauseToOMPpragma(XobjList threadsClause, XobjList reductionRefList,
                                                  CforBlock loopBlock, XobjList loopIterList) throws XMPexception {
    Xobject parallelClause = Xcons.statementList();
    XobjList forClause = Xcons.List();

    if (threadsClause != null) {
      for (Xobject c : threadsClause) {
        OMPpragma p = OMPpragma.valueOf(c.getArg(0));
        switch (p) {
          case DATA_PRIVATE:
          case DATA_FIRSTPRIVATE:
          case DATA_LASTPRIVATE:
            {
              compile_THREADS_name_list(c.getArg(1));
              forClause.add(c);
            } break;
          case DIR_NUM_THREADS:
          case DIR_IF:
            parallelClause.add(c);
            break;
          default:
            throw new XMPexception("unknown threads clause");
        }
      }
    }

    // FIXME compare loopIterList with private, firstprivate var list
    if (loopIterList != null) {
      String schedLoopIterName = loopBlock.getInductionVar().getSym();
      XobjList privateList = Xcons.List();
      XobjList firstPrivateList = Xcons.List();
      Iterator<Xobject> iter = loopIterList.iterator();
      while (iter.hasNext()) {
        Xobject x = iter.next();
        String iterName = x.getName();
        if (!iterName.equals(schedLoopIterName)) {
          privateList.add(Xcons.Symbol(Xcode.IDENT, iterName));
          firstPrivateList.add(Xcons.Symbol(Xcode.IDENT, "_XMP_loop_init_" + iterName));
          firstPrivateList.add(Xcons.Symbol(Xcode.IDENT, "_XMP_loop_cond_" + iterName));
          firstPrivateList.add(Xcons.Symbol(Xcode.IDENT, "_XMP_loop_step_" + iterName));
        }
      }

      forClause.add(Xcons.List(Xcode.LIST, Xcons.String(OMPpragma.DATA_PRIVATE.toString()), privateList));
      forClause.add(Xcons.List(Xcode.LIST, Xcons.String(OMPpragma.DATA_FIRSTPRIVATE.toString()), firstPrivateList));
    }

    addReductionClauseToOMPclause(forClause, reductionRefList);

    return createOMPpragmaBlock(OMPpragma.PARALLEL, parallelClause,
                                createOMPpragmaBlock(OMPpragma.FOR, forClause,
                                                     loopBlock));
  }

  private void addReductionClauseToOMPclause(XobjList OMPclause, XobjList reductionRefList) throws XMPexception {
    if (reductionRefList == null) {
      return;
    }

    for (Xobject c : reductionRefList) {
      OMPpragma redOp = null;

      XobjInt reductionOp = (XobjInt)c.getArg(0);
      switch (reductionOp.getInt()) {
        case XMPcollective.REDUCE_SUM:
          redOp = OMPpragma.DATA_REDUCTION_PLUS;
          break;
        case XMPcollective.REDUCE_MINUS:
          redOp = OMPpragma.DATA_REDUCTION_MINUS;
          break;
        case XMPcollective.REDUCE_PROD:
          redOp = OMPpragma.DATA_REDUCTION_MUL;
          break;
        case XMPcollective.REDUCE_BAND:
          redOp = OMPpragma.DATA_REDUCTION_BITAND;
          break;
        case XMPcollective.REDUCE_LAND:
          redOp = OMPpragma.DATA_REDUCTION_LOGAND;
          break;
        case XMPcollective.REDUCE_BOR:
          redOp = OMPpragma.DATA_REDUCTION_BITOR;
          break;
        case XMPcollective.REDUCE_LOR:
          redOp = OMPpragma.DATA_REDUCTION_LOGOR;
          break;
        case XMPcollective.REDUCE_BXOR:
          redOp = OMPpragma.DATA_REDUCTION_BITXOR;
          break;
        case XMPcollective.REDUCE_LXOR:
        case XMPcollective.REDUCE_MAX:
        case XMPcollective.REDUCE_MIN:
        case XMPcollective.REDUCE_FIRSTMAX:
        case XMPcollective.REDUCE_FIRSTMIN:
        case XMPcollective.REDUCE_LASTMAX:
        case XMPcollective.REDUCE_LASTMIN:
          throw new XMPexception("the operation cannot be translated to OpenMP clause");
        default:
          throw new XMPexception("unknown reduction operation");
      }

      XobjList redVarList = Xcons.List();
      for (Xobject x : (XobjList)c.getArg(1)) {
        redVarList.add(Xcons.Symbol(Xcode.IDENT, x.getArg(0).getName()));
      }

      OMPclause.add(omp_pg_list(redOp, redVarList));
    }
  }

  private Xobject omp_pg_list(OMPpragma pg, Xobject args) {
    return Xcons.List(Xcode.LIST, Xcons.String(pg.toString()), args);
  }

  // XXX not implemented yet, check variables
  private void compile_THREADS_name_list(Xobject name_list) throws XMPexception {
    if (name_list == null) {
      return;
    }
  }

  private Block createReductionClauseBlock(PragmaBlock pb, BlockList reductionBody, XobjList schedVarList) throws XMPexception {
    XobjList loopDecl      = (XobjList)pb.getClauses();
    XobjList onRef         = (XobjList)loopDecl.getArg(1);
    String onRefObjName    = onRef.getArg(0).getString();
    XobjList subscriptList = (XobjList)onRef.getArg(1);

    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    XMPobject onRefObj = _globalDecl.getXMPobject(onRefObjName, pb);
    if (onRefObj == null) {
      throw new XMPexception("cannot find '" + onRefObjName + "' nodes/template");
    }

    String initFuncSuffix = null;
    switch (onRefObj.getKind()) {
      case XMPobject.TEMPLATE:
        initFuncSuffix = "TEMPLATE";
        break;
      case XMPobject.NODES:
        initFuncSuffix = "NODES";
        break;
      default:
        throw new XMPexception("unknown object type");
    }

    // create function arguments
    XobjList initFuncArgs = Xcons.List(onRefObj.getDescId().Ref());

    boolean initComm = false;
    for (XobjArgs i = subscriptList.getArgs(); i != null; i = i.nextArgs()) {
      String subscript = i.getArg().getString();
      if (XMPutil.hasElmt(schedVarList, subscript)) {
        initFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(0)));
      }
      else {
        initComm = true;
        initFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
      }
    }

    if (initComm) {
      // setup finalizer
      Ident finFuncId = _globalDecl.declExternFunc("_XMP_pop_n_free_nodes");
      setupFinalizer(reductionBody, finFuncId, null);

      // create function call
      Ident initFuncId = _globalDecl.declExternFunc("_XMP_init_reduce_comm_" + initFuncSuffix, Xtype.intType);

      return Bcons.IF(BasicBlock.Cond(initFuncId.Call(initFuncArgs)), reductionBody, null);
    }
    else {
      return Bcons.COMPOUND(reductionBody);
    }
  }

  private void translateFollowingLoop(PragmaBlock pb, CforBlock schedBaseBlock) throws XMPexception {
    XobjList loopDecl              = (XobjList)pb.getClauses();
    ArrayList<String> iteraterList = new ArrayList<String>();  // Not used
    boolean[] isOmitSchedLoopFunc  = new boolean[1];    // Initialize false automatically
    
    isOmitSchedLoopFunc[0] = scheduleLoop(pb, schedBaseBlock, schedBaseBlock, iteraterList);
    if(isOmitSchedLoopFunc[0] == false)
      insertScheduleIndexFunction(pb, schedBaseBlock, schedBaseBlock, iteraterList);
  }

  private void translateMultipleLoop(PragmaBlock pb, CforBlock schedBaseBlock) throws XMPexception {
    // start translation
    XobjList loopDecl  = (XobjList)pb.getClauses();
    BlockList loopBody = pb.getBody();

    // iterate index variable list
    XobjList loopVarList = (XobjList)loopDecl.getArg(0);

    Vector<CforBlock> loopVector = new Vector<CforBlock>(XMPutil.countElmts(loopVarList));
    int num = 0;
    for (XobjArgs i = loopVarList.getArgs(); i != null; i = i.nextArgs()) {
      loopVector.add(findLoopBlock(loopBody, i.getArg().getString()));
      num++;
    }

    // schedule loop
    Iterator<CforBlock> it         = loopVector.iterator();
    ArrayList<String> iteraterList = new ArrayList<String>();
    boolean[] isOmitSchedLoopFunc  = new boolean[num];
    int i = 0;
    while (it.hasNext()) {
      CforBlock forBlock = it.next();
      isOmitSchedLoopFunc[i++] = scheduleLoop(pb, forBlock, schedBaseBlock, iteraterList);
    }

    it = loopVector.iterator();
    i = 0;
    while (it.hasNext()) {
      CforBlock forBlock = it.next();
      if(isOmitSchedLoopFunc[i++] == false)
        insertScheduleIndexFunction(pb, forBlock, schedBaseBlock, iteraterList);
    }
  }

  private final static int LOOP_EXPAND        = 410;
  private final static int LOOP_MARGIN        = 411;
  private final static int LOOP_PEEL_AND_WAIT = 412;
  private final static int LOOP_NONE          = 413;

  private void insertScheduleIndexFunction(PragmaBlock pb, CforBlock forBlock, CforBlock schedBaseBlock, 
					   ArrayList iteraterList) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(schedBaseBlock);
    Xobject onRef = loopDecl.getArg(1);
    String onRefObjName = onRef.getArg(0).getString();
    XMPobject onRefObj = _globalDecl.getXMPobject(onRefObjName, schedBaseBlock);

    XMPtemplate templateObj = (XMPtemplate)onRefObj;
    XobjList templateSubscriptList = (XobjList)onRef.getArg(1);
    Xobject loopIndex = forBlock.getInductionVar();
    String loopIndexName = loopIndex.getSym();
    XobjList funcArgs = Xcons.List();
    Xobject lb = forBlock.getLowerBound();
    Xobject up = forBlock.getUpperBound();
    Xobject step = forBlock.getStep();
    funcArgs.add(lb);
    funcArgs.add(up);
    funcArgs.add(step);

    Ident parallelInitId = declIdentWithBlock(schedBaseBlock, "_XMP_loop_init_" + loopIndexName, Xtype.intType);
    Ident parallelCondId = declIdentWithBlock(schedBaseBlock, "_XMP_loop_cond_" + loopIndexName, Xtype.intType);
    Ident parallelStepId = declIdentWithBlock(schedBaseBlock, "_XMP_loop_step_" + loopIndexName, Xtype.intType);
    
    funcArgs.add(parallelInitId.getAddr());
    funcArgs.add(parallelCondId.getAddr());
    funcArgs.add(parallelStepId.getAddr());

    XobjInt templateIndexArg = null;
    int templateIndex = 0;
    int distManner = 0;
    String distMannerString = null;
    for (XobjArgs i = templateSubscriptList.getArgs(); i != null; i = i.nextArgs()) {
      String s = i.getArg().getString();
      if (s.equals(loopIndexName)) {
        templateIndexArg = Xcons.IntConstant(templateIndex);
        distManner = templateObj.getDistMannerAt(templateIndex);
        distMannerString = templateObj.getDistMannerStringAt(templateIndex);
      }
      templateIndex++;
    }
    funcArgs.add(templateObj.getDescId().Ref());
    funcArgs.add(templateIndexArg);

    //
    // for EXPAND/MARGIN/PEEL_AND_WAIT
    //

    //if (distManner == XMPtemplate.BLOCK){
    
      Xobject expandDecl = loopDecl.getArg(5);
      int origLoopType, loopType;
      Xobject asyncId;
      Xobject expandList;
      if (expandDecl != null && !expandDecl.isEmptyList()){

	origLoopType = expandDecl.getArg(0).getInt();

	if (origLoopType == LOOP_PEEL_AND_WAIT){
	  asyncId = expandDecl.getArg(1);
	  expandList = expandDecl.getArg(2);
	}
	else {
	  expandList = expandDecl.getArg(1);
	}
	
	int t_idx = templateIndexArg.getInt();
	Xobject lwidth = expandList.getArg(t_idx).getArg(0);
	Xobject uwidth = expandList.getArg(t_idx).getArg(1);;
	Xobject unboundFlag = expandList.getArg(t_idx).getArg(2);

	if (origLoopType == LOOP_MARGIN && unboundFlag.getInt() == -1){
	  loopType = LOOP_EXPAND;
	}
	else {
	  loopType = origLoopType;
	}

	funcArgs.add(Xcons.IntConstant(loopType));
	funcArgs.add(lwidth);
	funcArgs.add(uwidth);
	funcArgs.add(unboundFlag);
      }
      else {
	funcArgs.add(Xcons.IntConstant(LOOP_NONE));
	funcArgs.add(Xcons.IntConstant(0));
	funcArgs.add(Xcons.IntConstant(0));
	funcArgs.add(Xcons.IntConstant(0));
      }

      //    }

    //Ident funcId = _globalDecl.declExternFunc("_XMP_sched_loop_template_" + distMannerString);
    Ident funcId = _globalDecl.declExternFunc("xmpc_loop_sched");
    int[] position = {iteraterList.size()};
    boolean[] flag = {false, false, false};
    String[] insertedIteraterList = new String[3];
    flag[0] = getPositionInsertScheFuncion(lb,   0, position, iteraterList, insertedIteraterList);
    flag[1] = getPositionInsertScheFuncion(up,   1, position, iteraterList, insertedIteraterList);
    flag[2] = getPositionInsertScheFuncion(step, 2, position, iteraterList, insertedIteraterList);

    if(position[0] == iteraterList.size()){
      Block b = getOuterSchedPoint(schedBaseBlock);
      b.insert(funcId.Call(funcArgs));
    }
    else{
      if(flag[0]){ 
    	insertScheFuncion(lb,   flag, position[0], 0, schedBaseBlock, funcArgs, templateObj, 
    			  funcId, iteraterList, insertedIteraterList);
      }
      if(flag[1]){
    	insertScheFuncion(up,   flag, position[0], 1, schedBaseBlock, funcArgs, templateObj, 
    			  funcId, iteraterList, insertedIteraterList);
      }
      if(flag[2]){
    	insertScheFuncion(step, flag, position[0], 2, schedBaseBlock, funcArgs, templateObj,
    			  funcId, iteraterList, insertedIteraterList);
      }
    }
  }

  private void insertScheFuncion(Xobject v, boolean[] flag, int position, int num, Block schedBaseBlock, XobjList funcArgs, 
				 XMPtemplate templateObj, Ident funcId, ArrayList iteraterList, String[] insertedIteraterList) throws XMPexception {

    //Block b = schedBaseBlock;
    Block b = getOuterSchedPoint(schedBaseBlock);

    for(int i=0;i<iteraterList.size()-position;i++){
      b = b.getBody().getHead();
    }

    Xobject func = funcId.Call(funcArgs);
    int loop_depth = 0;
    String index = insertedIteraterList[num];
    for(int i=0;i<iteraterList.size();i++){
      if(iteraterList.get(i).toString().equals(index)){
	loop_depth = i;
      }
    }
    v = XMPrewriteExpr.calcLtoG(templateObj, loop_depth, v, _globalDecl);
    funcArgs.setArg(num, v);

    // Insert Function
    if(num == 0){
      b.insert(func);
      return;
    }

    if(num == 1){
      if(flag[0]){
	return; 
      }
      else{
        b.insert(func);
      }
      return;
    }

    if(num == 2){
      if(flag[0] || flag[1]){
	return;
      }
      else{
	b.insert(func);
      }
      return;
    }
  }

  private boolean getPositionInsertScheFuncion(Xobject v, int num, int[] position, ArrayList iteraterList, 
					       String[] insertedIteraterList) throws XMPexception {
    if(v.Opcode() != Xcode.INT_CONSTANT){
      XobjList vList = null;
      if(v.isVariable() || v instanceof XobjConst){
        vList = Xcons.List(v);
      }
      else{
        vList = XobjArgs2XobjList(v, Xcons.List());
      }
      for (XobjArgs i = vList.getArgs(); i != null; i = i.nextArgs()) {
	if(i.getArg().isVariable()){
          for(int j=0;j<iteraterList.size();j++){
            if(i.getArg().getSym().equals(iteraterList.get(j))){
	      if(position[0] > j){
		position[0] = j;
	      }
	      insertedIteraterList[num] = iteraterList.get(j).toString();
	      return true;
            }
	  }
	}
      }
    }
    return false;
  }

  private XobjList XobjArgs2XobjList(Xobject x, XobjList a){
    for (XobjArgs i = x.getArgs(); i != null; i = i.nextArgs()) {

      if(i.getArg().isBinaryOp()){
	XobjArgs2XobjList(i.getArg(), a);
      } 
      else{
	a.add(i.getArg());
      }
    }
    return a;
  }

  private BlockList createReductionClauseBody(PragmaBlock pb, XobjList reductionRefList,
                                              CforBlock schedBaseBlock) throws XMPexception {
    // create init block
    Ident getRankFuncId = _globalDecl.declExternFunc("_XMP_get_execution_nodes_rank", Xtype.intType);
    IfBlock reductionInitIfBlock = (IfBlock)Bcons.IF(BasicBlock.Cond(Xcons.binaryOp(Xcode.LOG_NEQ_EXPR, getRankFuncId.Call(null),
                                                                                    Xcons.IntConstant(0))), null, null);

    // create function call
    Iterator<Xobject> it = reductionRefList.iterator();
    BlockList reductionBody = Bcons.emptyBody();
    while (it.hasNext()) {
      XobjList reductionRef = (XobjList)it.next();
      Vector<XobjList> reductionFuncArgsList = createReductionArgsList(reductionRef, pb,
                                                                       true, schedBaseBlock, reductionInitIfBlock);
      String reductionFuncType = createReductionFuncType(reductionRef, pb, false);

      reductionBody.add(createReductionFuncCallBlock(false, reductionFuncType + "_CLAUSE",
                                                     null, reductionFuncArgsList));
    }

    if (reductionInitIfBlock.getThenBody() != null) {
      // schedBaseBlock.insert(reductionInitIfBlock); # 418
      schedBaseBlock.getParentBlock().insert(reductionInitIfBlock);
    }

    return reductionBody;
  }

  private boolean scheduleLoop(PragmaBlock pb, CforBlock forBlock, CforBlock schedBaseBlock,
                               ArrayList<String> iteraterList) throws XMPexception {
    XobjList loopDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(schedBaseBlock);

    // analyze <on-ref>
    Xobject onRef       = loopDecl.getArg(1);
    String onRefObjName = onRef.getArg(0).getString();
    XMPobject onRefObj  = _globalDecl.getXMPobject(onRefObjName, schedBaseBlock);
    if (onRefObj == null) {
      throw new XMPexception("cannot find '" + onRefObjName + "' nodes/template");
    }

    switch (onRefObj.getKind()) {
      case XMPobject.TEMPLATE:
        {
          XMPtemplate onRefTemplate = (XMPtemplate)onRefObj;
          if (!onRefTemplate.isDistributed()) {
            throw new XMPexception("template '" + onRefObjName + "' is not distributed");
          }

	  Xobject expandDecl = loopDecl.getArg(5);
	  boolean withExpand = (expandDecl != null && !expandDecl.isEmptyList());

          return callLoopSchedFuncTemplate(onRefTemplate, (XobjList)onRef.getArg(1), forBlock, schedBaseBlock, iteraterList,
					   withExpand);
        }
      case XMPobject.NODES:
        callLoopSchedFuncNodes((XMPnodes)onRefObj, (XobjList)onRef.getArg(1), forBlock, schedBaseBlock);
        return false;
      default:
        throw new XMPexception("unknown object type");
    }
  }

  private static Block getOuterSchedPoint(Block b){

    Block bb = b.getParentBlock();

    while (bb != null){
      if (bb.Opcode() == Xcode.OMP_PRAGMA || bb.Opcode() == Xcode.ACC_PRAGMA) return bb;
      bb = bb.getParentBlock();
    }

    return b;

  }

  private static CforBlock getOutermostLoopBlock(BlockList body) throws XMPexception {
    Block b = body.getHead();
    if (b != null) {
      if (b.Opcode() == Xcode.FOR_STATEMENT) {
        LineNo blockLnObj = b.getLineNo();

        // XXX too strict?
        if (b.getNext() != null) {
          throw new XMPexception("only one loop statement is allowed in loop directive");
        }

        CforBlock forBlock = (CforBlock)b;
        forBlock.Canonicalize();
        if (forBlock.isCanonical()) {
          return forBlock;
        } else {
          throw new XMPexception("loop statement is not canonical");
        }
      }
      else if (b.Opcode() == Xcode.COMPOUND_STATEMENT ||
	       b.Opcode() == Xcode.OMP_PRAGMA ||
	       b.Opcode() == Xcode.ACC_PRAGMA) {
        return getOutermostLoopBlock(b.getBody());
      }

    }

    throw new XMPexception("cannot find the loop statement");
  }

  private static BlockList getLoopBody(Block b) throws XMPexception {
    return b.getBody();
  }

  private static CforBlock findLoopBlock(BlockList body, String loopVarName) throws XMPexception {
    Block b = body.getHead();
    if (b != null) {
      switch (b.Opcode()) {
        case FOR_STATEMENT:
          {
            CforBlock forBlock = (CforBlock)b;
            forBlock.Canonicalize();
            if (!(forBlock.isCanonical())) {
              throw new XMPexception("loop is not canonical");
            }

            if (forBlock.getInductionVar().getSym().equals(loopVarName)) {
              return (CforBlock)b;
            } else {
              return findLoopBlock(forBlock.getBody(), loopVarName);
            }
          }
        case COMPOUND_STATEMENT:
          return findLoopBlock(b.getBody(), loopVarName);
        case XMP_PRAGMA:
          throw new XMPexception("reached to a xcalablemp directive");
        case OMP_PRAGMA:
        case ACC_PRAGMA:
	  return findLoopBlock(b.getBody(), loopVarName);
      }
    }

    throw new XMPexception("cannot find the loop statement");
  }

  private int calcParallelCond(XMPtemplate t, int i, int manner) throws XMPexception
  {
    XMPnodes n       = t.getOntoNodes();
    int templateSize = XMPutil.foldIntConstant(t.getSizeAt(i)).getInt();
    
    if(manner == XMPtemplate.DUPLICATION)
      return templateSize;

    int nodeIndex    = t.getOntoNodesIndexAt(i).getInt();
    int nodeSize     = XMPutil.foldIntConstant(n.getSizeAt(nodeIndex)).getInt();

    return templateSize/nodeSize;
  }
  
  private boolean is_parallelCondConstant(XMPtemplate t, CforBlock forBlock, int i, int manner) throws XMPexception
  {
    XMPnodes n = t.getOntoNodes();
    
    if(n.isInherit() == true)                                      return false;
    if(manner == XMPtemplate.GBLOCK)                               return false;
    if(! XMPutil.is_Constant(t.getSizeAt(i), i))                   return false;
    if(! XMPutil.is_Constant(forBlock.getUpperBound(), i))         return false;
    if(manner != XMPtemplate.DUPLICATION){
      int nodeIndex = t.getOntoNodesIndexAt(i).getInt();
      if(! XMPutil.is_Constant(n.getSizeAt(nodeIndex), nodeIndex)) return false;
    }
    if(manner == XMPtemplate.BLOCK_CYCLIC)
      if(! XMPutil.is_Constant(t.getWidthAt(i), i))                return false;
    
    int templateSize = XMPutil.foldIntConstant(t.getSizeAt(i)).getInt();
    int forUpperSize = XMPutil.foldIntConstant(forBlock.getUpperBound()).getInt();
    if(templateSize != forUpperSize)
      return false;

    int blockSize = (manner == XMPtemplate.BLOCK_CYCLIC)? XMPutil.foldIntConstant(t.getWidthAt(i)).getInt() : 1;
    if(manner == XMPtemplate.DUPLICATION){
      return true;
    }
    else{
      int nodeIndex = t.getOntoNodesIndexAt(i).getInt();
      int nodeSize  = XMPutil.foldIntConstant(n.getSizeAt(nodeIndex)).getInt();
      if(templateSize%(nodeSize*blockSize) != 0)
        return false;
    }
    return true;
  }
  
  private boolean callLoopSchedFuncTemplate(XMPtemplate templateObj, XobjList templateSubscriptList, CforBlock forBlock,
                                            CforBlock schedBaseBlock, ArrayList<String> iteraterList,
					    boolean withExpand) throws XMPexception {
    Xobject loopIndex    = forBlock.getInductionVar();
    String loopIndexName = loopIndex.getSym();
    iteraterList.add(loopIndexName);

    int templateIndex        = 0;
    int templateDim          = templateObj.getDim();
    XobjInt templateIndexArg = null;
    int distManner           = 0;
    String distMannerString  = null;
    int targetIndex          = 0;
    
    for (XobjArgs i = templateSubscriptList.getArgs(); i != null; i = i.nextArgs()) {
      if (templateIndex >= templateDim) {
        throw new XMPexception("wrong template dimensions, too many");
      }

      if (i.getArg() != null){
	String s = i.getArg().getString();
	if (s.equals(loopIndexName)) {
	  if (templateIndexArg != null) {
	    throw new XMPexception("loop index '" + loopIndexName + "' is already described");
	  }

	  templateIndexArg = Xcons.IntConstant(templateIndex);
	  distManner       = templateObj.getDistMannerAt(templateIndex);
	  distMannerString = templateObj.getDistMannerStringAt(templateIndex);
	  targetIndex      = templateIndex;
	}
      }

      templateIndex++;
    }

    if(templateIndexArg == null) {
      throw new XMPexception("cannot find index '" + loopIndexName + "' reference in <on-ref>");
    }

    if(templateIndex != templateDim) {
      throw new XMPexception("wrong template dimensions, too few");
    }

    boolean isOmitSchedLoopFunc = true;
    Xobject parallelInit;
    if(!withExpand && forBlock.getLowerBound().equals(Xcons.IntConstant(0)))
      parallelInit        = Xcons.IntConstant(0);
    else{
      parallelInit = declIdentWithBlock(schedBaseBlock,
                                        "_XMP_loop_init_" + loopIndexName, Xtype.intType).Ref();
      isOmitSchedLoopFunc = false;
    }
    
    Xobject parallelCond;
    if(templateObj.isFixed() && !withExpand &&
       is_parallelCondConstant(templateObj, forBlock, targetIndex, distManner))
      parallelCond = Xcons.IntConstant(calcParallelCond(templateObj, targetIndex, distManner));
    else{
      parallelCond = declIdentWithBlock(schedBaseBlock,
                                        "_XMP_loop_cond_" + loopIndexName, Xtype.intType).Ref();
      isOmitSchedLoopFunc = false;
    }
    
    Xobject parallelStep;
    if(!withExpand && forBlock.getStep().equals(Xcons.IntConstant(1)))
      parallelStep = Xcons.IntConstant(1);
    else if(!withExpand && forBlock.getStep().equals(Xcons.IntConstant(-1)))
      parallelStep = Xcons.IntConstant(-1);
    else{
      parallelStep = declIdentWithBlock(schedBaseBlock,
                                        "_XMP_loop_step_" + loopIndexName, Xtype.intType).Ref();
      isOmitSchedLoopFunc = false;
    }

    XMPutil.putLoopIter(schedBaseBlock, loopIndexName,
                        Xcons.List(parallelInit, parallelCond, parallelStep));
    
    switch (distManner) {
      case XMPtemplate.DUPLICATION:
      case XMPtemplate.BLOCK:
      case XMPtemplate.CYCLIC:
      case XMPtemplate.BLOCK_CYCLIC:
      case XMPtemplate.GBLOCK:
        forBlock.setLowerBound(parallelInit);
        forBlock.setUpperBound(parallelCond);
        forBlock.setStep(parallelStep);
        break;
      default:
        throw new XMPexception("unknown distribute manner");
    }

    // rewrite loop index in loop
    BasicBlockExprIterator iter = new BasicBlockExprIterator(getLoopBody(forBlock));
    for (iter.init(); !iter.end(); iter.next()) {
      XMPrewriteExpr.rewriteLoopIndexInLoop(iter.getExpr(), loopIndexName,
                                            templateObj, templateIndexArg.getInt(),
                                            _globalDecl, forBlock);
    }

    // rewrite loop index in initializer in loop
    BlockList body = getLoopBody(forBlock);
    while(true){
      for (Block b = body.getHead(); b != null; b = b.getNext()){
        if (b.getBody() == null) continue;
        topdownXobjectIterator iter_decl = new topdownXobjectIterator(b.getBody().getDecls());

        for (iter_decl.init(); !iter_decl.end(); iter_decl.next()) {
          int num = 0;
          for (XobjArgs i = templateSubscriptList.getArgs(); i != null; i = i.nextArgs()) {
            String indexName = i.getArg().getString();
            XMPrewriteExpr.rewriteLoopIndexInLoop(iter_decl.getXobject(), indexName,
                                                  templateObj, num, _globalDecl, forBlock);
            num++;
          }
        }
      }
      
      body = body.getHead().getBody();
      if(body == null) break;
      if(body.getHead() == null) break;
    }
    return isOmitSchedLoopFunc;
  }

  private void callLoopSchedFuncNodes(XMPnodes nodesObj, XobjList nodesSubscriptList,
                                      CforBlock forBlock, CforBlock schedBaseBlock) throws XMPexception {
    Xobject loopIndex = forBlock.getInductionVar();
    Xtype loopIndexType = loopIndex.Type();

    if (!XMPutil.isIntegerType(loopIndexType)) {
      throw new XMPexception("loop index variable has a non-integer type");
    }

    String loopIndexName = loopIndex.getSym();

    int nodesIndex = 0;
    int nodesDim = nodesObj.getDim();
    XobjInt nodesIndexArg = null;
    for (XobjArgs i = nodesSubscriptList.getArgs(); i != null; i = i.nextArgs()) {
      if (nodesIndex >= nodesDim) {
        throw new XMPexception("wrong nodes dimensions, too many");
      }

      String s = i.getArg().getString();
      if (s.equals(loopIndexName)) {
        if (nodesIndexArg != null) {
          throw new XMPexception("loop index '" + loopIndexName + "' is already described");
        }

        nodesIndexArg = Xcons.IntConstant(nodesIndex);
      }

      nodesIndex++;
    }

    if (nodesIndexArg == null) {
      throw new XMPexception("cannot find index '" + loopIndexName + "' reference in <on-ref>");
    }

    if (nodesIndex != nodesDim) {
      throw new XMPexception("wrong nodes dimensions, too few");
    }

    Ident parallelInitId = declIdentWithBlock(schedBaseBlock,
                                              "_XMP_loop_init_" + loopIndexName, loopIndexType);
    Ident parallelCondId = declIdentWithBlock(schedBaseBlock,
                                              "_XMP_loop_cond_" + loopIndexName, loopIndexType);
    Ident parallelStepId = declIdentWithBlock(schedBaseBlock,
                                              "_XMP_loop_step_" + loopIndexName, loopIndexType);

    XMPutil.putLoopIter(schedBaseBlock, loopIndexName,
                        Xcons.List(parallelInitId, parallelCondId, parallelStepId));

    forBlock.setLowerBound(parallelInitId.Ref());
    forBlock.setUpperBound(parallelCondId.Ref());
    forBlock.setStep(parallelStepId.Ref());

    forBlock.getCondBBlock().setExpr(Xcons.binaryOp(Xcode.LOG_LT_EXPR, loopIndex, parallelCondId.Ref()));
  }

  static Ident declIdentWithBlock(Block b, String identName, Xtype type)
  {
    Block bb = getOuterSchedPoint(b);
    BlockList bl = bb.getParent();

    // FIXME consider variable scope
    return bl.declLocalIdent(identName, type);
  }

  private Block createCommTaskBlock(BlockList body, String execFuncSuffix, XobjList execFuncArgs) throws XMPexception {
    // setup barrier finalizer
    setupFinalizer(body, _globalDecl.declExternFunc("_XMP_end_task"), null);
    
    // create function call
    BlockList taskBody = Bcons.emptyBody();
    Ident taskDescId = taskBody.declLocalIdent("_XMP_TASK_desc", Xtype.voidPtrType, StorageClass.AUTO,
                                               Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)));
    execFuncArgs.cons(taskDescId.getAddr());
    Ident execFuncId = _globalDecl.declExternFunc("_XMP_exec_task_" + execFuncSuffix, Xtype.intType);
    Block execBlock = Bcons.IF(BasicBlock.Cond(execFuncId.Call(execFuncArgs)), body, null);
    taskBody.add(execBlock);

    Ident taskFinalizeId = _globalDecl.declExternFunc("_XMP_exec_task_NODES_FINALIZE", Xtype.voidType);
    XobjList args = Xcons.List(Xcode.POINTER_REF, taskDescId.Ref());
    taskBody.add(taskFinalizeId.Call(args));
    
    return Bcons.COMPOUND(taskBody);
  }

  // private Block createCommTaskBlock(BlockList body, PragmaBlock pb) throws XMPexception {
  //   BlockList taskBody = Bcons.emptyBody();

  //   XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(pb.getClauses().getArg(1), _globalDecl, pb);

  //   // when '*' = the executing node set is specified
  //   if (on_ref == null) return Bcons.COMPOUND(pb.getBody());

  //   Block b = on_ref.buildConstructor(_globalDecl);
  //   BasicBlock bb = b.getBasicBlock();

  //   Ident taskNodesDescId = pb.getParentBlock().getBody().declLocalIdent(tmpSym.getStr("_XMP_TASK_NODES"),
  // 									 Xtype.voidPtrType);

  //   Ident f1 = _globalDecl.declExternFunc("_XMP_create_task_nodes", Xtype.voidType);
  //   bb.add(f1.callSubroutine(Xcons.List(taskNodesDescId.getAddr(), on_ref.getDescId().Ref())));

  //   Ident f2 = _globalDecl.declExternFunc("_XMP_test_task_on_nodes", Xtype.intType);
  //   Xobject cond = f2.Call(Xcons.List(taskNodesDescId.Ref()));
  //   taskBody.add(Bcons.IF(cond, Bcons.COMPOUND(body), null));

  //   Ident f3 = _globalDecl.declExternFunc("_XMP_end_task", Xtype.voidType);
  //   pb.getBody().add(f3.Call(Xcons.List()));

  //   Ident g1 = _globalDecl.declExternFunc("_XMP_finalize_nodes", Xtype.voidType);
  //   Ident g2 = _globalDecl.declExternFunc("_XMP_ref_dealloc", Xtype.voidType);
  //   taskBody.add(g1.Call(Xcons.List(taskNodesDescId.Ref())));
  //   taskBody.add(g2.Call(Xcons.List(on_ref.getDescId().Ref())));

  //   return Bcons.COMPOUND(taskBody);
  // }
  
  private void translateBarrier(PragmaBlock pb) throws XMPexception {

    BlockList ret_body = Bcons.emptyBody();

    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(pb.getClauses().getArg(0), _globalDecl, pb);
    Xobject on_ref_arg;

    if (on_ref != null){
      ret_body.add(on_ref.buildConstructor(_globalDecl));
      on_ref_arg = on_ref.getDescId().Ref();
    }
    else on_ref_arg = Xcons.Cnull();

    Ident f = _globalDecl.declExternFunc("_XMP_barrier", Xtype.voidType);
    ret_body.add(f.Call(Xcons.List(on_ref_arg)));

    if (on_ref != null){
      Ident g = _globalDecl.declExternFunc("_XMP_ref_dealloc", Xtype.voidType);
      ret_body.add(g.Call(Xcons.List(on_ref_arg)));
    }

    pb.replace(Bcons.COMPOUND(ret_body));
  }

  // private void translateBarrier(PragmaBlock pb) throws XMPexception {
  //   // start translation
  //   XobjList barrierDecl = (XobjList)pb.getClauses();
  //   XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);

  //   // create function call
  //   Block barrierFuncCallBlock = null;
  //   XobjList onRef = (XobjList)barrierDecl.getArg(0);
  //   if (onRef == null || onRef.Nargs() == 0) {
  //     barrierFuncCallBlock = _globalDecl.createFuncCallBlock("_XMP_barrier_EXEC", null);
  //   }
  //   else {
  //     XMPquadruplet<String, Boolean, XobjList, XMPobject> execOnRefArgs = createExecOnRefArgs(onRef, pb);
  //     String execFuncSuffix = execOnRefArgs.getFirst();
  //     boolean splitComm = execOnRefArgs.getSecond().booleanValue();
  //     XobjList execFuncArgs = execOnRefArgs.getThird();
  //     if (splitComm) {
  //       BlockList barrierBody = Bcons.blockList(_globalDecl.createFuncCallBlock("_XMP_barrier_EXEC", null));
  // 	barrierFuncCallBlock = createCommTaskBlock(barrierBody, execFuncSuffix, execFuncArgs);
  //     }
  //     else {
  // 	barrierFuncCallBlock = _globalDecl.createFuncCallBlock("_XMP_barrier_" + execFuncSuffix, execFuncArgs);
  //     }
  //   }

  //   pb.replace(barrierFuncCallBlock);

  //   Xobject profileClause = barrierDecl.getArg(1);
  //   addProfileFunctions(profileClause, barrierFuncCallBlock, "barrier", pb);
  // }


  private Xobject setStartLengthSize(Xobject obj, Xtype varType, int dims, Xobject[] start, Xobject[] length, Xobject[] size) throws XMPexception {
    Xobject total_length = null;
    
    for(int j=0;j<dims;j++){
      Xobject triplet = obj.getArg(1).getArg(j);
      size[j] = XMPutil.getArrayElmt(varType, j);
      if(triplet.isVariable() || triplet.isIntConstant()){
        start[j]  = triplet;
        length[j] = Xcons.IntConstant(1);
      }
      else{
        start[j] = triplet.getArg(0);
        if(triplet.getArg(1).isVariable() || triplet.getArg(1).isIntConstant()){
          length[j] = triplet.getArg(1);
        }
        else{
          length[j] = (triplet.getArg(1) == null || triplet.getArg(1).isEmpty())?
            Xcons.binaryOp(Xcode.MINUS_EXPR, size[j], start[j]) : triplet.getArg(1);
        }
      }
      total_length = (j==0)? length[j] : Xcons.binaryOp(Xcode.MUL_EXPR, total_length, length[j]);
    }
    return total_length;
  }


  private void setAccSize(int dims, Xobject[] size, Xobject[] acc_size){
    for(int j=0;j<dims;j++)
      for(int k=j+1;k<dims;k++)
        acc_size[j] = (k==j+1)? size[k] : Xcons.binaryOp(Xcode.MUL_EXPR, acc_size[j], size[k]);

    acc_size[dims-1] = Xcons.IntConstant(1);
  }

  private Xobject calcOffsetSize(int dims, Xobject[] start, Xobject[] acc_size, Xtype varType){
    Xobject offset_size = Xcons.binaryOp(Xcode.MUL_EXPR, start[0], acc_size[0]);
    for(int j=1;j<dims;j++){
      offset_size = Xcons.binaryOp(Xcode.PLUS_EXPR, offset_size, Xcons.binaryOp(Xcode.MUL_EXPR, start[j], acc_size[j]));
    }
    return Xcons.binaryOp(Xcode.MUL_EXPR, offset_size, Xcons.SizeOf(varType.getArrayElementType()));
  }
  
  private void createLocReduction(PragmaBlock pb, XobjList reductionRef, Xobject reductionOp, XobjList onRef) throws XMPexception {
    Xobject reductionVariable = reductionRef.getArg(1).getArg(0).getArg(0);
    int numLocationVariables = reductionRef.getArg(1).Nargs() - 1;

    // Create xmp_reduce_loc_init()
    BlockList reductionBody = Bcons.emptyBody();
    String reductionVariableName = reductionVariable.getName();
    Ident reductionVariableId = pb.findVarIdent(reductionVariableName);
    Xtype reductionVariableType = reductionVariableId.Type();
    XobjList args = Xcons.List(Xcons.IntConstant(numLocationVariables), Xcons.Cast(BasicType.longdoubleType, reductionVariable),
                               reductionVariableId.getAddr(), XMP.createBasicTypeConstantObj(reductionVariableType));
    reductionBody.add(_globalDecl.createFuncCallBlock("xmp_reduce_loc_init", args));

    // Create xmp_reduce_loc_set();
    for(int i=0;i<numLocationVariables;i++){
      Xobject reductionLocation = reductionRef.getArg(1).getArg(i+1);
      boolean is_scalar = reductionLocation.isVarRef();
      Xobject varAddr, varLength, varSize;
      if(is_scalar){ // scalar
        String varName = reductionLocation.getName();
        Ident varId = pb.findVarIdent(varName);
        Xtype varType = varId.Type();

        varAddr   = varId.getAddr();
        varLength = Xcons.IntConstant(1);
        varSize   = Xcons.SizeOf(varType);
      }
      else{ // array
        String varName = reductionLocation.getArg(0).getName();
        Ident varId = pb.findVarIdent(varName);
        Xtype varType = varId.Type();
        int dims = varType.getNumDimensions();

        Xobject[] start  = new Xobject[dims];
        Xobject[] length = new Xobject[dims];
        Xobject[] size   = new Xobject[dims];
        Xobject total_length = setStartLengthSize(reductionLocation, varType, dims, start, length, size);

        // Check the array is contiguous or not.
        // Note that when XMP runtime supports stride bcast communication,
        // the following if-statment will be removed.
        if(! check_contiguous_of_array(dims, length, size))
          throw new XMPexception("Stride bcast operation is not supported");

        Xobject[] acc_size = new Xobject[dims];
        setAccSize(dims, size, acc_size);
        Xobject offsetSize = calcOffsetSize(dims, start, acc_size, varType);

        varAddr   = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.Cast(Xtype.Pointer(BasicType.charType), varId.Ref()), offsetSize);
        varLength = total_length;
        varSize   = Xcons.SizeOf(varType.getArrayElementType());
      }
      
      args = Xcons.List(varAddr, varLength, varSize);
      reductionBody.add(_globalDecl.createFuncCallBlock("xmp_reduce_loc_set", args));
    }

    // Create xmp_reduce_loc_execute()
    args = Xcons.List(reductionOp);
    reductionBody.add(_globalDecl.createFuncCallBlock("xmp_reduce_loc_execute", args));

    // Output
    Block reductionBodyBlock = null;
    if (onRef != null && onRef.Nargs() != 0){
      XMPquadruplet<String, Boolean, XobjList, XMPobject> execOnRefArgs = createExecOnRefArgs(onRef, pb);
      String execFuncSuffix = execOnRefArgs.getFirst();
      boolean splitComm     = execOnRefArgs.getSecond().booleanValue();
      XobjList execFuncArgs = execOnRefArgs.getThird();
      if(splitComm){
        reductionBodyBlock = createCommTaskBlock(reductionBody, execFuncSuffix, execFuncArgs);
      }
      else{
        reductionBodyBlock = Bcons.COMPOUND(reductionBody);
      }
    }
    else{
      reductionBodyBlock = Bcons.COMPOUND(reductionBody);
    }
    
    pb.replace(reductionBodyBlock);
  }

  private Block createReductionFuncCallBlock(PragmaBlock pb, Boolean isAcc) throws XMPexception {

    BlockList ret_body = Bcons.emptyBody();
    
    XobjList reductionRef = (XobjList)pb.getClauses().getArg(0);

    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(pb.getClauses().getArg(1), _globalDecl, pb);
    Xobject on_ref_arg;

    if (on_ref != null){
      if (XMPpragma.valueOf(pb.getPragma()) != XMPpragma.LOOP){
	ret_body.add(on_ref.buildConstructor(_globalDecl));
	on_ref_arg = on_ref.getDescId().Ref();
      }
      else {
	XMPobjectsRef on_ref_copy = on_ref.convertLoopToReduction();
	ret_body.add(on_ref_copy.buildConstructor(_globalDecl));
	on_ref_arg = on_ref_copy.getDescId().Ref();
      }
    }
    else on_ref_arg = Xcons.Cnull();

    Ident f = _globalDecl.declExternFunc(isAcc? "_XMP_reduction_acc" : "_XMP_reduction", Xtype.voidType);
    Ident f2 = _globalDecl.declExternFunc("_XMP_reduction_loc", Xtype.voidType);

    Vector<XobjList> reductionFuncArgsList = createReductionArgsList(reductionRef, pb, false, null, null);

    Vector<Vector<Ident>> pos_vars = new Vector<Vector<Ident>>();
    for (Xobject w: (XobjList)reductionRef.getArg(1)){
      Vector<Ident> plist = new Vector<Ident>();
      if (w.getArg(1) != null){
	for (Xobject p: (XobjList)w.getArg(1)){
	  Ident pid = pb.findVarIdent(p.getName());
	  if (pid == null){
	    XMP.error(pb.getLineNo(), "variable '" + p.getName() + "' in reduction is not found");
	  }
	  plist.add(pid);
	}
      }
      pos_vars.add(plist);
    }

    int i = 0;
    for (XobjList args : reductionFuncArgsList){
      int j = 0;
      Vector<Ident> plist = pos_vars.elementAt(i);
      for (Ident pos: plist){
      	Xobject args2 = Xcons.List(Xcons.IntConstant(j), pos.Ref(), XMP.createBasicTypeConstantObj(pos.Type()));
      	ret_body.add(f2.Call(args2));
      	j++;
      }
      args.add(on_ref_arg);
      args.add(Xcons.IntConstant(plist.size()));
      ret_body.add(f.Call(args));
      i++;
    }      
    
    if (on_ref != null){
      Ident g = _globalDecl.declExternFunc("_XMP_ref_dealloc", Xtype.voidType);
      ret_body.add(g.Call(Xcons.List(on_ref.getDescId().Ref())));
    }

    // if (isAcc){
    //   XobjList vars = Xcons.List();
    //   for(Ident id : info.getReductionVars()){
    //     vars.add(id.Ref());
    //   }
    //   ret_body = Bcons.blockList(buildAccHostData(vars, ret_body));
    // }

    return Bcons.COMPOUND(ret_body);

  }

  private void translateReduction(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList reductionDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);

    // Check host or acc clause for XcalableACC
    XobjList accOrHost = (XobjList)reductionDecl.getArg(3);
    boolean isHost     = accOrHost.hasIdent("host");
    boolean isACC      = accOrHost.hasIdent("acc");

    if(!isHost && !isACC){
      isHost = true;
    }
    else if(isHost && isACC){
      throw new XMPexception(pb.getLineNo(), "reduction for both acc and host is unimplemented");
    }

    XobjList reductionRef = (XobjList)reductionDecl.getArg(0);
    XobjInt reductionOp = (XobjInt)reductionRef.getArg(0);
    XobjList onRef = (XobjList)reductionDecl.getArg(1);

    // When MAXLOC or MINLOC, another flow, which does not use a variadic function in runtime, is executed.
    if(reductionOp.getInt() == XMPcollective.REDUCE_MAXLOC || reductionOp.getInt() == XMPcollective.REDUCE_MINLOC){
      createLocReduction(pb, reductionRef, reductionOp, onRef);
      return;
    }

    Block reductionFuncCallBlock = createReductionFuncCallBlock(pb, isACC);

    // // create function arguments
    // XobjList reductionRef = (XobjList)reductionDecl.getArg(0);
    // XobjInt reductionOp = (XobjInt)reductionRef.getArg(0);

    // // create function call
    // XobjList onRef = (XobjList)reductionDecl.getArg(1);
    
    // // When MAXLOC or MINLOC, another flow, which does not use a variadic function in runtime, is executed.
    // if(reductionOp.getInt() == XMPcollective.REDUCE_MAXLOC || reductionOp.getInt() == XMPcollective.REDUCE_MINLOC){
    //   createLocReduction(pb, reductionRef, reductionOp, onRef);
    //   return;
    // }

    // Block reductionFuncCallBlock = null;
    // Vector<XobjList> reductionFuncArgsList = createReductionArgsList(reductionRef, pb, false, null, null);
    // String reductionFuncType = createReductionFuncType(reductionRef, pb, isACC);

    // if (onRef == null || onRef.Nargs() == 0){
    //   reductionFuncCallBlock = createReductionFuncCallBlock(true, reductionFuncType + "_EXEC", null, reductionFuncArgsList);
    // }
    // else{
    //   XMPquadruplet<String, Boolean, XobjList, XMPobject> execOnRefArgs = createExecOnRefArgs(onRef, pb);
    //   String execFuncSuffix = execOnRefArgs.getFirst();
    //   boolean splitComm = execOnRefArgs.getSecond().booleanValue();
    //   XobjList execFuncArgs = execOnRefArgs.getThird();
    //   if (splitComm) {
    //     BlockList reductionBody =
    //       Bcons.blockList(createReductionFuncCallBlock(true, reductionFuncType + "_EXEC", null, reductionFuncArgsList));
    // 	reductionFuncCallBlock = createCommTaskBlock(reductionBody, execFuncSuffix, execFuncArgs);
    //   }
    //   else {
    //     reductionFuncCallBlock = createReductionFuncCallBlock(false, reductionFuncType + "_" + execFuncSuffix,
    //                                                           execFuncArgs.operand(), reductionFuncArgsList);
    //   }
    // }

    if(isACC){
      XobjList vars = Xcons.List();
      XobjList reductionSpecList = (XobjList)reductionRef.getArg(1);
      for(Xobject x : reductionSpecList){
        vars.add(x.getArg(0));
      }
      reductionFuncCallBlock = Bcons.PRAGMA(Xcode.ACC_PRAGMA, "HOST_DATA", Xcons.List(Xcons.List(Xcons.String("USE_DEVICE"), vars)),
					    Bcons.blockList(reductionFuncCallBlock));
    }

    Xobject async = reductionDecl.getArg(2);
    if(async.Opcode() != Xcode.LIST){
      if(!XmOption.isAsync())
	XMP.error(pb.getLineNo(), "MPI-3 is required to use the async clause on a reduction directive");

      BlockList bl = reductionFuncCallBlock.getBody();
      bl.insert(_globalDecl.declExternFunc("xmpc_init_async").Call(Xcons.List(async)));
      bl.add(_globalDecl.declExternFunc("xmpc_start_async").Call(Xcons.List()));
    }
    
    pb.replace(reductionFuncCallBlock);
    
    Xobject profileClause = reductionDecl.getArg(4);
    addProfileFunctions(profileClause, reductionFuncCallBlock, "reduction", pb);
  }

  private String createReductionFuncType(XobjList reductionRef, PragmaBlock pb, boolean isACC) throws XMPexception {
    XobjInt reductionOp = (XobjInt)reductionRef.getArg(0);
    switch (reductionOp.getInt()) {
      case XMPcollective.REDUCE_SUM:
      case XMPcollective.REDUCE_MINUS:
      case XMPcollective.REDUCE_PROD:
      case XMPcollective.REDUCE_BAND:
      case XMPcollective.REDUCE_LAND:
      case XMPcollective.REDUCE_BOR:
      case XMPcollective.REDUCE_LOR:
      case XMPcollective.REDUCE_BXOR:
      case XMPcollective.REDUCE_LXOR:
      case XMPcollective.REDUCE_MAX:
      case XMPcollective.REDUCE_MIN:
      case XMPcollective.REDUCE_MAXLOC:
      case XMPcollective.REDUCE_MINLOC:
        return isACC? new String("reduce_acc") : new String("reduce");
      case XMPcollective.REDUCE_FIRSTMAX:
      case XMPcollective.REDUCE_FIRSTMIN:
      case XMPcollective.REDUCE_LASTMAX:
      case XMPcollective.REDUCE_LASTMIN:
        return isACC? new String("reduce_acc_FLMM") : new String("reduce_FLMM");
      default:
        throw new XMPexception("unknown reduce operation");
    }
  }

  private Vector<XobjList> createReductionArgsList(XobjList reductionRef, PragmaBlock pb, boolean isClause,
                                                   CforBlock schedBaseBlock, IfBlock reductionInitIfBlock) throws XMPexception {
    Vector<XobjList> returnVector = new Vector<XobjList>();
    XobjInt reductionOp = (XobjInt)reductionRef.getArg(0);
    XobjList reductionSpecList = (XobjList)reductionRef.getArg(1);

    for (XobjArgs i = reductionSpecList.getArgs(); i != null; i = i.nextArgs()) {
      XobjList reductionSpec = (XobjList)i.getArg();
      String specName = reductionSpec.getArg(0).getString();
      Ident specId;
      Xtype specType;
      
      XMPalignedArray specAlignedArray = _globalDecl.getXMPalignedArray(specName, pb);
      if (specAlignedArray != null){
	// specName is an aligned array and its original entry of the symbol table may have been removed.
	specId = specAlignedArray.getAddrId();
	specType = specAlignedArray.getArrayType();
      }
      else {
	XMPpair<Ident, Xtype> typedSpec = XMPutil.findTypedVar(specName, pb);
	specId = typedSpec.getFirst();
	specType = typedSpec.getSecond();
      }

      boolean isArray = false;
      boolean isPointer = false;
      Xobject specRef = null;
      Xobject count = null;
      Xobject elmtType = null;
      BasicType basicSpecType = null;
      switch (specType.getKind()) {
      case Xtype.BASIC:
	{
	  basicSpecType = (BasicType)specType;
	  checkReductionType(specName, basicSpecType);
	  
	  specRef = specId.getAddr();
	  count = Xcons.LongLongConstant(0, 1);
	  elmtType = XMP.createBasicTypeConstantObj(basicSpecType);
	} break;
      case Xtype.ARRAY:
	{
	  isArray = true;
	  ArrayType arraySpecType = (ArrayType)specType;
	  if (arraySpecType.getArrayElementType().getKind() != Xtype.BASIC)
	    throw new XMPexception("array '" + specName + "' has a wrong data type for reduction");
	  
	  basicSpecType = (BasicType)arraySpecType.getArrayElementType();
	  checkReductionType(specName, basicSpecType);
	  
	  // FIXME not good implementation
	  XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
	  //XMPalignedArray specAlignedArray   = _globalDecl.getXMPalignedArray(specName, pb);
	  if(specAlignedArray == null){
	    specRef = specId.Ref();
	    count = Xcons.LongLongConstant(0, XMPutil.getArrayElmtCount(arraySpecType));
	  }
	  else{
	    if(isClause){
	      throw new XMPexception("aligned arrays cannot be used in reduction clause");
	    }
	    
	    if (specAlignedArray.hasShadow()) {
	      throw new XMPexception("arrays which have shadow cannot be used in reduction directive/clause");
	    }
	    
	    specRef = specAlignedArray.getAddrIdVoidRef();
	    Ident getTotalElmtFuncId = _globalDecl.declExternFunc("_XMP_get_array_total_elmts",
								  Xtype.unsignedlonglongType);
	    count = getTotalElmtFuncId.Call(Xcons.List(specAlignedArray.getDescId().Ref()));
	  }

	  elmtType = XMP.createBasicTypeConstantObj(basicSpecType);
	} break;
        case Xtype.POINTER:
        {
          isPointer = true;
          basicSpecType = (BasicType)specType.getRef();
          specRef = specId.getAddr();
          count = Xcons.LongLongConstant(0, 1);
          elmtType = XMP.createBasicTypeConstantObj(basicSpecType);
        } break;
      default:
	throw new XMPexception("'" + specName + "' has a wrong data type for reduction");
      }

      XobjList reductionFuncArgs = Xcons.List(specRef, count, elmtType, reductionOp);
      // declare temp variable for reduction
      if (isClause) {
        createReductionInitStatement(specId, isArray, count, basicSpecType, reductionOp.getInt(),
                                     schedBaseBlock, reductionInitIfBlock);
      }
      if(isPointer){
	Xobject varaddr = (Xobject)reductionFuncArgs.getArg(0);
	reductionFuncArgs.setArg(0, Xcons.PointerRef(varaddr));
      }

      // add extra args for (firstmax, firstmin, lastmax, lastmin) if needed
      createFLMMreductionArgs(reductionOp.getInt(), (XobjList)reductionSpec.getArg(1), reductionFuncArgs, pb);
      returnVector.add(reductionFuncArgs);
    }

    return returnVector;
  }

  private void createReductionInitStatement(Ident varId, boolean isArray, Xobject count, BasicType type, int reductionOp,
                                            CforBlock schedBaseBlock, IfBlock reductionInitIfBlock) throws XMPexception {
    if (!needsInitialization(reductionOp)) {
      return;
    }

    BlockList initPart = reductionInitIfBlock.getThenBody();
    if (initPart == null) {
      initPart = Bcons.emptyBody();
      reductionInitIfBlock.setThenBody(initPart);
    }

    Xobject statement = null;
    if (isArray) {
      Ident initLoopIndexId = declIdentWithBlock(schedBaseBlock, _globalDecl.genSym(XMP.TEMP_PREFIX), Xtype.unsignedlonglongType);
      initPart.add(createReductionArrayInit(varId, createReductionInitValueObj(varId, type, reductionOp), count, type, initLoopIndexId));
    }
    else {
      initPart.add(Xcons.Set(varId.Ref(), createReductionInitValueObj(varId, type, reductionOp)));
    }
  }

  private boolean needsInitialization(int reductionOp) throws XMPexception {
    switch (reductionOp) {
      case XMPcollective.REDUCE_SUM:
      case XMPcollective.REDUCE_MINUS:
      case XMPcollective.REDUCE_PROD:
      case XMPcollective.REDUCE_BXOR:
      case XMPcollective.REDUCE_LXOR:
        return true;
      case XMPcollective.REDUCE_BAND:
      case XMPcollective.REDUCE_LAND:
      case XMPcollective.REDUCE_BOR:
      case XMPcollective.REDUCE_LOR:
      case XMPcollective.REDUCE_MAX:
      case XMPcollective.REDUCE_MIN:
      case XMPcollective.REDUCE_FIRSTMAX:
      case XMPcollective.REDUCE_FIRSTMIN:
      case XMPcollective.REDUCE_LASTMAX:
      case XMPcollective.REDUCE_LASTMIN:
        return false;
      default:
        throw new XMPexception("unknown reduce operation");
    }
  }

  private Block createReductionArrayInit(Ident tempId, Xobject initValueObj, Xobject count, BasicType type,
                                         Ident loopIndexId) {
    Xobject loopIndexRef = loopIndexId.Ref();

    Xobject tempArrayRef = Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.Cast(Xtype.Pointer(type), tempId.Ref()),
							   loopIndexRef));

    Block loopBlock = Bcons.FOR(Xcons.Set(loopIndexRef, Xcons.IntConstant(0)),
                                Xcons.binaryOp(Xcode.LOG_LT_EXPR, loopIndexRef, count),
                                Xcons.asgOp(Xcode.ASG_PLUS_EXPR, loopIndexRef, Xcons.IntConstant(1)),
                                Bcons.Statement(Xcons.Set(tempArrayRef, initValueObj)));

    return loopBlock;
  }

  // FIXME
  private Xobject createReductionInitValueObj(Ident varId, BasicType type, int reductionOp) throws XMPexception {
    Xobject varRef = varId.Ref();
    Xobject intZero = Xcons.IntConstant(0);
    Xobject intOne = Xcons.IntConstant(1);
    Xobject floatZero = Xcons.FloatConstant(0.);
    Xobject floatOne = Xcons.FloatConstant(1.);

    switch (type.getBasicType()) {
      case BasicType.BOOL:
        // FIXME correct???
        return selectReductionInitValueObj(reductionOp, varRef, intZero, intOne);
      case BasicType.CHAR:
      case BasicType.UNSIGNED_CHAR:
      case BasicType.SHORT:
      case BasicType.UNSIGNED_SHORT:
      case BasicType.INT:
      case BasicType.UNSIGNED_INT:
      case BasicType.LONG:
      case BasicType.UNSIGNED_LONG:
      case BasicType.LONGLONG:
      case BasicType.UNSIGNED_LONGLONG:
        return selectReductionInitValueObj(reductionOp, varRef, intZero, intOne);
      case BasicType.FLOAT:
      case BasicType.DOUBLE:
      case BasicType.LONG_DOUBLE:
        return selectReductionInitValueObj(reductionOp, varRef, floatZero, floatOne);
      case BasicType.FLOAT_IMAGINARY:
      case BasicType.DOUBLE_IMAGINARY:
      case BasicType.LONG_DOUBLE_IMAGINARY:
      case BasicType.FLOAT_COMPLEX:
      case BasicType.DOUBLE_COMPLEX:
      case BasicType.LONG_DOUBLE_COMPLEX:
        // FIXME not implemented yet
        throw new XMPexception("not implemented yet");
      default:
        throw new XMPexception("wrong data type for reduction");
    }
  }

  // FIXME needs type checking
  private Xobject selectReductionInitValueObj(int reductionOp, Xobject varRef, Xobject zero, Xobject one) throws XMPexception {
    switch (reductionOp) {
      case XMPcollective.REDUCE_SUM:
      case XMPcollective.REDUCE_MINUS:
        return zero;
      case XMPcollective.REDUCE_PROD:
        return one;
      case XMPcollective.REDUCE_BAND:
      case XMPcollective.REDUCE_LAND:
      case XMPcollective.REDUCE_BOR:
      case XMPcollective.REDUCE_LOR:
      case XMPcollective.REDUCE_MAX:
      case XMPcollective.REDUCE_MIN:
      case XMPcollective.REDUCE_FIRSTMAX:
      case XMPcollective.REDUCE_FIRSTMIN:
      case XMPcollective.REDUCE_LASTMAX:
      case XMPcollective.REDUCE_LASTMIN:
        throw new XMPexception("the operation does not need initialization");
      case XMPcollective.REDUCE_BXOR:
        return Xcons.unaryOp(Xcode.BIT_NOT_EXPR, varRef);
      case XMPcollective.REDUCE_LXOR:
        return Xcons.unaryOp(Xcode.LOG_NOT_EXPR, varRef);
      default:
        throw new XMPexception("unknown reduce operation");
    }
  }

  private void createFLMMreductionArgs(int op, XobjList locationVars, XobjList funcArgs, PragmaBlock pb) throws XMPexception {
    switch (op) {
      case XMPcollective.REDUCE_SUM:
      case XMPcollective.REDUCE_MINUS:
      case XMPcollective.REDUCE_PROD:
      case XMPcollective.REDUCE_BAND:
      case XMPcollective.REDUCE_LAND:
      case XMPcollective.REDUCE_BOR:
      case XMPcollective.REDUCE_LOR:
      case XMPcollective.REDUCE_BXOR:
      case XMPcollective.REDUCE_LXOR:
      case XMPcollective.REDUCE_MAX:
      case XMPcollective.REDUCE_MIN:
      case XMPcollective.REDUCE_MAXLOC:
      case XMPcollective.REDUCE_MINLOC:
        return;
      case XMPcollective.REDUCE_FIRSTMAX:
      case XMPcollective.REDUCE_FIRSTMIN:
      case XMPcollective.REDUCE_LASTMAX:
      case XMPcollective.REDUCE_LASTMIN:
        break;
      default:
        throw new XMPexception("unknown reduce operation");
    }

    funcArgs.add(Xcons.IntConstant(XMPutil.countElmts(locationVars)));

    // check <location-variables> and add to funcArgs
    for (XobjArgs i = locationVars.getArgs(); i != null; i = i.nextArgs()) {
      String varName = i.getArg().getString();

      XMPpair<Ident, Xtype> typedVar = XMPutil.findTypedVar(varName, pb);
      Ident varId = typedVar.getFirst();
      Xtype varType = typedVar.getSecond();

      switch (varType.getKind()) {
        case Xtype.BASIC:
          {
	    //            if (!XMPutil.isIntegerType(varType))
	    //              throw new XMPexception("'" + varName + "' should have a integer type for reduction");

            BasicType basicVarType = (BasicType)varType;

            funcArgs.add(Xcons.Cast(Xtype.voidPtrType, varId.getAddr()));
            funcArgs.add(Xcons.Cast(Xtype.intType, XMP.createBasicTypeConstantObj(basicVarType)));
          } break;
        case Xtype.ARRAY:
          {
            ArrayType arrayVarType = (ArrayType)varType;
            if (arrayVarType.getArrayElementType().getKind() != Xtype.BASIC)
              throw new XMPexception("array '" + varName + "' has has a wrong data type for reduction");

            BasicType basicVarType = (BasicType)arrayVarType.getArrayElementType();

            if (!XMPutil.isIntegerType(basicVarType))
              throw new XMPexception("'" + varName + "' should have a integer type for reduction");

            funcArgs.add(Xcons.Cast(Xtype.voidPtrType, varId.Ref()));
            funcArgs.add(Xcons.Cast(Xtype.intType, XMP.createBasicTypeConstantObj(basicVarType)));
          } break;
        case Xtype.POINTER:
  	  {
	    PointerType ptrVarType = (PointerType)varType;
	    BasicType basicVarType = (BasicType)ptrVarType.getRef();
	    funcArgs.add(Xcons.Cast(Xtype.voidPtrType, varId.Ref()));
	    funcArgs.add(Xcons.Cast(Xtype.intType, XMP.createBasicTypeConstantObj(basicVarType)));
	  } break;
        default:
          throw new XMPexception("'" + varName + "' has a wrong data type for reduction");
      }
    }
  }

  private void checkReductionType(String name, BasicType type) throws XMPexception {
    switch (type.getBasicType()) {
      case BasicType.BOOL:
      case BasicType.CHAR:
      case BasicType.UNSIGNED_CHAR:
      case BasicType.SHORT:
      case BasicType.UNSIGNED_SHORT:
      case BasicType.INT:
      case BasicType.UNSIGNED_INT:
      case BasicType.LONG:
      case BasicType.UNSIGNED_LONG:
      case BasicType.LONGLONG:
      case BasicType.UNSIGNED_LONGLONG:
      case BasicType.FLOAT:
      case BasicType.DOUBLE:
      case BasicType.LONG_DOUBLE:
      case BasicType.FLOAT_IMAGINARY:
      case BasicType.DOUBLE_IMAGINARY:
      case BasicType.LONG_DOUBLE_IMAGINARY:
      case BasicType.FLOAT_COMPLEX:
      case BasicType.DOUBLE_COMPLEX:
      case BasicType.LONG_DOUBLE_COMPLEX:
        break;
      default:
        throw new XMPexception("'" + name + "' has a wrong data type for reduction");
    }
  }

  private Block createReductionFuncCallBlock(boolean isMacroFunc, String funcType,
                                             Xobject execDesc, Vector<XobjList> funcArgsList) {
    Ident funcId = null;
    if (isMacroFunc) funcId = XMP.getMacroId("_XMP_M_" + funcType.toUpperCase());
    else             funcId = _globalDecl.declExternFunc("_XMP_" + funcType);

    BlockList funcCallList = Bcons.emptyBody();
    Iterator<XobjList> it = funcArgsList.iterator();
    while (it.hasNext()) {
      XobjList funcArgs = it.next();
      if (execDesc != null) funcArgs.cons(execDesc);

      funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
    }
    return Bcons.COMPOUND(funcCallList);
  }

  private Block createBcastFuncCallBlock(PragmaBlock pb, boolean isAcc) throws XMPexception {

    BlockList ret_body = Bcons.emptyBody();

    XMPobjectsRef from_ref = XMPobjectsRef.parseDecl(pb.getClauses().getArg(1), _globalDecl, pb);
    Xobject from_ref_arg;
    if (from_ref != null){
      ret_body.add(from_ref.buildConstructor(_globalDecl));
      from_ref_arg = from_ref.getDescId().Ref();
    }
    else from_ref_arg = Xcons.Cnull();

    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(pb.getClauses().getArg(2), _globalDecl, pb);
    Xobject on_ref_arg;
    if (on_ref != null){
      ret_body.add(on_ref.buildConstructor(_globalDecl));
      on_ref_arg = on_ref.getDescId().Ref();
    }
    else on_ref_arg = Xcons.Cnull();

    Ident f = _globalDecl.declExternFunc(isAcc? "_XMP_bcast_acc" : "_XMP_bcast", Xtype.voidType);

    for (Xobject v: (XobjList)pb.getClauses().getArg(0)){
      XobjList args = createBcastArgsList(v, pb);
      args.add(from_ref_arg);
      args.add(on_ref_arg);
      ret_body.add(f.Call(args));
    }

    if (on_ref != null){
      Ident g = _globalDecl.declExternFunc("_XMP_ref_dealloc", Xtype.voidType);
      ret_body.add(g.Call(Xcons.List(on_ref_arg)));
    }

    if (from_ref != null){
      Ident g = _globalDecl.declExternFunc("_XMP_ref_dealloc", Xtype.voidType);
      ret_body.add(g.Call(Xcons.List(from_ref_arg)));
    }

    return Bcons.COMPOUND(ret_body);
  }
  
  private void translateBcast(PragmaBlock pb) throws XMPexception {
    // start translation
    XobjList bcastDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);

    // acc or host
    XobjList accOrHost = (XobjList)bcastDecl.getArg(4);
    boolean isACC = accOrHost.hasIdent("acc");
    boolean isHost = accOrHost.hasIdent("host");
    if(!isACC && !isHost){
      isHost = true;
    }
    if(isACC && isHost){
      throw new XMPexception(pb.getLineNo(), "bcast for both acc and host is unimplemented");
    }
    
    Block bcastFuncCallBlock = createBcastFuncCallBlock(pb, isACC);
    // // create function arguments
    // XobjList varList = (XobjList)bcastDecl.getArg(0);
    // Vector<XobjList> bcastArgsList = createBcastArgsList(varList, pb);

    // // create function call
    // Block bcastFuncCallBlock = null;
    // XobjList fromRef = (XobjList)bcastDecl.getArg(1);
    // XMPpair<String, XobjList> execFromRefArgs = null;
    // if(fromRef != null && fromRef.Nargs() != 0){
    //   execFromRefArgs = createExecFromRefArgs(fromRef, pb);
    // }

    // XobjList onRef = (XobjList)bcastDecl.getArg(2);
    // boolean splitComm = false;
    // if(onRef == null || onRef.getArgs() == null) {
    //   bcastFuncCallBlock = createBcastFuncCallBlock(true, "EXEC", null, bcastArgsList, execFromRefArgs, isACC);
    // }
    // else{
    //   XMPquadruplet<String, Boolean, XobjList, XMPobject> execOnRefArgs = createExecOnRefArgs(onRef, pb);
    //   String execFuncSuffix = execOnRefArgs.getFirst();
    //   splitComm = execOnRefArgs.getSecond().booleanValue();
    //   XobjList execFuncArgs = execOnRefArgs.getThird();
    //   if(splitComm){
    //     BlockList bcastBody = Bcons.blockList(createBcastFuncCallBlock(true, "EXEC", null, bcastArgsList,
    //                                                                    execFromRefArgs, isACC));
    // 	bcastFuncCallBlock = createCommTaskBlock(bcastBody, execFuncSuffix, execFuncArgs);
    //   }
    //   else{
    // 	bcastFuncCallBlock = createBcastFuncCallBlock(false, execFuncSuffix, execFuncArgs.operand(),
    //                                                   bcastArgsList, execFromRefArgs, isACC);
    //   }
    // }
    
    if(isACC){
      XobjList varList = (XobjList)bcastDecl.getArg(0);
      bcastFuncCallBlock = Bcons.PRAGMA(Xcode.ACC_PRAGMA, "HOST_DATA",
                                        Xcons.List(Xcons.List(Xcons.String("USE_DEVICE"),varList)),
                                        Bcons.blockList(bcastFuncCallBlock));
    }

    Xobject async = bcastDecl.getArg(3);
    if(async.Opcode() != Xcode.LIST){
      if(!XmOption.isAsync()){
	XMP.error(pb.getLineNo(), "MPI-3 is required to use the async clause on a bcast directive");
      }

      BlockList bl = bcastFuncCallBlock.getBody();
      
      bl.insert(_globalDecl.declExternFunc("xmpc_init_async").Call(Xcons.List(async)));
      bl.add(_globalDecl.declExternFunc("xmpc_start_async").Call(Xcons.List()));
    }

    pb.replace(bcastFuncCallBlock);

    Xobject profileClause = bcastDecl.getArg(5);
    addProfileFunctions(profileClause, bcastFuncCallBlock, "bcast", pb);
  }

  private boolean check_all(Xobject length, Xobject size){
    return length.equals(size);
  }
  
  private boolean check_one(Xobject length){
    return length.equals(Xcons.IntConstant(1));
  }
  
  private boolean check_contiguous_of_array(int dims, Xobject length[], Xobject size[]) throws XMPexception{
    boolean is_contiguous = false;

    switch (dims){
    case 1: is_contiguous = true;
      break;
    case 2:
      if(check_one(length[0]) || check_all(length[1], size[1]))
        is_contiguous = true;
      break;
    case 3:
      if((check_one(length[0]) && check_one(length[1])) ||
         (check_one(length[0]) && check_all(length[2], size[2])) ||
         (check_all(length[1], size[1]) && check_all(length[2], size[2])))
        is_contiguous = true;
      break;
    case 4:
      if((check_one(length[0]) && check_one(length[1]) && check_one(length[2])) ||
         (check_one(length[0]) && check_one(length[1]) && check_all(length[3], size[3])) ||
         (check_one(length[0]) && check_all(length[2], size[2]) && check_all(length[3], size[3])) ||
         (check_all(length[1], size[1]) && check_all(length[2], size[2]) && check_all(length[3], size[3])))
        is_contiguous = true;
      break;
    case 5:
      if((check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_one(length[3])) ||
         (check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_all(length[4], size[4])) ||
         (check_one(length[0]) && check_one(length[1]) && check_all(length[3], size[3]) && check_all(length[4], size[4])) ||
         (check_one(length[0]) && check_all(length[2], size[2]) && check_all(length[3], size[3]) && check_all(length[4], size[4])) ||
         (check_all(length[1], size[1]) && check_all(length[2], size[2]) && check_all(length[3], size[3]) && check_all(length[4], size[4])))
        is_contiguous = true;
      break;
    case 6:
      if((check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_one(length[3]) && check_one(length[4])) ||
         (check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_one(length[3]) && check_all(length[5], size[5])) ||
         (check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_all(length[4], size[4]) && check_all(length[5], size[5])) ||
         (check_one(length[0]) && check_one(length[1]) && check_all(length[3], size[3]) && check_all(length[4], size[4]) && check_all(length[5], size[5])) ||
         (check_one(length[0]) && check_all(length[2], size[2]) && check_all(length[3], size[3]) && check_all(length[4], size[4]) && check_all(length[5], size[5])) ||
         (check_all(length[1], size[1]) && check_all(length[2], size[2]) && check_all(length[3], size[3]) && check_all(length[4], size[4]) && check_all(length[5], size[5])))
        is_contiguous = true;
      break;
    case 7:
      if((check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_one(length[3]) && check_one(length[4]) && check_one(length[5])) ||
         (check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_one(length[3]) && check_one(length[4]) && check_all(length[6], size[6])) ||
         (check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_one(length[3]) && check_all(length[5], size[5]) && check_all(length[6], size[6])) ||
         (check_one(length[0]) && check_one(length[1]) && check_one(length[2]) && check_all(length[4], size[4]) && check_all(length[5], size[5]) && check_all(length[6], size[6])) ||
         (check_one(length[0]) && check_one(length[1]) && check_all(length[3], size[3]) && check_all(length[4], size[4]) && check_all(length[5], size[5]) && check_all(length[6], size[6])) ||
         (check_one(length[0]) && check_all(length[2], size[2]) && check_all(length[3], size[3]) && check_all(length[4], size[4]) && check_all(length[5], size[5]) && check_all(length[6], size[6])) ||
         (check_all(length[1], size[1]) && check_all(length[2], size[2]) && check_all(length[3], size[3]) && check_all(length[4], size[4]) && check_all(length[5], size[5]) && check_all(length[6], size[6])))
        is_contiguous = true;
      break;
    default:
      throw new XMPexception("A wrong data type for broadcast");
    }

    return is_contiguous;
  }
  
  private XobjList createBcastArgsList(Xobject var, PragmaBlock pb) throws XMPexception {
    XobjList argsList = null;

    if (var.isVariable()){
      // The variable indicated in bcast directive is variable name.
      // For example,
      // ---
      // int a[10], b[10][10];
      // #pragma xmp bcast (a,b)
      // ---
      String varName = var.getString();
      XMPpair<Ident, Xtype> typedSpec = XMPutil.findTypedVar(varName, pb);
      Ident varId = typedSpec.getFirst();
      Xtype varType = typedSpec.getSecond();
        
      XobjLong count = null;
      switch (varType.getKind()) {
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.UNION:
	{
	  count = Xcons.LongLongConstant(0, 1);
	  argsList = Xcons.List(varId.getAddr(), count, Xcons.SizeOf(varType));
	} break;
      case Xtype.POINTER:
	{
	  count = Xcons.LongLongConstant(0, 1);
	  argsList = Xcons.List(varId.Ref(), count, Xcons.SizeOf(varType));
	} break;
      case Xtype.ARRAY:
	{
	  ArrayType arrayVarType = (ArrayType)varType;
	  switch (arrayVarType.getArrayElementType().getKind()) {
	  case Xtype.BASIC:
	  case Xtype.STRUCT:
	  case Xtype.UNION:
	    break;
	  default:
	    throw new XMPexception("array '" + varName + "' has has a wrong data type for broadcast");
	}
            
	if (arrayVarType.getArraySizeExpr() == null){
	  // Number of elements is static
	  // int a[10];
	  // #pragma xmp bcast (a)
	  count = Xcons.LongLongConstant(0, XMPutil.getArrayElmtCount(arrayVarType));
	  argsList = Xcons.List(varId.Ref(), count, Xcons.SizeOf(arrayVarType.getArrayElementType()));
	}
	else {
	  // Number of elements is defined in arguments
	  // void hoge(int n, int a[n]){
	  // #pragma xmp bcast (a)

	  Xobject size = Xcons.SizeOf(arrayVarType.getArrayElementType());
	  Xobject len  = arrayVarType.getArraySizeExpr();
	  ArrayType tmpType = arrayVarType;
	  for (int j = 1; j < arrayVarType.getNumDimensions(); j++){
	    tmpType = (ArrayType)arrayVarType.getRef();
	    len = Xcons.binaryOp(Xcode.MUL_EXPR, len, tmpType.getArraySizeExpr());
	  }
              
	  argsList = Xcons.List(varId.Ref(), len, size);
	    
	  // argsList = Xcons.List(varId.Ref(), arrayVarType.getArraySizeExpr(),
	  // 		       Xcons.SizeOf(arrayVarType.getArrayElementType()));
	}
	} break;
      default:
	throw new XMPexception("'" + varName + "' has a wrong data type for broadcast");
      }
    }
    else{
      // The variable indicated in bcast directive is array with array section.
      // For example,
      // ---
      // int a[10], b[10][10];
      // #pragma xmp bcast (a[:],b[2:5][:])
      // ---
      String varName = var.getArg(0).getString();
      XMPpair<Ident, Xtype> typedSpec = XMPutil.findTypedVar(varName, pb);
      Ident varId = typedSpec.getFirst();
      Xtype varType = typedSpec.getSecond();

      int dims = varType.getNumDimensions();
      if (dims != var.getArg(1).Nargs())
	throw new XMPexception(varName + " has a wrong dimension");
        
      Xobject[] start  = new Xobject[dims];
      Xobject[] length = new Xobject[dims];
      Xobject[] size   = new Xobject[dims];
      Xobject total_length = setStartLengthSize(var, varType, dims, start, length, size);

      // Check the array is contiguous or not.
      // Note that when XMP runtime supports stride bcast communication,
      // the following if-statment will be removed.
      if (!check_contiguous_of_array(dims, length, size))
	throw new XMPexception("Stride bcast operation is not supported");

      Xobject[] acc_size = new Xobject[dims];
      setAccSize(dims, size, acc_size);
      Xobject offset = calcOffsetSize(dims, start, acc_size, varType);

      argsList = Xcons.List(Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.Cast(Xtype.Pointer(BasicType.charType), varId.Ref()), offset),
			   total_length, Xcons.SizeOf(varType.getArrayElementType()));
    }

    return argsList;
  }

  // private Vector<XobjList> createBcastArgsList(XobjList varList, PragmaBlock pb) throws XMPexception {
  //   Vector<XobjList> returnVector = new Vector<XobjList>();

  //   for (XobjArgs i = varList.getArgs(); i != null; i = i.nextArgs()) {
  //     if(i.getArg().isVariable()){
  //       // The variable indicated in bcast directive is variable name.
  //       // For example,
  //       // ---
  //       // int a[10], b[10][10];
  //       // #pragma xmp bcast (a,b)
  //       // ---
  //       String varName = i.getArg().getString();
  //       XMPpair<Ident, Xtype> typedSpec = XMPutil.findTypedVar(varName, pb);
  //       Ident varId = typedSpec.getFirst();
  //       Xtype varType = typedSpec.getSecond();
        
  //       XobjLong count = null;
  //       switch (varType.getKind()) {
  //       case Xtype.BASIC:
  //       case Xtype.STRUCT:
  //       case Xtype.UNION:
  //         {
  //           count = Xcons.LongLongConstant(0, 1);
  //           returnVector.add(Xcons.List(varId.getAddr(), count, Xcons.SizeOf(varType)));
  //         } break;
  //       case Xtype.POINTER:
  // 	  {
  // 	    count = Xcons.LongLongConstant(0, 1);
  // 	    returnVector.add(Xcons.List(varId.Ref(), count, Xcons.SizeOf(varType)));
  // 	  } break;
  //       case Xtype.ARRAY:
  //         {
  //           ArrayType arrayVarType = (ArrayType)varType;
  //           switch (arrayVarType.getArrayElementType().getKind()) {
  //             case Xtype.BASIC:
  //             case Xtype.STRUCT:
  //             case Xtype.UNION:
  //               break;
  //             default:
  //               throw new XMPexception("array '" + varName + "' has has a wrong data type for broadcast");
  //           }
            
  //           if(arrayVarType.getArraySizeExpr() == null){
  //             // Number of elements is static
  //             // int a[10];
  //             // #pragma xmp bcast (a)
  //             count = Xcons.LongLongConstant(0, XMPutil.getArrayElmtCount(arrayVarType));
  //             returnVector.add(Xcons.List(varId.Ref(), count, Xcons.SizeOf(arrayVarType.getArrayElementType())));
  //           }
  //           else{
  //             // Number of elements is defined in arguments
  //             // void hoge(int n, int a[n]){
  //             // #pragma xmp bcast (a)
  //             returnVector.add(Xcons.List(varId.Ref(), arrayVarType.getArraySizeExpr(),
  //                                         Xcons.SizeOf(arrayVarType.getArrayElementType())));
  //           }
  //         } break;
  //       default:
  //         throw new XMPexception("'" + varName + "' has a wrong data type for broadcast");
  //       }
  //     }
  //     else{
  //       // The variable indicated in bcast directive is array with array section.
  //       // For example,
  //       // ---
  //       // int a[10], b[10][10];
  //       // #pragma xmp bcast (a[:],b[2:5][:])
  //       // ---
  //       String varName = i.getArg().getArg(0).getString();
  //       XMPpair<Ident, Xtype> typedSpec = XMPutil.findTypedVar(varName, pb);
  //       Ident varId = typedSpec.getFirst();
  //       Xtype varType = typedSpec.getSecond();

  //       int dims = varType.getNumDimensions();
  //       if(dims != i.getArg().getArg(1).Nargs())
  //         throw new XMPexception(varName + " has a wrong dimension");
        
  //       Xobject[] start  = new Xobject[dims];
  //       Xobject[] length = new Xobject[dims];
  //       Xobject[] size   = new Xobject[dims];
  //       Xobject total_length = setStartLengthSize(i.getArg(), varType, dims, start, length, size);

  //       // Check the array is contiguous or not.
  //       // Note that when XMP runtime supports stride bcast communication,
  //       // the following if-statment will be removed.
  //       if(! check_contiguous_of_array(dims, length, size))
  //         throw new XMPexception("Stride bcast operation is not supported");

  //       Xobject[] acc_size = new Xobject[dims];
  //       setAccSize(dims, size, acc_size);
  //       Xobject offset = calcOffsetSize(dims, start, acc_size, varType);

  //       returnVector.add(Xcons.List(Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.Cast(Xtype.Pointer(BasicType.charType), varId.Ref()), offset),
  //                                   total_length, Xcons.SizeOf(varType.getArrayElementType())));
  //     }
  //   }

  //   return returnVector;
  // }

  // private Block createBcastFuncCallBlock(boolean isMacro, String funcType, Xobject execDesc, Vector<XobjList> funcArgsList,
  //                                        XMPpair<String, XobjList> execFromRefArgs, boolean isACC) throws XMPexception {
  //   String funcSuffix = null;
  //   XobjList fromRef = null;
  //   if (execFromRefArgs == null) funcSuffix = new String(funcType + "_OMITTED");
  //   else {
  //     funcSuffix = new String(funcType + "_" + execFromRefArgs.getFirst());
  //     fromRef = execFromRefArgs.getSecond();
  //   }

  //   String accSuffix = isACC? "acc_" : "";
  //   Ident funcId = null;
  //   if (isMacro) funcId = XMP.getMacroId("_XMP_M_BCAST_" + accSuffix.toUpperCase() + funcSuffix);
  //   else         funcId = _globalDecl.declExternFunc("_XMP_bcast_" + accSuffix + funcSuffix);

  //   BlockList funcCallList = Bcons.emptyBody();
  //   Iterator<XobjList> it = funcArgsList.iterator();
  //   while (it.hasNext()) {
  //     XobjList funcArgs = it.next();
  //     if (execDesc != null) funcArgs.cons(execDesc);
  //     if (execFromRefArgs != null) funcArgs.mergeList(fromRef);

  //     funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
  //   }
  //   return Bcons.COMPOUND(funcCallList);
  // }

  private XMPpair<String, XobjList> createExecFromRefArgs(XobjList fromRef, Block block) throws XMPexception {
    if (fromRef.getArg(0) == null) {
      // execute on global communicator
      XobjList globalRef = (XobjList)fromRef.getArg(1);

      XobjList execFuncArgs = Xcons.List();
      // lower
      if (globalRef.getArg(0) == null)
        throw new XMPexception("lower bound cannot be omitted in <from-ref>");
      else execFuncArgs.add(globalRef.getArg(0));

      // upper
      if (globalRef.getArg(1) == null)
        throw new XMPexception("upper bound cannot be omitted in <from-ref>");
      else execFuncArgs.add(globalRef.getArg(1));

      // stride
      if (globalRef.getArg(2) == null) execFuncArgs.add(Xcons.IntConstant(1));
      else execFuncArgs.add(globalRef.getArg(2));

      return new XMPpair<String, XobjList>(new String("GLOBAL"), execFuncArgs);
    }
    else {
      // execute on <object-ref>

      // check object name collision
      String objectName = fromRef.getArg(0).getString();
      XMPobject fromRefObject = _globalDecl.getXMPobject(objectName, block);
      if (fromRefObject == null) {
        throw new XMPexception("cannot find '" + objectName + "' nodes/template");
      }

      if (fromRefObject.getKind() == XMPobject.TEMPLATE)
        throw new XMPexception("template cannot be used in <from-ref>");

      String kind_bracket = fromRef.getArg(1).getTail().getString();
      boolean isSquare    = kind_bracket.equals("SQUARE");
      fromRef.getArg(1).removeLastArgs();  // remove ROUND or SQUARE information
      if(isSquare) ((XobjList)fromRef.getArg(1)).reverse();

      // create arguments
      if (fromRef.getArg(1) == null)
        throw new XMPexception("multiple source nodes indicated in bcast directive");
      else {
        XobjList execFuncArgs = Xcons.List(fromRefObject.getDescId().Ref());

        int refIndex = 0;
        int refDim = fromRefObject.getDim();
        
        for (XobjArgs i = fromRef.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
          if (refIndex == refDim)
            throw new XMPexception("wrong nodes dimension indicated, too many");

          XobjList t = (XobjList)i.getArg();
          if (t == null || t.isEmptyList()){
            execFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
          }
          else {
            Xobject lower = null;
            Xobject upper = null;
            Xobject stride = null;

            execFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(0)));
            
            // lower
            if (t.getArg(0) == null)
              throw new XMPexception("lower bound cannot be omitted in <from-ref>");
            else{
              lower = t.getArg(0);
              if(fromRefObject.getKind() == XMPobject.NODES && isSquare == true)
                lower = Xcons.binaryOp(Xcode.PLUS_EXPR, lower, Xcons.IntConstant(1));
            }

            // upper
            if (t.getArg(1) == null)
              throw new XMPexception("upper bound cannot be omitted in <from-ref>");
            else{
               upper =  t.getArg(1);
               if(fromRefObject.getKind() == XMPobject.NODES && isSquare == true){
                 upper = Xcons.binaryOp(Xcode.PLUS_EXPR, t.getArg(0), t.getArg(1));
              }
            }

            // stride
            stride = (t.getArg(2) == null)? Xcons.IntConstant(1) : t.getArg(2);

            // Output
            execFuncArgs.add(Xcons.Cast(Xtype.intType, lower));
            execFuncArgs.add(Xcons.Cast(Xtype.intType, upper));
            execFuncArgs.add(Xcons.Cast(Xtype.intType, stride));
          }

          refIndex++;
        }

        if (refIndex != refDim)
          throw new XMPexception("the number of <nodes/template-subscript> should be the same with the dimension");

        return new XMPpair<String, XobjList>(new String("NODES"), execFuncArgs);
      }
    }
  }

  private final static int GMOVE_ALL   = 0;
  private final static int GMOVE_INDEX = 1;
  private final static int GMOVE_RANGE = 2;
  private final static int GMOVE_COLL  = 400;
  private final static int GMOVE_IN    = 401;
  private final static int GMOVE_OUT   = 402;

  private boolean istheSameShape(Xobject leftExpr, Xobject rightExpr)
  {
    if(rightExpr.Opcode() == Xcode.ARRAY_REF || rightExpr.Opcode() == Xcode.VAR)
      return true;

    if(rightExpr.Opcode() != leftExpr.Opcode())
      return false;

    int leftDims = leftExpr.getArg(1).Nargs();
    int leftNumTriplets = 0;
    for(int i=0;i<leftDims;i++)
      if(leftExpr.getArg(1).getArg(i).Opcode() == Xcode.INDEX_RANGE)
        leftNumTriplets++;

    int rightDims = rightExpr.getArg(1).Nargs();
    int rightNumTriplets = 0;
    for(int i=0;i<rightDims;i++)
      if(rightExpr.getArg(1).getArg(i).Opcode() == Xcode.INDEX_RANGE)
        rightNumTriplets++;

    if(leftNumTriplets == rightNumTriplets)
      return true;
    else
      return false;
  }
  
  private void translateGmove(PragmaBlock pb) throws XMPexception {

    // start translation
    XobjList gmoveDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    BlockList gmoveBody = pb.getBody();
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    // GMOVE_NORMAL | GMOVE_IN | GMOVE_OUT
    Xobject gmoveClause = gmoveDecl.getArg(0);

    // async id
    Xobject async = gmoveDecl.getArg(1);

    // acc or host
    XobjList accOrHost = (XobjList)gmoveDecl.getArg(2);
    boolean isACC = accOrHost.hasIdent("acc");
    boolean isHost = accOrHost.hasIdent("host");
    if(!isACC && !isHost){
      isHost = true;
    }else if(isHost && isACC){
      throw new XMPexception(pb.getLineNo(), "gmove for both acc and host is unimplemented");
    }
    String funcPrefix = "_XMP_gmove_";
    if(isACC){
      funcPrefix += "acc_";
    }
    
    // check body
    Xobject assignStmt = null;
    String checkBodyErrMsg = new String("gmove directive should be written before one assign statement");
    Block gmoveBodyHead = gmoveBody.getHead();
    if(gmoveBodyHead instanceof SimpleBlock) {
      if (gmoveBodyHead.getNext() != null) {
        throw new XMPexception(checkBodyErrMsg);
      }

      Statement gmoveStmt = gmoveBodyHead.getBasicBlock().getHead();
      if (gmoveStmt.getNext() != null) {
        throw new XMPexception(checkBodyErrMsg);
      }

      if(gmoveStmt.getExpr().Opcode() == Xcode.ASSIGN_EXPR) {
        assignStmt = gmoveStmt.getExpr();
      } else {
        throw new XMPexception(checkBodyErrMsg);
      }
    } else {
      throw new XMPexception(checkBodyErrMsg);
    }

    Xobject leftExpr = assignStmt.left();
    XMPpair<XMPalignedArray, XobjList> leftExprInfo = getXMPalignedArrayExpr(pb, leftExpr);
    XMPalignedArray leftAlignedArray = null;
    if (leftExprInfo != null) leftAlignedArray = leftExprInfo.getFirst();

    Xobject rightExpr = assignStmt.right();
    XMPpair<XMPalignedArray, XobjList> rightExprInfo = getXMPalignedArrayExpr(pb, assignStmt.right());
    XMPalignedArray rightAlignedArray = null;
    if (rightExprInfo != null) rightAlignedArray = rightExprInfo.getFirst();

    // check LHS and RHS

    if (rightAlignedArray == null && leftAlignedArray == null){
      pb.replace(pb.getBody().getHead()); // ignore the gmove directive
      return;
    }
    
    if (rightAlignedArray != null){
      StorageClass sclass = rightAlignedArray.getArrayId().getStorageClass();
      if (gmoveClause.getInt() == XMPcollective.GMOVE_IN &&
  	  sclass != StorageClass.EXTDEF && sclass != StorageClass.EXTERN)
      XMP.fatal("Current limitation: Only a SAVE or MODULE variable can be the target of gmove in/out.");
    }

    if (leftAlignedArray != null){
      StorageClass sclass = leftAlignedArray.getArrayId().getStorageClass();
      if (gmoveClause.getInt() == XMPcollective.GMOVE_OUT &&
  	  sclass != StorageClass.EXTDEF && sclass != StorageClass.EXTERN)
  	XMP.fatal("Current limitation: Only a SAVE or MODULE variable can be the target of gmove in/out.");
    }

    if(! istheSameShape(leftExpr, assignStmt.right())){
      throw new XMPexception("The shape of the right side and the shape of the left side are different.");
    }
    
    if (rightAlignedArray == null && gmoveClause.getInt() == XMPcollective.GMOVE_IN){
      XMP.fatal("gmove in cannot be applied to a local rhs.");
    }

    if (leftAlignedArray == null && gmoveClause.getInt() == XMPcollective.GMOVE_OUT){
      XMP.fatal("gmove out cannot be applied to a local lhs.");
    }

    // convert gmove to array

    if (leftExpr.Opcode() == Xcode.SUB_ARRAY_REF &&
	leftAlignedArray != null && rightAlignedArray == null &&
	gmoveClause.getInt() == XMPcollective.GMOVE_NORMAL){
      Block arrayBlock = convertGmoveToArray(pb, leftAlignedArray, leftExpr, rightExpr);
      pb.replace(arrayBlock);
      translateArray((PragmaBlock)arrayBlock);
      return;
    }

    // generate gmove funcs.
    
    Ident left_desc = buildGmoveDesc(leftExpr, bb, pb, isACC);
    Ident right_desc = buildGmoveDesc(rightExpr, bb, pb, isACC);
    
    Ident f = _globalDecl.declExternFunc(isACC? "xmpc_gmv_do_acc" : "xmpc_gmv_do", Xtype.FsubroutineType);
    Xobject args = Xcons.List(left_desc.Ref(), right_desc.Ref(), gmoveClause);
    bb.add(f.callSubroutine(args));

    Ident d = _globalDecl.declExternFunc("xmpc_gmv_dealloc", Xtype.FsubroutineType);
    Xobject args_l = Xcons.List(left_desc.Ref());
    bb.add(d.callSubroutine(args_l));
    Xobject args_r = Xcons.List(right_desc.Ref());
    bb.add(d.callSubroutine(args_r));
	    
    if (async.Opcode() != Xcode.LIST){
      Ident g;
      Xobject arg = Xcons.List(async);
      g = _globalDecl.declExternFunc("xmpc_init_async", Xtype.FsubroutineType);
      bb.insert(g.callSubroutine(arg));
      g = _globalDecl.declExternFunc("xmpc_start_async", Xtype.FsubroutineType);
      bb.add(g.callSubroutine(arg));
    }
    
    if(isACC){
      XobjList vars = Xcons.List();
      vars.add(Xcons.Symbol(Xcode.VAR, leftAlignedArray.getName()));
      vars.add(Xcons.Symbol(Xcode.VAR, rightAlignedArray.getName()));
      
      b = 
      Bcons.PRAGMA(Xcode.ACC_PRAGMA, "HOST_DATA",
            Xcons.List(Xcons.List(Xcons.String("USE_DEVICE"), vars)), Bcons.blockList(b));
    }
    
    Block gmoveBlock = Bcons.COMPOUND(Bcons.blockList(b));
    pb.replace(gmoveBlock);

    Xobject profileClause = gmoveDecl.getArg(3);
    addProfileFunctions(profileClause, b, "gmove", pb);
  }

  
  private Ident buildGmoveDesc(Xobject x, BasicBlock bb, PragmaBlock pb, boolean isACC) throws XMPexception {

    Ident descId = pb.getParentBlock().getBody().declLocalIdent(tmpSym.getStr("gmv"), Xtype.voidPtrType);

    XMPalignedArray array = null;
    String arrayName;
    Xtype type;
    Xobject a;

    Ident f;
    Xobject args;

    switch (x.Opcode()){

    case ARRAY_REF:
    case SUB_ARRAY_REF:
      
      a = x.getArg(0);
      arrayName = getArrayName(x);
      array = _globalDecl.getXMPalignedArray(arrayName, pb);
      type = (array != null) ? array.getArrayType() : a.Type();
      int n = type.getNumDimensions();
      
      if (array != null){ // global
	
	f = _globalDecl.declExternFunc(isACC? "xmpc_gmv_g_alloc_acc" : "xmpc_gmv_g_alloc", Xtype.FsubroutineType);
	args = Xcons.List(descId.getAddr(), array.getDescId().Ref());
	if(isACC) args.add(array.getAddrId().Ref());
	bb.add(f.callSubroutine(args));

	XobjList subscripts = (XobjList)x.getArg(1);
	
	f = _globalDecl.declExternFunc("xmpc_gmv_g_dim_info", Xtype.FsubroutineType);

	for (int i = 0; i < n; i++, type = type.getRef()){

	  Xobject sub = subscripts.getArg(i);

	  if (! sub.isIndexRange()){ // index
	    args = Xcons.List(descId.Ref(), Xcons.IntConstant(i),
			      Xcons.IntConstant(GMOVE_INDEX),
			      sub,
			      Xcons.IntConstant(0), Xcons.IntConstant(0));
	  }
	  else { // triplet
	  
	    Xobject lb = ((XobjList)sub).getArg(0);
	    if (lb == null) lb = Xcons.IntConstant(0);
	    Xobject len = ((XobjList)sub).getArg(1);
	    Xobject st = ((XobjList)sub).getArg(2);
	    if (st == null) st = Xcons.IntConstant(1);

	    args = Xcons.List(descId.Ref(), Xcons.IntConstant(i),
			      Xcons.IntConstant(GMOVE_RANGE),
			      lb, len, st);
	  }
	  
	  bb.add(f.callSubroutine(args));
	}

      }
      else { // local
	
	f = _globalDecl.declExternFunc("xmpc_gmv_l_alloc", Xtype.FsubroutineType);
	type = a.Type();

	// if (!type.isArray())
	//   XMP.fatal("buildGmoveDesc: SUB_ARRAY_REF for non-array");
	// args = Xcons.List(descId.getAddr(), a,
	// 		  Xcons.IntConstant(type.getNumDimensions()));

	if (type.isArray()){
	  ;
	}
	else if (type.isPointer()){
	  // current limitation: local pointers are always considered to be one-dimensional.
	  n = 1;
	}
	else {
	  XMP.fatal("buildGmoveDesc: SUB_ARRAY_REF for non-array");
	}
	    
	args = Xcons.List(descId.getAddr(), a, Xcons.IntConstant(n));
	bb.add(f.callSubroutine(args));

	Xobject a_lb = Xcons.IntConstant(0);
	Xobject a_len;

	XobjList subscripts = (XobjList)x.getArg(1);
	
	f = _globalDecl.declExternFunc("xmpc_gmv_l_dim_info", Xtype.FsubroutineType);

	for (int i = 0; i < n; i++, type = type.getRef()){

	  if (type.getKind() == Xtype.POINTER){
	    a_len = Xcons.LongLongConstant(0, 1); // dummy
	  }
	  else {
	    long dimSize = type.getArraySize();
	    if (dimSize == 0){
	      Ident ret = declIdentWithBlock(pb, "XMP_" + arrayName + "_ret" + Integer.toString(i),
					     Xtype.intType);
	      Ident sz = declIdentWithBlock(pb, "XMP_" + arrayName + "_len" + Integer.toString(i),
					    Xtype.intType);

	      f = _globalDecl.declExternFunc("xmp_array_ubound", Xtype.intType);
	      args = Xcons.List(array.getDescId().Ref(), Xcons.IntConstant(i+1), sz.getAddr());

	      pb.insert(Xcons.Set(ret.Ref(), Xcons.binaryOp(Xcode.MINUS_EXPR, f.Call(args), Xcons.IntConstant(1))));
	      a_len = sz.Ref();
	    }
	    else if (dimSize == -1){
	      a_len = type.getArraySizeExpr();
	    }
	    else {
	      a_len = Xcons.LongLongConstant(0, dimSize);
	    }
	  }

	  Xobject sub = subscripts.getArg(i);

	  if (! sub.isIndexRange()){ // index
	    args = Xcons.List(descId.Ref(), Xcons.IntConstant(i),
			      a_lb, a_len,
			      Xcons.IntConstant(GMOVE_INDEX),
			      sub, Xcons.IntConstant(0), Xcons.IntConstant(0));
	  }
	  else { // triplet
	    
	    Xobject lb = ((XobjList)sub).getArg(0);
	    if (lb == null) lb = Xcons.IntConstant(0);
	    Xobject len = ((XobjList)sub).getArg(1);
	    Xobject st = ((XobjList)sub).getArg(2);
	    if (st == null) st = Xcons.IntConstant(1);

	    args = Xcons.List(descId.Ref(), Xcons.IntConstant(i),
			      a_lb, a_len,
			      Xcons.IntConstant(GMOVE_INDEX),
			      lb, len, st);
	  }

	  bb.add(f.callSubroutine(args));
	}
      }

      break;

    case VAR:

      arrayName = x.getName();
      array = _globalDecl.getXMPalignedArray(arrayName, pb);

      if (array != null){ // name of a global array
    	f = _globalDecl.declExternFunc("xmpc_gmv_g_alloc", Xtype.FsubroutineType);
    	args = Xcons.List(descId.getAddr(), array.getDescId().Ref());
    	bb.add(f.callSubroutine(args));
	
    	f = _globalDecl.declExternFunc("xmpc_gmv_g_dim_info", Xtype.FsubroutineType);
    	for (int i = 0; i < array.getDim(); i++){
    	  args = Xcons.List(descId.Ref(), Xcons.IntConstant(i),
    			    Xcons.IntConstant(GMOVE_ALL),
    			    Xcons.IntConstant(0),
    			    Xcons.IntConstant(0), Xcons.IntConstant(0));
    	  bb.add(f.callSubroutine(args));
    	}
      }
      else { // scalar or name of a local array
	f = _globalDecl.declExternFunc(isACC? "xmpc_gmv_l_alloc_acc" : "xmpc_gmv_l_alloc", Xtype.FsubroutineType);
    	args = Xcons.List(descId.Ref(), Xcons.AddrOf(x), Xcons.IntConstant(0));
    	bb.add(f.callSubroutine(args));
      }

      break;

    default:
      XMP.fatal("gmove must be followed by a simple assignment");
    }
    
    return descId;
  }

  
  // private void translateGmove(PragmaBlock pb) throws XMPexception {
  //   // start translation
  //   XobjList gmoveDecl = (XobjList)pb.getClauses();
  //   XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
  //   BlockList gmoveBody = pb.getBody();
  //   Block gmoveFuncCallBlock = null;

  //   // GMOVE_NORMAL | GMOVE_IN | GMOVE_OUT
  //   Xobject gmoveClause = gmoveDecl.getArg(0);

  //   // acc or host
  //   XobjList accOrHost = (XobjList)gmoveDecl.getArg(2);
  //   boolean isACC = accOrHost.hasIdent("acc");
  //   boolean isHost = accOrHost.hasIdent("host");
  //   if(!isACC && !isHost){
  //     isHost = true;
  //   }else if(isHost && isACC){
  //     throw new XMPexception(pb.getLineNo(), "gmove for both acc and host is unimplemented");
  //   }
  //   String funcPrefix = "_XMP_gmove_";
  //   if(isACC){
  //     funcPrefix += "acc_";
  //   }
    
  //   // check body
  //   Xobject assignStmt = null;
  //   String checkBodyErrMsg = new String("gmove directive should be written before one assign statement");
  //   Block gmoveBodyHead = gmoveBody.getHead();
  //   if(gmoveBodyHead instanceof SimpleBlock) {
  //     if (gmoveBodyHead.getNext() != null) {
  //       throw new XMPexception(checkBodyErrMsg);
  //     }

  //     Statement gmoveStmt = gmoveBodyHead.getBasicBlock().getHead();
  //     if (gmoveStmt.getNext() != null) {
  //       throw new XMPexception(checkBodyErrMsg);
  //     }

  //     if(gmoveStmt.getExpr().Opcode() == Xcode.ASSIGN_EXPR) {
  //       assignStmt = gmoveStmt.getExpr();
  //     } else {
  //       throw new XMPexception(checkBodyErrMsg);
  //     }
  //   } else {
  //     throw new XMPexception(checkBodyErrMsg);
  //   }

  //   Xobject leftExpr = assignStmt.left();
  //   XMPpair<XMPalignedArray, XobjList> leftExprInfo = getXMPalignedArrayExpr(pb, leftExpr);
  //   XMPalignedArray leftAlignedArray = null;
  //   if (leftExprInfo != null) leftAlignedArray = leftExprInfo.getFirst();

  //   Xobject rightExpr = assignStmt.right();
  //   XMPpair<XMPalignedArray, XobjList> rightExprInfo = getXMPalignedArrayExpr(pb, assignStmt.right());
  //   XMPalignedArray rightAlignedArray = null;
  //   if (rightExprInfo != null) rightAlignedArray = rightExprInfo.getFirst();

  //   if (rightAlignedArray != null){
  //     StorageClass sclass = rightAlignedArray.getArrayId().getStorageClass();
  //     if (gmoveClause.getInt() == XMPcollective.GMOVE_IN &&
  // 	  sclass != StorageClass.EXTDEF && sclass != StorageClass.EXTERN)
  //     XMP.fatal("Current limitation: Only a SAVE or MODULE variable can be the target of gmove in/out.");
  //   }

  //   if (leftAlignedArray != null){
  //     StorageClass sclass = leftAlignedArray.getArrayId().getStorageClass();
  //     if (gmoveClause.getInt() == XMPcollective.GMOVE_OUT &&
  // 	  sclass != StorageClass.EXTDEF && sclass != StorageClass.EXTERN)
  // 	XMP.fatal("Current limitation: Only a SAVE or MODULE variable can be the target of gmove in/out.");
  //   }

  //   if (rightAlignedArray == null && gmoveClause.getInt() == XMPcollective.GMOVE_IN){
  //     XMP.fatal("gmove in cannot be applied to a local rhs.");
  //   }

  //   if (leftAlignedArray == null && gmoveClause.getInt() == XMPcollective.GMOVE_OUT){
  //     XMP.fatal("gmove out cannot be applied to a local lhs.");
  //   }

  //   boolean leftHasSubArrayRef = (leftExpr.Opcode() == Xcode.SUB_ARRAY_REF);
  //   boolean rightHasSubArrayRef = (rightExpr.Opcode() == Xcode.SUB_ARRAY_REF);
  //   if (leftHasSubArrayRef) {
  //     if (rightHasSubArrayRef) {
  //       if (leftAlignedArray == null) {
  //         if (rightAlignedArray == null) {	// !leftIsAlignedArray && !rightIsAlignedArray  |-> local assignment (every node)
  // 	    gmoveFuncCallBlock = Bcons.COMPOUND(gmoveBody);
  //           // String arrayName = getArrayName(leftExpr);
  //           // Ident arrayId = pb.findVarIdent(arrayName);
  //           // Xtype arrayElmtType = arrayId.Type().getArrayElementType();

  //           // XobjList gmoveFuncArgs = null;
  //           // if (arrayElmtType.getKind() == Xtype.BASIC) {
  //           //   gmoveFuncArgs = Xcons.List(XMP.createBasicTypeConstantObj(arrayElmtType));
  //           // } else {
  //           //   gmoveFuncArgs = Xcons.List(Xcons.IntConstant(XMP.NONBASIC_TYPE));
  //           // }
  //           // gmoveFuncArgs.add(Xcons.SizeOf(arrayElmtType));

  //           // gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  //           // gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	    // gmoveFuncArgs.add(gmoveClause);
  // 	    // gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "LOCALCOPY_ARRAY", gmoveFuncArgs);
  //         } else {				// !leftIsAlignedArray &&  rightIsAlignedArray  |-> broadcast
  //           Xtype arrayElmtType = rightAlignedArray.getType();

  //           XobjList gmoveFuncArgs = Xcons.List(rightAlignedArray.getDescId().Ref());
  //           if (arrayElmtType.getKind() == Xtype.BASIC) {
  //             gmoveFuncArgs.add(XMP.createBasicTypeConstantObj(arrayElmtType));
  //           } else {
  //             gmoveFuncArgs.add(Xcons.IntConstant(XMP.NONBASIC_TYPE));
  //           }
  //           gmoveFuncArgs.add(Xcons.SizeOf(arrayElmtType));

  //           gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  //           gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	    gmoveFuncArgs.add(gmoveClause);
  // 	    gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "BCAST_ARRAY", gmoveFuncArgs);
  //         }
  //       }
  //       else {
  //         if (rightAlignedArray == null) {	//  leftIsAlignedArray && !rightIsAlignedArray  |-> local assignment (home node)
  //           Xtype arrayElmtType = leftAlignedArray.getType();

  //           XobjList gmoveFuncArgs = Xcons.List(leftAlignedArray.getDescId().Ref());
  //           if (arrayElmtType.getKind() == Xtype.BASIC) {
  //             gmoveFuncArgs.add(XMP.createBasicTypeConstantObj(arrayElmtType));
  //           } else {
  //             gmoveFuncArgs.add(Xcons.IntConstant(XMP.NONBASIC_TYPE));
  //           }
  //           gmoveFuncArgs.add(Xcons.SizeOf(arrayElmtType));

  //           gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  //           gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	    gmoveFuncArgs.add(gmoveClause);
  // 	    gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "HOMECOPY_ARRAY", gmoveFuncArgs);
  //         } else {				//  leftIsAlignedArray &&  rightIsAlignedArray  |-> send/recv
  //           Xtype arrayElmtType = leftAlignedArray.getType();

  //           XobjList gmoveFuncArgs = Xcons.List(leftAlignedArray.getDescId().Ref(),
  //                                               rightAlignedArray.getDescId().Ref());
  //           if(isACC){
  //             gmoveFuncArgs.add(leftAlignedArray.getAddrIdVoidRef());
  //             gmoveFuncArgs.add(rightAlignedArray.getAddrIdVoidRef());
  //           }
  //           if (arrayElmtType.getKind() == Xtype.BASIC) {
  //             gmoveFuncArgs.add(XMP.createBasicTypeConstantObj(arrayElmtType));
  //           } else {
  //             gmoveFuncArgs.add(Xcons.IntConstant(XMP.NONBASIC_TYPE));
  //           }
  //           gmoveFuncArgs.add(Xcons.SizeOf(arrayElmtType));

  //           gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  //           gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	    gmoveFuncArgs.add(gmoveClause);

  // 	    gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "SENDRECV_ARRAY", gmoveFuncArgs);
  //         }
  //       }
  //     }
  //     else { // leftHasSubArrayRef && !rightHasSubArrayRef
  //       if (leftAlignedArray == null){
  //         if (rightAlignedArray == null){
  // 	    gmoveFuncCallBlock = Bcons.COMPOUND(gmoveBody);
  // 	  }
  // 	  else {
  //           Xtype arrayElmtType = rightAlignedArray.getType();

  //           XobjList gmoveFuncArgs = Xcons.List(rightAlignedArray.getDescId().Ref());

  //           if (arrayElmtType.getKind() == Xtype.BASIC) {
  //             gmoveFuncArgs.add(XMP.createBasicTypeConstantObj(arrayElmtType));
  //           }
  // 	    else {
  //             gmoveFuncArgs.add(Xcons.IntConstant(XMP.NONBASIC_TYPE));
  //           }
  //           gmoveFuncArgs.add(Xcons.SizeOf(arrayElmtType));

  //           gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  //           gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	    gmoveFuncArgs.add(gmoveClause);

  // 	    gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "LSECTION_GSCALAR", gmoveFuncArgs);
  // 	  }
  // 	}
  // 	else {
  //         if (rightAlignedArray == null){
  // 	    if (gmoveClause.getInt() == XMPcollective.GMOVE_NORMAL){
  // 	      Block arrayBlock = convertGmoveToArray(pb, leftAlignedArray, leftExpr, rightExpr);
  // 	      pb.replace(arrayBlock);
  // 	      translateArray((PragmaBlock)arrayBlock);
  // 	      return;
  // 	    }
  // 	    else if (gmoveClause.getInt() == XMPcollective.GMOVE_OUT){
  // 	      XobjList gmoveFuncArgs = Xcons.List(leftAlignedArray.getDescId().Ref(), Xcons.AddrOf(rightExpr));
  // 	      gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  // 	      gmoveFuncArgs.add(gmoveClause);
  // 	      gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "INOUT_SCALAR", gmoveFuncArgs);
  // 	    }
  // 	    else { // GMOVE_IN
  // 	      XMP.fatal("gmove in cannot be applied to a local rhs.");
  // 	    }
  // 	  }
  // 	  else { // G(:) = g(i)
  //           Xtype arrayElmtType = leftAlignedArray.getType();

  //           XobjList gmoveFuncArgs = Xcons.List(leftAlignedArray.getDescId().Ref(),
  //                                               rightAlignedArray.getDescId().Ref());

  //           if (arrayElmtType.getKind() == Xtype.BASIC) {
  //             gmoveFuncArgs.add(XMP.createBasicTypeConstantObj(arrayElmtType));
  //           }
  // 	    else {
  //             gmoveFuncArgs.add(Xcons.IntConstant(XMP.NONBASIC_TYPE));
  //           }
  //           gmoveFuncArgs.add(Xcons.SizeOf(arrayElmtType));

  //           gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  //           gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	    gmoveFuncArgs.add(gmoveClause);

  // 	    gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "GSECTION_GSCALAR", gmoveFuncArgs);
  // 	  }
  // 	}
	
  //       // FIXME implement
  //       //throw new XMPexception("not implemented yet");
  // 	//System.err.printf("not implemented yet\n");
  //     }
  //   }
  //   else {
  //     if (rightHasSubArrayRef) {
  //       throw new XMPexception("syntax error in gmove assign statement");
  //     }

  //     if (leftAlignedArray == null) {
  //       if (rightAlignedArray == null) { // !leftIsAlignedArray && !rightIsAlignedArray	|-> local assignment (every node)
  // 	  gmoveFuncCallBlock = Bcons.COMPOUND(gmoveBody);
  //       } else {				// !leftIsAlignedArray &&  rightIsAlignedArray	|-> broadcast

  // 	  Xobject rightAddr = (gmoveClause.getInt() == XMPcollective.GMOVE_IN) ?
  // 	    Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)) : Xcons.AddrOf(rightExpr);
  //         XobjList gmoveFuncArgs = Xcons.List(Xcons.AddrOf(leftExpr), rightAddr,
  //                                             rightAlignedArray.getDescId().Ref());

  //         // XobjList gmoveFuncArgs = Xcons.List(Xcons.AddrOf(leftExpr), Xcons.AddrOf(rightExpr),
  //         //                                     rightAlignedArray.getDescId().Ref());

  //         gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	  gmoveFuncArgs.add(gmoveClause);

  // 	  gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "BCAST_SCALAR", gmoveFuncArgs);
  //       }
  //     }
  //     else {
  //       if (rightAlignedArray == null) { //  leftIsAlignedArray && !rightIsAlignedArray	|-> local assignment (home node)
  // 	  if (gmoveClause.getInt() == XMPcollective.GMOVE_NORMAL){
  // 	    XobjList gmoveFuncArgs = Xcons.List(leftAlignedArray.getDescId().Ref());
  // 	    gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  // 	    gmoveFuncArgs.add(gmoveClause);

  // 	    Ident gmoveFuncId = _globalDecl.declExternFunc(funcPrefix + "HOMECOPY_SCALAR", Xtype.intType);
  // 	    gmoveFuncCallBlock = Bcons.IF(BasicBlock.Cond(gmoveFuncId.Call(gmoveFuncArgs)), gmoveBody, null);
  // 	  }
  // 	  else if (gmoveClause.getInt() == XMPcollective.GMOVE_OUT){
  // 	    XobjList gmoveFuncArgs = Xcons.List(leftAlignedArray.getDescId().Ref(), Xcons.AddrOf(rightExpr));

  // 	    //gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  // 	    XobjList accList = getArrayAccList(pb, assignStmt.left());
  // 	    int dim = 0;
  // 	    for (Xobject x : leftExprInfo.getSecond()){
  // 	      gmoveFuncArgs.add(x);
  // 	      gmoveFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
  // 	      gmoveFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(0)));
  // 	      gmoveFuncArgs.add(Xcons.Cast(Xtype.unsignedlonglongType, accList.getArg(dim)));
  // 	      dim++;
  // 	    }

  // 	    gmoveFuncArgs.add(gmoveClause);
  // 	    gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "INOUT_SCALAR", gmoveFuncArgs);
  // 	  }
  //       }
  // 	else {				//  leftIsAlignedArray &&  rightIsAlignedArray	|-> send/recv

  // 	  Xobject leftAddr = (gmoveClause.getInt() == XMPcollective.GMOVE_OUT) ?
  // 	    Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)) : Xcons.AddrOf(leftExpr);
  // 	  Xobject rightAddr = (gmoveClause.getInt() == XMPcollective.GMOVE_IN) ?
  // 	    Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)) : Xcons.AddrOf(rightExpr);
  //         XobjList gmoveFuncArgs = Xcons.List(leftAddr, rightAddr,
  //                                             leftAlignedArray.getDescId().Ref(), rightAlignedArray.getDescId().Ref());

  //         // XobjList gmoveFuncArgs = Xcons.List(Xcons.AddrOf(leftExpr), Xcons.AddrOf(rightExpr),
  //         //                                     leftAlignedArray.getDescId().Ref(), rightAlignedArray.getDescId().Ref());
  //         gmoveFuncArgs.mergeList(leftExprInfo.getSecond());
  //         gmoveFuncArgs.mergeList(rightExprInfo.getSecond());
  // 	  gmoveFuncArgs.add(gmoveClause);

  // 	  gmoveFuncCallBlock = _globalDecl.createFuncCallBlock(funcPrefix + "SENDRECV_SCALAR", gmoveFuncArgs);
  //       }
  //     }
  //   }
    
  //   if(isACC){
  //     XobjList vars = Xcons.List();
  //     vars.add(Xcons.Symbol(Xcode.VAR, leftAlignedArray.getName()));
  //     vars.add(Xcons.Symbol(Xcode.VAR, rightAlignedArray.getName()));
      
  //     gmoveFuncCallBlock = 
  //     Bcons.PRAGMA(Xcode.ACC_PRAGMA, "HOST_DATA",
  //           Xcons.List(Xcons.List(Xcons.String("USE_DEVICE"), vars)), Bcons.blockList(gmoveFuncCallBlock));
  //   }
    
  //   Xobject async = gmoveDecl.getArg(1);
  //   if(async.Opcode() != Xcode.LIST){
  //     if(!XmOption.isAsync()){
  // 	XMP.error(pb.getLineNo(), "MPI-3 is required to use the async clause on a bcast directive");
  //     }
  //   }

  //   Block gmoveBlock = Bcons.COMPOUND(Bcons.blockList(gmoveFuncCallBlock));
  //   pb.replace(gmoveBlock);

  //   Xobject profileClause = gmoveDecl.getArg(3);
  //   addProfileFunctions(profileClause, gmoveFuncCallBlock, "gmove", pb);
  // }

  private Block convertGmoveToArray(PragmaBlock pb, XMPalignedArray leftAlignedArray, Xobject leftExpr, Xobject rightExpr){

    Xobject onRef = Xcons.List();
    Xobject t = Xcons.Symbol(Xcode.VAR, leftAlignedArray.getAlignTemplate().getName());
    onRef.add(t);

    Xobject subscripts = Xcons.List();
    for (int i = 0; i < leftAlignedArray.getAlignTemplate().getDim(); i++){
      subscripts.add(null);
    }

    XobjList arrayRefs = (XobjList)leftExpr.getArg(1);

    for (int i = 0; i < leftAlignedArray.getDim(); i++){

      int alignSubscriptIndex = leftAlignedArray.getAlignSubscriptIndexAt(i);
      Xobject alignSubscriptOffset = leftAlignedArray.getAlignSubscriptExprAt(i);

      Xobject sub = arrayRefs.getArg(i);
      Xobject triplet;
      if (sub.isIndexRange()){
	Xobject lb = sub.getArg(0);
	if (alignSubscriptOffset != null) lb = Xcons.binaryOp(Xcode.PLUS_EXPR, lb, alignSubscriptOffset);
	Xobject ub = sub.getArg(1);
	Xobject st = sub.getArg(2);
	triplet = Xcons.List(lb, ub, st);
      }
      else {
	Xobject lb = sub.getArg(0);
	if (alignSubscriptOffset != null) lb = Xcons.binaryOp(Xcode.PLUS_EXPR, lb, alignSubscriptOffset);
	Xobject ub = lb;
	Xobject st = Xcons.IntConstant(1);
	triplet = Xcons.List(lb, ub, st);
      }	

      subscripts.setArg(alignSubscriptIndex, triplet);
    }
    onRef.add(subscripts);

    Xobject args = Xcons.List(onRef);
    return Bcons.PRAGMA(Xcode.XMP_PRAGMA, "LOOP", args, pb.getBody());

  }

  private XMPpair<XMPalignedArray, XobjList> getXMPalignedArrayExpr(PragmaBlock pb, Xobject expr) throws XMPexception {
    switch (expr.Opcode()) {
      case ARRAY_REF:
        return parseArrayRefExpr(pb, expr);
      case SUB_ARRAY_REF:
        return parseSubArrayRefExpr(pb, expr, getArrayAccList(pb, expr));
      case VAR:
	return null;
      default:
        throw new XMPexception("unsupported expression: gmove");
    }
  }

  private String getArrayName(Xobject expr) throws XMPexception {
    if ((expr.Opcode() == Xcode.ARRAY_REF) ||
        (expr.Opcode() == Xcode.SUB_ARRAY_REF)) {
      return expr.getArg(0).getSym();
    } else {
      throw new XMPexception("cannot find array ref");
    }
  }

  private XobjList getArrayAccList(PragmaBlock pb, Xobject expr) throws XMPexception {
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);

    XobjList accList = Xcons.List();

    String arrayName = expr.getArg(0).getSym();
    //XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, localXMPsymbolTable);
    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, pb);
    if (alignedArray == null) {
      Ident arrayId = pb.findVarIdent(arrayName);
      Xtype arrayType = arrayId.Type();

      int arrayDim = arrayType.getNumDimensions();
      if (arrayDim > XMP.MAX_DIM) {
        throw new XMPexception("array dimension should be less than " + (XMP.MAX_DIM + 1));
      }

      arrayType = arrayType.getRef();
      for (int i = 0; i < arrayDim - 1; i++, arrayType = arrayType.getRef()) {
        accList.add(XMPutil.getArrayElmtsObj(arrayType));
      }
      accList.add(Xcons.IntConstant(1));
    } else {
      int arrayDim = alignedArray.getDim();
      for (int i = 0; i < arrayDim; i++) {
        accList.add(alignedArray.getAccIdAt(i).Ref());
      }
    }

    return accList;
  }

  private XMPpair<XMPalignedArray, XobjList> parseSubArrayRefExpr(PragmaBlock pb,
                                                                  Xobject expr, XobjList accList) throws XMPexception {
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    Xobject arrayAddr = expr.getArg(0);
    String arrayName = arrayAddr.getSym();
    //XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, localXMPsymbolTable);
    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, pb);

    XobjList arrayRefs = (XobjList)expr.getArg(1);
    XobjList castedArrayRefs = Xcons.List();

    int dim = 0;
    if (arrayRefs != null) {
      for (Xobject x : arrayRefs) {
        if (x.isIndexRange()) {
          castedArrayRefs.add(Xcons.Cast(Xtype.intType, x.getArg(0)));
          castedArrayRefs.add(Xcons.Cast(Xtype.intType, x.getArg(1)));
          castedArrayRefs.add(Xcons.Cast(Xtype.intType, x.getArg(2)));
          castedArrayRefs.add(Xcons.Cast(Xtype.unsignedlonglongType, accList.getArg(dim)));
        } else {
          castedArrayRefs.add(Xcons.Cast(Xtype.intType, x));
          //castedArrayRefs.add(Xcons.Cast(Xtype.intType, x)); // for C-style triplet
          castedArrayRefs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
          //castedArrayRefs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
	  castedArrayRefs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(0)));
          castedArrayRefs.add(Xcons.Cast(Xtype.unsignedlonglongType, accList.getArg(dim)));
        }

       dim++;
      }
    }

    if (alignedArray == null) {
      castedArrayRefs.cons(Xcons.Cast(Xtype.intType, Xcons.IntConstant(dim)));
      castedArrayRefs.cons(Xcons.Cast(Xtype.voidPtrType, arrayAddr));
    }

    return new XMPpair<XMPalignedArray, XobjList>(alignedArray, castedArrayRefs);
  }

  private XMPpair<XMPalignedArray, XobjList> parseArrayRefExpr(PragmaBlock pb, Xobject expr) throws XMPexception {
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    String arrayName = expr.getArg(0).getSym();

    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, pb);
    XobjList castedArrayRefs = Xcons.List();
    XobjList arrayRefs = (XobjList)expr.getArg(1);
    if (arrayRefs != null) {
      for (Xobject x : arrayRefs) {
        castedArrayRefs.add(Xcons.Cast(Xtype.intType, x));
      }
    }

    return new XMPpair<XMPalignedArray, XobjList>(alignedArray, castedArrayRefs);
  }

  private XMPquadruplet<String, Boolean, XobjList, XMPobject> createExecOnRefArgs(XobjList onRef,
                                                                                  Block block) throws XMPexception {
    if (onRef.getArg(0) == null) {

      // Now this case never ocuur.
      return null;
      
      // // execute on global communicator
      // XobjList globalRef = (XobjList)onRef.getArg(1);

      // boolean splitComm = false;
      // XobjList tempArgs = Xcons.List();
      // // lower
      // if (globalRef.getArg(0) == null) tempArgs.add(Xcons.IntConstant(1));
      // else {
      //   splitComm = true;
      //   tempArgs.add(globalRef.getArg(0));
      // }
      // // upper
      // if (globalRef.getArg(1) == null) tempArgs.add(_globalDecl.getWorldSizeId().Ref());
      // else {
      //   splitComm = true;
      //   tempArgs.add(globalRef.getArg(1));
      // }
      // // stride
      // if (globalRef.getArg(2) == null) tempArgs.add(Xcons.IntConstant(1));
      // else {
      //   splitComm = true;
      //   tempArgs.add(globalRef.getArg(2));
      // }

      // String execFuncSuffix = null;
      // XobjList execFuncArgs = null;
      // if (splitComm) {
      //   execFuncSuffix = "GLOBAL_PART";
      //   execFuncArgs = tempArgs;
      // }
      // else {
      //   execFuncSuffix = "NODES_ENTIRE";
      //   execFuncArgs = Xcons.List(_globalDecl.getWorldDescId().Ref());
      // }

      // return new XMPquadruplet<String, Boolean, XobjList, XMPobject>(execFuncSuffix, new Boolean(splitComm), execFuncArgs, null);
    }
    else {
      // execute on <object-ref>

      // check object name collision
      String objectName = onRef.getArg(0).getString();
      XMPobject onRefObject = _globalDecl.getXMPobject(objectName, block);
      if (onRefObject == null)
        throw new XMPexception("cannot find '" + objectName + "' nodes/template");

      Xobject ontoNodesRef = null;
      Xtype castType = null;
      switch (onRefObject.getKind()) {
        case XMPobject.NODES:
          ontoNodesRef = onRefObject.getDescId().Ref();
          castType = Xtype.intType;
          break;
        case XMPobject.TEMPLATE:
          XMPtemplate ontoTemplate = (XMPtemplate)onRefObject;
          
          if (!ontoTemplate.isDistributed())
            throw new XMPexception("template '" + objectName + "' is not distributed");
          
          XMPnodes ontoNodes = ((XMPtemplate)onRefObject).getOntoNodes();
          
          ontoNodesRef = ontoNodes.getDescId().Ref();
          castType = Xtype.longlongType;
          break;
        default:
          throw new XMPexception("unknown object type");
      }

      // create arguments
      if (onRef.getArg(1) == null || onRef.getArg(1).getArgs() == null){
        return new XMPquadruplet<String, Boolean, XobjList, XMPobject>(new String("NODES_ENTIRE"), new Boolean(false), Xcons.List(ontoNodesRef), onRefObject);
      }
      else {
        boolean splitComm   = false;
        int refIndex        = 0;
        int refDim          = onRefObject.getDim();
        String kind_bracket = onRef.getArg(1).getTail().getString();
        boolean isSquare    = kind_bracket.equals("SQUARE");
        onRef.getArg(1).removeLastArgs(); // Remove information of ROUND or SQUARE
        if(isSquare) ((XobjList)onRef.getArg(1)).reverse();

        XobjList tempArgs = Xcons.List();
        for (XobjArgs i = onRef.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
          if (refIndex == refDim)
            throw new XMPexception("wrong nodes dimension indicated, too many");

          XobjList t = (XobjList)i.getArg();
          if (t == null || t.getArgs() == null) {
            splitComm = true;
            tempArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(1)));
          }
          else {
            Xobject lower  = null;
            Xobject upper  = null;
            Xobject stride = null;
            
            tempArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(0)));

            // lower
            if (t.getArg(0) == null || (t.getArg(0) instanceof XobjList && t.getArg(0).getArgs() == null)) {
              lower = onRefObject.getLowerAt(refIndex);
            }
            else {
              splitComm = true;
              lower = t.getArg(0);
              if(onRefObject.getKind() == XMPobject.NODES && isSquare == true)
                lower = Xcons.binaryOp(Xcode.PLUS_EXPR, lower, Xcons.IntConstant(1));
            }

            // upper
            if (t.getArg(1) == null || (t.getArg(1) instanceof XobjList && t.getArg(1).getArgs() == null)) {
              upper = onRefObject.getUpperAt(refIndex);
            }
            else {
              splitComm = true;
              if(onRefObject.getKind() == XMPobject.NODES && isSquare == true){
                upper = Xcons.binaryOp(Xcode.PLUS_EXPR, t.getArg(0), t.getArg(1));
              }
              else if(onRefObject.getKind() == XMPobject.TEMPLATE && isSquare == true){
                upper = Xcons.binaryOp(Xcode.PLUS_EXPR, lower, t.getArg(1));
                upper = Xcons.binaryOp(Xcode.MINUS_EXPR, upper, Xcons.IntConstant(1));
              }
              else
                upper = t.getArg(1);
            }
            
            // stride
            if (t.getArg(2) == null || t.getArg(2).equals(Xcons.IntConstant(1))){
              stride = Xcons.IntConstant(1);
            }
            else {
              splitComm = true;
              stride = t.getArg(2);
            }

            // Output
            tempArgs.add(Xcons.Cast(castType, lower));
            tempArgs.add(Xcons.Cast(castType, upper));
            tempArgs.add(Xcons.Cast(castType, stride));
          }

          refIndex++;
        }

        if (refIndex != refDim)
          throw new XMPexception("the number of <nodes/template-subscript> should be the same with the dimension");

        if (splitComm) {
          String execFuncSuffix = null;
          XobjList execFuncArgs = null;
          execFuncArgs = tempArgs;
          switch (onRefObject.getKind()) {
            case XMPobject.NODES:
              execFuncSuffix = "NODES_PART";
              execFuncArgs.cons(ontoNodesRef);
              break;
            case XMPobject.TEMPLATE:
              execFuncSuffix = "TEMPLATE_PART";
              execFuncArgs.cons(((XMPtemplate)onRefObject).getDescId().Ref());
              break;
            default:
              throw new XMPexception("unknown object type");
          }

          return new XMPquadruplet<String, Boolean, XobjList, XMPobject>(execFuncSuffix, new Boolean(splitComm), execFuncArgs, onRefObject);
        }
        else
          return new XMPquadruplet<String, Boolean, XobjList, XMPobject>(new String("NODES_ENTIRE"),
                                                              new Boolean(splitComm), Xcons.List(ontoNodesRef),
                                                              onRefObject);
      }
    }
  }


  private void translateArray(PragmaBlock pb) throws XMPexception {

    // start translation
    XobjList arrayDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    BlockList arrayBody = pb.getBody();

    // check body
    Statement arrayStmt = null;
    Xobject assignStmt = null;
    String checkBodyErrMsg = new String("array directive should be written before one assign statement");
    Block arrayBodyHead = arrayBody.getHead();
    if (arrayBodyHead instanceof SimpleBlock) {
      if (arrayBodyHead.getNext() != null) {
        throw new XMPexception(checkBodyErrMsg);
      }

      arrayStmt = arrayBodyHead.getBasicBlock().getHead();
      if (arrayStmt.getNext() != null) {
        throw new XMPexception(checkBodyErrMsg);
      }

      if (arrayStmt.getExpr().Opcode() == Xcode.ASSIGN_EXPR) {
        assignStmt = arrayStmt.getExpr();
      }
      else {
        throw new XMPexception(checkBodyErrMsg);
      }
    }
    else {
      throw new XMPexception(checkBodyErrMsg);
    }

    Block loopBlock = convertArrayToLoop(pb, arrayStmt);
    pb.replace(loopBlock);

    translateLoop((PragmaBlock)loopBlock, false);

  }


  private Block convertArrayToLoop(PragmaBlock pb, Statement arrayStmt)  throws XMPexception {

    Xobject assignStmt = arrayStmt.getExpr();

    Xobject left = assignStmt.left();
    XMPpair<XMPalignedArray, XobjList> leftExprInfo = getXMPalignedArrayExpr(pb, left);
    XMPalignedArray leftAlignedArray = leftExprInfo.getFirst();

    List<Ident> varList = new ArrayList<Ident>(XMP.MAX_DIM);
    List<Ident> varListTemplate = new ArrayList<Ident>(XMP.MAX_DIM);
    for (int i = 0; i < XMP.MAX_DIM; i++) varListTemplate.add(null);
    List<Xobject> lbList = new ArrayList<Xobject>(XMP.MAX_DIM);
    List<Xobject> lenList = new ArrayList<Xobject>(XMP.MAX_DIM);
    List<Xobject> stList = new ArrayList<Xobject>(XMP.MAX_DIM);

    //
    // convert LHS
    //

    if (left.Opcode() != Xcode.SUB_ARRAY_REF){
      throw new XMPexception("ARRAY not followed by array ref.");
    }

    String arrayName = getArrayName(left);
    XMPalignedArray array = _globalDecl.getXMPalignedArray(arrayName, pb);
    Xtype arrayType = null;
    if (array != null){
      arrayType = array.getArrayType();
    }
    else {
      Ident arrayId = pb.findVarIdent(arrayName);
      if (arrayId != null){
	arrayType = arrayId.Type();
      }
    }
	
    if (arrayType == null) throw new XMPexception("array should be declared statically");

    Xtype elemType = arrayType.getArrayElementType();
    int n = arrayType.getNumDimensions();

    XobjList subscripts = (XobjList)left.getArg(1);

    for (int i = 0; i < n; i++, arrayType = arrayType.getRef()){

      long dimSize = arrayType.getArraySize();
      Xobject sizeExpr;
      if (dimSize == 0 || arrayType.getKind() == Xtype.POINTER){
	Ident ret = declIdentWithBlock(pb, "XMP_" + arrayName + "_ret" + Integer.toString(i),
				       Xtype.intType);
	Ident sz = declIdentWithBlock(pb, "XMP_" + arrayName + "_ub" + Integer.toString(i),
				      Xtype.intType);

	Ident f = _globalDecl.declExternFunc("xmp_array_ubound", Xtype.intType);
	Xobject args = Xcons.List(array.getDescId().Ref(), Xcons.IntConstant(i+1),
				  sz.getAddr());

	pb.insert(Xcons.Set(ret.Ref(), Xcons.binaryOp(Xcode.PLUS_EXPR, f.Call(args), Xcons.IntConstant(i+1))));
	sizeExpr = sz;
      }
      else if (dimSize == -1){
        sizeExpr = arrayType.getArraySizeExpr();
      }
      else {
	sizeExpr = Xcons.LongLongConstant(0, dimSize);
      }

      Xobject sub = subscripts.getArg(i);

      Ident var;
      Xobject lb, len, st;

      if (! sub.isIndexRange()) continue;

      var = declIdentWithBlock(pb, "_XMP_loop_i" + Integer.toString(i), Xtype.intType);
      varList.add(var);

      if (leftAlignedArray.getAlignMannerAt(i) != XMPalignedArray.NOT_ALIGNED){
	varListTemplate.set(leftAlignedArray.getAlignSubscriptIndexAt(i), var);
      }

      lb = ((XobjList)sub).getArg(0);
      if (lb == null) lb = Xcons.IntConstant(0);
      len = ((XobjList)sub).getArg(1);
      //if (len == null) len = Xcons.binaryOp(Xcode.MINUS_EXPR, sizeExpr, lb);
      if (len == null) len = sizeExpr;
      st = ((XobjList)sub).getArg(2);
      if (st == null) st = Xcons.IntConstant(1);

      lbList.add(lb);
      lenList.add(len);
      stList.add(st);

      Xobject expr;
      expr = Xcons.binaryOp(Xcode.MUL_EXPR, var.Ref(), st);
      expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

      subscripts.setArg(i, expr);

    }

    Xobject new_left = Xcons.arrayRef(elemType, left.getArg(0), subscripts);

    //
    // convert RHS
    //

    // NOTE: Since the top level object cannot be replaced, the following conversion is applied to
    //       the whole assignment.
    XobjectIterator j = new topdownXobjectIterator(assignStmt);
    for (j.init(); !j.end(); j.next()) {
      Xobject x = j.getXobject();

      if (x.Opcode() != Xcode.SUB_ARRAY_REF) continue;

      int k = 0;

      String arrayName1 = getArrayName(x);

      //Ident arrayId1 = pb.findVarIdent(arrayName1);
      //Xtype arrayType1 = arrayId.Type();

      XMPalignedArray array1 = _globalDecl.getXMPalignedArray(arrayName1, pb);
      Xtype arrayType1 = null;
      if (array1 != null){
	arrayType1 = array1.getArrayType();
      }
      else {
	Ident arrayId1 = pb.findVarIdent(arrayName1);
	if (arrayId1 != null){
	  arrayType1 = arrayId1.Type();
	}
      }
	
      if (arrayType1 == null) throw new XMPexception("array should be declared statically");

      Xtype elemType1 = arrayType1.getArrayElementType();
      int m = arrayType1.getNumDimensions();

      XobjList subscripts1 = (XobjList)x.getArg(1);

      for (int i = 0; i < m; i++, arrayType1 = arrayType1.getRef()){

	Xobject sub = subscripts1.getArg(i);

	Ident var;
	Xobject lb, st;

	if (! sub.isIndexRange()) continue;
	//if (array1.getAlignMannerAt(i) == XMPalignedArray.NOT_ALIGNED) continue;

	lb = ((XobjList)sub).getArg(0);
	if (lb == null) lb = Xcons.IntConstant(0);
	st = ((XobjList)sub).getArg(2);
	if (st == null) st = Xcons.IntConstant(1);

	Xobject expr;
	expr = Xcons.binaryOp(Xcode.MUL_EXPR, varList.get(k).Ref(), st);
	//Ident loopVar = varListTemplate.get(array1.getAlignSubscriptIndexAt(i));
	//if (loopVar == null) XMP.fatal("array on rhs does not conform to that on lhs.");
	//expr = Xcons.binaryOp(Xcode.MUL_EXPR, loopVar.Ref(), st);
	expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

	subscripts1.setArg(i, expr);
	k++;
      }

      Xobject new_x = Xcons.arrayRef(elemType1, x.getArg(0), subscripts1);
      j.setXobject(new_x);

    }

    //
    // construct loop
    //

    BlockList loop = null;

    BlockList body = Bcons.emptyBody();
    body.add(Xcons.Set(new_left, assignStmt.right()));

    for (int i = varList.size() - 1; i >= 0; i--){
      loop = Bcons.emptyBody();
      // Xobject ub = Xcons.binaryOp(Xcode.MINUS_EXPR, ubList.get(i), lbList.get(i));
      // ub = Xcons.binaryOp(Xcode.PLUS_EXPR, ub, stList.get(i));
      // ub = Xcons.binaryOp(Xcode.DIV_EXPR, ub, stList.get(i));
      // ub = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, Xcons.IntConstant(1));
      loop.add(Bcons.FORall(varList.get(i).Ref(), Xcons.IntConstant(0), lenList.get(i), Xcons.IntConstant(1),
			    Xcode.LOG_LT_EXPR, body));
      body = loop;
    }

    //
    // convert ARRAY to LOOP directive
    //

    Xobject args = Xcons.List();

    XobjList loopIterList = Xcons.List();
    for (int i = 0; i < varList.size(); i++){
      if (leftAlignedArray.getAlignMannerAt(i) != XMPalignedArray.NOT_ALIGNED){
	loopIterList.add(varList.get(i).Ref());
      }
    }
    args.add(loopIterList);

    Xobject onRef = Xcons.List();

    String templateName = pb.getClauses().getArg(0).getArg(0).getName();
    XMPtemplate template = _globalDecl.getXMPtemplate(templateName, pb);
    if (template == null) throw new XMPexception("template '" + templateName + "' not found");

    onRef.add(pb.getClauses().getArg(0).getArg(0));
    Xobject subscriptList = Xcons.List();

    Xobject onSubscripts = pb.getClauses().getArg(0).getArg(1);

    if (onSubscripts != null){
      for (int i = 0; i < onSubscripts.Nargs(); i++){
    	Xobject sub = onSubscripts.getArg(i);
    	if (sub.Opcode() == Xcode.LIST){ // triplet
    	  Xobject lb = ((XobjList)sub).getArg(0);
    	  if (lb == null || (lb.Opcode() == Xcode.LIST && lb.Nargs() == 0)){
	    if (template.isFixed()){
	      lb = template.getLowerAt(i);
	    }
	    else {
	      Ident ret = declIdentWithBlock(pb, "XMP_" + template.getName() + "_ret" + Integer.toString(i),
					     Xtype.intType);
	      Ident tlb = declIdentWithBlock(pb, "XMP_" + template.getName() + "_lb" + Integer.toString(i),
					     Xtype.intType);
	      
	      Ident f = _globalDecl.declExternFunc("xmp_template_lbound", Xtype.intType);
	      Xobject args1 = Xcons.List(template.getDescId().Ref(), Xcons.IntConstant(i+1), tlb.getAddr());
	      pb.insert(Xcons.Set(ret.Ref(), f.Call(args1)));

	      lb = tlb.Ref();
	    }
	  }
	  Xobject st = ((XobjList)sub).getArg(2);
	  if (st != null){
	    if (st.Opcode() == Xcode.INT_CONSTANT && ((XobjInt)st).getInt() == 0){ // scalar
	      subscriptList.add(sub);
	      continue;
	    }
	  }
	  else st = Xcons.IntConstant(1);

	  Ident loopVar = varListTemplate.get(i);
	  if (loopVar != null){
	    Xobject expr = Xcons.binaryOp(Xcode.MUL_EXPR, loopVar.Ref(), st);
	    expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);
	    subscriptList.add(expr);
	  }
	  else {
	    subscriptList.add(null);
	  }
    	}
	else { // scalar
    	  subscriptList.add(sub);
	}
      }
    }
    else {
      for (int i = 0; i < template.getDim(); i++){
	Xobject lb;
	if (template.isFixed()){
	  lb = template.getLowerAt(i);
	}
	else {
	  Ident ret = declIdentWithBlock(pb, "XMP_" + template.getName() + "_ret" + Integer.toString(i),
					 Xtype.intType);
	  Ident tlb = declIdentWithBlock(pb, "XMP_" + template.getName() + "_lb" + Integer.toString(i),
					 Xtype.intType);
	      
	  Ident f = _globalDecl.declExternFunc("xmp_template_lbound", Xtype.intType);
	  Xobject args1 = Xcons.List(template.getDescId().Ref(), Xcons.IntConstant(i+1), tlb.getAddr());
	  pb.insert(Xcons.Set(ret.Ref(), f.Call(args1)));

	  lb = tlb.Ref();
	}

	Ident loopVar = varListTemplate.get(i);
	if (loopVar != null){
	  Xobject expr = Xcons.binaryOp(Xcode.PLUS_EXPR, loopVar.Ref(), lb);
	  subscriptList.add(expr);
	}
	else {
	  subscriptList.add(null);
	}
      }
    }

    onRef.add(subscriptList);
    args.add(onRef);

    args.add(null); // reduction
    args.add(null); // multicore clause ?
    args.add(null); // prof
    args.add(null); // expand/margin/peel_and_wait

    return Bcons.PRAGMA(Xcode.XMP_PRAGMA, "LOOP", args, loop);
  }


  private void analyzeStaticDesc(PragmaBlock pb){

    Block parentBlock = pb.getParentBlock();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable2(parentBlock);

    XobjList xmpObjList = (XobjList)pb.getClauses().getArg(0);
    for (Xobject xx: xmpObjList){
      String objName = xx.getString();
      localXMPsymbolTable.putStaticDesc(objName);
    }

  }


  private void setupFinalizer(BlockList body, Ident funcId, XobjList args) throws XMPexception {
    BlockIterator i = new topdownBlockIterator(body);
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() == Xcode.GOTO_STATEMENT)
        throw new XMPexception("cannot use goto statement here");
      else if (b.Opcode() == Xcode.RETURN_STATEMENT)
        b.insert(funcId.Call(args));
    }
    body.add(funcId.Call(args));
  }

  public static void checkDeclPragmaLocation(PragmaBlock pb) throws XMPexception {
/*
    String pragmaNameL = pb.getPragma().toLowerCase();
    String errMsg = new String(pragmaNameL +
                               " directive should be written before declarations, statements and executable directives");

    // check parent
    Block parentBlock = pb.getParentBlock();
    if (parentBlock.Opcode() != Xcode.COMPOUND_STATEMENT)
      throw new XMPexception(errMsg);
    else {
      BlockList parent = pb.getParent();
      Xobject declList = parent.getDecls();
      if (declList != null) {
        if (declList.operand() != null)
          throw new XMPexception(errMsg);
      }

      if (parentBlock.getParentBlock().Opcode() != Xcode.FUNCTION_DEFINITION)
        throw new XMPexception(errMsg);
    }

    // check previous blocks
    for (Block prevBlock = pb.getPrev(); prevBlock != null; prevBlock = prevBlock.getPrev()) {
      if (prevBlock.Opcode() == Xcode.XMP_PRAGMA) {
        XMPpragma prevPragma = XMPpragma.valueOf(((PragmaBlock)prevBlock).getPragma());
        switch (XMPpragma.valueOf(((PragmaBlock)prevBlock).getPragma())) {
          case NODES:
          case TEMPLATE:
          case DISTRIBUTE:
          case ALIGN:
          case SHADOW:
            continue;
          default:
            throw new XMPexception(errMsg);
        }
      }
      else
        throw new XMPexception(errMsg);
    }
*/
    // XXX delete this
    return;
  }

  public void set_all_profile(){
      _all_profile = true;
  }

  public void set_selective_profile(){
      _selective_profile = true;
  }

  public void setScalascaEnabled(boolean v) {
      doScalasca = v;
  }

  public void setTlogEnabled(boolean v) {
      doTlog = v;
  }

    private Block createScalascaStartProfileCall(XobjList funcArgs) {
        Ident funcId = XMP.getMacroId("_XMP_M_EPIK_USER_START");
        BlockList funcCallList = Bcons.emptyBody();

        funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
        return Bcons.COMPOUND(funcCallList);
    }

    private Block createScalascaEndProfileCall(XobjList funcArgs) {
        Ident funcId = XMP.getMacroId("_XMP_M_EPIK_USER_END");
        BlockList funcCallList = Bcons.emptyBody();

        funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
        return Bcons.COMPOUND(funcCallList);
    }

    private Block createScalascaProfileOffCall(XobjList funcArgs) {
        Ident funcId = XMP.getMacroId("_XMP_M_EPIK_GEN_OFF");
        BlockList funcCallList = Bcons.emptyBody();

        funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
        return Bcons.COMPOUND(funcCallList);
    }

    private Block createScalascaProfileOnfCall(XobjList funcArgs) {
        Ident funcId = XMP.getMacroId("_XMP_M_EPIK_GEN_ON");
        BlockList funcCallList = Bcons.emptyBody();

        funcCallList.add(Bcons.Statement(funcId.Call(funcArgs)));
        return Bcons.COMPOUND(funcCallList);
    }

    private Block createTlogMacroInvoke(String macro, XobjList funcArgs) {
        Ident macroId = XMP.getMacroId(macro);
        BlockList funcCallList = Bcons.emptyBody();
        funcCallList.add(Bcons.Statement(macroId.Call(funcArgs)));
        return Bcons.COMPOUND(funcCallList);
    }

  private static Block divideMarginLoop(PragmaBlock pb){

    // The type is XMP.LOOP_MARGIN

    BlockList loops = Bcons.emptyBody();
    boolean flag = false;
    
    XobjList expandOpt = (XobjList)pb.getClauses().getArg(5);

    // System.out.println("("+expandOpt.getArg(1).getArg(0).getArg(0).getInt()+" : "+expandOpt.getArg(1).getArg(0).getArg(1).getInt()+" ,"
    // 		       +expandOpt.getArg(1).getArg(1).getArg(0).getInt()+" : "+expandOpt.getArg(1).getArg(1).getArg(1).getInt()+")");

    PragmaBlock pb1, pb2;
    XobjList expandOpt1 = null, expandOpt2 = null;

    for (int i = 0; i < expandOpt.getArg(1).Nargs(); i++){

      Xobject expandWidth = expandOpt.getArg(1).getArg(i);
      Xobject lower = expandWidth.getArg(0);
      Xobject upper = expandWidth.getArg(1);

      Xobject stride = expandWidth.getArg(2);
      if (stride.isIntConstant() && stride.getInt() == -1) continue;

      if (!lower.isZeroConstant() && !upper.isZeroConstant()){

	flag = true;

	// PragmaBlock pb1, pb2;
	// XobjList expandOpt1, expandOpt2;
	
	// for lower margin
	pb1 = (PragmaBlock)pb.copy();
	expandOpt1 = (XobjList)pb1.getClauses().getArg(5);

	for (int j = 0; j < expandOpt1.getArg(1).Nargs(); j++){
	  Xobject expandWidth1 = expandOpt1.getArg(1).getArg(j);
	  if (j == i){
	    expandWidth1.setArg(0, lower);
	    expandWidth1.setArg(1, Xcons.IntConstant(0));
	  }
	  else if (j > i){
	    expandWidth1.setArg(2, Xcons.IntConstant(-1)); // edge of margin
	  }
	  else { // j < i
	    expandWidth1.setArg(0, Xcons.IntConstant(0));
	    expandWidth1.setArg(1, Xcons.IntConstant(0));
	  }
	}

	// System.out.println(" ("+expandOpt1.getArg(1).getArg(0).getArg(0).getInt()+" : "+expandOpt1.getArg(1).getArg(0).getArg(1).getInt()+" ,"
	// 		   +expandOpt1.getArg(1).getArg(1).getArg(0).getInt()+" : "+expandOpt1.getArg(1).getArg(1).getArg(1).getInt()+")");

	loops.add(pb1);
			     
	// for upper margin
	pb2 = (PragmaBlock)pb.copy();
	expandOpt2 = (XobjList)pb2.getClauses().getArg(5);

	for (int j = 0; j < expandOpt1.getArg(1).Nargs(); j++){
	  Xobject expandWidth2 = expandOpt2.getArg(1).getArg(j);
	  if (j == i){
	    expandWidth2.setArg(0, Xcons.IntConstant(0));
	    expandWidth2.setArg(1, upper);
	  }
	  else if (j > i){
	    expandWidth2.setArg(2, Xcons.IntConstant(-1)); // edge of margin
	  }
	  else { // j < i
	    expandWidth2.setArg(0, Xcons.IntConstant(0));
	    expandWidth2.setArg(1, Xcons.IntConstant(0));
	  }	    
	}

      // System.out.println(" ("+expandOpt2.getArg(1).getArg(0).getArg(0).getInt()+" : "+expandOpt2.getArg(1).getArg(0).getArg(1).getInt()+" ,"
      // 			 +expandOpt2.getArg(1).getArg(1).getArg(0).getInt()+" : "+expandOpt2.getArg(1).getArg(1).getArg(1).getInt()+")");

    
	loops.add(pb2);

      }

    }

    if (flag){
      return Bcons.COMPOUND(loops);
    }
    else {
      return null;
    }

  }

  private static Block peelLoop(PragmaBlock pb){

    // The type is XMP.LOOP_PEEL_AND_WAIT

    BlockList bl = Bcons.emptyBody();

    // First, create the kernel loop
    PragmaBlock pb3 = (PragmaBlock)pb.copy();
    pb3.getClauses().getArg(5).setArg(0, Xcons.IntConstant(LOOP_EXPAND));
    pb3.getClauses().getArg(5).setArg(1, pb.getClauses().getArg(5).getArg(2));
    bl.add(pb3);

    // Second, create wait_async
    XobjList clauses = Xcons.List();
    clauses.add(Xcons.List(pb.getClauses().getArg(5).getArg(1)));
    clauses.add(null);
    PragmaBlock pb2 = new PragmaBlock(Xcode.XMP_PRAGMA, "WAIT_ASYNC", clauses, null);
    bl.add(pb2);
    
    // Third, create the peeled Loop
    PragmaBlock pb1 = (PragmaBlock)pb.copy();
    pb1.getClauses().getArg(5).setArg(0, Xcons.IntConstant(LOOP_MARGIN));
    pb1.getClauses().getArg(5).setArg(1, pb.getClauses().getArg(5).getArg(2));
    bl.add(pb1);

    return Bcons.COMPOUND(bl);

  }

}
