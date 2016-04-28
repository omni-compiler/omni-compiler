/* 
 * $TSUKUBA_Release: Omni XcalableMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import java.util.ArrayList;
import java.util.List;

import exc.object.*;
import exc.block.*;
import java.util.*;
import xcodeml.util.XmOption;

/**
 * process analyze XcalableMP pragma
 * pass1: check directive and allocate descriptor
 */
public class XMPanalyzePragma
{
  XMPenv env;

  public XMPanalyzePragma() {}

  public void run(FuncDefBlock def, XMPenv env) {
    this.env = env;
    env.setCurrentDef(def);

    Block b;
    Block fblock = def.getBlock();


    // pass1: traverse to collect information about XMP pramga
    XMP.debug("pass1:");

    // check module use 
    checkUseDecl(fblock.getBody().getDecls());

    b = fblock.getBody().getHead();
    if(b == null) return; // null body, do nothing
    b.setProp(XMP.prop, new XMPinfo(XMPpragma.FUNCTION_BODY, null, b, env));

    // scan by topdown iterator
    BlockIterator i = new topdownBlockIterator(fblock);
    for(i.init(); !i.end(); i.next()) {
      b = i.getBlock();
      if(XMP.debugFlag)	System.out.println("pass1=" + b);

      if (b.Opcode() == Xcode.XMP_PRAGMA){

	PragmaBlock pb = (PragmaBlock)b;

	// These two pragmas must be processed in advance because they may be converted into another
	// ones, which will be processed again later.

	if (XMPpragma.valueOf(pb.getPragma()) == XMPpragma.GMOVE){
	  analyzePragma(pb);
	}

	if (XMPpragma.valueOf(pb.getPragma()) == XMPpragma.ARRAY){
	  b = analyzeArray(pb);
	  if (b != null) i.setBlock(b);
	}

	analyzePragma((PragmaBlock)b);
      }
    }
  }

  private void checkUseDecl(Xobject decls){
    if(decls == null) return;
    for(Xobject decl: (XobjList)decls){
      if(decl.Opcode() != Xcode.F_USE_DECL) continue;
      String name = decl.getArg(0).getName();
      env.useModule(name);
      env.findModule(name); // import 
    }
  }

  private void analyzePragma(PragmaBlock pb) {
    String pragmaName = pb.getPragma();

    // debug
    if(XMP.debugFlag){
      System.out.println("Pragma: "+pragmaName);
      System.out.println(" Clause= "+pb.getClauses());
      System.out.println(" Body= "+pb.getBody());
    }

    XMPinfo outer = null;
    for(Block bp = pb.getParentBlock(); bp != null; bp = bp.getParentBlock()) {
      if(bp.Opcode() == Xcode.XMP_PRAGMA) {
	outer = (XMPinfo)bp.getProp(XMP.prop);
      }
    }
    XMPinfo info = new XMPinfo(XMPpragma.valueOf(pb.getPragma()), outer, pb, env);
    pb.setProp(XMP.prop, info);

    XMPpragma p = info.pragma;
    switch (p){
    case NODES:
      {
	Xobject clauses = pb.getClauses();
	XMPnodes.analyzePragma(clauses, env, pb);
      }
      break;

    case TEMPLATE:
      {
	Xobject templateDecl = pb.getClauses();
	XobjList templateNameList = (XobjList)templateDecl.getArg(0);
	  
	for(Xobject xx:templateNameList){
	  XMPtemplate.analyzeTemplate(xx,templateDecl.getArg(1), env, pb);
	}
      }
      break;

    case DISTRIBUTE:
      {
	Xobject distributeDecl = pb.getClauses();
	XobjList distributeNameList = (XobjList)distributeDecl.getArg(0);
	Xobject distributeDeclCopy = distributeDecl.copy();

	for(Xobject xx:distributeNameList){
	  XMPtemplate.analyzeDistribute(xx,distributeDecl.getArg(1),
					distributeDecl.getArg(2),env, pb);
	}
      }
      break;

    case ALIGN:
      {
	Xobject alignDecl = pb.getClauses();
	XobjList alignNameList = (XobjList)alignDecl.getArg(0);

	for(Xobject xx: alignNameList){
	  XMParray.analyzeAlign(xx, alignDecl.getArg(1),
				  alignDecl.getArg(2),
				  alignDecl.getArg(3),
				  env, pb);
	  if(XMP.hasError()) break;
	}
      }
      break;

    case SHADOW:
      {
 	Xobject shadowDecl = pb.getClauses();
 	XobjList shadowNameList = (XobjList) shadowDecl.getArg(0);
	Xobject shadow_w_list = shadowDecl.getArg(1);

	for(Xobject xx: shadowNameList){
	  XMParray.analyzeShadow(xx,shadow_w_list,env,pb);
	  if(XMP.hasError()) break;
 	}
      }
      break;

    case LOCAL_ALIAS:
      analyzeLocalAlias(pb.getClauses(), env, pb);
      break;

    case SAVE_DESC:
      {
	XobjList xmpObjList = (XobjList)pb.getClauses();

	for (Xobject xx: xmpObjList){
	  String objName = xx.getString();
	  XMPobject xmpObject = env.findXMPobject(objName, pb);
	  if (xmpObject != null){
	    xmpObject.setSaveDesc(true);
	    continue;
	  }
	  Ident id = env.findVarIdent(objName, pb);
	  if (id != null){
	    XMParray array =  XMParray.getArray(id);
	    if (array != null){
	      array.setSaveDesc(true);
	    }
	    else {
	      XMP.errorAt(pb, "object '" + objName + "' is not an aligned array");
	    }
	    continue;
	  }
	  XMP.errorAt(pb, "object '" + objName + "' is not declared");
 	}
      }
      break;

    case TEMPLATE_FIX:
      analyzeTemplateFix(pb.getClauses(), info, pb);
      break;

    case LOOP:
      analyzeLoop(pb.getClauses(), pb.getBody(), pb,info);
      break; 

    case REFLECT:
      analyzeReflect(pb.getClauses(),info,pb);
      break;

    case BARRIER:
      analyzeBarrier(pb.getClauses(),info,pb);
      break;

    case REDUCTION:
      analyzeReduction(pb.getClauses(),info,pb);
      break;

    case BCAST:
      analyzeBcast(pb.getClauses(),info,pb);
      break;

    case WAIT_ASYNC:
      analyzeWaitAsync(pb.getClauses(),info,pb);
      break;

    case TASK:
      analyzeTask(pb.getClauses(), pb.getBody(), info, pb);
      break;

    case TASKS:
      analyzeTasks(pb.getClauses(), pb.getBody(), info, pb);
      break;

    case GMOVE:
      analyzeGmove(pb.getClauses(), pb.getBody(), info, pb);
      break;

    case COARRAY:
      XMPcoarray.analyzeCoarrayDirective(pb.getClauses(), env, pb);
      break;

    case IMAGE:
      XMPtransCoarrayRun.analyzeImageDirective(pb.getClauses(), env, pb);
      break;

    case ARRAY:
      //analyzeArray(pb.getClauses(), pb.getBody(), info, pb);
      break;

    default:
      XMP.fatal("'" + pragmaName.toLowerCase() + 
		"' directive is not supported yet");
    }
  }

  private static boolean isEqualVar(Xobject v1, Xobject v2){
    return (v1.isVariable() && v2.isVariable() &&
	    v1.getName().equals(v2.getName()));
  }

  public static void analyzeLocalAlias(Xobject localAliasDecl, 
				 XMPenv env, PragmaBlock pb){

    // global array

    String gName = localAliasDecl.getArg(1).getName();

    Ident gArrayId = env.findVarIdent(gName, pb);
    if (gArrayId == null) {
      XMP.errorAt(pb, "global array '" + gName + "' is not declared");
      return;
    }

    XMParray gObject = XMParray.getArray(gArrayId);
    if (gObject == null){
      XMP.errorAt(pb, "global array '" + gName  + "' is not aligned");
      return;
    }

    // local array

    String lName = localAliasDecl.getArg(0).getName();

    Ident lArrayId = env.findVarIdent(lName, pb);
    if (lArrayId == null) {
      XMP.errorAt(pb, "local alias '" + lName + "' is not declared");
      return;
    }

    XMParray lObject = XMParray.getArray(lArrayId);
    if (lObject != null){
      XMP.errorAt(pb, "local alias '" + lName  + "' is aligned");
      return;
    }

    // check type matching

    FarrayType gType = (FarrayType)gArrayId.Type();
    FarrayType lType = (FarrayType)lArrayId.Type();

    if (!lType.isFassumedShape()){
      XMP.errorAt(pb, "local alias must be declared as an assumed-shape array.");
      return;
    }      

    if (gType.getNumDimensions() != lType.getNumDimensions()){
      XMP.errorAt(pb, "The rank is different between the global array and the local alias");
      return;
    }

    if (!gType.getRef().equals(lType.getRef())){
      XMP.errorAt(pb, "The element type unmatched between the global array and the local alias");
      return;
    }

    // replace name

    Ident origLocalId = gObject.getLocalId();
    Xtype localType = origLocalId.Type();
    StorageClass sclass = origLocalId.getStorageClass();

    env.removeIdent(lName, pb);
    env.removeIdent(origLocalId.getName(), pb);

    Ident newLocalId = env.declIdent(lName, localType, false, pb);
    newLocalId.setStorageClass(sclass);
    newLocalId.setValue(Xcons.Symbol(Xcode.VAR, localType, lName));

    gObject.setLocalId(newLocalId);

  }

  private void analyzeTemplateFix(Xobject tfixDecl, 
				  XMPinfo info, PragmaBlock pb){

    XobjList distList = (XobjList)tfixDecl.getArg(0);
    Xobject t = tfixDecl.getArg(1);
    XobjList sizeList = (XobjList)tfixDecl.getArg(2);

    // get template object
    String tName = t.getString();
    XMPtemplate tObject = env.findXMPtemplate(tName, pb);

    if (tObject == null) {
      XMP.errorAt(pb, "template '" + tName  + "' is not declared");
      return;
    }

    if (tObject.isFixed()) {
      XMP.errorAt(pb, "template '" + tName + "' is alreday fixed");
    }

    if (!sizeList.isEmptyList()){
      for (int i = 0; i < tObject.getDim(); i++){
	if (sizeList.getArg(i).Opcode() != Xcode.LIST){
	  sizeList.setArg(i, Xcons.List(Xcons.IntConstant(1), sizeList.getArg(i)));
	}
      }
    }

    // set info
    info.setTemplateFix(tObject, sizeList, distList);

  }


  /* 
   * analyze Loop directive:
   *  loopDecl = (on_ref | ...)
   */
  void analyzeLoop(Xobject loopDecl, BlockList loopBody,
		   PragmaBlock pb, XMPinfo info) {
    // get block to schedule
    Vector<XMPdimInfo> dims = new Vector<XMPdimInfo>();
    
    // schedule loop
    XobjList loopIterList = (XobjList)loopDecl.getArg(0);
    if (loopIterList == null || loopIterList.Nargs() == 0) {
      ForBlock loopBlock = getOutermostLoopBlock(loopBody);
      if(loopBlock == null){
    	  XMP.errorAt(pb,"loop is not found after loop directive");
    	  return;
      }
      dims.add(XMPdimInfo.loopInfo(loopBlock));
      loopBody = loopBlock.getBody();
    } else {
      while(true){
    	  ForBlock loopBlock = getOutermostLoopBlock(loopBody);
    	  if(loopBlock == null) break;
    	  boolean is_found = false;
    	  for(Xobject x: loopIterList){
    		  if(x.Opcode() == Xcode.LIST) x = x.getArg(0);
    		  if(isEqualVar(loopBlock.getInductionVar(),x)){
    			  is_found = true;
    			  break;
    		  }
    	  }
    	  if(is_found)
    		  dims.add(XMPdimInfo.loopInfo(loopBlock));
    	  loopBody = loopBlock.getBody();
      }

      /* check loopIterList */
      for(Xobject x: loopIterList){
    	  if(x.Opcode() == Xcode.LIST){
    		  if(x.getArgOrNull(1) != null ||
    		     x.getArgOrNull(2) != null){
    			  XMP.errorAt(pb,"bad syntax in loop directive");
    		  }
    		  x = x.getArg(0);
    	  }
    	  boolean is_found = false;
    	  for(XMPdimInfo d_info: dims){
    		  if(isEqualVar(d_info.getLoopVar(),x)){
    			  is_found = true;
    			  break;
    		  }
    	  }
    	  if(!is_found)
    		  XMP.errorAt(pb,"loop index is not found in loop varaibles");
      }
    }
    
    XMPobjectsRef on_ref = XMPobjectsRef.parseDecl(loopDecl.getArg(1),env,pb);
    if(XMP.hasError()) return;

    /* check on ref: it should be v+off */
    int on_ref_idx = 0;
    Vector<XMPdimInfo> on_ref_dims = on_ref.getSubscripts();
    for(int k = 0; k < on_ref_dims.size(); k++){
      XMPdimInfo d_info = on_ref_dims.elementAt(k);
      if(d_info.isStar()) continue;
      if(d_info.isTriplet()){
	  //XMP.errorAt(pb,"on-ref in loop must not be triplet");
	  continue;
      } else {
	Xobject t = d_info.getIndex();
	if(t == null) continue;
	Xobject v = null;
	Xobject off = null;
	if(t.isVariable()){
	  v = t;
	} else {
	  switch(t.Opcode()){
	  case PLUS_EXPR:
	  case MINUS_EXPR:
	    if(!t.left().isVariable())
	      XMP.errorAt(pb,"left-hand side in align-subscript must be a variable");
	    else {
	      v = t.left();
	      off = t.right();
	      if(t.Opcode() == Xcode.MINUS_EXPR)
		off = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR,off);
	    }
	    // check right-hand side?
	    break;
	  default:
	    XMP.errorAt(pb,"bad expression in subsript of on-ref");
	  }
	}
	if(v == null) continue; // some error occurs
	// find varaible
	int idx = -1;
	for(int i = 0; i < dims.size(); i++){
	  if(isEqualVar(v,dims.elementAt(i).getLoopVar())){
	    idx = i;
	    dims.elementAt(i).setLoopOnIndex(k);
	    break;
	  }
	}
	if(idx < 0)
	  XMP.errorAt(pb,"loop variable is not found in on_ref: '"
		      +v.getName()+"'");
	d_info.setLoopOnRefInfo(idx,off);
      }
    }
    
    Xobject reductionSpec = loopDecl.getArg(2);
    if(reductionSpec != null) 
      analyzeReductionSpec(info, reductionSpec, pb);

    on_ref.setLoopDimInfo(dims);  // set back pointer
    
    checkLocalizableLoop(dims,on_ref,pb);

    info.setBody(loopBody);  // inner most body
    info.setLoopInfo(dims, on_ref);
  }

  private static ForBlock getOutermostLoopBlock(BlockList body) {
    Block b = body.getHead();
    while (b != null) {
      if (b.Opcode() == Xcode.F_DO_STATEMENT) {
//         if (b.getNext() != null){
//           // XMP.error("only one loop statement is allowed in loop directive");
// 	  return null;
// 	}
        ForBlock forBlock = (ForBlock)b;
        forBlock.Canonicalize();
        if (!(forBlock.isCanonical())){
        	// XMP.error("loop statement is not canonical");
        	return null;
        }
        return forBlock;
      }
      else if (b.Opcode() == Xcode.COMPOUND_STATEMENT)
    	  return getOutermostLoopBlock(b.getBody());
      else if (b.Opcode() == Xcode.OMP_PRAGMA)
    	  return getOutermostLoopBlock(b.getBody());
//       else if(b.Opcode() == Xcode.F_STATEMENT_LIST &&
// 	      b.getBasicBlock().getHead().getExpr().Opcode() 
// 	      == Xcode.PRAGMA_LINE) 
// 	b = b.getNext();   // skip pragma_line
//       else return null;  // otherwise, failed.
      b = b.getNext();
    } 
    return null;
  }

  /*
   * distributed loop is localizable if:
   * (1) step is 1
   * (2) distribution is BLOCK
   */
  private void checkLocalizableLoop(Vector<XMPdimInfo> dims,
				    XMPobjectsRef on_ref, Block b){
    for(int i = 0; i < dims.size(); i++){
      boolean localizable = false;
      XMPdimInfo d_info = dims.elementAt(i);
      if(d_info.getStride().isOneConstant()) 
	localizable = true;
      else {
	if(!(on_ref.getRefObject() instanceof XMPtemplate))
	  continue;
	XMPtemplate tmpl = (XMPtemplate)on_ref.getRefObject();
	if(tmpl.isDistributed()){
	  if(tmpl.getDistMannerAt(d_info.getLoopOnIndex()) 
	     == XMPtemplate.BLOCK)
	    localizable = true;
	} else 
	  XMP.errorAt(b,"template '"+tmpl.getName()+"' is not distributed");
      }

      if(XMP.debugFlag)
	System.out.println("localizable(i="+i+")="+localizable);

      if(localizable){
	Xobject loop_var = d_info.getLoopVar();
	/* if localiable, allocate local */
	Ident local_loop_var = 
	  env.declIdent(XMP.genSym(loop_var.getName()),
			loop_var.Type(),b);
	d_info.setLoopLocalVar(local_loop_var);
      }
    }
  }

  private void analyzeReflect(Xobject reflectDecl, 
			      XMPinfo info, PragmaBlock pb){
    XobjList reflectNameList = (XobjList) reflectDecl.getArg(0);
    XobjList widthOpt = (XobjList) reflectDecl.getArg(1);
    Xobject asyncOpt = reflectDecl.getArg(2);

//     if(reflectOpt != null){
//       XMP.fatal("reflect opt is not supported yet, sorry!");
//       return;
//     }

    Vector<XMParray> reflectArrays = new Vector<XMParray>();
    // check array
    for(Xobject x: reflectNameList){
      if(!x.isVariable()){
	XMP.errorAt(pb,"Bad array name in reflect name list");
	continue;
      }
      String name = x.getName();
      Ident id = env.findVarIdent(name,pb);
      if(id == null){
	XMP.errorAt(pb,"variable '" + name + "'for reflect is not declared");
	continue;
      }
      XMParray array =  XMParray.getArray(id);
      if(array == null){
	XMP.errorAt(pb,"array '" + name + "'for reflect is not declared");
	continue;
      }
      reflectArrays.add(array);
    }

    Vector<XMPdimInfo> widthList = new Vector<XMPdimInfo>();
    // width list
    for (Xobject x: widthOpt){
	XMPdimInfo width = new XMPdimInfo();

	if(XMP.debugFlag)
	    System.out.println("width = ("+x.getArg(0)+":"+x.getArg(1)+":"+x.getArg(2)+")");

	width.parse(x);
	widthList.add(width);
    }

    info.setReflectArrays(reflectArrays, widthList);
    info.setAsyncId(asyncOpt);

  }

  void analyzeBarrier(Xobject barrierDecl, 
		      XMPinfo info, PragmaBlock pb) {
    Xobject barrierOnRef = barrierDecl.getArg(0);
    Xobject barrierOpt = barrierDecl.getArg(1);

    if(barrierOpt != null){
      System.out.println("opt="+barrierOpt);
      XMP.fatal("barrier opt is not supported yet, sorry!");
      return;
    }

    info.setOnRef(XMPobjectsRef.parseDecl(barrierOnRef,env,pb));
  }

  void analyzeReduction(Xobject reductionDecl, 
			XMPinfo info, PragmaBlock pb){
    Xobject reductionSpec = reductionDecl.getArg(0);
    Xobject reductionOnRef = reductionDecl.getArg(1);
    Xobject asyncOpt = reductionDecl.getArg(2);

    // if(reductionOpt != null){
    //   XMP.fatal("redution opt is not supported yet, sorry!");
    //   return;
    // }

    analyzeReductionSpec(info, reductionSpec, pb);
    info.setOnRef(XMPobjectsRef.parseDecl(reductionOnRef,env,pb));

    if (asyncOpt != null && !XmOption.isAsync()){
      XMP.errorAt(pb, "MPI-3 is required to use the async clause on a reduction directive");
    }

    info.setAsyncId(asyncOpt);
  }

  private void analyzeReductionSpec(XMPinfo info, Xobject reductionSpec,
				    PragmaBlock pb){
    Xobject op = reductionSpec.getArg(0);
    if(!op.isIntConstant()) XMP.fatal("reduction: op is not INT");

    Vector<Ident> reduction_vars = new Vector<Ident>();
    Vector<Vector<Ident>> reduction_pos_vars = new Vector<Vector<Ident>>();

    for(Xobject w: (XobjList)reductionSpec.getArg(1)){

      Xobject v = w.getArg(0);
      if(!v.isVariable()){
	XMP.errorAt(pb,"not variable in reduction spec list");
      }
      Ident id = env.findVarIdent(v.getName(),pb);
      if(id == null){
	XMP.errorAt(pb,"variable '"+v.getName()+"' in reduction is not found");
      }
      reduction_vars.add(id);

      Vector<Ident> plist = new Vector<Ident>();
      if (w.getArg(1) != null){
	for (Xobject p: (XobjList)w.getArg(1)){
	  Ident pid = env.findVarIdent(p.getName(), pb);
	  if (pid == null){
	    XMP.errorAt(pb, "variable '" + p.getName() + "' in reduction is not found");
	  }
	  plist.add(pid);
	}
      }

      reduction_pos_vars.add(plist);
    }

    info.setReductionInfo(op.getInt(), reduction_vars, reduction_pos_vars);
  }

  private void analyzeBcast(Xobject bcastDecl, 
			XMPinfo info, PragmaBlock pb){
    XobjList bcastNameList = (XobjList) bcastDecl.getArg(0);
    Xobject fromRef = bcastDecl.getArg(1);
    Xobject onRef = bcastDecl.getArg(2);
    Xobject asyncOpt = bcastDecl.getArg(3);

    // if(bcastOpt != null){
    //   XMP.fatal("bcast opt is not supported yet, sorry!");
    //   return;
    // }

    Vector<Ident> bcast_vars = new Vector<Ident>();
    for(Xobject v: bcastNameList){
      if(!v.isVariable()){
	XMP.errorAt(pb,"not variable in bcast variable list");
      }
      Ident id = env.findVarIdent(v.getName(),pb);
      if(id == null){
	XMP.errorAt(pb,"variable '"+v.getName()+"' in reduction is not found");
      }
      bcast_vars.add(id);
    }
    
    info.setBcastInfo(XMPobjectsRef.parseDecl(fromRef,env,pb),
		      XMPobjectsRef.parseDecl(onRef,env,pb),
		      bcast_vars);

    if (asyncOpt != null && !XmOption.isAsync()){
      XMP.errorAt(pb, "MPI-3 is required to use the async clause on a bcast directive");
    }

    info.setAsyncId(asyncOpt);
  }

  private void analyzeWaitAsync(Xobject waitAsyncDecl, 
				XMPinfo info, PragmaBlock pb){

    XobjList asyncIdList = (XobjList) waitAsyncDecl.getArg(0);
    Vector<Xobject> asyncIds = new Vector<Xobject>();
    for (Xobject x: asyncIdList){
	asyncIds.add(x);
    }
    info.setWaitAsyncIds(asyncIds);

    Xobject onRef = waitAsyncDecl.getArg(1);
    info.setOnRef(XMPobjectsRef.parseDecl(onRef, env, pb));

  }

  void analyzeTask(Xobject taskDecl, BlockList taskBody,
		   XMPinfo info, PragmaBlock pb) {
    Xobject onRef = taskDecl.getArg(0);
    Xobject nocomm = taskDecl.getArg(1);
    // Xobject taskOpt = taskDecl.getArg(1);
    
    // if(taskOpt != null){
    //   XMP.fatal("task opt is not supported yet, sorry!");
    //   return;
    // }
    info.setOnRef(XMPobjectsRef.parseDecl(onRef,env,pb));
    info.setNocomm(nocomm);
  }

  private void analyzeTasks(Xobject tasksDecl, BlockList taskList,
			    XMPinfo info, PragmaBlock pb){
    //XMP.fatal("analyzeTasks");
  }

  // private void analyzeTasks(PragmaBlock pb) {
  //   XMP.fatal("analyzeTasks");
  // }

  private void analyzeGmove(Xobject gmoveDecl, BlockList body, 
			    XMPinfo info, PragmaBlock pb) {
    Xobject gmoveOpt = gmoveDecl.getArg(0); // NORMAL | IN | OUT
    Xobject asyncOpt = gmoveDecl.getArg(1);
    //Xobject Opt = gmoveDecl.getArg(2);

    // check body is single statement.
    Block b = body.getHead();
    if(b.getNext() != null) XMP.fatal("not single block for Gmove");
    Statement s = b.getBasicBlock().getHead();
    if(b.Opcode() != Xcode.F_STATEMENT_LIST ||s.getNext() != null)
      XMP.fatal("not single statment for Gmove");
    Xobject x = s.getExpr();
    if(x.Opcode() != Xcode.F_ASSIGN_STATEMENT)
      XMP.fatal("not assignment for Gmove");
    Xobject left = x.left();
    Xobject right = x.right();
    
    // opcode must be VAR or ARRAY_REF
    boolean left_is_global = checkGmoveOperand(left, gmoveOpt.getInt() == XMP.GMOVE_OUT, pb);
    boolean right_is_global = checkGmoveOperand(right, gmoveOpt.getInt() == XMP.GMOVE_IN, pb);

    boolean left_is_scalar = isScalar(left);
    boolean right_is_scalar = isScalar(right);

    if (left_is_scalar && !right_is_scalar){
      XMP.fatal("Incompatible ranks in assignment.");
    }

    if (!right_is_global && gmoveOpt.getInt() == XMP.GMOVE_IN)
      XMP.errorAt(pb, "RHS should be global in GMOVE IN.");

    if (!left_is_global && gmoveOpt.getInt() == XMP.GMOVE_OUT)
      XMP.errorAt(pb, "LHS should be global in GMOVE OUT.");

    if (!left_is_global){
      if (!right_is_global){
	// local assignment
	info.setGmoveOperands(null, null);
	return;
      }
      // else if (right_is_scalar){
      // 	// make bcast
      // 	info.setGmoveOperands(null, null);
      // 	return;
      // }
    }
    else if (left_is_global){
      if (!right_is_global){
	if (gmoveOpt.getInt() == XMP.GMOVE_NORMAL &&
	    !left_is_scalar &&
	    convertGmoveToArray(pb, left, right)) return;
      }
      // else if (right_is_scalar){
      // 	// make bcast
      // 	if (convertGmoveToArray(pb, left, right)) return;
      // }
    }

    if(XMP.hasError()) return;
    
    info.setGmoveOperands(left,right);

    info.setGmoveOpt(gmoveOpt);

    if (asyncOpt != null && !XmOption.isAsync()){
      XMP.errorAt(pb, "MPI-3 is required to use the async clause on a gmove directive");
    }

    info.setAsyncId(asyncOpt);
  }

  private boolean checkGmoveOperand(Xobject x, boolean remotely_accessed, PragmaBlock pb){

    Ident id = null;
    XMParray array = null;

    switch(x.Opcode()){
    case F_ARRAY_REF:
      Xobject a = x.getArg(0);
      if(a.Opcode() != Xcode.F_VAR_REF)
	XMP.fatal("not F_VAR_REF for F_ARRAY_REF");
      a = a.getArg(0);
      if(a.Opcode() != Xcode.VAR)
	XMP.fatal("not VAR for F_VAR_REF");
      id = env.findVarIdent(a.getName(),pb);
      if(id == null)
	XMP.fatal("array in F_ARRAY_REF is not declared");
      array = XMParray.getArray(id);
      if(array != null){
	if (remotely_accessed && id.getStorageClass() != StorageClass.FSAVE){
	  XMP.fatal("Current limitation: Only a SAVE or MODULE variable can be the target of gmove in/out.");
	}
	if (XMPpragma.valueOf(pb.getPragma()) != XMPpragma.ARRAY) a.setProp(XMP.RWprotected, array);
	return true;
      }
      break;
    case VAR:
      id = env.findVarIdent(x.getName(), pb);
      if (id == null)
	XMP.fatal("variable" + x.getName() + "is not declared");
      array = XMParray.getArray(id);
      if (array != null){
	if (remotely_accessed && id.getStorageClass() != StorageClass.FSAVE){
	  XMP.fatal("Current limitation: Only a SAVE or MODULE variable can be the target of gmove in/out.");
	}
	if (XMPpragma.valueOf(pb.getPragma()) != XMPpragma.ARRAY) x.setProp(XMP.RWprotected, array);
	return true;
      }
      break;
    case F_LOGICAL_CONSTATNT:
    case F_CHARACTER_CONSTATNT:
    case F_COMPLEX_CONSTATNT:
    case STRING_CONSTANT:
    case INT_CONSTANT:
    case FLOAT_CONSTANT:
    case LONGLONG_CONSTANT:
    case MOE_CONSTANT:
      return false;
    default:
      XMP.errorAt(pb,"gmove must be followed by simple assignment");
    }
    return false;
  }

  private boolean isScalar(Xobject x){
    switch (x.Opcode()){
    case F_ARRAY_REF:
      XobjList subscripts = (XobjList)x.getArg(1);
      Xobject var = x.getArg(0).getArg(0);
      int n = var.Type().getNumDimensions();
      for (int i = 0; i < n; i++){
	Xobject sub = subscripts.getArg(i);
	if (sub.Opcode() == Xcode.F_INDEX_RANGE) return false;
      }
      break;
    case VAR:
      if (x.Type().getKind() == Xtype.F_ARRAY) return false;
      break;
    }
    return true;
  }

  private boolean convertGmoveToArray(PragmaBlock pb, Xobject left, Xobject right){

    // boolean lhs_is_scalar = isScalar(left);
    // boolean rhs_is_scalar = isScalar(right);

    // if (!lhs_is_scalar && rhs_is_scalar){

      pb.setPragma("ARRAY");

      Xobject onRef = Xcons.List();

      Xobject a = left.getArg(0).getArg(0);
      a.remProp(XMP.RWprotected);
      Ident id = env.findVarIdent(a.getName(), pb);
      assert id != null;
      XMParray array = XMParray.getArray(id);
      assert array != null;

      Xobject t = Xcons.Symbol(Xcode.VAR, array.getAlignTemplate().getName());
      onRef.add(t);

      Xobject subscripts = Xcons.List();
      for (int i = 0; i < array.getAlignTemplate().getDim(); i++){
	subscripts.add(null);
      }

      for (int i = 0; i < array.getDim(); i++){
	int alignSubscriptIndex = array.getAlignSubscriptIndexAt(i);
	Xobject alignSubscriptOffset = array.getAlignSubscriptOffsetAt(i);

	Xobject lb = left.getArg(1).getArg(i).getArg(0);
	if (alignSubscriptOffset != null) lb = Xcons.binaryOp(Xcode.PLUS_EXPR, lb, alignSubscriptOffset);
	Xobject ub = left.getArg(1).getArg(i).getArg(1);
	if (alignSubscriptOffset != null) ub = Xcons.binaryOp(Xcode.PLUS_EXPR, ub, alignSubscriptOffset);
	Xobject st = left.getArg(1).getArg(i).getArg(2);
	Xobject triplet = Xcons.List(lb, ub, st);

	subscripts.setArg(alignSubscriptIndex, triplet);
      }
      onRef.add(subscripts);

      Xobject decl = Xcons.List(onRef);
      pb.setClauses(decl);

    // }
    // else { // lhs_is_scalar || !rhs_is_scalar){
    //   return false;
    // }

    return true;

  }

  // private boolean isScalar(Xobject x){
  //   if (x.Opcode() == Xcode.F_ARRAY_REF){
  //     for (XobjArgs args = x.getArg(1).getArgs(); args != null;
  // 	   args = args.nextArgs()){
  // 	if (args.getArg().Opcode() != Xcode.F_ARRAY_INDEX) return false;
  //     }
  //   }
  //   return true;
  // }


  private Block analyzeArray(PragmaBlock pb){

    Xobject arrayDecl = pb.getClauses();
    BlockList body = pb.getBody();

    XMPinfo outer = null;
    for(Block bp = pb.getParentBlock(); bp != null; bp = bp.getParentBlock()) {
      if(bp.Opcode() == Xcode.XMP_PRAGMA) {
	outer = (XMPinfo)bp.getProp(XMP.prop);
      }
    }
    XMPinfo info = new XMPinfo(XMPpragma.ARRAY, outer, pb, env);
    pb.setProp(XMP.prop, info);

    Xobject onRef = arrayDecl.getArg(0);

    // check body is single statement.
    Block b = body.getHead();
    if (b.getNext() != null) XMP.fatal("not single block for Array");
    Statement s = b.getBasicBlock().getHead();
    if (b.Opcode() != Xcode.F_STATEMENT_LIST ||s.getNext() != null)
      XMP.fatal("not single statment for Array");
    Xobject x = s.getExpr();
    if (x.Opcode() != Xcode.F_ASSIGN_STATEMENT)
      XMP.fatal("not assignment for Array");

    XobjectIterator i = new topdownXobjectIterator(x);
    for (i.init(); !i.end(); i.next()) {
      Xobject y = i.getXobject();
      if (y == null) continue;
      if (y.Opcode() == Xcode.VAR && y.Type().isFarray() &&
    	  (i.getParent() == null || i.getParent().Opcode() != Xcode.F_VAR_REF)){
    	i.setXobject(Xcons.FarrayRef(y));
      }
    }

    Xobject left = x.left();
    Xobject right = x.right();

    // opcode must be VAR or ARRAY_REF
    boolean left_is_global = checkGmoveOperand(left, false, pb);
    //boolean right_is_global = checkGmoveOperand(right, pb);

    //if (!left_is_global && !right_is_global)
    //XMP.errorAt(pb, "local assignment for array");
    
    if (XMP.hasError()) return null;
    
    info.setGmoveOperands(left, right);
    info.setOnRef(XMPobjectsRef.parseDecl(onRef, env, pb));

    return convertArrayToLoop(pb, info);

  }

  private Block convertArrayToLoop(PragmaBlock pb, XMPinfo info){

    List<Ident> varList = new ArrayList<Ident>();
    List<Ident> varListTemplate = new ArrayList<Ident>(XMP.MAX_DIM);
    for (int i = 0; i < XMP.MAX_DIM; i++) varListTemplate.add(null);
    List<Xobject> lbList = new ArrayList<Xobject>();
    List<Xobject> ubList = new ArrayList<Xobject>();
    List<Xobject> stList = new ArrayList<Xobject>();

    //
    // convert LHS
    //

    Xobject left = info.getGmoveLeft();

    if (left.Opcode() != Xcode.F_ARRAY_REF) XMP.fatal(pb, "ARRAY not followed by array ref.");

    Xobject left_var = left.getArg(0).getArg(0);
    int n = left_var.Type().getNumDimensions();
    XobjList subscripts = (XobjList)left.getArg(1);
    Xobject[] sizeExprs = left_var.Type().getFarraySizeExpr();

    String name = left_var.getName();
    Ident id = env.findVarIdent(name, pb);
    XMParray leftArray =  XMParray.getArray(id);

    for (int i = 0; i < n; i++){

      Xobject sub = subscripts.getArg(i);

      Ident var;
      Xobject lb, ub, st;

      if (sub.Opcode() == Xcode.F_ARRAY_INDEX){
	continue;
      }
      else if (sub.Opcode() == Xcode.F_INDEX_RANGE){

	int tidx = leftArray.getAlignSubscriptIndexAt(i);
	if (tidx == -1) continue;

    	var = env.declIdent(XMP.genSym("XMP_loop_i"), Xtype.intType, pb);
    	varList.add(var);
	varListTemplate.set(tidx, var);

	lb = ((XobjList)sub).getArg(0);
	if (lb == null){
	  if (!left_var.Type().isFallocatable()){
	    lb = sizeExprs[i].getArg(0);
	  }
	  else {
	    lb = env.declIntrinsicIdent("lbound", Xtype.FintFunctionType).
	      Call(Xcons.List(left_var, Xcons.IntConstant(i+1)));
	  }
	}

	ub = ((XobjList)sub).getArg(1);
	if (ub == null){
	  if(!left_var.Type().isFallocatable()){
	    ub = sizeExprs[i].getArg(1);
	  }
	  else {
	    ub = env.declIntrinsicIdent("ubound", Xtype.FintFunctionType).
	      Call(Xcons.List(left_var, Xcons.IntConstant(i+1)));
	  }
	}

	st = ((XobjList)sub).getArg(2);
	if (st == null) st = Xcons.IntConstant(1);

    	lbList.add(lb);
    	ubList.add(ub);
    	stList.add(st);

	Xobject expr;
	expr = Xcons.binaryOp(Xcode.MUL_EXPR, var.Ref(), st);
	expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

    	subscripts.setArg(i, Xcons.FarrayIndex(expr));

      }

    }

    //
    // convert RHS
    //

    XobjectIterator j = new topdownXobjectIterator(info.getGmoveRight());
    for (j.init(); !j.end(); j.next()) {
      Xobject x = j.getXobject();

      if (x.Opcode() == Xcode.F_ARRAY_REF){
	int k = 0;
	Xobject x_var = x.getArg(0).getArg(0);
	XobjList subscripts1 = (XobjList)x.getArg(1);
	Xobject[] sizeExprs1 = x_var.Type().getFarraySizeExpr();

	String name1 = x_var.getName();
	Ident id1 = env.findVarIdent(name1, pb);
	XMParray array1 =  XMParray.getArray(id1);

	for (int i = 0; i < x_var.Type().getNumDimensions(); i++){

	  Xobject sub = subscripts1.getArg(i);
	  if (sub.Opcode() == Xcode.F_INDEX_RANGE){

	    //int tidx = array1.getAlignSubscriptIndexAt(i);

	    Xobject lb, st;

	    lb = ((XobjList)sub).getArg(0);
	    if (lb == null){
	      if (!x_var.Type().isFallocatable()){
		lb = sizeExprs1[i].getArg(0);
	      }
	      else {
		lb = env.declIntrinsicIdent("lbound", Xtype.FintFunctionType).
		  Call(Xcons.List(x_var, Xcons.IntConstant(i+1)));
	      }
	    }

	    st = ((XobjList)sub).getArg(2);
	    if (st == null) st = Xcons.IntConstant(1);

	    Xobject expr;
	    //expr = Xcons.binaryOp(Xcode.MUL_EXPR, varList.get(k).Ref(), st);

	    Ident loopVar;
	    if (array1 != null){
	      int tidx = array1.getAlignSubscriptIndexAt(i);
	      loopVar = varListTemplate.get(tidx);
	    }
	    else
	      loopVar = varList.get(k);

	    if (loopVar == null) XMP.fatal("array on rhs does not conform to that on lhs.");
	    expr = Xcons.binaryOp(Xcode.MUL_EXPR, loopVar.Ref(), st);
	    expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

	    subscripts1.setArg(i, Xcons.FarrayIndex(expr));
	    k++;
	  }

	}

      }
    }

    //
    // construct loop
    //

    BlockList loop = null;

    BlockList body = Bcons.emptyBody();
    body.add(Xcons.Set(info.getGmoveLeft(), info.getGmoveRight()));

    for (int i = 0; i < varList.size(); i++){
      loop = Bcons.emptyBody();
      Xobject ub = Xcons.binaryOp(Xcode.MINUS_EXPR, ubList.get(i), lbList.get(i));
      ub = Xcons.binaryOp(Xcode.PLUS_EXPR, ub, stList.get(i));
      ub = Xcons.binaryOp(Xcode.DIV_EXPR, ub, stList.get(i));
      ub = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, Xcons.IntConstant(1));
      loop.add(Bcons.Fdo(varList.get(i).Ref(),
       			 Xcons.List(Xcons.IntConstant(0), ub, Xcons.IntConstant(1)), body, null));
      body = loop;
    }

    //
    // convert ARRAY to LOOP directive
    //

    Xobject args = Xcons.List();

    XobjList loopIterList = Xcons.List();
    for (int i = 0; i < varList.size(); i++){
      loopIterList.add(varList.get(i).Ref());
    }
    args.add(loopIterList);

    Xobject onRef = Xcons.List();

    String templateName = pb.getClauses().getArg(0).getArg(0).getName();
    XMPtemplate template = env.findXMPtemplate(templateName, pb);
    if (template == null) XMP.errorAt(pb,"template '" + templateName + "' not found");

    onRef.add(pb.getClauses().getArg(0).getArg(0));
    Xobject subscriptList = Xcons.List();

    Xobject onSubscripts = pb.getClauses().getArg(0).getArg(1);

    if (onSubscripts != null && !onSubscripts.isEmptyList()){
      int k = 0;
      for (int i = 0; i < onSubscripts.Nargs(); i++){
    	Xobject sub = onSubscripts.getArg(i);
    	if (sub.Opcode() == Xcode.LIST){ // triplet

    	  Xobject lb = ((XobjList)sub).getArg(0);
    	  if (lb == null){
	    if (template.isFixed()){
	      lb = template.getLowerAt(i);
	    }
	    else {
	      Ident ret = env.declIdent(XMP.genSym("XMP_ret_"), Xtype.intType, pb);
	      Ident tlb = env.declIdent(XMP.genSym("XMP_" + template.getName() + "_lb"), Xtype.intType, pb);
	      
	      Ident f = env.declInternIdent("xmp_template_lbound", Xtype.FintFunctionType);
	      Xobject args1 = Xcons.List(template.getDescId().Ref(), Xcons.IntConstant(i+1), tlb.Ref());
	      pb.insert(Xcons.Set(ret.Ref(), f.Call(args1)));

	      lb = tlb.Ref();
	    }
	  }
	  else if (lb.Opcode() == Xcode.VAR){
	    String lbName = lb.getString();
	    Ident lbId = env.findVarIdent(lbName, pb);
	    if (lbId == null) {
	      XMP.errorAt(pb, "variable '" + lbName + "' is not declared");
	      return null;
	    }
	    lb = lbId.Ref();
	  }

	  Xobject st = ((XobjList)sub).getArg(2);
	  if (st != null){
	    if (st.Opcode() == Xcode.INT_CONSTANT && ((XobjInt)st).getInt() == 0){ // scalar
	      subscriptList.add(sub);
	      continue;
	    }
	    else if (st.Opcode() == Xcode.VAR){
	      String stName = st.getString();
	      Ident stId = env.findVarIdent(stName, pb);
	      if (stId == null) {
		XMP.errorAt(pb, "variable '" + stName + "' is not declared");
		return null;
	      }
	      st = stId.Ref();
	    }
	  }
	  else st = Xcons.IntConstant(1);
	  Xobject expr;
	  //expr = Xcons.binaryOp(Xcode.MUL_EXPR, varList.get(k).Ref(), st);
	  Ident loopVar = varListTemplate.get(i);
	  if (loopVar == null) XMP.fatal("template-ref does not conform to the array on lhs.");
	  expr = Xcons.binaryOp(Xcode.MUL_EXPR, loopVar.Ref(), st);
	  expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);
    	  subscriptList.add(expr);
	  k++;
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
	  Ident ret = env.declIdent(XMP.genSym("XMP_ret_"), Xtype.intType, pb);
	  Ident tlb = env.declIdent(XMP.genSym("XMP_" + template.getName() + "_lb"), Xtype.intType, pb);
	      
	  Ident f = env.declInternIdent("xmp_template_lbound", Xtype.FintFunctionType);
	  Xobject args1 = Xcons.List(template.getDescId().Ref(), Xcons.IntConstant(i+1), tlb.Ref());
	  pb.insert(Xcons.Set(ret.Ref(), f.Call(args1)));
	  
	  lb = tlb.Ref();
	}

	//Xobject expr = Xcons.binaryOp(Xcode.PLUS_EXPR, varList.get(i).Ref(), lb);
	Ident loopVar = varListTemplate.get(i);
	if (loopVar == null) XMP.fatal("template-ref does not conform to the array on lhs.");
	Xobject expr = Xcons.binaryOp(Xcode.PLUS_EXPR, loopVar.Ref(), lb);
    	subscriptList.add(expr);
      }
    }

    onRef.add(subscriptList);
    args.add(onRef);

    args.add(null);

    return Bcons.PRAGMA(Xcode.XMP_PRAGMA, "LOOP", args, loop);
  }

}
