package exc.xcalablemp;

import java.util.Map;

import exc.block.*;
import exc.object.*;
import exc.openacc.ACCpragma;

public class XACCrewriteACCdata {
  private PragmaBlock pb;
  private XobjList clauses; 
  private XMPglobalDecl _globalDecl;
  private BlockList mainBody;
  private BlockList deviceLoopBody;
  private BlockList initDeviceLoopBody;
  private BlockList finalizeDeviceLoopBody;
  //private Map<String, XACCdeviceArray> layoutedArrayMap;
  private XMPsymbolTable localSymbolTable;
  private XMPdevice device = null;
  private XMPlayout layout = null;
  static final String XACC_DESC_PREFIX = "_XACC_DESC_";
  private Block replaceBlock;
  
  public XACCrewriteACCdata(XMPglobalDecl decl, PragmaBlock pb) {
    this.pb = pb;
    this.clauses = (XobjList)pb.getClauses();
    // TODO Auto-generated constructor stub
    this._globalDecl = decl;
    mainBody = Bcons.emptyBody();
    deviceLoopBody = Bcons.emptyBody();
    replaceBlock = Bcons.COMPOUND(mainBody);
    localSymbolTable = XMPlocalDecl.declXMPsymbolTable2(replaceBlock); //new XMPsymbolTable();
  }
  
  public Block makeReplaceBlock(){
    XACCtranslatePragma trans = new XACCtranslatePragma(_globalDecl);
    
    //XMPdevice device = null;
    //XMPlayout layout = null;
    XMPon on = null;
    if (clauses != null){
      device = trans.getXACCdevice((XobjList)clauses, pb);
       layout = trans.getXACClayout((XobjList)clauses);
      on = trans.getXACCon((XobjList)clauses, pb);
    }
    
    Ident fid = _globalDecl.declExternFunc("acc_set_device_num");

    if(device == null) return null;
    
    //base
    //Ident baseDeviceLoopVarId = mainBody.declLocalIdent("_XACC_device_" + device.getName(), Xtype.intType);
    //mainBody.add(Bcons.FORall(baseDeviceLoopVarId.Ref(), device.getLower(), device.getUpper(),
        //device.getStride(), Xcode.LOG_LE_EXPR, deviceLoopBody));
    //deviceLoopBody.add(fid.Call(Xcons.List(baseDeviceLoopVarId.Ref(), device.getDeviceRef())));
    //rewriteACCClauses(clauses, pb, newBody, baseDeviceLoopBody, baseDeviceLoopVarId, device, layout);

    BlockList pbBody  = null;

    deviceLoopBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, pb.getPragma(), clauses, pbBody));
    
    XobjList createArgs = Xcons.List();
    XobjList updateDeviceArgs = Xcons.List();
    XobjList updateHostArgs = Xcons.List();
    XobjList deleteArgs = Xcons.List();
    analyzeClause(createArgs, updateDeviceArgs, updateHostArgs, deleteArgs);
    Block beginDeviceLoop = makeBeginDeviceLoop(createArgs, updateDeviceArgs);
    Block endDeviceLoop = makeEndDeviceLoop(deleteArgs, updateHostArgs);
    
    add(beginDeviceLoop);
    add(Bcons.COMPOUND(pb.getBody()));
    add(endDeviceLoop);
    
    return replaceBlock; //Bcons.COMPOUND(mainBody);
  }
  
  public void analyzeClause(XobjList createArgs, XobjList updateDeviceArgs, XobjList updateHostArgs, XobjList deleteArgs){
    for(XobjArgs arg = clauses.getArgs(); arg != null; arg = arg.nextArgs()){
      Xobject clause = arg.getArg();
      if(clause == null) continue; 
      Xobject clauseArg = clause.right();
      String clauseName = clause.left().getName();
      ACCpragma pragma = ACCpragma.valueOf(clauseName);
      switch(pragma){
      case COPY:
        createArgs.mergeList((XobjList)clauseArg);
        updateDeviceArgs.mergeList((XobjList)clauseArg);
        updateHostArgs.mergeList((XobjList)clauseArg);
        deleteArgs.mergeList((XobjList)clauseArg);
        break;
      case CREATE:
        createArgs.mergeList((XobjList)clauseArg);
        deleteArgs.mergeList((XobjList)clauseArg);
        break;
      case COPYIN:
        createArgs.mergeList((XobjList)clauseArg);
        updateDeviceArgs.mergeList((XobjList)clauseArg);
        deleteArgs.mergeList((XobjList)clauseArg);
        break;
      case COPYOUT:
        createArgs.mergeList((XobjList)clauseArg);
        updateHostArgs.mergeList((XobjList)clauseArg);
        deleteArgs.mergeList((XobjList)clauseArg);
        break;
      default:
        continue;
      }
    }
  }
  
  private Block makeBeginDeviceLoop(XobjList createArgs, XobjList updateDeviceArgs)
  {
    Ident fid = _globalDecl.declExternFunc("acc_set_device_num");
    BlockList loopBody = Bcons.emptyBody();
    BlockList body = Bcons.emptyBody();
    
    Ident deviceLoopVarId = body.declLocalIdent("_XACC_device_" + device.getName(), Xtype.intType);
    body.add(Bcons.FORall(deviceLoopVarId.Ref(), device.getLower(), device.getUpper(),
        device.getStride(), Xcode.LOG_LE_EXPR, loopBody));
    loopBody.add(fid.Call(Xcons.List(deviceLoopVarId.Ref(), device.getDeviceRef())));

    rewriteXACCClause(ACCpragma.CREATE, createArgs, loopBody, deviceLoopVarId);
    rewriteXACCClause(ACCpragma.DEVICE, updateDeviceArgs, loopBody, deviceLoopVarId);
    
    loopBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.ENTER_DATA.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.CREATE.toString()), createArgs)), null));
    loopBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.UPDATE.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.DEVICE.toString()), updateDeviceArgs)), null));
    
    return Bcons.COMPOUND(body);
  }
  
  private Block makeEndDeviceLoop(XobjList deleteArgs, XobjList updateHostArgs)
  {
    Ident fid = _globalDecl.declExternFunc("acc_set_device_num");
    BlockList loopBody = Bcons.emptyBody();
    BlockList body = Bcons.emptyBody();
    
    Ident deviceLoopVarId = body.declLocalIdent("_XACC_device_" + device.getName(), Xtype.intType);
    body.add(Bcons.FORall(deviceLoopVarId.Ref(), device.getLower(), device.getUpper(),
        device.getStride(), Xcode.LOG_LE_EXPR, loopBody));
    loopBody.add(fid.Call(Xcons.List(deviceLoopVarId.Ref(), device.getDeviceRef())));

    rewriteXACCClause(ACCpragma.HOST, updateHostArgs, loopBody, deviceLoopVarId);
    rewriteXACCClause(ACCpragma.DELETE, deleteArgs, loopBody, deviceLoopVarId);
    
    loopBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.UPDATE.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.HOST.toString()), updateHostArgs)), null));
    loopBody.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.EXIT_DATA.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.DELETE.toString()), deleteArgs)), null));

    return Bcons.COMPOUND(body);
  }


  
  private void rewriteXACCClause(ACCpragma clause, XobjList clauseArgs, BlockList body, Ident deviceLoopVarId){
    switch(clause){
    case HOST:
    case DEVICE:
    case USE_DEVICE:
    case PRIVATE:
    case FIRSTPRIVATE:   
    case DEVICE_RESIDENT:
      break;
    default:
      if(!clause.isDataClause()) return;
    }
    
    for(XobjArgs args = clauseArgs.getArgs(); args != null; args = args.nextArgs()){
      Xobject item = args.getArg();
      if(item.Opcode() == Xcode.VAR){
        //item is variable or arrayAddr
        try{
          args.setArg(rewriteXACCArrayAddr(item, body, clause, deviceLoopVarId));
        }catch(XMPexception e){
          XMP.error(item.getLineNo(), e.getMessage());
        }
      }
    }
  }
  
  private void add(Block b)
  {
    mainBody.add(b);
  }
  
  private Xobject rewriteXACCArrayAddr(Xobject arrayAddr, BlockList body, ACCpragma clause, Ident deviceLoopCounterId) throws XMPexception{
    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayAddr.getSym(), pb);
    XMPcoarray coarray = _globalDecl.getXMPcoarray(arrayAddr.getSym(), pb);

    if (alignedArray == null && coarray == null) {
        return arrayAddr;
    }
    else if(alignedArray != null && coarray == null){ // only alignedArray
      if (alignedArray.checkRealloc() || (alignedArray.isLocal() && !alignedArray.isParameter()) ||
          alignedArray.isParameter()){
        Xobject arrayAddrRef = alignedArray.getAddrId().Ref();
        Ident descId = alignedArray.getDescId();

        //normal array
        if(device == null){
          String arraySizeName = "_ACC_size_" + arrayAddr.getSym();
          Ident arraySizeId = body.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);

          Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_get_array_total_elmts", Xcons.List(descId.Ref()));
          body.insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));

          XobjList arrayRef = Xcons.List(arrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
          return arrayRef;
        }
        


        //device array
        
        String arrayName = arrayAddr.getSym();
        XACCdeviceArray layoutedArray = null;
        layoutedArray = _globalDecl.getXACCdeviceArray(arrayName);
        if(layoutedArray == null && pb != null){
          layoutedArray = _globalDecl.getXACCdeviceArray(arrayName, pb);
        }
        if(layoutedArray == null){
          layoutedArray = localSymbolTable.getXACCdeviceArray(arrayName);
        }
        if(layoutedArray == null && layout != null){
          Ident layoutedArrayDescId = mainBody.declLocalIdent(XACC_DESC_PREFIX + arrayName, Xtype.voidPtrType);
          Block initDeviceArrayFuncCall = _globalDecl.createFuncCallBlock("_XACC_init_layouted_array", Xcons.List(layoutedArrayDescId.getAddr(), descId.Ref(), device.getDescId().Ref()));
          add(initDeviceArrayFuncCall);
          for(int dim = alignedArray.getDim() - 1; dim >= 0; dim--){
            int distManner = layout.getDistMannerAt(dim);
            String mannerStr = XMPlayout.getDistMannerString(distManner);
            Block splitDeviceArrayBlockCall = _globalDecl.createFuncCallBlock("_XACC_split_layouted_array_" + mannerStr, Xcons.List(layoutedArrayDescId.Ref(), Xcons.IntConstant(dim)));
            add(splitDeviceArrayBlockCall);
            if(! layout.hasShadow()) continue;
            int shadowType = layout.getShadowTypeAt(dim);
            if(shadowType != XMPlayout.SHADOW_NONE){
              String shadowTypeStr = XMPlayout.getShadowTypeString(shadowType);

              Block setShadowFuncCall = _globalDecl.createFuncCallBlock("_XACC_set_shadow_" + shadowTypeStr, 
                  Xcons.List(layoutedArrayDescId.Ref(), Xcons.IntConstant(dim), layout.getShadowLoAt(dim), layout.getShadowHiAt(dim)));
              add(setShadowFuncCall);
            }
          }
          Block calcDeviceArraySizeCall = _globalDecl.createFuncCallBlock("_XACC_calc_size", Xcons.List(layoutedArrayDescId.Ref()));
          add(calcDeviceArraySizeCall);

          layoutedArray = new XACCdeviceArray(layoutedArrayDescId, alignedArray, layout);
          if(pb != null){
            localSymbolTable.putXACCdeviceArray(layoutedArray);
          }else{            
            _globalDecl.putXACCdeviceArray(layoutedArray);
          }
        }

        Ident layoutedArrayDescId = layoutedArray.getDescId(); 

        Xobject arrayRef = null;
        switch(clause){
        case CREATE:
        case DELETE:
        {
          String arraySizeName = "_XACC_size_" + arrayAddr.getSym();
          String arrayOffsetName = "_XACC_offset_" + arrayAddr.getSym();
          Ident arraySizeId = body.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);
          Ident arrayOffsetId = body.declLocalIdent(arrayOffsetName, Xtype.unsignedlonglongType);
          Block getRangeFuncCall = _globalDecl.createFuncCallBlock("_XACC_get_size", Xcons.List(layoutedArrayDescId.Ref(), arrayOffsetId.getAddr(), arraySizeId.getAddr(), deviceLoopCounterId.Ref()));
          body.add(getRangeFuncCall);
          arrayRef = Xcons.List(arrayAddrRef, Xcons.List(arrayOffsetId.Ref(), arraySizeId.Ref()));
        } break;
        case HOST:
        case DEVICE:
        {
          String arrayCopySizeName = "_XACC_copy_size_" + arrayAddr.getSym();
          String arrayCopyOffsetName = "_XACC_copy_offset_" + arrayAddr.getSym();
          Ident arrayCopySizeId = body.declLocalIdent(arrayCopySizeName, Xtype.unsignedlonglongType);
          Ident arrayCopyOffsetId = body.declLocalIdent(arrayCopyOffsetName, Xtype.unsignedlonglongType);
          Block getCopyRangeFuncCall = _globalDecl.createFuncCallBlock("_XACC_get_copy_size", Xcons.List(layoutedArrayDescId.Ref(), arrayCopyOffsetId.getAddr(), arrayCopySizeId.getAddr(), deviceLoopCounterId.Ref()));
          body.add(getCopyRangeFuncCall);
          arrayRef = Xcons.List(arrayAddrRef, Xcons.List(arrayCopyOffsetId.Ref(), arrayCopySizeId.Ref()));
        } break;
        }
        arrayRef.setIsRewrittedByXmp(true);
        return arrayRef; 
      }else{
        return arrayAddr;
      }
    } else if(alignedArray == null && coarray != null){  // only coarray
      Xobject coarrayAddrRef = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + arrayAddr.getSym()).Ref();
      Ident descId = coarray.getDescId();
      
      String arraySizeName = "_ACC_size_" + arrayAddr.getSym();
      Ident arraySizeId = body.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);
      
      Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_get_array_total_elmts", Xcons.List(descId.Ref()));
      body.insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));
      
      XobjList arrayRef = Xcons.List(coarrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
      
      arrayRef.setIsRewrittedByXmp(true);
      return arrayRef;
    } else{ // no execute
      return arrayAddr;
    }
  }
}