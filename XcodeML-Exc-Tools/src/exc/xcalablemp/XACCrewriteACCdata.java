package exc.xcalablemp;

import java.util.*;

import exc.block.*;
import exc.object.*;
import exc.openacc.ACCpragma;

public class XACCrewriteACCdata {
  protected PragmaBlock pb;
  protected XobjList clauses; 
  protected XMPglobalDecl _globalDecl;
  protected BlockList mainBody;
  private BlockList deviceLoopBody;
  private BlockList initDeviceLoopBody;
  private BlockList finalizeDeviceLoopBody;
 // protected DeviceLoop initDeviceLoop;
  //protected DeviceLoop mainDeviceLoop;
  //protected DeviceLoop finalizeDeviceLoop;
  
  //private Map<String, XACCdeviceArray> layoutedArrayMap;
  private XMPsymbolTable localSymbolTable;
  protected XACCdevice device = null;
  protected XACClayout layout = null;
  protected XACClayoutRef on = null;
  static final String XACC_DESC_PREFIX = "_XACC_DESC_";
  protected Block replaceBlock;
  protected Boolean isGlobal = false;
  
  public XACCrewriteACCdata()
  {
    
  }
  
  public XACCrewriteACCdata(XMPglobalDecl decl, PragmaBlock pb) {
    this.pb = pb;
    this.clauses = (XobjList)pb.getClauses();
    this._globalDecl = decl;
    mainBody = Bcons.emptyBody();
    deviceLoopBody = Bcons.emptyBody();
    replaceBlock = Bcons.COMPOUND(mainBody);
    localSymbolTable = XMPlocalDecl.declXMPsymbolTable2(replaceBlock); //new XMPsymbolTable();

    XACCtranslatePragma trans = new XACCtranslatePragma(_globalDecl);
    if (clauses != null){
      device = trans.getXACCdevice((XobjList)clauses, pb);
      layout = trans.getXACClayout((XobjList)clauses);
      on = trans.getXACClayoutRef((XobjList)clauses, pb);//trans.getXACCon((XobjList)clauses, pb);
    }
  }
  
  public Block makeReplaceBlock(){
    if(device == null) return null;

    
    XobjList createArgs = Xcons.List();
    XobjList updateDeviceArgs = Xcons.List();
    XobjList updateHostArgs = Xcons.List();
    XobjList deleteArgs = Xcons.List();
    analyzeClause(createArgs, updateDeviceArgs, updateHostArgs, deleteArgs);
    Block beginDeviceLoopBlock = makeBeginDeviceLoop(createArgs, updateDeviceArgs);
    Block endDeviceLoopBlock = makeEndDeviceLoop(deleteArgs, updateHostArgs);
    
    add(beginDeviceLoopBlock);
    add(Bcons.COMPOUND(pb.getBody()));
    add(endDeviceLoopBlock);
    
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
      case HOST:
        updateHostArgs.mergeList((XobjList)clauseArg);
        break;
      case DEVICE:
        updateDeviceArgs.mergeList((XobjList)clauseArg);
        break;
      default:
        continue;
      }
    }
  }
  
  protected Block makeBeginDeviceLoop(XobjList createArgs, XobjList updateDeviceArgs)
  {
    
    DeviceLoop deviceLoop = new DeviceLoop(device);

    if(! createArgs.isEmpty()){
      rewriteXACCClause(ACCpragma.CREATE, createArgs, deviceLoop);
      deviceLoop.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.ENTER_DATA.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.CREATE.toString()), createArgs)), null));
    }
    if(! updateDeviceArgs.isEmpty()){
      rewriteXACCClause(ACCpragma.DEVICE, updateDeviceArgs, deviceLoop);
      deviceLoop.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.UPDATE.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.DEVICE.toString()), updateDeviceArgs)), null));
    }
    
    return deviceLoop.makeBlock();
  }
  
  protected Block makeEndDeviceLoop(XobjList deleteArgs, XobjList updateHostArgs)
  {
    DeviceLoop deviceLoop = new DeviceLoop(device);
    
    if(! updateHostArgs.isEmpty()){
      rewriteXACCClause(ACCpragma.HOST, updateHostArgs, deviceLoop);
      deviceLoop.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.UPDATE.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.HOST.toString()), updateHostArgs)), null));
    }
    if(! deleteArgs.isEmpty()){
      rewriteXACCClause(ACCpragma.DELETE, deleteArgs, deviceLoop);
      deviceLoop.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.EXIT_DATA.toString(), Xcons.List(Xcons.List(Xcons.String(ACCpragma.DELETE.toString()), deleteArgs)), null));
    }
    
    return deviceLoop.makeBlock();
  }


  
  protected void rewriteXACCClause(ACCpragma clause, XobjList clauseArgs, DeviceLoop deviceLoop){
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
          args.setArg(rewriteXACCArrayAddr(item, clause, deviceLoop));
        }catch(XMPexception e){
          XMP.error(item.getLineNo(), e.getMessage());
        }
      }
    }
  }
  
  protected void add(Block b)
  {
    mainBody.add(b);
  }
  
  private Xobject rewriteXACCArrayAddr(Xobject arrayAddr, ACCpragma clause, DeviceLoop deviceLoop) throws XMPexception{
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
          Ident arraySizeId = deviceLoop.getBody().declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);

          Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_get_array_total_elmts", Xcons.List(descId.Ref()));
          deviceLoop.getBody().insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));

          XobjList arrayRef = Xcons.List(arrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
          return arrayRef;
        }
        


        //device array
        
        String arrayName = arrayAddr.getSym();
        XACClayoutedArray layoutedArray = null;
//        layoutedArray = _globalDecl.getXACCdeviceArray(arrayName);
//        if(layoutedArray == null && pb != null){
//          layoutedArray = _globalDecl.getXACCdeviceArray(arrayName, pb);
//        }
//        if(layoutedArray == null){
//          layoutedArray = localSymbolTable.getXACCdeviceArray(arrayName);
//        }
        layoutedArray = getXACClayoutedArray(arrayName);
        if(layoutedArray == null && layout != null){
          Ident layoutedArrayDescId = declIdent(XACC_DESC_PREFIX + arrayName, Xtype.voidPtrType);
          Block initDeviceArrayFuncCall = _globalDecl.createFuncCallBlock("_XACC_init_layouted_array", Xcons.List(layoutedArrayDescId.getAddr(), descId.Ref(), device.getDescId().Ref()));
          add(initDeviceArrayFuncCall);
          for(int dim = alignedArray.getDim() - 1; dim >= 0; dim--){
            int distManner = layout.getDistMannerAt(dim);
            String mannerStr = XACClayout.getDistMannerString(distManner);
            Block splitDeviceArrayBlockCall = _globalDecl.createFuncCallBlock("_XACC_split_layouted_array_" + mannerStr, Xcons.List(layoutedArrayDescId.Ref(), Xcons.IntConstant(dim)));
            add(splitDeviceArrayBlockCall);
            if(! layout.hasShadow()) continue;
            int shadowType = layout.getShadowTypeAt(dim);
            if(shadowType != XACClayout.SHADOW_NONE){
              String shadowTypeStr = XACClayout.getShadowTypeString(shadowType);

              Block setShadowFuncCall = _globalDecl.createFuncCallBlock("_XACC_set_shadow_" + shadowTypeStr, 
                  Xcons.List(layoutedArrayDescId.Ref(), Xcons.IntConstant(dim), layout.getShadowLoAt(dim), layout.getShadowHiAt(dim)));
              add(setShadowFuncCall);
            }
          }
          Block calcDeviceArraySizeCall = _globalDecl.createFuncCallBlock("_XACC_calc_size", Xcons.List(layoutedArrayDescId.Ref()));
          add(calcDeviceArraySizeCall);

          layoutedArray = new XACClayoutedArray(layoutedArrayDescId, alignedArray, layout);
//          if(pb != null){
//            localSymbolTable.putXACCdeviceArray(layoutedArray);
//          }else{            
//            _globalDecl.putXACCdeviceArray(layoutedArray);
//          }
          putXACClayoutedArray(layoutedArray);
        }

        Ident layoutedArrayDescId = layoutedArray.getDescId(); 

        Xobject arrayRef = null;
        switch(clause){
        case CREATE:
        case DELETE:
        case PRESENT:
        {
          String arraySizeName = "_XACC_size_" + arrayAddr.getSym();
          String arrayOffsetName = "_XACC_offset_" + arrayAddr.getSym();
          Ident arraySizeId = deviceLoop.getBody().declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);
          Ident arrayOffsetId = deviceLoop.getBody().declLocalIdent(arrayOffsetName, Xtype.unsignedlonglongType);
          Block getRangeFuncCall = _globalDecl.createFuncCallBlock("_XACC_get_size", Xcons.List(layoutedArrayDescId.Ref(), arrayOffsetId.getAddr(), arraySizeId.getAddr(), deviceLoop.getLoopVarId().Ref()));
          deviceLoop.add(getRangeFuncCall);
          arrayRef = Xcons.List(arrayAddrRef, Xcons.List(arrayOffsetId.Ref(), arraySizeId.Ref()));
          
          if(clause == ACCpragma.CREATE){
            Block setDevicePtrFuncCall = _globalDecl.createFuncCallBlock("_XACC_set_deviceptr", Xcons.List(layoutedArrayDescId.Ref(), arrayAddrRef, deviceLoop.getLoopVarId().Ref()));
            deviceLoop.addToEnd(Bcons.PRAGMA(Xcode.ACC_PRAGMA, ACCpragma.HOST_DATA.toString(), Xcons.List(Xcons.List(Xcons.String("USE_DEVICE"), Xcons.List(Xcons.List(arrayAddrRef, Xcons.List(arrayOffsetId.Ref(),arraySizeId.Ref()))))), Bcons.blockList(setDevicePtrFuncCall)));
          }
        } break;
        case HOST:
        case DEVICE:
        {
          String arrayCopySizeName = "_XACC_copy_size_" + arrayAddr.getSym();
          String arrayCopyOffsetName = "_XACC_copy_offset_" + arrayAddr.getSym();
          Ident arrayCopySizeId = deviceLoop.getBody().declLocalIdent(arrayCopySizeName, Xtype.unsignedlonglongType);
          Ident arrayCopyOffsetId = deviceLoop.getBody().declLocalIdent(arrayCopyOffsetName, Xtype.unsignedlonglongType);
          Block getCopyRangeFuncCall = _globalDecl.createFuncCallBlock("_XACC_get_copy_size", Xcons.List(layoutedArrayDescId.Ref(), arrayCopyOffsetId.getAddr(), arrayCopySizeId.getAddr(), deviceLoop.getLoopVarId().Ref()));
          deviceLoop.add(getCopyRangeFuncCall);
          arrayRef = Xcons.List(arrayAddrRef, Xcons.List(arrayCopyOffsetId.Ref(), arrayCopySizeId.Ref()));
        } break;
        default:
          throw new XMPexception("unimplemented for "+clause.getName());
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
      Ident arraySizeId = deviceLoop.getBody().declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);
      
      Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_get_array_total_elmts", Xcons.List(descId.Ref()));
      deviceLoop.getBody().insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));
      
      XobjList arrayRef = Xcons.List(coarrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
      
      arrayRef.setIsRewrittedByXmp(true);
      return arrayRef;
    } else{ // no execute
      return arrayAddr;
    }
    
  }
  
  protected XACClayoutedArray getXACClayoutedArray(String arrayName)
  {
    XACClayoutedArray layoutedArray = null;
    if(! isGlobal){
      layoutedArray = localSymbolTable.getXACCdeviceArray(arrayName);
      if(layoutedArray == null){
        layoutedArray = _globalDecl.getXACCdeviceArray(arrayName, pb);
      }
    }
    
    if(layoutedArray == null){
      layoutedArray = _globalDecl.getXACCdeviceArray(arrayName);
    }
    return layoutedArray;
  }
  
  protected void putXACClayoutedArray(XACClayoutedArray layoutedArray)
  {
    if(isGlobal){
      _globalDecl.putXACCdeviceArray(layoutedArray);
    }else{            
      localSymbolTable.putXACCdeviceArray(layoutedArray);
    }
  }
  
  protected Ident declIdent(String name, Xtype t){
    if(isGlobal){
      return _globalDecl.declGlobalIdent(name, t);
    }else{
      return mainBody.declLocalIdent(name, t);
    }
  }
  
  
  class DeviceLoop
  {
    private BlockList loopBody;
    private BlockList body;
    private Ident loopVarId;
    private XACCdevice dev;
    private List<Block> beginBlocks;
    private List<Block> endBlocks;

    public DeviceLoop(XACCdevice d){
      this.dev = d;
      loopBody = Bcons.emptyBody();
      body = Bcons.emptyBody();
      loopVarId = body.declLocalIdent("_XACC_device_" + dev.getName(), Xtype.intType);
      beginBlocks = new ArrayList<Block>();
      endBlocks = new ArrayList<Block>();
    }
    
    public Ident getLoopVarId() {
      return loopVarId;
    }

    public Block makeBlock(){
      if(loopBody.isEmpty()) return Bcons.emptyBlock();
      body.add(Bcons.FORall(loopVarId.Ref(), dev.getLower(), dev.getUpper(),
          dev.getStride(), Xcode.LOG_LE_EXPR, loopBody));
      Ident fid = _globalDecl.declExternFunc("acc_set_device_num");
      for(Block b : beginBlocks) loopBody.insert(b);
      for(Block b : endBlocks) loopBody.add(b);
      loopBody.insert(fid.Call(Xcons.List(loopVarId.Ref(), dev.getDeviceRef())));
      return Bcons.COMPOUND(body);
    }
    
    public void add(Block b){
      loopBody.add(b);
    }
    public void addToBegin(Block b){
      beginBlocks.add(b);
    }
    public void addToEnd(Block b){
      endBlocks.add(b);
    }
    public BlockList getBody()
    {
      return loopBody;
    }
    public Ident declLocalIdent(String name, Xtype t){
      return loopBody.declLocalIdent(name, t);
    }
  }
}