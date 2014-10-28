package exc.openacc;
import java.util.*;

import exc.block.*;
import exc.object.*;

public class ACCtranslateData {
  //private PragmaBlock pb;
  private ACCinfo dataInfo;
  private List<Block> initBlockList;
  private List<Block> copyinBlockList;
  private List<Block> copyoutBlockList;
  private List<Block> finalizeBlockList;
//  private List<Ident> idList;
  private XobjList idList;
  private XobjList declList;
  private boolean _isGlobal;
  
  private final static String GPU_DEVICE_PTR_PREFIX = "_ACC_GPU_DEVICE_ADDR_";
  private final static String GPU_HOST_DESC_PREFIX = "_ACC_GPU_HOST_DESC_";
  
  
  ACCtranslateData(PragmaBlock pb){
    init(pb);
    _isGlobal = false;
  }
  
  ACCtranslateData(Xobject px){
    init(px);
    _isGlobal = true;
  }
  
  private void init(PropObject po){
    dataInfo = ACCutil.getACCinfo(po);
    if(dataInfo == null){
      ACC.fatal("cannot get accinfo");
    }
    initBlockList = new ArrayList<Block>();
    copyinBlockList = new ArrayList<Block>();
    copyoutBlockList = new ArrayList<Block>();
    finalizeBlockList = new ArrayList<Block>();
    idList = Xcons.IDList();
    declList = Xcons.List();
  }
  
  public void translate() throws ACCexception{
    ACC.debug("translate data");
    
    if(dataInfo.isDisabled()) return;

    VarScope varScope = _isGlobal? VarScope.GLOBAL : VarScope.LOCAL;
    
    for(Iterator<ACCvar> iter = dataInfo.getVars(); iter.hasNext(); ){
      ACCvar var = iter.next();
      Ident varId = var.getId();
      
      if(! var.allocatesDeviceMemory()){
    	  if(var.isFirstprivate() || var.isPrivate() || var.isReduction()) continue;
    	  
    	  Ident hostDescId = dataInfo.getHostDescId(var.getName());
    	  if(hostDescId != null) continue;
      }
      
      if(dataInfo.getParent() != null && dataInfo.getParent().isVarAllocated(varId)){
        continue;
      }

      
      String varName = var.getName();
      Xobject addrObj = var.getAddress();
      Ident deviceAddr = Ident.Var(GPU_DEVICE_PTR_PREFIX + varName, Xtype.voidPtrType, Xtype.Pointer(Xtype.voidPtrType), varScope);
      Ident hostDesc = Ident.Var(GPU_HOST_DESC_PREFIX + varName, Xtype.voidPtrType, Xtype.Pointer(Xtype.voidPtrType), varScope);
      var.setDevicePtr(deviceAddr);
      var.setHostDesc(hostDesc);
      idList.add(deviceAddr);
      idList.add(hostDesc);
      
      Block initializeBlock = Bcons.emptyBlock();
      Block finalizeBlock = Bcons.emptyBlock();

      //setup array dim
      Xtype varType = var.getId().Type();
      Xtype elementType = var.getElementType();
      int dim = var.getDim();

      XobjList suffixArgs = Xcons.List();
      {
	  for(Xobject x : var.getSubscripts()){
	      suffixArgs.add(x.left());
	      suffixArgs.add(x.right());
	  }
      }

      XobjList lowerList = Xcons.List();
      XobjList lengthList = Xcons.List();
      BlockList body = Bcons.emptyBody();
      for(Xobject x : var.getSubscripts()){
        lowerList.add(x.left());
        lengthList.add(x.right());
      }

      XobjList initArgs = Xcons.List(hostDesc.getAddr(), deviceAddr.getAddr(), addrObj, Xcons.SizeOf(elementType), Xcons.IntConstant(dim));
      //initArgs.mergeList(suffixArgs);
      String initFuncName;
      if(dataInfo.pragma != ACCpragma.EXIT_DATA){
        if(var.isPresent()){
          initFuncName = ACC.FIND_DATA_FUNC_NAME;
        }else if(var.isPresentOr()){
          initFuncName = ACC.PRESENT_OR_INIT_DATA_FUNC_NAME;
        }else{
          initFuncName = ACC.INIT_DATA_FUNC_NAME;
        }
      }else{
        initFuncName = ACC.FIND_DATA_FUNC_NAME;
      }
      initializeBlock = ACCutil.createFuncCallBlockWithArrayRange(initFuncName, initArgs, Xcons.List(lowerList, lengthList));
      initBlockList.add(initializeBlock);

      //if(dataInfo.pragma != ACCpragma.ENTER_DATA){    
        int finalizeKind = 0;
        if(dataInfo.pragma == ACCpragma.ENTER_DATA) finalizeKind = 1;
        else if(dataInfo.pragma == ACCpragma.EXIT_DATA)finalizeKind = 2;
        finalizeBlock = createFuncCallBlock(ACC.FINALIZE_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref(), Xcons.IntConstant(finalizeKind)));
        finalizeBlockList.add(finalizeBlock);
      //}
      
      //copy data
      Block copyHostToDeviceFunc = Bcons.emptyBlock();
      Block copyDeviceToHostFunc = Bcons.emptyBlock();
      boolean copyHtoD = var.copiesHtoD();
      boolean copyDtoH = var.copiesDtoH();
      if(var.isPresentOr()){
        if(copyHtoD){
	    copyHostToDeviceFunc = createFuncCallBlock(ACC.PRESENT_OR_COPY_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref(), Xcons.IntConstant(ACC.HOST_TO_DEVICE), Xcons.IntConstant(ACC.ACC_ASYNC_SYNC)));
        }
        if(copyDtoH){
	  copyDeviceToHostFunc = createFuncCallBlock(ACC.PRESENT_OR_COPY_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref(), Xcons.IntConstant(ACC.DEVICE_TO_HOST), Xcons.IntConstant(ACC.ACC_ASYNC_SYNC)));
        }
      }else if(var.allocatesDeviceMemory()){
        if(copyHtoD){
	    copyHostToDeviceFunc = createFuncCallBlock(ACC.COPY_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref(), Xcons.IntConstant(ACC.HOST_TO_DEVICE), Xcons.IntConstant(ACC.ACC_ASYNC_SYNC)));
        }
        if(copyDtoH){
	    copyDeviceToHostFunc = createFuncCallBlock(ACC.COPY_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref(), Xcons.IntConstant(ACC.DEVICE_TO_HOST), Xcons.IntConstant(ACC.ACC_ASYNC_SYNC)));
        }
      }
      copyinBlockList.add(copyHostToDeviceFunc);
      copyoutBlockList.add(copyDeviceToHostFunc);
    }
    
    //build
    BlockList beginBody = Bcons.emptyBody();
    for(Block b : initBlockList) beginBody.add(b);
    for(Block b : copyinBlockList) beginBody.add(b);
    BlockList endBody = Bcons.emptyBody();
    for(Block b : copyoutBlockList) endBody.add(b);
    for(Block b : finalizeBlockList) endBody.add(b);
    
    Block beginBlock = Bcons.COMPOUND(beginBody);
    Block endBlock = Bcons.COMPOUND(endBody);
    
    if(dataInfo.isEnabled()){
      dataInfo.setBeginBlock(beginBlock);
      dataInfo.setEndBlock(endBlock);
    }else{
      Ident condId = Ident.Local("_ACC_DATA_IF_COND", Xtype.intType);
      condId.setIsDeclared(true);
      Xobject condDecl = Xcons.List(Xcode.VAR_DECL, condId.Ref(), dataInfo.getIfCond());
      dataInfo.setBeginBlock(Bcons.IF(condId.Ref(), beginBlock, null));
      dataInfo.setEndBlock(Bcons.IF(condId.Ref(), endBlock, null));
      dataInfo.setDeclList(Xcons.List(condDecl));
      idList.add(condId);
    }
    dataInfo.setIdList(idList);
  }
  
  public Block createFuncCallBlock(String funcName, XobjList funcArgs) {
    Ident funcId = ACC.getMacroId(funcName, Xtype.voidType);
    return Bcons.Statement(funcId.Call(funcArgs));
  }

  private XobjList getAddressAndSize(String varName, Ident varId, Xtype varType) throws ACCexception{    
    Xobject addrObj = null;
    Xobject sizeObj = null;
    
    switch (varType.getKind()) {
    case Xtype.BASIC:
    case Xtype.STRUCT:
    case Xtype.UNION:
      addrObj = varId.getAddr();
      sizeObj = Xcons.SizeOf(varType);
      break;
    case Xtype.ARRAY:
    {
      ArrayType arrayVarType = (ArrayType)varType;
      switch (arrayVarType.getArrayElementType().getKind()) {
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.UNION:
        break;
      default:
        throw new ACCexception("array '" + varName + "' has has a wrong data type for acc data");
      }

      addrObj = varId.Ref();
      sizeObj = Xcons.binaryOp(Xcode.MUL_EXPR, 
          //Xcons.LongLongConstant(0, ACCutil.getArrayElmtCount(arrayVarType)),
          ACCutil.getArrayElmtCountObj(arrayVarType),
          Xcons.SizeOf(((ArrayType)varType).getArrayElementType()));
      break;
    }
    default:
      throw new ACCexception("'" + varName + "' has a wrong data type for acc data");
    }
    return Xcons.List(addrObj, sizeObj);
  }

}
