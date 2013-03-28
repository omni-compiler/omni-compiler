package exc.openacc;
import java.util.*;

import exc.block.*;
import exc.object.*;

public class ACCtranslateData {
  private PragmaBlock pb;
  private ACCinfo dataInfo;
  private List<Block> initBlockList;
  private List<Block> copyinBlockList;
  private List<Block> copyoutBlockList;
  private List<Block> finalizeBlockList;
//  private List<Ident> idList;
  private XobjList idList;
  private XobjList declList;
  
  private final static String GPU_DEVICE_PTR_PREFIX = "_ACC_GPU_DEVICE_ADDR_";
  private final static String GPU_HOST_DESC_PREFIX = "_ACC_GPU_HOST_DESC_";
  private final static String GPU_INIT_DATA_FUNC_NAME = "_ACC_gpu_init_data";
  private final static String GPU_FINALIZE_DATA_FUNC_NAME = "_ACC_gpu_finalize_data";
  private final static String GPU_COPY_DATA_FUNC_NAME = "_ACC_gpu_copy_data";
  
  private final static int GPU_COPY_HOST_TO_DEVICE = 400;
  private final static int GPU_COPY_DEVICE_TO_HOST = 401;
  
  ACCtranslateData(PragmaBlock pb){
    this.pb = pb;
    this.dataInfo = ACCutil.getACCinfo(pb);
    if(this.dataInfo == null){
      ACC.fatal("cannot get accinfo");
    }
    initBlockList = new ArrayList<Block>();
    copyinBlockList = new ArrayList<Block>();
    copyoutBlockList = new ArrayList<Block>();
    finalizeBlockList = new ArrayList<Block>();
//    idList = new ArrayList<Ident>();
    idList = Xcons.IDList();
    declList = Xcons.List();
  }
  
  public void translate() throws ACCexception{
    if(ACC.debugFlag){
      System.out.println("translate data");
    }
    
    if(dataInfo.isDisabled()) return;
        
    for(Iterator<ACCvar> iter = dataInfo.getVars(); iter.hasNext(); ){
      ACCvar var = iter.next();
      if(var.allocatesDeviceMemory()){
        String varName = var.getName();
        Ident varId = var.getId();
        if(var.isPresentOr()){
          if(dataInfo.getParent() != null){
            if(dataInfo.getParent().isVarAllocated(varName))continue;            
          }
        }
        boolean copyHtoD = var.copiesHtoD();
        boolean copyDtoH = var.copiesDtoH();
        
        //allocate device memory
        XobjList objAddrSize = getAddressAndSize(varName, varId, varId.Type());
        Xobject addrObj = objAddrSize.left();
        Xobject sizeObj = objAddrSize.right();
        
        Ident deviceAddr = Ident.Local(GPU_DEVICE_PTR_PREFIX + varName, Xtype.voidPtrType);
        Ident hostDesc = Ident.Local(GPU_HOST_DESC_PREFIX + varName, Xtype.voidPtrType);
        Block initBlock = createFuncCallBlock(GPU_INIT_DATA_FUNC_NAME, Xcons.List(hostDesc.getAddr(), deviceAddr.getAddr(), addrObj, sizeObj));
        Block finalizeBlock = createFuncCallBlock(GPU_FINALIZE_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref()));
        
        var.setDevicePtr(deviceAddr);
        var.setHostDesc(hostDesc);
        idList.add(deviceAddr);
        idList.add(hostDesc);        
        
        initBlockList.add(initBlock);
        finalizeBlockList.add(finalizeBlock);
        
        //copy data
        if(copyHtoD){
          Block copyHtoD_Func = createFuncCallBlock(GPU_COPY_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref(), Xcons.IntConstant(GPU_COPY_HOST_TO_DEVICE)));
          copyinBlockList.add(copyHtoD_Func);
        }
        if(copyDtoH){
          Block copyDtoH_Func = createFuncCallBlock(GPU_COPY_DATA_FUNC_NAME, Xcons.List(hostDesc.Ref(), Xcons.IntConstant(GPU_COPY_DEVICE_TO_HOST)));
          copyoutBlockList.add(copyDtoH_Func);
        }
      }
    }
    
//    List<Block> beginBlockList = new ArrayList<Block>();
//    List<Block> endBlockList = new ArrayList<Block>();
//    beginBlockList.addAll(initBlockList);
//    beginBlockList.addAll(copyinBlockList);
//    endBlockList.addAll(copyoutBlockList);
//    endBlockList.addAll(finalizeBlockList);
    
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

    
//    if(dataInfo.isEnabled()){
//      dataInfo.setBeginBlockList(beginBlockList);
//      dataInfo.setEndBlockList(endBlockList);
//      dataInfo.setIdList(idList);
//    }else{
//      //case like if(var)
//    }
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
