package exc.openacc;

import exc.block.*;
import exc.object.*;
import java.util.*;

class AccData extends AccDirective {
  private final static String DEVICE_PTR_PREFIX = "_ACC_DEVICE_ADDR_";
  private final static String HOST_DESC_PREFIX = "_ACC_HOST_DESC_";
  final List<Block> initBlockList = new ArrayList<Block>();
  final List<Block> copyinBlockList = new ArrayList<Block>();
  final List<Block> copyoutBlockList = new ArrayList<Block>();
  final List<Block> finalizeBlockList = new ArrayList<Block>();
  final XobjList idList = Xcons.IDList();

  AccData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }
  AccData(ACCglobalDecl decl, AccInformation info) {
    super(decl, info);
  }

  boolean isAcceptableClause(ACCpragma clauseKind) {
    switch (clauseKind) {
    case IF:
      return true;
    default:
      return clauseKind.isDataClause();
    }
  }

  /*
  @Override
  void analyze() throws ACCexception {
    super.analyze();
    //シンボルのidを調べて存在するかチェック
    //setVarIdents();

    //TODO 親のpragmaで既に確保されているか調べる
  }
  */

  boolean isDisabled(){
    Xobject ifExpr = _info.getIntExpr(ACCpragma.IF);
    return ifExpr != null && ifExpr.isZeroConstant();
  }

  @Override
  void generate() throws ACCexception {
    if(isDisabled()) return;

    for(ACCvar var : _info.getDeclarativeACCvarList()){
      //FIXME
      {
        ACCvar redVar = _info.findReductionACCvar(var.getSymbol());
        if (redVar != null) {
          continue;
        }
      }

      if(var.getParent() != null) continue;

      if(var.isPrivate() || var.isFirstprivate()){
        continue;
      }

      String varName = var.getName();
      var.setHostDesc(declHostDesc(varName));
      var.setDevicePtr(declDevicePtr(varName));

      initBlockList.add(makeInitFuncCallBlock(var));

      int finalizeKind = 0;
      finalizeBlockList.add(makeFinalizeFuncCallBlock(var, finalizeKind));

      copyinBlockList.add(makeCopyBlock(var, true));
      copyoutBlockList.add(makeCopyBlock(var, false));
    }
  }

  Ident declDevicePtr(String varSymbol){
    return declVar(DEVICE_PTR_PREFIX + varSymbol);
  }

  Ident declHostDesc(String varSymbol) {
    return declVar(HOST_DESC_PREFIX + varSymbol);
  }

  private Ident declVar(String varSymbol){
    boolean isGlobal = _pb == null;
    VarScope varScope = isGlobal? VarScope.GLOBAL : VarScope.LOCAL;
    Ident id = Ident.Var(varSymbol, Xtype.voidPtrType, Xtype.Pointer(Xtype.voidPtrType), varScope);
    idList.add(id);
    return id;
  }

  Block makeInitFuncCallBlock(ACCvar var) throws ACCexception{
    Xobject addrObj = var.getAddress();
    Ident hostDescId = var.getHostDesc();
    Ident devicePtrId = var.getDevicePtr();
    Xtype elementType = var.getElementType();
    int dim = var.getDim();
    XobjList lowerList = Xcons.List();
    XobjList lengthList = Xcons.List();
    for(Xobject x : var.getSubscripts()){
      lowerList.add(x.left());
      lengthList.add(x.right());
    }
    XobjList initArgs = Xcons.List(hostDescId.getAddr(), devicePtrId.getAddr(), addrObj, Xcons.SizeOf(elementType), Xcons.IntConstant(dim));
    String initFuncName = getInitFuncName(var);

    return ACCutil.createFuncCallBlockWithArrayRange(initFuncName, initArgs, Xcons.List(lowerList, lengthList));
  }

  Block makeFinalizeFuncCallBlock(ACCvar var, int finalizeKind){
    Ident hostDescId = var.getHostDesc();
    return ACCutil.createFuncCallBlock(ACC.FINALIZE_DATA_FUNC_NAME, Xcons.List(hostDescId.Ref(), Xcons.IntConstant(finalizeKind)));
  }

  Block makeCopyBlock(ACCvar var, boolean isHostToDevice){
    boolean doCopy = (isHostToDevice)? var.copiesHtoD() : var.copiesDtoH();
    if(doCopy){
      String copyFuncName = getCopyFuncName(var);
      Ident hostDescId = var.getHostDesc();
      int direction = (isHostToDevice)? ACC.HOST_TO_DEVICE : ACC.DEVICE_TO_HOST;
      return ACCutil.createFuncCallBlock(copyFuncName,
              Xcons.List(hostDescId.Ref(), Xcons.IntConstant(direction), Xcons.IntConstant(ACC.ACC_ASYNC_SYNC)));
    }else{
      return Bcons.emptyBlock();
    }
  }

  private String getCopyFuncName(ACCvar var) {
    if(var.isPresentOr()){
      return ACC.PRESENT_OR_COPY_DATA_FUNC_NAME;
    }else{
      return ACC.COPY_DATA_FUNC_NAME;
    }
  }

  String getInitFuncName(ACCvar var) {
    String initFuncName;
    if (var.isPresent()) {
      initFuncName = ACC.FIND_DATA_FUNC_NAME;
    } else if (var.isPresentOr()) {
      initFuncName = ACC.PRESENT_OR_INIT_DATA_FUNC_NAME;
    } else {
      initFuncName = ACC.INIT_DATA_FUNC_NAME;
    }
    return initFuncName;
  }

  @Override
  void rewrite() throws ACCexception{
    //build
    BlockList beginBody = Bcons.emptyBody();
    for(Block b : initBlockList) beginBody.add(b);
    for(Block b : copyinBlockList) beginBody.add(b);
    BlockList endBody = Bcons.emptyBody();
    for(Block b : copyoutBlockList) endBody.add(b);
    for(Block b : finalizeBlockList) endBody.add(b);

    Block beginBlock = Bcons.COMPOUND(beginBody);
    Block endBlock = Bcons.COMPOUND(endBody);

    BlockList resultBody = Bcons.emptyBody();
    for(Xobject x: idList){
      resultBody.addIdent((Ident)x);
    }

    Xobject ifExpr = _info.getIntExpr(ACCpragma.IF);
    boolean isEnabled = (ifExpr == null || ifExpr.isIntConstant());
    if(isEnabled){
      resultBody.add(beginBlock);
      resultBody.add(Bcons.COMPOUND(_pb.getBody()));
      resultBody.add(endBlock);
    }else {
      Ident condId = resultBody.declLocalIdent("_ACC_DATA_IF_COND", Xtype.charType, StorageClass.AUTO, ifExpr);
      resultBody.add(Bcons.IF(condId.Ref(), beginBlock, null));
      resultBody.add(Bcons.COMPOUND(_pb.getBody()));
      resultBody.add(Bcons.IF(condId.Ref(), endBlock, null));
    }

    _pb.replace(Bcons.COMPOUND(resultBody));
  }
}
