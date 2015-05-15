package exc.openacc;

import exc.block.*;
import exc.object.*;

import java.util.ArrayList;
import java.util.List;

class AccUpdate extends AccData{
  private final List<Block> copyBlockList = new ArrayList<Block>();

  AccUpdate(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  boolean isAcceptableClause(ACCpragma clauseKind){
    switch (clauseKind){
    case IF:
    case ASYNC:
    case HOST:
    case DEVICE:
      return true;
    default:
      return false;
    }
  }

  int getCopyDirection(ACCvar var) throws ACCexception {
    if(var.copiesDtoH()){ //update host
      return ACC.DEVICE_TO_HOST;
    }else if(var.copiesHtoD()){ //update device
      return ACC.HOST_TO_DEVICE;
    }else{
      throw new ACCexception(var + " does not have update direction");
    }
  }

  Xobject getAsyncExpr(){
    boolean isAsync = _info.hasClause(ACCpragma.ASYNC);

    if(! isAsync){
      return Xcons.IntConstant(ACC.ACC_ASYNC_SYNC);
    }

    Xobject expr = _info.getIntExpr(ACCpragma.ASYNC);

    if(expr == null){
      return Xcons.IntConstant(ACC.ACC_ASYNC_NOVAL);
    }

    return expr;
  }

  @Override
  void generate() throws ACCexception{
    if(isDisabled()){
      return;
    }

    for(ACCvar var : _info.getACCvarList()){
      //TODO analyzeでチェックすること
      Ident hostDescId = var.getHostDesc();
      if(hostDescId == null){
        throw new ACCexception(var + " is not allocated in device memory");
      }

      Xobject asyncExpr = getAsyncExpr();

      int copyDirection = getCopyDirection(var);
      XobjList copyFuncArgs = Xcons.List(hostDescId.Ref(), Xcons.IntConstant(copyDirection), asyncExpr);
      Block copyFunc;
      if(var.isSubarray()){
        XobjList subarrayList = var.getSubscripts();
        XobjList lowerList = Xcons.List();
        XobjList lengthList = Xcons.List();
        for(Xobject x : subarrayList){
          XobjList rangeList = (XobjList)x;
          lowerList.add(rangeList.left());
          lengthList.add(rangeList.right());
        }
        String copyFuncName = ACC.COPY_SUBDATA_FUNC_NAME;
        copyFunc = ACCutil.createFuncCallBlockWithArrayRange(copyFuncName, copyFuncArgs, Xcons.List(lowerList, lengthList));
      }else{
        String copyFuncName = ACC.COPY_DATA_FUNC_NAME;
        copyFunc = ACCutil.createFuncCallBlock(copyFuncName, copyFuncArgs);
      }

      copyBlockList.add(copyFunc);
    }
  }

  @Override
  void rewrite() throws ACCexception{
    //build
    BlockList body = Bcons.emptyBody();
    for(Block b : copyBlockList) body.add(b);

    BlockList resultBody = Bcons.emptyBody();

    Xobject ifExpr = _info.getIntExpr(ACCpragma.IF);
    boolean isEnabled = (ifExpr == null || ifExpr.isIntConstant());
    if(isEnabled){
      resultBody.add(Bcons.COMPOUND(body));
    }else {
      Ident condId = resultBody.declLocalIdent("_ACC_DATA_IF_COND", Xtype.charType, StorageClass.AUTO, ifExpr);
      resultBody.add(Bcons.IF(condId.Ref(), Bcons.COMPOUND(body), null));
    }

    _pb.replace(Bcons.COMPOUND(resultBody));
  }
}
