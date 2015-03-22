package exc.openacc;

import exc.block.*;

public class AccExitData extends AccData{
  AccExitData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  @Override
  void translate() throws ACCexception {
    ACC.debug("translate data");

    if(isDisabled()) return;

    for(ACCvar var : _info.getDeclarativeACCvarList()){
      if(var.getParent() != null) continue;

      String varName = var.getName();
      var.setHostDesc(declHostDesc(varName));
      var.setDevicePtr(declDevicePtr(varName));

      initBlockList.add(makeInitFuncCallBlock(var));

      int finalizeKind = 2;
      finalizeBlockList.add(makeFinalizeFuncCallBlock(var, finalizeKind));

      copyoutBlockList.add(makeCopyBlock(var, false));
    }
  }

  @Override
  String getInitFuncName(ACCvar var) {
    return ACC.FIND_DATA_FUNC_NAME;
  }

  public static boolean isAcceptableClause(ACCpragma clauseKind) {
    switch (clauseKind){
    case IF:
    case ASYNC:
    case WAIT:
    case COPYOUT:
    case DELETE:
      return true;
    default:
      return false;
    }
  }
}
