package exc.openacc;

import exc.block.*;

public class AccEnterData extends AccData{

  AccEnterData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  @Override
  void translate() throws ACCexception {
    ACC.debug("translate enter data");

    if(isDisabled()) return;

    for(ACCvar var : _info.getDeclarativeACCvarList()){
      if(var.getParent() != null) continue;

      String varName = var.getName();
      var.setHostDesc(declHostDesc(varName));
      var.setDevicePtr(declDevicePtr(varName));

      initBlockList.add(makeInitFuncCallBlock(var));

      int finalizeKind = 1;
      finalizeBlockList.add(makeFinalizeFuncCallBlock(var, finalizeKind));

      copyinBlockList.add(makeCopyBlock(var, true));
    }
  }

  public static boolean isAcceptableClause(ACCpragma clauseKind) {
    switch (clauseKind) {
    case IF:
    case ASYNC:
    case WAIT:
    case COPYIN:
    case CREATE:
    case PRESENT_OR_COPYIN:
    case PRESENT_OR_CREATE:
      return true;
    default:
      return false;
    }
  }
}
