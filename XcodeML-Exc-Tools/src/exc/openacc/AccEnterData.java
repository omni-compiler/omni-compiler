package exc.openacc;

import exc.block.*;
import exc.object.StorageClass;

class AccEnterData extends AccData{

  AccEnterData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  @Override
  void generate() throws ACCexception {
    if(isDisabled()) return;

    for(ACCvar var : _info.getDeclarativeACCvarList()){
      if(var.getParent() != null) continue;

      String varName = var.getName();
      StorageClass storageClass = var.getId().getStorageClass();
      var.setHostDesc(declHostDesc(varName, storageClass));
      var.setDevicePtr(declDevicePtr(varName, storageClass));

      initBlockList.add(makeInitFuncCallBlock(var));

      int finalizeKind = 1;
      finalizeBlockList.add(makeFinalizeFuncCallBlock(var, finalizeKind));

      copyinBlockList.add(makeCopyBlock(var, true, getAsyncExpr()));
    }
  }

  boolean isAcceptableClause(ACCpragma clauseKind) {
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
