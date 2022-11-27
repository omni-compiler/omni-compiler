/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.*;
import exc.object.StorageClass;

public class AccExitData extends AccData{
  public AccExitData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
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

      int finalizeKind = 2;
      finalizeBlockList.add(makeFinalizeFuncCallBlock(var, finalizeKind));

      copyoutBlockList.add(makeCopyBlock(var, false, getAsyncExpr()));
    }
  }

  @Override
  String getInitFuncName(ACCvar var) {
    return ACC.FIND_DATA_FUNC_NAME;
  }

  boolean isAcceptableClause(ACCpragma clauseKind) {
    switch (clauseKind){
    case IF:
    case ASYNC:
    case WAIT_CLAUSE:
    case WAIT:
    case COPYOUT:
    case DELETE:
      return true;
    default:
      return false;
    }
  }
}
