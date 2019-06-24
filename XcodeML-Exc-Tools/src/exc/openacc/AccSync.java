package exc.openacc;

import exc.block.Block;
import exc.block.PragmaBlock;
import exc.object.Xcons;
import exc.object.Xobject;

class AccSync extends AccDirective{

  private static final String ACC_SYNC_FUNC_NAME = "_ACC_sync";
  private static final String ACC_SYNC_ALL_FUNC_NAME = "_ACC_sync_all";
  private Block replaceBlock;

  AccSync(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  @Override
  void generate() throws ACCexception {
  }

  @Override
  void rewrite() throws ACCexception {
  }

  @Override
  boolean isAcceptableClause(ACCpragma clauseKind) {
    return clauseKind == ACCpragma.SYNC;
  }

  Block makeSyncBlock() throws ACCexception {
    Xobject syncExpr = _info.getIntExpr(ACCpragma.SYNC);
    if(syncExpr != null){ //sync(expr)
      return ACCutil.createFuncCallBlock(ACC_SYNC_FUNC_NAME, Xcons.List(syncExpr));
    }else{ //sync all
      return ACCutil.createFuncCallBlock(ACC_SYNC_ALL_FUNC_NAME, Xcons.List());
    }
  }
}
