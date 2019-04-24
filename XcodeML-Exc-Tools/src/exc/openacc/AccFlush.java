package exc.openacc;

import exc.block.Block;
import exc.block.PragmaBlock;
import exc.object.Xcons;
import exc.object.Xobject;

class AccFlush extends AccDirective{

  private static final String ACC_FLUSH_FUNC_NAME = "_ACC_flush";
  private static final String ACC_FLUSH_ALL_FUNC_NAME = "_ACC_flush_all";
  private Block replaceBlock;

  AccFlush(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
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
    return clauseKind == ACCpragma.FLUSH;
  }

  Block makeFlushBlock() throws ACCexception {
    Xobject flushExpr = _info.getIntExpr(ACCpragma.FLUSH);
    if(flushExpr != null){ //flush(expr)
      return ACCutil.createFuncCallBlock(ACC_FLUSH_FUNC_NAME, Xcons.List(flushExpr));
    }else{ //flush all
      return ACCutil.createFuncCallBlock(ACC_FLUSH_ALL_FUNC_NAME, Xcons.List());
    }
  }
}
