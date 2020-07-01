package exc.openacc;

import exc.block.Block;
import exc.block.PragmaBlock;
import exc.object.Xcons;
import exc.object.Xobject;

class AccYield extends AccDirective{

  private static final String ACC_YIELD_FUNC_NAME = "_ACC_yield";
  private Block replaceBlock;

  AccYield(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
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
    return clauseKind == ACCpragma.YIELD;
  }

  Block makeYieldBlock() throws ACCexception {
    return ACCutil.createFuncCallBlock(ACC_YIELD_FUNC_NAME, Xcons.List());
  }
}
