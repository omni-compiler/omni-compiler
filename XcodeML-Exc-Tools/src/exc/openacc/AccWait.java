package exc.openacc;

import exc.block.Block;
import exc.block.PragmaBlock;
import exc.object.Xcons;
import exc.object.Xobject;

class AccWait extends AccDirective{

  private static final String ACC_GPU_WAIT_FUNC_NAME = "_ACC_gpu_wait";
  private static final String ACC_GPU_WAIT_ALL_FUNC_NAME = "_ACC_gpu_wait_all";
  private Block replaceBlock;

  AccWait(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  @Override
  void generate() throws ACCexception {
    Xobject waitExpr = _info.getIntExpr(ACCpragma.WAIT);
    if(waitExpr != null){ //wait(expr)
      replaceBlock = ACCutil.createFuncCallBlock(ACC_GPU_WAIT_FUNC_NAME, Xcons.List(waitExpr));
    }else{ //wait all
      replaceBlock = ACCutil.createFuncCallBlock(ACC_GPU_WAIT_ALL_FUNC_NAME, Xcons.List());
    }
  }

  @Override
  void rewrite() throws ACCexception {
    _pb.replace(replaceBlock);
  }

  @Override
  boolean isAcceptableClause(ACCpragma clauseKind) {
    return clauseKind == ACCpragma.WAIT;
  }
}
