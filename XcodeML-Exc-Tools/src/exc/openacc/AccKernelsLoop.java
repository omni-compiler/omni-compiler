package exc.openacc;

import exc.block.PragmaBlock;

class AccKernelsLoop extends AccKernels{
  private final AccLoop loop;
  AccKernelsLoop(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
    loop = new AccLoop(decl, info, pb);
  }

  @Override
  void analyze() throws ACCexception{
    //loop.analyze();
    loop.checkParallelism();
    loop.addInductionVariableAsPrivate();
    super.analyze();
  }

  boolean isAcceptableClause(ACCpragma clauseKind) {
    return super.isAcceptableClause(clauseKind) || loop.isAcceptableClause(clauseKind);
  }
}
