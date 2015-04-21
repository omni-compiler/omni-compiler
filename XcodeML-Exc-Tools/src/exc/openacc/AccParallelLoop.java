package exc.openacc;

import exc.block.PragmaBlock;

class AccParallelLoop extends AccParallel{
  private final AccLoop loop;
  AccParallelLoop(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
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

  boolean isAcceptableClause(ACCpragma clauseKind){
    return super.isAcceptableClause(clauseKind) || loop.isAcceptableClause(clauseKind);
  }
}
