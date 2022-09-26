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
    System.out.println("AccParallelLoop: analyze _info="+_info);

    // don't call loop.analyze(); instead ...
    loop.checkParallelism();
    loop.addInductionVariableAsPrivate();

    super.analyze(); // analyze for PARALLEL directive
  }

  boolean isAcceptableClause(ACCpragma clauseKind){
    return super.isAcceptableClause(clauseKind) || loop.isAcceptableClause(clauseKind);
  }
}
