package exc.openacc;

import exc.block.PragmaBlock;

public class AccParallelLoop extends AccParallel{
  private final AccLoop loop;
  AccParallelLoop(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
    loop = new AccLoop(decl, info, pb);
  }

  @Override
  void analyze() throws ACCexception{
    loop.analyze();
    super.analyze();
  }

  public static boolean isAcceptableClause(ACCpragma clauseKind){
    return AccParallel.isAcceptableClause(clauseKind) || AccLoop.isAcceptableClause(clauseKind);
  }
}
