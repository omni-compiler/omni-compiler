package exc.openacc;

import exc.block.PragmaBlock;

public class AccKernelsLoop extends AccKernels{
  private final AccLoop loop;
  AccKernelsLoop(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
    loop = new AccLoop(decl, info, pb);
  }

  @Override
  void analyze() throws ACCexception{
    loop.analyze();
    super.analyze();
  }

  public static boolean isAcceptableClause(ACCpragma clauseKind) {
    return AccKernels.isAcceptableClause(clauseKind) || AccLoop.isAcceptableClause(clauseKind);
  }
}
