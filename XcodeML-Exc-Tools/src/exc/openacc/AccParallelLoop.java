/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.PragmaBlock;

public class AccParallelLoop extends AccParallel{
  public final AccLoop loop;

  public AccParallelLoop(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
    loop = new AccLoop(decl, info, pb);
  }

  @Override
  void analyze() throws ACCexception{
    if(ACC.debug_flag) System.out.println("AccParallelLoop: analyze _info="+_info);

    // don't call loop.analyze(); instead ...
    loop.checkParallelism();
    loop.addInductionVariableAsPrivate();

    super.analyze(); // analyze for PARALLEL directive
  }

  boolean isAcceptableClause(ACCpragma clauseKind){
    return super.isAcceptableClause(clauseKind) || loop.isAcceptableClause(clauseKind);
  }
}
