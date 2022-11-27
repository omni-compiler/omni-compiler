/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import java.util.*;

import exc.block.*;
import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

//
// OMPTargetParallelLoop: convert to AccParallelLoop which generates kernel invocation & loop
//
public class OMPTargetParallelLoop extends AccParallelLoop {

  public OMPTargetParallelLoop(ACCglobalDecl decl, AccInformation info, PragmaBlock pb,
                        boolean has_teams, boolean has_distribute, boolean has_parallel, boolean has_simd) {
    super(decl, info, pb);
    info.omp_pragma = OMPpragma.valueOf(pb.getPragma());

    loop._info.omp_has_teams = has_teams;
    loop._info.omp_has_distribute = has_distribute;
    loop._info.omp_has_parallel = has_parallel;
    loop._info.omp_has_simd = has_simd;
  }

  boolean isAcceptableClause(OMPpragma clauseKind) {
    if(!OMPTarget.isTargetClause(clauseKind)) return false;
    if(loop._info.omp_has_teams && !OMPTarget.isTeamsClause(clauseKind)) return false;
    if(loop._info.omp_has_distribute && !OMPTarget.isDistributeClause(clauseKind)) return false;
    if(loop._info.omp_has_parallel && !OMPTarget.isParallelClause(clauseKind)) return false;
    if(loop._info.omp_has_simd && !OMPTarget.isSIMDClause(clauseKind)) return false;
    return true;
  }
}
