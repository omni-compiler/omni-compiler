/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import java.util.*;

import exc.block.*;
import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

//
// OMPTargeLoop: convert to AccLoop which generates loop
//
public class OMPTargetLoop extends AccLoop {
  OMPTargetLoop(ACCglobalDecl decl, AccInformation info, PragmaBlock pb,
                boolean has_teams, boolean has_distribute, boolean has_parallel, boolean has_simd) {
    super(decl, info, pb);
    info.omp_pragma = OMPpragma.valueOf(pb.getPragma());

    _info.omp_has_teams = has_teams;
    _info.omp_has_distribute = has_distribute;
    _info.omp_has_parallel = has_parallel;
    _info.omp_has_simd = has_simd;
  }

  boolean isAcceptableClause(OMPpragma clauseKind) {
    if(_info.omp_has_teams && !OMPTarget.isTeamsClause(clauseKind)) return false;
    if(_info.omp_has_distribute && !OMPTarget.isDistributeClause(clauseKind)) return false;
    if(_info.omp_has_parallel && !OMPTarget.isParallelClause(clauseKind)) return false;
    if(_info.omp_has_simd && !OMPTarget.isSIMDClause(clauseKind)) return false;
    return true;
  }
}
