/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import java.util.*;

import exc.block.*;
import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

//
// OMPTargetParallel: convert to AccParallel which generates kernel invocation
//
public class OMPTargetParallel extends AccParallel {
  OMPTargetParallel(ACCglobalDecl decl, AccInformation info, PragmaBlock pb,
                    boolean has_teams, boolean has_parallel) {
    super(decl, info, pb);
    _info.omp_pragma = OMPpragma.valueOf(pb.getPragma());
    _info.omp_has_teams = has_teams;
    _info.omp_has_parallel = has_parallel;
  }

  boolean isAcceptableClause(OMPpragma clauseKind) {
    if(!OMPTarget.isTargetClause(clauseKind)) return false;
    if(_info.omp_has_teams && !OMPTarget.isTeamsClause(clauseKind)) return false;
    if(_info.omp_has_parallel && !OMPTarget.isParallelClause(clauseKind)) return false;
    return true;
  }
}
