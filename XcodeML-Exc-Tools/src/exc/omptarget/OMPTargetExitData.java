/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import java.util.*;

import exc.block.*;
import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

// TargetData: convert to AccData
public class OMPTargetExitData extends AccExitData {
  public OMPTargetExitData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
    info.omp_pragma = OMPpragma.TARGET_EXIT_DATA;
  }

  boolean isAcceptableClause(OMPpragma clauseKind) {
    switch (clauseKind) {
    case DIR_IF:
    case TARGET_DEVICE:
    case TARGET_DATA_MAP:
    case DEPEND:
    case DIR_NOWAIT:
      return true;
    default:
      return false;
    }
  }
}
