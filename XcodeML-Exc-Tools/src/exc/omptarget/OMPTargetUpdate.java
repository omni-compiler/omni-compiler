/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import java.util.*;

import exc.block.*;
import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

// TargetData: convert to AccData
public class OMPTargetUpdate extends AccUpdate {
  OMPTargetUpdate(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
    info.omp_pragma = OMPpragma.TARGET_UPDATE;
  }

  boolean isAcceptableClause(OMPpragma clauseKind) {
    switch (clauseKind) {
    case DIR_IF:
    case TARGET_DEVICE:
    case DEPEND:
    case DIR_NOWAIT:
    case TARGET_UPDATE_TO:
    case TARGET_UPDATE_FROM:
      return true;
    default:
      return false;
    }
  }
}
