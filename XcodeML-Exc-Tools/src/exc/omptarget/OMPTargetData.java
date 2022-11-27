/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import java.util.*;

import exc.block.*;
import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

// TargetData: convert to AccData
public class OMPTargetData extends AccData {
  //  contructor
  public OMPTargetData(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
    info.omp_pragma = OMPpragma.TARGET_DATA;
  }

  boolean isAcceptableClause(OMPpragma clauseKind) {
    switch (clauseKind) {
    case DIR_IF:
    case TARGET_DEVICE:
    case TARGET_DATA_MAP:
    case USE_DEVICE_PTR:
      return true;
    default:
      return false;
   }
  }
}
