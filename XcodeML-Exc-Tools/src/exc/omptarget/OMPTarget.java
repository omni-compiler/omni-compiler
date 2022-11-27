/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import exc.object.*;
import xcodeml.util.XmLog;
import java.util.*;

import exc.openmp.OMPpragma;
import exc.openmp.OMP;

/* static class for this pacakge */
public class OMPTarget {
  private static boolean errorFlag = false;
  public static boolean debug_flag = false;

  public static void exitByError() {
    if (errorFlag) System.exit(1);
  }

  public static void error(LineNo l, String msg) {
    errorFlag = true;
    XmLog.error(l, "[OpenMP target] " + msg);
  }

  public static void warning(String msg) {
    XmLog.warning("[OpenMP target] " + msg);
  }

  public static void warning(LineNo l, String msg){
    XmLog.warning(l, "[OpenMP target] " + msg);
  }

  public static void fatal(String msg) {
    XmLog.fatal("[OpenMP target] " + msg);
  }
  
  public static void debug(String msg) {
    // XmLog.debug("[OpenMP target] " + msg);
    if(OMP.debugFlag) XmLog.warning("[OpenMP target] " + msg);
  }

  static boolean isTargetClause(OMPpragma clauseKind){
    switch(clauseKind){
    case DIR_IF:
    case TARGET_DEVICE:
    case DATA_PRIVATE:
    case DATA_FIRSTPRIVATE:
    case TARGET_DATA_MAP:
    case IS_DEVICE_PTR:
    case DEFAULTMAP:
    case DIR_NOWAIT:
    case DEPEND:
      return true;
    default:
      return false;
    }
  }

  static boolean isTeamsClause(OMPpragma clauseKind){
    switch(clauseKind){
    case NUM_TEAMS:
    case THREAD_LIMIT:
    case DATA_DEFAULT:
    case DATA_PRIVATE:
    case DATA_FIRSTPRIVATE:
    case DATA_SHARED:
      /* DATA_DEFAULT_* */
      return true;
    default:
      return false;
    }
  }
  
  static boolean isDistributeClause(OMPpragma clauseKind) {
    switch(clauseKind){
    case DATA_PRIVATE:
    case DATA_FIRSTPRIVATE:
    case DATA_LASTPRIVATE:
    case COLLAPSE:
    case  DIST_SCHEDULE:
      return true;
    default:
      return false;
    }
  }

  static boolean isParallelClause(OMPpragma clauseKind) {
    switch(clauseKind){
    case DIR_IF:
    case DIR_NUM_THREADS:
    case DATA_DEFAULT:
    case DATA_PRIVATE:
    case DATA_FIRSTPRIVATE:
    case DATA_SHARED:
    case DATA_COPYIN:
    case PROC_BIND:
      return true;
    default:
      return false;
    }
  }

  static boolean isForClause(OMPpragma clauseKind) {
    switch(clauseKind){
    case DATA_PRIVATE:
    case DATA_FIRSTPRIVATE:
    case DATA_LASTPRIVATE:
    case DATA_LINEAR:
    case DIR_SCHEDULE:
    case COLLAPSE:
    case ORDERED:
    case DIR_NOWAIT:
      return true;
    default:
      return false;
    }
  }
  
  static boolean isSIMDClause(OMPpragma clauseKind) {
    switch(clauseKind){
    case SIMD_SAFELEN: /* safelen(length) */
    case SIMD_SIMDLEN: /* simdlen(length) */
    case SIMD_ALIGNED: /* aligned(arg-list[:alignment:) */

    case DATA_LINEAR:
    case DATA_PRIVATE:
    case DATA_LASTPRIVATE:
    case COLLAPSE:
      return true;
    default:
      return false;
    }
  }
}
