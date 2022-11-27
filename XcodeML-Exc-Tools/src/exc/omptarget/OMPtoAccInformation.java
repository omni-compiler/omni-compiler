/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.omptarget;

import java.util.*;

import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

public class OMPtoAccInformation extends AccInformation {

  //
  // constructor: OMPtoAccInformation
  //
  public OMPtoAccInformation(OMPpragma pragma, Xobject arg) throws ACCexception {
    super(ACCpragma.NONE,arg);
    omp_pragma = pragma;
    switch(pragma){
    case TARGET_DATA:        /* target data <clause_list> */
      _pragma = ACCpragma.DATA;
      break;
    case TARGET_ENTER_DATA:
      _pragma = ACCpragma.ENTER_DATA;
      break;
    case TARGET_EXIT_DATA:
      _pragma = ACCpragma.EXIT_DATA;
      break;
    case TARGET_UPDATE:
      _pragma = ACCpragma.UPDATE;
      break;

    case TARGET:             /* target <clause_list> */
    case TARGET_PARALLEL:    /* target parallel <clause_list> */
    case TARGET_TEAMS:       /* target teams <clause_list> */
      _pragma = ACCpragma.PARALLEL;
      break;

    case TARGET_SIMD:
    case TARGET_PARALLEL_LOOP: /* target parallel for <clause_list> */
    case TARGET_PARALLEL_LOOP_SIMD:
    case TARGET_TEAMS_DISTRIBUTE: /* target teams distribute <clause_list>  */
    case TARGET_TEAMS_DISTRIBUTE_SIMD:
    case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP: /* target teams distribute parallel for <clause_list> */
    case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD:
      _pragma = ACCpragma.PARALLEL_LOOP;
      break;

    case TEAMS:              /* teams <clause_list> *//* not loop */
    case TEAMS_DISTRIBUTE:   /* teams distribute <clause_list> */
    case TEAMS_DISTRIBUTE_SIMD:
    case TEAMS_DISTRIBUTE_PARALLEL_LOOP: /* teams distribute parallel for <clause_list> */
    case TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD:
    case DISTRIBUTE:         /* distribute <clause_list> */
    case DISTRIBUTE_SIMD:
    case DISTRIBUTE_PARALLEL_LOOP: /* distribute parallel for <clause_list> */
    case DISTRIBUTE_PARALLEL_LOOP_SIMD:
      _pragma = ACCpragma.LOOP;
      break;

    case PARALLEL:
    case PARALLEL_FOR:
    default:
      break;
    }

    default_var_attr = ACCpragma.NONE;

    if(OMPTarget.debug_flag) System.out.println("OMPtoAccInformation: pragma="+pragma+", _pragma="+_pragma+", args="+arg);

    for (Xobject o : (XobjList)arg) {
      XobjList clause = (XobjList) o;
      OMPpragma clauseKind = OMPpragma.valueOf(clause.getArg(0));
      Xobject clauseArg = clause.getArgOrNull(1);
      addClause(makeClause(clauseKind, clauseArg));
    }
  }
  
  Clause makeClause(OMPpragma clauseKind, Xobject arg) throws ACCexception {
    switch (clauseKind) {
    case DIR_IF: /* if(scalar-expr) */
      return new IntExprClause(ACCpragma.IF, arg);
    case DIR_NUM_THREADS: /* num_threads(expr) */
      return new IntExprClause(ACCpragma.OMP_NUM_THREADS, arg);
    case NUM_TEAMS:  /* num_teams(expr) */
      return new IntExprClause(ACCpragma.OMP_NUM_TEAMS, arg);
    case THREAD_LIMIT: /* thread_limit(expr) */
      return new IntExprClause(ACCpragma.OMP_THREAD_LIMIT, arg);
    case COLLAPSE: /* collapse(n) */
      return new IntExprClause(ACCpragma.COLLAPSE, arg);

    case TARGET_DEVICE: /* device(expr) */
    case USE_DEVICE_PTR: /* use_device_ptr(list) */
    case IS_DEVICE_PTR: /* is_device_ptr(list) */

    case TARGET_UPDATE_TO: /* to(list) */
    case TARGET_UPDATE_FROM: /* from(list) */

    case DEFAULTMAP: /* defaultmap(tofrom:scalar) */
    case DEPEND:   /* depend([in|out|inout]: list */
    case DIST_SCHEDULE: /* dist_schedule(kind[,chunk_size]) */
    case PROC_BIND:  /* proc_bind([master|cloase|spread]) */
    case DATA_LINEAR: /* linaer( ) */

    case TARGET_DATA_MAP: /*  */
      {
        String map_type = arg.getArg(0).getString();
        
        if(map_type.equals("from")) /* COPYIN */
          return new VarListClause(ACCpragma.COPYIN,arg.getArg(1));
        if(map_type.equals("tofrom")) /* COPYIN */
          return new VarListClause(ACCpragma.COPY,arg.getArg(1));
        if(map_type.equals("to")) /* COPYIN */
          return new VarListClause(ACCpragma.COPYOUT,arg.getArg(1));
        /* alloc, release, delete + always*/

        OMPTarget.fatal("unknown TARGET_DATA_MAP map_type="+map_type);
      }
      break;

    case DATA_DEFAULT:   /* default(shared|none) */
    case DATA_PRIVATE:   /* private(list) */
    case DATA_SHARED:    /* shared(list) */
    case DATA_FIRSTPRIVATE: /* firstprivate(list) */
    case DATA_LASTPRIVATE:  /* lastprivate(list) */
    case DATA_COPYPRIVATE: /* copyprivate(list) */
    case DATA_COPYIN:  /* copyin(list) */

    // case DEFAULT:
    //   {
    //     Xobject atr = arg.getArg(0);
    //     if(atr != null){
    //       if(atr.getName().equals("none")) default_var_attr = ACCpragma.DEFAULT_NONE;
    //       else if(atr.getName().equals("present")) default_var_attr = ACCpragma.PRESENT;
    //       else {
    //         throw new ACCexception("bad default clause");
    //       }
    //     }            
    //     return new VarListClause(clauseKind,Xcons.List());
    //   }
    default:
      // return new VarListClause(clauseKind, (XobjList)arg);
      OMPTarget.fatal("unknown OMPtoACC clause="+clauseKind);
    }
    return null; // error 
  }
}
