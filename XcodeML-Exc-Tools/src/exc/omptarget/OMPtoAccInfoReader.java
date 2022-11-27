package exc.omptarget;

import exc.block.*;
import exc.object.*;
import exc.openacc.*;

import exc.openmp.OMPpragma;

class OMPtoAccInfoReader extends AccProcessor {
  public OMPtoAccInfoReader(ACCglobalDecl globalDecl) {
    super(globalDecl, true, true);
  }

  //
  // declare and routine pragma
  //
  public void doGlobalAccPragma(Xobject def) throws ACCexception {
    String directiveName = def.getArg(0).getString();
    OMPpragma directive = OMPpragma.valueOf(directiveName);
    if(OMPTarget.debug_flag) System.out.println("ONPtoAccInfoReader.doGlobalAccProgram ... pragma="+directiveName);

    if(!directive.isGlobalDirective()){
      throw new ACCexception(directive.getName() + " is not global directive");
    }

    Xobject clauseList = def.getArg(1);
    AccInformation info = new OMPtoAccInformation(directive, clauseList);

    XobjectDef xobjDef = (XobjectDef)def.getParent();
    switch (directive){
    case DECLARE_TARGET:
    case DECLARE_TARGET_START:
    case DECLARE_TARGET_END:
        // def.setProp(AccDirective.prop, new AccRoutine(_globalDecl, info, xobjDef));
        break;
      default:
        ACC.fatal("unknown directive: " + directive.getName());
    }
  }

  // data, parallel, parallel loop, ...
  public void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    String directiveName = pb.getPragma();
    OMPpragma directive = OMPpragma.valueOf(directiveName);
    if(OMPTarget.debug_flag) System.out.println("OMPtoAccInfoReader.doLocalAccProgram ... pragma="+directiveName);

    if(directive.isGlobalDirective()){
      throw new ACCexception(directiveName + " is not local directive");
    }

    Xobject clauseList = pb.getClauses();
    AccInformation info = new OMPtoAccInformation(directive, clauseList);

    // create each object for directive and put it under AccDirective,prop
    switch (directive){
    case TARGET_DATA:        /* target data <clause_list> */
      pb.setProp(AccDirective.prop, new OMPTargetData(_globalDecl, info, pb));
      break;
    case TARGET_ENTER_DATA:
      pb.setProp(AccDirective.prop, new OMPTargetEnterData(_globalDecl, info, pb));
      break;
    case TARGET_EXIT_DATA:
      pb.setProp(AccDirective.prop, new OMPTargetExitData(_globalDecl, info, pb));
      break;
    case TARGET_UPDATE:
      pb.setProp(AccDirective.prop, new OMPTargetUpdate(_globalDecl, info, pb));
      break;

    case TARGET:             /* target <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallel(_globalDecl, info, pb, false, false));
      break;
    case TARGET_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallelLoop(_globalDecl, info, pb, false, false, false, true));
      break;

    case TARGET_PARALLEL:    /* target parallel <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallel(_globalDecl, info, pb, false, true));
      break;

    case TARGET_PARALLEL_LOOP: /* target parallel for <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallelLoop(_globalDecl, info, pb, false, false, true, false));
      break;
    case TARGET_PARALLEL_LOOP_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallelLoop(_globalDecl, info, pb, false, false, true, true));
      break;

    case TARGET_TEAMS:       /* target teams <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallel(_globalDecl, info, pb, true, false));
      break;
    case TARGET_TEAMS_DISTRIBUTE: /* target teams distribute <clause_list>  */
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallelLoop(_globalDecl, info, pb, true, true, false, false));
      break;
    case TARGET_TEAMS_DISTRIBUTE_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallelLoop(_globalDecl, info, pb, true, true, false, true));
      break;

    case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP: /* target teams distribute parallel for <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallelLoop(_globalDecl, info, pb, true, true, true, false));
      break;
    case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetParallelLoop(_globalDecl, info, pb, true, true, true, true));
      break;

    case TEAMS:              /* teams <clause_list> *//* not loop */
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, true, false, false, false));
      break;
    case TEAMS_DISTRIBUTE:   /* teams distribute <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, true, true, false, false));
      break;
    case TEAMS_DISTRIBUTE_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, true, true, false, true));
      break;
    case TEAMS_DISTRIBUTE_PARALLEL_LOOP: /* teams distribute parallel for <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, true, true, true, false));
      break;
    case TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, true, true, true, true));
      break;
      
    case DISTRIBUTE:         /* distribute <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, false, true, false, false));
      break;
    case DISTRIBUTE_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, false, true, false, true));
      break;
    case DISTRIBUTE_PARALLEL_LOOP: /* distribute parallel for <clause_list> */
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, false, true, true, false));
      break;
    case DISTRIBUTE_PARALLEL_LOOP_SIMD:
      pb.setProp(AccDirective.prop,
                 new OMPTargetLoop(_globalDecl, info, pb, false, true, true, true));
      break;

    case PARALLEL:
    case FOR: /*PARALLEL_LOOP*/
      
    default:
      ACC.fatal("unknown directive: " + directive.getName());
    }
  }
}
