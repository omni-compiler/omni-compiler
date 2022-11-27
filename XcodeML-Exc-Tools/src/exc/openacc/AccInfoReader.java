/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.*;
import exc.object.*;

class AccInfoReader extends AccProcessor{
  public AccInfoReader(ACCglobalDecl globalDecl) {
    super(globalDecl, true, true);
  }

  //
  // declare and routine pragma
  //
  public void doGlobalAccPragma(Xobject def) throws ACCexception {
    String directiveName = def.getArg(0).getString();
    ACCpragma directive = ACCpragma.valueOf(directiveName);
    if(ACC.debug_flag) System.out.println("AccInfoReader.doGlobalAccProgram ... pragma="+directiveName);

    if(!directive.isGlobalDirective()){
      throw new ACCexception(directive.getName() + " is not global directive");
    }

    Xobject clauseList = def.getArg(1);
    AccInformation info = new AccInformation(directive, clauseList);

    XobjectDef xobjDef = (XobjectDef)def.getParent();
    switch (directive){
      case DECLARE:
        def.setProp(AccDirective.prop, new AccDeclare(_globalDecl, info, xobjDef));
        break;
      case ROUTINE:
        def.setProp(AccDirective.prop, new AccRoutine(_globalDecl, info, xobjDef));
        break;
      default:
        ACC.fatal("unknown directive: " + directive.getName());
    }
  }

  // data, parallel, parallel loop, ...
  public void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    String directiveName = pb.getPragma();
    ACCpragma directive = ACCpragma.valueOf(directiveName);
    if(ACC.debug_flag) System.out.println("AccInfoReader.doLocalAccProgram ... pragma="+directiveName);
    if(! directive.isLocalDirective()){
      throw new ACCexception(directiveName + " is not local directive");
    }

    Xobject clauseList = pb.getClauses();
    AccInformation info = new AccInformation(directive, clauseList);

    // create each object for directive and put it under AccDirective,prop
    switch (directive){
    case DATA:
      pb.setProp(AccDirective.prop, new AccData(_globalDecl, info, pb));
      break;
    case PARALLEL:
      pb.setProp(AccDirective.prop, new AccParallel(_globalDecl, info, pb));
      break;
    case PARALLEL_LOOP:
      pb.setProp(AccDirective.prop, new AccParallelLoop(_globalDecl, info, pb));
      break;
    case LOOP:
      pb.setProp(AccDirective.prop, new AccLoop(_globalDecl, info, pb));
      break;
    case UPDATE:
      pb.setProp(AccDirective.prop, new AccUpdate(_globalDecl, info, pb));
      break;
    case HOST_DATA:
      pb.setProp(AccDirective.prop, new AccHostData(_globalDecl, info, pb));
      break;
    case WAIT:
      pb.setProp(AccDirective.prop, new AccWait(_globalDecl, info, pb));
      break;
    case KERNELS:
      pb.setProp(AccDirective.prop, new AccKernels(_globalDecl, info, pb));
      break;
    case KERNELS_LOOP:
      pb.setProp(AccDirective.prop, new AccKernelsLoop(_globalDecl, info, pb));
      break;
    case ENTER_DATA:
      pb.setProp(AccDirective.prop, new AccEnterData(_globalDecl, info, pb));
      break;
    case EXIT_DATA:
      pb.setProp(AccDirective.prop, new AccExitData(_globalDecl, info, pb));
      break;
    case ATOMIC:
      pb.setProp(AccDirective.prop, new AccAtomic(_globalDecl, info, pb));
      break;
    case DECLARE:
      pb.setProp(AccDirective.prop, new AccDeclare(_globalDecl, info, pb));
      break;
    // case SYNC:
    //   pb.setProp(AccDirective.prop, new AccSync(_globalDecl, info, pb));
    //   break;
    // case FLUSH:
    //   pb.setProp(AccDirective.prop, new AccFlush(_globalDecl, info, pb));
    //   break;
    // case YIELD:
    //   pb.setProp(AccDirective.prop, new AccYield(_globalDecl, info, pb));
    //   break;
    default:
      ACC.fatal("unknown directive: " + directive.getName());
    }
  }
}
