package exc.openacc;

import exc.block.*;
import exc.object.*;

class AccInfoReader extends AccProcessor{
  public AccInfoReader(ACCglobalDecl globalDecl) {
    super(globalDecl, true, false, true);
  }

  void doGlobalAccPragma(Xobject def) throws ACCexception {
    String pragmaName = def.getArg(0).getString();
    ACCpragma pragma = ACCpragma.valueOf(pragmaName);

    if(! pragma.isGlobalDirective()){
      throw new ACCexception(pragma.getName() + " is not global directive");
    }

    Xobject clauseList = def.getArg(1);
    AccInformation info = new AccInformation(pragma, clauseList);

    def.setProp(AccDirective.prop, new AccDeclare(_globalDecl, info));
  }

  void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    String directiveName = pb.getPragma();
    ACCpragma directive = ACCpragma.valueOf(directiveName);
    if(! directive.isLocalDirective()){
      throw new ACCexception(directiveName + " is not local directive");
    }

    Xobject clauseList = pb.getClauses();
    AccInformation info = new AccInformation(directive, clauseList);

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
    default:
    }
  }
}
