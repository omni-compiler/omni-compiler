package exc.openacc;

import exc.block.*;
import exc.object.*;

public class AccInfoReader extends AccProcessor{
  public AccInfoReader(ACCglobalDecl globalDecl) {
    super(globalDecl, true);
  }

  void doGlobalAccPragma(Xobject def) throws ACCexception {
    String pragmaName = def.getArg(0).getString();
    Xobject clauseList = def.getArg(1);
    ACCpragma pragma = ACCpragma.valueOf(pragmaName);
    ACC.debug(pragma.getName() + " directive : " + clauseList);

    if(! pragma.isGlobal()){
      throw new ACCexception(pragma.getName() + " is not global directive");
    }

    AccInformation info = new AccInformation(pragma, clauseList);
    //def.setProp(AccInformation.prop, info);

    def.setProp(AccDirective.prop, new AccDeclare(_globalDecl, info));
  }

  void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    String directiveName = pb.getPragma();
    ACCpragma directive = ACCpragma.valueOf(directiveName);
    Xobject clauseList = pb.getClauses();
    ACC.debug(directiveName + " directive : " + clauseList);

    if(! directive.isLocal()){
      throw new ACCexception(directiveName + " is not local directive");
    }

    AccInformation info = new AccInformation(directive, clauseList);
    //pb.setProp(AccInformation.prop, info);

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
    default:
    }
  }

//  private void fixXobject(Xobject x, Block b) throws ACCexception {
//    topdownXobjectIterator xIter = new topdownXobjectIterator(x);
//    for (xIter.init(); !xIter.end(); xIter.next()) {
//      Xobject xobj = xIter.getXobject();
//      if (xobj.Opcode() == Xcode.VAR) {
//        String name = xobj.getName();
//        Ident id = findVarIdent(b, name);
//        if (id == null) throw new ACCexception("'" + name + "' is not exist");
//        xIter.setXobject(id.Ref());
//      }
//    }
//  }
}
