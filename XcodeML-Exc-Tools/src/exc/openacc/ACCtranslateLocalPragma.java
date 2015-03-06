package exc.openacc;
import exc.block.*;
import exc.object.*;


public class ACCtranslateLocalPragma {
  private ACCglobalDecl   _globalDecl;
  private XobjectDef currentDef;
  
  public ACCtranslateLocalPragma(ACCglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }
  
  public void translate(FuncDefBlock def) {
    FunctionBlock fb = def.getBlock();
    currentDef = def.getDef();

    BlockIterator i = new topdownBlockIterator(fb);
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() ==  Xcode.ACC_PRAGMA) {
        PragmaBlock pb = (PragmaBlock)b;
        try {
          translatePragma(pb);
        } catch (ACCexception e) {
          ACC.error(pb.getLineNo(), e.getMessage());
        }
      }else if(b.Opcode() == Xcode.PRAGMA_LINE){
        PragmaBlock pb = (PragmaBlock)b;
        ACC.error(pb.getLineNo(), "unknown pragma : " + pb.getClauses());
      }
    }
  }
  
  private void translatePragma(PragmaBlock pb) throws ACCexception{
    String pragmaName = pb.getPragma();

    switch (ACCpragma.valueOf(pragmaName)) {
    case PARALLEL:
      translateParallel(pb); break;
    case KERNELS:
      translateKernels(pb); break;
    case DATA:
    case ENTER_DATA:
    case EXIT_DATA:
      translateData(pb); break;
    case HOST_DATA:
      //translateHostData(pb); 
      break;
    case LOOP: break;
    case CACHE:
      //translateCache(pb); 
      break;
    case PARALLEL_LOOP:
      translateParallel(pb);
      break;
    case KERNELS_LOOP:
      translateKernels(pb); break;
    case DECLARE:
      translateDeclare(pb); break;
    case UPDATE:
      translateUpdate(pb); break;
    case WAIT:
      translateWait(pb); break;
    default:
      throw new ACCexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }
  
  private void translateParallel(PragmaBlock pb) throws ACCexception{
    ACCtranslateParallel translator = new ACCtranslateParallel(pb);
    translator.translate();
  }
  
  private void translateKernels(PragmaBlock pb) throws ACCexception{
    ACCtranslateKernels translator = new ACCtranslateKernels(pb);
    translator.translate();
  }
  
  private void translateData(PragmaBlock pb) throws ACCexception{
    ACCtranslateData translator = new ACCtranslateData(pb);
    translator.translate();
  }
  
  private void translateDeclare(PragmaBlock pb) throws ACCexception{
    ACCtranslateDeclare translator = new ACCtranslateDeclare(pb);
    translator.translate();
  }
  
  private void translateUpdate(PragmaBlock pb) throws ACCexception{
    ACCtranslateUpdate translator = new ACCtranslateUpdate(pb);
    translator.translate();
  }
  
  private void translateWait(PragmaBlock pb) throws ACCexception{
    ACCtranslateWait translator = new ACCtranslateWait(pb);
    translator.translate();
  }
}
