package exc.openacc;

import xcodeml.util.XmOption;
import exc.object.*;
import exc.block.*;

public class ACCtranslatePragma implements XobjectDefVisitor{
  private ACCglobalDecl _globalDecl;
  private ACCtranslateLocalPragma _localPragmaTranslator;
  private ACCtranslateGlobalPragma _globalPragmaTranslator;
  
  public ACCtranslatePragma(ACCglobalDecl globalDecl) {
    // FIXME current implementation only supports C language
    if (!XmOption.isLanguageC())
      ACC.fatal("current version only supports C language.");

    _globalDecl = globalDecl;
    _localPragmaTranslator = new ACCtranslateLocalPragma(globalDecl);
    _globalPragmaTranslator = new ACCtranslateGlobalPragma(globalDecl);
  }
  
  public void finalize() {
    _globalDecl.finalize();
  }

  public void doDef(XobjectDef def) {
    translate(def);
  }
  
  private void translate(XobjectDef def) {
    if (def.isFuncDef()) {
      FuncDefBlock fd = new FuncDefBlock(def);
      _localPragmaTranslator.translate(fd);
    }else{
      _globalPragmaTranslator.translate(def);      
      return;
    }
  }
}

class ACCtranslateGlobalPragma{
  ACCglobalDecl _globalDecl;
  public ACCtranslateGlobalPragma(ACCglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }
  
  public void translate(XobjectDef def) {
    Xobject x = def.getDef();
    if(x.Opcode() == Xcode.ACC_PRAGMA){
      try{
        translatePragma(x);
      }catch(ACCexception e){
        ACC.error(x.getLineNo(), e.getMessage());
      }
    }else if(x.Opcode() == Xcode.PRAGMA_LINE){
      ACC.error(x.getLineNo(), "unknown pragma : " + x);
    }
  }
  
  private void translatePragma(Xobject x) throws ACCexception{
    String pragmaName = x.getArg(0).getString();
    switch(ACCpragma.valueOf(pragmaName)){
    case DECLARE:
      translateDeclare(x);
      break;
    default:
      throw new ACCexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }
  
  private void translateDeclare(Xobject x) throws ACCexception{
    ACCtranslateDeclare translator = new ACCtranslateDeclare(x);
    translator.translate();
  }
}
