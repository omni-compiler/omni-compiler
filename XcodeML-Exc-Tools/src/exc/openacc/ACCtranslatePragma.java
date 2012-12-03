package exc.openacc;

import xcodeml.util.XmOption;
import exc.object.*;
import exc.block.*;

public class ACCtranslatePragma implements XobjectDefVisitor{
  private ACCglobalDecl _globalDecl;
  private ACCtranslateLocalPragma _localPragmaTranslator;
  //private ACCtranslateGlobalPragma _globalPragmaTranslator;
  
  public ACCtranslatePragma(ACCglobalDecl globalDecl) {
    // FIXME current implementation only supports C language
    if (!XmOption.isLanguageC())
      ACC.fatal("current version only supports C language.");

    _globalDecl = globalDecl;
    _localPragmaTranslator = new ACCtranslateLocalPragma(globalDecl);
    //_globalPragmaTranslator = new ACCanalyzeGlobalPragma(globalDecl);
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
      //_globalPragmaTranslator.analyze(def);      
      return;
    }
  }
}
