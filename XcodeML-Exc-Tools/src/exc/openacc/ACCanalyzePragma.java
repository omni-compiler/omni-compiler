package exc.openacc;

import exc.block.*;
import exc.object.*;
import xcodeml.util.XmOption;

public class ACCanalyzePragma implements XobjectDefVisitor{
  private ACCglobalDecl _globalDecl;
  private ACCanalyzeLocalPragma _localPragmaAnalyzer;
  private ACCanalyzeGlobalPragma _globalPragmaAnalyzer;
  
  public ACCanalyzePragma(ACCglobalDecl globalDecl) {
    // FIXME current implementation only supports C language
    if (!XmOption.isLanguageC())
      ACC.fatal("current version only supports C language.");

    _globalDecl = globalDecl;
    _localPragmaAnalyzer = new ACCanalyzeLocalPragma(globalDecl);
    _globalPragmaAnalyzer = new ACCanalyzeGlobalPragma(globalDecl);
  }
  
  public void finalize() {
    _globalDecl.checkPresentData();
    _globalDecl.finalize();
  }

  public void doDef(XobjectDef def) {
    analyze(def);
  }
  
  private void analyze(XobjectDef def) {
    if (def.isFuncDef()) {
      FuncDefBlock fd = new FuncDefBlock(def);
      _localPragmaAnalyzer.analyze(fd);
    }else{
      _globalPragmaAnalyzer.analyze(def);      
      return;
    }
  }
}
