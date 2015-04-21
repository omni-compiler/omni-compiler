package exc.openacc;

import exc.object.*;
import xcodeml.util.XmOption;

public class AccTranslator implements XobjectDefVisitor {
  private final ACCglobalDecl _globalDecl;
  private final AccInfoReader _infoReader;
  private final AccInfoWriter _infoWriter;
  private final AccAnalyzer _analyzer;
  private final AccGenerator _generator;
  private final AccRewriter _rewrite;
  private final boolean _onlyAnalyze;

  public AccTranslator(XobjectFile xobjFile, boolean onlyAnalyze){
    if (!XmOption.isLanguageC()) {
      ACC.fatal("current version only supports C language.");
    }

    _globalDecl = new ACCglobalDecl(xobjFile);
    _infoReader = new AccInfoReader(_globalDecl);
    _infoWriter = new AccInfoWriter(_globalDecl);
    _analyzer = new AccAnalyzer(_globalDecl);
    _generator = new AccGenerator(_globalDecl);
    _rewrite = new AccRewriter(_globalDecl);
    _onlyAnalyze = onlyAnalyze;
  }

  @Override
  public void doDef(XobjectDef def) {
    _infoReader.doDef(def);
    ACC.exitByError();
    _analyzer.doDef(def);
    ACC.exitByError();

    if(_onlyAnalyze) {
      _infoWriter.doDef(def);
      ACC.exitByError();
      return;
    }

    _generator.doDef(def);
    ACC.exitByError();
    _rewrite.doDef(def);
    ACC.exitByError();
  }

  public void finish(){
    if(!_onlyAnalyze) {
      ACCgpuDecompiler gpuDecompiler = new ACCgpuDecompiler();
      gpuDecompiler.decompile(_globalDecl);

      _globalDecl.setupGlobalConstructor();
      _globalDecl.setupGlobalDestructor();
      _globalDecl.setupMain();
      _globalDecl.setupHeaderInclude();
    }

    _globalDecl.finish();
  }
}
