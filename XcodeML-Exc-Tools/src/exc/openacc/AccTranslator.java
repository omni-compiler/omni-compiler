/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.FuncDefBlock;
import exc.block.FunctionBlock;
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
    if (def.isFuncDef()) {
      FuncDefBlock fd = new FuncDefBlock(def);
      FunctionBlock fb = fd.getBlock();
      doFuncDef(fb);
      fd.finalizeBlock();
    } else {
      Xobject x = def.getDef();
      doNonFuncDef(x);
    }
  }

  private void doFuncDef(FunctionBlock fb){
    System.out.println("### doFuncDef infoReader ...");
    _infoReader.doFuncDef(fb); // infoReader: set info?
    ACC.exitByError();

    System.out.println("### doFuncDef analyzer ...");
    _analyzer.doFuncDef(fb); // analyze
    ACC.exitByError();

    if(_onlyAnalyze) {
      _infoWriter.doFuncDef(fb);
      ACC.exitByError();
      return;
    }

    System.out.println("### doFuncDef generator ...");
    _generator.doFuncDef(fb);  // geneate
    ACC.exitByError();

    System.out.println("### doFuncDef rewrite ...");
    _rewrite.doFuncDef(fb);  // rewrite
    ACC.exitByError();
  }

  private void doNonFuncDef(Xobject x){
    _infoReader.doNonFuncDef(x);
    ACC.exitByError();
    _analyzer.doNonFuncDef(x);
    ACC.exitByError();

    if(_onlyAnalyze) {
      _infoWriter.doNonFuncDef(x);
      ACC.exitByError();
      return;
    }

    _generator.doNonFuncDef(x);
    ACC.exitByError();
    _rewrite.doNonFuncDef(x);
    ACC.exitByError();
  }

  public void finish(){
    if(!_onlyAnalyze) {
      ACCgpuDecompiler gpuDecompiler = new ACCgpuDecompiler();
      gpuDecompiler.decompile(_globalDecl);

      _globalDecl.setupKernelsInitAndFinalize();
      _globalDecl.setupGlobalConstructor();
      _globalDecl.setupGlobalDestructor();
      _globalDecl.setupMain();
      _globalDecl.setupHeaderInclude();
    }

    _globalDecl.finish();
  }
}
