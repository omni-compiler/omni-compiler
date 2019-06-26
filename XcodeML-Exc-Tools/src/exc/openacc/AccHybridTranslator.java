package exc.openacc;

import exc.block.FuncDefBlock;
import exc.block.FunctionBlock;
import exc.object.*;
import xcodeml.util.XmOption;

public class AccHybridTranslator implements XobjectDefVisitor {

    public AccHybridTranslator(XobjectFile xobjFile) {
        if (!XmOption.isLanguageC()) {
            ACC.fatal("current version only supports C language.");
        }
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

    public void finish() {
        // ヘッダーを出力する？

        if (!_onlyAnalyze) {
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