package exc.openacc;

import exc.object.*;
import exc.block.*;

abstract class AccProcessor implements XobjectDefVisitor {
  final ACCglobalDecl _globalDecl;
  final private boolean _isTopdown;
  final private boolean _isFinal;

  AccProcessor(ACCglobalDecl globalDecl, boolean isTopdown, boolean isFinal) {
    _globalDecl = globalDecl;
    _isTopdown = isTopdown;
    _isFinal = isFinal;
  }

  AccProcessor(ACCglobalDecl globalDecl, boolean isTopdown) {
    this(globalDecl, isTopdown, false);
  }

  public void doDef(XobjectDef def) {
    if (def.isFuncDef()) {
      doFuncDef(def);
    } else {
      doNonFuncDef(def);
    }
  }

  private void doNonFuncDef(XobjectDef def) {
    Xobject x = def.getDef();
    switch (x.Opcode()){
    case ACC_PRAGMA:
      try{
        doGlobalAccPragma(x);
      }catch(ACCexception e){
        ACC.error(x.getLineNo(), e.getMessage());
      }
      break;
    case PRAGMA_LINE:
      ACC.error(x.getLineNo(), "unknown pragma : " + x);
      break;
    default:
    }
  }

  private void doFuncDef(XobjectDef def){
    FuncDefBlock fd = new FuncDefBlock(def);
    FunctionBlock fb = fd.getBlock();

    BlockIterator blockIterator;
    if(_isTopdown){
      blockIterator = new topdownBlockIterator(fb);
    }else{
      blockIterator = new bottomupBlockIterator(fb);
    }

    for (blockIterator.init(); !blockIterator.end(); blockIterator.next()) {
      Block b = blockIterator.getBlock();
      if (b.Opcode() == Xcode.ACC_PRAGMA) {
        PragmaBlock pb = (PragmaBlock) b;
        try {
          doLocalAccPragma(pb);
        } catch (ACCexception e) {
          ACC.error(pb.getLineNo(), e.getMessage());
        }
      }
    }

    if(_isFinal){
      fd.Finalize();
    }
  }

  abstract void doGlobalAccPragma(Xobject x) throws ACCexception;
  abstract void doLocalAccPragma(PragmaBlock pb) throws ACCexception;
}
