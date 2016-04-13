package exc.openacc;

import exc.object.*;
import exc.block.*;

abstract class AccProcessor {
  final ACCglobalDecl _globalDecl;
  final private boolean _isTopdown;
  final private boolean _warnUnknownPragma;

  AccProcessor(ACCglobalDecl globalDecl, boolean isTopdown, boolean warnUnknownPragma) {
    _globalDecl = globalDecl;
    _isTopdown = isTopdown;
    _warnUnknownPragma = warnUnknownPragma;
  }

  void doNonFuncDef(Xobject x) {
    switch (x.Opcode()){
    case ACC_PRAGMA:
      try{
        doGlobalAccPragma(x);
      }catch(ACCexception e){
        ACC.error(x.getLineNo(), e.getMessage());
      }
      break;
    case PRAGMA_LINE:
      if(_warnUnknownPragma) {
        ACC.warning(x.getLineNo(), "unknown pragma : " + x);
      }
      break;
    default:
    }
  }

  void doFuncDef(FunctionBlock fb){
    BlockIterator blockIterator;
    if(_isTopdown){
      blockIterator = new topdownBlockIterator(fb);
    }else{
      blockIterator = new bottomupBlockIterator(fb);
    }

    for (blockIterator.init(); !blockIterator.end(); blockIterator.next()) {
      Block b = blockIterator.getBlock();
      switch (b.Opcode()) {
      case ACC_PRAGMA:
        try {
          doLocalAccPragma((PragmaBlock) b);
        } catch (ACCexception e) {
          ACC.error(b.getLineNo(), e.getMessage());
        }
        break;
      case PRAGMA_LINE:
        if (_warnUnknownPragma) {
          ACC.warning(b.getLineNo(), "unknown pragma : " + b);
        }
        break;
      default:
      }
    }
  }

  abstract void doGlobalAccPragma(Xobject x) throws ACCexception;
  abstract void doLocalAccPragma(PragmaBlock pb) throws ACCexception;
}
