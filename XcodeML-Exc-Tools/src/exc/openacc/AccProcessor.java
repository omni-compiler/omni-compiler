/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.object.*;
import exc.block.*;

public abstract class AccProcessor {
  public final ACCglobalDecl _globalDecl;
  final private boolean _isTopdown;  // order of traverse
  final private boolean _warnUnknownPragma;

  public AccProcessor(ACCglobalDecl globalDecl, boolean isTopdown, boolean warnUnknownPragma) {
    _globalDecl = globalDecl;
    _isTopdown = isTopdown;
    _warnUnknownPragma = warnUnknownPragma;
  }

  public void doNonFuncDef(Xobject x) {
    switch (x.Opcode()){
    case ACC_PRAGMA:
    case OMP_PRAGMA:
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
      break;
    }
  }

  public void doFuncDef(FunctionBlock fb){
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
      case OMP_PRAGMA:
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
        break;
      }
    }
  }

  public abstract void doGlobalAccPragma(Xobject x) throws ACCexception;
  public abstract void doLocalAccPragma(PragmaBlock pb) throws ACCexception;
}
