package exc.openacc;

import exc.block.PragmaBlock;
import exc.object.PropObject;
import exc.object.Xobject;

class AccRewriter extends AccProcessor{
  public AccRewriter(ACCglobalDecl globalDecl) {
    super(globalDecl, false, false);
  }

  void doGlobalAccPragma(Xobject def) throws ACCexception {
    doAccPragma(def);
  }

  void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    doAccPragma(pb);
  }

  void doAccPragma(PropObject po) throws ACCexception {
    Object obj = po.getProp(AccDirective.prop);
    if (obj == null) return;
    AccDirective dire = (AccDirective) obj;
    dire.rewrite();
  }
}
