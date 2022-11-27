/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.PragmaBlock;
import exc.object.PropObject;
import exc.object.Xobject;

class AccGenerator extends AccProcessor{
  public AccGenerator(ACCglobalDecl globalDecl) {
    super(globalDecl, true, false);
  }

  public void doGlobalAccPragma(Xobject def) throws ACCexception {
    doAccPragma(def);
  }

  public void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    doAccPragma(pb);
  }

  void doAccPragma(PropObject po) throws ACCexception {
    Object obj = po.getProp(AccDirective.prop);
    if (obj == null) return;
    AccDirective dire = (AccDirective) obj;
    dire.generate(); // call generate method.
  }
}
