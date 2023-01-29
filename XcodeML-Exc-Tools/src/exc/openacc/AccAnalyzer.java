/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.PragmaBlock;
import exc.object.PropObject;
import exc.object.Xobject;

class AccAnalyzer extends AccProcessor {
  public AccAnalyzer(ACCglobalDecl globalDecl) {
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
    if (obj == null)  return;
    AccDirective dire = (AccDirective) obj;
    // System.out.println("doAccPragma dire="+dire);
    dire.analyze(); // call analyze method.
  }
}
