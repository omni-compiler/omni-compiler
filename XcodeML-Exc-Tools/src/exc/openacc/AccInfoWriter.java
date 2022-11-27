/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.*;
import exc.object.*;

class AccInfoWriter extends AccProcessor {
  public AccInfoWriter(ACCglobalDecl globalDecl) {
    super(globalDecl, true, false);
  }

  public void doGlobalAccPragma(Xobject def) throws ACCexception {
    AccDirective directive = (AccDirective)def.getProp(AccDirective.prop);
    AccInformation info = directive.getInfo();
    Xobject clauses = info.toXobject();
    def.setArg(1, clauses);
  }

  public void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    AccDirective directive = (AccDirective)pb.getProp(AccDirective.prop);
    AccInformation info = directive.getInfo(); //(AccInformation) pb.getProp(AccInformation.prop);
    Xobject clauses = info.toXobject();
    pb.setClauses(clauses);
  }
}
