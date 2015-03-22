package exc.openacc;

import exc.block.*;
import exc.object.*;

public class AccInfoWriter extends AccProcessor {
  public AccInfoWriter(ACCglobalDecl globalDecl) {
    super(globalDecl, true, true);
  }

  void doGlobalAccPragma(Xobject def) throws ACCexception {
    AccInformation info = (AccInformation) def.getProp(AccInformation.prop);
    Xobject clauses = info.toXobject();
    def.setArg(1, clauses);
  }

  void doLocalAccPragma(PragmaBlock pb) throws ACCexception {
    AccDirective directive = (AccDirective)pb.getProp(AccDirective.prop);
    AccInformation info = directive.getInfo(); //(AccInformation) pb.getProp(AccInformation.prop);
    Xobject clauses = info.toXobject();
    pb.setClauses(clauses);
  }
}