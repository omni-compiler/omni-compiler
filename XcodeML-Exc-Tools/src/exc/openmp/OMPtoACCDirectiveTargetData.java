package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACCDirectiveTargetData extends OMPtoACCDirective {
    public OMPtoACCDirectiveTargetData() {
        super();
    }

    @Override
    public void convert(Xobject xobj,
                        XobjArgs currentArgs) {
        if (xobj.getArg(1) == null ||
            xobj.getArg(1).Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Not found clause list.");
            return;
        }

        XobjList ompClauses = (XobjList) xobj.getArg(1);
        XobjList accClauses = Xcons.List();
        for (Iterator<Xobject> it = ompClauses.iterator(); it.hasNext();) {
            XobjList clause = (XobjList) it.next();
            if (clause.Opcode() != Xcode.LIST ||
                clause.Nargs() < 1) {
                OMP.error((LineNo)xobj.getLineNo(),
                          "Clause list does not exist or number of clauses is too small.");
                return;
            }

            XobjList l = null;
            switch (OMPpragma.valueOf(clause.getArg(0))) {
            case TARGET_DATA_MAP:
                l = clauseConverter.convertFromMap(xobj, clause);
                break;
            case DIR_IF:
                l = clauseConverter.convertFromIf(xobj, clause,
                                                  new OMPpragma[]{OMPpragma.TARGET_DATA});
                break;
            }

            if (OMP.hasError()) {
                return;
            }

            if (l != null) {
                accClauses.add(l);
            }
        }

        currentArgs.setArg(createAccPragma(ACCpragma.PARALLEL,
                                           accClauses, xobj, 2));
    }
}
