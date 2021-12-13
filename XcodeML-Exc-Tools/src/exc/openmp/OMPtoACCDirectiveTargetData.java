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
        XobjList accDataClauses = Xcons.List();
        for (Iterator<Xobject> it = ompClauses.iterator(); it.hasNext();) {
            XobjList clause = (XobjList) it.next();
            if (clause.Opcode() != Xcode.LIST ||
                clause.Nargs() < 1) {
                OMP.error((LineNo)xobj.getLineNo(),
                          "Clause list does not exist or number of clauses is too small.");
                return;
            }

            XobjList l = null;
            OMPpragma pragmaClause = OMPpragma.valueOf(clause.getArg(0));
            switch (pragmaClause) {
            case TARGET_DATA_MAP:
                l = clauseConverters.get(pragmaClause).convert(xobj, clause);
                break;
            case DIR_IF:
                l = clauseConverters.get(pragmaClause).
                    convert(xobj, clause,
                            new OMPpragma[]{OMPpragma.TARGET_DATA},
                            new OMPpragma[]{});
                break;
            case TARGET_DEVICE:
            case USE_DEVICE_PTR:
                OMP.error((LineNo)xobj.getLineNo(),
                          "Not implemented clause. ('" +
                          notImplementedClauseStr(pragmaClause) +
                          "').");
                break;
            default:
                OMP.error((LineNo)xobj.getLineNo(),
                          "Cannot be specified is cause.");
                break;
            }

            if (OMP.hasError()) {
                return;
            }

            if (l != null) {
                if (pragmaClause == OMPpragma.TARGET_DATA_MAP) {
                    accDataClauses.add(l);
                } else {
                    accClauses.add(l);
                }
            }
        }

        XobjList accData = createAccPragma(ACCpragma.DATA,
                                           accDataClauses, xobj);
        XobjList acc = createAccPragma(ACCpragma.PARALLEL,
                                       accClauses, xobj, 2);
        XobjArgs accArgs = new XobjArgs(acc, currentArgs.nextArgs());
        currentArgs.setNext(accArgs);
        currentArgs.setArg(accData);
    }
}
