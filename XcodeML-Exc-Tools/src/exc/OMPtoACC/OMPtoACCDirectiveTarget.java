package exc.OMPtoACC;

import exc.object.*;
import exc.openmp.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACCDirectiveTarget extends OMPtoACCDirective {
    public OMPtoACCDirectiveTarget() {
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

            System.out.println("clause="+clause);
            XobjList l = null;
            OMPpragma pragmaClause = OMPpragma.valueOf(clause.getArg(0));
            switch (pragmaClause) {
            case TARGET_DATA_MAP:
            case DATA_PRIVATE:
            case DATA_FIRSTPRIVATE:
            case TARGET_DEVICE:
            case IS_DEVICE_PTR:
            case DEFAULTMAP:
            case DIR_NOWAIT:
            case DEPEND:
                l = clauseConverters.get(pragmaClause).convert(xobj, clause);
                break;
            case DIR_IF:
                l = clauseConverters.get(pragmaClause).
                    convert(xobj, clause,
                            new OMPpragma[]{OMPpragma.TARGET},
                            new OMPpragma[]{});
                break;
            default:
                OMP.error((LineNo)xobj.getLineNo(),
                          "clause cannot be specified");
                break;
            }

            if (OMP.hasError()) {
                return;
            }

            if (l != null) {
                if (pragmaClause == OMPpragma.TARGET_DATA_MAP) {
                    accDataClauses.add(l);
                } else {
                    // Delay all but copyXX/create clause.
                    setContextClause(pragmaClause, l);
                }
            }
        }

        // If nested task-offload is contained, convert to
        // 'acc data' with copyXX/create clause.
        // If not, convert to 'acc parallel' with all clause
        // (Include delayed clauses).
        XobjList acc = null;
        if (containsNestedTaskOffload(xobj)) {
            acc = createAccPragma(ACCpragma.DATA,
                                  accDataClauses, xobj, 2);
        } else {
            accClauses.mergeList(accDataClauses);
            accClauses.mergeList(getContextClauses());

            acc = createAccPragma(ACCpragma.PARALLEL,
                                  accClauses, xobj, 2);
            resetContextClauses();
        }
        currentArgs.setArg(acc);
    }
}
