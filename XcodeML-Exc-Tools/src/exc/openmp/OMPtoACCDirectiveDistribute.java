package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACCDirectiveDistribute extends OMPtoACCDirective {
    public OMPtoACCDirectiveDistribute() {
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

        XobjList ompThreadLimitClause = null;
        XobjList ompNumThreadsClause = null;

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
            case DATA_PRIVATE:
            case DATA_FIRSTPRIVATE:
                l = clauseConverters.get(pragmaClause).convert(xobj, clause);
                break;
            case DATA_LASTPRIVATE:
            case COLLAPSE:
            case DIST_SCHEDULE:
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
                accClauses.add(l);
            }
        }

        // If 'thread_limit()' and 'num_threads()' are specified together,
        // 'num_threads()' will take precedence.
        if (ompThreadLimitClause != null && ompNumThreadsClause != null) {
            accClauses.add(ompNumThreadsClause);
        } else if (ompNumThreadsClause != null) {
            accClauses.add(ompNumThreadsClause);
        } else if (ompThreadLimitClause != null) {
            accClauses.add(ompThreadLimitClause);
        }

        currentArgs.setArg(createAccPragma(ACCpragma.PARALLEL_LOOP,
                                           accClauses, xobj, 2));
    }
}
