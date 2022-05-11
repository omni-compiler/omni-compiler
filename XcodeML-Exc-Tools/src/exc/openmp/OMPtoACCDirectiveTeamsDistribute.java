package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACCDirectiveTeamsDistribute extends OMPtoACCDirective {
    public OMPtoACCDirectiveTeamsDistribute() {
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
            OMPpragma pragmaClause = OMPpragma.valueOf(clause.getArg(0));
            switch (pragmaClause) {
            case NUM_TEAMS:
            case THREAD_LIMIT:
            case DATA_PRIVATE:
            case DATA_FIRSTPRIVATE:
            case DATA_REDUCTION_PLUS:
            case DATA_REDUCTION_MINUS:
            case DATA_REDUCTION_MUL:
            case DATA_REDUCTION_LOGAND:
            case DATA_REDUCTION_LOGOR:
            case DATA_REDUCTION_MIN:
            case DATA_REDUCTION_MAX:
            case DATA_REDUCTION_BITAND:
            case DATA_REDUCTION_BITOR:
            case DATA_REDUCTION_BITXOR:
                l = clauseConverters.get(pragmaClause).convert(xobj, clause);
                break;
            case DATA_DEFAULT:
            case DATA_SHARED:
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
                          "Cannot be specified is clause.");
                break;
            }

            if (OMP.hasError()) {
                return;
            }

            if (l != null) {
                setContextClause(pragmaClause, l);
            }
        }

        // Merge delayed clauses.
        accClauses.mergeList(getContextClauses());
        currentArgs.setArg(createAccPragma(ACCpragma.PARALLEL_LOOP,
                                           accClauses, xobj, 2));
        resetContextClauses();
    }
}
