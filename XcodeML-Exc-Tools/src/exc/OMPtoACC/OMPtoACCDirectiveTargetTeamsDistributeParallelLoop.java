package exc.OMPtoACC;

import exc.object.*;
import exc.openmp.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACCDirectiveTargetTeamsDistributeParallelLoop extends OMPtoACCDirective {
    public OMPtoACCDirectiveTargetTeamsDistributeParallelLoop() {
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
            case TARGET_DATA_MAP:
            case NUM_TEAMS:
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
            case TARGET_DEVICE:
            case IS_DEVICE_PTR:
            case DEFAULTMAP:
            case DIR_NOWAIT:
            case DEPEND:
            case DATA_DEFAULT:
            case DATA_SHARED:
            case DATA_LASTPRIVATE:
            case COLLAPSE:
            case DIST_SCHEDULE:
            case DATA_COPYIN:
            case PROC_BIND:
            case DATA_LINEAR:
            case DIR_SCHEDULE:
            case DIR_ORDERED:
                l = clauseConverters.get(pragmaClause).convert(xobj, clause);
                break;
            case DIR_IF:
                l = clauseConverters.get(pragmaClause).
                    convert(xobj, clause,
                            new OMPpragma[]{OMPpragma.TARGET},
                            new OMPpragma[]{OMPpragma.PARALLEL_FOR});
                break;
            case THREAD_LIMIT:
                ompThreadLimitClause =
                    clauseConverters.get(pragmaClause).convert(xobj, clause);
                break;
            case DIR_NUM_THREADS:
                ompNumThreadsClause =
                    clauseConverters.get(pragmaClause).convert(xobj, clause);
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

        // If 'thread_limit()' and 'num_threads()' are specified together,
        // 'num_threads()' will take precedence.
        if (ompThreadLimitClause != null && ompNumThreadsClause != null) {
            setContextClause(OMPpragma.DIR_NUM_THREADS, ompNumThreadsClause);
        } else if (ompNumThreadsClause != null) {
            setContextClause(OMPpragma.DIR_NUM_THREADS, ompNumThreadsClause);
        } else if (ompThreadLimitClause != null) {
            setContextClause(OMPpragma.THREAD_LIMIT, ompThreadLimitClause);
        }

        // Merge delayed clauses.
        accClauses.mergeList(accDataClauses);
        accClauses.mergeList(getContextClauses());
        currentArgs.setArg(createAccPragma(ACCpragma.PARALLEL_LOOP,
                                           accClauses, xobj, 2));
        resetContextClauses();
    }
}
