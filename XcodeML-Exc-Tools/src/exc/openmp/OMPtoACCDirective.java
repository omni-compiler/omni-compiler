package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACCDirective {
    private OMPtoACCClause clauseConverter = new OMPtoACCClause();

    public OMPtoACCDirective() {
    }

    private String notImplementedClauseStr(OMPpragma clause){
        switch (clause) {
        case TARGET_DEVICE:
            return "device";
        case USE_DEVICE_PTR:
            return "use_device_ptr";
        case IS_DEVICE_PTR:
            return "is_device_ptr";
        case DEFAULTMAP:
            return "defaultmap";
        case DIR_NOWAIT:
            return "nowait";
        case DEPEND:
            return "depend";
        case DATA_DEFAULT:
            return "default";
        case DATA_SHARED:
            return "shared";
        case DATA_LASTPRIVATE:
            return "lastprivate";
        case COLLAPSE:
            return "collapse";
        case DIST_SCHEDULE:
            return "dist_schedule";
        case DATA_COPYIN:
            return "copyin";
        case PROC_BIND:
            return "proc_bind";
        case DATA_LINEAR:
            return "linear";
        case DIR_SCHEDULE:
            return "schedule";
        case DIR_ORDERED:
            return "ordered";
        }

        return clause.getName();
    }

    private XobjList createAccPragma(ACCpragma directive, XobjList clauses,
                                     Xobject xobj, int addArgsHeadPos) {
        XobjList accPragma = Xcons.List(Xcode.ACC_PRAGMA, xobj.Type());
        accPragma.setLineNo(xobj.getLineNo());
        accPragma.add(Xcons.String(directive.toString()));
        accPragma.add(clauses);

        int pos = addArgsHeadPos;
        Xobject arg = null;
        while ((arg = xobj.getArgOrNull(pos)) != null) {
            accPragma.add(arg);
            pos++;
        }

        return accPragma;
    }

    public void convertFromTargetData(Xobject xobj,
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

    public void convertFromTargetTeamsDistributeParallelLoop(Xobject xobj,
                                                             XobjArgs currentArgs) {
        if (xobj.getArg(1) == null ||
            xobj.getArg(1).Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Not found clause list.");
            return;
        }

        XobjList ompClauses = (XobjList) xobj.getArg(1);
        XobjList gangClause = Xcons.List(Xcons.String(ACCpragma.GANG.toString()));
        XobjList accClauses = Xcons.List(gangClause);

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
            switch (OMPpragma.valueOf(clause.getArg(0))) {
            case TARGET_DATA_MAP:
                l = clauseConverter.convertFromMap(xobj, clause);
                break;
            case DIR_IF:
                l = clauseConverter.convertFromIf(xobj, clause,
                                                  new OMPpragma[]{OMPpragma.TARGET},
                                                  new OMPpragma[]{OMPpragma.PARALLEL_FOR});
                break;
            case NUM_TEAMS:
                l = clauseConverter.convertFromNumTeams(xobj, clause);
                break;
            case DATA_PRIVATE:
                l = clauseConverter.convertFromPrivate(xobj, clause);
                break;
            case DATA_FIRSTPRIVATE:
                l = clauseConverter.convertFromFirstprivate(xobj, clause);
                break;
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
                l = clauseConverter.convertFromReduction(xobj, clause);
                break;
            case THREAD_LIMIT:
                ompThreadLimitClause =
                    clauseConverter.convertFromThreadLimit(xobj, clause);
                break;
            case DIR_NUM_THREADS:
                ompNumThreadsClause =
                    clauseConverter.convertFromNumThreads(xobj, clause);
                break;
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
                OMP.error((LineNo)xobj.getLineNo(),
                          "Not implemented clause. ('" +
                          notImplementedClauseStr(OMPpragma.valueOf(clause.getArg(0))) +
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

    public void convertFromParallelLoop(Xobject xobj,
                                        XobjArgs currentArgs) {
        if (xobj.getArg(1) == null ||
            xobj.getArg(1).Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Not found clause list.");
            return;
        }

        XobjList ompClauses = (XobjList) xobj.getArg(1);
        XobjList gangClause = Xcons.List(Xcons.String(ACCpragma.GANG.toString()));
        XobjList accClauses = Xcons.List(gangClause);

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
            case DIR_IF:
                l = clauseConverter.convertFromIf(xobj, clause,
                                                  new OMPpragma[]{},
                                                  new OMPpragma[]{OMPpragma.PARALLEL_FOR});
                break;
            }

            if (OMP.hasError()) {
                return;
            }

            if (l != null) {
                accClauses.add(l);
            }
        }

        currentArgs.setArg(createAccPragma(ACCpragma.PARALLEL_LOOP,
                                           accClauses, xobj, 2));
    }
}
