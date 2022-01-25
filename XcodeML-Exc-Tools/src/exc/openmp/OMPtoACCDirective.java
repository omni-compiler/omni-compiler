package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;

public class OMPtoACCDirective {
    protected HashMap<OMPpragma, OMPtoACCClause> clauseConverters =
        new HashMap<OMPpragma, OMPtoACCClause>() {
            {
                put(OMPpragma.DATA_FIRSTPRIVATE, new OMPtoACCClauseFirstprivate());
                put(OMPpragma.DIR_IF, new OMPtoACCClauseIf());
                put(OMPpragma.TARGET_DATA_MAP, new OMPtoACCClauseMap());
                put(OMPpragma.NUM_TEAMS, new OMPtoACCClauseNumTeams());
                put(OMPpragma.DIR_NUM_THREADS, new OMPtoACCClauseNumThreads());
                put(OMPpragma.DATA_PRIVATE, new OMPtoACCClausePrivate());
                put(OMPpragma.DATA_REDUCTION_PLUS, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_MINUS, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_MUL, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_LOGAND, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_LOGOR, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_MIN, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_MAX, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_BITAND, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_BITOR, new OMPtoACCClauseReduction());
                put(OMPpragma.DATA_REDUCTION_BITXOR, new OMPtoACCClauseReduction());
                put(OMPpragma.THREAD_LIMIT, new OMPtoACCClauseThreadLimit());
            }
        };

    protected static HashMap<OMPpragma, XobjList> contextClauses =
        new HashMap<OMPpragma, XobjList>();

    public OMPtoACCDirective() {
    }

    protected void setContextClause(OMPpragma pragma, XobjList list) {
        switch (pragma) {
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
            if (contextClauses.get(pragma) == null) {
                contextClauses.put(pragma, list);
            } else {
                XobjList clauseXobjs = (XobjList) contextClauses.get(pragma).
                    getArgs().nextArgs().getArg();
                XobjList xobjs = (XobjList)list.getArgs().nextArgs().getArg();
                clauseXobjs.mergeList(xobjs);
            }
            break;
        case DIR_NUM_THREADS:
            contextClauses.put(pragma, list);
            contextClauses.remove(OMPpragma.THREAD_LIMIT);
            break;
        case THREAD_LIMIT:
            contextClauses.put(pragma, list);
            contextClauses.remove(OMPpragma.DIR_NUM_THREADS);
        default:
            contextClauses.put(pragma, list);
        }
    }

    protected void resetContextClauses() {
        contextClauses.clear();
    }

    protected XobjList getContextClauses() {
        XobjList list = new XobjList();
        for (Map.Entry<OMPpragma, XobjList> e : contextClauses.entrySet()) {
            list.add(e.getValue());
        }
        return list;
    }

    private boolean containsNestedTaskOffloadInternal(Xobject xobj) {
        if ((xobj == null) || !(xobj instanceof XobjList)) {
            return false;
        }

        if (xobj.Opcode() == Xcode.OMP_PRAGMA) {
            Xobject directive = xobj.left();
            switch (OMPpragma.valueOf(directive)) {
            case TARGET:
            case TARGET_DATA:
            case TARGET_PARALLEL:
            case TARGET_PARALLEL_LOOP:
            case TARGET_TEAMS:
            case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP:
            case TARGET_TEAMS_DISTRIBUTE:
            case DISTRIBUTE_PARALLEL_LOOP:
            case DISTRIBUTE:
            case PARALLEL_FOR:
            case PARALLEL:
            case FOR:
            case TEAMS:
            case TEAMS_DISTRIBUTE:
            case TEAMS_DISTRIBUTE_PARALLEL_LOOP:
                return true;
            }
        }

        for (XobjArgs a = xobj.getArgs(); a != null; a = a.nextArgs()) {
            if (containsNestedTaskOffloadInternal(a.getArg())) {
                return true;
            }
        }

        return false;
    }

    protected boolean containsNestedTaskOffload(Xobject xobj) {
        for (XobjArgs a = xobj.getArgs(); a != null; a = a.nextArgs()) {
            if (containsNestedTaskOffloadInternal(a.getArg())) {
                return true;
            }
        }
        return false;
    }

    protected String notImplementedClauseStr(OMPpragma clause){
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

    protected XobjList createAccPragma(ACCpragma directive, XobjList clauses,
                                       Xobject xobj, Integer addArgsHeadPos) {
        XobjList accPragma = Xcons.List(Xcode.ACC_PRAGMA, xobj.Type());
        accPragma.setLineNo(xobj.getLineNo());
        accPragma.add(Xcons.String(directive.toString()));
        accPragma.add(clauses);

        if (addArgsHeadPos != null) {
            int pos = addArgsHeadPos.intValue();
            Xobject arg = null;
            while ((arg = xobj.getArgOrNull(pos)) != null) {
                accPragma.add(arg);
                pos++;
            }
        }

        return accPragma;
    }

    protected XobjList createAccPragma(ACCpragma directive, XobjList clauses,
                                       Xobject xobj) {
        return createAccPragma(directive, clauses, xobj, null);
    }

    public void convert(Xobject xobj,
                        XobjArgs currentArgs) {
        throw new UnsupportedOperationException(toString());
    }
}
