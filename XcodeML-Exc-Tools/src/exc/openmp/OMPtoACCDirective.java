package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.HashMap;

public class OMPtoACCDirective {
    protected HashMap<OMPpragma, OMPtoACCClause> clauseConverters =
        new HashMap<>() {
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


    public OMPtoACCDirective() {
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
