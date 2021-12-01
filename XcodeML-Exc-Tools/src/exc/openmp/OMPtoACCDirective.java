package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;

public class OMPtoACCDirective {
    protected OMPtoACCClause clauseConverter = new OMPtoACCClause();

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

    public void convert(Xobject xobj,
                        XobjArgs currentArgs) {
        throw new UnsupportedOperationException(toString());
    }
}
