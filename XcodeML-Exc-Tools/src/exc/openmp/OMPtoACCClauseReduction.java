package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;

public class OMPtoACCClauseReduction extends OMPtoACCClause {
    public OMPtoACCClauseReduction() {
        super();
    }

    @Override
    public XobjList convert(Xobject xobj,
                            XobjList clause) {
        if (clause.Nargs() != 2) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
            return null;
        }

        ACCpragma acc = null;

        switch (OMPpragma.valueOf(clause.getArg(0))) {
        case DATA_REDUCTION_PLUS:
            acc = ACCpragma.REDUCTION_PLUS;
            break;
        case DATA_REDUCTION_MUL:
            acc = ACCpragma.REDUCTION_MUL;
            break;
        case DATA_REDUCTION_LOGAND:
            acc = ACCpragma.REDUCTION_LOGAND;
            break;
        case DATA_REDUCTION_LOGOR:
            acc = ACCpragma.REDUCTION_LOGOR;
            break;
        case DATA_REDUCTION_MIN:
            acc = ACCpragma.REDUCTION_MIN;
            break;
        case DATA_REDUCTION_MAX:
            acc = ACCpragma.REDUCTION_MAX;
            break;
        case DATA_REDUCTION_BITAND:
            acc = ACCpragma.REDUCTION_BITAND;
            break;
        case DATA_REDUCTION_BITOR:
            acc = ACCpragma.REDUCTION_BITOR;
            break;
        case DATA_REDUCTION_BITXOR:
            acc = ACCpragma.REDUCTION_BITXOR;
            break;
        case DATA_REDUCTION_MINUS:
            OMP.error((LineNo)xobj.getLineNo(),
                      "'reduction-identifier' of MINUS(-) cannot be specified.");
            return null;
        default:
            OMP.error((LineNo)xobj.getLineNo(),
                      "Not found 'reduction-identifier'.");
            return null;
        }
        return Xcons.List(Xcons.String(acc.toString()),
                          clause.getArg(1));
    }
}
