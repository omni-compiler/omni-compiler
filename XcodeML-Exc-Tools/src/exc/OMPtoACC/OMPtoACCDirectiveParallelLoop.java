package exc.OMPtoACC;

import exc.object.*;
import exc.openmp.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACCDirectiveParallelLoop extends OMPtoACCDirective {
    public OMPtoACCDirectiveParallelLoop() {
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
            case DATA_PRIVATE:
            case DATA_FIRSTPRIVATE:
            case DIR_NUM_THREADS:
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
            case DATA_DEFAULT:
            case DATA_SHARED:
            case DATA_COPYIN:
            case PROC_BIND:
            case DATA_LASTPRIVATE:
            case DATA_LINEAR:
            case DIR_SCHEDULE:
            case COLLAPSE:
            case DIR_ORDERED:
            case DIR_NOWAIT:
              l = clauseConverters.get(pragmaClause).convert(xobj, clause);
                break;
            case DIR_IF:
                l = clauseConverters.get(pragmaClause).
                    convert(xobj, clause,
                            new OMPpragma[]{},
                            new OMPpragma[]{OMPpragma.PARALLEL_FOR});
                break;
            default:
              OMP.error((LineNo)xobj.getLineNo(),"clause cannot be specified");
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
