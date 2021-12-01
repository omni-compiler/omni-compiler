package exc.openmp;

import exc.object.*;
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
            OMPpragma pragmaClause = OMPpragma.valueOf(clause.getArg(0));
            switch (pragmaClause) {
            case DIR_IF:
                l = clauseConverters.get(pragmaClause).
                    convert(xobj, clause,
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
