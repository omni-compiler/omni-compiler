package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;

public class OMPtoACCClauseThreadLimit extends OMPtoACCClause {
    public OMPtoACCClauseThreadLimit() {
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

        return Xcons.List(Xcons.String(ACCpragma.VECT_LEN.toString()),
                          clause.getArg(1));
    }
}
