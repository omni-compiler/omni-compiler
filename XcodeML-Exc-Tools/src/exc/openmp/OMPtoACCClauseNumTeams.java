package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;

public class OMPtoACCClauseNumTeams extends OMPtoACCClause {
    public OMPtoACCClauseNumTeams() {
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

        return Xcons.List(Xcons.String(ACCpragma.NUM_GANGS.toString()),
                          clause.getArg(1));
    }
}
