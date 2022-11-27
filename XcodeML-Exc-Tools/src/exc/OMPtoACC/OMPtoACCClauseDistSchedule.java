/* -*- Mode: java; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
package exc.OMPtoACC;

import exc.object.*;
import exc.openmp.*;
import exc.openacc.ACCpragma;

public class OMPtoACCClauseDistSchedule extends OMPtoACCClause {
    public OMPtoACCClauseDistSchedule() {
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

        return Xcons.List(Xcons.String(ACCpragma.OMP_DIST_SCHEDULE.toString()),
                          clause.getArg(1));
    }
}
