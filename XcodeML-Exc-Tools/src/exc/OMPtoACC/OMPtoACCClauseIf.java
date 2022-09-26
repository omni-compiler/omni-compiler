package exc.OMPtoACC;

import exc.object.*;
import exc.openmp.*;
import exc.openacc.ACCpragma;

import java.util.Arrays;

public class OMPtoACCClauseIf extends OMPtoACCClause {
    public OMPtoACCClauseIf() {
        super();
    }

    private String ifModifierStr(OMPpragma modifier) {
        switch (modifier) {
        case TARGET_DATA:
            return "target data";
        case TARGET:
            return "target";
        case PARALLEL_FOR:
            return "parallel";
        default:
            return modifier.getName().replace("_", " ");
        }
    }

    @Override
    public XobjList convert(Xobject xobj,
                            XobjList clause,
                            OMPpragma[] modifiers,
                            OMPpragma[] ignoreModifiers) {
        if (clause.Nargs() != 3) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
            return null;
        }

        // check modifier.
        if (!clause.getArg(1).isEmpty()) {
            String modifierStr = clause.getArg(1).getArg(0).getName();

            if (Arrays.asList(ignoreModifiers).
                contains(OMPpragma.valueOf(modifierStr))) {
                OMP.warning((LineNo)xobj.getLineNo(),
                            "modifier('" +
                            ifModifierStr(OMPpragma.valueOf(modifierStr)) +
                            "') cannot be specified. Ignore 'if' clause.");
                return null;
            } else if (!Arrays.asList(modifiers).
                       contains(OMPpragma.valueOf(modifierStr))) {
                OMP.error((LineNo)xobj.getLineNo(),
                          "modifier('" +
                          ifModifierStr(OMPpragma.valueOf(modifierStr)) +
                          "') cannot be specified.");
                return null;
            }
        }

        // create IF().
        return Xcons.List(Xcons.String(ACCpragma.IF.toString()),
                          clause.getArg(2));
    }
}
