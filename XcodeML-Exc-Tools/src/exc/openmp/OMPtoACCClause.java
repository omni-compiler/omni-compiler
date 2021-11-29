package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.Arrays;

public class OMPtoACCClause {
    public OMPtoACCClause() {
    }

    public XobjList convertFromMap(Xobject xobj,
                                   XobjList clause) {
        if (clause.Nargs() != 2) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
            return null;
        }

        XobjList mapArgs = (XobjList) clause.getArg(1);
        if (mapArgs.Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Clause list does not exist");
            return null;
        }

        XobjString mapType = (XobjString) mapArgs.getArg(0);
        if (mapType.Opcode() != Xcode.STRING) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Map type does not exist.");
            return null;
        }

        // check map-type-modifier.
        if (mapType.getName().split(" ").length > 1) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "'map-type-modifier (" +
                      mapType.getName().split(" ")[0] +
                      ")' is not supported.");
            return null;
        }

        XobjList mapValues = (XobjList) mapArgs.getArg(1);

        // NOTE: OpenMP xcode has an empty list
        //       if the array range specification is omitted.
        //       So, remove the empty list.
        if (mapValues.getArg(0).getArg(1).isEmpty()) {
            mapValues = Xcons.List(mapValues.getArg(0).getArg(0));
        }

        // create COPY()/CREATE().
        XobjList list = null;
        switch (mapType.getName()) {
        case "alloc":
            list = Xcons.List(Xcons.String(ACCpragma.CREATE.toString()),
                              mapValues);
            break;
        case "to":
            list = Xcons.List(Xcons.String(ACCpragma.COPYIN.toString()),
                              mapValues);
            break;
        case "from":
            list = Xcons.List(Xcons.String(ACCpragma.COPYOUT.toString()),
                              mapValues);
            break;
        case "tofrom":
            list = Xcons.List(Xcons.String(ACCpragma.COPY.toString()),
                              mapValues);
            break;
        default:
            OMP.error((LineNo)xobj.getLineNo(),
                      "'" + mapType.getName() + "'" +
                      " cannot be specified for map.");
            return null;
        }
        return list;
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

    public XobjList convertFromIf(Xobject xobj,
                                  XobjList clause,
                                  OMPpragma[] modifiers) {
        return convertFromIf(xobj, clause, modifiers, new OMPpragma[]{});
    }

    public XobjList convertFromIf(Xobject xobj,
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
                            "') cannot be specified. ignore.");
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

    public XobjList convertFromNumTeams(Xobject xobj,
                                        XobjList clause) {
        if (clause.Nargs() != 2) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
            return null;
        }

        return Xcons.List(Xcons.String(ACCpragma.NUM_GANGS.toString()),
                          clause.getArg(1));
    }

    public XobjList convertFromPrivate(Xobject xobj,
                                       XobjList clause) {
        if (clause.Nargs() != 2) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
            return null;
        }

        return Xcons.List(Xcons.String(ACCpragma.PRIVATE.toString()),
                          clause.getArg(1));
    }

    public XobjList convertFromFirstprivate(Xobject xobj,
                                            XobjList clause) {
        if (clause.Nargs() != 2) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
            return null;
        }

        return Xcons.List(Xcons.String(ACCpragma.FIRSTPRIVATE.toString()),
                          clause.getArg(1));
    }

    public XobjList convertFromReduction(Xobject xobj,
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
