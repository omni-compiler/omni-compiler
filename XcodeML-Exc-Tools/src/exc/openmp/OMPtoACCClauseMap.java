package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;

public class OMPtoACCClauseMap extends OMPtoACCClause {
    public OMPtoACCClauseMap() {
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
}
