package exc.openmp;

import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

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
        for (Iterator<Xobject> it = ((XobjList) mapValues.getArg(0)).iterator(); it.hasNext();) {
            Xobject value = (Xobject) it.next();
            if (value.Opcode() == Xcode.LIST &&
                ((XobjList) value).isEmpty()) {
                ((XobjList) mapValues.getArg(0)).remove(value);
            }
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
