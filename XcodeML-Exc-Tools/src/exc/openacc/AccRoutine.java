package exc.openacc;

import exc.object.XobjectDef;

/**
 * Created by tabuchi on 16/05/10.
 */
public class AccRoutine extends AccDirective{
    AccRoutine(ACCglobalDecl decl, AccInformation info, XobjectDef xobjDef) {
        super(decl, info, xobjDef);
    }

    @Override
    void analyze() throws ACCexception{
        super.analyze();

        //XobjectDef def = _def;
        //_decl.getEnv().getDefs().indexOf()
    }

    @Override
    void generate() throws ACCexception {
        ACC.warning("not implemented");
    }

    @Override
    void rewrite() throws ACCexception {
        ACC.warning("not implemented");
    }

    @Override
    boolean isAcceptableClause(ACCpragma clauseKind) {
        switch (clauseKind){
        case ROUTINE_ARG:
        case GANG:
        case WORKER:
        case VECTOR:
        case SEQ:
        case BIND:
        case DEVICE_TYPE:
        case NOHOST:
            return true;
        default:
            return clauseKind.isDataClause();
        }
    }
}
