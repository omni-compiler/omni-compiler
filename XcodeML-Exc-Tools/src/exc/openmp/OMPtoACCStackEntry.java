package exc.openmp;

import exc.object.*;

public class OMPtoACCStackEntry {
    private OMPpragma pragma;
    private Xobject xobj;

    public OMPtoACCStackEntry(OMPpragma pragma) {
        this.pragma = pragma;
    }

    public OMPtoACCStackEntry(OMPpragma pragma,
                              Xobject xobj) {
        this.pragma = pragma;
        this.xobj = xobj;
    }

    public OMPpragma getPragma() {
        return pragma;
    }

    public Xobject getXobj() {
        return xobj;
    }

    @Override
    public String toString(){
        return "StackEntry(pragma=" + pragma +
            ", xobj =" + xobj + ")";
    }
}
