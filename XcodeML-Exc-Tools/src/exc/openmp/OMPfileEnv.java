package exc.openmp;

import exc.object.*;
import exc.block.*;

interface OMPfileEnv
{
    public XobjectFile getFile();
    
    public boolean isThreadPrivate(Ident id);

    public Ident findThreadPrivate(Block b, String name);

    public void declThreadPrivate(Xobject x, IVarContainer vc, Xobject args);
}
