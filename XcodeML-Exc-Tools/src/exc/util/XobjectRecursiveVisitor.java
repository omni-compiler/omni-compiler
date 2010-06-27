/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.util;

import exc.object.XobjList;
import exc.object.Xobject;
import exc.object.XobjectDef;
import exc.object.XobjectDefEnv;

/**
 * Xobject Visitor which visits objects recursively
 */
public class XobjectRecursiveVisitor implements XobjectVisitor
{
    @Override
    public boolean enter(Xobject v)
    {
        if(v instanceof XobjList) {
            for(Xobject a : (XobjList)v) {
                if(a != null && !a.enter(this))
                    return false;
            }
        }
        
        return true;
    }

    @Override
    public boolean enter(XobjectDef v)
    {
        if((v.getDef() != null) && !v.getDef().enter(this)) {
            return false;
        }
        if(v.hasChildren()) {
            for(XobjectDef d : v.getChildren()) {
                if(!d.enter(this))
                    return false;
            }
        }
        return true;
    }

    @Override
    public boolean enter(XobjectDefEnv v)
    {
        for(XobjectDef def : v) {
            if(def != null && !def.enter(this))
                return false;
        }
        return true;
    }
}
