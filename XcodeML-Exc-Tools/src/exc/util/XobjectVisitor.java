/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.util;

import exc.object.Xobject;
import exc.object.XobjectDef;
import exc.object.XobjectDefEnv;

/**
 * Xobject visitor interface
 */
public interface XobjectVisitor
{
    public boolean enter(Xobject v);

    public boolean enter(XobjectDef v);

    public boolean enter(XobjectDefEnv v);
}
