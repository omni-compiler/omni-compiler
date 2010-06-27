/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

public abstract class XobjectIterator
{
    Xobject currentXobject;
    Xobject topXobject;
    XobjArgs currentArgs;

    public XobjectIterator()
    {
    }

    public abstract void init();

    public abstract void init(Xobject x);

    public abstract void next();

    public abstract boolean end();

    public abstract Xobject getParent();

    public Xobject getXobject()
    {
        return currentXobject;
    }

    public void setXobject(Xobject x)
    {
        currentXobject = x;
        if(currentArgs == null)
            topXobject = x;
        else
            currentArgs.setArg(x);
    }

    public Xobject topXobject()
    {
        return topXobject;
    }
}
