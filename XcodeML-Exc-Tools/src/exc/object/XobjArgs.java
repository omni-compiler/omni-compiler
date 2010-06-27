/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * Iterator for the list of Xobject, which is used to represent the argment
 * list.
 */
public class XobjArgs
{
    Xobject arg;
    XobjArgs next;

    public XobjArgs(Xobject a, XobjArgs next)
    {
        this.arg = a;
        this.next = next;
    }

    public Xobject getArg()
    {
        return arg;
    }

    public void setArg(Xobject x)
    { // rewrite
        arg = x;
    }

    public XobjArgs nextArgs()
    {
        return next;
    }

    public void setNext(XobjArgs n)
    {
        next = n;
    }

    public static XobjArgs cons(Xobject x, XobjArgs a)
    {
        return new XobjArgs(x, a);
    }
    
    @Override
    public String toString()
    {
        return arg != null ? arg.toString() : "()";
    }
}
