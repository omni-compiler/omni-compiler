/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;
import exc.block.Block;

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
    
    public boolean hasNullArg()
    {
        if (arg == null)
            return true;
        if (next == null)
            return false;
        return next.hasNullArg();
    }

    public XobjArgs cfold(Block block)
    {
        Xobject arg2 = (arg == null) ? null : arg.cfold(block);
        XobjArgs next2 = (next == null) ? null : next.cfold(block);
        return new XobjArgs(arg2, next2);
    }

    @Override
    public String toString()
    {
        return arg != null ? arg.toString() : "()";
    }
}
