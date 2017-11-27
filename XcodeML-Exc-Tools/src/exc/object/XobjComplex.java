package exc.object;

/**
 * represents complex constant.
 */
public class XobjComplex extends XobjList
{
    public XobjComplex(Xcode code, Xtype type, Xobject re, Xobject im)
    {
        super(code, type);
        add(re);
        add(im);
    }
    
    public Xobject getReValue()
    {
        return getArg(0);
    }

    public Xobject getImValue()
    {
        return getArg(1);
    }
}
