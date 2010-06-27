/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
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
