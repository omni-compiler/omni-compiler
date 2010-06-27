/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * represents Fortran Logical Constant.
 */
public class XobjBool extends XobjConst
{
    private boolean value;
    
    public XobjBool(Xcode code, Xtype type, boolean value, String fkind)
    {
        super(code, type, fkind);
        this.value = value;
    }
    
    public XobjBool(boolean value)
    {
        this(Xcode.F_LOGICAL_CONSTATNT, null, value, null);
    }
    
    public boolean getBoolValue()
    {
        return value;
    }

    @Override
    public Xobject copy()
    {
        return copyTo(new XobjBool(code, type, value, getFkind()));
    }

    @Override
    public String toString()
    {
        return "(" + OpcodeName() + " " + value + ")";
    }
}
