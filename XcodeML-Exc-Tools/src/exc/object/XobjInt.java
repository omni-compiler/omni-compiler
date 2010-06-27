/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

public class XobjInt extends XobjConst
{
    int value;

    public XobjInt(Xcode code, Xtype type, int value, String fkind)
    {
        super(code, type, fkind);
        this.value = value;
    }

    public XobjInt(Xcode code, int value)
    {
        this(code, Xtype.intType, value, null);
    }

    @Override
    public int getInt()
    {
        return value;
    }

    @Override
    public Xobject copy()
    {
        return copyTo(new XobjInt(code, type, value, getFkind()));
    }

    @Override
    public boolean equals(Xobject x)
    {
        return super.equals(x) && value == x.getInt();
    }

    @Override
    public String getName()
    {
        if(code == Xcode.REG)
            return "r_" + Long.toHexString(value);
        else
            return Long.toString(value);
    }
    
    @Override
    public String toString()
    {
        return "(" + OpcodeName() + " 0x" + Long.toHexString(value) + ")";
    }
}
