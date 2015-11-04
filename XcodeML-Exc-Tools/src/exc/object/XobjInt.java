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
    public boolean canGetInt()
    {
        return true;
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

    @Override
    public boolean isZeroConstant() {
      if (value == 0) {
        return true;
      } else {
        return false;
      }
    }

    @Override
    public boolean isOneConstant() {
      if (value == 1) {
        return true;
      } else {
        return false;
      }
    }
}
