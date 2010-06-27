/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.math.BigDecimal;

import xcodeml.XmException;

/**
 * Represents float/double/long double constant.
 */
public class XobjFloat extends XobjConst
{
    private BigDecimal value;
    
    private String float_str;
    
    public XobjFloat(Xcode code, Xtype type, String float_str, BigDecimal value, String fkind)
    {
        super(code, type, fkind);
        this.float_str = float_str;
        this.value = value;
    }
    
    public XobjFloat(Xcode code, Xtype type, String float_str, String fkind)
    {
        this(code, type, float_str, null, fkind);
    }

    public XobjFloat(Xcode code, Xtype type, String float_str)
    {
        this(code, type, float_str, null);
    }
    
    public XobjFloat(Xcode code, Xtype type, double d)
    {
        this(code, type, null, new BigDecimal(d), null);
    }
    
    public XobjFloat(double d) throws XmException
    {
        this(Xcode.FLOAT_CONSTANT, Xtype.doubleType, d);
    }
    
    @Override
    public double getFloat()
    {
        return value.doubleValue();
    }
    
    @Override
    public String getFloatString()
    {
        if(float_str == null)
            return value.toEngineeringString();
        return float_str;
    }
    
    @Override
    public Xobject copy()
    {
        return copyTo(new XobjFloat(code, type, float_str, value, getFkind()));
    }

    @Override
    public String toString()
    {
        return "(" + OpcodeName() + " " + getFloatString() + ")";
    }
}
