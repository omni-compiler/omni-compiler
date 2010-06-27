/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * type of 'float', 'double', 'long double'
 */
public abstract class XcDecimalType extends XcNumType
{
    public XcDecimalType(XcBaseTypeEnum basicTypeEnum, String typeId)
    {
        super(basicTypeEnum, typeId);
    }
    
    public static final class Float extends XcDecimalType
    {
        public Float()
        {
            this(null);
        }
        
        public Float(String typeId)
        {
            super(XcBaseTypeEnum.FLOAT, typeId);
        }
    };

    public static final class Double extends XcDecimalType
    {
        public Double()
        {
            this(null);
        }
        
        public Double(String typeId)
        {
            super(XcBaseTypeEnum.DOUBLE, typeId);
        }
    };

    public static final class LongDouble extends XcDecimalType
    {
        public LongDouble()
        {
            this(null);
        }
        
        public LongDouble(String typeId)
        {
            super(XcBaseTypeEnum.LONGDOUBLE, typeId);
        }
    };
}
