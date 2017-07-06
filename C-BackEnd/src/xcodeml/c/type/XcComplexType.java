package xcodeml.c.type;

/**
 * type of '_Complex'
 */
public abstract class XcComplexType extends XcNumType
{
    public XcComplexType(XcBaseTypeEnum basicTypeEnum, String typeId)
    {
        super(basicTypeEnum, typeId);
    }

    public static final class FloatComplex extends XcComplexType
    {
        public FloatComplex()
        {
            this(null);
        }
        
        public FloatComplex(String typeId)
        {
            super(XcBaseTypeEnum.FLOAT_COMPLEX, typeId);
        }
    }

    public static final class DoubleComplex extends XcComplexType
    {
        public DoubleComplex()
        {
            this(null);
        }
        
        public DoubleComplex(String typeId)
        {
            super(XcBaseTypeEnum.DOUBLE_COMPLEX, typeId);
        }
    }

    public static final class LongDoubleComplex extends XcComplexType
    {
        public LongDoubleComplex()
        {
            this(null);
        }
        
        public LongDoubleComplex(String typeId)
        {
            super(XcBaseTypeEnum.LONGDOUBLE_COMPLEX, typeId);
        }
    }
}
