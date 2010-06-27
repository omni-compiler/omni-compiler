/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * type of '_Imaginary'
 */
public abstract class XcImagType extends XcNumType
{
    public XcImagType(XcBaseTypeEnum basicTypeEnum, String typeId)
    {
        super(basicTypeEnum, typeId);
    }

    public static final class FloatImag extends XcImagType
    {
        public FloatImag()
        {
            this(null);
        }
        
        public FloatImag(String typeId)
        {
            super(XcBaseTypeEnum.FLOAT_IMAG, typeId);
        }
    }

    public static final class DoubleImag extends XcImagType
    {
        public DoubleImag()
        {
            this(null);
        }
        
        public DoubleImag(String typeId)
        {
            super(XcBaseTypeEnum.DOUBLE_IMAG, typeId);
        }
    }

    public static final class LongDoubleImag extends XcImagType
    {
        public LongDoubleImag()
        {
            this(null);
        }
        
        public LongDoubleImag(String typeId)
        {
            super(XcBaseTypeEnum.LONGDOUBLE_IMAG, typeId);
        }
    }
}
