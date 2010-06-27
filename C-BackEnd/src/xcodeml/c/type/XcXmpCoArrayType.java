/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * type of 'coarray'
 */
public final class XcXmpCoArrayType extends XcArrayLikeType
{
    public XcXmpCoArrayType(String typeId)
    {
        super(XcTypeEnum.COARRAY, typeId);
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder(128);
        b.append("[");
        commonToString(b);
        b.append("coArraySize=").append(getArraySize()).append("]");
        return b.toString();
    }

    public XcType getElementType()
    {
        XcType refType = getRefType();

        if(refType.getTypeEnum() == XcTypeEnum.COARRAY)
            return ((XcXmpCoArrayType)refType).getElementType();
        else
            return refType;
    }
}
