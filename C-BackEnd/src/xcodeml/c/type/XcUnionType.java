/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * type of 'union'
 */
public final class XcUnionType extends XcCompositeType
{
    public XcUnionType(String typeId)
    {
        super(XcTypeEnum.UNION, typeId);
    }

    @Override
    public final String getTypeNameHeader()
    {
        return "union";
    }
}
