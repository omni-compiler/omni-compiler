/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * type of 'struct'
 */
public final class XcStructType extends XcCompositeType
{
    public XcStructType(String typeId)
    {
        super(XcTypeEnum.STRUCT, typeId);
    }

    public XcStructType(String typeId, String tagName)
    {
        super(XcTypeEnum.STRUCT, typeId, tagName);
    }

    @Override
    public final String getTypeNameHeader()
    {
        return "struct";
    }
}
