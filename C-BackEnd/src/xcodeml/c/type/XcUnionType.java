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
