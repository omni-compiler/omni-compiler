package xcodeml.c.type;

/**
 * type which has a reference type
 */
public abstract class XcExtType extends XcType
{
    public XcExtType(XcTypeEnum typeEnum, String typeId)
    {
        super(typeEnum, typeId);
    }
}
