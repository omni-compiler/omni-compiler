package xcodeml.c.type;

/**
 * type of 'void'
 */
public final class XcVoidType extends XcBaseType
{
    public XcVoidType()
    {
        this(null);
    }
    
    public XcVoidType(String typeId)
    {
        super(XcBaseTypeEnum.VOID, typeId);
    }
}
