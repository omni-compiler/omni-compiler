package xcodeml.c.type;

/**
 * represent gcc builtin type.
 */
public class XcGccBuiltinVaListType extends XcBaseType
{
    public XcGccBuiltinVaListType()
    {
        this(null);
    }

    public XcGccBuiltinVaListType(String typeId)
    {
        super(XcBaseTypeEnum.GCC_BUILTIN_VA_LIST, typeId);
    }
}
