/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
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
