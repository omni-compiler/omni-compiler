/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
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
