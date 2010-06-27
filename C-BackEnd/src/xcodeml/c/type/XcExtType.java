/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
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
