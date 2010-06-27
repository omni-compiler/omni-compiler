/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.binding;

/**
 * Concrete classes of the interface are equivalent to XcodeML tag which has the type attribute.
 */
public interface IXbTypedExpr
{
    /**
     * Gets a content of type attribute.
     *
     * @return a content of type attribute.
     */
    public String getType();

    public void setType(String type);
}
