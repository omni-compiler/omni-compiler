/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

/**
 * Implements the interface allows an type object to have XcGccAttribute.
 */
public interface XcGccAttributable
{
    /**
     * Sets gccAttributes.
     *
     * @param attrs list of gccAttribute.
     */
    public void setGccAttribute(XcGccAttributeList attrs);

    /**
     * Gets gccAttributes.
     *
     * @return list of gccAttribute.
     */
    public XcGccAttributeList getGccAttribute();
}