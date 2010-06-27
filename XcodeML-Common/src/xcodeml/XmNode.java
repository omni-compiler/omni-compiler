/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml;

/**
 * The tree interface for XcodeML binding object.
 * Implementing this interface allows an object the tree operation.
 */
public interface XmNode extends Iterable<XmNode>
{
    /**
     * Gets children of the node.
     *
     * @return children of the node.
     */
    public XmNode[] getChildren();

    /**
     * Sets parent of the node.
     *
     * @param n parent of the node.
     */
    public void setParent(XmNode n);


    /**
     * Gets parent of the node.
     *
     * @return parent of the node.
     */
    public XmNode getParent();
}
