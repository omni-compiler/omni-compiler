/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.obj;

/**
 * Implements this interface allows an object to operated by XcodeML API.
 */
public interface XcNode
{
    /**
     * Adds child to the node.
     *
     * @param child set to the child of the node.
     */
    public void addChild(XcNode child);

    /**
     * Gets children of the node.
     *
     * @return childen of the node.
     */
    public XcNode[] getChild();

    /**
     * Replaces the child of the node.
     *
     * @param index The index of a child be to be replaced.
     * @param child A node replaced to index'th child.
     */
    public void setChild(int index, XcNode child);

    /**
     * Checks is node valid.
     */
    public void checkChild();
}
