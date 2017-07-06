package xcodeml.c.decompile;

/**
 * Implementing this interface allows an expression object
 * to be a parent of XcXmpCoArrayRefObj which is able to be turn over by XcXmpCoArrayRefObj.
 */
public interface XcXmpCoArrayParent
{
    /**
     * Sets the content of coArrayRef operation to the child of the object.
     * 
     * @param expr the content of coArrayRef operation and is to be the child of the object. 
     */
    public void setCoArrayContent(XcExprObj expr);
}
