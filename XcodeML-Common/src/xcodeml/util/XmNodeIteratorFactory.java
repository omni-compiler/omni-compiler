/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.util.Iterator;

import xcodeml.XmNode;

/**
 * The factory interface creates iterator of XmNode.
 */
public abstract class XmNodeIteratorFactory
{
    private static XmNodeIteratorFactory factory = null;

    /**
     * create a XmNode iterator.
     * 
     * @return an iterator
     * @param n node
     */
    public abstract Iterator<XmNode> createXmNodeIterator(XmNode n);

    /**
     * Gets an iterator factory.
     * 
     * @return an iterator factory
     */
    public static XmNodeIteratorFactory getFactory()
    {
        return factory;
    }

    /**
     * Sets an iterator factory.
     * 
     * @param factory a node iterator factory
     */
    public static void setFactory(XmNodeIteratorFactory factory)
    {
        XmNodeIteratorFactory.factory = factory;
    }
}
