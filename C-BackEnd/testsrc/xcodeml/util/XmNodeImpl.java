/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.util.Iterator;
import java.util.ArrayList;

import xcodeml.XmNode;

public class XmNodeImpl implements XmNode
{
    private String name;
    private ArrayList<XmNodeImpl> children;
    private XmNodeImpl parent;

    public XmNodeImpl(String name)
    {
        this.name = name;
        children = new ArrayList<XmNodeImpl>();
        parent = null;
    }

    public String getName()
    {
        return name;
    }

    public void addChild(XmNode n)
    {
        n.setParent(this);
        children.add((XmNodeImpl)n);
    }

    @Override
    public XmNode[] getChildren()
    {
        XmNode[] a = new XmNode[children.size()];
        return (XmNode[])children.toArray(a);
    }

    @Override
    public XmNode getParent()
    {
        return parent;
    }

    @Override
    public void setParent(XmNode n)
    {
        this.parent = (XmNodeImpl)n;
    }

    @Override
    public Iterator<XmNode> iterator()
    {
        return new XmNodePreOrderIterator(this);
    }

}
