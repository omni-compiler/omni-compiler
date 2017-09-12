package xcodeml.c.decompile;

import xcodeml.c.obj.XcNode;

/**
 * Top class which represents X-CodeML elements
 */
public abstract class XcObj implements XcAppendable, XcNode
{
    protected static final XcNode[] toNodeArray(XcNode ... nodes)
    {
        if(nodes == null || nodes.length == 0)
            return null;
        
        return nodes;
    }
}
