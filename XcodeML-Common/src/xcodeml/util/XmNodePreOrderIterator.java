/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.util.Iterator;
import java.util.Stack;

import xcodeml.XmNode;


/**
 * XmNode iterator traverse XmNode tree in pre order.
 */
public class XmNodePreOrderIterator implements Iterator<XmNode>
{
    Stack<XmNode> stack;

    /**
     * Creates an iterator.
     *
     * @param n tree traverse start from this node
     */
    public XmNodePreOrderIterator(XmNode n)
    {
        stack = new Stack<XmNode>();
        if(n != null) {
            stack.push(n);
        }
    }

    @Override
    public boolean hasNext()
    {
        return !stack.empty();
    }

    @Override
    public XmNode next()
    {
        if(stack.empty()) {
            return null;
        } else {
            XmNode n = stack.pop();
            XmNode child[] = n.getChildren();
            for(int i = child.length - 1; i >= 0; --i) {
                stack.push(child[i]);
            }
            return n;
        }
    }

    @Override
    public void remove()
    {
        throw new UnsupportedOperationException();
    }
}
