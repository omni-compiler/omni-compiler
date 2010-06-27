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
 * XmNode iterator traverse XmNode tree in post order.
 */
public class XmNodePostOrderIterator implements Iterator<XmNode>
{
    private Stack<XmNode> stack;

    /**
     * Creates an iterator.
     *
     * @param n tree traverse start from this node
     */
    public XmNodePostOrderIterator(XmNode n)
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
            if(stack.peek() == null) {
                stack.pop();
                return stack.pop();
            } else {
                XmNode n = stack.peek();
                while(n.getChildren().length != 0) {
                    stack.push(null);
                    XmNode child[] = n.getChildren();
                    for(int i = child.length - 1; i >= 0; --i) {
                        stack.push(child[i]);
                    }
                    n = stack.peek();
                }
                return stack.pop();
            }
        }
    }

    @Override
    public void remove()
    {
        throw new UnsupportedOperationException();
    }

}
