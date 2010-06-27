/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.util.Iterator;

import xcodeml.XmNode;
import xcodeml.XmObj;

/**
 * XmObj converter.
 */
public class XmObjConverter
{
    private XmNodeIteratorFactory factory;
    private XmObjMatcher matcher;
    private XmObjMatchAction action;

    /**
     * Traverses XmObj tree and convert a node if the node match the condition.
     *
     * @param n original XmObj tree.
     * @return XmObj tree converted from n.
     */
    public XmObj convert(XmObj n)
    {
        XmNodeIteratorFactory.setFactory(factory);
        XmObj result = n;
        Iterator<XmNode> it = n.iterator();
        while(it.hasNext()) {
            XmNode current = it.next();
            if(matcher.match((XmObj)current)) {
                result = action.execute((XmObj)current);
            }
        }
        return result;
    }

    /**
     * Sets iterator used with XmObj tree traverse.
     *
     * @param factory provides XmNode iterator.
     */
    public void setFactory(XmNodeIteratorFactory factory)
    {
        this.factory = factory;
    }

    /**
     * Sets XmObj matcher which arise convert action.
     *
     * @param matcher provides condition.
     */
    public void setMatcher(XmObjMatcher matcher)
    {
        this.matcher = matcher;
    }

    /**
     * Sets action convert XmObj.
     *
     * @param action provides action.
     */
    public void setAction(XmObjMatchAction action)
    {
        this.action = action;
    }
}
