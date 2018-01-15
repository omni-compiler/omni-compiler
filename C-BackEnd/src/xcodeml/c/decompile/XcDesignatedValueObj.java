package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents designatedValue.
 */
public class XcDesignatedValueObj extends XcObj implements XcExprObj
{
    private String    _member;

    private XcExprObj _expr;

    /**
     * Sets a name of member.
     *  
     * @param member a name of member.
     */
    public void setMember(String member)
    {
        _member = member;
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _expr = (XcExprObj)child;
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return toNodeArray(_expr);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
    	if(index == 0)
    		_expr = (XcExprObj)child;
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        w.add("{").add(".").add(_member).add(" =").add(_expr).add("}");
    }
}
