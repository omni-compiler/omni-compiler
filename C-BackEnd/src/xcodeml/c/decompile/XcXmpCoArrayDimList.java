package xcodeml.c.decompile;

import java.util.List;
import java.util.ArrayList;
import xcodeml.c.obj.XcNode;

/**
 * Internal object represents coarray dimension.
 */
public class XcXmpCoArrayDimList implements XcNode
{
    List<XcExprObj> _exprList = new ArrayList<XcExprObj>();

    /**
     * Adds a coarray dimension. 
     * 
     * @param expr a coarray dimension.
     */
    public void add(XcExprObj expr)
    {
        _exprList.add(expr);
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _exprList.add((XcExprObj)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return _exprList.toArray(new XcNode[_exprList.size()]);
    }

    @Override
    public void setChild(int index, XcNode child)
    {
        if((child instanceof XcExprObj) == false)
            throw new IllegalArgumentException(child.getClass().getName());

        if(index >= 0 && index < _exprList.size())
            _exprList.set(index, (XcExprObj)child);
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }
}
