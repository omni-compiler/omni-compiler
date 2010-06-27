/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents coArrayAssignExpr.
 */
public class XcXmpCoArrayAssignObj extends XcObj implements XcExprObj
{
    XcExprObj _dst;

    XcExprObj _src;

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        XcExprObj obj = XcXmpFactory.createCoAPutFuncCall(this);

        w.add(obj);
    }

    /**
     * Gets expression object of coaray put destination.
     * 
     * @return a coarray assignment destination.
     */
    public XcExprObj getDst()
    {
        return _dst;
    }

    /**
     * Gets expression object of coaray put source.
     * 
     * @return a coarray assignment source.
     */
    public XcExprObj getSrc()
    {
        return _src;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj) {
            if(_dst == null)
                _dst = (XcExprObj)child;
            else
                _src = (XcExprObj)child;
        } else {
            throw new IllegalArgumentException(child.getClass().getName());
        }
    }

    @Override
    public void checkChild()
    {
        if(_src == null || _dst == null)
            throw new IllegalArgumentException();
    }

    @Override
    public XcNode[] getChild()
    {
        return toNodeArray(_dst, _src);
    }

    @Override
    public void setChild(int index, XcNode child)
    {
        if(index == 0) {
            if(child instanceof XcXmpCoArrayRefObj) {
                _dst = (XcXmpCoArrayRefObj)child;
                return;
            } else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        if(index == 1) {
            if(child instanceof XcExprObj) {
                _src = (XcExprObj)child;
                return;
            } else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }
}