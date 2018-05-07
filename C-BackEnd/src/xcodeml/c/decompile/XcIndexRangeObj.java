package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

public class XcIndexRangeObj extends XcObj implements XcExprObj
{
    private XcExprObj _base;
    private XcExprObj _length;
    private XcExprObj _step;

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        w.add(_base).add(":").add(_length).add(":").add(_step);
    }

    @Override
    public void addChild(XcNode child)
    {
        if (child instanceof XcExprObj){
            if(_base == null){
                _base = (XcExprObj)child;
            }else if(_length == null){
                _length = (XcExprObj)child;
            }else if(_step == null){
                _step = (XcExprObj)child;
            }else{
                throw new IllegalArgumentException(child.getClass().getName());
            }
        }else {
            throw new IllegalArgumentException(child.getClass().getName());
        }
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return new XcNode[] {_base, _length, _step};
    }

    @Override
    public void setChild(int index, XcNode child)
    {
        if(child instanceof XcExprObj) {
            switch (index) {
            case 0:
                _base = (XcExprObj) child;
                return;
            case 1:
                _length = (XcExprObj) child;
                return;
            case 2:
                _step = (XcExprObj) child;
                return;
            default:
            }
        }
        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    /**
     * Sets a base of the operator.
     * 
     * @param base a base of the operator.
     */
    public void setBase(XcExprObj base)
    {
        _base = base;
    }

    /**
     * Gets a base of the operator.
     * 
     * @return a base of the operator.
     */
    public XcExprObj getBase()
    {
        return _base;
    }

    /**
     * Sets an length of the operator.
     * 
     * @param length an length of the operator.
     */
    public void setLength(XcExprObj length)
    {
        _length = length;
    }

    /**
     * Gets an length of the operator.
     * 
     * @return an length of the operator.
     */
    public XcExprObj getLength()
    {
        return _length;
    }

    /**
     * Sets a step of the operator.
     * 
     * @param step a step of the operator.
     */
    public void setStep(XcExprObj step)
    {
        _step = step;
    }

    /**
     * Gets a step of the operator.
     * 
     * @return a step of the operator.
     */
    public XcExprObj getStep()
    {
        return _step;
    }
}
