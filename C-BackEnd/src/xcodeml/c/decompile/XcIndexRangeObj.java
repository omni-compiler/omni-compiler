package xcodeml.c.decompile;

import java.util.List;
import java.util.ArrayList;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

public class XcIndexRangeObj extends XcObj implements XcExprObj
{
    private XcExprObj _lowerBound;
    private XcExprObj _upperBound;
    private XcExprObj _step;

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
	w.add(_lowerBound).add(":")
         .add(_upperBound).add(":").add(_step);
    }

    @Override
    public void addChild(XcNode child)
    {
        if (child instanceof UpperBound){
            _upperBound = ((UpperBound)child).getExpr();
        }
	else if(child instanceof LowerBound){
            _lowerBound = ((LowerBound)child).getExpr();
        }
	else if(child instanceof Step){
            _step = ((Step)child).getExpr();
        }
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
        return new XcNode[] {_lowerBound, _upperBound, _step};
    }

    @Override
    public void setChild(int index, XcNode child)
    {
        if ((child instanceof XcExprObj) == false)
            throw new IllegalArgumentException(child.getClass().getName());

        XcExprObj expr = (XcExprObj)child;

        switch (index){
        case 0:
            _lowerBound = expr;
            break;
        case 1:
            _upperBound = expr;
            break;
        case 2:
            _step = expr;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    /**
     * Sets a lowerBound of the operator.
     * 
     * @param lowerBound a lowerBound of the operator.
     */
    public void setLowerBound(XcExprObj lowerBound)
    {
        _lowerBound = lowerBound;
    }

    /**
     * Gets a lowerBound of the operator.
     * 
     * @return a lowerBound of the operator.
     */
    public XcExprObj getLowerBound()
    {
        return _lowerBound;
    }

    /**
     * Sets an upperBound of the operator.
     * 
     * @param upperBound an upperBound of the operator.
     */
    public void setUpperBound(XcExprObj upperBound)
    {
        _upperBound = upperBound;
    }

    /**
     * Gets an upperBound of the operator.
     * 
     * @return an upperBound of the operator.
     */
    public XcExprObj getUpperBound()
    {
        return _upperBound;
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

    /**
     * Internal object represents upperBound,lowerBound,step.
     */
    private static class Index extends XcObj
    {
        private XcExprObj _expr;

        /**
         * get sub array index expression object.
         */
        public XcExprObj getExpr()
        {
            return _expr;
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                _expr = (XcExprObj)child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public void checkChild()
        {
            if(_expr == null)
                throw new IllegalArgumentException();
        }

        @Override
        public XcNode[] getChild()
        {
            return new XcNode[] {_expr};
        }

        @Override
        public void setChild(int index, XcNode child)
        {
            if((child instanceof XcExprObj) == false)
                throw new IllegalArgumentException(child.getClass().getName());

            if(index != 0)
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());

            _expr = (XcExprObj)child;
        }

        @Override
        public void appendCode(XmcWriter w) throws XmException
        {
            w.add(_expr);
        }
    }

    /**
     * Internal object represents subArrayRef.upperBound.
     */
    public static class UpperBound extends Index
    {
    }

    /**
     * Internal object represents subArrayRef.lowerBound.
     */
    public static class LowerBound extends Index
    {
    }

    /**
     * Internal object represents subArrayRef.step.
     */
    public static class Step extends Index
    {
    }

}
