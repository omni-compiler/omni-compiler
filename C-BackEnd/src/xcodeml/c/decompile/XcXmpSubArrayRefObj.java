/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import java.util.List;
import java.util.ArrayList;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcBaseTypeEnum;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents subArrayRef.
 */
public class XcXmpSubArrayRefObj extends XcObj implements XcExprObj
{
    private XcExprObj _expr;

    private XcExprObj _upperBound;

    private XcExprObj _lowerBound;

    private XcExprObj _step;

    private XcType _arrayType;

    private XcType _unitType;

    private XcExprObj _arraySizeExpr;

    /**
     * Gets an upperBound of subArrayRef.
     *
     * @return an upper bound expression for sub array.
     */
    public XcExprObj getUpperBound()
    {
        return _upperBound;
    }

    /**
     * Gets a lowerBound of subArrayRef.
     *
     * @return a lower bound expression for sub array.
     */
    public XcExprObj getLowerBound()
    {
        return _lowerBound;
    }

    /**
     * Gets a step of subArrayRef.
     *
     * @return a step expression for sub array.
     */
    public XcExprObj getStep()
    {
        return _step;
    }

    /**
     * Sets an upperBound of subArrayRef.
     *
     * @param upperBound an upper bound expression for sub array. 
     */
    public void setUpperBound(XcExprObj upperbound)
    {
        _upperBound = upperbound;
    }

    /**
     * Sets a lowerBound of subArrayRef.
     *
     * @param lowerBound a lower bound expression for sub array.
     */
    public void setLowerBound(XcExprObj lowerbound)
    {
        _lowerBound = lowerbound;
    }

    /**
     * Sets a step of subArrayRef.
     *
     * @param step a step expression for sub array.
     */
    public void setStep(XcExprObj step)
    {
        _step = step;
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

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        w.add(_expr);
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof UpperBound) {
            _upperBound = ((UpperBound)child).getExpr();
        } else if(child instanceof LowerBound) {
            _lowerBound = ((LowerBound)child).getExpr();
        } else if(child instanceof Step) {
            _step = ((Step)child).getExpr();
        } else if(child instanceof XcExprObj) {
            _expr = (XcExprObj)child;
        } else
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
        List<XcNode> list = new ArrayList<XcNode>();
        if(_expr != null)
            list.add(_expr);
        if(_upperBound != null)
            list.add(_upperBound);
        if(_lowerBound != null)
            list.add(_lowerBound);
        if(_step != null)
            list.add(_step);

        return list.toArray(new XcNode[list.size()]);
    }

    @Override
    public void setChild(int index, XcNode child)
    {
        if((child instanceof XcExprObj) == false)
            throw new IllegalArgumentException(child.getClass().getName());

        XcExprObj expr = (XcExprObj)child;

        switch(index) {
        case 0:
            _expr = expr;
            break;
        case 1:
            _upperBound = expr;
            break;
        case 2:
            _lowerBound = expr;
            break;
        case 3:
            _step = expr;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    /**
     * Gets a type of an array the operation referred.
     * 
     * @return a type of an array the operation referred.
     */
    public XcType getArrayType()
    {
        return _arrayType;
    }

    /**
     * Sets a type of an array  the operation referred.
     * 
     * @param type a type of an array the operation referred.
     */
    public void setArrayType(XcType type)
    {
        _arrayType = type;
    }

    /**
     * Gets a unit type of an array the operation referred.
     * 
     * @return a unit type of an array the operation referred.
     */    
    public XcType getUnitType()
    {
        return _unitType;
    }

    /**
     * Sets a unit type of an array the operation referred.
     * 
     * @param type a unit type of an array the operation referred.
     */    
    public void setUnitType(XcType type)
    {
        _unitType = type;
    }

    /**
     * Sets a size expression of an array the operation referred.
     * 
     * @param arraySizeExpr a size of an array the operation referred.
     */    
    public void setArraySize(XcExprObj arraySizeExpr)
    {
        _arraySizeExpr = arraySizeExpr;
    }

    /**
     * Gets a size expression of an array the operation referred.
     * 
     * @return a size of an array the operation referred.
     */
    public XcExprObj getArraySize()
    {
        return _arraySizeExpr;
    }

    /**
     * Gets an expresion the operation applied to.
     * 
     * @return an expresion the operation applied to.
     */
    public XcExprObj getExpr()
    {
        return _expr;
    }

    /**
     * Makes up for lowerbound/step expression by array type.
     */
    public void makeCompleteInfo()
    {
        if(_lowerBound == null)
            _lowerBound = new XcConstObj.IntConst(0, XcBaseTypeEnum.INT);

        if(_step == null)
            _step = new XcConstObj.IntConst(1, XcBaseTypeEnum.INT);
    }

}
