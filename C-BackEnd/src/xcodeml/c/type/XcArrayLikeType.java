package xcodeml.c.type;

import java.util.Set;
import java.util.HashSet;

import xcodeml.util.XmException;
import xcodeml.c.decompile.XcConstObj;
import xcodeml.c.decompile.XcExprObj;
import xcodeml.c.obj.XcNode;

/**
 * Abstract class for type of array or coArray.
 */
public abstract class XcArrayLikeType extends XcExtType implements XcLazyEvalType
{
    /* array size */
    private int                   _arraySize;

    /* array size of variable length array */
    private XcExprObj             _arraySizeExpr;
    
    private boolean               _isArraySize = false;

    private boolean               _isArraySizeExpr = false;

    private boolean               _isLazyEvalType = false;

    // private IRVisitable[]         _arraySizeBindings;

    private org.w3c.dom.Node[]    _arraySizeBindingNodes;

    private Set<String>           _dependVariable = new HashSet<String>();
    
    public XcArrayLikeType(XcTypeEnum typeEnum, String typeId)
    {
        super(typeEnum, typeId);
    }

    /**
     * Gets array size.
     * @return array size
     */
    public final int getArraySize()
    {
        return _arraySize;
    }

    /**
     * Sets array size.
     * @param len array size
     */
    public final void setArraySize(int len)
    {
        _isArraySizeExpr = false;
        _isArraySize = true;
        _arraySize = len;
    }

    /**
     * Gets array size expression.
     * @return array size expression
     */
    public XcExprObj getArraySizeExpr()
    {
        return _arraySizeExpr;
    }

    /**
     * Gets array size expression,
     * but it has array size as integer,
     * then creates and return array size as expression.
     *
     * @return array size expression
     */
    public XcExprObj getArraySizeAsExpr()
    {
        if(_isArraySizeExpr)
            return _arraySizeExpr;
        else if(_isArraySize)
            return new XcConstObj.IntConst(_arraySize, XcBaseTypeEnum.INT);
        else
            return null;
    }

    /**
     * Sets array size expression.
     * @param expr array size expression
     */
    public final void setArraySizeExpr(XcExprObj expr)
    {
        _isArraySizeExpr = true;
        _isArraySize = false;
        _arraySizeExpr = expr;
    }

    public final boolean isArraySize()
    {
        return _isArraySize;
    }
    
    public final void setIsArraySize(boolean enable)
    {
        _isArraySize = enable;
    }

    public boolean isArraySizeExpr()
    {
        return _isArraySizeExpr;
    }

    public final void setIsArraySizeExpr(boolean enable)
    {
        _isArraySizeExpr = enable;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            setArraySizeExpr((XcExprObj)child);
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
        XcGccAttributeList _gccAttrs = getGccAttribute();

        if(_arraySizeExpr == null) {
            if(_gccAttrs == null)
                return null;
            else
                return toNodeArray(_gccAttrs);
        } else {
            if(_gccAttrs == null)
                return toNodeArray(_arraySizeExpr);
            else
                return toNodeArray(_arraySizeExpr, _gccAttrs);
        }
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _arraySizeExpr = (XcExprObj)child;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    @Override
    protected final void resolveOverride(XcIdentTableStack itStack) throws XmException
    {
    }

    public final boolean isLazyEvalType()
    {
        return _isLazyEvalType;
    }

    @Override
    public void setIsLazyEvalType(boolean enable) {
        _isLazyEvalType = enable;
    }

    // @Override
    // public void setLazyBindings(IRVisitable[] bindings)
    // {
    //     _arraySizeBindings = bindings;
    // }

    @Override
    public void setLazyBindings(org.w3c.dom.Node[] nodes) {
        _arraySizeBindingNodes = nodes;
    }

    // @Override
    // public IRVisitable[] getLazyBindings() {
    //     return _arraySizeBindings;
    // }

    @Override
    public org.w3c.dom.Node[] getLazyBindingNodes() {
        return _arraySizeBindingNodes;
    }

    @Override
    public Set<String> getDependVar()
    {
        return _dependVariable;
    }
}
