package xcodeml.c.decompile;

import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcType;

/**
 * Internal object represents following elements:
 * sizeOfExpr, gccAlignOfExpr
 */
public class XcSizeOfExprObj extends XcOperatorObj
{
    private XcType _typeName;

    /**
     * Gets a term of sizeof/__alignof__ if the term is type.
     * 
     * @return a term of sizeof/__alignof__
     */
    public XcType getTypeName()
    {
        return _typeName;
    }

    /**
     * Sets a single term of sizeof/__alignof__ if the term is type.
     * 
     * @param type a term of sizeof/__alignof__.
     */
    public void setTypeName(XcType type)
    {
        _typeName = type;
    }

    /**
     * Creates a XcSizeOfExprObj.
     * 
     * @param opeEnum an operator code enumerator.
     * @param typeName  a term of sizeof/__alignof__.
     */
    public XcSizeOfExprObj(XcOperatorEnum opeEnum, XcType typeName)
    {
        super(opeEnum);
        _typeName = typeName;
    }

    /**
     * Creates a XcSizeOfExprObj.
     * 
     * @param opeEnum an operator code enumerator.
     * @param expr  a term of sizeof/__alignof__.
     */
    public XcSizeOfExprObj(XcOperatorEnum opeEnum, XcExprObj expr)
    {
        super(opeEnum);
        addChild(expr);
    }

    @Override
    public void checkChild()
    {
        if(super.getExprObjs() == null && _typeName == null)
            throw new IllegalArgumentException("number of expression for the operator is invalid : 0");
    }

    @Override
    public XcNode[] getChild()
    {
        XcExprObj[] exprs = super.getExprObjs();

        if(exprs == null) {
            if(_typeName == null)
                return null;
            else
                return toNodeArray(_typeName);
        } else {
            if(_typeName == null)
                return toNodeArray(exprs[0]);
            else
                return toNodeArray(exprs[0], _typeName);
        }
    }
}
