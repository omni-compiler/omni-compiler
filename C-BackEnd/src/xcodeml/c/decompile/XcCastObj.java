package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents castExpr.
 */
public class XcCastObj extends XcObj implements XcExprObj
{
    private XcType _type;

    private XcExprObj _expr;

    private boolean _isGccExtension;

    /**
     * Creates a XcCastObj.
     */
    public XcCastObj()
    {
    }

    /**
     * Creates a XcCastObj.
     * 
     * @param type the operator object casts the term to.
     */
    public XcCastObj(XcType type)
    {
        _type = type;
    }
    
    /**
     * Creates a XcCastObj.
     * 
     * @param expr the term of the operator object.
     * @param type the operator object casts the term to.
     */
    public XcCastObj(XcType type, XcExprObj expr)
    {
        _type = type;
        _expr = expr;
    }

    /**
     * Sets expression which casted a type by the object.
     * 
     * @param expr the term of the operator object.
     */
    public final void setExpr(XcExprObj expr)
    {
        _expr = expr;
    }

    /**
     * Gets if is the operator object used with __extension__. 
     * 
     * @param enable true if the object is used with __extension__.
     */
    public final void setIsGccExtension(boolean enable)
    {
        _isGccExtension = enable;
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            setExpr((XcExprObj)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
        if(_expr == null)
            throw new IllegalArgumentException("no expression");
    }

    @Override
    public XcNode[] getChild()
    {
        if(_expr == null) {
            if(_type == null)
                return null;
            else
                return toNodeArray(_type);
        } else {
            if(_type == null)
                return toNodeArray(_expr);
            else
                return toNodeArray(_expr, _type);
        }
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _expr = (XcExprObj)child;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        boolean isValueBrace = true;
        if(_expr instanceof XcCompoundValueObj || _expr instanceof XcDesignatedValueObj)
            isValueBrace = false;

        if(_isGccExtension)
            w.addSpc("__extension__");

        w.addBrace(_type);
        if(isValueBrace) w.add("(");
        w.add(_expr);
        if(isValueBrace) w.add(")");
    }
}
