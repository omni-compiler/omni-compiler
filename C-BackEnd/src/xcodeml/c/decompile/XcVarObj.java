package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   Var, Param
 */
public final class XcVarObj extends XcObj implements XcExprObj
{
    private XcIdent _ident;
    
    /**
     * Creates a XcVarObj.
     */
    public XcVarObj()
    {
    }
    
    /**
     * Creates a XcVarObj.
     * 
     * @param a identifier of variable referred by the operator object.
     */
    public XcVarObj(XcIdent ident)
    {
        _ident = ident;
    }
    
    /**
     * Sets a identifier of variable referred by the operator object.
     * 
     * @param ident a identifier of variable.
     */
    public final void setIdent(XcIdent ident)
    {
        _ident = ident;
    }
    
    /**
     * Gets a identifier of variable referred by the operator object.
     * 
     * @return a identifier of variable referred by the operator object.
     */
    public final XcIdent getIdent()
    {
        return _ident;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcIdent)
            _ident = (XcIdent)child;
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
        if(_ident == null)
            throw new IllegalArgumentException("no identifier");
    }

    @Override
    public XcNode[] getChild()
    {
        return toNodeArray(_ident);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _ident = (XcIdent)child;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }
    
    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        w.addSpc(_ident);
    }
}
