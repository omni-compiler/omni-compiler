package xcodeml.c.type;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;

/**
 * type of function
 */
public final class XcFuncType extends XcExtType
{
    /* function prototype */
    private boolean _isPrototype;
    
    /* type qualifier: inline */
    private boolean _isInline;

    /* type qualifier: static */
    private boolean _isStatic;

    /* pre declaration function */
    private boolean _isPreDeclaration;
    
    /* types of parameters */
    private XcParamList _paramList = new XcParamList();

    public boolean isEllipsised()
    {
        return _paramList.isEllipsised();
    }
    
    public void setIsEllipsised(boolean enable)
    {
        _paramList.setIsEllipsised(enable);
    }
    
    public XcFuncType(String typeId)
    {
        super(XcTypeEnum.FUNC, typeId);
    }

    @Override
    public XcFuncType copy()
    {
        try {
            XcFuncType obj = (XcFuncType)clone();
            obj._paramList = (XcParamList)_paramList.copy();
            obj.setIsEllipsised(isEllipsised());
            return obj;
        } catch(CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public final boolean isPrototype()
    {
        return _isPrototype;
    }

    public final void setIsPrototype(boolean enable)
    {
        _isPrototype = enable;
    }

    public final boolean isPreDecl()
    {
        return _isPreDeclaration;
    }

    public final void setIsPreDecl(boolean enable)
    {
        _isPreDeclaration = enable;
    }

    public final boolean isInline()
    {
        return _isInline;
    }

    public final void setIsInline(boolean enable)
    {
        _isInline = enable;
    }

    public final boolean isStatic()
    {
        return _isStatic;
    }
    
    public final void setIsStatic(boolean enable)
    {
        _isStatic = enable;
    }
    
    public final XcParamList getParamList()
    {
        return _paramList;
    }

    public final void setParamList(XcParamList paramList)
    {
        _paramList = paramList;
    }

    public final void addParam(XcIdent param)
    {
        _paramList.add(param);
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcIdent)
            addParam((XcIdent)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
    }

    @Override
    public final XcNode[] getChild()
    {
        if(_paramList.isEmpty())
            return null;

        return _paramList.toArray(new XcNode[_paramList.size()]);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        if(index < _paramList.size())
            _paramList.set(index, (XcIdent)child);
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    protected final void resolveOverride(XcIdentTableStack itStack) throws XmException
    {
        // assume int for no-type function
        if(getTempRefTypeId() == null)
            setRefType(XcBaseTypeEnum.INT.getSingletonType());
        
        _paramList.resolve(itStack);
    }
}
