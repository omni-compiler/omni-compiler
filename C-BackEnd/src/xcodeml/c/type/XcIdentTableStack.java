package xcodeml.c.type;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;
import xcodeml.util.XmException;
import xcodeml.c.util.XcLazyVisitor;

/**
 * represents stack of symbol table
 */
public final class XcIdentTableStack
{
    /* ident table */
    private Stack<XcIdentTable>     _stack = new Stack<XcIdentTable>();

    /* type table */
    private Map<String, XcType>     _typeMap = new HashMap<String, XcType>();

    /* pre-resolved type list */
    private XcTypeList              _preResolvedTypeList = new XcTypeList();

    public XcIdentTableStack()
    {
    }

    public final XcIdentTable getLast()
    {
        return _stack.peek();
    }

    public final XcIdentTable push()
    {
        XcTagAndDefSet definedTypeSet = null;

        if(!_stack.isEmpty()) {
            definedTypeSet = getLast().getDefinedTypeSet();
        }

        XcIdentTable it = new XcIdentTable();

        it.copyTypeSet(definedTypeSet);

        _stack.push(it);
        return getLast();
    }

    public final void pop()
    {
        _stack.pop();
    }

    private XcIdentTableEnum _getIdentTableEnum(XcSymbolKindEnum kind)
    {
        if(kind == null)
            throw new IllegalArgumentException();

        switch(kind) {
        case FUNC:
        case VAR:
        case MOE:
        case TYPE:
            return XcIdentTableEnum.MAIN;
        case TAGNAME:
            return XcIdentTableEnum.TAGNAME;
        case LABEL:
            return XcIdentTableEnum.LABEL;
        case ANONYM:
            return XcIdentTableEnum.ANONYM;
        default:
            /* not reachable */
            throw new IllegalArgumentException(kind.toString());
        }
    }

    public void addIdent(XcSymbolKindEnum kind, XcIdent ident) throws XmException
    {
        if(ident == null)
            throw new IllegalArgumentException();
        if(_stack.isEmpty())
            throw new IllegalStateException();

        XcIdentTable it = getLast();
        XcIdentTableEnum itEnum = _getIdentTableEnum(kind);

        if(it.containsIdent(itEnum, ident.getSymbol())) {
            throw new XmException("redeclaration of '" + ident.getSymbol() + "'");
        }

        ident.setSymbolKindEnum(kind);
        it.add(itEnum, ident);

        if(kind == XcSymbolKindEnum.TAGNAME) {
            ((XcTaggedType)ident.getType()).setTagName(ident.getSymbol());
        }
    }

    public void addAnonIdent(XcIdent ident) throws XmException
    {
        if(ident == null)
            throw new IllegalArgumentException();

        if(_stack.isEmpty())
            throw new IllegalStateException();

        XcIdentTable it = getLast();

        it.addAnonIdent(ident);
    }

    public void resolveDependency(XcLazyVisitor visitor) throws XmException
    {
        //_lazyListEval();

        XcIdentTable it = getLast();
        it.resolveDependency(visitor);
    }

    public XcIdent getIdent(XcSymbolKindEnum kind, String symbol)
    {
        if(symbol == null)
            throw new IllegalArgumentException();
        if(_stack.isEmpty())
            throw new IllegalStateException();

        XcIdentTableEnum itEnum = _getIdentTableEnum(kind);

        XcIdent ident = null;

        for(XcIdentTable itTable : _stack) {
            XcIdent tmpIdent = itTable.getIdent(itEnum, symbol);
            if(tmpIdent != null)
                ident = tmpIdent;
        }

        return ident;
    }

    public void addType(XcType type) throws XmException
    {
        if(type == null)
            throw new IllegalArgumentException();

        if(_stack.isEmpty())
            throw new IllegalStateException();

        String typeId = type.getTypeId();

        if(_typeMap.containsKey(typeId)) {
            throw new XmException("redeclaration of '" + type.getTypeId() + "'");
        }

        _typeMap.put(type.getTypeId(), type);
        _preResolvedTypeList.add(type);
    }
    
    public XcType getTypeOrNull(String typeId)
    {
        if(typeId == null)
            throw new IllegalArgumentException();
        if(_stack.isEmpty())
            throw new IllegalStateException();

        return _typeMap.get(typeId);
    }

    public boolean contains(String typeId)
    {
        if(XcBaseTypeEnum.isBuiltInType(typeId))
            return true;

        if(_typeMap.containsKey(typeId))
            return true;

        return false;
    }

    public XcType getType(String typeId) throws XmException
    {
        XcType baseType = XcBaseTypeEnum.createTypeByXcode(typeId);

        if(baseType != null)
            return baseType;

        XcType type = getTypeOrNull(typeId);

        if(type == null)
            throw new XmException("type '" + typeId + "' is not found");

        return type;
    }

    public XcType getTypeAs(XcTypeEnum typeEnum, String typeId) throws XmException
    {
        XcType type = getType(typeId);

        if(type.getTypeEnum().equals(typeEnum) == false)
            throw new XmException("type '" + typeId + "' is not " + typeEnum.getDescription());

        return type;
    }

    public void resolveType() throws XmException
    {
        _preResolvedTypeList.resolve(this);
        _preResolvedTypeList.clear();
    }

    public final XcType getRealType(XcType type)
    {
        while(type != null && type instanceof XcBasicType) {
            type = type.getRefType();
        }

        return type;
    }
}
