/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import java.util.Set;
import java.util.HashSet;

import xcodeml.util.XmException;
//import xcodeml.c.binding.gen.IRVisitable;
import xcodeml.c.decompile.XcExprObj;
import xcodeml.c.decompile.XcObj;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;
import xcodeml.c.util.XcLazyVisitor;

/**
 * Internal object represent identifier.
 */
public class XcIdent extends XcObj implements XcExprObj, XcGccAttributable, XcLazyEvalType
{
    private String                _symbol;

    private XcType                _type;

    /* pre-resolved type ID */
    private String                _tempTypeId;

    /*
     * _valueExpr represent one of follows
     *
     * 1) initial  value if indent is variable
     * 2) bitfield value if indent is member of struct
     * 3) initial  value of member of enum
     */
    private XcExprObj             _valueExpr;

    private boolean               _isStatic;

    private boolean               _isAuto;

    private boolean               _isExtern;

    private boolean               _isExternDef;

    private boolean               _isRegister;

    private boolean               _isTypedef;

    private boolean               _isGccExtension;

    private boolean               _isGccThread;

    private boolean               _isBitField;

    private int                   _bitField;

    private boolean               _isBitFieldExpr;

    private XcSymbolKindEnum      _symbolKindEnum;

    private XcVarKindEnum         _varKindEnum;

    private boolean               _isLazyEvalType = false;

    private boolean               _isIncompleteType;

    // private IRVisitable[]         _arraySizeBindings;

    private org.w3c.dom.Node[]    _arraySizeBindingNodes;

    /* GCC attribute */
    private XcGccAttributeList    _gccAttrs;

    /* parent type for composite type member */
    private XcType                _parentType;

    private boolean               _isOutput = false;

    private boolean               _isDeclared = false;

    private Set<String>           _dependVariables = new HashSet<String>();

    public XcIdent(String symbol)
    {
        _symbol = symbol;
    }

    public XcIdent(String symbol, XcType type)
    {
        _symbol = symbol;
        _type   = type;
    }

    public final String getSymbol()
    {
        return _symbol;
    }

    public final void setSymbol(String symbol)
    {
        _symbol = symbol;
    }

    public final XcType getType()
    {
        return _type;
    }

    public final void setType(XcType type)
    {
        _type = type;
    }

    public final String getTempTypeId()
    {
        return _tempTypeId;
    }

    public final void setTempTypeId(String typeId)
    {
        _tempTypeId = typeId;
    }

    public final XcExprObj getValue()
    {
        return _valueExpr;
    }

    public final void setValue(XcExprObj value)
    {
        _valueExpr = value;
    }
    
    public final boolean isStatic()
    {
        return _isStatic;
    }

    public final void setIsStatic(boolean enable)
    {
        _isStatic = enable;
    }
    
    public final boolean isAuto()
    {
        return _isAuto;
    }

    public final void setIsAuto(boolean enable)
    {
        _isAuto = enable;
    }
    
    public final boolean isExtern()
    {
        return _isExtern;
    }

    public final void setIsExtern(boolean enable)
    {
        _isExtern = enable;
    }
    
    public final boolean isExternDef()
    {
        return _isExternDef;
    }

    public final void setIsExternDef(boolean enable)
    {
        _isExternDef = enable;
    }
    
    public final boolean isTypedef()
    {
        return _isTypedef;
    }

    public final void setIsTypedef(boolean enable)
    {
        _isTypedef = enable;
    }
    
    public final boolean isRegister()
    {
        return _isRegister;
    }
    
    public final void setIsRegister(boolean enable)
    {
        _isRegister = enable;
    }
    
    public final boolean isGccExtension()
    {
        return _isGccExtension;
    }
    
    public final void setIsGccExtension(boolean enable)
    {
        _isGccExtension = enable;
    }

    public final boolean isGccThread()
    {
        return _isGccThread;
    }
    
    public final void setIsGccThread(boolean enable)
    {
        _isGccThread = enable;
    }

    public final XcType getParentType()
    {
        return _parentType;
    }
    
    public final void setParentType(XcType type)
    {
        _parentType = type;
    }
    
    public final XcSymbolKindEnum getSymbolKindEnum()
    {
        return _symbolKindEnum;
    }
    
    final void setSymbolKindEnum(XcSymbolKindEnum symbolKindEnum)
    {
        _symbolKindEnum = symbolKindEnum;
    }
    
    public final XcVarKindEnum getVarKindEnum()
    {
        return _varKindEnum;
    }
    
    public final void setVarKindEnum(XcVarKindEnum varKindEnum)
    {
        _varKindEnum = varKindEnum;
    }

    public final void setIsBitField(boolean enable)
    {
        _isBitField = enable;
    }

    public final boolean getIsBitField()
    {
        return _isBitField;
    }
    
    public final void setIsBitFieldExpr(boolean enable)
    {
        _isBitFieldExpr = enable;
    }

    public boolean getIsBitFieldExpr()
    {
        return _isBitFieldExpr;
    }
    
    public final int getBitField()
    {
        return _bitField;
    }
    
    public final void setBitField(int len)
    {
        _isBitField = true;
        _isBitFieldExpr = false;
        _bitField = len;
    }

    public XcExprObj getBitFieldExpr()
    {
        return _valueExpr;
    }

    public final void setBitFieldExpr(XcExprObj expr)
    {
        _isBitField = false;
        _isBitFieldExpr = true;
        _valueExpr = expr;
    }

    public final void appendGccExtension(XmcWriter w) throws XmException
    {
        if(_isGccExtension)
            w.addSpc("__extension__");
    }

    public final void appendGccThread(XmcWriter w) throws XmException
    {
      if(_isGccThread)
        w.addSpc("__thread");
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        w.addSpc(_symbol);
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _valueExpr = (XcExprObj)child;
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
        if(_valueExpr == null) {
            if(_gccAttrs == null)
                return null;
            else
                return toNodeArray(_gccAttrs);
        } else {
            if(_gccAttrs == null)
                return toNodeArray(_valueExpr);
            else
                return toNodeArray(_valueExpr, _gccAttrs);
        }
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _valueExpr = (XcExprObj)child;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    public final void resolve(XcIdentTableStack itStack) throws XmException
    {
        if(_type != null)
            return;

        _type = XcBaseTypeEnum.createTypeByXcode(_tempTypeId);

        if(_type == null)
            _type = itStack.getType(_tempTypeId);

        _tempTypeId = null;
    }

    public final void appendTagOrTypedef(XmcWriter w) throws XmException
    {
        w.noLfOrLf();

        appendGccExtension(w);

        if(_isTypedef) {
            w.addSpc("typedef ");
            _type.appendTypeDefCode(w, _symbol);
        } else {
            _type.appendBodyCode(w, _isIncompleteType);
        }
        w.eos();
    }

    public final void appendFuncDeclCode(XmcWriter w,
        boolean isPreDecl, XcGccAttributeList attrOfFunc) throws XmException
    {

        appendDeclAndAttrCode(w, isPreDecl, attrOfFunc);
    }

    public final void appendDeclCode(XmcWriter w, boolean isPreDecl) throws XmException
    {
        XcType type = _type.getRealType();

        if(type instanceof XcArrayType)
            appendArrayDeclCode(w, isPreDecl);
        else
            appendDeclAndAttrCode(w, isPreDecl, _gccAttrs);
    }

    public final void appendArrayDeclCode(XmcWriter w, boolean isPreDecl) throws XmException
    {
        appendDeclAndAttrCode(w, isPreDecl, null);
        w.add(_gccAttrs);
    }

    public final void appendDeclAndAttrCode(XmcWriter w,
        boolean isPreDecl,
        XcGccAttributeList attrOfDeclOrDef) throws XmException
    {
        /*
          append as follows.

          case if ident of {var, func}Decl
          __extension__ {static,auto,extern,register} __thread (type and symbol and attr)
        */

        appendGccExtension(w);

        if(_isStatic)
            w.addSpc("static");
        else if(_isAuto)
            w.addSpc("auto");
        else if(_isExtern)
            w.addSpc("extern");
        else if(_isRegister)
            w.addSpc("register");

        appendGccThread(w);

        _type.appendDeclCode(w, _symbol, true, isPreDecl, attrOfDeclOrDef);
    }

    public final void appendInitCode(XmcWriter w, boolean withType) throws XmException
    {
        if(withType && _type != null)
            _type.appendDeclCode(w, _symbol, false, false);

        w.add(_gccAttrs).add(_symbol);

        if(_valueExpr != null)
            w.addSpc("=").addSpc(_valueExpr);
    }

    @Override
    public String toString()
    {
        return "[symbol=" + _symbol + "]";
    }

    public static final class MoeConstant extends XcIdent
    {
        private XcEnumType _enumType;

        public MoeConstant(String symbol, XcEnumType enumType)
        {
            super(symbol);
            _enumType = enumType;
        }

        public XcEnumType getEnumType()
        {
            return _enumType;
        }
    }

    @Override
    public void setGccAttribute(XcGccAttributeList attrs)
    {
        _gccAttrs = attrs;
    }

    @Override
    public XcGccAttributeList getGccAttribute()
    {
        return _gccAttrs;
    }

    public final boolean isLazyEvalType()
    {
        return _isLazyEvalType;
    }

    @Override
    public void setIsLazyEvalType(boolean enable) {
        _isLazyEvalType = enable;
    }

//     @Override
//     public void setLazyBindings(IRVisitable[] bindings)
//     {
//         _arraySizeBindings = bindings;
//     }
// 
//     @Override
//     public IRVisitable[] getLazyBindings()
//     {
//         return _arraySizeBindings;
//     }

    @Override
    public org.w3c.dom.Node[] getLazyBindingNodes() {
        return _arraySizeBindingNodes;
    }

    public void setLazyBindings(org.w3c.dom.Node[] nodes) {
        _arraySizeBindingNodes = nodes;
    }

    public void setIsIncomplete(boolean enable)
    {
        _isIncompleteType = enable;
    }

    public XcIdent getIncomp()
    {
        XcIdent incomp = new XcIdent(_symbol);
        incomp.setType(_type);
        incomp.setIsGccExtension(_isGccExtension);
        incomp.setIsGccThread(_isGccThread);
        incomp.setGccAttribute(_gccAttrs);
        incomp.setIsIncomplete(true);
        return incomp;
    }

    public boolean isOutput()
    {
        return _isOutput;
    }

    public void setOutput()
    {
        _isOutput = true;
    }

    public boolean isDeclared()
    {
        return _isDeclared;
    }

    public void setDeclared()
    {
        _isDeclared = true;
    }

    private void _lazyEval(XcType type, XcLazyVisitor visitor)
    {
        if((type == null) || (type instanceof XcBaseType))
            return;

        XcGccAttributeList attrs = type.getGccAttribute();

        if(attrs != null)
            visitor.lazyEnter(attrs);

        if(type instanceof XcArrayType)
            visitor.lazyEnter((XcLazyEvalType) type);

        type = type.getRefType();

        _lazyEval(type, visitor);
    }

    private void _lazyEval(XcIdent ident, XcLazyVisitor visitor)
    {
        if(ident == null)
            return;

        visitor.lazyEnter(ident);

        if(ident.getGccAttribute() != null)
            visitor.lazyEnter(ident.getGccAttribute());

        XcType type = ident.getType();

        _lazyEval(type, visitor);
    }

    public void lazyEval(XcLazyVisitor visitor)
    {
        XcType type = _type;

        if(_gccAttrs != null) {
            visitor.lazyEnter(_gccAttrs);
        }

        _lazyEval(type, visitor); // lazy evalueate for type

        while(type instanceof XcBasicType) {
            type = type.getRefType();
        }

        switch(_type.getTypeEnum()) { // lazy evaluate for child ident
        case STRUCT:
        case UNION:
            XcCompositeType ct = (XcCompositeType)type;

            if(ct.getMemberList() != null) {
                for(XcIdent child : ct.getMemberList()) {
                    _lazyEval(child, visitor);
                }
            }
            break;
        case ENUM:
            XcEnumType et = (XcEnumType)type;

            if(et.getEnumeratorList() != null) {
                for(XcIdent child : et.getEnumeratorList()) {
                    _lazyEval(child, visitor);
                }
            }
            break;
        case FUNC:
            XcFuncType ft = (XcFuncType)type;

            if(ft.getParamList() != null) {
                visitor.pushParamListIdentTable(ft.getParamList());
                for(XcIdent child : ft.getParamList()) {
                    _lazyEval(child, visitor);
                }
                visitor.popIdentTable();
            }
            break;
        default:
            break;
        }
    }

    @Override
    public Set<String> getDependVar()
    {
        return _dependVariables;
    }

    private void _gatherVar(XcIdent ident)
    {
        Set<String> vars = ident.getDependVar();

        if(vars.isEmpty() == false)
            _dependVariables.addAll(vars);

        XcType type = ident.getType();

        if(type == null)
            return;

        type = type.getRealType();

        while(type instanceof XcArrayType) {
            _dependVariables.addAll(((XcArrayType)type).getDependVar());

            type = type.getRefType();
            if(type == null)
                break;

            type = type.getRealType();
        }
    }

    public void gatherVar()
    {
        XcType type = _type;

        while(type instanceof XcBasicType) {
            type = type.getRefType();
        }

        switch(_type.getTypeEnum()) {
        case STRUCT:
        case UNION:
            XcCompositeType xc = (XcCompositeType)type;

            if(xc.getMemberList() != null) {
                for(XcIdent child : xc.getMemberList()) {
                    _gatherVar(child);
                }
            }
            break;
        case ENUM:
            XcEnumType xe = (XcEnumType)type;

            if(xe.getEnumeratorList() != null) {
                for(XcIdent child : xe.getEnumeratorList()) {
                    _gatherVar(child);
                }
            }
            break;
        default: /* type is unable to be tagged */
            break;
        }
    }
}
