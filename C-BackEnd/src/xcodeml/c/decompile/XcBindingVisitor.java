/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import java.util.Iterator;

import xcodeml.XmException;
import xcodeml.XmObj;
import xcodeml.binding.IXbLineNo;
import xcodeml.binding.IXbStringContent;
import xcodeml.c.binding.IXbcStatement;
import xcodeml.c.binding.IXbcTypedExpr;
import xcodeml.c.binding.IXbcArrayType;
import xcodeml.c.binding.IXbcBinaryExpr;
import xcodeml.c.binding.IXbcCompositeType;
import xcodeml.c.binding.IXbcMember;
import xcodeml.c.binding.IXbcSizeOrAlignExpr;
import xcodeml.c.binding.IXbcSymbolAddr;
import xcodeml.c.binding.IXbcType;
import xcodeml.c.binding.IXbcUnaryExpr;
import xcodeml.c.binding.IXbcVar;
import xcodeml.c.binding.gen.*;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcArrayLikeType;
import xcodeml.c.type.XcArrayType;
import xcodeml.c.type.XcIdentTableEnum;
import xcodeml.c.type.XcXmpCoArrayType;
import xcodeml.c.type.XcGccAttributable;
import xcodeml.c.type.XcGccAttribute;
import xcodeml.c.type.XcGccAttributeList;
import xcodeml.c.type.XcBaseTypeEnum;
import xcodeml.c.type.XcBasicType;
import xcodeml.c.type.XcCompositeType;
import xcodeml.c.type.XcEnumType;
import xcodeml.c.type.XcFuncType;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcIdentTable;
import xcodeml.c.type.XcIdentTableStack;
import xcodeml.c.type.XcLazyEvalType;
import xcodeml.c.type.XcParamList;
import xcodeml.c.type.XcPointerType;
import xcodeml.c.type.XcStructType;
import xcodeml.c.type.XcSymbolKindEnum;
import xcodeml.c.type.XcType;
import xcodeml.c.type.XcTypeEnum;
import xcodeml.c.type.XcUnionType;
import xcodeml.c.type.XcVarKindEnum;
import xcodeml.c.type.XcVoidType;
import xcodeml.c.util.XmcBindingUtil;
import xcodeml.util.XmBindingException;
import xcodeml.util.XmStringUtil;

/**
 * Converter of XcodeML's binding object to internal AST object (XmObj).
 */
public class XcBindingVisitor extends RVisitorBase
{
    private XcNode _parentNode;

    private XcNode _visitedNode;

    private XcIdentTableStack _identTableStack;

    private ScopeEnum _scopeEnum = ScopeEnum.GLOBAL;

    enum ScopeEnum
    {
        GLOBAL, LOCAL,
    }

    /**
     * Gets a node visitor stays.
     * 
     * @return
     */
    public XcNode getVisitedNode()
    {
        return _visitedNode;
    }

    /**
     * Creates a XcBindingVisitor.
     * 
     * @param identTableStack an identifier table stack.
     * @param parentNode a parent node visitor came from.
     * @param symContextEnum indicates scope of a node.
     */
    private XcBindingVisitor(XcIdentTableStack identTableStack, XcNode parentNode,
        ScopeEnum symContextEnum)
    {
        _identTableStack = identTableStack;
        _parentNode = parentNode;
        _scopeEnum = symContextEnum;
    }

    /**
     * Creates a XcBindingVisitor in the global scope.
     * 
     * @param identTableStack an identifier table stack.
     * @param parentNode a parent node visitor came from.
     */
    public XcBindingVisitor(XcIdentTableStack identTableStack, XcNode parentNode)
    {
        _identTableStack = identTableStack;
        _parentNode = parentNode;
        _scopeEnum = ScopeEnum.GLOBAL;
    }

    /**
     * Creates a XcBindingVisitor in the global scope.
     * 
     * @param identTableStack an identifier table stack.
     */
    public XcBindingVisitor(XcIdentTableStack identTableStack)
    {
        _identTableStack = identTableStack;
        _scopeEnum = ScopeEnum.GLOBAL;
    }

    /**
     * Creates a XcProgramObj from a XcodeProgram tag object.

     * @param xprog a XcodeProgram tag object.
     * @return
     * @throws XmBindingException thrown if there is fault description in XcodeML.
     */
    public static final XcProgramObj createXcProgramObj(XbcXcodeProgram xprog)
        throws XmBindingException
    {
        XcIdentTableStack itStack = new XcIdentTableStack();
        XcIdentTable itable = itStack.push();
        XcBindingVisitor visitor = new XcBindingVisitor(itStack, null, ScopeEnum.GLOBAL);

        visitor.enter(xprog);

        XcProgramObj prog = (XcProgramObj)visitor.getVisitedNode();

        prog.setIdentTable(itable);

        _checkChild(prog);
        prog.reduce();

        return prog;
    }

    private static void _checkChild(XcNode node)
    {
        if(node == null)
            return;

        node.checkChild();
        XcNode[] children = node.getChild();
        if(children != null) {
            for(XcNode child : children)
                _checkChild(child);
        }
    }

    private boolean _enter(XcBindingVisitor visitor, IRVisitable... visitable)
    {
        if(visitable != null && visitable.length > 0) {
            if(visitor._parentNode instanceof XcControlStmtObj.For) {
                for(IRVisitable v : visitable) {
                    if(v == null) {
                        visitor._setAsLeaf(XcNullExpr.createXcNullExpr(), null);
                    } else {
                        v.enter(visitor);
                    }
                }

            } else {
                for(IRVisitable v : visitable) {
                    if(v != null) {
                        v.enter(visitor);
                    }
                }
            }
        }

        return true;
    }

    private boolean _setAsLeaf(XcObj obj, XmObj xobj)
    {
        _visitedNode = obj;

        if(_parentNode != null)
            _parentNode.addChild(obj);

        if((obj instanceof XcStAndDeclObj) && (xobj instanceof IXbcStatement)
            && ((IXbcStatement)xobj).getLineno() != null) {
            XcStAndDeclObj stmtAndDecl = (XcStAndDeclObj)obj;
            IXbcStatement xstmt = (IXbcStatement)xobj;
            _setSourcePos(stmtAndDecl, xstmt);
        }

        return true;
    }

    private XcBindingVisitor _setAsNode(XcObj obj, XmObj xobj)
    {
        _setAsLeaf(obj, xobj);
        return new XcBindingVisitor(_identTableStack, obj, _scopeEnum);
    }

    private String _getContentString(IXbStringContent strContent)
    {
        if(strContent == null)
            return null;

        return XmStringUtil.trim(strContent.getContent());
    }

    private String _getContent(IXbStringContent xstr)
    {
        String name = XmStringUtil.trim(xstr.getContent());
        if(name == null)
            throw new XmBindingException((XmObj)xstr, "content is empty");

        return name;
    }

    private XcIdent _getIdent(XcSymbolKindEnum kind, IXbStringContent xstr)
    {
        String name = _getContent(xstr);
        XcIdent ident = _identTableStack.getIdent(kind, name);

        if(ident == null)
            throw new XmBindingException((XmObj)xstr, "variable '" + name + "' is not defined");

        return ident;
    }

    private XcIdent _getIdentVarOrFunc(IXbStringContent xstr)
    {
        String name = _getContent(xstr);

        if(name == null) {
            throw new XmBindingException((XmObj)xstr, "variable or function name is not specified");
        }

        XcIdent ident = _identTableStack.getIdent(XcSymbolKindEnum.VAR, name);

        if(ident == null) {
            ident = _identTableStack.getIdent(XcSymbolKindEnum.FUNC, name);

            if(ident == null) {
                if (name.startsWith("_XMP_")) return new XcIdent(name);

                throw new XmBindingException((XmObj)xstr, "variable or function '" + name
                    + "' is not defined");
            }
        }

        return ident;
    }

    private XcIdent _getIdentFunc(XbcName xname)
    {
        return _getIdent(XcSymbolKindEnum.FUNC, xname);
    }

    private XcIdent _getIdentFunc(XbcName xname, XbcParams xparams)
    {
        XcIdent ident = _getIdentFunc(xname);
        XcType type = ident.getType();

        if(XcTypeEnum.FUNC.equals(type.getTypeEnum()) == false)
            throw new XmBindingException(xname, "symbol '" + ident.getSymbol()
                + "' is not function type");

        XcFuncType funcType = (XcFuncType)type;
        XcParamList paramList = funcType.getParamList();

        if(paramList.isEmpty() == false) {
            // TODO strict parameter check
            int sizeName = (xparams != null) ? xparams.sizeName() : 0;
            if(paramList.isVoid() == false && (sizeName != paramList.size()))
                throw new XmBindingException(xparams,
                    "parameter type is not applicable as function '" + type.getTypeId() + "'");

        } else {
            // replace explicit parameters instead of empty parameters.

            funcType = funcType.copy();
            ident.setType(funcType);

            for(XbcName xparamName : xparams.getName()) {
                String paramTypeId = XmStringUtil.trim(xparamName.getType());
                _ensureAttr(xparamName, paramTypeId, "type");
                String paramName = _getContent(xparamName);
                XcIdent paramIdent = new XcIdent(paramName);
                paramIdent.setTempTypeId(paramTypeId);
                funcType.addParam(paramIdent);
            }

            try {
                funcType.getParamList().resolve(_identTableStack);
            } catch(XmException e) {
                throw new XmBindingException(xname, e);
            }
        }

        return ident;
    }

    private XcIdent _getIdentEnumerator(XmObj xobj, String typeId, String moe)
    {
        XcEnumType enumType;

        try {
            enumType = (XcEnumType)_identTableStack.getTypeAs(XcTypeEnum.ENUM, typeId);
        } catch(XmException e) {
            throw new XmBindingException(xobj, e);
        }

        XcIdent ident = enumType.getEnumerator(moe);

        if(ident == null)
            throw new XmBindingException(xobj, "enum type '" + typeId
                + "' does not have enumerator '" + moe + "'");

        return ident;
    }

    private String _getChildTypeId(IXbcExpressionsChoice xexpr)
    {
        if(xexpr == null)
            throw new XmBindingException((XmObj)xexpr, "no array type is specified");

        if(xexpr instanceof IXbcTypedExpr) {
            return ((IXbcTypedExpr)xexpr).getType();
        }

        throw new XmBindingException((XmObj)xexpr, "invalid expression");
    }

    private String _getChildTypeId(IXbcMember xmember)
    {
        IXbcExpressionsChoice xexpr = xmember.getExpressions();

        if(xexpr == null)
            throw new XmBindingException((XmObj)xmember, "no composite type is specified");

        if(xexpr instanceof IXbcTypedExpr) {
            return ((IXbcTypedExpr)xexpr).getType();
        }

        throw new XmBindingException((XmObj)xmember, "invalid expression");
    }

    private XcIdent _getIdentCompositeTypeMember(IXbcMember xmember)
    {
        String typeId = _getChildTypeId(xmember);
        String member = xmember.getMember();
        XmObj xobj = (XmObj)xmember;
        _ensureAttr(xobj, typeId, "type");
        _ensureAttr(xobj, member, "member");
        XcType ptype, type;

        try {
            ptype = _identTableStack.getType(typeId).getRealType();
            type = ptype.getRefType().getRealType();
        } catch(XmException e) {
            throw new XmBindingException(xobj, e);
        }

        if((type instanceof XcCompositeType) == false)
            throw new XmBindingException(xobj, "type '" + typeId + "' is not struct/union pointer type");

        XcCompositeType compType = (XcCompositeType)type;
        XcIdent ident = compType.getMember(member);

        if(ident == null)
            throw new XmBindingException(xobj, "symbol '" + member + "' is not a member of type '"
                + compType.getTypeId() + "'");
        return ident;
    }
    
    private void _setSourcePos(XcSourcePositioned obj, IXbLineNo lineNo)
    {
        obj.setSourcePos(new XcSourcePosObj(
            lineNo.getFile(),
            lineNo.getLineno(),
            lineNo.getRawlineno()));
    }

    @Override
    public boolean enter(XbcXcodeProgram visitable)
    {
        XcProgramObj obj = new XcProgramObj();

        obj.setLanguage(visitable.getLanguage());
        obj.setTime(visitable.getTime());
        obj.setSource(visitable.getSource());
        obj.setVersion(visitable.getVersion());
        obj.setCompilerInfo(visitable.getCompilerInfo());

        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getTypeTable(), visitable.getGlobalSymbols(), visitable
            .getGlobalDeclarations());
    }

    @Override
    public boolean enter(XbcGlobalDeclarations visitable)
    {
        // globalDeclarations combiles to XcProgram
        return _enter(this, visitable.getContent());
    }

    @Override
    public boolean enter(XbcDeclarations visitable)
    {
        XcDeclsObj obj = new XcDeclsObj();
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getContent());
    }

    @Override
    public boolean enter(XbcExprStatement visitable)
    {
        XcExprStmtObj obj = new XcExprStmtObj();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getExpressions());
    }

    @Override
    public boolean enter(XbcTypeTable visitable)
    {
        _enter(this, visitable.getTypes());

        try {
            _identTableStack.resolveType();
        } catch(XmException e) {
            throw new XmBindingException(visitable, e);
        }

        return true;
    }

    @Override
    public boolean enter(XbcGlobalSymbols visitable)
    {
        _enter(this, visitable.getId());

        try {
            _identTableStack.resolveDependency(this);
        } catch (XmException e) {
            throw new XmBindingException(visitable, e);
        }

        return true;
    }

    @Override
    public boolean enter(XbcSymbols visitable)
    {
        _enter(this, visitable.getContent());

        try {
            _identTableStack.resolveDependency(this);
        } catch (XmException e) {
            throw new XmBindingException(visitable, e);
        }

        return true;
    }

    private static void _ensureAttr(XmObj obj, Object attr, String msg)
    {
        if(attr == null || (attr instanceof String && XmStringUtil.trim((String)attr) == null))
            throw new XmBindingException(obj, "no " + msg);
    }

    private static void _setTypeAttr(XcType type, IXbcType xtype)
    {
        String isConstStr = xtype.getIsConst();
        String isVolatileStr = xtype.getIsVolatile();
        String isRestrictStr = xtype.getIsRestrict();
        XmObj xobj = (XmObj)xtype;

        if(isConstStr != null)
            type.setIsConst(XmStringUtil.getAsBool(xobj, isConstStr));
        if(isVolatileStr != null)
            type.setIsVolatile(XmStringUtil.getAsBool(xobj, isVolatileStr));
        if(isRestrictStr != null)
            type.setIsRestrict(XmStringUtil.getAsBool(xobj, isRestrictStr));
    }

    private void _addType(XcType type, XmObj xobj)
    {
        try {
            _identTableStack.addType(type);
        } catch(XmException e) {
            throw new XmBindingException(xobj, e);
        }
    }

    private void _addGccAttribute(XcGccAttributable parent, XbcGccAttributes attrs)
    {
        if (attrs != null) {
            /*
             * set XcGccAttributeList as LazyEvalType
             */
            parent.setGccAttribute(new XcGccAttributeList(attrs));
        } else {
            parent.setGccAttribute(null);
        }
    }

    @Override
    public boolean enter(XbcGccAttribute visitable)
    {
        XcGccAttribute obj = new XcGccAttribute();

        String name = visitable.getName();

        obj.setName(name);

        XcBindingVisitor visitor =  _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getExpressions());
    }

    @Override
    public boolean enter(XbcGccAttributes visitable)
    {
        XcGccAttributeList attrs = new XcGccAttributeList();

        XcBindingVisitor visitor = _setAsNode(attrs, visitable);

        return _enter(visitor, visitable.getGccAttribute());
    }

    @Override
    public boolean enter(XbcBasicType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        _ensureAttr(visitable, typeId, "type");

        String typeName = XmStringUtil.trim(visitable.getName());
        _ensureAttr(visitable, typeName, "name");

        XcBasicType type = new XcBasicType(typeId);

        type.setTempRefTypeId(typeName);

        _setTypeAttr(type, visitable);

        _addType(type, visitable);

        _addGccAttribute(type, visitable.getGccAttributes());

        return true;
    }

    private boolean _enterArrayType(XcArrayLikeType type, IXbcArrayType visitable)
    {
        String ArraySizeStr = visitable.getArraySize1();

        if(ArraySizeStr == null) {
            type.setIsArraySize(false);
            type.setIsArraySizeExpr(false);

        } else if(ArraySizeStr.equals("*")) {
            type.setIsArraySize(false);
            type.setIsArraySizeExpr(true);

            XbcArraySize arraySizeBinding = visitable.getArraySize2();

            XcLazyEvalType lazyType = type;
            lazyType.setIsLazyEvalType(true);

            IXbcExpressionsChoice _expr = arraySizeBinding.getExpressions();
            lazyType.setLazyBindings(new IRVisitable[] { _expr });

        } else {
            type.setIsArraySize(true);
            type.setIsArraySizeExpr(false);

            type.setArraySize(XmStringUtil.getAsCInt((XmObj)visitable, ArraySizeStr));
        }

        type.setTempRefTypeId(visitable.getElementType());

        _addType((XcType)type, (XmObj)visitable);

        return true;
    }

    @Override
    public boolean enter(XbcArrayType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        _ensureAttr(visitable, typeId, "type");

        XcArrayType type = new XcArrayType(typeId);
        _setTypeAttr(type, visitable);

        String isStaticStr = visitable.getIsStatic();
        if(isStaticStr != null)
            type.setIsStatic(XmStringUtil.getAsBool(visitable, isStaticStr));

        _addGccAttribute(type, visitable.getGccAttributes());

        return _enterArrayType(type, visitable);
    }

    @Override
    public boolean enter(XbcArraySize visitable)
    {
        _enter(this, visitable.getExpressions());
        return true;
    }

    @Override
    public boolean enter(XbcPointerType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        String refTypeId = XmStringUtil.trim(visitable.getRef());
        _ensureAttr(visitable, typeId, "type");
        _ensureAttr(visitable, refTypeId, "reference type");

        XcPointerType type = new XcPointerType(typeId);
        _setTypeAttr(type, visitable);
        type.setTempRefTypeId(refTypeId);

        _addType(type, visitable);

        _addGccAttribute(type, visitable.getGccAttributes());

        return true;
    }

    @Override
    public boolean enter(XbcEnumType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        _ensureAttr(visitable, typeId, "type");

        XcEnumType type = new XcEnumType(typeId);
        _setTypeAttr(type, visitable);

        XbcSymbols xsymbols = visitable.getSymbols();

        boolean isAllIntConstant = true;
        int i = 0;
        if(xsymbols != null)
        for(IXbcSymbolsChoice xsc : xsymbols.getContent()) {
            if(xsc instanceof XbcId) {
                XbcId xid = (XbcId)xsc;

                XbcName xname = xid.getName();

                String name = XmStringUtil.trim(xname.getContent());
                _ensureAttr(xname, name, "name");

                XcIdent.MoeConstant ident = new XcIdent.MoeConstant(name, type);

                XbcValue xvalue = xid.getValue();
                if(xvalue != null) {
                    IXbcValueChoice xexpr = xvalue.getContent();

                    if(xexpr instanceof XbcIntConstant) {
                        i = XmStringUtil.getAsCInt((XmObj)xexpr, ((XbcIntConstant)xexpr).getContent());
                        ident.setValue(new XcConstObj.IntConst(i++, XcBaseTypeEnum.INT));
                    } else if(xexpr instanceof IXbcExpressionsChoice) {
                        XcLazyEvalType lazyIdent = ident;
                        lazyIdent.setIsLazyEvalType(true);
                        lazyIdent.setLazyBindings(new IRVisitable[] { xexpr });

                        isAllIntConstant = false;
                    }
                } else {
                    if(isAllIntConstant) {
                        ident.setValue(new XcConstObj.IntConst(i++, XcBaseTypeEnum.INT));
                    } else {
                        ident.setValue(null);
                    }
                }

                type.addEnumerator(ident);
            }
        }

        _addType(type, visitable);

        _addGccAttribute(type, visitable.getGccAttributes());

        return true;
    }

    private boolean _enterCompositeType(XcCompositeType type, IXbcCompositeType xtype)
    {
        _setTypeAttr(type, xtype);

        XbcSymbols xsymbols = xtype.getSymbols();

        if(xsymbols == null) {
            type.setMemberList(null);
            _addType(type, (XmObj)xtype);
            return true;
        }

        for(IXbcSymbolsChoice xsc : xsymbols.getContent()) {
            if (xsc instanceof XbcId) {
                XbcId xid = (XbcId)xsc;
                String typeId = XmStringUtil.trim(xid.getType());
                XbcName xname = xid.getName();

                if(typeId == null && xname != null)
                    typeId = XmStringUtil.trim(xname.getType());

                _ensureAttr(xid, typeId, "type");

                String name = null;
                if(xname != null)
                    name = XmStringUtil.trim(xname.getContent());

                XcIdent ident = new XcIdent(name);
                ident.setTempTypeId(typeId);

                String isGccExtensionStr = xid.getIsGccExtension();
                String isGccThreadStr = xid.getIsGccThread();

                if (isGccExtensionStr != null)
                    ident.setIsGccExtension(XmStringUtil.getAsBool(xid, isGccExtensionStr));
                if (isGccThreadStr != null)
                    ident.setIsGccThread(XmStringUtil.getAsBool(xid, isGccThreadStr));

                String bitFieldStr = xid.getBitField1();

                if(bitFieldStr != null) {
                    if (bitFieldStr.equals("*")) {
                        XbcBitField bitFiledExpr = xid.getBitField2();

                        if(bitFiledExpr == null)
                            throw new XmBindingException((XmObj)xid, "bitFidld must be specified");

                        IXbcExpressionsChoice expr = bitFiledExpr.getExpressions();

                        ident.setIsBitField(true);
                        ident.setIsBitFieldExpr(true);

                        XcLazyEvalType lazyIdent = (XcLazyEvalType)ident;
                        lazyIdent.setIsLazyEvalType(true);
                        lazyIdent.setLazyBindings(new IRVisitable[] { expr });
                    } else {
                        ident.setIsBitField(true);
                        ident.setIsBitFieldExpr(false);
                        ident.setBitField(XmStringUtil.getAsCInt(xid, bitFieldStr));
                    }
                }

                _addGccAttribute(ident, xid.getGccAttributes());

                type.addMember(ident);
            }
        }

        _addType(type, (XmObj)xtype);

        return true;
    }

    @Override
    public boolean enter(XbcStructType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        _ensureAttr(visitable, typeId, "type");

        XcStructType type = new XcStructType(typeId);
        _addGccAttribute(type, visitable.getGccAttributes());

        return _enterCompositeType(type, visitable);
    }

    @Override
    public boolean enter(XbcUnionType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        _ensureAttr(visitable, typeId, "type");

        XcUnionType type = new XcUnionType(typeId);

        _addGccAttribute(type, visitable.getGccAttributes());

        return _enterCompositeType(type, visitable);
    }

    @Override
    public boolean enter(XbcFunctionType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        _ensureAttr(visitable, typeId, "type");

        XcFuncType type = new XcFuncType(typeId);
        _setTypeAttr(type, visitable);

        String inlineStr = visitable.getIsInline();
        if(inlineStr != null)
            type.setIsInline(XmStringUtil.getAsBool((XmObj)visitable, inlineStr));

        String staticStr = visitable.getIsStatic();
        if(staticStr != null)
            type.setIsStatic(XmStringUtil.getAsBool((XmObj)visitable, staticStr));

        if(visitable.getReturnType() != null)
            type.setTempRefTypeId(visitable.getReturnType());

        XbcParams xparams = visitable.getParams();

        if(xparams != null) {
            for(XbcName xname : xparams.getName()) {
                String name = XmStringUtil.trim(xname.getContent());
                String paramTypeId = XmStringUtil.trim(xname.getType());
                _ensureAttr(xname, paramTypeId, "type");

                XcIdent ident = new XcIdent(name);
                ident.setTempTypeId(paramTypeId);
                type.addParam(ident);
            }

            if(xparams.getEllipsis() != null)
                type.setIsEllipsised(true);
        }

        // TODO if needed
        // throw exception if
        // param list is empty and argument has ellipsis

        _addGccAttribute(type, visitable.getGccAttributes());
        _addType(type, visitable);

        return true;
    }

    @Override
    public boolean enter(XbcId visitable)
    {
        // called by enter(XbSymbols|XbGlobalSymbols)

        XbcName xname = visitable.getName();
        _ensureAttr(visitable, xname, "name");

        String name = xname.getContent();
        _ensureAttr(xname, name, "name");

        String typeId = visitable.getType();
        if(typeId == null) {
            typeId = xname.getType();
            _ensureAttr(xname, typeId, "type");
        }

        XcIdent ident = new XcIdent(name);

        _addGccAttribute(ident, visitable.getGccAttributes());

        ident.setTempTypeId(typeId);

        String isGccExtensionStr = visitable.getIsGccExtension();
        String isGccThreadStr = visitable.getIsGccThread();

        if (isGccExtensionStr != null)
            ident.setIsGccExtension(XmStringUtil.getAsBool((XmObj)visitable, isGccExtensionStr));

        if (isGccThreadStr != null)
            ident.setIsGccThread(XmStringUtil.getAsBool((XmObj)visitable, isGccThreadStr));

        try {
            ident.resolve(_identTableStack);
        } catch(XmException e) {
            throw new XmBindingException(visitable, e);
        }

        String sclass = XmStringUtil.trim(visitable.getSclass());
        XcSymbolKindEnum kind = null;

        if(sclass != null) {
            if("auto".equals(sclass)) {
                //function type must not be decorated by 'auto',
                //and other type do not need to be decorated by 'auto'.
                //ident.setIsAuto(true);
            } else if("extern".equals(sclass)) {
                ident.setIsExtern(true);
            } else if("extern_def".equals(sclass)) {
                ident.setIsExternDef(true);
            } else if("static".equals(sclass)) {
                ident.setIsStatic(true);
            } else if("register".equals(sclass)) {
                ident.setIsRegister(true);
            } else if("typedef_name".equals(sclass)) {
                ident.setIsTypedef(true);
                kind = XcSymbolKindEnum.TYPE;
            } else if("param".equals(sclass)) {
                ident.setVarKindEnum(XcVarKindEnum.PARAM);
                kind = XcSymbolKindEnum.VAR;
            } else if("label".equals(sclass)) {
                kind = XcSymbolKindEnum.LABEL;
            } else if("tagname".equals(sclass)) {
                kind = XcSymbolKindEnum.TAGNAME;
            } else if("moe".equals(sclass)) {
                kind = XcSymbolKindEnum.MOE;
            }
        }

        XcType type = ident.getType();
        
        if(kind == null) {
            if(type != null && XcTypeEnum.FUNC.equals(type.getTypeEnum())) {
                kind = XcSymbolKindEnum.FUNC;

            } else {
                kind = XcSymbolKindEnum.VAR;
            }
        }

        if(XcSymbolKindEnum.VAR.equals(kind) && ident.getVarKindEnum() == null) {
            switch(_scopeEnum) {
            case GLOBAL:
                ident.setVarKindEnum(XcVarKindEnum.GLOBAL);
                break;
            case LOCAL:
                ident.setVarKindEnum(XcVarKindEnum.LOCAL);
                break;
            default:
                /* not reachable */
                throw new IllegalArgumentException();
            }
        }

        try {
            _identTableStack.addIdent(kind, ident);
        } catch(XmException e) {
            throw new XmBindingException(visitable, e.getMessage());
        }

        /*
           if ident is member of enum 
           then create and add anonymous enum.
        */
        if(XcSymbolKindEnum.MOE.equals(kind)) {
            if(((XcEnumType)type).getTagName() == null) {
                ident = new XcIdent(null);
                ident.setType(type);

                try {
                    _identTableStack.addAnonIdent(ident);
                } catch(XmException e) {
                    throw new XmBindingException(visitable, e.getMessage());
                }
            }
        }

        return true;
    }

    @Override
    public boolean enter(XbcVarDecl visitable)
    {
        XcIdent ident = _getIdent(XcSymbolKindEnum.VAR, visitable.getName());

        XcDeclObj obj = new XcDeclObj(ident);

        _setSourcePos(obj, visitable);
        String gccAsmCodeStr = null;
        XbcStringConstant xsc = null;
        XbcGccAsm xasm = visitable.getGccAsm();
        if(xasm != null)
            xsc = xasm.getStringConstant();

        if(xsc != null)
            gccAsmCodeStr = xsc.getContent();

        obj.setGccAsmCode(gccAsmCodeStr);

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getValue());
    }

    @Override
    public boolean enter(XbcValue visitable)
    {
        return _enter(this, visitable.getContent());
    }

    @Override
    public boolean enter(XbcDesignatedValue visitable)
    {
        XcDesignatedValueObj obj = new XcDesignatedValueObj();
        obj.setMember(visitable.getMember());

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getContent());
    }

    @Override
    public boolean enter(XbcFunctionDecl visitable)
    {
        XcIdent ident = _getIdentFunc(visitable.getName());

        XcType type = ident.getType();

        if((type instanceof XcFuncType) == false)
            throw new XmBindingException(visitable, "symbol is declared as function, but type of symbol is not function.");
        XcGccAttributeList attrs = ident.getGccAttribute();

        if(attrs != null && attrs.containsAttrAlias()) {
            if(ident.isDeclared()) {
                return true;
            }
        }

        ident.setDeclared();

        XcDeclObj obj = new XcDeclObj(ident);
        _setSourcePos(obj, visitable);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcFunctionDefinition visitable)
    {
        XcIdent ident = _getIdentFunc(visitable.getName(), visitable.getParams());

        XcFuncDefObj obj = new XcFuncDefObj(ident);
        _setSourcePos(obj, visitable);
        String isGccExtensionStr = visitable.getIsGccExtension();

        if (isGccExtensionStr != null)
            obj.setIsGccExtension(XmStringUtil.getAsBool((XmObj)visitable, isGccExtensionStr));

        XbcParams xparams = visitable.getParams();
        XcParamList pList = ((XcFuncType) ident.getType()).getParamList();

        if(xparams == null && pList != null)
            throw new XmBindingException(visitable, "mismatch with type by parameter size.");

        Iterator<XcIdent> pIdentIter = pList.iterator();

        if(xparams != null) {
            for(XbcName xname : xparams.getName()) {
                if(pIdentIter.hasNext() == false) {
                    throw new XmBindingException(visitable, "mismatch with type by parameter size.");
                }
                pIdentIter.next();

                String name = XmStringUtil.trim(xname.getContent());
                String paramTypeId = XmStringUtil.trim(xname.getType());

                _ensureAttr(xname, paramTypeId, "type");

                XcIdent paramIdent = new XcIdent(name);
                XcType  paramType;

                try {
                    paramType = _identTableStack.getType(paramTypeId);
                } catch(XmException e) {
                    throw new XmBindingException(visitable, e);
                }

                // TODO if needed
                // if(paramTypeId.equals(paramType.getTypeId()))
                //    throw new XbcBindingException(visitable, "mismatch with type by paramemter type.");

                paramIdent.setType(paramType);

                obj.addParam(paramIdent);
            }

            if(pIdentIter.hasNext()) {
                XcIdent restParamIdent = pIdentIter.next();
                if(!(restParamIdent.getType() instanceof XcVoidType))
                    throw new XmBindingException(visitable, "mismatch with type of parameter.");

                obj.addParam(restParamIdent);
            }

            obj.setIsEllipsised(pList.isEllipsised());
        }

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        XcIdentTable it = _identTableStack.push();

        ScopeEnum symContextEnum0 = _scopeEnum;
        _scopeEnum = ScopeEnum.LOCAL;

        _enter(visitor, visitable.getSymbols(), visitable.getParams(), visitable.getBody(), visitable.getGccAttributes());

        if(obj.getCompStmt() != null && obj.getCompStmt().getIdentTable() == null) {
            obj.getCompStmt().setIdentTable(it);
        }

        _identTableStack.pop();
        _scopeEnum = symContextEnum0;

        return true;
    }

    @Override
    public boolean enter(XbcBody visitable)
    {
        XcBindingVisitor visitor = null;

        if(_parentNode instanceof XcCompStmtObj ||
            (_parentNode instanceof XcFuncDefObj && visitable.getStatements().length == 1 &&
                visitable.getStatements(0) instanceof XbcCompoundStatement)) {
            visitor = this;

        } else {
            XcCompStmtObj obj = new XcCompStmtObj();
            obj.setIdentTable(_identTableStack.getLast());

            visitor = _setAsNode(obj, visitable);
        }

        return _enter(visitor, visitable.getStatements());
    }

    @Override
    public boolean enter(XbcCompoundStatement visitable)
    {
        XcCompStmtObj obj = new XcCompStmtObj();
        _setSourcePos(obj, visitable);
        obj.setIdentTable(_identTableStack.push());
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        _enter(visitor, visitable.getSymbols(), visitable.getDeclarations(), visitable.getBody());
        _identTableStack.pop();

        return true;
    }

    @Override
    public boolean enter(XbcIfStatement visitable)
    {
        XcControlStmtObj.If obj = new XcControlStmtObj.If();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getCondition(), visitable.getThen(), visitable.getElse());
    }

    @Override
    public boolean enter(XbcCondition visitable)
    {
        return _enter(this, visitable.getExpressions());
    }

    @Override
    public boolean enter(XbcInit visitable)
    {
        return _enter(this, visitable.getExpressions());
    }

    @Override
    public boolean enter(XbcIter visitable)
    {
        return _enter(this, visitable.getExpressions());
    }

    private boolean _enterStmtBlock(XmObj visitable, IXbcStatementsChoice choice)
    {
        if(choice == null)
            return true;

        _enter(this, choice);

        return true;
    }

    @Override
    public boolean enter(XbcThen visitable)
    {
        IXbcStatementsChoice stmt = visitable.getStatements();
        if(stmt == null) {
            XbcCompoundStatement cstmt = new XbcCompoundStatement();
            XbcSymbols symbols = new XbcSymbols();
            XbcDeclarations declrs = new XbcDeclarations();
            cstmt.setSymbols(symbols);
            cstmt.setDeclarations(declrs);
            stmt = cstmt;
        }
        
        return _enterStmtBlock(visitable, stmt);
    }

    @Override
    public boolean enter(XbcElse visitable)
    {
        return _enterStmtBlock(visitable, visitable.getStatements());
    }

    @Override
    public boolean enter(XbcWhileStatement visitable)
    {
        XcControlStmtObj.While obj = new XcControlStmtObj.While();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getCondition(), visitable.getBody());
    }

    @Override
    public boolean enter(XbcDoStatement visitable)
    {
        XcControlStmtObj.Do obj = new XcControlStmtObj.Do();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getCondition(), visitable.getBody());
    }

    @Override
    public boolean enter(XbcForStatement visitable)
    {
        XcControlStmtObj.For obj = new XcControlStmtObj.For();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor,
                      visitable.getInit(),
                      visitable.getCondition(),
                      visitable.getIter(),
                      visitable.getBody());
    }

    @Override
    public boolean enter(XbcBreakStatement visitable)
    {
        XcControlStmtObj.Break obj = new XcControlStmtObj.Break();
        _setSourcePos(obj, visitable);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcContinueStatement visitable)
    {
        XcControlStmtObj.Continue obj = new XcControlStmtObj.Continue();
        _setSourcePos(obj, visitable);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcReturnStatement visitable)
    {
        XcControlStmtObj.Return obj = new XcControlStmtObj.Return();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getExpressions());
    }

    @Override
    public boolean enter(XbcGotoStatement visitable)
    {
        XcControlStmtObj.Goto obj = new XcControlStmtObj.Goto();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getContent());
    }

    @Override
    public boolean enter(XbcName visitable)
    {
        // called by enter(XbGotoStatement)
        XcNameObj obj = new XcNameObj(visitable.getContent());
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcSwitchStatement visitable)
    {
        XcControlStmtObj.Switch obj = new XcControlStmtObj.Switch();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getValue(), visitable.getBody());
    }

    @Override
    public boolean enter(XbcCaseLabel visitable)
    {
        XcControlStmtObj.CaseLabel obj = new XcControlStmtObj.CaseLabel();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);
        return _enter(visitor, visitable.getValue());
    }

    @Override
    public boolean enter(XbcDefaultLabel visitable)
    {
        XcControlStmtObj.DefaultLabel obj = new XcControlStmtObj.DefaultLabel();
        _setSourcePos(obj, visitable);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcStatementLabel visitable)
    {
        String name = _getContentString(visitable.getName());

        if(name == null)
            throw new XmBindingException(visitable, "no label name");

        XcControlStmtObj.Label obj = new XcControlStmtObj.Label(name);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcFloatConstant visitable)
    {
        String text = _getContentString(visitable);

        if(text == null)
            throw new XmBindingException(visitable, "invalid float fraction/exponential");

        XcConstObj.FloatConst obj;
        XcBaseTypeEnum btEnum;
        String typeId = XmStringUtil.trim(visitable.getType());

        if(typeId == null)
            btEnum = XcBaseTypeEnum.DOUBLE;
        else {
            btEnum = XcBaseTypeEnum.getByXcode(typeId);

            if(btEnum == null)
                throw new XmBindingException(visitable, "invalid type '" + typeId + "' as float constant");
        }

        obj = new XcConstObj.FloatConst(text, btEnum);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcIntConstant visitable)
    {
        String text = _getContentString(visitable);
        if(text == null)
            throw new XmBindingException(visitable, "invalid constant value");

        XcBaseTypeEnum btEnum;
        String typeId = XmStringUtil.trim(visitable.getType());

        if(typeId == null)
            btEnum = XcBaseTypeEnum.INT;
        else
            btEnum = XcBaseTypeEnum.getByXcode(typeId);
        
        if(btEnum == null)
            throw new XmBindingException(visitable, "invalid type '" + typeId + "' as int constant");

        XcConstObj.IntConst obj;

        try {
            obj = new XcConstObj.IntConst(text, btEnum);
        } catch(XmException e) {
            throw new XmBindingException(visitable, e.getMessage());
        }
        return _setAsLeaf(obj, visitable);
    }
    
    @Override
    public boolean enter(XbcLonglongConstant visitable)
    {
        XcConstObj.LongLongConst obj = XmcBindingUtil.createLongLongConst(visitable);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcStringConstant visitable)
    {
        String text = visitable.getContent();
        XcConstObj.StringConst obj = new XcConstObj.StringConst(text);

        boolean isWide = false;

        String isWideStr = visitable.getIsWide();

        if(isWideStr != null)
            isWide = XmStringUtil.getAsBool((XmObj)visitable, isWideStr);

        obj.setIsWide(isWide);

        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcMoeConstant visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        String moe = XmStringUtil.trim(visitable.getContent());

        _ensureAttr(visitable, typeId, "type");
        _ensureAttr(visitable, moe, "enumerator symbol");

        XcIdent ident = _getIdentEnumerator(visitable, typeId, moe);

        return _setAsLeaf(ident, visitable);
    }

    private XcExprObj _shiftUpCoArray(XcOperatorObj op)
    {
        if(op.getOperatorEnum() != XcOperatorEnum.PLUS)
            return op;

        XcExprObj exprs[] = op.getExprObjs();

        if((exprs[0] instanceof XcXmpCoArrayRefObj) == false)
            return op;

        XcXmpCoArrayRefObj coaRefObj = (XcXmpCoArrayRefObj)exprs[0];

        XcType elemetType = coaRefObj.getElementType().getRealType();

        if((elemetType.getTypeEnum() == XcTypeEnum.ARRAY) == false)
            return op;

        XcArrayType at = (XcArrayType)elemetType;

        XcPointerType pt = new XcPointerType("XMPP", at.getRefType());

        coaRefObj.turnOver(op, pt);
        
        return coaRefObj;
    }

    private boolean _enterExpr(IXbcUnaryExpr visitable, XcOperatorEnum opeEnum)
    {
        XcOperatorObj obj = new XcOperatorObj(opeEnum);
        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);
        return _enter(visitor, visitable.getExpressions());
    }

    private boolean _enterExpr(IXbcBinaryExpr visitable, XcOperatorEnum opeEnum)
    {
        XcOperatorObj obj = new XcOperatorObj(opeEnum);
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions1(), visitable.getExpressions2());

        XcExprObj expr = _shiftUpCoArray(obj);

        _setAsLeaf((XcObj)expr, (XmObj)visitable);

        return true;
    }

    @Override
    public boolean enter(XbcUnaryMinusExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.UNARY_MINUS);
    }

    @Override
    public boolean enter(XbcPostDecrExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.POST_DECR);
    }

    @Override
    public boolean enter(XbcPostIncrExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.POST_INCR);
    }

    @Override
    public boolean enter(XbcPreDecrExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.PRE_DECR);
    }

    @Override
    public boolean enter(XbcPreIncrExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.PRE_INCR);
    }

    @Override
    public boolean enter(XbcAsgBitAndExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_BIT_AND);
    }

    @Override
    public boolean enter(XbcAsgBitOrExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_BIT_OR);
    }

    @Override
    public boolean enter(XbcAsgBitXorExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_BIT_XOR);
    }

    @Override
    public boolean enter(XbcAsgDivExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_DIV);
    }

    @Override
    public boolean enter(XbcAsgLshiftExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_LSHIFT);
    }

    @Override
    public boolean enter(XbcAsgMinusExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_MINUS);
    }

    @Override
    public boolean enter(XbcAsgModExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_MOD);
    }

    @Override
    public boolean enter(XbcAsgMulExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_MUL);
    }

    @Override
    public boolean enter(XbcAsgPlusExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_PLUS);
    }

    @Override
    public boolean enter(XbcAsgRshiftExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN_RSHIFT);
    }

    @Override
    public boolean enter(XbcAssignExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.ASSIGN);
    }

    @Override
    public boolean enter(XbcBitAndExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.BIT_AND);
    }

    @Override
    public boolean enter(XbcBitNotExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.BIT_NOT);
    }

    @Override
    public boolean enter(XbcBitOrExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.BIT_OR);
    }

    @Override
    public boolean enter(XbcBitXorExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.BIT_XOR);
    }

    @Override
    public boolean enter(XbcDivExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.DIV);
    }

    @Override
    public boolean enter(XbcLogAndExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_AND);
    }

    @Override
    public boolean enter(XbcLogEQExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_EQ);
    }

    @Override
    public boolean enter(XbcLogGEExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_GE);
    }

    @Override
    public boolean enter(XbcLogGTExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_GT);
    }

    @Override
    public boolean enter(XbcLogLEExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_LE);
    }

    @Override
    public boolean enter(XbcLogLTExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_LT);
    }

    @Override
    public boolean enter(XbcLogNEQExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_NEQ);
    }

    @Override
    public boolean enter(XbcLogNotExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_NOT);
    }

    @Override
    public boolean enter(XbcLogOrExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LOG_OR);
    }

    @Override
    public boolean enter(XbcLshiftExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.LSHIFT);
    }

    @Override
    public boolean enter(XbcModExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.MOD);
    }

    @Override
    public boolean enter(XbcMulExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.MUL);
    }

    @Override
    public boolean enter(XbcMinusExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.MINUS);
    }

    @Override
    public boolean enter(XbcPlusExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.PLUS);
    }

    @Override
    public boolean enter(XbcRshiftExpr visitable)
    {
        return _enterExpr(visitable, XcOperatorEnum.RSHIFT);
    }

    @Override
    public boolean enter(XbcCondExpr visitable)
    {
        XcOperatorObj obj = new XcOperatorObj(XcOperatorEnum.COND,
                                              new XcExprObj[3]);
        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);

        _enter(visitor, visitable.getExpressions1());
        _enter(visitor, visitable.getExpressions2());
        _enter(visitor, visitable.getExpressions3());
        return true;
    }

    @Override
    public boolean enter(XbcCommaExpr visitable)
    {
        XcOperatorObj obj = new XcOperatorObj(XcOperatorEnum.COMMA,
                                              new XcExprObj[visitable.sizeExpressions()]);
        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);
        _enter(visitor, visitable.getExpressions());

        return true;
    }

    @Override
    public boolean enter(XbcCastExpr visitable)
    {
        String typeId = visitable.getType();
        _ensureAttr(visitable, typeId, "type");
        XcType type;
        try {
            type = _identTableStack.getType(typeId);
        } catch(XmException e) {
            throw new XmBindingException(visitable, e);
        }
        XcCastObj obj = new XcCastObj(type);

        String isGccExtensionStr = visitable.getIsGccExtension();

        if (isGccExtensionStr != null)
            obj.setIsGccExtension(XmStringUtil.getAsBool((XmObj)visitable, isGccExtensionStr));

        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);

        return _enter(visitor, visitable.getContent());
    }

    private boolean _enterVar(IXbcVar var)
    {
        XcIdent ident = _getIdent(XcSymbolKindEnum.VAR, var);
        return _setAsLeaf(ident, (XmObj)var);
    }

    private boolean _enterSymbolAddr(IXbcSymbolAddr var)
    {
        XcIdent ident = _getIdentVarOrFunc(var);
        XcRefObj.Addr obj = new XcRefObj.Addr(ident);
        return _setAsLeaf(obj, (XmObj)var);
    }

    @Override
    public boolean enter(XbcVar visitable)
    {
        return _enterVar(visitable);
    }

    @Override
    public boolean enter(XbcArrayRef visitable)
    {

        XcArrayRefObj obj = new XcArrayRefObj();

	XbcArrayAddr array = visitable.getArrayAddr();

	XcIdent ident = _getIdent(XcSymbolKindEnum.VAR, array);
	XcVarObj arrayObj = new XcVarObj(ident);

	obj.setType(ident.getType());
	obj.setElementType(ident.getType().getRefType());
	obj.setArrayAddr(arrayObj);

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getExpressions());

        //return _enterVar(visitable);
    }

    @Override
    public boolean enter(XbcVarAddr visitable)
    {
        return _enterSymbolAddr(visitable);
    }

    @Override
    public boolean enter(XbcFuncAddr visitable)
    {
        XcIdent ident = _getIdentVarOrFunc(visitable);
        return _setAsLeaf(ident, visitable);
    }

    @Override
    public boolean enter(XbcArrayAddr visitable)
    {
        return _enterSymbolAddr(visitable);
    }

    private XcExprObj _shiftUpCoArray(XcRefObj refObj, IXbcTypedExpr visitable)
    {
        XcExprObj expr = refObj.getExpr();

        if((expr instanceof XcXmpCoArrayRefObj) == false)
            return refObj;

        XcXmpCoArrayRefObj coaRef =(XcXmpCoArrayRefObj)expr;
        String typeId = visitable.getType();
        XcType elemType = null;

        try {
            elemType = _identTableStack.getType(typeId);
        } catch (XmException e) {
            throw new XmBindingException((XmObj)visitable, "type " + typeId + "is not found ");
        }

        coaRef.turnOver(refObj, elemType);
        return coaRef;
    }

    @Override
    public boolean enter(XbcMemberAddr visitable)
    {
        XcIdent ident = _getIdentCompositeTypeMember(visitable);
        XcRefObj.MemberAddr obj = new XcRefObj.MemberAddr(ident);
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());

        XcExprObj expr = _shiftUpCoArray(obj, visitable);

        _setAsLeaf((XcObj)expr, visitable);
        return true;
    }

    @Override
    public boolean enter(XbcMemberRef visitable)
    {
        XcIdent ident = _getIdentCompositeTypeMember(visitable);
        XcRefObj.MemberRef obj = new XcRefObj.MemberRef(ident);
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());

        XcExprObj expr = _shiftUpCoArray(obj, visitable);

        _setAsLeaf((XcObj)expr, visitable);
        return true;
    }

    @Override
    public boolean enter(XbcMemberArrayAddr visitable)
    {
        XcIdent ident = _getIdentCompositeTypeMember(visitable);
        XcRefObj.MemberAddr obj = new XcRefObj.MemberAddr(ident);
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());

        XcExprObj expr = _shiftUpCoArray(obj, visitable);

        _setAsLeaf((XcObj)expr, visitable);
        return true;
    }

    @Override
    public boolean enter(XbcMemberArrayRef visitable)
    {
        XcIdent ident = _getIdentCompositeTypeMember(visitable);
        XcRefObj.MemberRef obj = new XcRefObj.MemberRef(ident);
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());

        XcExprObj expr = _shiftUpCoArray(obj, visitable);

        _setAsLeaf((XcObj)expr, visitable);
        return true;
    }

    @Override
    public boolean enter(XbcPointerRef visitable)
    {
        XcRefObj.PointerRef obj = new XcRefObj.PointerRef();
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());

        XcExprObj expr = obj.getExpr();

        if(expr instanceof XcXmpCoArrayRefObj) {
            XcXmpCoArrayRefObj coaRef =(XcXmpCoArrayRefObj)expr;
            XcType elemType = coaRef.getElementType();

            if(elemType.getTypeEnum() == XcTypeEnum.POINTER) {
                elemType = elemType.getRefType();
                coaRef.turnOver(obj, elemType);

                _setAsLeaf(coaRef, (XmObj)visitable);
                return true;
            } else if (coaRef.isNeedPointerRef()) {
                coaRef.unsetPointerRef();
                _setAsLeaf(coaRef, (XmObj)visitable);
                return true;
            }
        }
        _setAsLeaf(obj, (XmObj)visitable);
        return true;
    }

    @Override
    public boolean enter(XbcFunctionCall visitable)
    {
        XcFuncCallObj obj = new XcFuncCallObj();
        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);
        return _enter(visitor, visitable.getFunction(), visitable.getArguments());
    }

    @Override
    public boolean enter(XbcFunction visitable)
    {
        // function is combined to XmFuncCall
        return _enter(this, visitable.getExpressions());
    }

    @Override
    public boolean enter(XbcArguments visitable)
    {
        // arguments is combined to XmFuncCall
        return _enter(this, visitable.getExpressions());
    }

    public boolean _enterExprOrType(IXbcSizeOrAlignExpr visitable, XbcTypeName typeName, XcOperatorEnum opeEnum)
    {
        XcType _type = null;
        try {
            _type = _identTableStack.getType(typeName.getType());
        } catch (XmException e) {
            throw new XmBindingException((XmObj)visitable, e.getMessage());
        }

        XcSizeOfExprObj obj;
        obj =  new XcSizeOfExprObj(opeEnum, _type);

        if(_type instanceof XcLazyEvalType) {
            lazyEnter((XcLazyEvalType)_type);
            ((XcLazyEvalType)_type).setIsLazyEvalType(false);
        }

        return _setAsLeaf(obj, (XmObj)visitable);
    }

    public boolean _enterExprOrType(IXbcSizeOrAlignExpr visitable, IXbcExpressionsChoice expr, XcOperatorEnum opeEnum)
    {
        XcOperatorObj obj = new XcOperatorObj(opeEnum);
        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);
        return _enter(visitor, expr);
    }

    public boolean _enterExprOrType(IXbcSizeOrAlignExpr visitable,
                                    IXbcExprOrTypeChoice expr,
                                    XcOperatorEnum opeEnum)
    {
        if (expr instanceof IXbcExpressionsChoice) {
            return _enterExprOrType(visitable, (IXbcExpressionsChoice)expr, opeEnum);
        } else {
            return _enterExprOrType(visitable, (XbcTypeName)expr, opeEnum);
        }
    }

    @Override
    public boolean enter(XbcSizeOfExpr visitable)
    {
        return _enterExprOrType((IXbcSizeOrAlignExpr)visitable, visitable.getExprOrType(), XcOperatorEnum.SIZEOF);
    }

    @Override
    public boolean enter(XbcGccAlignOfExpr visitable)
    {
        return _enterExprOrType((IXbcSizeOrAlignExpr)visitable, visitable.getExprOrType(), XcOperatorEnum.ALIGNOF);
    }

    @Override
    public boolean enter(XbcGccLabelAddr visitable)
    {
        XcOperatorObj.LabelAddrExpr obj;
        obj =  (new XcOperatorObj()).new LabelAddrExpr(visitable.getContent());
        return _setAsLeaf(obj, (XmObj)visitable);
    }

    @Override
    public boolean enter(XbcGccAsmDefinition visitable)
    {
        XcGccAsmDefinition obj = new XcGccAsmDefinition();

        String isGccExtensionStr = visitable.getIsGccExtension();

        if (isGccExtensionStr != null)
            obj.setIsGccExtension(XmStringUtil.getAsBool((XmObj)visitable, isGccExtensionStr));

        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);
        return _enter(visitor, visitable.getStringConstant());
    }

    @Override
    public boolean enter(XbcPragma visitable)
    {
        XcDirectiveObj obj = new XcDirectiveObj();

        String directiveContent = visitable.getContent();
        obj.setLine("#pragma " + directiveContent);

        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcText visitable)
    {
        XcDirectiveObj obj = new XcDirectiveObj();

        String directiveContent = visitable.getContent();
        obj.setLine("# " + directiveContent);
        _setSourcePos(obj, visitable);
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcGccAsmStatement visitable)
    {
        XcGccAsmStmtObj obj = new XcGccAsmStmtObj();
        _setSourcePos(obj, visitable);
        XbcStringConstant sc = visitable.getStringConstant();
        if (sc == null)
            throw new XmBindingException((XmObj)visitable, "content is empty");

        obj.setAsmCode(new XcConstObj.StringConst(sc.getContent()));

        String isVolatileStr = visitable.getIsVolatile();

        if(isVolatileStr != null)
            obj.setIsVolatile(XmStringUtil.getAsBool(visitable, isVolatileStr));

        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);

        if(visitable.getGccAsmOperands1() != null) {
            obj.initInputOperands();
            _enter(visitor, visitable.getGccAsmOperands1());
        }

        obj.setInputOperandsEnd();

        if(visitable.getGccAsmOperands2() != null) {
            obj.initOutputOperands();
            _enter(visitor, visitable.getGccAsmOperands2());
        }

        return _enter(visitor, visitable.getGccAsmClobbers());
    }

    @Override
    public boolean enter(XbcGccAsmOperand visitable)
    {
        XcOperandObj obj = new XcOperandObj();

        obj.setMatch(visitable.getMatch());
        obj.setConstraint(visitable.getConstraint());

        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);

        return _enter(visitor, visitable.getExpressions());
    }

    @Override
    public boolean enter(XbcGccAsmOperands visitable)
    {
        return _enter(this, visitable.getGccAsmOperand());
    }

    @Override
    public boolean enter(XbcGccAsmClobbers visitable)
    {
        return _enter(this, visitable.getStringConstant());
    }

    @Override
    public boolean enter(XbcBuiltinOp visitable)
    {
        XcBltInOpObj obj = new XcBltInOpObj();

        String name = visitable.getName();
        _ensureAttr(visitable, name, "name");
        obj.setName(name);

        String isIdStr = visitable.getIsId();
        String isAddrOfStr = visitable.getIsAddrOf();

        if (isIdStr != null)
            obj.setIsId(XmStringUtil.getAsBool((XmObj)visitable, isIdStr));
        if (isAddrOfStr != null)
            obj.setIsAddrOf(XmStringUtil.getAsBool((XmObj)visitable, isAddrOfStr));

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getContent());
    }

    @Override
    public boolean enter(XbcTypeName visitable)
    {
        String typeId = visitable.getType();
        XcType _type;
        try {
            _type = _identTableStack.getType(typeId);
          _setAsLeaf(_type, visitable);
        } catch (XmException e) {
            throw new XmBindingException((XmObj)visitable, "type " + typeId + "is not found ");
        }

        return true;
    }

    @Override
    public boolean enter(XbcGccMemberDesignator visitable)
    {
        /* ex)
         * <gccMemberDesignator ref="S1" member="c1">
         *   <gccMemberDesignator ref="S0" member="c"/>
         * </gccMemberDesignator>
         *
         * 'ref' attribute must indicate type id of struct/union type.
         * 'member' attribute must indicate member name of struct/union type.
         */
        XcMemberDesignator obj = new XcMemberDesignator();

        String referenceStr = visitable.getRef();
        String memberStr = visitable.getMember();

        XcType refType = null;

        try {
            refType = _identTableStack.getType(referenceStr);
        } catch (XmException e) {
            throw new XmBindingException((XmObj)visitable, "type " + referenceStr + " is not found ");
        }

        if(refType == null)
            throw new XmBindingException((XmObj)visitable, "type " + referenceStr + " is not found ");

        refType = _identTableStack.getRealType(refType);

        if(refType instanceof XcCompositeType && memberStr != null) {
            XcCompositeType compType = (XcCompositeType)refType;
            XcIdent ident = compType.getMember(memberStr);

            if(ident == null)
                throw new XmBindingException((XmObj)visitable, "symbol '" + memberStr + "' is not a member of type '"
                                              + compType.getTypeId() + "'");

            obj.setMember(memberStr);
        }

        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);

        _enter(visitor, visitable.getExpressions(), visitable.getGccMemberDesignator());

        return (true);
    }

    @Override
    public boolean enter(XbcGccRangedCaseLabel visitable) {
        XcControlStmtObj.GccRangedCaseLabel obj = new XcControlStmtObj.GccRangedCaseLabel();
        _setSourcePos(obj, visitable);
        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getValue1(), visitable.getValue2());
    }

    @Override
    public boolean enter(XbcGccCompoundExpr visitable) {
        XcGccCompoundExprObj expr = new XcGccCompoundExprObj();
        XcBindingVisitor visitor = _setAsNode(expr, visitable);

        String isGccExtensionStr = visitable.getIsGccExtension();

        if (isGccExtensionStr != null)
            expr.setIsGccExtension(XmStringUtil.getAsBool((XmObj)visitable, isGccExtensionStr));

        return _enter(visitor, visitable.getCompoundStatement());
    }

    @Override
    public boolean enter(XbcCompoundValue visitable)
    {
        XcCompoundValueObj obj = new XcCompoundValueObj();

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getCompoundLiteral());
    }

    @Override
    public boolean enter(XbcCompoundValueExpr visitable)
    {
        String typeId = visitable.getType();
        _ensureAttr(visitable, typeId, "type");
        XcType type;
        try {
            type = _identTableStack.getType(typeId);
        } catch(XmException e) {
            throw new XmBindingException(visitable, e);
        }

        XcCompoundValueObj.Ref obj = new XcCompoundValueObj.Ref();

        obj.setType(type);

        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);

        return _enter(visitor, visitable.getValue());
    }

    @Override
    public boolean enter(XbcCompoundValueAddrExpr visitable)
    {
        String typeId = visitable.getType();
        _ensureAttr(visitable, typeId, "type");

        XcType type;
        try {
            type = _identTableStack.getType(typeId);
        } catch(XmException e) {
            throw new XmBindingException(visitable, e);
        }

        type = type.getRealType().getRefType();

        XcCompoundValueObj.AddrRef obj = new XcCompoundValueObj.AddrRef();

        obj.setType(type);

        XcBindingVisitor visitor = _setAsNode(obj, (XmObj)visitable);

        return _enter(visitor, visitable.getValue());
    }

    /**
     * Sets the argument to a parent and enters his children which is <br>
     * XcodeML binding object not yet visited by a XcBindingVisitor.<br>
     * A lazyEnter function is used to lazy evaluate XcodeML binding objects<br>
     * those are not able to be evaluate at some timig but another timing<br>
     * such as the timing after evaluating while variables.
     * 
     * @param lazyType has XcodeML binding objects are not visited by XcBindingVisitor.
     */
    public void lazyEnter(XcLazyEvalType lazyType) {
        if(lazyType.isLazyEvalType() == false)
            return;

        XcVarVisitor varVisitor = new XcVarVisitor(lazyType.getDependVar());

        for(IRVisitable visitable : lazyType.getLazyBindings()) {
            visitable.enter(varVisitor);
        }

        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)lazyType, _scopeEnum);

        _enter(visitor, lazyType.getLazyBindings());

        lazyType.setIsLazyEvalType(false);
    }

    /* XcarableMP extension */

    @Override
    public boolean enter(XbcCoArrayRef visitable)
    {
        XcXmpCoArrayRefObj obj = new XcXmpCoArrayRefObj();

	/*
        XbcName xname = visitable.getName();
        String name = XmStringUtil.trim(xname.getContent());
        _ensureAttr(xname, name, "name");

        XcIdent ident = _identTableStack.getIdent(XcSymbolKindEnum.VAR, name);
        XcExprObj content = ident;

        obj.setType(ident.getType());
        obj.setElementType(ident.getType().getRefType());

        if((obj.getElementType().getTypeEnum() == XcTypeEnum.ARRAY) == false) {
            content = new XcRefObj.Addr(content);
        }
	*/

	IXbcCoArrayRefChoice1 coarray = visitable.getContent();

	if (coarray instanceof XbcVar){
	    XcIdent ident = _getIdent(XcSymbolKindEnum.VAR, (XbcVar)coarray);
	    XcVarObj content = new XcVarObj(ident);
	    obj.setType(ident.getType());
	    obj.setElementType(ident.getType().getRefType());
	    obj.setContent(content);
	}
	else if (coarray instanceof XbcArrayRef){
	    XcArrayRefObj content = new XcArrayRefObj();

	    XbcArrayAddr array = ((XbcArrayRef)coarray).getArrayAddr();
	    XcIdent ident = _getIdent(XcSymbolKindEnum.VAR, array);
	    XcVarObj arrayObj = new XcVarObj(ident);

	    content.setType(ident.getType());
	    content.setElementType(ident.getType().getRefType());
	    content.setArrayAddr(arrayObj);

	    XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)content, _scopeEnum);
	    _enter(visitor, ((XbcArrayRef)coarray).getExpressions());
	    
	    obj.setType(ident.getType());
	    obj.setElementType(ident.getType().getRefType());
	    obj.setContent(content);

        }
	else if (coarray instanceof XbcSubArrayRef){
	    XcXmpSubArrayRefObj content = new XcXmpSubArrayRefObj();

	    XbcArrayAddr array = ((XbcSubArrayRef)coarray).getArrayAddr();
	    XcIdent ident = _getIdent(XcSymbolKindEnum.VAR, array);
	    XcVarObj arrayObj = new XcVarObj(ident);

	    content.setType(ident.getType());
	    content.setElementType(ident.getType().getRefType());
	    content.setArrayAddr(arrayObj);

	    XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)content, _scopeEnum);

	    _enter(visitor, ((XbcSubArrayRef)coarray).getSubArrayDimension());

	    obj.setType(ident.getType());
	    obj.setElementType(ident.getType().getRefType());
	    obj.setContent(content);

	}
	else if (coarray instanceof XbcMemberRef){
	    XcIdent ident = _getIdentCompositeTypeMember((XbcMemberRef)coarray);
	    XcRefObj.MemberRef content = new XcRefObj.MemberRef(ident);

	    XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, (XcNode)content, _scopeEnum);
	    _enter(visitor, ((XbcMemberRef)coarray).getExpressions());

	    obj.setType(ident.getType());
	    obj.setElementType(ident.getType().getRefType());
	    obj.setContent(content);
	}
	else {
            throw new XmBindingException(visitable, "content must be either Var, ArrayRef, SubArrayRef, or MemberRef.");
	}

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

        return _enter(visitor, visitable.getExpressions());
    }

//     @Override
//     public boolean enter(XbcCoArrayAssignExpr visitable)
//     {
//         XcXmpCoArrayAssignObj obj = new XcXmpCoArrayAssignObj();

//         XcBindingVisitor visitor = _setAsNode(obj, visitable);

//         return _enter(visitor, visitable.getExpressions1(), visitable.getExpressions2());
//     }

    @Override
    public boolean enter(XbcCoArrayType visitable)
    {
        String typeId = XmStringUtil.trim(visitable.getType());
        _ensureAttr(visitable, typeId, "type");

        XcXmpCoArrayType type = new XcXmpCoArrayType(typeId);

        return _enterArrayType((XcArrayLikeType)type, (IXbcArrayType)visitable);
    }

//     @Override
//     public boolean enter(XbcSubArrayRef visitable)
//     {
//         XcXmpSubArrayRefObj obj = new XcXmpSubArrayRefObj();

//         String typeId = _getChildTypeId(visitable.getExpressions());

//         XcType type = null;
//         try {
//             type = _identTableStack.getRealType(_identTableStack.getType(typeId));
//         } catch(XmException e) {
//             throw new XmBindingException(visitable, e);
//         }

//         if(type == null || type.getTypeEnum() != XcTypeEnum.ARRAY)
//             throw new XmBindingException(visitable, "invalid expression.");

//         obj.setArrayType(type);

//         XcBindingVisitor visitor = _setAsNode(obj, visitable);

//         _enter(visitor, visitable.getExpressions());
//         _enter(visitor, visitable.getSubArrayRefLowerBound());
//         _enter(visitor, visitable.getSubArrayRefUpperBound());
//         _enter(visitor, visitable.getSubArrayRefStep());

//         return true;
//     }

    @Override
    public boolean enter(XbcSubArrayRef visitable)
    {
        XcXmpSubArrayRefObj obj = new XcXmpSubArrayRefObj();

	XbcArrayAddr array = visitable.getArrayAddr();

	XcIdent ident = _getIdent(XcSymbolKindEnum.VAR, array);
	XcVarObj arrayObj = new XcVarObj(ident);

	obj.setType(ident.getType());
	obj.setElementType(ident.getType().getRefType());
	obj.setArrayAddr(arrayObj);

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

	//        _enter(visitor, visitable.getArrayAddr());
        _enter(visitor, visitable.getSubArrayDimension());

        return true;
    }

    @Override
    public boolean enter(XbcIndexRange visitable)
    {
        XcIndexRangeObj obj = new XcIndexRangeObj();

        XcBindingVisitor visitor = _setAsNode(obj, visitable);

	_enter(visitor, visitable.getLowerBound());
	_enter(visitor, visitable.getUpperBound());
	_enter(visitor, visitable.getStep());

        return true;
    }

    @Override
    public boolean enter(XbcLowerBound visitable)
    {
        XcIndexRangeObj.LowerBound obj = new XcIndexRangeObj.LowerBound();
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcUpperBound visitable)
    {
        XcIndexRangeObj.UpperBound obj = new XcIndexRangeObj.UpperBound();
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());
        return _setAsLeaf(obj, visitable);
    }

    @Override
    public boolean enter(XbcStep visitable)
    {
        XcIndexRangeObj.Step obj = new XcIndexRangeObj.Step();
        XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, obj, _scopeEnum);
        _enter(visitor, visitable.getExpressions());
        return _setAsLeaf(obj, visitable);
    }

//     @Override
//     public boolean enter(XbcSubArrayRefUpperBound visitable)
//     {
//         XcXmpSubArrayRefObj.UpperBound obj = new XcXmpSubArrayRefObj.UpperBound();
//         XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, obj, _scopeEnum);
//         _enter(visitor, visitable.getExpressions());
//         return _setAsLeaf(obj, visitable);
//     }

//     @Override
//     public boolean enter(XbcSubArrayRefLowerBound visitable)
//     {
//         XcXmpSubArrayRefObj.LowerBound obj = new XcXmpSubArrayRefObj.LowerBound();
//         XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, obj, _scopeEnum);
//         _enter(visitor, visitable.getExpressions());
//         return _setAsLeaf(obj, visitable);
//     }

//     @Override
//     public boolean enter(XbcSubArrayRefStep visitable)
//     {
//         XcXmpSubArrayRefObj.Step obj = new XcXmpSubArrayRefObj.Step();
//         XcBindingVisitor visitor = new XcBindingVisitor(_identTableStack, obj, _scopeEnum);
//         _enter(visitor, visitable.getExpressions());
//         return _setAsLeaf(obj, visitable);
//     }
    
    public void pushParamListIdentTable(XcParamList paramList)
    {
        XcIdentTable it = _identTableStack.push();
        for(XcIdent ident : paramList) {
            it.add(XcIdentTableEnum.MAIN, ident);
        }
    }
    
    public void popIdentTable()
    {
        _identTableStack.pop();
    }
}
