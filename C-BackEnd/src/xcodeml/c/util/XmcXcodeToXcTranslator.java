package xcodeml.c.util;

import xcodeml.c.decompile.*;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcArrayLikeType;
import xcodeml.c.type.XcArrayType;
import xcodeml.c.type.XcBaseType;
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
import xcodeml.util.XmStringUtil;
import xcodeml.util.XmDomUtil;
import xcodeml.util.XmTranslationException;
import xcodeml.util.XmException;
import static xcodeml.util.XmDomUtil.getElement;
import static xcodeml.util.XmDomUtil.getAttr;
import static xcodeml.util.XmDomUtil.getContent;
import static xcodeml.util.XmDomUtil.getContentText;
import static xcodeml.util.XmDomUtil.getAttrBool;
import static xcodeml.util.XmDomUtil.collectElements;

import org.w3c.dom.*;

import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;
import java.util.ArrayList;

/**
 * Translator of XcodeML DOM nodes to internal AST node.
 */
public class XmcXcodeToXcTranslator {
    enum ScopeEnum {
        GLOBAL, LOCAL,
    }

    public XmcXcodeToXcTranslator() {
        createVisitorMap(pairs);
    }

    public XcProgramObj trans(Document xcodeDoc) {
        Node rootNode = xcodeDoc.getDocumentElement();

        XcIdentTableStack itStack = new XcIdentTableStack();
        XcIdentTable itable = itStack.push();
        TranslationContext tc = new TranslationContext(itStack,
                                                       ScopeEnum.GLOBAL);

        XcProgramObj prog = createXcodeProgram(tc, rootNode);
        prog.setIdentTable(itable);

        _checkChild(prog);
        return prog;
    }

    private static void _checkChild(XcNode node) {
        if (node == null)
            return;

        node.checkChild();
        XcNode[] children = node.getChild();
        if (children != null) {
            for (XcNode child : children)
                _checkChild(child);
        }
    }

    private void trans(TranslationContext tc, Node n, XcNode parent) {
        XcodeNodeVisitor visitor = getVisitor(n.getNodeName());
        if (visitor == null) {
            throw new XmTranslationException(null, "unknown node : " + n.getNodeName());
        }
        visitor.enter(tc, n, parent);
    }

    private void transChildren(TranslationContext tc, Node n, XcNode parent) {
        NodeList list = n.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                continue;
            }
            trans(tc, childNode, parent);
        }
    }

    private void transChildren(TranslationContext tc, XcNode parent,
                               List<Node> childNodes) {
        for (Node childNode : childNodes) {
            if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                continue;
            }
            trans(tc, childNode, parent);
        }
    }

    private XcNode addChild(XcNode parent, XcNode obj) {
        if (parent != null) {
            parent.addChild(obj);
        }
        return parent;
    }

    private XcodeNodeVisitor getVisitor(String nodeName) {
        return visitorMap.get(nodeName);
    }

    abstract class XcodeNodeVisitor {
        public abstract void enter(TranslationContext tc, Node n, XcNode parent);
    }

    private XcProgramObj createXcodeProgram(TranslationContext tc, Node n) {
        XcProgramObj obj = new XcProgramObj();

        obj.setLanguage(getAttr(n, "language"));
        obj.setTime(getAttr(n, "time"));
        obj.setSource(getAttr(n, "source"));
        obj.setVersion(getAttr(n, "version"));
        obj.setCompilerInfo(getAttr(n, "compiler-info"));

        enterNodes(tc, obj,
                   getElement(n, "typeTable"),
                   getElement(n, "globalSymbols"),
                   getElement(n, "globalDeclarations"));

        return obj;
    }

    // globalDeclarations
    class GlobalDeclarationsNodeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            transChildren(tc, n, parent);
        }
    }

    // declarations
    class DeclarationsNodeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcDeclsObj obj = new XcDeclsObj();
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // exprStatement
    class ExprStatementNodeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcExprStmtObj obj = new XcExprStmtObj();
            setSourcePos(obj, n);
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // typeTable
    class TypeTableNodeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            transChildren(tc, n, parent);
            try {
                tc.identTableStack.resolveType();
            } catch (XmException e) {
                throw new XmTranslationException(n, e);
            }
        }
    }

    // globalSymbols
    class GlobalSymbolsNodeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            transChildren(tc, n, parent);

            XcLazyVisitor lazyVisitor = new XmcXcodeToXcLazyVisitor(tc);
            try {
                tc.identTableStack.resolveDependency(lazyVisitor);
            } catch (XmException e) {
                throw new XmTranslationException(n, e);
            }
        }
    }

    // symbols
    class SymbolsNodeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            transChildren(tc, n, parent);

            XcLazyVisitor lazyVisitor = new XmcXcodeToXcLazyVisitor(tc);
            try {
                tc.identTableStack.resolveDependency(lazyVisitor);
            } catch (XmException e) {
                throw new XmTranslationException(n, e);
            }
        }
    }

    // gccAttribute
    class GccAttributeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcGccAttribute obj = new XcGccAttribute();
            obj.setName(getAttr(n, "name"));
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // gccAttributes
    class GccAttributesVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcGccAttributeList attrs = new XcGccAttributeList();
            addChild(parent, attrs);
            transChildren(tc, n, attrs);
        }
    }

    // basicType
    class BasicTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            _ensureAttr(n, typeId, "type");

            String typeName = XmStringUtil.trim(getAttr(n, "name"));
            _ensureAttr(n, typeName, "name");

            XcBasicType type = new XcBasicType(typeId);
            type.setTempRefTypeId(typeName);

            _setTypeAttr(type, n);

            _addType(tc, type, n);

            _addGccAttribute(type, n);
        }
    }

    // arrayType
    class ArrayTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            _ensureAttr(n, typeId, "type");

            XcArrayType type = new XcArrayType(typeId);
            _setTypeAttr(type, n);
            type.setIsStatic(getAttrBool(n, "is_static"));

            _addGccAttribute(type, n);

            _enterArrayType(tc, type, n);
        }
    }

    // arraySize
    class ArraySizeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            enterNodes(tc, parent,
                       getElement(n, "expressions"));
        }
    }

    // pointerType
    class PointerTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            String refTypeId = XmStringUtil.trim(getAttr(n, "ref"));
            _ensureAttr(n, typeId, "type");
            _ensureAttr(n, refTypeId, "reference type");

            XcPointerType type = new XcPointerType(typeId);
            _setTypeAttr(type, n);
            type.setTempRefTypeId(refTypeId);

            _addType(tc, type, n);

            _addGccAttribute(type, n);
        }
    }

    // enumType
    class EnumTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            _ensureAttr(n, typeId, "type");

            XcEnumType type = new XcEnumType(typeId);
            _setTypeAttr(type, n);

            // XXX 全体的に見直す。

            Node symbolsNode = getElement(n, "symbols");
            if (symbolsNode != null) {
                boolean isAllIntConstant = true;
                int i = 0;
                NodeList nodeList = symbolsNode.getChildNodes();
                for (int nodeCount = 0; nodeCount < nodeList.getLength(); ++nodeCount) {
                    Node cn = nodeList.item(nodeCount);
                    if (cn.getNodeType() != Node.ELEMENT_NODE) {
                        continue;
                    }
                    String cnName = cn.getNodeName();
                    if ("id".equals(cnName)) {
                        Node nameNode = getElement(cn, "name");
                        String name =
                            XmStringUtil.trim(getContentText(nameNode));
                        _ensureAttr(nameNode, name, "name");

                        XcIdent.MoeConstant ident = new XcIdent.MoeConstant(name, type);

                        Node valueNode = getElement(cn, "value");
                        if (valueNode != null) {
                            Node valueChildNode = getContent(valueNode);
                            String valueChildNodeName = valueChildNode.getNodeName();
                            if ("intConstant".equals(valueChildNodeName)) {
                                i = XmStringUtil.getAsCInt(getContentText(valueChildNode));

                                ident.setValue(new XcConstObj.IntConst(i++, XcBaseTypeEnum.INT));
                            } else {
                                // FIXME 'expressions に属するものすべて' と
                                // いう判定はできないので、その他で
                                // まとめてある。
                                XcLazyEvalType lazyIdent = ident;
                                lazyIdent.setIsLazyEvalType(true);
                                lazyIdent.setLazyBindings(new Node[] { valueChildNode });

                                isAllIntConstant = false;
                            }
                        } else {
                            if (isAllIntConstant) {
                                ident.setValue(new XcConstObj.IntConst(i++, XcBaseTypeEnum.INT));
                            } else {
                                ident.setValue(null);
                            }
                        }

                        type.addEnumerator(ident);
                    }
                }
            }

            _addType(tc, type, n);

            _addGccAttribute(type, n);
        }
    }

    // structType
    class StructTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            _ensureAttr(n, typeId, "type");

            XcStructType type = new XcStructType(typeId);
            _addGccAttribute(type, n);

            _enterCompositeType(tc, type, n);
        }
    }

    // unionType
    class UnionTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            _ensureAttr(n, typeId, "type");

            XcUnionType type = new XcUnionType(typeId);
            _addGccAttribute(type, n);

            _enterCompositeType(tc, type, n);
        }
    }

    // functionType
    class FunctionTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            _ensureAttr(n, typeId, "type");

            XcFuncType type = new XcFuncType(typeId);
            _setTypeAttr(type, n);

            type.setIsInline(getAttrBool(n, "is_inline"));
            type.setIsStatic(getAttrBool(n, "is_static"));

            String returnTypeId = getAttr(n, "return_type");
            if (returnTypeId != null) {
                type.setTempRefTypeId(returnTypeId);
            }

            Node params = getElement(n, "params");
            if (params != null) {
                NodeList list = params.getChildNodes();
                for (int i = 0; i < list.getLength(); i++) {
                    Node cn = list.item(i);
                    if (cn.getNodeType() != Node.ELEMENT_NODE) {
                        continue;
                    }
                    String cnName = cn.getNodeName();
                    if ("name".equals(cnName)) {
                        String name = XmStringUtil.trim(getContentText(cn));
                        String paramTypeId = XmStringUtil.trim(getType(cn));
                        _ensureAttr(cn, paramTypeId, "type");

                        XcIdent ident = new XcIdent(name);
                        ident.setTempTypeId(paramTypeId);
                        type.addParam(ident);
                    } else if ("ellipsis".equals(cnName)) {
                        type.setIsEllipsised(true);
                    }
                }
            }

            // TODO if needed
            // throw exception if
            // param list is empty and argument has ellipsis

            _addGccAttribute(type, n);
            _addType(tc, type, n);
        }
    }

    // id
    class IdVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            // called by enter(<symbols> | <globalSymbols>)

            Node nameNode = getElement(n, "name");
            String name = getContentText(nameNode);
            _ensureAttr(nameNode, name, "name");

            String typeId = getType(n);
            if (typeId == null) {
                typeId = getType(nameNode);
                _ensureAttr(nameNode, typeId, "type");
            }

            XcIdent ident = new XcIdent(name);
            _addGccAttribute(ident, n);
            ident.setTempTypeId(typeId);
            ident.setIsGccExtension(getAttrBool(n, "is_gccExtension"));
            ident.setIsGccThread(getAttrBool(n, "is_gccThread"));

            try {
                ident.resolve(tc.identTableStack);
            } catch (XmException e) {
                throw new XmTranslationException(n, e);
            }

            String sclass = XmStringUtil.trim(getAttr(n, "sclass"));
            XcSymbolKindEnum kind = null;

            if (sclass != null) {
                if ("auto".equals(sclass)) {
                    //function type must not be decorated by 'auto',
                    //and other type do not need to be decorated by 'auto'.
                    //ident.setIsAuto(true);
                } else if ("extern".equals(sclass)) {
                    ident.setIsExtern(true);
                } else if ("extern_def".equals(sclass)) {
                    ident.setIsExternDef(true);
                } else if ("static".equals(sclass)) {
                    ident.setIsStatic(true);
                } else if ("register".equals(sclass)) {
                    ident.setIsRegister(true);
                } else if ("typedef_name".equals(sclass)) {
                    ident.setIsTypedef(true);
                    kind = XcSymbolKindEnum.TYPE;
                } else if ("param".equals(sclass)) {
                    ident.setVarKindEnum(XcVarKindEnum.PARAM);
                    kind = XcSymbolKindEnum.VAR;
                } else if ("label".equals(sclass)) {
                    kind = XcSymbolKindEnum.LABEL;
                } else if ("tagname".equals(sclass)) {
                    kind = XcSymbolKindEnum.TAGNAME;
                } else if ("moe".equals(sclass)) {
                    kind = XcSymbolKindEnum.MOE;
                }
            }

            XcType type = ident.getType();
            if (kind == null) {
                if (type != null &&
                    XcTypeEnum.FUNC.equals(type.getTypeEnum())) {
                    kind = XcSymbolKindEnum.FUNC;
                } else {
                    kind = XcSymbolKindEnum.VAR;
                }
            }

            if (XcSymbolKindEnum.VAR.equals(kind) &&
                ident.getVarKindEnum() == null) {
                switch(tc.scopeEnum) {
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
                tc.identTableStack.addIdent(kind, ident);
            } catch(XmException e) {
                throw new XmTranslationException(n, e);
            }

            /*
              if ident is member of enum 
              then create and add anonymous enum.
            */
            if (XcSymbolKindEnum.MOE.equals(kind)) {
                if (((XcEnumType)type).getTagName() == null) {
                    ident = new XcIdent(null);
                    ident.setType(type);

                    try {
                        tc.identTableStack.addAnonIdent(ident);
                    } catch(XmException e) {
                        throw new XmTranslationException(n, e);
                    }
                }
            }
        }
    }

    // varDecl
    class VarDeclVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdent(tc,
                                      XcSymbolKindEnum.VAR,
                                      getElement(n, "name"));

            XcDeclObj obj = new XcDeclObj(ident);

            setSourcePos(obj, n);
            String gccAsmCodeStr = null;
            Node gccAsmNode = getElement(n, "gccAsm");
            if (gccAsmNode != null) {
                Node gccAsmExprNode = getElement(gccAsmNode,
                                                 "gccAsmExpression");
                if (gccAsmExprNode != null) {
                    gccAsmCodeStr = getContentText(getElement(gccAsmExprNode,
                                                              "stringConstant"));
                }
            }
            obj.setGccAsmCode(gccAsmCodeStr);

            addChild(parent, obj);

            enterNodes(tc, obj,
                       getElement(n, "value"));
        }
    }

    // value
    class ValueVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            Node contentNode = getContent(n);
            Node parentNode = n.getParentNode();
            if (parentNode != null &&
                parentNode.getNodeName().equals("value")) {
                // inner 'value' node is treated as a 'compoundValue' node.
                enterCompoundValue(tc, n, parent);
            } else {
                enterNodes(tc, parent,
                           contentNode);
            }
        }
    }

    // designatedValue
    class DesignatedValueVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcDesignatedValueObj obj = new XcDesignatedValueObj();
            obj.setMember(getAttr(n, "member"));
            addChild(parent, obj);
            transChildren(tc, n, obj);
            enterNodesWithNull(tc, obj,
                               getContent(n));
        }
    }

    // functionDecl
    class FunctionDeclVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdentFunc(tc, getElement(n, "name"));
            XcType type = ident.getType();

            if ((type instanceof XcFuncType) == false) {
                throw new XmTranslationException(n, "symbol is declared as function, but type of symbol is not function.");
            }
            XcGccAttributeList attrs = ident.getGccAttribute();

            if (attrs != null && attrs.containsAttrAlias()) {
                if (ident.isDeclared()) {
                    return;
                }
            }
            ident.setDeclared();

            XcDeclObj obj = new XcDeclObj(ident);
            setSourcePos(obj, n);

            Node gccAsmNode = getElement(n, "gccAsm");
            if (gccAsmNode != null) {
                String gccAsmStr = getContentText(getElement(gccAsmNode, "stringConstant"));
                obj.setGccAsmCode(gccAsmStr);
            }

            addChild(parent, obj);
        }
    }

    // functionDefinition
    class FunctionDefinitionVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            Node paramsNode = getElement(n, "params");
            XcIdent ident = _getIdentFunc(tc,
                                          getElement(n, "name"),
                                          paramsNode);

            XcFuncDefObj obj = new XcFuncDefObj(ident);
            setSourcePos(obj, n);
            obj.setIsGccExtension(getAttrBool(n, "is_gccExtension"));

            XcParamList pList = ((XcFuncType) ident.getType()).getParamList();
            if (paramsNode == null && pList != null) {
                throw new XmTranslationException(n, "mismatch with type by parameter size.");
            }

            Iterator<XcIdent> pIdentIter = pList.iterator();

            if (paramsNode != null) {
                NodeList list = paramsNode.getChildNodes();
                for (int i = 0; i < list.getLength(); i++) {
                    Node cn = list.item(i);
                    if (cn.getNodeType() != Node.ELEMENT_NODE) {
                        continue;
                    }
                    String cnName = cn.getNodeName();
                    if ("name".equals(cnName)) {
                        if (pIdentIter.hasNext() == false) {
                            throw new XmTranslationException(n, "mismatch with type by parameter size.");
                        }
                        pIdentIter.next();

                        String name = XmStringUtil.trim(getContentText(cn));
                        String paramTypeId = XmStringUtil.trim(getType(cn));

                        _ensureAttr(cn, paramTypeId, "type");

                        XcIdent paramIdent = new XcIdent(name);
                        XcType  paramType;

                        try {
                            paramType = tc.identTableStack.getType(paramTypeId);
                        } catch (XmException e) {
                            throw new XmTranslationException(n, e);
                        }

                        // TODO if needed
                        // if(paramTypeId.equals(paramType.getTypeId()))
                        //    throw new XbcBindingException(visitable, "mismatch with type by paramemter type.");

                        paramIdent.setType(paramType);

                        obj.addParam(paramIdent);
                    }
                }

                if (pIdentIter.hasNext()) {
                    XcIdent restParamIdent = pIdentIter.next();
                    if (!(restParamIdent.getType() instanceof XcVoidType))
                        throw new XmTranslationException(n, "mismatch with type of parameter.");

                    obj.addParam(restParamIdent);
                }

                obj.setIsEllipsised(pList.isEllipsised());
            }

            XcIdentTable it = tc.identTableStack.push();
            ScopeEnum symContextEnum0 = tc.scopeEnum;
            tc.scopeEnum = ScopeEnum.LOCAL;

            enterNodes(tc, obj,
                       getElement(n, "symbols"),
                       //paramsNode, // FIXME
                       getElement(n, "body"),
                       getElement(n, "gccAttributes"));

            if (obj.getCompStmt() != null &&
                obj.getCompStmt().getIdentTable() == null) {
                obj.getCompStmt().setIdentTable(it);
            }

            tc.identTableStack.pop();
            tc.scopeEnum = symContextEnum0;

            addChild(parent, obj);
        }
    }

    // body
    class BodyVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            List<Node> childNodes = XmDomUtil.collectChildNodes(n);
            if (parent instanceof XcCompStmtObj ||
                (parent instanceof XcFuncDefObj && childNodes.size() == 1 &&
                 childNodes.get(0).getNodeName().equals("compoundStatement"))) {
                transChildren(tc, parent, childNodes);
            } else {
                XcCompStmtObj obj = new XcCompStmtObj();
                obj.setIdentTable(tc.identTableStack.getLast());
                addChild(parent, obj);
                transChildren(tc, obj, childNodes);
            }
        }
    }

    // compoundStatement
    class CompoundStatementNodeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcCompStmtObj obj = new XcCompStmtObj();
            setSourcePos(obj, n);
            addChild(parent, obj);
            obj.setIdentTable(tc.identTableStack.push());
            enterNodes(tc, obj,
                       getElement(n, "symbols"),
                       getElement(n, "declarations"),
                       getElement(n, "body"));

            tc.identTableStack.pop();
        }
    }

    // ifStatement
    class IfStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.If obj = new XcControlStmtObj.If();
            setSourcePos(obj, n);
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "condition"),
                       getElement(n, "then"),
                       getElement(n, "else"));
        }
    }

    // condition
    class ConditionVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            enterNodesWithNull(tc, parent,
                               getContent(n));
        }
    }

    // init
    class InitVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            enterNodesWithNull(tc, parent,
                               getContent(n));
        }
    }

    // iter
    class IterVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            enterNodes(tc, parent,
                       getContent(n));
        }
    }

    // then
    class ThenVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            List<Node> stmtNodes = XmDomUtil.collectChildNodes(n);
            if (stmtNodes.isEmpty()) {
                XcCompStmtObj obj = new XcCompStmtObj();
                XcDeclsObj declsObj = new XcDeclsObj();
                addChild(obj, declsObj);

                addChild(parent, obj);
            } else {
                transChildren(tc, parent, stmtNodes);
            }
        }
    }

    // else
    class ElseVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            transChildren(tc, n, parent);
        }
    }

    // whileStatement
    class WhileStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.While obj = new XcControlStmtObj.While();
            setSourcePos(obj, n);
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "condition"),
                       getElement(n, "body"));
        }
    }

    // doStatement
    class DoStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.Do obj = new XcControlStmtObj.Do();
            setSourcePos(obj, n);
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "condition"),
                       getElement(n, "body"));
        }
    }

    // forStatement
    class ForStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.For obj = new XcControlStmtObj.For();
            setSourcePos(obj, n);
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "init"),
                       getElement(n, "condition"),
                       getElement(n, "iter"),
                       getElement(n, "body"));
        }
    }

    // breakStatement
    class BreakStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.Break obj = new XcControlStmtObj.Break();
            setSourcePos(obj, n);
            addChild(parent, obj);
        }
    }

    // continueStatement
    class ContinueStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.Continue obj = new XcControlStmtObj.Continue();
            setSourcePos(obj, n);
            addChild(parent, obj);
        }
    }

    // returnStatement
    class ReturnStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.Return obj = new XcControlStmtObj.Return();
            setSourcePos(obj, n);
            addChild(parent, obj);
            enterNodesWithNull(tc, obj,
                               getContent(n));
        }
    }

    // gotoStatement
    class GotoStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.Goto obj = new XcControlStmtObj.Goto();
            setSourcePos(obj, n);
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // name
    class NameVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            // called by enter(XbGotoStatement)
            XcNameObj obj = new XcNameObj(getContentText(n));
            addChild(parent, obj);
        }
    }

    // switchStatement
    class SwitchStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.Switch obj = new XcControlStmtObj.Switch();
            setSourcePos(obj, n);
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "value"),
                       getElement(n, "body"));
        }
    }

    // caseLabel
    class CaseLabelVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.CaseLabel obj = new XcControlStmtObj.CaseLabel();
            setSourcePos(obj, n);
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "value"));
        }
    }

    // defaultLabel
    class DefaultLabelVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.DefaultLabel obj = new XcControlStmtObj.DefaultLabel();
            setSourcePos(obj, n);
            addChild(parent, obj);
        }
    }

    // statementLabel
    class StatementLabelVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String name = _getContentString(getElement(n, "name"));
            if (name == null) {
                throw new XmTranslationException(n, "no label name");
            }

            XcControlStmtObj.Label obj = new XcControlStmtObj.Label(name);
            setSourcePos(obj, n);
            addChild(parent, obj);
        }
    }

    // floatConstant
    class FloatConstantVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String text = _getContentString(n);

            if (text == null) {
                throw new XmTranslationException(n, "invalid float fraction/exponential");
            }

            XcConstObj.FloatConst obj;
            XcBaseTypeEnum btEnum;
            String typeId = XmStringUtil.trim(getType(n));

            if (typeId == null) {
                btEnum = XcBaseTypeEnum.DOUBLE;
            } else {
                btEnum = XcBaseTypeEnum.getByXcode(typeId);

                if (btEnum == null) {
                    throw new XmTranslationException(n, "invalid type '" + typeId + "' as float constant");
                }
            }

            obj = new XcConstObj.FloatConst(text, btEnum);
            addChild(parent, obj);
        }
    }

    // intConstant
    class IntConstantVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String text = _getContentString(n);
            if (text == null) {
                throw new XmTranslationException(n, "invalid constant value");
            }

            XcBaseTypeEnum btEnum;
            String typeId = XmStringUtil.trim(getType(n));

            if (typeId == null) {
                btEnum = XcBaseTypeEnum.INT;
            } else {
                btEnum = XcBaseTypeEnum.getByXcode(typeId);
            }

            if (btEnum == null) {
                throw new XmTranslationException(n, "invalid type '" + typeId + "' as int constant");
            }

            XcConstObj.IntConst obj;

            try {
                obj = new XcConstObj.IntConst(text, btEnum);
            } catch(XmException e) {
                throw new XmTranslationException(n, e);
            }

            addChild(parent, obj);
        }
    }

    // longlongConstant
    class LonglongConstantVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            XcConstObj.LongLongConst obj =
                XmcBindingUtil.createLongLongConst(getContentText(n), typeId);
            addChild(parent, obj);
        }
    }

    // stringConstant
    class StringConstaantVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String text = getContentText(n);
            XcConstObj.StringConst obj = new XcConstObj.StringConst(text);
            obj.setIsWide(getAttrBool(n, "is_wide"));
            addChild(parent, obj);
        }
    }

    // moeConstant
    class MoeConstantVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            String moe = XmStringUtil.trim(getContentText(n));

            _ensureAttr(n, typeId, "type");
            _ensureAttr(n, moe, "enumerator symbol");

            XcIdent ident = _getIdentEnumerator(tc, n, typeId, moe);
            addChild(parent, ident);
        }
    }

    // unaryMinusExpr
    class UnaryMinusVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterUnaryExpr(tc, n, parent,
                            XcOperatorEnum.UNARY_MINUS);
        }
    }

    // postDecrExpr
    class PostDecrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterUnaryExpr(tc, n, parent,
                            XcOperatorEnum.POST_DECR);
        }
    }

    // postIncrExpr
    class PostIncrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterUnaryExpr(tc, n, parent,
                            XcOperatorEnum.POST_INCR);
        }
    }

    // preDecrExpr
    class PreDecrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterUnaryExpr(tc, n, parent,
                            XcOperatorEnum.PRE_DECR);
        }
    }

    // preIncrExpr
    class PreIncrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterUnaryExpr(tc, n, parent,
                            XcOperatorEnum.PRE_INCR);
        }
    }

    // asgBitAndExpr
    class AsgBitAndExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_BIT_AND);
        }
    }

    // asgBitOrExpr
    class AsgBitOrExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_BIT_OR);
        }
    }

    // asgBitXorExpr
    class AsgBitXorExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_BIT_XOR);
        }
    }

    // asgDivExpr
    class AsgDivExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_DIV);
        }
    }

    // asgLshiftExpr
    class AsgLshiftExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_LSHIFT);
        }
    }

    // asgMinusExpr
    class AsgMinusExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_MINUS);
        }
    }

    // asgModExpr
    class AsgModExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_MOD);
        }
    }

    // asgMulExpr
    class AsgMulExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_MUL);
        }
    }

    // asgPlusExpr
    class AsgPlusExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_PLUS);
        }
    }

    // asgRshiftExpr
    class AsgRshiftExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN_RSHIFT);
        }
    }

    // assignExpr
    class AssignExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.ASSIGN);
        }
    }

    // bitAndExpr
    class BitAndExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.BIT_AND);
        }
    }

    // bitOrExpr
    class BitOrExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.BIT_OR);
        }
    }

    // bitNotExpr
    class BitNotExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterUnaryExpr(tc, n, parent,
                            XcOperatorEnum.BIT_NOT);
        }
    }

    // bitXorExpr
    class BitXorExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.BIT_XOR);
        }
    }

    // divExpr
    class DivExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.DIV);
        }
    }

    // logAndExpr
    class LogAndExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_AND);
        }
    }

    // logEQExpr
    class LogEQExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_EQ);
        }
    }

    // logGEExpr
    class LogGEExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_GE);
        }
    }

    // logGTExpr
    class LogGTExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_GT);
        }
    }

    // logLEExpr
    class LogLEExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_LE);
        }
    }

    // logLTExpr
    class LogLTExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_LT);
        }
    }

    // logNEQExpr
    class LogNEQExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_NEQ);
        }
    }

    // logNotExpr
    class LogNotExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterUnaryExpr(tc, n, parent,
                            XcOperatorEnum.LOG_NOT);
        }
    }

    // logOrExpr
    class LogOrExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LOG_OR);
        }
    }

    // LshiftExpr
    class LshiftExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.LSHIFT);
        }
    }

    // modExpr
    class ModExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.MOD);
        }
    }

    // mulExpr
    class MulExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.MUL);
        }
    }

    // minusExpr
    class MinusExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.MINUS);
        }
    }

    // plusExpr
    class PlusExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.PLUS);
        }
    }

    // RshiftExpr
    class RshiftExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterBinaryExpr(tc, n, parent,
                             XcOperatorEnum.RSHIFT);
        }
    }

    // condExpr
    class CondExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcOperatorObj obj = new XcOperatorObj(XcOperatorEnum.COND,
                                                  new XcExprObj[3]);
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // commaExpr
    class CommaExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            List<Node> exprNodes = XmDomUtil.collectChildNodes(n);
            XcOperatorObj obj = new XcOperatorObj(XcOperatorEnum.COMMA,
                                                  new XcExprObj[exprNodes.size()]);
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // castExpr
    class CastExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = getType(n);
            _ensureAttr(n, typeId, "type");
            XcType type;
            try {
                type = tc.identTableStack.getType(typeId);
            } catch (XmException e) {
                throw new XmTranslationException(n, e);
            }
            XcCastObj obj = new XcCastObj(type);
            obj.setIsGccExtension(getAttrBool(n, "is_gccExtension"));
            addChild(parent, obj);
            transChildren(tc, n, obj);
            
            XcLazyVisitor lazyVisitor = new XmcXcodeToXcLazyVisitor(tc);
	    lazyEval(type, lazyVisitor);
        }
        public void lazyEval(XcType type, XcLazyVisitor visitor)
        {
            _lazyEval(type, visitor); // lazy evalueate for type

            while(type instanceof XcBasicType) {
                type = type.getRefType();
            }
            switch(type.getTypeEnum()) { // lazy evaluate for child ident
            case STRUCT:
            case UNION:
                XcCompositeType ct = (XcCompositeType)type;

                if(ct.getMemberList() != null) {
                    for(XcIdent child : ct.getMemberList()) {
                        child.lazyEval(visitor);
                    }
                }
                break;
            case ENUM:
                XcEnumType et = (XcEnumType)type;

                if(et.getEnumeratorList() != null) {
                    for(XcIdent child : et.getEnumeratorList()) {
                      child.lazyEval(visitor);
                    }
                }
                break;
            case FUNC:
                XcFuncType ft = (XcFuncType)type;

                if(ft.getParamList() != null) {
                    visitor.pushParamListIdentTable(ft.getParamList());
                    for(XcIdent child : ft.getParamList()) {
                      child.lazyEval(visitor);
                    }
                    visitor.popIdentTable();
                }
                break;
            default:
                break;
            }
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
    }

    // Var
    class VarVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdent(tc, XcSymbolKindEnum.VAR, n);
            addChild(parent, ident);
        }
    }

    // arrayRef
    class ArrayRefVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcArrayRefObj obj = new XcArrayRefObj();
            addChild(parent, obj);

            enterArrayRef(tc, n, obj);
        }
    }

    // varAddr
    class VarAddrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterSymbolAddr(tc, n, parent);
        }
    }

    // funcAddr
    class FuncAddrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdentVarOrFunc(tc, n);
            addChild(parent, ident);
        }
    }

    // arrayAddr
    class ArrayAddrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            //_enterSymbolAddr(tc, n, parent);
            XcIdent ident = _getIdent(tc, XcSymbolKindEnum.VAR, n);
            addChild(parent, ident);
        }
    }

    // memberAddr
    class MemberAddrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdentCompositeTypeMember(tc, n);
            XcRefObj.MemberAddr obj = new XcRefObj.MemberAddr(ident);
            transChildren(tc, n, obj);

            XcExprObj expr = _shiftUpCoArray(tc, obj, n);
            addChild(parent, expr);
        }
    }

    // memberRef
    class MemberRefVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdentCompositeTypeMember(tc, n);
            XcRefObj.MemberRef obj = new XcRefObj.MemberRef(ident);
            transChildren(tc, n, obj);

            XcExprObj expr = _shiftUpCoArray(tc, obj, n);
            addChild(parent, expr);
        }
    }

    // memberArrayAddr
    class MemberArrayAddrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdentCompositeTypeMember(tc, n);
            XcRefObj.MemberAddr obj = new XcRefObj.MemberAddr(ident);
            transChildren(tc, n, obj);

            XcExprObj expr = _shiftUpCoArray(tc, obj, n);
            addChild(parent, expr);
        }
    }

    // memberArrayRef
    class MemberArrayRefVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIdent ident = _getIdentCompositeTypeMember(tc, n);
            XcRefObj.MemberRef obj = new XcRefObj.MemberRef(ident);
            transChildren(tc, n, obj);

            XcExprObj expr = _shiftUpCoArray(tc, obj, n);
            addChild(parent, expr);
        }
    }

    // pointerRef
    class PointerRefVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcRefObj.PointerRef obj = new XcRefObj.PointerRef();
            transChildren(tc, n, obj);

            XcExprObj expr = obj.getExpr();
            if (expr instanceof XcXmpCoArrayRefObj) {
                XcXmpCoArrayRefObj coaRef = (XcXmpCoArrayRefObj)expr;
                XcType elemType = coaRef.getElementType();

                if (elemType.getTypeEnum() == XcTypeEnum.POINTER) {
                    elemType = elemType.getRefType();
                    coaRef.turnOver(obj, elemType);

                    addChild(parent, coaRef);
                    return;
                } else if (coaRef.isNeedPointerRef()) {
                    coaRef.unsetPointerRef();
                    addChild(parent, coaRef);
                    return;
                }
            }
            addChild(parent, obj);
        }
    }

    // functionCall
    class FunctionCallVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcFuncCallObj obj = new XcFuncCallObj();
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "function"),
                       getElement(n, "arguments"));
        }
    }

    // function
    class FunctionVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            // function is combined to XmFuncCall
            transChildren(tc, n, parent);
        }
    }

    // arguments
    class ArgumentsVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            // arguments is combined to XmFuncCall
            transChildren(tc, n, parent);
        }
    }

    // sizeOfExpr
    class SizeOfExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterExprOrType(tc, n, parent, XcOperatorEnum.SIZEOF);
        }
    }

    // addrOfExpr
    class AddrOfExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcOperatorObj obj = new XcOperatorObj(XcOperatorEnum.ADDROF);
            transChildren(tc, n, obj);
            addChild(parent, obj);
        }
    }

    // gccAlignOfExpr
    class GccAlignOfExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            _enterExprOrType(tc, n, parent, XcOperatorEnum.ALIGNOF);
        }
    }

    // xmpDescOf
    class XmpDescOfVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcOperatorObj obj = new XcOperatorObj(XcOperatorEnum.XMPDESCOF);
            transChildren(tc, n, obj);
        }
    }

    // gccLabelAddr
    class GccLabelAddrVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcOperatorObj.LabelAddrExpr obj = 
                (new XcOperatorObj()).new LabelAddrExpr(getContentText(n));
            addChild(parent, obj);
        }
    }

    // gccAsmDefinition
    class GccAsmDefinitionVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcGccAsmDefinition obj = new XcGccAsmDefinition();
            obj.setIsGccExtension(getAttrBool(n, "is_gccExtension"));
            addChild(parent, obj);
            enterNodes(tc, obj,
                       getElement(n, "stringConstant"));
        }
    }

    // pragma
    class PragmaVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcDirectiveObj obj = new XcDirectiveObj();
            obj.setLine("#pragma " + getContentText(n));
            addChild(parent, obj);
        }
    }

    // OMPPragma
    class OMPPragmaVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "OMPPragma" element in XcodeML/F.
         */
        @Override
	public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcDirectiveObj obj = new XcDirectiveObj();
	    addChild(parent, obj);
            
            // directive
            Node dir = n.getFirstChild();
            String dirName = XmDomUtil.getContentText(dir).toLowerCase();
            
	    if (dirName.equals("parallel_for")) dirName = "parallel for";

	    obj.setLine("#pragma omp " + dirName);

            if (dirName.equals("barrier"))
		return;

            if (dirName.equals("threadprivate")){
		obj.addToken("(");
            	
            	NodeList varList = dir.getNextSibling().getChildNodes();
		enterNodes(tc, obj, varList.item(0));
		for (int j = 1; j < varList.getLength(); j++){
		    Node var = varList.item(j);
		    obj.addToken(",");
		    enterNodes(tc, obj, var);
		}
        
		obj.addToken(")");
		
		return;
            }
            
            // clause
            Node clause = dir.getNextSibling();

            NodeList list0 = clause.getChildNodes();
            for (int i = 0; i < list0.getLength(); i++){          
            	Node childNode = list0.item(i);
                if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                    continue;
                }
                
                String clauseName = XmDomUtil.getContentText(childNode).toLowerCase();
                String operator = "";
                if (clauseName.equals("data_default"))               clauseName = "default";
                else if (clauseName.equals("data_private"))          clauseName = "private";
                else if (clauseName.equals("data_shared"))           clauseName = "shared";
                else if (clauseName.equals("data_firstprivate"))     clauseName = "firstprivate";
                else if (clauseName.equals("data_lastprivate"))      clauseName = "lastprivate";
                else if (clauseName.equals("data_copyin"))           clauseName = "copyin";
                else if (clauseName.equals("data_reduction_plus"))  {clauseName = "reduction"; operator = "+";}
                else if (clauseName.equals("data_reduction_minus")) {clauseName = "reduction"; operator = "-";}
                else if (clauseName.equals("data_reduction_mul"))   {clauseName = "reduction"; operator = "*";}
                else if (clauseName.equals("data_reduction_bitand")){clauseName = "reduction"; operator = "iand";}
                else if (clauseName.equals("data_reduction_bitor")) {clauseName = "reduction"; operator = "ior";}
                else if (clauseName.equals("data_reduction_bitxor")){clauseName = "reduction"; operator = "ieor";}
                else if (clauseName.equals("data_reduction_logand")){clauseName = "reduction"; operator = ".and.";}
                else if (clauseName.equals("data_reduction_logor")) {clauseName = "reduction"; operator = ".or.";}
                else if (clauseName.equals("data_reduction_min"))   {clauseName = "reduction"; operator = "min";}
                else if (clauseName.equals("data_reduction_max"))   {clauseName = "reduction"; operator = "max";}
                else if (clauseName.equals("data_reduction_eqv"))   {clauseName = "reduction"; operator = ".eqv.";}
                else if (clauseName.equals("data_reduction_neqv"))  {clauseName = "reduction"; operator = ".neqv.";}
                else if (clauseName.equals("dir_ordered"))           clauseName = "ordered";
                else if (clauseName.equals("dir_if"))                clauseName = "if";
                else if (clauseName.equals("dir_nowait"))            clauseName = "nowait";
                else if (clauseName.equals("dir_schedule"))          clauseName = "schedule";
                else if (clauseName.equals("dir_num_threads"))       clauseName = "num_threads";
            
		obj.addToken(clauseName);
                
		Node arg = childNode.getFirstChild().getNextSibling();
		if (arg != null){
		    obj.addToken("(");
		    if (operator != "") obj.addToken(operator + " :");

            if (!arg.getNodeName().equals("list")){
                String text = XmDomUtil.getContentText(arg);
                if (clauseName.equals("if") || clauseName.equals("num_threads"))
                    obj.addToken(text);
                else
                {   // 'default' clause
                    String kind = text.toLowerCase();
                    if (kind.equals("default_shared")) kind = "shared";
                    else if (kind.equals("default_none")) kind = "none";
                    obj.addToken(kind);
                }
            }
            else {
		      NodeList varList = arg.getChildNodes();

		      String kind = XmDomUtil.getContentText(varList.item(0)).toLowerCase();
		      
		      if (kind.equals("sched_static")) obj.addToken("static");
		      else if (kind.equals("sched_dynamic")) obj.addToken("static");
		      else if (kind.equals("sched_guided")) obj.addToken("guided");
		      else if (kind.equals("sched_auto")) obj.addToken("auto");
		      else if (kind.equals("sched_runtime")) obj.addToken("runtime");
		      else enterNodes(tc, obj, varList.item(0));

		      for (int j = 1; j < varList.getLength(); j++){
			Node var = varList.item(j);
			obj.addToken(",");
			enterNodes(tc, obj, var);
		      }
		    }

		    obj.addToken(")");
		}
	    }
                
            // body
            Node body = clause.getNextSibling();

	    //            writer.incrementIndentLevel();

            NodeList list2 = body.getChildNodes();
            for (int i = 0; i < list2.getLength(); i++){
                Node childNode = list2.item(i);
                if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                    continue;
                }
		enterNodes(tc, obj, childNode);
            }

	    //            writer.decrementIndentLevel();
            
        }
    }
    
    class ACCPragmaVisitor extends XcodeNodeVisitor {
	/**
	 * Decompile "ACCPragma" element in XcodeML/F.
	 */
	@Override
	public void enter(TranslationContext tc, Node n, XcNode parent) {
	    XcDirectiveObj obj = new XcDirectiveObj();
	    addChild(parent, obj);
	    // directive
	    Node dir = n.getFirstChild();
	    String dirName = XmDomUtil.getContentText(dir).toLowerCase();

	    if (dirName.equals("parallel_loop"))     dirName = "parallel loop";
	    else if (dirName.equals("kernels_loop")) dirName = "kernels loop";
	    else if (dirName.equals("enter_data")) dirName = "enter data";
	    else if (dirName.equals("exit_data")) dirName = "exit data";

	    obj.setLine("#pragma acc " + dirName);

	    if (dirName.equals("wait")){
		Node arg = dir.getNextSibling();
		if(! arg.getNodeName().equals("list")){
		    obj.addToken("(");
		    enterIntExprNode(tc, obj, arg);
		    obj.addToken(")");
		}
		return;
	    }

	    if (dirName.equals("cache")){
		NodeList varList = dir.getNextSibling().getChildNodes();
		obj.addToken("(");
		enterVarListNode(tc, obj, varList);
		obj.addToken(")");
		return;
	    }

            if (dirName.equals("host_data use_device")){
              NodeList varList = dir.getNextSibling().getChildNodes();
              obj.addToken("(");
              enterVarListNode(tc, obj, varList);
              obj.addToken(")");

              Node clause = dir.getNextSibling();
              Node body = clause.getNextSibling();
              NodeList list2 = body.getChildNodes();
              for (int i = 0; i < list2.getLength(); i++){
                Node childNode = list2.item(i);
                if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                  continue;
                }
                enterNodes(tc, parent, childNode);
              }
              return;
            }

	    // clause
	    Node clause = dir.getNextSibling();
	    NodeList list0 = clause.getChildNodes();

	    for (int i = 0; i < list0.getLength(); i++){          
		Node childNode = list0.item(i);
		if (childNode.getNodeType() != Node.ELEMENT_NODE) {
		    continue;
		}

		Node clauseNameNode = childNode.getFirstChild();
		String clauseName = XmDomUtil.getContentText(clauseNameNode).toLowerCase();
		String operator = "";

		if (clauseName.equals("dev_resident"))          clauseName = "device_resident";	  
		else if (clauseName.equals("vect_len"))         clauseName = "vector_length";
        else if (clauseName.equals("routine_arg"))      clauseName = "";
		else if (clauseName.equals("reduction_plus"))  {clauseName = "reduction"; operator = "+";}
		else if (clauseName.equals("reduction_mul"))   {clauseName = "reduction"; operator = "*";}
		else if (clauseName.equals("reduction_bitand")){clauseName = "reduction"; operator = "&";}
		else if (clauseName.equals("reduction_bitor")) {clauseName = "reduction"; operator = "|";}
		else if (clauseName.equals("reduction_bitxor")){clauseName = "reduction"; operator = "^";}
		else if (clauseName.equals("reduction_logand")){clauseName = "reduction"; operator = "&&";}
		else if (clauseName.equals("reduction_logor")) {clauseName = "reduction"; operator = "||";}
		else if (clauseName.equals("reduction_min"))   {clauseName = "reduction"; operator = "min";}
		else if (clauseName.equals("reduction_max"))   {clauseName = "reduction"; operator = "max";}

		obj.addToken(clauseName);

		Node arg = childNode.getFirstChild().getNextSibling();

		if (arg != null){
		    obj.addToken("(");
		    if (operator != "") obj.addToken(operator + " :");

		    if (!arg.getNodeName().equals("list")){
			enterNodes(tc, obj, arg);
		    }
		    else {
			NodeList varList = arg.getChildNodes();
			enterVarListNode(tc, obj, varList);
		    }
		    obj.addToken(")");
		}
	    }

	    // body
	    Node body = clause.getNextSibling();
	    NodeList list2 = body.getChildNodes();
	    for (int i = 0; i < list2.getLength(); i++){
		Node childNode = list2.item(i);
		if (childNode.getNodeType() != Node.ELEMENT_NODE) {
		    continue;
		}
		enterNodes(tc, obj, childNode);
	    }
	}

	private void enterVarListNode(TranslationContext tc, XcDirectiveObj obj, NodeList nl){
	    enterVarNode(tc, obj, nl.item(0));
	    for (int j = 1; j < nl.getLength(); j++){
		Node var = nl.item(j);
		obj.addToken(",");
		enterVarNode(tc, obj, var);
	    }
	}

	private void enterIntExprNode(TranslationContext tc, XcDirectiveObj obj, Node n){
	    String text = XmDomUtil.getContentText(n);
	    obj.addToken(text);
	}

	private void enterVarNode(TranslationContext tc, XcDirectiveObj obj, Node n){
	    if(n.getNodeName().equals("list")){
		//array
		String arrayDim = "";
		NodeList array = n.getChildNodes();
		enterNodes(tc, obj, array.item(0));
		for (int j = 1; j < array.getLength(); j++){
		    Node index = array.item(j);
		    String indexStr = "";
		    obj.addToken("[");
		    if(index.getNodeName().equals("list")){
			NodeList range = index.getChildNodes();
			enterNodes(tc, obj, range.item(0));
			obj.addToken(":");
			String length = "";
			if(range.item(1) != null){
			    enterNodes(tc, obj, range.item(1));
			}
		    }else{
		        enterNodes(tc,  obj, array.item(j));
		    }
		    obj.addToken("]");
		}
		obj.addToken(arrayDim);
	    }else{
		//var
		enterNodes(tc, obj, n);
	    }
	}
    }

    // text
    class TextVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcDirectiveObj obj = new XcDirectiveObj();
	    //            obj.setLine("# " + getContentText(n));
            obj.setLine(getContentText(n));
            setSourcePos(obj, n);
            addChild(parent, obj);
        }
    }

    // gccAsmStatement
    class GccAsmStatementVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcGccAsmStmtObj obj = new XcGccAsmStmtObj();
            setSourcePos(obj, n);
            addChild(parent, obj);
            Node strConstNode = getElement(n, "stringConstant");
            if (strConstNode == null) {
                throw new XmTranslationException(n, "content is empty");
            }

            obj.setAsmCode(new XcConstObj.StringConst(getContentText(strConstNode)));
            obj.setIsVolatile(getAttrBool(n, "is_volatile"));

            List<Node> asmOperandsNodes = collectElements(n, "gccAsmOperands");
            if(asmOperandsNodes.get(0) != null){
                obj.initInputOperands();
                enterNodes(tc, obj, asmOperandsNodes.get(0));
                obj.setInputOperandsEnd();
            }
            if(asmOperandsNodes.size() >= 2){
              if(asmOperandsNodes.get(1) != null){
                obj.initOutputOperands();
                enterNodes(tc, obj, asmOperandsNodes.get(1));
              }
            }

            enterNodes(tc, obj,
                       getElement(n, "gccAsmClobbers"));
        }
    }

    // gccAsmOperand
    class GccAsmOperandVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcOperandObj obj = new XcOperandObj();

            obj.setMatch(getAttr(n, "match"));
            obj.setConstraint(getAttr(n, "constraint"));

            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // gccAsmOperands
    class GccAsmOperandsVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            transChildren(tc, n, parent);
        }
    }

    // gccAsmClobbers
    class GccAsmClobbersVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            transChildren(tc, n, parent);
        }
    }

    // builtin_op
    class BuiltinOpoVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcBltInOpObj obj = new XcBltInOpObj();
            addChild(parent, obj);

            String name = getAttr(n, "name");
            _ensureAttr(n, name, "name");
            obj.setName(name);

            obj.setIsId(getAttrBool(n, "is_id"));
            obj.setIsAddrOf(getAttrBool(n, "is_addrOf"));

            transChildren(tc, n, obj);
        }
    }

    // typeName
    class TypeNameVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = getType(n);
            XcType _type;
            try {
                _type = tc.identTableStack.getType(typeId);
                addChild(parent, _type);
            } catch (XmException e) {
                throw new XmTranslationException(n, "type " + typeId + "is not found ");
            }
        }
    }

    // gccMemberDesignator
    class GccMemberDesignatorVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            /* ex)
             * <gccMemberDesignator ref="S1" member="c1">
             *   <gccMemberDesignator ref="S0" member="c"/>
             * </gccMemberDesignator>
             *
             * 'ref' attribute must indicate type id of struct/union type.
             * 'member' attribute must indicate member name of struct/union type.
             */
            XcMemberDesignator obj = new XcMemberDesignator();

            String referenceStr = getAttr(n, "ref");
            String memberStr = getAttr(n, "member");

            XcType refType = null;
            try {
                refType = tc.identTableStack.getType(referenceStr);
            } catch (XmException e) {
                throw new XmTranslationException(n, "type " + referenceStr + " is not found ");
            }

            if (refType == null) {
                throw new XmTranslationException(n, "type " + referenceStr + " is not found ");
            }

            refType = tc.identTableStack.getRealType(refType);

            if (refType instanceof XcCompositeType && memberStr != null) {
                XcCompositeType compType = (XcCompositeType)refType;
                XcIdent ident = compType.getMember(memberStr);

                if (ident == null) {
                    throw new XmTranslationException(n, "symbol '" + memberStr + "' is not a member of type '" +
                                                     compType.getTypeId() + "'");
                }
                obj.setMember(memberStr);
            }

            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // gccRangedCaseLabel
    class GccRangedCaseLabelVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcControlStmtObj.GccRangedCaseLabel obj = new XcControlStmtObj.GccRangedCaseLabel();
            setSourcePos(obj, n);
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // gccCompoundExpr
    class GccCompoundExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcGccCompoundExprObj expr = new XcGccCompoundExprObj();
            addChild(parent, expr);

            expr.setIsGccExtension(getAttrBool(n, "is_gccExtension"));
            enterNodes(tc, expr,
                       getElement(n, "compoundStatement"));
        }
    }

    // compoundValue
    class CompoundValueVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            enterCompoundValue(tc, n, parent);
        }
    }

    // compoundValueExpr
    class CompoundValueExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = getType(n);
            _ensureAttr(n, typeId, "type");
            XcType type;
            try {
                type = tc.identTableStack.getType(typeId);
            } catch (XmException e) {
                throw new XmTranslationException(n, e);
            }

            XcCompoundValueObj.Ref obj = new XcCompoundValueObj.Ref();
            obj.setType(type);
            addChild(parent, obj);

            transChildren(tc, n, obj);
        }
    }

    // compoundValueAddrExpr
    class CompoundValueAddrExprVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = getType(n);
            _ensureAttr(n, typeId, "type");

            XcType type;
            try {
                type = tc.identTableStack.getType(typeId);
            } catch (XmException e) {
                throw new XmTranslationException(n, e);
            }

            type = type.getRealType().getRefType();

            XcCompoundValueObj.AddrRef obj = new XcCompoundValueObj.AddrRef();
            obj.setType(type);
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    /* XcarableMP extension */

    // coArrayRef
    class CoArrayRefVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcXmpCoArrayRefObj obj = new XcXmpCoArrayRefObj();
            List<Node> childNodes = XmDomUtil.collectChildNodes(n);
            
            Node coArrayRefChoiceNode = childNodes.remove(0);

            String nodeName = coArrayRefChoiceNode.getNodeName();
            if (nodeName.equals("Var")) {
                XcIdent ident = _getIdent(tc, XcSymbolKindEnum.VAR,
                                          coArrayRefChoiceNode);
                XcVarObj content = new XcVarObj(ident);
                obj.setType(ident.getType());
                obj.setElementType(ident.getType().getRefType());
                obj.setContent(content);
            } else if (nodeName.equals("arrayRef")) {
                XcArrayRefObj content = new XcArrayRefObj();
                enterArrayRef(tc, coArrayRefChoiceNode, content);
            } else if (nodeName.equals("subArrayRef")) {
                XcXmpSubArrayRefObj content = new XcXmpSubArrayRefObj();
                enterArrayRef(tc, coArrayRefChoiceNode, content);
            } else if (nodeName.equals("memberRef")) {
                XcIdent ident = _getIdentCompositeTypeMember(tc,
                                                             coArrayRefChoiceNode);
                XcRefObj.MemberRef content = new XcRefObj.MemberRef(ident);
                addChild(parent, content);
                enterNodes(tc, content, getContent(coArrayRefChoiceNode));

                obj.setType(ident.getType());
                obj.setElementType(ident.getType().getRefType());
                obj.setContent(content);
            } else {
                throw new XmTranslationException(n, "content must be either Var, ArrayRef, SubArrayRef, or MemberRef.");
            }

            addChild(parent, obj);
            transChildren(tc, parent, childNodes);
        }
    }

    // coArrayType
    class CoArrayTypeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            String typeId = XmStringUtil.trim(getType(n));
            _ensureAttr(n, typeId, "type");

            XcXmpCoArrayType type = new XcXmpCoArrayType(typeId);

            _enterArrayType(tc, (XcArrayLikeType)type, n);
        }
    }

    // subArrayRef
    class SubArrayRefVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcXmpSubArrayRefObj obj = new XcXmpSubArrayRefObj();
            addChild(parent, obj);

            enterArrayRef(tc, n, obj);
        }
    }

    // indexRange
    class IndexRangeVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIndexRangeObj obj = new XcIndexRangeObj();
            addChild(parent, obj);

            enterNodes(tc, obj,
                       getElement(n, "lowerBound"),
                       getElement(n, "upperBound"),
                       getElement(n, "step"));
        }
    }

    // lowerBound
    class LowerBoundVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIndexRangeObj.LowerBound obj = new XcIndexRangeObj.LowerBound();
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // upperBound
    class UpperBoundVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIndexRangeObj.UpperBound obj = new XcIndexRangeObj.UpperBound();
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }

    // step
    class StepBoundVisitor extends XcodeNodeVisitor {
        @Override
        public void enter(TranslationContext tc, Node n, XcNode parent) {
            XcIndexRangeObj.Step obj = new XcIndexRangeObj.Step();
            addChild(parent, obj);
            transChildren(tc, n, obj);
        }
    }


    void enterNodes(TranslationContext tc, XcNode obj, Node ... nodes) {
        enterNodes(tc, obj, false, nodes);
    }

    void enterNodes(TranslationContext tc, XcNode obj, boolean isAllowNull,
                    Node ... nodes) {
        for (Node n : nodes) {
            if (n == null) {
                if (isAllowNull) {
                    addChild(obj, XcNullExpr.createXcNullExpr());
                }
            } else {
                trans(tc, n, obj);
            }
        }
    }

    void enterNodesWithNull(TranslationContext tc, XcNode obj, Node ... nodes) {
        enterNodes(tc, obj, true, nodes);
    }

    void setSourcePos(XcSourcePositioned obj, Node n) {
        obj.setSourcePos(
            new XcSourcePosObj(getAttr(n, "file"),
                               getAttr(n, "lineno"),
                               getAttr(n, "rawlineno")));
    }

    String getType(Node n) {
        return getAttr(n, "type");
    }

    class XmcXcodeToXcLazyVisitor implements XcLazyVisitor {
        public XmcXcodeToXcLazyVisitor(TranslationContext tc) {
            this.tc = tc;
        }

        @Override
        public void lazyEnter(XcLazyEvalType lazyType) {
            if (lazyType.isLazyEvalType() == false) {
                return;
            }

            XmcVarCollector varCollector = new XmcVarCollector(lazyType.getDependVar());
            for (Node n : lazyType.getLazyBindingNodes()) {
                varCollector.collectVars(n);
            }

            enterNodes(tc, (XcNode)lazyType,
                       lazyType.getLazyBindingNodes());

            lazyType.setIsLazyEvalType(false);
        }

        public void pushParamListIdentTable(XcParamList paramList) {
            XcIdentTable it = tc.identTableStack.push();
            for (XcIdent ident : paramList) {
                it.add(XcIdentTableEnum.MAIN, ident);
            }
        }

        public void popIdentTable() {
            tc.identTableStack.pop();
        }

        private TranslationContext tc;
    }

    static class TranslationContext {
        public XcIdentTableStack identTableStack;
        public ScopeEnum scopeEnum;
        Stack<XcNode> parentNodeStack;
        public TranslationContext(XcIdentTableStack identTableStack,
                                  ScopeEnum scopeEnum) {
            this.identTableStack = identTableStack;
            this.scopeEnum = scopeEnum;
            parentNodeStack = new Stack<XcNode>();
        }

        public void pushParentNode(XcNode node) {
            parentNodeStack.push(node);
        }
        public XcNode popParentNode() {
            return parentNodeStack.pop();
        }
    }
    
    void _enterArrayType(TranslationContext tc,
                         XcArrayLikeType type, Node arrayTypeNode) {
        String arraySizeStr = getAttr(arrayTypeNode, "array_size");

        if (arraySizeStr == null) {
            type.setIsArraySize(false);
            type.setIsArraySizeExpr(false);
        } else if (arraySizeStr.equals("*")) {
            type.setIsArraySize(false);
            type.setIsArraySizeExpr(true);

            Node arraySizeExprNode = getContent(getElement(arrayTypeNode, "arraySize"));

            XcLazyEvalType lazyType = type;
            lazyType.setIsLazyEvalType(true);

            lazyType.setLazyBindings(new Node[] { arraySizeExprNode });
        } else {
            type.setIsArraySize(true);
            type.setIsArraySizeExpr(false);

            type.setArraySize(XmStringUtil.getAsCInt(arraySizeStr));
        }

        type.setTempRefTypeId(getAttr(arrayTypeNode, "element_type"));

        _addType(tc, (XcType)type, arrayTypeNode);
    }

    void _enterCompositeType(TranslationContext tc,
                             XcCompositeType type, Node typeNode)
    {
        _setTypeAttr(type, typeNode);

        Node symbolsNode = getElement(typeNode, "symbols");
        if (symbolsNode == null) {
            type.setMemberList(null);
            _addType(tc, type, typeNode);
            return;
        }

        NodeList nodeList = symbolsNode.getChildNodes();
        for (int i = 0; i < nodeList.getLength(); ++i) {
            Node cn = nodeList.item(i);
            if (cn.getNodeType() != Node.ELEMENT_NODE) {
                continue;
            }
            String cnName = cn.getNodeName();
            if ("id".equals(cnName)) {
                String typeId = XmStringUtil.trim(getType(cn));
                Node nameNode = getElement(cn, "name");
                if (typeId == null && nameNode != null) {
                    typeId = XmStringUtil.trim(getType(nameNode));
                }

                _ensureAttr(cn, typeId, "type");

                String name = null;
                if (nameNode != null) {
                    name = XmStringUtil.trim(getContentText(nameNode));
                }

                XcIdent ident = new XcIdent(name);
                ident.setTempTypeId(typeId);
                ident.setIsGccExtension(getAttrBool(cn, "is_gccExtension"));
                ident.setIsGccThread(getAttrBool(cn, "is_gccThread"));

                String bitFieldStr = getAttr(cn, "bit_field");
                if (bitFieldStr != null) {
                    if (bitFieldStr.equals("*")) {
                        Node bitFieldExprNode = getElement(cn, "bitField");

                        if (bitFieldExprNode == null) {
                            throw new XmTranslationException(bitFieldExprNode,
                                                             "bitFidld must be specified");
                        }

                        ident.setIsBitField(true);
                        ident.setIsBitFieldExpr(true);

                        XcLazyEvalType lazyIdent = (XcLazyEvalType)ident;
                        lazyIdent.setIsLazyEvalType(true);
                        lazyIdent.setLazyBindings(new Node[] { getContent(bitFieldExprNode) });
                    } else {
                        ident.setIsBitField(true);
                        ident.setIsBitFieldExpr(false);
                        ident.setBitField(XmStringUtil.getAsCInt(bitFieldStr));
                    }
                }

                _addGccAttribute(ident, typeNode);

                type.addMember(ident);
            }
        }

        _addType(tc, type, typeNode);
    }

    static void _ensureAttr(Node n, Object attr, String msg) {
        if (attr == null ||
            (attr instanceof String && XmStringUtil.trim((String)attr) == null))
            throw new XmTranslationException(n, "no " + msg);
    }

    static void _setTypeAttr(XcType type, Node n) {
        type.setIsConst(getAttrBool(n, "is_const"));
        type.setIsVolatile(getAttrBool(n, "is_volatile"));
        type.setIsRestrict(getAttrBool(n, "is_restrict"));
    }

    void _addType(TranslationContext tc, XcType type, Node n) {
        try {
            tc.identTableStack.addType(type);
        } catch (XmException e) {
            throw new XmTranslationException(n, e);
        }
    }

    void _addGccAttribute(XcGccAttributable parent, Node n) {
        Node attrs = getElement(n, "gccAttributes");
        if (attrs != null) {
            /*
             * set XcGccAttributeList as LazyEvalType
             */
            parent.setGccAttribute(new XcGccAttributeList(attrs));
        } else {
            parent.setGccAttribute(null);
        }
    }

    String _getContentString(Node contentTextIncludedNode) {
        if (contentTextIncludedNode == null)
            return null;

        return XmStringUtil.trim(getContentText(contentTextIncludedNode));
    }

    String _getContent(Node n) {
        String name = XmStringUtil.trim(getContentText(n));
        if (name == null) {
            throw new XmTranslationException(n, "content is empty");
        }
        return name;
    }

    XcIdent _getIdent(TranslationContext tc,
                      XcSymbolKindEnum kind, Node nameNode) {
        String name = _getContent(nameNode);
        XcIdent ident = tc.identTableStack.getIdent(kind, name);

        if (ident == null) {
            throw new XmTranslationException(nameNode, "variable '" + name + "' is not defined");
        }

        return ident;
    }

    XcIdent _getIdentVarOrFunc(TranslationContext tc, Node stringContentNode) {
        String name = _getContent(stringContentNode);

        if (name == null) {
            throw new XmTranslationException(stringContentNode, "variable or function name is not specified");
        }

        XcIdent ident = tc.identTableStack.getIdent(XcSymbolKindEnum.VAR, name);

        if (ident == null) {
            ident = tc.identTableStack.getIdent(XcSymbolKindEnum.FUNC, name);

            if (ident == null) {
                if (name.startsWith("_XMP_") ||name.startsWith("xmp_") ||
		    name.startsWith("xmpc_")) return new XcIdent(name);
                if (name.startsWith("_ACC_") || name.startsWith("acc_")) return new XcIdent(name);

                throw new XmTranslationException(stringContentNode, "variable or function '" + name
                    + "' is not defined");
            }
        }

        return ident;
    }

    XcIdent _getIdentFunc(TranslationContext tc, Node nameNode) {
        return _getIdent(tc, XcSymbolKindEnum.FUNC, nameNode);
    }

    XcIdent _getIdentFunc(TranslationContext tc,
                          Node nameNode, Node paramsNode) {
        XcIdent ident = _getIdentFunc(tc, nameNode);
        XcType type = ident.getType();

        if (XcTypeEnum.FUNC.equals(type.getTypeEnum()) == false) {
            throw new XmTranslationException(nameNode, "symbol '" + ident.getSymbol()
                + "' is not function type");
        }

        XcFuncType funcType = (XcFuncType)type;
        XcParamList paramList = funcType.getParamList();

        ArrayList<Node> paramNameNodes = null;
        if (paramsNode != null) {
            paramNameNodes = XmDomUtil.collectElements(paramsNode, "name");
        }
        if (paramList.isEmpty() == false) {
            // TODO strict parameter check
            int sizeName = (paramNameNodes != null) ? paramNameNodes.size() : 0;
            if (paramList.isVoid() == false && (sizeName != paramList.size())) {
                
                throw new XmTranslationException(paramsNode,
                    "parameter type is not applicable as function '" + type.getTypeId() + "'");
            }
        } else {
            // replace explicit parameters instead of empty parameters.

            funcType = funcType.copy();
            ident.setType(funcType);

            for (Node n : paramNameNodes) {
                String paramTypeId = XmStringUtil.trim(getType(n));
                _ensureAttr(n, paramTypeId, "type");
                String paramName = _getContent(n);
                XcIdent paramIdent = new XcIdent(paramName);
                paramIdent.setTempTypeId(paramTypeId);
                funcType.addParam(paramIdent);
            }

            try {
                funcType.getParamList().resolve(tc.identTableStack);
            } catch(XmException e) {
                throw new XmTranslationException(nameNode, e);
            }
        }

        return ident;
    }

    XcIdent _getIdentEnumerator(TranslationContext tc,
                                Node n, String typeId, String moe) {
        XcEnumType enumType;

        try {
            enumType = (XcEnumType)tc.identTableStack.getTypeAs(XcTypeEnum.ENUM, typeId);
        } catch (XmException e) {
            throw new XmTranslationException(n, e);
        }

        XcIdent ident = enumType.getEnumerator(moe);

        if (ident == null) {
            throw new XmTranslationException(n, "enum type '" + typeId
                                             + "' does not have enumerator '" + moe + "'");
        }

        return ident;
    }

    String _getChildTypeIdMember(Node memberNode) {
        Node exprNode = getContent(memberNode);

        if (exprNode == null) 
            throw new XmTranslationException(memberNode, "no composite type is specified");

        // if ((exprNode instanceof IXbcTypedExpr) == false) {
        //     throw new XmTranslationException(memberNode, "invalid expression");
        // }

        return getType(exprNode);
    }

    XcIdent _getIdentCompositeTypeMember(TranslationContext tc,
                                         Node memberNode) {
        String typeId = _getChildTypeIdMember(memberNode);
        String member = getAttr(memberNode, "member");
        _ensureAttr(memberNode, typeId, "type");
        _ensureAttr(memberNode, member, "member");
        XcType ptype, type;

        try {
            ptype = tc.identTableStack.getType(typeId).getRealType();
            type = ptype.getRefType().getRealType();
        } catch (XmException e) {
            throw new XmTranslationException(memberNode, e);
        }

        if ((type instanceof XcCompositeType) == false) {
            throw new XmTranslationException(memberNode, "type '" + typeId + "' is not struct/union pointer type");
        }

        XcCompositeType compType = (XcCompositeType)type;
        XcIdent ident = compType.getMember(member);

        if (ident == null) {
            throw new XmTranslationException(memberNode, "symbol '" + member + "' is not a member of type '"
                + compType.getTypeId() + "'");
        }
        return ident;
    }

    XcExprObj _shiftUpCoArray(XcOperatorObj op) {
        if (op.getOperatorEnum() != XcOperatorEnum.PLUS) {
            return op;
        }

        XcExprObj exprs[] = op.getExprObjs();

        if ((exprs[0] instanceof XcXmpCoArrayRefObj) == false) {
            return op;
        }

        XcXmpCoArrayRefObj coaRefObj = (XcXmpCoArrayRefObj)exprs[0];

        XcType elemetType = coaRefObj.getElementType().getRealType();

        if ((elemetType.getTypeEnum() == XcTypeEnum.ARRAY) == false) {
            return op;
        }

        XcArrayType at = (XcArrayType)elemetType;

        XcPointerType pt = new XcPointerType("XMPP", at.getRefType());

        coaRefObj.turnOver(op, pt);

        return coaRefObj;
    }

    XcExprObj _shiftUpCoArray(TranslationContext tc,
                              XcRefObj refObj, Node typedNode) {
        XcExprObj expr = refObj.getExpr();

        if ((expr instanceof XcXmpCoArrayRefObj) == false) {
            return refObj;
        }

        XcXmpCoArrayRefObj coaRef = (XcXmpCoArrayRefObj)expr;
        String typeId = getType(typedNode);
        XcType elemType = null;

        try {
            elemType = tc.identTableStack.getType(typeId);
        } catch (XmException e) {
            throw new XmTranslationException(typedNode, "type " + typeId + "is not found ");
        }

        coaRef.turnOver(refObj, elemType);
        return coaRef;
    }

    void _enterUnaryExpr(TranslationContext tc,
                         Node unaryExprNode, XcNode parent,
                         XcOperatorEnum opeEnum) {
        XcOperatorObj obj = new XcOperatorObj(opeEnum);
        addChild(parent, obj);
        enterNodes(tc, obj,
                   getContent(unaryExprNode));
    }

    void _enterBinaryExpr(TranslationContext tc,
                          Node binExprNode, XcNode parent,
                          XcOperatorEnum opeEnum) {
        XcOperatorObj obj = new XcOperatorObj(opeEnum);
        transChildren(tc, binExprNode, obj);
        XcExprObj expr = _shiftUpCoArray(obj);
        addChild(parent, expr);
    }

    XcNode _enterVar(TranslationContext tc, Node varNode, XcNode parent) {
        XcIdent ident = _getIdent(tc, XcSymbolKindEnum.VAR, varNode);
        addChild(parent, ident);
        return ident;
    }

    void _enterSymbolAddr(TranslationContext tc, Node varNode, XcNode parent) {
        XcIdent ident = _getIdentVarOrFunc(tc, varNode);
        XcRefObj.Addr obj = new XcRefObj.Addr(ident);
        addChild(parent, obj);
    }

    void _enterExprOrType(TranslationContext tc, Node sizeOrAlignNode,
                          XcNode parent, XcOperatorEnum opeEnum) {
        Node childNode = getContent(sizeOrAlignNode);
        if ("typeName".equals(childNode.getNodeName())) {
            XcType _type = null;
            try {
                _type = tc.identTableStack.getType(getType(childNode));
            } catch (XmException e) {
                throw new XmTranslationException(sizeOrAlignNode, e);
            }

            XcSizeOfExprObj obj;
            obj = new XcSizeOfExprObj(opeEnum, _type);

            if (_type instanceof XcLazyEvalType) {
                XcLazyVisitor lazyVisitor = new XmcXcodeToXcLazyVisitor(tc);
                lazyVisitor.lazyEnter((XcLazyEvalType)_type);
                ((XcLazyEvalType)_type).setIsLazyEvalType(false);
            }

            addChild(parent, obj);
        } else {
            // expressions
            XcOperatorObj obj = new XcOperatorObj(opeEnum);
            addChild(parent, obj);
            enterNodes(tc, obj, childNode);
        }
    }

    void enterCompoundValue(TranslationContext tc, Node n, XcNode parent) {
        XcCompoundValueObj obj = new XcCompoundValueObj();
        addChild(parent, obj);
        transChildren(tc, n, obj);
    }

    void enterArrayRef(TranslationContext tc, Node arrayRefNode, 
                       XcNode arrayRefObj) {
        List<Node> childNodes = XmDomUtil.collectChildNodes(arrayRefNode);
        Node arrayAddrNode = childNodes.remove(0);
        String nodeName = arrayAddrNode.getNodeName();
        if (! nodeName.equals("arrayAddr") && ! nodeName.equals("Var")) {
            throw new XmTranslationException(arrayRefNode, "Invalid arrayRef: arrayAddr not found.");
        }

        XcIdent ident = _getIdent(tc, XcSymbolKindEnum.VAR, arrayAddrNode);
        XcVarObj arrayObj = new XcVarObj(ident);

        if (arrayRefObj instanceof XcArrayRefObj) {
            XcArrayRefObj obj = (XcArrayRefObj) arrayRefObj;
            obj.setType(ident.getType());
            obj.setElementType(ident.getType().getRefType());
            obj.setArrayAddr(arrayObj);
        } else if (arrayRefObj instanceof XcXmpSubArrayRefObj) {
            XcXmpSubArrayRefObj obj = (XcXmpSubArrayRefObj) arrayRefObj;
            obj.setType(ident.getType());
            obj.setElementType(ident.getType().getRefType());
            obj.setArrayAddr(arrayObj);
        } else if (arrayRefObj instanceof XcXmpSubArrayRefObj) {
            XcXmpSubArrayRefObj obj = (XcXmpSubArrayRefObj) arrayRefObj;
            obj.setType(ident.getType());
            obj.setElementType(ident.getType().getRefType());
            obj.setArrayAddr(arrayObj);
        } else {
            throw new XmTranslationException(arrayRefNode,
                                             "Unknown ArrayRef object");
        }
        transChildren(tc, arrayRefObj, childNodes);
    }

    static class Pair {
        public String codeName;
        public XcodeNodeVisitor visitor;
        public Pair(String codeName, XcodeNodeVisitor visitor) {
            this.codeName = codeName;
            this.visitor = visitor;
        }
    }

    Pair pairs[] = {
        new Pair("globalDeclarations", new GlobalDeclarationsNodeVisitor()),
        new Pair("declarations", new DeclarationsNodeVisitor()),
        new Pair("exprStatement", new ExprStatementNodeVisitor()),
        new Pair("typeTable", new TypeTableNodeVisitor()),
        new Pair("globalSymbols", new GlobalSymbolsNodeVisitor()),
        new Pair("symbols", new SymbolsNodeVisitor()),
        new Pair("gccAttribute", new GccAttributeVisitor()),
        new Pair("gccAttributes", new GccAttributesVisitor()),
        new Pair("basicType", new BasicTypeVisitor()),
        new Pair("arrayType", new ArrayTypeVisitor()),
        new Pair("arraySize", new ArraySizeVisitor()),
        new Pair("pointerType", new PointerTypeVisitor()),
        new Pair("enumType", new EnumTypeVisitor()),
        new Pair("structType", new StructTypeVisitor()),
        new Pair("unionType", new UnionTypeVisitor()),
        new Pair("functionType", new FunctionTypeVisitor()),
        new Pair("id", new IdVisitor()),
        new Pair("varDecl", new VarDeclVisitor()),
        new Pair("value", new ValueVisitor()),
        new Pair("designatedValue", new DesignatedValueVisitor()),
        new Pair("functionDecl", new FunctionDeclVisitor()),
        new Pair("functionDefinition", new FunctionDefinitionVisitor()),
        new Pair("body", new BodyVisitor()),
        new Pair("compoundStatement", new CompoundStatementNodeVisitor()),
        new Pair("ifStatement", new IfStatementVisitor()),
        new Pair("condition", new ConditionVisitor()),
        new Pair("init", new InitVisitor()),
        new Pair("iter", new IterVisitor()),
        new Pair("then", new ThenVisitor()),
        new Pair("else", new ElseVisitor()),
        new Pair("whileStatement", new WhileStatementVisitor()),
        new Pair("doStatement", new DoStatementVisitor()),
        new Pair("forStatement", new ForStatementVisitor()),
        new Pair("breakStatement", new BreakStatementVisitor()),
        new Pair("continueStatement", new ContinueStatementVisitor()),
        new Pair("returnStatement", new ReturnStatementVisitor()),
        new Pair("gotoStatement", new GotoStatementVisitor()),
        new Pair("name", new NameVisitor()),
        new Pair("switchStatement", new SwitchStatementVisitor()),
        new Pair("caseLabel", new CaseLabelVisitor()),
        new Pair("defaultLabel", new DefaultLabelVisitor()),
        new Pair("statementLabel", new StatementLabelVisitor()),
        new Pair("floatConstant", new FloatConstantVisitor()),
        new Pair("intConstant", new IntConstantVisitor()),
        new Pair("longlongConstant", new LonglongConstantVisitor()),
        new Pair("stringConstant", new StringConstaantVisitor()),
        new Pair("string", new StringConstaantVisitor()),
        new Pair("moeConstant", new MoeConstantVisitor()),
        new Pair("unaryMinusExpr", new UnaryMinusVisitor()),
        new Pair("postDecrExpr", new PostDecrVisitor()),
        new Pair("postIncrExpr", new PostIncrVisitor()),
        new Pair("preDecrExpr", new PreDecrVisitor()),
        new Pair("preIncrExpr", new PreIncrVisitor()),
        new Pair("asgBitAndExpr", new AsgBitAndExprVisitor()),
        new Pair("asgBitOrExpr", new AsgBitOrExprVisitor()),
        new Pair("asgBitXorExpr", new AsgBitXorExprVisitor()),
        new Pair("asgDivExpr", new AsgDivExprVisitor()),
        new Pair("asgLshiftExpr", new AsgLshiftExprVisitor()),
        new Pair("asgMinusExpr", new AsgMinusExprVisitor()),
        new Pair("asgModExpr", new AsgModExprVisitor()),
        new Pair("asgMulExpr", new AsgMulExprVisitor()),
        new Pair("asgPlusExpr", new AsgPlusExprVisitor()),
        new Pair("asgRshiftExpr", new AsgRshiftExprVisitor()),
        new Pair("assignExpr", new AssignExprVisitor()),
        new Pair("bitAndExpr", new BitAndExprVisitor()),
        new Pair("bitOrExpr", new BitOrExprVisitor()),
        new Pair("bitNotExpr", new BitNotExprVisitor()),
        new Pair("bitXorExpr", new BitXorExprVisitor()),
        new Pair("divExpr", new DivExprVisitor()),
        new Pair("logAndExpr", new LogAndExprVisitor()),
        new Pair("logEQExpr", new LogEQExprVisitor()),
        new Pair("logGEExpr", new LogGEExprVisitor()),
        new Pair("logGTExpr", new LogGTExprVisitor()),
        new Pair("logLEExpr", new LogLEExprVisitor()),
        new Pair("logLTExpr", new LogLTExprVisitor()),
        new Pair("logNEQExpr", new LogNEQExprVisitor()),
        new Pair("logNotExpr", new LogNotExprVisitor()),
        new Pair("logOrExpr", new LogOrExprVisitor()),
        new Pair("LshiftExpr", new LshiftExprVisitor()),
        new Pair("modExpr", new ModExprVisitor()),
        new Pair("mulExpr", new MulExprVisitor()),
        new Pair("minusExpr", new MinusExprVisitor()),
        new Pair("plusExpr", new PlusExprVisitor()),
        new Pair("RshiftExpr", new RshiftExprVisitor()),
        new Pair("condExpr", new CondExprVisitor()),
        new Pair("commaExpr", new CommaExprVisitor()),
        new Pair("castExpr", new CastExprVisitor()),
        new Pair("Var", new VarVisitor()),
        new Pair("arrayRef", new ArrayRefVisitor()),
        new Pair("varAddr", new VarAddrVisitor()),
        new Pair("funcAddr", new FuncAddrVisitor()),
        new Pair("arrayAddr", new ArrayAddrVisitor()),
        new Pair("memberAddr", new MemberAddrVisitor()),
        new Pair("memberRef", new MemberRefVisitor()),
        new Pair("memberArrayAddr", new MemberArrayAddrVisitor()),
        new Pair("memberArrayRef", new MemberArrayRefVisitor()),
        new Pair("pointerRef", new PointerRefVisitor()),
        new Pair("functionCall", new FunctionCallVisitor()),
        new Pair("function", new FunctionVisitor()),
        new Pair("arguments", new ArgumentsVisitor()),
        new Pair("sizeOfExpr", new SizeOfExprVisitor()),
        new Pair("addrOfExpr", new AddrOfExprVisitor()),
        new Pair("gccAlignOfExpr", new GccAlignOfExprVisitor()),
        new Pair("xmpDescOf", new XmpDescOfVisitor()),
        new Pair("gccLabelAddr", new GccLabelAddrVisitor()),
        new Pair("gccAsmDefinition", new GccAsmDefinitionVisitor()),
        new Pair("pragma", new PragmaVisitor()),
        new Pair("OMPPragma", new OMPPragmaVisitor()),
        new Pair("ACCPragma", new ACCPragmaVisitor()),
        new Pair("text", new TextVisitor()),
        new Pair("gccAsmStatement", new GccAsmStatementVisitor()),
        new Pair("gccAsmOperand", new GccAsmOperandVisitor()),
        new Pair("gccAsmOperands", new GccAsmOperandsVisitor()),
        new Pair("gccAsmClobbers", new GccAsmClobbersVisitor()),
        new Pair("builtin_op", new BuiltinOpoVisitor()),
        new Pair("typeName", new TypeNameVisitor()),
        new Pair("gccMemberDesignator", new GccMemberDesignatorVisitor()),
        new Pair("gccRangedCaseLabel", new GccRangedCaseLabelVisitor()),
        new Pair("gccCompoundExpr", new GccCompoundExprVisitor()),
        new Pair("compoundValue", new CompoundValueExprVisitor()),
        new Pair("compoundValueAddr", new CompoundValueAddrExprVisitor()),
        new Pair("coArrayRef", new CoArrayRefVisitor()),
        new Pair("coArrayType", new CoArrayTypeVisitor()),
        new Pair("subArrayRef", new SubArrayRefVisitor()),
        new Pair("indexRange", new IndexRangeVisitor()),
        new Pair("lowerBound", new LowerBoundVisitor()),
        new Pair("upperBound", new UpperBoundVisitor()),
        new Pair("step", new StepBoundVisitor()),
    };

    void createVisitorMap(Pair[] pairs) {
        visitorMap = new HashMap<String, XcodeNodeVisitor>();
        for (Pair p : pairs) {
            visitorMap.put(p.codeName, p.visitor);
        }
    }

    private Map<String, XcodeNodeVisitor> visitorMap;
}
