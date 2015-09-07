/*
 * $tsukuba_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
// package exc.xcodeml;

package exc.xcodeml;


import static xcodeml.util.XmDomUtil.collectChildNodes;
import static xcodeml.util.XmDomUtil.collectElementsExclude;
import static xcodeml.util.XmDomUtil.getAttr;
import static xcodeml.util.XmDomUtil.getAttrBool;
import static xcodeml.util.XmDomUtil.getContent;
import static xcodeml.util.XmDomUtil.getContentText;
import static xcodeml.util.XmDomUtil.getElement;

import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import xcodeml.c.decompile.XcConstObj;
import xcodeml.c.util.XmcBindingUtil;
import xcodeml.util.XmStringUtil;
import exc.object.ArrayType;
import exc.object.BasicType;
import exc.object.EnumType;
import exc.object.FunctionType;
import exc.object.Ident;
import exc.object.PointerType;
import exc.object.StorageClass;
import exc.object.StructType;
import exc.object.UnionType;
import exc.object.VarScope;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XmpCoArrayType;
import exc.object.XobjArgs;
import exc.object.XobjList;
import exc.object.XobjString;
import exc.object.Xobject;
import exc.object.XobjectFile;
import exc.object.Xtype;

import static xcodeml.util.XmDomUtil.getElement;
import static xcodeml.util.XmDomUtil.getAttr;
import static xcodeml.util.XmDomUtil.getContent;
import static xcodeml.util.XmDomUtil.getContentText;
import static xcodeml.util.XmDomUtil.getAttrBool;
import static xcodeml.util.XmDomUtil.collectChildNodes;
import static xcodeml.util.XmDomUtil.collectElementsExclude;

/**
 * tools for XcodeML/C to Xcode translation.
 */
public class XcodeMLtools_C extends XcodeMLtools {
  private XcodeMLNameTable_C nameTable = new XcodeMLNameTable_C();

  enum PragmaScope {
    Compexpr, Nestedfunc,
      }
  private static Stack<PragmaScope> _pScopeStack = new Stack<PragmaScope>();

  public XcodeMLtools_C() {
  }

  @Override
  void enterType(Node n) {
    String name = n.getNodeName();

    if (name.equals("basicType")) {
      declBasicType(n);
    } else if (name.equals("pointerType")) {
      declPointerType(n);
    } else if (name.equals("functionType")) {
      declFunctionType(n);
    } else if (name.equals("arrayType")) {
      declArrayType(n);
    } else if (name.equals("structType")) {
      declStructType(n);
    } else if (name.equals("unionType")) {
      declUnionType(n);
    } else if (name.equals("enumType")) {
      declEnumType(n);
    } else if (name.equals("coArrayType")) {
      declCoArrayType(n);
    } else {
      fatal("Unknown node in typeTable: " + n);
    }
  }

  /*
   * global Ident section
   */
  @Override
  void enterGlobalIdent(Node n) {
    if (n.getNodeName() == "id") {
      Ident id = toIdent(n);
      xobjFile.getGlobalIdentList().add(id);
    } else
      fatal("Unknown node in globalSybmols: " + n);
  }

  @Override
  void enterGlobalDecl(Node n) {
    Xobject xobj = toXobject(n);
    //xobj.setParentRecursively(null);
    switch (xobj.Opcode()) {
    case FUNCTION_DEFINITION:
    case VAR_DECL:
    case FUNCTION_DECL:
    case GCC_ASM_DEFINITION:
    case PRAGMA_LINE:
    case TEXT:
    case ACC_PRAGMA:
    case XMP_PRAGMA:
      xobjFile.add(xobj);
      break;
    default:
      fatal("Unknown node in globalDeclarations: " + n);
    }
  }

  @Override
  Xobject toXobject(Node n) {
    if (n == null)
      return null;

    Xcode code = nameTable.getXcode(n.getNodeName());
    if (code == null) {
      fatal("unknown Xcode=" + n);
    }

    Xtype type = null;
    Xobject x;

    String typeId = getAttr(n, "type");
    if (typeId != null)
      type = getType(typeId);

    switch (code) {
    case FUNCTION_DEFINITION: {

      _pScopeStack.push(PragmaScope.Nestedfunc);

      XobjList xlist = enterAsXobjList(n,
				       code,
				       type,
				       Xcode.IDENT,
				       null, // symbolTypeName
				       getContentText(getElement(n, "name")), // symbolName
				       null, // symbolScope
				       getElement(n, "symbols"),
				       null, // params, process later.
				       getElement(n, "body"),
				       getElement(n, "gccAttributes"));
      _pScopeStack.pop();

      // enter params node.
      XobjList paramList = enterFuncDefParams(getElement(n, "params"));
      // (name symbols params body gccAttributes)
      xlist.setArg(2, paramList);

      // change body to COMPOUND_STATTEMENT.
      // body of FUNCTION_DEFINITION must be COMPOUND_STATEMENT.
      XobjArgs arg = xlist.getArgs();
      XobjArgs argBody = arg.nextArgs().nextArgs().nextArgs();
      Xobject stmts = argBody.getArg();
      if (stmts != null) {
        switch (stmts.Opcode()) {
        case LIST:
          stmts = Xcons.CompoundStatement(
					  Xcons.IDList(), Xcons.List(), stmts);
          argBody.setArg(stmts);
          break;
        case COMPOUND_STATEMENT:
          break;
        default:
          fatal("Invalid function body");
        }
      }
      return xlist;
    }

    case VAR_DECL:
      return enterAsXobjList(n,
			     code,
			     type,
			     Xcode.IDENT,
			     null, // symbolTypeName
			     getContentText(getElement(n, "name")), // symbolName
			     null, // symbolScope
			     getContent(getElement(n, "value")),
			     getElement(n, "gccAsm"));

    case FUNCTION_DECL:
      return enterAsXobjList(n,
			     code,
			     type,
			     Xcode.IDENT,
			     null, // symbolTypeName
			     getContentText(getElement(n, "name")), // symbolName
			     null, // symbolScope
			     getElement(n, "gccAsm"));

    case ID_LIST: // symbols
      return toIdentList(n);

    case IDENT: // name
      return Xcons.Symbol(code, type, getContentText(n));

    case STRING: 
      return Xcons.String(getContentText(n));

    case OMP_PRAGMA: {
      XobjList xlist = Xcons.List(Xcode.OMP_PRAGMA);
      setCommonAttributes(n, xlist);
      return getChildList(n, xlist);
    }

    case TEXT:
    case PRAGMA_LINE: {
      String contentText = getContentText(n);
      x = Xcons.List(code,
		     new XobjString(Xcode.STRING, contentText));
      setCommonAttributes(n, x);
      return x;
    }

    case STRING_CONSTANT:
      return Xcons.StringConstant(getContentText(n));

    case INT_CONSTANT:
      return Xcons.Int(code,
		       type,
		       getAsCInt(getContentText(n)));

    case FLOAT_CONSTANT:
      return Xcons.Float(code,
			 type,
			 getContentText(n));

    case LONGLONG_CONSTANT: {
      XcConstObj.LongLongConst llConst =
	XmcBindingUtil.createLongLongConst(getContentText(n),
					   typeId);
      if (llConst == null) {
	fatal("Invalid long long value");
      }
      return Xcons.Long(code,
			type,
			llConst.getHigh(),
			llConst.getLow());
    }

    case MOE_CONSTANT:
      return new XobjString(code,
			    type,
			    getContentText(n));


    case COMPOUND_STATEMENT:
      return enterAsXobjList(n,
			     code,
			     type,
			     getElement(n, "symbols"),
			     getElement(n, "declarations"),
			     getElement(n, "body"));

    case WHILE_STATEMENT:
      return enterAsXobjList(n,
			     code,
			     type,
			     getContent(getElement(n, "condition")),
			     getElement(n, "body"));

    case DO_STATEMENT:
      return enterAsXobjList(n,
			     code,
			     type,
			     getElement(n, "body"),
			     getContent(getElement(n, "condition")));

    case FOR_STATEMENT:
      return enterAsXobjList(n,
			     code,
			     type,
			     getContent(getElement(n, "init")),
			     getContent(getElement(n, "condition")),
			     getContent(getElement(n, "iter")),
			     getElement(n, "body"));

    case IF_STATEMENT:
      return enterAsXobjList(n,
			     code,
			     type,
			     getContent(getElement(n, "condition")),
			     getContent(getElement(n, "then")),
			     getContent(getElement(n, "else")));

    case SWITCH_STATEMENT:
      return enterAsXobjList(n,
			     code,
			     type,
			     getContent(getElement(n, "value")),
			     getElement(n, "body"));

    case STATEMENT_LABEL:
      return enterAsXobjList(n,
			     code,
			     type,
			     getElement(n, "name"));

    case CASE_LABEL:
      return enterAsXobjList(n,
			     code,
			     type,
			     getContent(getElement(n, "value")));

    case CONDITIONAL_EXPR: {
      NodeList exprsNodeList = n.getChildNodes();
      List<Node> exprNodes = new ArrayList<Node>();
      for (int i = 0; i < exprsNodeList.getLength(); ++i) {
	Node thisNode = exprsNodeList.item(i);
	if (thisNode.getNodeType() != Node.ELEMENT_NODE) {
	  continue;
	}
	exprNodes.add(thisNode);
      }

      if (exprNodes.size() != 3) {
	fatal("Invalid condExpr");
      }
      Xobject xobj =
	Xcons.List(Xcode.CONDITIONAL_EXPR,
		   type,
		   toXobject(exprNodes.get(0)),
		   Xcons.List(Xcode.LIST,
			      toXobject(exprNodes.get(1)),
			      toXobject(exprNodes.get(2))));
      setCommonAttributes(n, xobj);
      return xobj;
    }

    case FUNCTION_CALL:
      return enterAsXobjList(n,
			     code,
			     type,
			     getContent(getElement(n, "function")),
			     getElement(n, "arguments"));

    case SIZE_OF_EXPR:
      return enterSizeOrAlignExpr(code, type, getContent(n));

    case VAR:
    case VAR_ADDR:
    case ARRAY_ADDR:
      return Xcons.Symbol(code,
			  type,
			  getContentText(n),
			  VarScope.get(getAttr(n, "scope")));

    case ARRAY_REF:
      return enterArrayRef(code, type, n);

    case FUNC_ADDR:
      return Xcons.Symbol(code,
			  type,
			  getContentText(n));

    case MEMBER_REF:
    case MEMBER_ARRAY_REF:
    case MEMBER_ADDR:
    case MEMBER_ARRAY_ADDR:
      return enterMember(n, code, type);

    case BUILTIN_OP: {
      // (CODE name is_id is_addr content...)
      XobjList objList = enterAsXobjList(n,
					 code,
					 type);
      objList.add(new XobjString(Xcode.IDENT, getAttr(n, "name")));
      objList.add(getAttrIntFlag(n, "is_id"));
      objList.add(getAttrIntFlag(n, "is_addrOf"));

      XobjList args = enterChildren(n);
      if (args != null) {
	objList.add(args);
      }
      return objList;
    }

    case GCC_ATTRIBUTE: {
      XobjList objList = enterAsXobjList(n,
					 code,
					 type);
      Xobject symbolObj = Xcons.Symbol(Xcode.IDENT, getAttr(n, "name"));
      objList.add(symbolObj);
      objList.add(enterChildren(n));
      return objList;
    }
    
    case GCC_ASM_OPERAND: {
        // (CODE var match constraint)
        XobjList objList = enterAsXobjList(n, code, type);
        getChildList(n, objList);
        String match = getAttr(n, "match");
        objList.add((match != null)? Xcons.StringConstant(null, match) : null);
        objList.add(Xcons.StringConstant(null,getAttr(n, "constraint")));
        return objList;
    }
    
    case GCC_ASM_STATEMENT: {
        // (CODE is_volatile string_constant operand1 operand2 clobbers)
        XobjList objList = enterAsXobjList(n, code, type);
        objList.add(getAttrIntFlag(n, "is_volatile"));
        getChildList(n, objList);
        return objList;
    }

    case GCC_LABEL_ADDR:
      return Xcons.Symbol(code, type, getContentText(n));

    case GCC_COMPOUND_EXPR: {
      _pScopeStack.push(PragmaScope.Compexpr);
      XobjList objList = enterAsXobjList(n,
					 code,
					 type,
					 getElement(n, "compoundStatement"));
      _pScopeStack.pop();
      return objList;
    }

    case XMP_DESC_OF:
      return enterXmpDescOf(code, type, getContent(n));

    case SUB_ARRAY_REF:
      //return enterArrayRef(code, type, n);
      return enterSubArrayRef(code, type, n);

    case INDEX_RANGE:
      return enterIndexRange(code, type, n);

    case CO_ARRAY_REF: {
      ArrayList<Node> childNodes = collectChildNodes(n);
      Node contentNode = childNodes.remove(0);
      String contentNodeName = contentNode.getNodeName();
      if (("Var".equals(contentNodeName) ||
	   "arrayRef".equals(contentNodeName) ||
	   "subArrayRef".equals(contentNodeName) ||
	   "memberRef".equals(contentNodeName)) == false) {
	fatal("Invalid coArrayRef");
      }
      XobjList objList = enterAsXobjList(n,
					 code,
					 type,
					 contentNode);
      Xobject exprs = Xcons.List();
      for (Node childNode : childNodes) {
	exprs.add(toXobject(childNode));
      }
      objList.add(exprs);
      objList.setScope(VarScope.get(getAttr(n, "scope")));
      return objList;
    }

    case COMPOUND_VALUE:
    case COMPOUND_VALUE_ADDR:
      return enterAsXobjList(n, code, type, getContent(getElement(n, "value")));

    default: // default action, make list
      XobjList xobjs = new XobjList(code, type);
      //if(code == Xcode.LIST && list.getLength() == 0)
      //  return null;
      setCommonAttributes(n, xobjs);
      return getChildList(n, xobjs);
    }
  }

  Ident toIdent(Node n) {
    String name = getContentText(getElement(n, "name"));

    // get type
    String tid = getAttr(n, "type");
    Xtype type = null;

    if (tid == null)
      tid = getAttr(getElement(n, "name"), "type");

    if (tid != null) {
      type = getType(tid);
    }

    // get storage class
    StorageClass sclass = StorageClass.SNULL;
    String sclassStr = getAttr(n, "sclass");

    if (sclassStr != null) {
      sclass = StorageClass.get(sclassStr);
    }

    // get bit field
    String bitFieldStr = getAttr(n, "bit_field");
    int bitField = 0;
    Xobject bitFieldExpr = null;
    if (bitFieldStr != null) {
      if ("*".equals(bitFieldStr)) {
	bitFieldExpr = toXobject(getElement(n, "bitField"));
      } else {
	bitField = getAsCInt(bitFieldStr);
      }
    }

    // get enum member value
    Xobject enumValue = toXobject(getContent(getElement(n, "value")));

    // get codimensions #284
    Xobject codims = toXobject(getElement(n, "codimensions"));

    // get gcc attributes
    Xobject gccAttrs = getGccAttributes(n);

    // get optional flags
    int optionalFlags =
      (getAttrBool(n, "is_gccExtension") ? Xobject.OPT_GCC_EXTENSION : 0) |
      (getAttrBool(n, "is_gccThread") ? Xobject.OPT_GCC_THREAD : 0);

    // addr of symbol
    Xobject addr = null;
    if (sclass != null && sclass.canBeAddressed()) {
      switch (type.getKind()) {
      case Xtype.BASIC:
      case Xtype.POINTER:
      case Xtype.FUNCTION:
      case Xtype.ENUM:
      case Xtype.STRUCT:
      case Xtype.UNION: {
	Xtype ptype = Xtype.Pointer(type, xobjFile);
	addr = Xcons.Symbol(Xcode.VAR_ADDR, ptype,
			    name, VarScope.LOCAL);
      }
	break;
      case Xtype.ARRAY: {
	Xtype ptype = Xtype.Pointer(type, xobjFile);
	addr = Xcons.Symbol(Xcode.ARRAY_ADDR, ptype,
			    name, VarScope.LOCAL);
      }
	break;
      }
    }

    // create ident
    // for coarray, set codiemnsions (#284)
    Ident ident = new Ident(name, sclass, type, addr,
			    optionalFlags, gccAttrs,
			    bitField, bitFieldExpr, enumValue, null, codims);
    //if (codims != null)
    //  ident.setCodimensions(codims);

    // declaring
    if (sclass != null && sclass.isVarOrFunc()) {
      ident.setIsDeclared(true);
    }

    return ident;
  }

//   public XobjectFile read(Reader reader) {
//     XobjectFile objFile = super.read(reader);
//     return objFile;
//   }

  private XobjList enterAsXobjList(Node baseNode, Xcode code, Xtype type, Node ... nodes) {
    return toXobjList(baseNode, Xcons.List(code, type), nodes);
  }

  private XobjList enterAsXobjList(Node baseNode,
				   Xcode code,
				   Xtype type,
				   Xcode symbolCode,
				   String symbolTypeName,
				   String symbolName,
				   String symbolScope,
				   Node ... nodes) {
    Xtype symbolType = null;
    if (symbolTypeName != null) {
      symbolType = getType(symbolTypeName);
    }
    VarScope scope = null;
    if (symbolScope != null) {
      scope = VarScope.get(symbolScope);
    }
    Xobject symbolObj = Xcons.Symbol(symbolCode,
				     symbolType,
				     symbolName,
				     scope);
    XobjList objList = Xcons.List(code, type);
    objList.add(symbolObj);
    return toXobjList(baseNode, objList, nodes);
  }

  private XobjList toXobjList(Node baseNode, XobjList objList, Node ... nodes) {
    for (Node node : nodes) {
      Xobject obj = toXobject(node);
      objList.add(obj); // add even if obj is null
    }
    setCommonAttributes(baseNode, objList);
    return objList;
  }

  private XobjList enterChildren(Node parentNode) {
    XobjList objList = Xcons.List();
    NodeList childNodeList = parentNode.getChildNodes();
    for (int i = 0; i < childNodeList.getLength(); ++i) {
      Node childNode = childNodeList.item(i);
      if (childNode.getNodeType() != Node.ELEMENT_NODE) {
	continue;
      }
      objList.add(toXobject(childNode));
    }
    return objList;
  }

  private XobjList enterFuncDefParams(Node paramsNode) {
    XobjList xobj = Xcons.List();
    NodeList list = paramsNode.getChildNodes();
    boolean hasEllipsis = false;
    for (int i = 0; i < list.getLength(); i++) {
      Node nameNode = list.item(i);
      if (nameNode.getNodeType() != Node.ELEMENT_NODE) {
	continue;
      }
      String nodeName = nameNode.getNodeName();
      if ("name".equals(nodeName)) {
	String name = getContentText(nameNode);
	if (name != null &&
	    !name.equals("")) {
	  xobj.add(Xcons.List(Xcode.VAR_DECL,
			      Xcons.Symbol(Xcode.IDENT, name),
			      null));
	}
      } else if ("ellipsis".equals(nodeName)) {
	hasEllipsis = true;
      }
    }

    if (hasEllipsis) {
      /*
       * FIXME
       * This statement is not enabled in
       * XmcXmObjToXobjectTranslator.java:enter(XbcParams xmobj).
       * Next code in the method would be always evaluated to false:
       *
       *     if(toBool(xmobj, xmobj.getEllipsis())) {
       */
      //xobj.add(null);
    }

    return xobj;
  }

  private XobjList enterMember(Node n, Xcode code, Xtype type) {
    XobjList objList = enterAsXobjList(n,
				       code,
				       type,
				       getContent(n));
    Xobject symbolObj = Xcons.Symbol(Xcode.IDENT, getAttr(n, "member"));
    objList.add(symbolObj);
    return objList;
  }

  private XobjList enterSizeOrAlignExpr(Xcode code, Xtype type, Node argNode) {
    String argTypeName = getAttr(argNode, "type");
    Xtype argType = getType(argTypeName);

    if (argType instanceof ArrayType &&
	((ArrayType) argType).getArraySizeExpr() != null) {
      argType = Xtype.voidPtrType;
    }

    XobjList child = Xcons.List(Xcode.TYPE_NAME, argType);
    XobjList objList = Xcons.List(code, type);
    objList.add(child);
    return objList;
  }

  /** process arrayRef. */
  private XobjList enterArrayRef(Xcode code, Xtype type, Node arrayRefNode) {
    ArrayList<Node> childNodes = collectElementsExclude(arrayRefNode,
							"arrayAddr");
    Node arrayAddrNode = getElement(arrayRefNode, "arrayAddr");
    XobjList objList = enterAsXobjList(arrayRefNode,
				       code,
				       type,
				       arrayAddrNode);
    objList.add(enterAsXobjList(arrayRefNode,
				Xcode.LIST,
				getType(getAttr(arrayRefNode, "type")),
				childNodes.toArray(new Node[0])));
    return objList;
  }

  /** process xmpDescOf. */
  private XobjList enterXmpDescOf(Xcode code, Xtype type, Node argNode){
    XobjList objList = Xcons.List(code, type);
    objList.add(toXobject(argNode));
    return objList;
  }

  /** process subArrayRef. */
  private XobjList enterSubArrayRef(Xcode code, Xtype type, Node subArrayRefNode) {

    Node arrayAddrNode = getElement(subArrayRefNode, "arrayAddr");
    ArrayList<Node> childNodes;
    if (arrayAddrNode != null){
      childNodes = collectElementsExclude(subArrayRefNode, "arrayAddr");
    }
    else { // pointer
      arrayAddrNode = getElement(subArrayRefNode, "Var");
      childNodes = collectElementsExclude(subArrayRefNode, "Var");
    }

    XobjList objList = enterAsXobjList(subArrayRefNode,
				       code,
				       type,
				       arrayAddrNode);

    XobjList subList = Xcons.List();
    for (Node childNode : childNodes){
      Xobject indexRange = toXobject(childNode);
      if (indexRange instanceof XobjList){
	subList.add(indexRange.getArg(0));
      }
      else {
	subList.add(indexRange);
      }
    }
    objList.add(subList);

    // objList.add(enterAsXobjList(subArrayRefNode,
    // 				Xcode.LIST,
    // 				getType(getAttr(subArrayRefNode, "type")),
    // 				childNodes.toArray(new Node[0])));

    return objList;
  }

  /** process indexRange. */
  private XobjList enterIndexRange(Xcode code, Xtype type, Node indexRangeNode) {
    ArrayList<Node> childNodes = collectChildNodes(indexRangeNode);
    XobjList objList = Xcons.List(code, type);
    // objList.add(enterAsXobjList(indexRangeNode,
    // 				Xcode.LIST,
    // 				type,
    // 				childNodes.toArray(new Node[0])));

    if (childNodes.size() == 1){
      objList.add(toXobject(childNodes.get(0)));
    }
    else {
      XobjList subList = Xcons.List();
      for (Node childNode : childNodes){
	Xobject arg = toXobject(childNode);
	if (arg != null && arg.Nargs() != 0) subList.add(arg.getArg(0));
	else subList.add(null);
      }
      objList.add(subList);
    }

    return objList;
  }

  private int getTypeQualFlags(Node n, boolean isFunctionType) {
    int tqConst = getAttrBool(n, "is_const") ? Xtype.TQ_CONST : 0;
    int tqRestrict = getAttrBool(n, "is_restrict") ? Xtype.TQ_RESTRICT : 0;
    int tqVolatile = getAttrBool(n, "is_volatile") ? Xtype.TQ_VOLATILE : 0;
    int tqInline = (isFunctionType && getAttrBool(n, "is_inline")) ? Xtype.TQ_INLINE : 0;
    int tqFuncStatic = (isFunctionType && getAttrBool(n, "is_static")) ? Xtype.TQ_FUNC_STATIC : 0;

    return tqConst | tqRestrict | tqVolatile | tqInline | tqFuncStatic;
  }

  private Xobject getGccAttributes(Node n) {
    return toXobject(getElement(n, "gccAttributes"));
  }

  private void declBasicType(Node n) {
    String typeId = getAttr(n, "type");
    String name = getAttr(n, "name");
    int tq = getTypeQualFlags(n, false);
    Xobject gccAttrs = getGccAttributes(n);
    BasicType.TypeInfo ti = BasicType.getTypeInfoByCName(name);

    Xtype type;
    if (ti == null) {
      // inherited type
      Xtype ref = getType(name);
      type = ref.inherit(typeId);
      type.setTypeQualFlags(tq);
      type.setGccAttributes(gccAttrs);
    } else {
      type = new BasicType(ti.type.getBasicType(),
			   typeId,
			   tq,
			   gccAttrs,
			   null,
			   null);
    }
    xobjFile.addType(type);
  }

  private void declPointerType(Node n) {
    Xtype refType = getType(getAttr(n, "ref"));
    Xobject gccAttrs = getGccAttributes(n);
    Xtype type = new PointerType(getAttr(n, "type"),
				 refType,
				 getTypeQualFlags(n, false),
				 gccAttrs);
    xobjFile.addType(type);
  }

  private void declFunctionType(Node n) {
    Xtype returnType = getType(getAttr(n, "return_type"));
    if (returnType == null) {
      fatal("Invalid functionType");
    }

    Xobject gccAttrs = getGccAttributes(n);

    Xobject params = null;
    Node paramsNode = getElement(n, "params");
    if (paramsNode != null) {
      NodeList paramsNodeChildList = paramsNode.getChildNodes();
      List<Node> nameNodes = new ArrayList<Node>();
      boolean hasEllipsis = false;
      for (int i = 0; i < paramsNodeChildList.getLength(); ++i) {
	Node thisNode = paramsNodeChildList.item(i);
	if (thisNode.getNodeType() != Node.ELEMENT_NODE) {
	  continue;
	}
	String nodeName = thisNode.getNodeName();
	if ("name".equals(nodeName)) {
	  nameNodes.add(thisNode);
	} else if ("ellipsis".equals(nodeName)) {
	  hasEllipsis = true;
	}
      }

      int numNameNodes = nameNodes.size();
      if (numNameNodes > 0) {
	params = Xcons.List();
	if (numNameNodes >= 2 ||
	    !"void".equals(getAttr(nameNodes.get(0), "type"))) {
	  for (Node nameNode : nameNodes) {
	    Xtype type = getType(getAttr(nameNode, "type"));
	    params.add(Xcons.Symbol(Xcode.IDENT,
				    type,
				    getContentText(nameNode)));
	  }
	}
      }

      if (hasEllipsis) {
	if (params == null) {
	  params = Xcons.List();
	}
	params.add(null);
      }
    }

    Xtype type = new FunctionType(getAttr(n, "type"),
				  returnType,
				  params,
				  getTypeQualFlags(n, true),
				  false,
				  gccAttrs,
				  null);
    xobjFile.addType(type);
  }

  private void declArrayType(Node n) {
    String arraySize1 = getAttr(n, "array_size");
    long size = -1;
    Xobject sizeExpr = null;

    if ("*".equals(arraySize1)) {
      sizeExpr = toXobject(getContent(getElement(n, "arraySize")));
    } else if (arraySize1 != null) {
      size = getAsCLong(arraySize1);
    }

    Xtype refType = getType(getAttr(n, "element_type"));
    Xobject gccAttrs = getGccAttributes(n);
    Xtype type = new ArrayType(getAttr(n, "type"),
			       refType,
			       getTypeQualFlags(n, false),
			       size,
			       sizeExpr,
			       gccAttrs);
    xobjFile.addType(type);
  }

  private void declStructType(Node n) {
    XobjList identList = (XobjList)toXobject(getElement(n, "symbols"));
    Xobject gccAttrs = getGccAttributes(n);
    Xtype type = new StructType(getAttr(n, "type"),
				identList,
				getTypeQualFlags(n, false),
				gccAttrs);
    xobjFile.addType(type);
  }

  private void declUnionType(Node n) {
    XobjList identList = (XobjList)toXobject(getElement(n, "symbols"));
    Xobject gccAttrs = getGccAttributes(n);
    Xtype type = new UnionType(getAttr(n, "type"),
			       identList,
			       getTypeQualFlags(n, false),
			       gccAttrs);
    xobjFile.addType(type);
  }

  private void declEnumType(Node n) {
    XobjList identList = (XobjList)toXobject(getElement(n, "symbols"));
    Xobject gccAttrs = getGccAttributes(n);
    Xtype type = new EnumType(getAttr(n, "type"),
			      identList,
			      getTypeQualFlags(n, false),
			      gccAttrs);
    xobjFile.addType(type);
  }

  private void declCoArrayType(Node n) {
    String arraySize1 = getAttr(n, "array_size");
    long size = -1;
    Xobject sizeExpr = null;

    if ("*".equals(arraySize1)) {
      sizeExpr = toXobject(getContent(getElement(n, "arraySize")));
    } else if (arraySize1 != null) {
      size = getAsCLong(arraySize1);
    }

    Xtype refType = getType(getAttr(n, "element_type"));
    Xtype type = new XmpCoArrayType(getAttr(n, "type"),
				    refType,
				    0,
				    (int) size,
				    sizeExpr);
    xobjFile.addType(type);
  }

  @Override
  Xobject setCommonAttributes(Node n, Xobject object) {
    Xobject obj = super.setCommonAttributes(n, object);
        
    if (getAttrBool(n, "is_gccSyntax")) {
      obj.setIsGccSyntax(true);
    }
    if (getAttrBool(n, "is_modified")) {
      obj.setIsSyntaxModified(true);
    }
    if (getAttrBool(n, "is_gccExtension")) {
      obj.setIsGccExtension(true);
    }
    return obj;
  }

  /*

   */
  private int getAsCInt(String str) {
    return XmStringUtil.getAsCInt(null, str);
  }
  private long getAsCLong(String str) {
    return XmStringUtil.getAsCLong(null, str);
  }
}
