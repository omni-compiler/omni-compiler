/*
 * $tsukuba_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
// package exc.xcodeml;

package exc.xcodeml;

import static xcodeml.util.XmDomUtil.getAttr;
import static xcodeml.util.XmDomUtil.getAttrBool;
import static xcodeml.util.XmDomUtil.getContent;
import static xcodeml.util.XmDomUtil.getContentText;
import static xcodeml.util.XmDomUtil.getElement;

import java.io.Reader;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import exc.object.BasicType;
import exc.object.FarrayType;
import exc.object.FunctionType;
import exc.object.Ident;
import exc.object.StorageClass;
import exc.object.StructType;
import exc.object.VarScope;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XobjList;
import exc.object.XobjString;
import exc.object.Xobject;
import exc.object.XobjectFile;
import exc.object.Xtype;


/**
 * tools for XcodeML/Fortran to Xcode translation.
 */
public class XcodeMLtools_F extends XcodeMLtools {
  private XcodeMLNameTable_F nameTable = new XcodeMLNameTable_F();

  // constructor
  public XcodeMLtools_F() {
  }

  void enterType(Node n) {
    String name = n.getNodeName();
    if (name == "FbasicType") {
      declFbasicType(n);
    } else if (name == "FfunctionType") {
      declFfunctionType(n);
    } else if (name == "FstructType") {
      declFstructType(n);
    } else
      fatal("Unknown node in typeTable: " + n);
  }

  /*
   * FbasicType: (kind?, (len | (arrayIndex | indexRange)+)?)
   *
   * ex1:"integer(kind=8)" <FbasicType type="TYPE_NAME" ref="Fint">
   * <kind>8</kind> </FbasicType>
   *
   * ex2:　"integer dimension(10, 1:10)" <FbasicType type="TYPE_NAME"
   * ref="Fint"> <arrayIndex> <FintConstant type="Fint">10</FintConstant>
   * </arrayIndex> <indexRange> <lowerBound> <FintConstant
   * type="Fint">1</FintConstant> </lowerBound> <upperBound> <FintConstant
   * type="Fint">10</FintConstant> </upperBound> </indexRange> </FbasicType>
   *
   * ex3:　"character(len=10)" <FbasicType type="TYPE_NAME" ref="Fcharacter">
   * <len> <FintConstant type="Fint">10</FintConstant> </len> </FbasicType>
   */

  private void declFbasicType(Node n) {
    String tid = getAttr(n, "type");
    BasicType.TypeInfo ti = BasicType.getTypeInfoByFName(
                             (getAttrBool(n, "is_class") ||
                              getAttrBool(n, "is_procedure")) &&
                             (getAttr(n, "ref") == null) ?
                               "Fvoid" : getAttr(n, "ref"));
    int tq = (getAttrBool(n, "is_allocatable") ? Xtype.TQ_FALLOCATABLE : 0)
      | (getAttrBool(n, "is_optional") ? Xtype.TQ_FOPTIONAL : 0)
      | (getAttrBool(n, "is_parameter") ? Xtype.TQ_FPARAMETER : 0)
      | (getAttrBool(n, "is_pointer") ? Xtype.TQ_FPOINTER : 0)
      | (getAttrBool(n, "is_private") ? Xtype.TQ_FPRIVATE : 0)
      | (getAttrBool(n, "is_public") ? Xtype.TQ_FPUBLIC : 0)
      | (getAttrBool(n, "is_save") ? Xtype.TQ_FSAVE : 0)
      | (getAttrBool(n, "is_target") ? Xtype.TQ_FTARGET : 0)
      | (getAttrBool(n, "is_cray_pointer") ? Xtype.TQ_FCRAY_POINTER : 0) //#060c
      | (getAttrBool(n, "is_volatile") ? Xtype.TQ_FVOLATILE : 0)
      | (getAttrBool(n, "is_class") ? Xtype.TQ_FCLASS : 0)
      | (getAttrBool(n, "is_value") ? Xtype.TQ_FVALUE : 0)
      | (getAttrBool(n, "is_procedure") ? Xtype.TQ_FPROCEDURE : 0);

    String intent = getAttr(n, "intent");

    if (intent != null) {
      if (intent.equals("in")) {
	tq |= Xtype.TQ_FINTENT_IN;
      } else if (intent.equals("out")) {
	tq |= Xtype.TQ_FINTENT_OUT;
      } else if (intent.equals("inout")) {
	tq |= Xtype.TQ_FINTENT_INOUT;
      }
    }

    Xobject fkind = toXobject(getContent(getElement(n, "kind")));
    Xobject flen = null, sizeExprs[] = null, cosizeExprs[] = null;
    XobjList typeParamValues = null;
    Node nn, nnn;

    if ((nn = getElement(n, "len")) != null) {
      flen = toXobject(getContent(nn));
      if (flen == null)
        if (getAttrBool(nn, "is_assumed_shape"))
	  flen = Xcons.IntConstant(-1); // means variable length, "(len=:)"
        else if (getAttrBool(nn, "is_assumed_size"))
	  flen = Xcons.IntConstant(-2); // means variable length, "(len=*)"
        else
          fatal("array length unknown:" + nn);
    } else if ((nn = getElement(n, "typeParamValues")) != null) {
      typeParamValues = (XobjList)toXobject(nn);
    } else {
      NodeList list = n.getChildNodes();
      if (list.getLength() > 0) {
	List<Xobject> sizeExprList = new ArrayList<Xobject>();
	List<Xobject> cosizeExprList = new ArrayList<Xobject>();            // #060
	for (int i = 0; i < list.getLength(); i++) {
	  nn = list.item(i);
	  if (nn.getNodeType() != Node.ELEMENT_NODE)
	    continue;
	  if (nn.getNodeName() == "kind") {
	    continue;
	  }
	  Xobject x = toXobject(nn);
	  if (x.Opcode() == Xcode.F_ARRAY_INDEX
	      || x.Opcode() == Xcode.F_INDEX_RANGE) {
	    sizeExprList.add(x);
	  } else if (x.Opcode() == Xcode.F_CO_SHAPE) {              // #060
            NodeList colist = nn.getChildNodes();
            for (int j = 0; j < colist.getLength(); j++) {
              nnn = colist.item(j);
              if (nnn.getNodeType() != Node.ELEMENT_NODE)
                continue;
              if (nnn.getNodeName() == "kind") {
                continue;
              }
              Xobject xx = toXobject(nnn);
              if (xx.Opcode() == Xcode.F_ARRAY_INDEX
                  || xx.Opcode() == Xcode.F_INDEX_RANGE) {
                cosizeExprList.add(xx);
              } else
                fatal("bad coindex in type:" + nn);
            }
          } else
	    fatal("bad index in type:" + nn);
	}
	if (sizeExprList.size() > 0) {
	  sizeExprs = sizeExprList.toArray(new Xobject[0]);
	}
	if (cosizeExprList.size() > 0) {                                // #060
	  cosizeExprs = cosizeExprList.toArray(new Xobject[0]);
	}
      }
    }

    Xtype type;

    if (sizeExprs == null) {
      if (ti == null) { // inherited type such as structure
        String pass = getAttr(n, "pass");
        String pass_arg_name = getAttr(n, "pass_arg_name");
	Xtype ref = getType(getAttr(n, "ref"));
	xobjFile.addType(ref);
	type = ref.inherit(tid);
	type.setTypeQualFlags(tq);
        type.setCodimensions(cosizeExprs);                           // #060
        type.setFTypeParamValues(typeParamValues);
        type.setPass(pass);
        type.setPassArgName(pass_arg_name);
      } else {
	type = new BasicType(ti.type.getBasicType(), tid, tq, null,
			     fkind, flen, cosizeExprs);             // #060
      }
    } else {
      Xtype ref = getType(getAttr(n, "ref"));
      type = new FarrayType(tid, ref, tq, sizeExprs, cosizeExprs);    // #060
    }

    String bind = getAttr(n, "bind");
    if(bind != null){
        type.setBind(bind);
    }

    String bind_name = getAttr(n, "bind_name");
    if(bind_name != null){
        type.setBindName(bind_name);
    }

    Xtype dummy = getType(tid);
    if (dummy.getKind() == Xtype.UNDEF)
      dummy.assign(type);
    else
      fatal("type table entry is doubled: " + tid);
    xobjFile.addType(type);
  }

  /*
   * (params?)
   *
   * function foo(a, b) integer a, b real foo
   *
   * <FfunctionType type="F0" return_type="Freal"> <params> <name
   * type="Fint">a</name> <name type="Fint">b</name> </params>
   * </FfunctionType>
   */
  private void declFfunctionType(Node n) {
    String tid = getAttr(n, "type");
    Xtype retType = getType(getAttr(n, "return_type"));

    int tq = (getAttrBool(n, "is_external") ? Xtype.TQ_FEXTERNAL : 0)
      | (getAttrBool(n, "is_internal") ? Xtype.TQ_FINTERNAL : 0)
      | (getAttrBool(n, "is_intrinsic") ? Xtype.TQ_FINTRINSIC : 0)
      | (getAttrBool(n, "is_private") ? Xtype.TQ_FPRIVATE : 0)
      | (getAttrBool(n, "is_public") ? Xtype.TQ_FPUBLIC : 0)
      | (getAttrBool(n, "is_program") ? Xtype.TQ_FPROGRAM : 0)
      | (getAttrBool(n, "is_recursive") ? Xtype.TQ_FRECURSIVE : 0)
      | (getAttrBool(n, "is_module") ? Xtype.TQ_FMODULE : 0);

    Xobject params = toXobject(getElement(n, "params"));
    FunctionType type = new FunctionType(tid, retType, params, tq, false,
        null, getAttr(n, "result_name"));

    String bind = getAttr(n, "bind");
    if(bind != null){
        type.setBind(bind);
    }

    String bind_name = getAttr(n, "bind_name");
    if(bind_name != null){
        type.setBindName(bind_name);
    }

    xobjFile.addType(type);
  }

  /*
   * (symbols)
   *
   * type S integer x1, y1; end type S
   *
   * <FstructType type="TYPE_NAME"> <symbols> <id type="Fint"> <name
   * type="Fint">x1</name> </id> <id type="Fint"> <name type="Fint">y1</name>
   * </id> </symbols> </FstructType>
   */
  private void declFstructType(Node n) {
    String tid = getAttr(n, "type");
    String parent_tid = getAttr(n, "extends");
    int tq = (getAttrBool(n, "is_internal_private") ? Xtype.TQ_FINTERNAL_PRIVATE
	      : 0)
      | (getAttrBool(n, "is_private") ? Xtype.TQ_FPRIVATE : 0)
      | (getAttrBool(n, "is_public") ? Xtype.TQ_FPUBLIC : 0)
      | (getAttrBool(n, "is_sequence") ? Xtype.TQ_FSEQUENCE : 0);

    XobjList tparam_list = (XobjList) toXobject(getElement(n, "typeParams"));
    XobjList id_list = (XobjList) toXobject(getElement(n, "symbols"));
    XobjList proc_list = (XobjList) toXobject(getElement(n, "typeBoundProcedures"));
    StructType type = new StructType(tid, parent_tid, id_list, proc_list, tq, null, tparam_list);

    String bind = getAttr(n, "bind");
    if(bind != null){
        type.setBind(bind);
    }

    xobjFile.addType(type);
  }

  /*
   * global Ident section
   */
  void enterGlobalIdent(Node n) {
    if (n.getNodeName() == "id") {
      Ident id = toIdent(n);
      xobjFile.getGlobalIdentList().add(id);
    } else
      fatal("Unknown node in globalSybmols: " + n);
  }

  void enterGlobalDecl(Node n) {
    Xobject xobj = toXobject(n);
    xobj.setParentRecursively(null);
    switch (xobj.Opcode()) {
    case FUNCTION_DEFINITION:
    case F_MODULE_DEFINITION:
    case F_BLOCK_DATA_DEFINITION:
      xobjFile.add(xobj);
      break;
    default:
      fatal("Unknown node in globalDeclarations: " + n);
    }
  }

  Xobject toXobject(Node n) {
    if (n == null)
      return null;

    Xcode code = nameTable.getXcode(n.getNodeName());
    if (code == null) {
      fatal("unknown Xcode=" + n);
    }

    Xtype type = null;
    Xobject attr = null;
    Xobject x;

    String t = getAttr(n, "type");
    if (t != null)
      type = getType(t);

    switch (code) {
    case F_MODULE_PROCEDURE_DEFINITION:
    case FUNCTION_DEFINITION:
      x = Xcons.List(code, type, toXobject(getElement(n, "name")),
		     toXobject(getElement(n, "symbols")),
		     toXobject(getElement(n, "declarations")),
		     toXobject(getElement(n, "body")), null);

      markModuleVariable((XobjList)x.getArgOrNull(1),
			 (XobjList)x.getArgOrNull(2));
      setCommonAttributes(n, x);
      return x;

    case F_MODULE_DEFINITION:
      x = Xcons.List(code, type, getSymbol(n, "name"),
		     toXobject(getElement(n, "symbols")),
		     toXobject(getElement(n, "declarations"))
		     );
      x.add(toXobject(getElement(n, "FcontainsStatement")));
      x.add(getAttrIntFlag(n, "is_sub"));
      x.add(getSymbol(n, "parent_name"));

      markModuleVariable((XobjList)x.getArgOrNull(1),
			 (XobjList)x.getArgOrNull(2));
      setCommonAttributes(n, x);
      return x;

    case VAR_DECL:
      return setCommonAttributes(n,
				 Xcons.List(code, type, toXobject(getElement(n, "name")),
					    toXobject(getElement(n, "value")),	null));

    case F_USE_DECL:
    case F_USE_ONLY_DECL:
      x = getSymbol(n, "name");
      boolean isIntrinsic = getAttrBool(n, "intrinsic");
      return setCommonAttributes(n,
        Xcons.List(code, type, x, Xcons.IntConstant(isIntrinsic ? 1 : 0), getChildList(n)));

    case F_IMPORT_STATEMENT:
      return setCommonAttributes(n,
         Xcons.List(code, type, getChildList(n)));

    case F_MODULE_PROCEDURE_DECL:
      return setCommonAttributes(n, Xcons.List(code, type,
                                               getAttrIntFlag(n, "is_module_specified"),
                                               getChildList(n)));

    case F_INTERFACE_DECL:
      return setCommonAttributes(n, Xcons.List(code, type,
                                               getSymbol(n, "name"),
                                               getAttrIntFlag(n, "is_operator"),
                                               getAttrIntFlag(n, "is_assignment"),
                                               getChildList(n)));

    case F_BLOCK_DATA_DEFINITION:
      x = getSymbol(n, "name");
      return setCommonAttributes(n,
				 Xcons.List(code, type, x,
					    toXobject(getElement(n, "symbols")),  toXobject(getElement(n, "declarations"))));

    case FUNCTION_DECL:
      return setCommonAttributes(n, Xcons.List(code, type, toXobject(getElement(n, "name")),
					       null, null, toXobject(getElement(n, "declarations"))));

    case STRING:
      return Xcons.String(getContentText(n));

    case OMP_PRAGMA:
      XobjList xlist = Xcons.List(Xcode.OMP_PRAGMA);
      setCommonAttributes(n, xlist);
      return getChildList(n, xlist);

    case PRAGMA_LINE:
      String contentText = getContentText(n);
      x = Xcons.List(Xcode.PRAGMA_LINE,
		     new XobjString(Xcode.STRING, contentText));
      setCommonAttributes(n, x);
      return x;

    case INT_CONSTANT:
      BigInteger bi = new BigInteger(getContentText(n));
      int bl = bi.bitLength();
      String kind = getAttr(n, "kind");

      if (bl <= 31) {
	return Xcons.IntConstant(bi.intValue(), type, kind);
      } else {
	return Xcons.LongLongConstant(bi, type, kind);
      }

    case FLOAT_CONSTANT:
      return Xcons.FloatConstant(type, getContentText(n),
				 getAttr(n, "kind"));

    case F_CHARACTER_CONSTATNT:
      return Xcons.FcharacterConstant(type, getContentText(n),
				      getAttr(n, "kind"));

    case F_LOGICAL_CONSTATNT: {
      boolean value = false;
      if (getContentText(n).equalsIgnoreCase(".TRUE."))
	value = true;
      return Xcons.FlogicalConstant(type, value, getAttr(n, "kind"));
    }

    case FUNC_ADDR:
      return Xcons.Symbol(code, type, getContentText(n));

    case F_ALLOC:
    case F_ARRAY_REF:
    case CO_ARRAY_REF: {
      NodeList list = n.getChildNodes();
      int i;
      x = null;
      for (i = 0; i < list.getLength(); i++) {
	Node nn = list.item(i);
	if (nn.getNodeType() != Node.ELEMENT_NODE)
	  continue;
	x = toXobject(nn);
	i++;
	break;
      }
      XobjList xobjs = new XobjList(Xcode.LIST);
      for (; i < list.getLength(); i++) {
	Node nn = list.item(i);
	if (nn.getNodeType() != Node.ELEMENT_NODE)
	  continue;
	xobjs.add(toXobject(nn));
      }
      return Xcons.List(code, type, x, xobjs);
    }

    case MEMBER_REF:
      return Xcons.List(code, type, toXobject(getElement(n, "varRef")), getSymbol(n, "member"));

    case F_USER_BINARY_EXPR: {
      XobjList xx = Xcons.List(code, type);
      x = getChildList(n, xx);
    }
      t = getAttr(n, "name");
      x.add(Xcons.String(t));
      return x;

    case ID_LIST: // symbols
      return toIdentList(n);

    case IDENT: // name
      return Xcons.Symbol(code, type, getContentText(n));

    case VAR:
      VarScope scope = VarScope.get(getAttr(n, "scope"));
      return Xcons.Symbol(code, type, getContentText(n), scope);

    case F_INDEX_RANGE:
      return Xcons.List(code, type,
			toXobject(getContent(getElement(n, "lowerBound"))),
			toXobject(getContent(getElement(n, "upperBound"))),
			toXobject(getContent(getElement(n, "step"))),
			getAttrIntFlag(n, "is_assumed_shape"),
			getAttrIntFlag(n, "is_assumed_size"));

    case F_LEN:
      return Xcons.List(code, type,
			toXobject(getContent(n)),
			getAttrIntFlag(n, "is_assumed_shape"),
			getAttrIntFlag(n, "is_assumed_size"));

    case F_VALUE:
      {
	x = null;
	Node rc = getElement(n, "repeat_count");
	if (rc != null) {
	  x = toXobject(getContent(rc));
	}
	return Xcons.List(code, type, toXobject(getContent(n)), x);
      }

    case F_NAMED_VALUE: {
      String val1 = getAttr(n, "value");
      if (val1 != null) {
	x = Xcons.String(val1);
      } else {
	x = toXobject(getContent(n));
      }
      return Xcons.List(Xcode.F_NAMED_VALUE,
			Xcons.Symbol(Xcode.IDENT, getAttr(n, "name")), x);
    }

    case FUNCTION_CALL:
      Node n_func = getElement(n, "name");
      if (n_func == null) {
        n_func = getElement(n, "FmemberRef");
      }
      if (n_func == null) {
        fatal("unknown functionCall function.");
      }
      return Xcons.List(code, type, toXobject(n_func),
			toXobject(getElement(n, "arguments")),
			getAttrIntFlag(n, "is_intrinsic"));

    case F_IF_STATEMENT:
      attr = getSymbol(n, "construct_name");
      return setCommonAttributes(n, Xcons.List(code, type, attr,
					       toXobject(getContent(getElement(n, "condition"))),
					       toXobject(getContent(getElement(n, "then"))),
					       toXobject(getContent(getElement(n, "else")))));

    case F_DO_STATEMENT:
      attr = getSymbol(n, "construct_name");
      return setCommonAttributes(n, Xcons.List(code, type, attr,
					       toXobject(getElement(n, "Var")),
					       toXobject(getElement(n, "indexRange")),
					       toXobject(getElement(n, "body"))));

    case F_STATEMENT_LIST:
      {
	XobjList xx = Xcons.List(code, type);
	return getChildList(n, xx);
      }

    case F_RENAMABLE:
      return Xcons.List(code, type, getSymbol(n, "use_name"), getSymbol(n, "local_name"));

    case F_DO_WHILE_STATEMENT:
      attr = getSymbol(n, "construct_name");
      return setCommonAttributes(n, Xcons.List(code, type, attr,
					       toXobject(getContent(getElement(n, "condition"))),
					       toXobject(getElement(n, "body"))));

    case F_CONTINUE_STATEMENT:
    case RETURN_STATEMENT:
      return setCommonAttributes(n, Xcons.List(code, type));

    case F_CYCLE_STATEMENT:
    case F_EXIT_STATEMENT:
      attr = getSymbol(n, "construct_name");
      return setCommonAttributes(n, Xcons.List(code, type, attr));

    case GOTO_STATEMENT:
      x = getSymbol(n, "label_name");
      return setCommonAttributes(n, Xcons.List(code, type, x, toXobject(getElement(n, "value")), toXobject(getElement(n, "params"))));

    case STATEMENT_LABEL:
      t = getAttr(n, "label_name");
      attr = Xcons.Symbol(Xcode.IDENT, t);
      return setCommonAttributes(n, Xcons.List(code, type, attr));

    case F_SELECT_CASE_STATEMENT:
      {
	x = new XobjList(code, type);
	x.add(getSymbol(n, "construct_name"));
	NodeList list = n.getChildNodes();
	XobjList caseLabels = new XobjList();
	for (int i = 0; i < list.getLength(); i++) {
	  Node nn = list.item(i);
	  if (nn.getNodeType() != Node.ELEMENT_NODE)
	    continue;
	  if (nn.getNodeName() == "FcaseLabel") {
	    caseLabels.add(toXobject(nn));
	  } else {
	    x.add(toXobject(nn));
	  }
	}
	x.add(caseLabels);
	return setCommonAttributes(n, x);
      }

    case F_CASE_LABEL:
      {
	x = new XobjList(code, type);
	x.add(getSymbol(n, "construct_name"));
	NodeList list = n.getChildNodes();
	XobjList values = new XobjList();
	x.add(values);
	for (int i = 0; i < list.getLength(); i++) {
	  Node nn = list.item(i);
	  if (nn.getNodeType() != Node.ELEMENT_NODE)
	    continue;
	  if (nn.getNodeName() != "body") {
	    values.add(toXobject(nn));
	  } else {
	    x.add(toXobject(nn));
	  }
	}
	return setCommonAttributes(n, x);
      }

    case SELECT_TYPE_STATEMENT:
        {
          x = new XobjList(code, type);
          x.add(getSymbol(n, "construct_name"));
          NodeList list = n.getChildNodes();
          XobjList typeGuards = new XobjList();
          for (int i = 0; i < list.getLength(); i++) {
              Node nn = list.item(i);
              if (nn.getNodeType() != Node.ELEMENT_NODE)
                  continue;
              if (nn.getNodeName() == "typeGuard") {
                typeGuards.add(toXobject(nn));
              } else if (nn.getNodeName() == "id") {
                x.add(toIdent(nn));
              }
          }
          x.add(typeGuards);
          return setCommonAttributes(n, x);
        }
    case TYPE_GUARD:
        {
            x = new XobjList(code, type);
            x.add(getSymbol(n, "construct_name"));
            x.add(getSymbol(n, "kind"));
            x.add(getSymbol(n, "type"));
            NodeList list = n.getChildNodes();
            XobjList values = new XobjList();
            x.add(toXobject(getElement(n, "body")));
            return setCommonAttributes(n, x);
        }



    case F_WHERE_STATEMENT:
      x = new XobjList(code, type);
      x.add(null);
      x.add(toXobject(getContent(getElement(n, "condition"))));
      x.add(toXobject(getContent(getElement(n, "then"))));
      x.add(toXobject(getContent(getElement(n, "else"))));
      return setCommonAttributes(n, x);

    case F_STOP_STATEMENT:
      {
	t = getAttr(n, "code");
	Xobject cd = (t == null ? null : Xcons.String(t));
	t = getAttr(n, "message");
	Xobject mes = (t == null ? null : Xcons.FcharacterConstant(Xtype.FcharacterType, t, null));
	return setCommonAttributes(n, Xcons.List(code, type, cd, mes));
      }

    case F_PAUSE_STATEMENT:
      {
	t = getAttr(n, "code");
	Xobject cd = (t == null) ? null : Xcons.String(t);
	t = getAttr(n, "message");
	Xobject mes = (t == null) ? null : Xcons.FcharacterConstant(Xtype.FcharacterType, t, null);
	return setCommonAttributes(n, Xcons.List(code, type, cd, mes));
      }

    case F_READ_STATEMENT:
    case F_WRITE_STATEMENT:
      return setCommonAttributes(n, Xcons.List(code, type,
					       toXobject(getElement(n, "namedValueList")),
					       toXobject(getElement(n, "valueList"))
					       ));

    case F_PRINT_STATEMENT:
      t = getAttr(n, "format");
      return setCommonAttributes(n, Xcons.List(code, type,
					       Xcons.String(t),
					       toXobject(getElement(n, "valueList"))
					       ));

    case F_DO_LOOP:
      x = Xcons.List();
      {
	NodeList list = n.getChildNodes();
	for (int i = 0; i < list.getLength(); i++) {
	  Node nn = list.item(i);
	  if (nn.getNodeType() != Node.ELEMENT_NODE)
	    continue;
	  if (nn.getNodeName() != "value") {
	    continue;
	  }
	  x.add(toXobject(nn));
	}
      }
      return setCommonAttributes(n, Xcons.List(code, type,
					       toXobject(getElement(n, "Var")),
					       toXobject(getElement(n, "indexRange")),
					       x
					       ));

    case F_VAR_LIST:
      return Xcons.List(code, type, getSymbol(n, "name"), getChildList(n));

    case F_FORMAT_DECL:
      t = getAttr(n, "format");
      x = Xcons.String(t);
      return setCommonAttributes(n, Xcons.List(code, type, x));

    case F_DATA_STATEMENT:
    case F_DATA_DECL:
      return setCommonAttributes(n, Xcons.List(code, type, Xcons.List(toXobject(getElement(n, "varList")), toXobject(getElement(n, "valueList")))));

    case F_EQUIVALENCE_DECL:
      return setCommonAttributes(n, Xcons.List(code, type, getChildList(n)));

    case F_ALLOCATE_STATEMENT:
    case F_DEALLOCATE_STATEMENT:
      {
	x = Xcons.List(code, type);
	//x.add(getSymbol(n, "stat_name"));
	//x.add(getChildList(n));

	XobjList xx = Xcons.List();

	NodeList list = n.getChildNodes();
	for (int i = 0; i < list.getLength(); i++) {
	  Node nn = list.item(i);
	  String name = nn.getNodeName();
	  if (name == "allocOpt"){
	    switch (getAttr(nn, "kind")){
	    case "stat":
	      XobjList v = getChildList(nn);
	      x.add(v.getArg(0));
	      break;
	    default:
	      // for this moment, do nothing.
	    }
	    continue;
	  }
	  else if (name == "alloc"){
	    xx.add(toXobject(nn));
	  }
	  else
	    continue;
	}

	if (x.Nargs() == 0) x.add(null);
	x.add(xx);

	return setCommonAttributes(n, x);
      }
    case F_CONTAINS_STATEMENT:
      {
	XobjList xx = Xcons.List(code, type);
	return getChildList(n, xx);
      }

    case F_CRITICAL_STATEMENT:
      {
        attr = getSymbol(n, "construct_name");
        return setCommonAttributes(n, Xcons.List(code, type, attr,
						 toXobject(getElement(n, "body"))
						 ));
      }
    case F_BLOCK_STATEMENT:
      {
        attr = getSymbol(n, "construct_name");
        return setCommonAttributes(n, Xcons.List(code, type, attr,
						 toXobject(getElement(n, "symbols")),
						 toXobject(getElement(n, "declarations")),
						 toXobject(getElement(n, "body"))
						 ));
      }

    case F_SYNC_STAT:
      {
        attr = getSymbol(n, "kind");
        return setCommonAttributes(n, Xcons.List(code, type, attr,
                                                 toXobject(getContent(n))
                                                 ));
      }

    case F_TYPE_PARAM:
      {
        attr = getSymbol(n, "attr");
        return setCommonAttributes(n, Xcons.List(code, type, attr,
						 toXobject(getElement(n, "name")),
						 toXobject(getElement(n, "value"))
						 ));
      }

    case F_TYPE_BOUND_PROCEDURE:
      {
        XobjString pass     = Xcons.String(getAttr(n, "pass"         ));
        XobjString pass_arg = Xcons.String(getAttr(n, "pass_arg_name"));
        int tq = (getAttrBool(n, "is_private") ? Xtype.TQ_FPRIVATE : 0)
               | (getAttrBool(n, "is_public" ) ? Xtype.TQ_FPUBLIC  : 0);
        Node bdg = getElement(n, "binding");
        return setCommonAttributes(n, Xcons.List(code, type, pass, pass_arg,
                                                 toXobject(getElement(n, "name")),
                                                 Xcons.IntConstant(tq),
                                                 (bdg != null) ? toXobject(getContent(bdg)) : null,
                                                 getAttrIntFlag(n, "is_non_overridable")
                                                ));
      }

    case F_TYPE_BOUND_GENERIC_PROCEDURE:
      {
        int tq = (getAttrBool(n, "is_private") ? Xtype.TQ_FPRIVATE : 0)
               | (getAttrBool(n, "is_public" ) ? Xtype.TQ_FPUBLIC  : 0);
        Node bdg = getElement(n, "binding");
        return setCommonAttributes(n, Xcons.List(code, (Xtype)null,
                                                 getAttrIntFlag(n, "is_operator"),
                                                 getAttrIntFlag(n, "is_assignment"),
                                                 toXobject(getElement(n, "name")),
                                                 Xcons.IntConstant(tq),
                                                 toXobject(bdg)
                                                ));
      }

    case F_ARRAY_CONSTRUCTOR:
      {
        XobjList xobjp = new XobjList(code, type);
        xobjp.add(getSymbol(n, "element_type"));
        XobjList xobjs = new XobjList();
        xobjp.add(xobjs);
        NodeList list = n.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
          Node nn = list.item(i);
          if (nn.getNodeType() != Node.ELEMENT_NODE)
            continue;
          xobjs.add(toXobject(nn));
        }
        return setCommonAttributes(n, xobjp);
      }

    case F_CONDITION:
      return toXobject(getContent(n));

    case F_FORALL_STATEMENT:
      {
        XobjList xobj = new XobjList(code, type);
        xobj.add(getSymbol(n, "construct_name"));
        NodeList list = n.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
          Node nn = list.item(i);
          if (nn.getNodeType() != Node.ELEMENT_NODE)
            continue;
          xobj.add(toXobject(nn));
        }
        return setCommonAttributes(n, xobj);
      }

    case F_TYPE_BOUND_PROCEDURES:
    case F_BINDING:
    case F_SYNCALL_STATEMENT:
    case F_SYNCIMAGE_STATEMENT:
    case F_SYNCMEMORY_STATEMENT:
    case F_LOCK_STATEMENT:
    case F_UNLOCK_STATEMENT:
    default: // default action, make list
      NodeList list = n.getChildNodes();
      if(code == Xcode.LIST && list.getLength() == 0)
	return null;
      XobjList xobjs = new XobjList(code, type);
      for (int i = 0; i < list.getLength(); i++) {
	Node nn = list.item(i);
	if (nn.getNodeType() != Node.ELEMENT_NODE)
	  continue;
	xobjs.add(toXobject(nn));
      }
      return setCommonAttributes(n, xobjs);
    }
  }

  private XobjString getSymbol(Node n, String attr) {
    String value = getAttr(n, attr);
    if (value == null) {
      return null;
    } else {
      return Xcons.Symbol(Xcode.IDENT, value);
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
    Xobject addr = null;
    Node valueNode = getElement(n, "value");

    if (sclassStr != null) {
      sclass = StorageClass.get(sclassStr);

      switch (sclass) {
      case FLOCAL:
      case FSAVE:
      case FCOMMON:
      case FPARAM:
	addr = Xcons.Symbol(Xcode.VAR, type, name);
	addr.setScope(sclass == StorageClass.FPARAM ? VarScope.PARAM
		      : VarScope.LOCAL);
	break;
      case FFUNC:
	addr = Xcons.Symbol(Xcode.FUNC_ADDR, type, name);
	addr.setScope(VarScope.LOCAL);
	break;
      }
    } else if (valueNode != null) {
        addr = toXobject(valueNode);
    }

    // create ident
    //   ### It might be better to set codimensions here...
    Ident ident = new Ident(name, sclass, type, addr, 0, null, 0, null,
			    null, null, null/*codimensions*/);

    if (type != null && StorageClass.FTYPE_NAME.equals(sclass))
      type.setTagIdent(ident);

    // declaring
    if (sclass != null && sclass.isVarOrFunc()) {
      ident.setIsDeclared(true);
    }

    String moduleStr = getAttr(n, "declared_in");
    if(moduleStr != null) ident.setFdeclaredModule(moduleStr);

    return ident;
  }

//   public XobjectFile read(Reader reader) {
//     XobjectFile objFile = super.read(reader);
//     return objFile;
//   }

  private void markModuleVariable(XobjList ids, XobjList decls) {
    if(ids == null)
      return;
    // mark module variable
    Set<String> declSet = new HashSet<String>();
    if(decls != null) {
      for(Xobject a : decls) {
	if(a.Opcode() == Xcode.VAR_DECL)
	  declSet.add(a.getArg(0).getName());
      }
    }

    for(Xobject a : ids) {
      Ident id = (Ident)a;
      if((id.getStorageClass() == StorageClass.FSAVE ||
	  id.getStorageClass() == StorageClass.FLOCAL) &&
	 !declSet.contains(id.getName())) {
	id.setIsFmoduleVar(true);
      }
    }
  }
}
