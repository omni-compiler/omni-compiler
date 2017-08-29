/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import static xcodeml.util.XmLog.fatal;
import static xcodeml.util.XmLog.fatal_dump;
import xcodeml.util.XmDomUtil;
import xcodeml.util.ILineNo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import org.w3c.dom.Element;
import org.w3c.dom.Node;

import exc.object.*;
import exc.openmp.OMPpragma;


/**
 * convert Xobject/F to XcodeML(DOM)/F
 */
public class XmfXobjectToXcodeTranslator extends XmXobjectToXcodeTranslator {
    private XcodeMLNameTable_F nameTable = new XcodeMLNameTable_F();

    @Override
    void transGlobalDeclarations(Element globalDecl, XobjectDefEnv defList) {
        addDeclForNotDeclared(defList,
                              (XobjList)defList.getGlobalIdentList(),
                              null);

        for (XobjectDef def : defList) {
            if (def == null) {
                fatal("def is null");
            }
            addChildNodes(globalDecl, transDef(def));
        }
    }

    @Override
    public Element trans(Xobject xobj) {
        if (xobj == null) {
            return null;
        }

	//System.out.println("trans="+xobj);

        Element e = null;

        if (xobj instanceof Ident) {
            Ident i = (Ident)xobj;
            switch (i.Type().getKind()) {
            case Xtype.FUNCTION:
                e = createElement("Ffunction");
                break;
            default:
                e = createElement("Var",
                                  "scope", i.getScope() != null ? i.getScope().toXcodeString() : null);
                break;
            }
            addChildNode(e, trans(i.getName()));
            addAttributes(e, "type", i.Type().getXcodeFId());
            return e;
        }

        final Xcode xcode = xobj.Opcode();
        final String name = nameTable.getName(xcode);
//System.out.println(xobj.toString());
        switch (xcode) {
        case F_ARRAY_INDEX:
            e = addChildNode(createElement(name), transExpr(xobj.getArg(0)));
            break;

        case F_INDEX_RANGE: {
            e = createElement(name,
                              "is_assumed_shape", intFlagToBoolStr(xobj.getArgOrNull(3)),
                              "is_assumed_size", intFlagToBoolStr(xobj.getArgOrNull(4)));
            Xobject lb = xobj.getArg(0);
            if (lb != null) {
                addChildNode(e,
                             addChildNode(createElement("lowerBound"),
                                          transExpr(lb)));
            }
            Xobject ub = xobj.getArg(1);
            if (ub != null) {
                addChildNode(e,
                             addChildNode(createElement("upperBound"),
                                          transExpr(ub)));
            }
            Xobject st = xobj.getArg(2);
            if (st != null) {
                addChildNode(e,
                             addChildNode(createElement("step"),
                                          transExpr(st)));
            }
        }
            break;

        case F_LEN: {
            e = createElement(name,
                              "is_assumed_shape", intFlagToBoolStr(xobj.getArgOrNull(1)),
                              "is_assumed_size", intFlagToBoolStr(xobj.getArgOrNull(2)));
            Xobject len = xobj.getArg(0);
            if (len != null) {
                addChildNode(e, transExpr(len));
            }
        }
            break;

        case FUNCTION_DEFINITION:
        case F_MODULE_PROCEDURE_DEFINITION: {
            XobjList symbols = (XobjList)xobj.getArg(1);
            Xobject body = xobj.getArg(3);
            XobjList decls =
                (XobjList)addDeclForNotDeclared((XobjList)xobj.getArg(2),
                                                symbols,
                                                body);
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)),
                              transSymbols(symbols),
                              transDeclarations(decls),
                              transBody(body));
        }
            break;

        case VAR_DECL: {
            // (CODE name init)
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)),
                              transValue(xobj.getArgOrNull(1)));
        }
            break;

        case FUNCTION_DECL: {
            // (CODE name () () declarations)
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)),
                              transDeclarations(xobj.getArgOrNull(3)));
        }
            break;

        case F_DATA_STATEMENT:
        case F_DATA_DECL: {
            e = createElement(name);
            for (Xobject xseq : (XobjList)xobj) {
                addChildNodes(e,
                              trans(xseq.getArg(0)),
                              trans(xseq.getArg(1)));
            }
        }
            break;

        case F_VAR_LIST:
            e = createElement(name,
                              "name", getArg0Name(xobj));
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNode(e, trans(a));
            }
            break;

        case F_VALUE_LIST:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, transValue(a));
            }
            break;

        case F_DO_LOOP:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)),
                              trans(xobj.getArg(1)));
            for (Xobject a : (XobjList)xobj.getArg(2)) {
                addChildNode(e, transValue(a));
            }
            break;

        case F_BLOCK_DATA_DEFINITION:
            e = addChildNodes(createElement(name,
                                            "name", getArg0Name(xobj)),
                              transSymbols(xobj.getArg(1)),
                              transDeclarations(xobj.getArg(2)));
            break;

        case F_ENTRY_DECL:
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)));
            break;

        case F_EXTERN_DECL:
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)));
            break;

        case F_EQUIVALENCE_DECL:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNodes(e,
                              trans(a.getArg(0)),
                              trans(a.getArg(1)));
            }
            break;

        case F_COMMON_DECL:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, trans(a));
            }
            break;

        case F_MODULE_DEFINITION: {
            Xobject parent_name = xobj.getArgOrNull(5);
            e = createElement(name,
                              "name", getArg0Name(xobj),
                              "is_sub", intFlagToBoolStr(xobj.getArgOrNull(4)),
                              "parent_name", (parent_name != null) ? parent_name.getName() : null);
            XobjList symbols = (XobjList)xobj.getArgOrNull(1);
            XobjList decls =
                (XobjList)addDeclForNotDeclared((XobjList)xobj.getArgOrNull(2),
                                                symbols,
                                                null);
	    //System.out.println("module arg3="+ xobj.getArgOrNull(3));
            addChildNodes(e,
                          transSymbols(symbols),
                          transDeclarations(decls));
        }
            break;

        case F_MODULE_PROCEDURE_DECL:
            e = createElement(name, "is_module_specified", intFlagToBoolStr(xobj.getArg(0)));
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNode(e, transName(a));
            }
            break;

        case F_INTERFACE_DECL:
            // (CODE is_operator is_assignment name (LIST ... ))
            e = createElement(name,
                              "name", getArg0Name(xobj),
                              "is_operator", intFlagToBoolStr(xobj.getArg(1)),
                              "is_assignment", intFlagToBoolStr(xobj.getArg(2)));
            for (Xobject a : (XobjList)xobj.getArg(3)) {
                addChildNode(e, trans(a));
            }
            break;

        case F_FORMAT_DECL:
            e = createElement(name,
                              "format", xobj.getArg(0).getString());
            break;

        case F_NAMELIST_DECL:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, trans(a));
            }
            break;

        case F_STRUCT_DECL:
            e = addChildNode(createElement(name), transName(xobj.getArg(0)));
            break;

        case F_USE_DECL:
            e = createElement(name,
                              "name", xobj.getArg(0).getName(),
                              "intrinsic", intFlagToBoolStr(xobj.getArg(1)));
            if (xobj.Nargs() > 2) {
                for (Xobject a : (XobjList)xobj.getArg(2)) {
                    addChildNode(e, trans(a));
                }
            }
            break;

        case F_USE_ONLY_DECL:
            e = createElement(name,
                              "name", xobj.getArg(0).getName(),
                              "intrinsic", intFlagToBoolStr(xobj.getArg(1)));
            for (Xobject a : (XobjList)xobj.getArg(2)) {
                addChildNode(e, trans(a));
            }
            break;

        case F_IMPORT_STATEMENT:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj.getArg(0)) {
                addChildNode(e, transName(a));
            }
            break;

        case F_RENAME:
            e = createElement(name,
                              "use_name", xobj.getArg(0).getName(),
                              "local_name", (xobj.getArgOrNull(1) != null ? xobj.getArg(1).getName() : null));
            break;

        case F_RENAMABLE:
            e = createElement(name,
                              "use_name", xobj.getArg(0).getName(),
                             "local_name", (xobj.getArgOrNull(1) != null ? xobj.getArg(1).getName() : null));
            break;

        case EXPR_STATEMENT: {
            Xobject xexpr = xobj.getArg(0);
            switch (xexpr.Opcode()) {
            case FUNCTION_CALL:
                e = addChildNode(createElement(name),
                                 transExpr(xexpr));
                break;
            case ASSIGN_EXPR:
                e = addChildNodes(createElement("FassignStatement"),
                                  trans(xexpr.getArg(0)),
                                  transExpr(xexpr.getArg(1)));
                break;
            default:
                fatal_dump("cannot convert Xcode to XcodeML.", xobj);
                break;
            }
        }
            break;

        case F_ASSIGN_STATEMENT:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)),
                              transExpr(xobj.getArg(1)));
            break;

        case F_POINTER_ASSIGN_STATEMENT:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)),
                              transExpr(xobj.getArg(1)));
            break;

        case F_DO_STATEMENT: {
            e = createElement(name,
                              "construct_name", getArg0Name(xobj));
            Xobject var = xobj.getArgOrNull(1);
            Xobject idxRange = xobj.getArgOrNull(2);
            if (var != null) {
                addChildNodes(e,
                              trans(var),
                              trans(idxRange));
            }
            addChildNode(e, transBody(xobj.getArg(3)));
        }
            break;

        case F_DO_WHILE_STATEMENT:
            e = addChildNodes(createElement(name,
                                            "construct_name", getArg0Name(xobj)),
                              transCondition(xobj.getArg(1)),
                              transBody(xobj.getArg(2)));
            break;

        case WHILE_STATEMENT:
            e = addChildNodes(createElement(name, // "FdoWhileStatement".
                                            "construct_name", getArg0Name(xobj)),
                              transCondition(xobj.getArg(0)),
                              transBody(xobj.getArg(1)));
            break;

        case F_SELECT_CASE_STATEMENT:
            e = addChildNodes(createElement(name,
                                            "construct_name", getArg0Name(xobj)),
                              transValue(xobj.getArg(1)));

            Xobject caseList = xobj.getArg(2);

            if (caseList.Opcode() == Xcode.F_STATEMENT_LIST){
                for (Xobject a : (XobjList)caseList) {
                    if (a.Opcode() == Xcode.F_STATEMENT_LIST){
                        for (Xobject b : (XobjList)a){
                            addChildNode(e, trans(b));
                        }
                    } else {
                        addChildNode(e, trans(a));
                    }
                }
            } else {
                addChildNode(e, trans(caseList));
            }
            break;

        case F_CASE_LABEL:
            {
                e = createElement(name,
                                  "construct_name", getArg0Name(xobj));
                XobjList values = (XobjList)xobj.getArg(1);
                if (values != null) {
                    for (Xobject a : values) {
                        addChildNode(e, trans(a));
                    }
                }
                addChildNode(e, transBody(xobj.getArg(2)));
            }
            break;
        case SELECT_TYPE_STATEMENT:
            e = createElement(name,
                              "construct_name", getArg0Name(xobj));
            addChildNode(e, transIdent((Ident)xobj.getArg(1)));
            XobjList typeGuards = (XobjList)xobj.getArg(2);
            for (Xobject a : typeGuards) {
                addChildNode(e, trans(a));
            }
            break;
        case TYPE_GUARD:
            {
                e = createElement(name, "construct_name", getArg0Name(xobj),
                                  "kind", (xobj.getArgOrNull(1) != null ? xobj.getArg(1).getName() : null),
                                  "type", (xobj.getArgOrNull(1) != null ? xobj.getArg(1).getName() : null)
                );
                addChildNode(e, transBody(xobj.getArg(3)));
            }
            break;


        case F_WHERE_STATEMENT: {
            e = addChildNodes(createElement(name),
                              transCondition(xobj.getArg(1)),
                              transThen(xobj.getArgOrNull(2)));
            Xobject xelse = xobj.getArgOrNull(3);
            if (xelse != null && xelse.Nargs() > 0) {
                addChildNode(e, transElse(xelse));
            }
        }
            break;

        case F_IF_STATEMENT: {
            e = addChildNodes(createElement(name,
                                            "construct_name", getArg0Name(xobj)),
                              transCondition(xobj.getArg(1)),
                              transThen(xobj.getArgOrNull(2)));
            Xobject xelse = xobj.getArgOrNull(3);
            if (xelse != null && xelse.Nargs() > 0) {
                addChildNode(e, transElse(xelse));
            }
        }
            break;

        case IF_STATEMENT: {
            e = addChildNodes(createElement(name), // "FifStatement"
                              transCondition(xobj.getArg(0)),
                              transThen(xobj.getArgOrNull(1)));
            Xobject xelse = xobj.getArgOrNull(2);
            if (xelse != null && xelse.Nargs() > 0) {
                addChildNode(e, transElse(xelse));
            }
        }
            break;

        case F_CYCLE_STATEMENT:
            e = createElement(name,
                              "construct_name", getArg0Name(xobj));
            break;

        case F_EXIT_STATEMENT:
            e = createElement(name,
                              "construct_name", getArg0Name(xobj));
            break;

        case F_CONTINUE_STATEMENT:
        case RETURN_STATEMENT:
            e = createElement(name);
            break;

        case GOTO_STATEMENT: {
            e = createElement(name,
                              "label_name", getArg0Name(xobj));
            Xobject value = xobj.getArgOrNull(1);
            Xobject params = xobj.getArgOrNull(2);
            if (value != null) {
                addChildNodes(e,
                              transParams((XobjList)params),
                              transValue(value));
            }
        }
            break;

        case STATEMENT_LABEL:
            e = createElement(name,
                              "label_name", getArg0Name(xobj));
            break;

        case F_CONTAINS_STATEMENT:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, trans(a));
            }
            break;

        case PRAGMA_LINE:
        case TEXT:
            e = addChildNode(createElement(name),
                             trans(xobj.getArg(0).getString()));
            break;

        case F_ALLOCATE_STATEMENT:
	  // System.out.println("xobj="+xobj);
             // e = createElement(name,
	     // 		       "stat_name", getArg0Name(xobj));
	    e = createElement(name);
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNode(e, trans(a));
            }
	    if (xobj.getArg(0) != null){
	      Element o = createElement("allocOpt", "kind", "stat");
	      addToBody(o, xobj.getArg(0));
	      addChildNode(e, o);
	    }

// 	  e = createElement(name);
// 	  for (Xobject a : (XobjList)xobj.getArg(0)) {
// 	    addChildNode(e, trans(a));
// 	  }
	    break
	      ;

        case F_DEALLOCATE_STATEMENT:
            e = createElement(name,
                              "stat_name", getArg0Name(xobj));
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNode(e, trans(a));
            }
            break;

        case F_NULLIFY_STATEMENT:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, trans(a));
            }
            break;

        case F_ALLOC:
            e = addChildNode(createElement(name),
                             trans(xobj.getArg(0)));
            if (xobj.getArgOrNull(1) != null) {
                for (Xobject a : (XobjList)xobj.getArgOrNull(1)) {
                    addChildNode(e, trans(a));
                }
            }
            break;

        case F_OPEN_STATEMENT:
        case F_CLOSE_STATEMENT:
        case F_END_FILE_STATEMENT:
        case F_REWIND_STATEMENT:
        case F_BACKSPACE_STATEMENT:
            e = addChildNode(createElement(name),
                             trans(xobj.getArgOrNull(0)));
            break;

        case F_NAMED_VALUE_LIST:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, trans(a));
            }
            break;

        case F_NAMED_VALUE: {
            e = createElement(name,
                              "name", xobj.getArg(0).getName());
            Xobject value = xobj.getArg(1);
            if (value.Opcode() == Xcode.STRING) {
                addAttributes(e,
                              "value", value.getString());
            } else {
                addChildNode(e, transExpr(value));
            }
        }
            break;

        case F_READ_STATEMENT:
        case F_WRITE_STATEMENT:
        case F_INQUIRE_STATEMENT:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)),
                              trans(xobj.getArg(1)));
            break;

        case F_PRINT_STATEMENT:
            e = addChildNode(createElement(name,
                                           "format", xobj.getArg(0).getString()),
                             trans(xobj.getArgOrNull(1)));
            break;

        case F_PAUSE_STATEMENT:
        case F_STOP_STATEMENT: {
            e = createElement(name);
            Xobject code = xobj.getArgOrNull(0);
            Xobject msg = xobj.getArgOrNull(1);
            if (code != null) {
                addAttributes(e,
                              "code", code.getString());
            }
            if (msg != null) {
                addAttributes(e,
                              "message", msg.getString());
            }
        }
            break;

        case F_CRITICAL_STATEMENT:
            e = createElement(name, "construct_name", getArg0Name(xobj));
            addChildNode(e, transBody((XobjList)xobj.getArg(1)));
            break;
        case F_BLOCK_STATEMENT:
            e = createElement(name, "construct_name", getArg0Name(xobj));
            XobjList identList = (XobjList)xobj.getArg(1);
            XobjList declList = (XobjList)xobj.getArg(2);
            Xobject  body = (XobjList)xobj.getArg(3);
            XobjList addDeclList = (XobjList)addDeclForNotDeclared(declList, identList, body);
            e = addChildNodes(e,
                              transSymbols(identList),
                              transDeclarations(declList),
                              transBody(body));
            break;

        case F_SYNCALL_STATEMENT:
        case F_SYNCIMAGE_STATEMENT:
        case F_SYNCMEMORY_STATEMENT:
        case F_LOCK_STATEMENT:
        case F_UNLOCK_STATEMENT:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, trans(a));
            }
            break;

        case F_SYNC_STAT:
            e = createElement(name, "kind", getArg0Name(xobj));
            addChildNode(e, trans(xobj.getArg(1)));
            break;

        case VAR:
            e = addChildNodes(createElement(name,
                                            "scope", xobj.getScope() != null ? xobj.getScope().toXcodeString() : null),
                              trans(xobj.getName()));
            break;

        case F_VAR_REF:
            e = addChildNode(createElement(name),
                             trans(xobj.getArg(0)));
            break;

        case F_ARRAY_REF:
            e = addChildNode(createElement(name),
                             trans(xobj.getArg(0)));
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNode(e, trans(a));
            }
            break;

        case CO_ARRAY_REF:                                     // #060
            e = addChildNode(createElement(name),
                             trans(xobj.getArg(0)));
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNode(e, trans(a));
            }
            break;

        case F_CO_SHAPE:                                        // #060
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, trans(a));
            }
            break;

        case F_USER_UNARY_EXPR:
        case LOG_NOT_EXPR:
        case UNARY_MINUS_EXPR:
            e = addChildNode(createElement(name),
                             transExpr(xobj.getArg(0)));
            if (xcode == Xcode.F_USER_UNARY_EXPR) {
                addAttributes(e,
                              "name", xobj.getArg(1).getString());
            }
            break;

        case PLUS_EXPR:
        case MINUS_EXPR:
        case MUL_EXPR:
        case DIV_EXPR:
        case F_POWER_EXPR:
        case LOG_AND_EXPR:
        case LOG_EQ_EXPR:
        case F_LOG_EQV_EXPR:
        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case LOG_NEQ_EXPR:
        case F_LOG_NEQV_EXPR:
        case LOG_OR_EXPR:
        case F_CONCAT_EXPR:
        case F_USER_BINARY_EXPR: {
            e = addChildNodes(createElement(name),
                              transExpr(xobj.getArg(0)),
                              transExpr(xobj.getArg(1)));
            if (xcode == Xcode.F_USER_BINARY_EXPR) {
                addAttributes(e,
                              "name", xobj.getArg(2).getString());
            }
        }
            break;

        case F_CHARACTER_REF:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)),
                              trans(xobj.getArg(1)));
            break;

        case INT_CONSTANT:
        case LONGLONG_CONSTANT:
        case FLOAT_CONSTANT:
        case F_LOGICAL_CONSTATNT:
        case F_CHARACTER_CONSTATNT:
        case STRING_CONSTANT:
            e = addChildNode(createElement(name,
                                           "kind", ((XobjConst)xobj).getFkind()),
                             trans(getConstContent(xobj)));
            break;

        case F_COMPLEX_CONSTATNT:
            e = addChildNodes(createElement(name),
                              transExpr(xobj.getArg(0)),
                              transExpr(xobj.getArg(1)));
            break;

        case FUNC_ADDR:
            e = addChildNode(createElement(name), // "Ffunction"
                             trans(xobj.getName()));
            break;

        case FUNCTION_CALL:
            e = addChildNodes(createElement(name,
                                            "is_intrinsic",
                                            intFlagToBoolStr(xobj.getArgOrNull(2))),
                              (xobj.getArg(0).Opcode() != Xcode.MEMBER_REF) ? transName(xobj.getArg(0)) :
                                                                              trans    (xobj.getArg(0)));
            if (xobj.getArg(1) != null) {
                Element argsElem = createElement("arguments");
                for (Xobject a : (XobjList)xobj.getArg(1)) {
                    addChildNode(argsElem, trans(a));
                }
                addChildNode(e, argsElem);
            }
            break;

        case MEMBER_REF:
            e = addChildNode(createElement(name,
                                           "member", xobj.getArg(1).getName()),
                             trans(xobj.getArg(0)));
            break;

        case F_ARRAY_CONSTRUCTOR:
            e = createElement(name, "element_type", getArg0Name(xobj));
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNode(e, transExpr(a));
            }
            break;
        case F_STRUCT_CONSTRUCTOR:
        case F_TYPE_PARAMS:
        case F_TYPE_PARAM_VALUES:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNode(e, transExpr(a));
            }
            break;

        case F_TYPE_PARAM:
            e = addChildNode(addChildNode(createElement(name, "attr", xobj.getArg(0).getName()),
                                          transName(xobj.getArg(1))),
                             trans(xobj.getArg(2)));
            break;

        case F_VALUE:
            e = transValue(xobj);
            break;

        case F_FORALL_STATEMENT: {
              e = createElement(name, "type", (xobj.Type() != null) ? xobj.Type().getXcodeFId() : null,
                                      "construct_name", (xobj.getArg(0) != null) ? xobj.getArg(0).getName() : null);
              addChildNode(e, trans(xobj.getArg(1))); // VAR 1
              addChildNode(e, trans(xobj.getArg(2))); // INDEX_RANGE 1
              int idx = 3;
              if (xobj.getArgOrNull(idx + 1) == null) {
                  // 0:LABEL , 1:VAR1 , 2:INDEX_RANGE1 , 3:BODY
                  // just continue...
              } else {
                  while (xobj.getArgOrNull(idx + 2) != null)
                      addChildNode(e, trans(xobj.getArg(idx++)));
                  if ((idx % 2) == 1)
                      addChildNode(e, addChildNode(createElement("condition"), trans(xobj.getArg(idx++))));
                  else
                      addChildNode(e, trans(xobj.getArg(idx++)));
              }
              addChildNode(e, transBody(xobj.getArg(idx)));
              break;
            }

        case NULL:
            return null;
        case STRING:
        	e =createElement("srting");
        	if(xobj.getName().equals("SCHED_DYNAMIC"))
        		xobj.setName("DYNAMIC");
        	if(xobj.getName().equals("SCHED_STATIC"))
        		xobj.setName("STATIC");
        	if(xobj.getName().equals("SCHED_RUNTIME"))
        		xobj.setName("RUNTIME");
        	if(xobj.getName().equals("SCHED_NONE"))
        		xobj.setName("NONE");
        	if(xobj.getName().equals("DEFAULT_NONE"))
        		xobj.setName("NONE");
        	if(xobj.getName().equals("DEFAULT_SHARED"))
        		xobj.setName("SHARED");
        	if(xobj.getName().equals("DEFAULT_PRIVATE"))
        		xobj.setName("PRIVATE");
        	switch(xobj.Opcode())
        	{
        	case OMP_PRAGMA:
        		switch(OMPpragma.valueOf(xobj))
        		{
    			case SCHED_STATIC:
    				xobj.setName("STATIC");
    				break;
    			case SCHED_DYNAMIC:
    				xobj.setName("DYNAMIC");
    				break;
    			case SCHED_GUIDED:
    				xobj.setName("GUIDED");
    				break;
    			case SCHED_RUNTIME:
    				xobj.setName("RUNTIME");
    				break;
    			case SCHED_NONE:
    				xobj.setName("NOME");
    				break;
    			default:
        			fatal_dump("OMP SCHED error",xobj);
        		}
        	}
        	addChildNode(e,trans(xobj.getName()));
        	break;

        case OMP_PRAGMA: {
	    e = createElement(name);

	    Element f0 = createElement("string");

	    addChildNode(f0, trans(xobj.getArg(0).getString()));
	    addChildNode(e, f0);

	    Element f1 = createElement("list");
	    Xobject clause = xobj.getArg(1);
	    if (clause != null){
		for (Xobject a : (XobjList)clause){

			if (a instanceof XobjString){
			addChildNode(f1, trans(a));
		    }
		    else {
			Element g = createElement("list");

			addChildNode(g, trans(a.getArg(0).getString()));
			Xobject vars = a.getArg(1);
			if (vars != null){
			    Element g1 = createElement("list");
			    if (vars instanceof XobjList){
				for (Xobject b : (XobjList)vars){
				    addChildNode(g1, trans(b));
				}
			    }
			    else {
				addChildNode(g1, trans(vars));
			    }
			    addChildNode(g, g1);
			}

			addChildNode(f1, g);
		    }
		}
            }
	    addChildNode(e, f1);

	    Element f2 = createElement("list");
	    body = xobj.getArg(2);
	    if (body != null){
		if (body.Opcode() == Xcode.F_STATEMENT_LIST){
		    for (Xobject a : (XobjList)body){
			if (a.Opcode() == Xcode.F_STATEMENT_LIST){
			    for (Xobject b : (XobjList)a){
				if (b.Opcode() == Xcode.F_STATEMENT_LIST){
				    for (Xobject c : (XobjList)b){
					addChildNode(f2, trans(c));
				    }
				}
				else {
				    addChildNode(f2, trans(b));
				}
			    }
			}
			else {
			    addChildNode(f2, trans(a));
			}
		    }
		}
		else {
		    addChildNode(f2, trans(body));
		}
            }
	    addChildNode(e, f2);

        }
//System.out.println(xcode.toString());
	    break;

	case XMP_PRAGMA: {
	    e = createElement(name);

	    Element f0 = createElement("string");

	    addChildNode(f0, trans(xobj.getArg(0).getString()));
	    addChildNode(e, f0);
	    addChildNode(e, trans(xobj.getArg(1)));
	    addChildNode(e, trans(xobj.getArg(2)));
	}
	    break;

        case ACC_PRAGMA: {
            e = createElement(name);

            //add directive-name
            Element f0 = createElement("string");
            addChildNode(f0, trans(xobj.getArg(0).getString()));
            addChildNode(e, f0);

            //add clauses
            Element f1 = createElement("list");
            XobjList clauseList = (XobjList)xobj.getArg(1);
            for(Xobject clause : clauseList){
                addChildNode(f1, transAccClause(clause));
            }
            addChildNode(e, f1);

            //add body
            Element f2 = createElement("list");
            body = xobj.getArg(2);
            addToBody(f2, body);
            addChildNode(e, f2);
        }
        break;
        default:
            fatal_dump("cannot convert Xcode to XcodeML.", xobj);
        }

        if (xobj.Type() != null) {
            String tid = xobj.Type().getXcodeFId();
            if (tid == null || tid.equals("null")) {
	      Xtype t = xobj.Type();
	      if(t.isBasic() &&
		 t.getBasicType() == BasicType.DOUBLE){
		addAttributes(e, "type", "Freal");
		addAttributes(e, "kind", "8");
	      } else
	      fatal("type is null");
            } else
	      addAttributes(e, "type", tid);
        }

        if (xobj.getLineNo() != null) {
            /*
             * XXX workaround
             * IXbfIOStatement, IXbfRWStatement を継承するクラスは
             * IXbLineNo を継承していないので LineNo を設定しないようにする。
             * これは旧コードとの互換のために行っている。
             */
            switch (xcode) {
            // IXbfIOStatement
//          case F_OPEN_STATEMENT:
//          case F_CLOSE_STATEMENT:
//          case F_END_FILE_STATEMENT:
//          case F_REWIND_STATEMENT:
//          case F_BACKSPACE_STATEMENT:
            // IXbfRWStatement
//          case F_READ_STATEMENT:
//          case F_WRITE_STATEMENT:
//          case F_INQUIRE_STATEMENT:
//              break;
            default:
                ILineNo lineNo = xobj.getLineNo();
                addAttributes(e,
                              "lineno", Integer.toString(lineNo.lineNo()),
                              "file", lineNo.fileName());
                break;
            }
        }
        return e;
    }

    @Override
    protected void preprocess(XobjectFile xobjFile) {
        xobjFile.setParentRecursively(null);
    }

    private void setBasicTypeFlags(Element basicTypeElem, Xtype type) {
        addAttributes(basicTypeElem,
                      "is_public", toBoolStr(type.isFpublic()),
                      "is_private", toBoolStr(type.isFprivate()),
                      "is_pointer", toBoolStr(type.isFpointer()),
                      "is_target", toBoolStr(type.isFtarget()),
                      "is_optional", toBoolStr(type.isFoptional()),
                      "is_save", toBoolStr(type.isFsave()),
                      "is_parameter", toBoolStr(type.isFparameter()),
                      "is_allocatable", toBoolStr(type.isFallocatable()),
                      "is_cray_pointer", toBoolStr(type.isFcrayPointer()),
                      "is_volatile", toBoolStr(type.isFvolatile()),
                      "is_class", toBoolStr(type.isFclass()),
                      "is_value", toBoolStr(type.isFvalue()));

        if (type.isFintentIN()) {
            addAttributes(basicTypeElem, "intent", "in");
        }
        if (type.isFintentOUT()) {
            addAttributes(basicTypeElem, "intent", "out");
        }
        if (type.isFintentINOUT()) {
            addAttributes(basicTypeElem, "intent", "inout");
        }
    }

    @Override
    protected Element transType(Xtype type) {
        Element typeElem = null;

        if (type.copied != null) {
            typeElem = createElement("FbasicType", "ref",
                                     (type.isFclass() || type.isFprocedure() && type.isFpointer())
                                      && type.isBasic() && (type.getBasicType() == BasicType.VOID) ?
                                       null : type.copied.getXcodeFId(),
                                     "is_procedure", toBoolStr(type.isFprocedure()),
                                     "pass", type.getPass(),
                                     "pass_arg_name", type.getPassArgName());
            setBasicTypeFlags(typeElem, type);
            XobjList typeParams = type.getFTypeParamValues();
            if (typeParams != null) {
              Element e_typeparams = trans(typeParams);
              typeElem = addChildNodes(typeElem, e_typeparams);
            }
        } else {
            switch (type.getKind()) {
            case Xtype.BASIC:
                typeElem = createElement("FbasicType");
                addAttributes(typeElem,
                              "ref", (type.isFclass() || type.isFprocedure() && type.isFpointer())
                                      && type.isBasic() && (type.getBasicType() == BasicType.VOID) ?
                                       null : BasicType.getTypeInfo(type.getBasicType()).fname,
                              "is_procedure", toBoolStr(type.isFprocedure()));
                addChildNodes(typeElem,
                              transKind(type.getFkind()),
                              transLen(type));
                setBasicTypeFlags(typeElem, type);
                break;

            case Xtype.F_ARRAY:
                typeElem = createElement("FbasicType");
                addAttributes(typeElem,
                              "ref", type.getRef().getXcodeFId(),
                              "bind", type.getBind(),
                              "bind_name", type.getBindName());
                for (Xobject sizeExpr : type.getFarraySizeExpr()) {
                    addChildNode(typeElem, trans(sizeExpr));
                }
                setBasicTypeFlags(typeElem, type);
                break;

            case Xtype.STRUCT:
                typeElem = createElement(
                    "FstructType",
                    "is_public", toBoolStr(type.isFpublic()),
                    "is_private", toBoolStr(type.isFprivate()),
                    "is_sequence", toBoolStr(type.isFsequence()),
                    "is_internal_private", toBoolStr(type.isFinternalPrivate()),
                    "extends", ((CompositeType)type).parentId(),
                    "bind", type.getBind());
                addChildNode(typeElem, trans(((StructType)type).getFTypeParams()));
                addChildNode(typeElem, transSymbols(type.getMemberList()));
                addChildNode(typeElem, transProcs(type.getProcList()));
                break;

            case Xtype.FUNCTION:
                typeElem = createElement(
                    "FfunctionType",
                    "return_type", type.getRef().getXcodeFId(),
                    "result_name", type.getFuncResultName(),
                    "is_recursive", toBoolStr(type.isFrecursive()),
                    "is_program", toBoolStr(type.isFprogram()),
                    "is_internal", toBoolStr(type.isFinternal()),
                    "is_intrinsic", toBoolStr(type.isFintrinsic()),
                    "is_external", toBoolStr(type.isFexternal()),
                    "is_public", toBoolStr(type.isFpublic()),
                    "is_private", toBoolStr(type.isFprivate()),
                    "bind", type.getBind(),
                    "bind_name", type.getBindName(),
                    "is_module", toBoolStr(type.isFmodule()));
                addChildNode(typeElem,
                             transParams((XobjList)type.getFuncParam()));
                break;

            default:
                fatal("cannot convert type_kind:" + Xtype.getKindName(type.getKind()));
            }
        }

        addAttributes(typeElem,
                      "type", type.getXcodeFId());

        /*
         *  add <coShape> block if it has codimensions (ID=060)
         */
        if (type.isCoarray()) {
          Element typeElem1 = createElement("coShape");
          addChildNode(typeElem, typeElem1);

          for (Xobject codimension : type.getCodimensions())
            addChildNode(typeElem1, trans(codimension));
        }

        return typeElem;
    }

    private Node transOrError(Xobject xobj) {
        if (xobj == null) {
            throw new NullPointerException("xobj");
        }
        Node n = trans(xobj);
        if (n == null)
            throw new NullPointerException("node : " + xobj.toString());
        return n;
    }

    private String getArg0Name(Xobject x) {
        Xobject a = x.getArgOrNull(0);
        if (a == null)
            return null;
        return a.getName();
    }

    private String getArgString(Xobject x, int idx) {
        Xobject a = x.getArgOrNull(idx);
        if (a == null)
            return null;
        return a.getString();
    }

    private Node transDef(XobjectDef def) {
        if (def == null) {
            fatal("def is null");
        }
        Xobject defObj = def.getDef();
        if (defObj == null)
            return null;
        Node defNode = transOrError(defObj);

        if (def.hasChildren()) {
            Element e = createElement("FcontainsStatement");
            for (XobjectDef childDef : def.getChildren()) {
                addChildNode(e, transDef(childDef));
            }

            String defNodeName = defNode.getNodeName();
            if ("FfunctionDefinition".equals(defNodeName)) {
                Node bodyNode = XmDomUtil.getElement(defNode, "body");
                addChildNode((Element)bodyNode, e);
            } else if ("FmoduleDefinition".equals(defNodeName)) {
                Node containsStmtNode = XmDomUtil.getElement(defNode, "FcontainsStatement");
                if (containsStmtNode != null) {
                    defNode.removeChild(containsStmtNode);
                }
                addChildNode((Element)defNode, e);
            } else {
                fatal("Invalid def: " + defNodeName);
            }
        }

        return defNode;
    }

    @Override
    protected Element transIdent(Ident ident) {
        if (ident == null) {
            return null;
        }
        Element e = createElement("id");

        // type
        if (ident.Type() != null) {
            addAttributes(e, "type", ident.Type().getXcodeFId());
        }

        // name
        Element nameElem = transName(ident);
        nameElem.removeAttribute("type");
        addChildNode(e, nameElem);

        // sclass
        if (ident.getStorageClass() != null) {
            addAttributes(e,
                          "sclass", ident.getStorageClass().toXcodeString());
        }

        // value
        Xobject val = ident.getValue();
        if (val != null && val.Opcode() == Xcode.F_VALUE) {
            Element ve = trans(val);
            addChildNode(e, ve);
        }

        // module declared in
        String mod_name = ident.getFdeclaredModule();
        if (mod_name != null) {
            addAttributes(e, "declared_in", mod_name);
        }

        return e;
    }

    private void addToBody(Element bodyElem, Xobject xobj) {
        if (xobj == null)
            return;
        switch (xobj.Opcode()) {
        case F_STATEMENT_LIST:
        case LIST:
            for (Xobject a : (XobjList)xobj) {
                addToBody(bodyElem, a);
            }
            break;
        case COMPOUND_STATEMENT:
            throw new IllegalArgumentException();
        default:
            addChildNode(bodyElem, trans(xobj));
            break;
        }
    }

    private Node transBody(Xobject xobj) {
        if (xobj == null)
            return null;
        Element e = createElement("body");
        addToBody(e, xobj);
        return e;
    }

    private Node transCondition(Xobject xobj) {
        return addChildNode(createElement("condition"),
                            transExpr(xobj));
    }

    private Node transThen(Xobject xobj) {
        if (xobj == null) {
            return null;
        }
        return addChildNode(createElement("then"),
                            transBody(xobj));
    }

    private Node transElse(Xobject xobj) {
        if (xobj == null) {
            return null;
        }
        return addChildNode(createElement("else"),
                            transBody(xobj));
    }

    private String getConstContent(Xobject xobj) {
        switch (xobj.Opcode()) {
        case INT_CONSTANT:
            return Integer.toString(xobj.getInt());
        case LONGLONG_CONSTANT:
            return ((XobjLong)xobj).getBigInteger().toString();
        case FLOAT_CONSTANT:
            return xobj.getFloatString();
        case F_CHARACTER_CONSTATNT:
        case STRING_CONSTANT:
            return xobj.getString();
        case F_LOGICAL_CONSTATNT:
            return ((XobjBool)xobj).getBoolValue() ? ".TRUE." : ".FALSE.";
        }
        throw new UnsupportedOperationException("not constant : " + xobj.OpcodeName());
    }

    private XobjContainer addDeclForNotDeclared(XobjContainer declList,
                                                XobjList identList,
                                                Xobject body) {
        if (identList == null)
            return declList;

        // collect identifiers which are set to 'delayed decl'
        if (body != null && body instanceof XobjList) {
            XobjectIterator i = new topdownXobjectIterator(body, true);
            for (i.init(); !i.end(); i.next()) {
                Xobject a = i.getXobject();
                if (a == null || !a.isDelayedDecl())
                    continue;
                String name = a.getName();
                if (identList.hasIdent(name))
                    continue;

                Ident id = Ident.Fident(name,
                                        a.Type(),
                                        a.isToBeFcommon(),
                                        true,
                                        null);
                identList.add(id);
            }
        }

        XobjList addDeclList = Xcons.List();

        // add declaration
        for (Xobject a : identList) {
            Ident id = (Ident)a;
            if (id.isDeclared())
                continue;

            String name = id.getName();
            if (declList instanceof XobjList) {
                boolean exists = false;
                for (Xobject decl : (XobjList)declList) {
                    if (decl != null && decl.Opcode() == Xcode.VAR_DECL &&
                        decl.getArg(0).getName().equals(name)) {
                        exists = true;
                        break;
                    }
                }
                if (exists)
                    continue;
            }
            if (id.getStorageClass().isVarOrFunc()) {
                addDeclList.add(Xcons.List(Xcode.VAR_DECL,
                                           Xcons.Symbol(Xcode.IDENT,
                                                        id.Type(),
                                                        name),
                                           id.getFparamValue()));
            }

            if (id.isToBeFcommon()) {
                Xobject cmnDecl =
                    Xcons.List(Xcode.F_COMMON_DECL,
                               Xcons.List(Xcode.F_VAR_LIST,
                                          Xcons.Symbol(Xcode.IDENT,
                                                       id.Type(),
                                                       name),
                                          Xcons.List(Xcode.LIST,
                                                     Xcons.FvarRef(id))));
                addDeclList.add(cmnDecl);
            }
        }

        if (!addDeclList.isEmpty()) {
            if (declList == null)
                declList = addDeclList;
            else {
                addDeclList.reverse();
                for (Xobject a : addDeclList) {
                    declList.insert(a);
                }
            }
        }

        if (declList instanceof XobjList) {
            // redorder var decls by dependency.
            new DeclSorter(identList).sort((XobjList)declList);
        }

        return declList;
    }

    private Node transKind(Xobject xobj) {
        if (xobj == null) {
            return null;
        }
        return addChildNode(createElement("kind"),
                            transExpr(xobj));
    }

    private Node transLen(Xtype type) {
        Xobject xobj = type.getFlen();
        if (xobj == null) {
            return null;
        }
        Element e = createElement("len");
        if (type.isFlenAssumedShape()) {
            addAttributes(e, "is_assumed_shape", "true");
        } else if (type.isFlenAssumedSize()) {
            addAttributes(e, "is_assumed_size", "true");
        } else {
            addChildNode(e, transExpr(xobj));
        }
        return e;
    }

    private Node transParams(XobjList paramList) {
        if (paramList == null || paramList.isEmpty()) {
            return null;
        }
        Element e = createElement("params");
        for (Xobject param : paramList) {
            addChildNode(e, transName(param));
        }
        return e;
    }

    private Element transName(Xobject xobj) {
        if (xobj == null)
            return null;
        Element e = createElement("name");
        if (xobj.Type() != null) {
            addAttributes(e, "type", xobj.Type().getXcodeFId());
        }
        addChildNode(e, trans(xobj.getName()));
        return e;
    }

    private Node transSymbols(Xobject xobj) {
        XobjList identList = (XobjList)xobj;
        Element e = createElement("symbols");
        if (identList != null) {
            for (Xobject ident : identList) {
                addChildNode(e, transIdent((Ident)ident));
            }
        }
        return e;
    }

    private Node transProcs(Xobject xobj) {
        XobjList procList = (XobjList)xobj;
        Element e = null;
        if (procList != null) {
            e = createElement("typeBoundProcedures");
            for (Xobject proc : procList) {
                addChildNode(e, transProc(proc));
            }
        }
        return e;
    }

    private Node transProc(Xobject xobj) {
        Element e = null;
        if (xobj.Opcode() == Xcode.F_TYPE_BOUND_PROCEDURE) {
          e = createElement("typeBoundProcedure");
          addAttributes(e, "type", xobj.Type().getXcodeFId());
          addAttributes(e, "pass"         , getArgString(xobj, 0));
          addAttributes(e, "pass_arg_name", getArgString(xobj, 1));
          addChildNode(e, transName(xobj.getArg(2)));
          int tq = ((XobjInt)xobj.getArg(3)).getInt();
          if ((tq & Xtype.TQ_FPRIVATE) != 0)
            addAttributes(e, "is_private", "true");
          else if ((tq & Xtype.TQ_FPUBLIC) != 0)
            addAttributes(e, "is_public", "true");
          addChildNode(e, addChildNode(createElement("binding"), transName(xobj.getArg(4))));
          addAttributes(e, "is_non_overridable", intFlagToBoolStr(xobj.getArgOrNull(5)));
        } else if (xobj.Opcode() == Xcode.F_TYPE_BOUND_GENERIC_PROCEDURE) {
          e = createElement("typeBoundGenericProcedure");
          addAttributes(e, "is_operator"  , intFlagToBoolStr(xobj.getArgOrNull(0)));
          addAttributes(e, "is_assignment", intFlagToBoolStr(xobj.getArgOrNull(1)));
          addChildNode(e, transName(xobj.getArg(2)));
          int tq = ((XobjInt)xobj.getArg(3)).getInt();
          if ((tq & Xtype.TQ_FPRIVATE) != 0)
            addAttributes(e, "is_private", "true");
          else if ((tq & Xtype.TQ_FPUBLIC) != 0)
            addAttributes(e, "is_public", "true");
          Element bdgn = createElement("binding");
          if (xobj.getArg(4) != null) {
              for (Xobject a : (XobjList)xobj.getArg(4)) {
                  if (a == null)
                      continue;
                  Node n = transName(a);
                  if (n == null)
                      continue;
                  addChildNode(bdgn, n);
              }
          }
          addChildNode(e, bdgn);
        } else {
           fatal("unknown type bound procedure type.");
        }
        return e;
    }

    private Node transDeclarations(Xobject xobj) {
        Element e = createElement("declarations");
        if (xobj != null) {
            for (Xobject a : (XobjList)xobj) {
                if (a == null)
                    continue;
                Node n = trans(a);
                if (n == null)
                    continue;
                addChildNode(e, n);
            }
        }
        return e;
    }

    private Node transExpr(Xobject xobj) {
        return trans(xobj);
    }

    private Element transValue(Xobject xobj) {
        if (xobj == null) {
            return null;
        }
        Element e = addChildNode(createElement("value"),
                                 transExpr(xobj.getArg(0)));
        Node rcExpr = transExpr(xobj.getArgOrNull(1));
        if (rcExpr != null) {
            Element rcElem = addChildNode(createElement("repeat_count"),
                                          rcExpr);
            addChildNode(e, rcElem);
        }
        return e;
    }

    private static final String PROP_KEY_IDSET = "DeclComparator.idSet";
    private static final String PROP_KEY_DEPSET = "DeclComparator.depSet";

    private Element transAccClauseArg(Xobject xobj){
        if(xobj == null) return null;

        if(xobj.Opcode() == Xcode.LIST && xobj.getArg(0).Opcode() == Xcode.STRING){
            return transAccClause(xobj);
        }

        return trans(xobj);
    }

    private Element transAccClause(Xobject xobj) {
        Element e = createElement("list");

        //clause name
        String clauseName = xobj.getArg(0).getString();
        Element f0 = createElement("string");
        addChildNode(f0, trans(clauseName));
        addChildNode(e, f0);

        //clause arg
        Xobject arg = xobj.getArgOrNull(1);
        if(arg == null){
            return e;
        }

        if(arg.Opcode() == Xcode.LIST && arg.getArg(0).Opcode() != Xcode.STRING){
            //normal list
            Element f1 = createElement("list");
            for(Xobject a : (XobjList)arg){
                addChildNode(f1, transAccClauseArg(a));
            }
            addChildNode(e, f1);
        }else{
            //accClause or normal element
            addChildNode(e, transAccClauseArg(arg));
        }

        return e;
    }

    /**
     * sort declarations by dependency order.
     * Copied from XmfXobjectToXmObjTranslator.java
     */
    static class DeclSorter {
        private XobjList ident_list;

        public DeclSorter(XobjList ident_list) {
            this.ident_list = ident_list;
        }

        public void sort(XobjList declList) {
            Map<String, Xobject> declMap = new LinkedHashMap<String, Xobject>();
            List<Xobject> headDeclList = new ArrayList<Xobject>();
            List<Xobject> tailDeclList = new ArrayList<Xobject>();

            for (Xobject decl : declList) {
                collectDependName(decl);
                switch (decl.Opcode()) {
                case VAR_DECL:
                    declMap.put(decl.getArg(0).getName(), decl);
                    break;
                case F_STRUCT_DECL:
                    declMap.put("$" + decl.getArg(0).getName(), decl);
                    break;
                case F_USE_DECL:
                case F_USE_ONLY_DECL:
                    headDeclList.add(decl);
                    break;
                default:
                    tailDeclList.add(decl);
                    break;
                }
            }

            for (Xobject decl : declMap.values()) {
                Set<String> idSet = getIdSet(decl);
                if (idSet == null)
                    throw new IllegalStateException(decl.toString());
                decl.remProp(PROP_KEY_IDSET);
                Set<Xobject> depSet = getDepSet(decl);
                if (depSet == null) {
                    depSet = new HashSet<Xobject>();
                    decl.setProp(PROP_KEY_DEPSET, depSet);
                }
                for (String n : idSet) {
                    Xobject depDecl = declMap.get(n);
                    if (depDecl == null)
                        continue;
                    depSet.add(depDecl);
                }
            }

            declList.clear();

            while (true) {
                int declMapSize = declMap.size();

                for (Iterator<Xobject> ite = declMap.values().iterator(); ite.hasNext(); ) {
                    Xobject decl = ite.next();
                    Set<Xobject> depSet = getDepSet(decl);
                    if (depSet.isEmpty()) {
                        ite.remove();
                        declList.add(decl);
                        decl.remProp(PROP_KEY_DEPSET);
                        for (Xobject d : declMap.values()) {
                            getDepSet(d).remove(decl);
                        }
                    }
                }

                if (declMapSize == declMap.size()) {
                    for (Xobject d : declMap.values())
                        declList.add(d);
                    break;
                }

                if (declMap.isEmpty())
                    break;
            }

            ListIterator<Xobject> ite = headDeclList.listIterator();
            for (; ite.hasNext(); ite.next());
            for (; ite.hasPrevious(); ) {
                Xobject x = ite.previous();
                declList.insert(x);
            }

            for (Xobject x : tailDeclList)
                declList.add(x);
        }

        @SuppressWarnings("unchecked")
        private Set<String> getIdSet(Xobject x) {
            return (Set<String>)x.getProp(PROP_KEY_IDSET);
        }

        @SuppressWarnings("unchecked")
        private Set<Xobject> getDepSet(Xobject x) {
            return (Set<Xobject>)x.getProp(PROP_KEY_DEPSET);
        }

        private void collectDependName(Xobject decl) {
            switch (decl.Opcode()) {
            case VAR_DECL:
            case F_STRUCT_DECL:
                break;
            default:
                return;
            }

            Set<String> idSet = getIdSet(decl);
            if (idSet != null)
                return;
            idSet = new HashSet<String>();
            decl.setProp(PROP_KEY_IDSET, idSet);

            Ident id = ident_list.find(decl.getArg(0).getName(),
                (decl.Opcode() == Xcode.VAR_DECL) ? IXobject.FINDKIND_VAR : IXobject.FINDKIND_TAGNAME);
            Xtype t = (id != null) ? id.Type() : null;
            _collectDependName(t, idSet, null);

            if (decl.Opcode() == Xcode.VAR_DECL) {
                _collectDependName(decl.getArgOrNull(1), idSet);
            } else {
                //remove dependency to self
                idSet.remove("$" + id.getName());
            }
        }

        private void _collectDependName(Xobject x, Set<String> idSet) {
            if (x == null)
                return;

            if (x.isVarRef()) {
                idSet.add(x.getName());
            } else if (x instanceof XobjList) {
                for (Xobject a : (XobjList)x)
                    _collectDependName(a, idSet);
            }
        }

        private void _collectDependName(Xtype t, Set<String> idSet, Set<Xtype> checkedSet) {
            if (t == null)
                return;

            if (checkedSet == null)
                checkedSet = new HashSet<Xtype>();
            else if (checkedSet.contains(t))
                return;
            else
                checkedSet.add(t);

            switch (t.getKind()) {
            case Xtype.BASIC:
                if (t.copied != null) {
                    _collectDependName(t.copied, idSet, checkedSet);
                } else {
                    _collectDependName(t.getFkind(), idSet);
                    _collectDependName(t.getFlen(), idSet);
                }
                break;
            case Xtype.F_ARRAY:
                _collectDependName(t.getRef(), idSet, checkedSet);
                for (Xobject s : t.getFarraySizeExpr())
                    _collectDependName(s, idSet);
                break;
            case Xtype.STRUCT: {
                    Ident typeName = ident_list.getStructTypeName(t);
                    if (typeName != null)
                        idSet.add("$" + typeName.getName());
                    for (Xobject a : t.getMemberList()) {

			if (a.Type().equals(t)) continue;
                        _collectDependName(a.Type(), idSet, checkedSet);
                        _collectDependName(((Ident)a).getValue(), idSet);
                    }
                }
		break;
	    case Xtype.FUNCTION:
                _collectDependName(t.getRef(), idSet, checkedSet);
                break;
            }
        }
    }
}
