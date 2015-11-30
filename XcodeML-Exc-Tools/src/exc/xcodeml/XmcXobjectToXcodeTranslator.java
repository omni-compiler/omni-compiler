package exc.xcodeml;

import static xcodeml.util.XmLog.fatal;
import static xcodeml.util.XmLog.fatal_dump;

import org.w3c.dom.Element;
import org.w3c.dom.Node;

import xcodeml.ILineNo;
import exc.object.BasicType;
import exc.object.Ident;
import exc.object.StorageClass;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XobjList;
import exc.object.XobjString;
import exc.object.Xobject;
import exc.object.XobjectDef;
import exc.object.XobjectDefEnv;
import exc.object.Xtype;

public class XmcXobjectToXcodeTranslator extends XmXobjectToXcodeTranslator {
    private XcodeMLNameTable_C nameTable = new XcodeMLNameTable_C();

    private Element transName(Xobject xobj) {
        if (xobj == null) {
            return null;
        }

        Element e = createElement("name");
        if (xobj.Type() != null) {
            addAttributes(e,
                          "type", xobj.Type().getXcodeCId());
        }
        addChildNodes(e, trans(xobj.getString()));

        return e;
    }

    private Element transBody(Xobject xobj) {
        Element e = createElement("body");
        if (xobj != null) {
            if (xobj.Opcode() == Xcode.LIST) {
                for (Xobject a : (XobjList)xobj) {
                    addChildNodes(e, trans(a));
                }
            } else {
                addChildNodes(e, trans(xobj));
            }
        }
        return e;
    }

    private Element transSymbols(Xobject xobj) {
        XobjList identList = (XobjList)xobj;
        Element e = createElement("symbols");
        if (identList != null) {
            for (Xobject ident : identList) {
                addChildNodes(e, transIdent((Ident)ident));
            }
        }
        return e;
    }

    @Override
    protected Element transIdent(Ident ident) {
        if (ident == null) {
            return null;
        }

        Element e = createElement("id");

        // type
        if (ident.Type() != null) {
            addAttributes(e, 
                          "type", ident.Type().getXcodeCId());
        }

        // name
        addChildNodes(e,
                      addChildNodes(createElement("name"),
                                    trans(ident.getName())));

        // sclass
        if (ident.getStorageClass() != null) {
            addAttributes(e,
                          "sclass", ident.getStorageClass().toXcodeString());
        }

        // bit field
        if (ident.getBitField() > 0) {
            addAttributes(e,
                          "bit_field", "" + ident.getBitField());
        } else if (ident.getBitFieldExpr() != null) {
            addAttributes(e,
                          "bit_field", "*");
            addChildNodes(e,
                          addChildNodes(createElement("bitField"),
                                        trans(ident.getBitFieldExpr())));
        }

        // gccThread
        if (ident.isGccThread()) {
            addAttributes(e,
                          "is_gccThread", "1");
        }

        // gccExtension
        if (ident.isGccExtension()) {
            addAttributes(e,
                          "is_gccExtension", "1");
        }

        // enum member value
        addChildNodes(e,
                      transValue(ident.getEnumValue()));

        // codimensions for coarray (ID=284)
        addChildNodes(e,
                      transCodimensions(ident.getCodimensions()));

        // gcc attributes
        addChildNodes(e,
                      trans(ident.getGccAttributes()));

        return e;
    }

    private Node transValue(Xobject xobj) {
        if (xobj == null) {
            return null;
        }

        return addChildNodes(createElement("value"),
                             transValueChild(xobj));
    }

    private Node transValueChild(Xobject xobj) {
        if (xobj.Opcode() == Xcode.LIST) {
            // The name of "ompoundValue" node is "value".
            Element e = createElement("value");
            for (Xobject a : (XobjList)xobj) {
                Node cn = transValueChild(a);
                if (cn != null) {
                    addChildNodes(e, cn);
                }
            }
            return e;
        } else {
            return trans(xobj);
        }
    }

    // ID=284
    private Node transCodimensions(Xobject xobj) {
        if (xobj == null) {
            return null;
        }
        Element codims = createElement("codimensions");
        for (Xobject a : (XobjList)xobj) {
            if (a.Opcode() == Xcode.LIST) {
                addChildNodes(codims, createElement("list"));
            } else {
                addChildNodes(codims, transExprOrError(a));
            }
        }
        return codims;
    }

    private Element transFuncDefParams(XobjList declList, XobjList identList) {
        Element eParams = createElement("params");

        if (declList != null) {
            for (Xobject a: declList) {
                if (a == null) {
                    addChildNodes(eParams,
                                  createElement("ellipsis"));
                    break;
                }
                if (a.Opcode() != Xcode.VAR_DECL) {
                    fatal("not VAR_DECL : " + a.toString());
                }
                String paramName = a.getArg(0).getName();
                Xtype type = null;
                for (Xobject i : identList) {
                    if (i.getName().equals(paramName)) {
                        type = i.Type();
                        break;
                    }
                }
                Element eName =
                    addChildNodes(createElement("name",
                                                "type",
                                                type.getXcodeCId()),
                                  trans(paramName));
                addChildNodes(eParams,
                              eName);
            }
        }

        if (identList != null) {
            for (Xobject a : identList) {
                Ident id = (Ident)a;
                if (id.isDeclared() ||
                    id.getStorageClass() != StorageClass.PARAM) {
                    continue;
                }
                Element eName =
                    addChildNodes(createElement("name",
                                                "type",
                                                id.Type().getXcodeCId()),
                                  trans(id.getName()));
                addChildNodes(eParams,
                              eName);
            }
        }

        return eParams;
    }

    private Element transDeclarations(Xobject xobj) {
        Element e = createElement("declarations");
        if (xobj != null) {
            for (Xobject a : (XobjList)xobj) {
                addChildNodes(e, transOrError(a));
            }
        }
        return e;
    }

    private Element transExpr(Xobject expr) {
        return trans(expr);
    }

    private Node transCondition(Xobject xobj) {
        if (xobj == null) {
            return null;
        }
        return addChildNodes(createElement("condition"),
                             transExpr(xobj));
    }

    private Element transOrError(Xobject xobj) {

	// null list should be treated as null
        if (xobj.Opcode() == Xcode.LIST && xobj.Nargs() == 0){
	    return null;
	}
        else if (xobj == null) {
            throw new NullPointerException("xobj");
        }
        Element e = trans(xobj);
        if (e == null) {
            throw new NullPointerException("xobj : " + xobj.toString());
        }
        return e;
    }

    private Element transExprOrError(Xobject expr) {
        return transOrError(expr);
    }

    @Override
    protected Element transType(Xtype type) {
        Element e = null;
        if (type.copied != null) {
            e = createTypeElement("basicType", type,
                                  "name", type.copied.getXcodeCId());
        } else {
            switch (type.getKind()) {
            case Xtype.BASIC:
                e = createTypeElement(
                    "basicType", type,
                    "name", BasicType.getTypeInfo(type.getBasicType()).cname);
                break;
            case Xtype.ENUM:
                e = addChildNodes(createTypeElement("enumType", type),
                                  transSymbols(type.getMemberList()));
                break;
            case Xtype.STRUCT:
                e = addChildNodes(createTypeElement("structType", type),
                                  transSymbols(type.getMemberList()));
                break;
            case Xtype.UNION:
                e = addChildNodes(createTypeElement("unionType", type),
                                  transSymbols(type.getMemberList()));
                break;
            case Xtype.ARRAY:
                e = createTypeElement(
                    "arrayType", type,
                    "element_type", type.getRef().getXcodeCId());
                if (type.getArraySize() >= 0) {
                    addAttributes(e,
                                  "array_size", "" + type.getArraySize());
                } else if (type.getArraySizeExpr() != null) {
                    addAttributes(e,
                                  "array_size", "*");
                    addChildNodes(e,
                                  transArraySize(type.getArraySizeExpr()));
                }
                addAttributes(e,
                              "is_static", toBoolStr(type.isArrayStatic()));
                break;
            case Xtype.FUNCTION:
                e = createTypeElement(
                    "functionType", type,
                    "return_type", type.getRef().getXcodeCId(),
                    "is_static", toBoolStr(type.isFuncStatic()),
                    "is_inline", toBoolStr(type.isInline()));
                addChildNodes(e,
                              transFuncTypeParams((XobjList)type.getFuncParam()));
                break;
            case Xtype.POINTER:
                e = createTypeElement("pointerType", type,
                                      "ref", type.getRef().getXcodeCId());
                break;
            case Xtype.XMP_CO_ARRAY:
                e = createTypeElement(
                    "coArrayType", type,
                    "element_type", type.getRef().getXcodeCId());
                if (type.getArraySize() >= 0) {
                    addAttributes(e,
                                  "array_size", "" + type.getArraySize());
                } else if (type.getArraySizeExpr() != null) {
                    addAttributes(e,
                                  "array_size", "*");
                    addChildNodes(e,
                                  transArraySize(type.getArraySizeExpr()));
                }
                break;
            default:
                fatal("cannot convert type_kind:" + type.getKind());
            }
        }

        addAttributes(e,
                      "type", type.getXcodeCId(),
                      "is_const", toBoolStr(type.isConst()),
                      "is_restrict", toBoolStr(type.isRestrict()),
                      "is_volatile", toBoolStr(type.isVolatile()));
        return e;
    }

    private Node transArraySize(Xobject expr) {
        if (expr == null) {
            return null;
        }
        return addChildNodes(createElement("arraySize"),
                             transExpr(expr));
    }

    private Node transFuncTypeParams(XobjList paramList) {
        Element e = createElement("params");
        if (paramList != null) {
            if (paramList.isEmpty()) {
                Element name =
                    createElement("name",
                                  "type",
                                  BasicType.getTypeInfo(BasicType.VOID).cname);
                addChildNodes(e, name);
            } else {
                for (Xobject param : paramList) {
                    if (param == null) {
                        addChildNodes(e,
                                      createElement("ellipsis"));
                    } else {
                        addChildNodes(e,
                                      transName(param));
                    }
                }
            }
        }
        return e;
    }

    private Element transSizeOrAlignOf(Xobject xobj) {
        final String name = nameTable.getName(xobj.Opcode());
        return addChildNodes(createElement(name),
                             transOrError(xobj.getArg(0)));
    }

    @Override
    void transGlobalDeclarations(Element globalDecl, XobjectDefEnv defList) {
        XobjList declList = getDeclForNotDeclared((XobjList)defList.getGlobalIdentList());
        if (declList != null) {
            declList.reverse();
            for (Xobject a : declList) {
                defList.insert(a);
            }
        }

        for (XobjectDef def : defList) {
            if (def == null) {
                fatal("def is null");
            }
            Xobject defObj = def.getDef();
            Node n = null;
            if (defObj == null) {
                n = null;
            } else {
                n = transOrError(defObj);
            }
            addChildNodes(globalDecl, n);
        }
    }

    @Override
    public Element trans(Xobject xobj) {
        if (xobj == null) {
            return null;
        }

	if (xobj instanceof Ident){
	  Element e = addChildNodes(createElement("Var"),
				    trans(xobj.getName()));
	  return e;
	}

        final String name = nameTable.getName(xobj.Opcode());
        Element e = null;

        switch (xobj.Opcode()) {
        case LIST: {
            switch(xobj.Nargs()) {
            case 0:
                return null;
            case 1:
                return trans(xobj.getArg(0));
            default: {
                // compound statement.
                e = addChildNodes(createElement("compoundStatement"),
                                  transSymbols(null),
                                  transDeclarations(null));
                Element eBody = transBody(null);
                for (Xobject a : (XobjList)xobj) {
                    addChildNodes(eBody, trans(a));
                }
                addChildNodes(e, eBody);
                break;
            }
            }
        }
            break;

        // constant
        case INT_CONSTANT:
            e = addChildNodes(createElement(name),
                              trans(Integer.toString(xobj.getInt())));
            break;
        case FLOAT_CONSTANT:
            e = addChildNodes(createElement(name),
                              trans(xobj.getFloatString()));
            break;
        case LONGLONG_CONSTANT:
            e = addChildNodes(createElement(name),
                              trans("0x" + Integer.toHexString((int)xobj.getLongHigh()) + " " +
                                    "0x" + Integer.toHexString((int)xobj.getLongLow())));
            break;
        case STRING_CONSTANT:
        case STRING:
            e = addChildNodes(createElement(name),
                              trans(xobj.getString()));
            break;

        case MOE_CONSTANT:
            e = addChildNodes(createElement(name),
                              trans(xobj.getString()));
            break;


        // definition and declaration

        case FUNCTION_DEFINITION: {
            // (CODE name symbols params body gccAttrs)
            XobjList identList = (XobjList)xobj.getArg(1);
            XobjList paramList = (XobjList)xobj.getArg(2);
            XobjList bodyList = (XobjList)xobj.getArgOrNull(3);
            Xobject gccAttrs = xobj.getArgOrNull(4);

            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)),
                              trans(gccAttrs),
                              transSymbols(identList),
                              transFuncDefParams(paramList, identList),
                              transBody(bodyList));
        }
            break;
        case VAR_DECL:
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)),
                              transValue(xobj.getArgOrNull(1)),
                              trans(xobj.getArgOrNull(2)));
            break;
        case FUNCTION_DECL:
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)),
                              trans(xobj.getArgOrNull(2)));
            break;

        // statements

        case COMPOUND_STATEMENT: {
            XobjList identList = (XobjList)xobj.getArg(0);
            XobjList declList = (XobjList)xobj.getArg(1);
            XobjList addDeclList = getDeclForNotDeclared(identList);

            if (addDeclList != null) {
                if (declList == null) {
                    declList = Xcons.List();
                }
                addDeclList.reverse();
                for (Xobject a : addDeclList) {
                    declList.insert(a);
                }
            }

            e = addChildNodes(createElement(name),
                              transSymbols(identList),
                              transDeclarations(declList),
                              transBody(xobj.getArg(2)));
        }
            break;
        case EXPR_STATEMENT:
            if (xobj.getArg(0).Opcode() == Xcode.GCC_ASM_STATEMENT) {
                return trans(xobj.getArg(0));
            } else {
                e = addChildNodes(createElement(name),
                                  transExpr(xobj.getArg(0)));
            }
            break;
        case IF_STATEMENT: {
            Node eCond = transCondition(xobj.getArg(0));
            Element eThen = addChildNodes(createElement("then"),
                                          trans(xobj.getArg(1)));

            Element eElse = createElement("else");
            Xobject xElse = xobj.getArgOrNull(2);
            if (xElse != null) {
                addChildNodes(eElse,
                              trans(xobj.getArg(2)));
            }
            e = addChildNodes(createElement(name),
                              eCond,
                              eThen,
                              eElse);
        }
            break;
        case WHILE_STATEMENT:
            e = addChildNodes(createElement(name),
                              transCondition(xobj.getArg(0)),
                              transBody(xobj.getArg(1)));
            break;
        case DO_STATEMENT:
            e = addChildNodes(createElement(name),
                              transBody(xobj.getArg(0)),
                              transCondition(xobj.getArg(1)));
            break;
        case FOR_STATEMENT:
            e = addChildNodes(createElement(name),
                              addChildNodes(createElement("init"),
                                            transExpr(xobj.getArg(0))),
                              transCondition(xobj.getArg(1)),
                              addChildNodes(createElement("iter"),
                                            transExpr(xobj.getArg(2))),
                              transBody(xobj.getArg(3)));
            break;
        case SWITCH_STATEMENT:
            e = addChildNodes(createElement(name),
                              transValue(xobj.getArg(0)),
                              transBody(xobj.getArg(1)));
            break;
        case BREAK_STATEMENT:
            e = addChildNodes(createElement(name));
            break;
        case CONTINUE_STATEMENT:
            e = addChildNodes(createElement(name));
            break;
        case RETURN_STATEMENT:{
              Xobject ret_exp = xobj.getArgOrNull(0);
              if(ret_exp != null){
                e = addChildNodes(createElement(name), transExpr(ret_exp));
              }else{
                e = addChildNodes(createElement(name));
              }
            //e = addChildNodes(createElement(name),
                              //transExpr(xobj.getArg(0)));
        }
            break;
        case GOTO_STATEMENT: {
            e = createElement(name);
            Xobject x = xobj.getArg(0);
            if (x.Opcode() == Xcode.IDENT) {
                addChildNodes(e, transName(x));
            } else {
                addChildNodes(e, transExpr(x));
            }
        }
            break;
        case STATEMENT_LABEL:
            e = addChildNodes(createElement(name),
                              transName(xobj.getArg(0)));
            break;
        case CASE_LABEL:
            e = addChildNodes(createElement(name),
                              transValue(xobj.getArg(0)));
            break;
        case DEFAULT_LABEL:
            e = addChildNodes(createElement(name));
            break;

        // expression

        case CONDITIONAL_EXPR: {
            Xobject a = xobj.getArg(1);
            e = addChildNodes(createElement(name),
                              transExprOrError(xobj.getArg(0)),
                              transExprOrError(a.getArg(0)),
                              transExprOrError(a.getArg(1)));
        }
            break;
        case COMMA_EXPR:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNodes(e,
                              transExprOrError(a));
            }
            break;
        case DESIGNATED_VALUE: {
            Xobject valueOrExpr = xobj.getArg(1);
            Node n = null;
            if (valueOrExpr.Opcode() == Xcode.LIST) {
                n = transValue(valueOrExpr);
            } else {
                n = transExpr(valueOrExpr);
            }
            e = addChildNodes(createElement(name,
                                            "member",
                                            xobj.getArg(0).getName()),
                              n);
        }
            break;
        case COMPOUND_VALUE:
            e = addChildNodes(createElement(name),
                              transValue(xobj.getArg(0)));
            break;
        case COMPOUND_VALUE_ADDR:
            e = addChildNodes(createElement(name),
                              transValue(xobj.getArg(0)));
            break;
        case ADDR_OF: {
            Element nSymAddr = null;
            Element nMember = null;
            Boolean isArrayRef = false;
            Xobject operand = xobj.operand();
            switch (operand.Opcode()) {
            case VAR:
            case VAR_ADDR:
                nSymAddr = createElement("varAddr");
                break;
            case ARRAY_REF:
              isArrayRef = true;
              break;
            case ARRAY_ADDR: /* illegal but convert */
                nSymAddr = createElement("arrayAddr");
                break;
            case FUNC_ADDR:
                nSymAddr = createElement("funcAddr");
                break;
            case MEMBER_REF:
            case MEMBER_ADDR:
                nMember = createElement("memberAddr");
                break;
            case MEMBER_ARRAY_REF:
            case MEMBER_ARRAY_ADDR:
                nMember = createElement("memberArrayAddr");
                break;
            case POINTER_REF:
                // reduce (ADDR_OF (POINTER_REF expr)) => expr
                return transOrError(operand.getArg(0));
            default:
                fatal("cannot apply ADDR_OF to " + operand.toString());
            }
            
            if (isArrayRef){
              e = addChildNodes(createElement("addrOfExpr"), transOrError(xobj.getArg(0)));
            }else if (nSymAddr != null) {
                addChildNodes(nSymAddr,
                              trans(operand.getName()));
                e = nSymAddr;
            } else {
                addAttributes(nMember,
                              "member", operand.getArg(1).getName());
                addChildNodes(nMember,
                              transExpr(operand.getArg(0)));
                e = nMember;
            }
        }
            break;
        case FUNCTION_CALL: {
            e = createElement(name);
            Element nFunc = createElement("function");
            Element nArgs = createElement("arguments");
            addChildNodes(nFunc,
                          transExprOrError(xobj.getArg(0)));
            XobjList params = (XobjList)xobj.getArg(1);
            if (params != null) {
                for (Xobject a : params) {
                    addChildNodes(nArgs, transExprOrError(a));
                }
            }

            addChildNodes(e,
                          nFunc,
                          nArgs);
        }
            break;
        case SIZE_OF_EXPR:
            e = transSizeOrAlignOf(xobj);
            break;
        case CAST_EXPR:
            e = addChildNodes(createElement(name),
                              transOrError(xobj.getArg(0)));
            break;
        case TYPE_NAME:
            e = createElement(name,
                              "type", xobj.Type().getXcodeCId());
            break;
        case BUILTIN_OP: {
            e = createElement(name,
                              "name", xobj.getArg(0).getName(),
                              "is_id", intFlagToBoolStr(xobj.getArg(1)),
                              "is_addrOf", intFlagToBoolStr(xobj.getArg(2)));

            XobjList args = (XobjList)xobj.getArgOrNull(3);
            if (args != null) {
                for (Xobject a : args) {
                    addChildNodes(e, transOrError(a));
                }
            }
        }
            break;

        // unary expression

        case POINTER_REF:
            // reduce (POINTER_REF (ADDR_OF expr)) => expr
            if (xobj.getArg(0).Opcode() == Xcode.ADDR_OF)
                return transOrError(xobj.getArg(0).getArg(0));
            // no break
        case UNARY_MINUS_EXPR:
        case LOG_NOT_EXPR:
        case BIT_NOT_EXPR:
        case PRE_INCR_EXPR:
        case PRE_DECR_EXPR:
        case POST_INCR_EXPR:
        case POST_DECR_EXPR:
            e = addChildNodes(createElement(name),
                              transOrError(xobj.getArg(0)));
            break;
        
        // binary expression
        
        case ASSIGN_EXPR:
        case PLUS_EXPR:
        case ASG_PLUS_EXPR:
        case MINUS_EXPR:
        case ASG_MINUS_EXPR:
        case MUL_EXPR:
        case ASG_MUL_EXPR:
        case DIV_EXPR:
        case ASG_DIV_EXPR:
        case MOD_EXPR:
        case ASG_MOD_EXPR:
        case LSHIFT_EXPR:
        case ASG_LSHIFT_EXPR:
        case RSHIFT_EXPR:
        case ASG_RSHIFT_EXPR:
        case BIT_AND_EXPR:
        case ASG_BIT_AND_EXPR:
        case BIT_OR_EXPR:
        case ASG_BIT_OR_EXPR:
        case BIT_XOR_EXPR:
        case ASG_BIT_XOR_EXPR:
        case LOG_EQ_EXPR:
        case LOG_NEQ_EXPR:
        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case LOG_AND_EXPR:
        case LOG_OR_EXPR:
            e = addChildNodes(createElement(name),
                              transOrError(xobj.getArg(0)),
                              transOrError(xobj.getArg(1)));
            break;

        // symbol reference

        case VAR:
        case VAR_ADDR:
        case ARRAY_ADDR:
            e = addChildNodes(createElement(name),
                              trans(xobj.getName()));
            if (xobj.getScope() != null) {
                addAttributes(e,
                              "scope", xobj.getScope().toXcodeString());
            }
            break;
        case FUNC_ADDR:
            e = addChildNodes(createElement(name),
                              trans(xobj.getName()));
            break;
        case MEMBER_REF:
        case MEMBER_ADDR:
        case MEMBER_ARRAY_REF:
        case MEMBER_ARRAY_ADDR:
            e = addChildNodes(createElement(name,
                                            "member",
                                            xobj.getArg(1).getString()),
                              transExpr(xobj.getArg(0)));
            break;

        // gcc syntax

        case GCC_ATTRIBUTES:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNodes(e, trans(a));
            }
            break;
        case GCC_ATTRIBUTE:
            e = createElement(name,
                              "name", xobj.getArg(0).getString());
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNodes(e, transExprOrError(a));
            }
            break;
        case GCC_ASM:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)));
            break;
        case GCC_ASM_DEFINITION:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)));
            break;
        case GCC_ASM_STATEMENT:
            // (CODE is_volatile string_constant operand1 operand2 clobbers)
            e = addChildNodes(createElement(name,
                                            "is_volatile",
                                            intFlagToBoolStr(xobj.getArg(0))),
                              transExpr(xobj.getArg(1)),
                              trans(xobj.getArgOrNull(2)),
                              trans(xobj.getArgOrNull(3)),
                              trans(xobj.getArgOrNull(4)));
            break;
        case GCC_ASM_OPERANDS:
            e = createElement(name);
            for (Xobject a : (XobjList)xobj) {
                addChildNodes(e, trans(a));
            }
            break;
        case GCC_ASM_OPERAND:
            e = addChildNodes(createElement(name),
                              transExpr(xobj.getArg(0)));

            if (xobj.getArg(1) != null) {
                addAttributes(e,
                              "match", xobj.getArg(1).getString());
            }
            if (xobj.getArg(2) != null) {
                addAttributes(e,
                              "constraint", xobj.getArg(2).getString());
            }
            break;
        case GCC_ASM_CLOBBERS:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)));
            break;
        case GCC_ALIGN_OF_EXPR:
            e = transSizeOrAlignOf(xobj);
            break;
        case GCC_MEMBER_DESIGNATOR:
            e = addChildNodes(createElement(name,
                                            "member", xobj.getArg(1).getName(),
                                            "ref", xobj.getArg(0).Type().getXcodeCId()),
                              transExpr(xobj.getArg(2)),
                              trans(xobj.getArg(3)));
            break;
        case GCC_LABEL_ADDR:
            e = addChildNodes(createElement(name),
                              trans(xobj.getName()));
            break;
        case GCC_COMPOUND_EXPR:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)));
            break;
        case GCC_RANGED_CASE_LABEL:
            e = addChildNodes(createElement(name),
                              transValue(xobj.getArg(0)),
                              transValue(xobj.getArg(0)));
            break;

        // xmp syntax
        case ARRAY_REF: {
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)));

            XobjList exprList = (XobjList)xobj.getArg(1);
            final int exprListSize = exprList.Nargs();
            for (int i = 0; i < exprListSize; i++) {
                addChildNodes(e, transExpr(exprList.getArg(i)));
            }
        }
            break;

        case SUB_ARRAY_REF: {
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)));

            XobjList exprList = (XobjList)xobj.getArg(1);
            final int exprListSize = exprList.Nargs();
            for (int i = 0; i < exprListSize; i++) {
                addChildNodes(e, trans(exprList.getArg(i)));
            }
        }
            break;
//         case LOWER_BOUND:
//             e = addChildNodes(createElement(name),
//                               transExpr(xobj.getArg(0)));
//             break;
//         case UPPER_BOUND:
//             e = addChildNodes(createElement(name),
//                               transExpr(xobj.getArg(0)));
//             break;
//         case STEP:
//             e = addChildNodes(createElement(name),
//                               transExpr(xobj.getArg(0)));
//             break;
        case CO_ARRAY_REF:
            e = addChildNodes(createElement(name),
                              transExpr(xobj.getArg(0)));
            for (Xobject a : (XobjList)xobj.getArg(1)) {
                addChildNodes(e, transExpr(a));
            }
            if (xobj.getScope() != null) {
                addAttributes(e,
                              "scope", xobj.getScope().toXcodeString());
            }
            break;
        case CO_ARRAY_ASSIGN_EXPR:
            e = addChildNodes(createElement(name),
                              transExpr(xobj.getArg(0)),
                              transExpr(xobj.getArg(1)));
            break;

        // directive
        case PRAGMA_LINE:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0).getString()));
            break;
        case TEXT:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0).getString()));
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

			if (a.Nargs() > 1){

			  Xobject vars = a.getArg(1);
			  if (vars != null){

			    if (vars instanceof XobjString){
			      addChildNode(g, trans(vars));
			    }
			    else {
			      Element g1 = createElement("list");
			      for (Xobject b : (XobjList)vars){
				addChildNode(g1, trans(b));
			      }
			      addChildNode(g, g1);
			    }

			  }

			}

                        addChildNode(f1, g);
                    }
		}
            }
            addChildNode(e, f1);

            Element f2 = createElement("list");
            Xobject body = xobj.getArg(2);
            if (body != null){
                if (body.Opcode() == Xcode.F_STATEMENT_LIST){
                    for (Xobject a : (XobjList)body){
                        if (a.Opcode() == Xcode.F_STATEMENT_LIST){
                            for (Xobject b : (XobjList)a){
                                addChildNode(f2, trans(b));
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

            break;

        case ACC_PRAGMA:
            e = createElement(name);

            Element f0 = createElement("string");
            addChildNode(f0, trans(xobj.getArg(0).getString()));
            addChildNode(e, f0);

            Xobject clause = xobj.getArg(1);
            if (clause != null){
        	if(clause instanceof XobjList){
        	    //clause list
        	    Element f1 = createElement("list");

        	    for(Xobject a : (XobjList)clause){
        		if(a instanceof XobjList){
        		    //clause name
        		    Element g = createElement("list");
        		    addChildNode(g, trans(a.getArg(0).getString()));

        		    Xobject vars = a.getArgOrNull(1);
        		    if (vars != null){
        			if (vars instanceof XobjList){
        			    Element g1 = createElement("list");
        			    for (Xobject b : (XobjList)vars){
        				addChildNode(g1, transACCPragmaVarOrArray(b));
        			    }
        			    addChildNode(g, g1);
        			}else{
        			    //int-expr of if, async, num_gangs, num_workers, vector_length, gang, worker, vector, collapse
        			    addChildNode(g, transExpr(vars));
        			}
        		    }
        		    addChildNode(f1, g);
        		}else{
        		    //var of cache
        		    addChildNode(f1, transACCPragmaVarOrArray(a));
        		}
        	    }
        	    addChildNode(e, f1);
        	}else{
        	    //int-expr of wait
        	    addChildNode(e, transExpr(clause));
        	}
            }

            Element f2 = createElement("list");
            Xobject body = xobj.getArgOrNull(2);
            if (body != null){
        	if (body.Opcode() == Xcode.F_STATEMENT_LIST){
        	    for (Xobject a : (XobjList)body){
        		if (a.Opcode() == Xcode.F_STATEMENT_LIST){
        		    for (Xobject b : (XobjList)a){
        			addChildNode(f2, trans(b));
        		    }
        		} else {
        		    addChildNode(f2, trans(a));
        		}
        	    }
        	} else {
        	    addChildNode(f2, trans(body));
        	}
            }
            addChildNode(e, f2);
            break;

        case XMP_PRAGMA:
            e = addChildNodes(createElement("text"),
                              trans("/* ignored Xcode." + xobj.Opcode().toXcodeString() + " */"));
            break;
        case INDEX_RANGE:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)),
                              trans(xobj.getArg(1)),
                              trans(xobj.getArg(2)));
            break;
        case LOWER_BOUND:
        case UPPER_BOUND:
        case STEP:
            e = addChildNodes(createElement(name),
                              trans(xobj.getArg(0)));
            break;
        // case ADDR_OF_EXPR:
        //     e = addChildNodes(createElement(name),
        //                       transExpr(xobj.getArg(0)));
        //     break;
        default:
            fatal_dump("cannot convert Xcode to XcodeML.", xobj);
        }

        if (xobj.Type() != null) {
            addAttributes(e,
                          "type", xobj.Type().getXcodeCId());
        }

        if (xobj.getLineNo() != null) {
            ILineNo lineNo = xobj.getLineNo();
            addAttributes(e,
                          "lineno", Integer.toString(lineNo.lineNo()),
                          "file", lineNo.fileName());
        }

        if (xobj.isGccSyntax()) {
            addAttributes(e,
                          "is_gccSyntax", TRUE_STR);
        }

        if (xobj.isSyntaxModified()) {
            addAttributes(e,
                          "is_modified", TRUE_STR);
        }

        if (xobj.isGccExtension()) {
            addAttributes(e,
                          "is_gccExtension", TRUE_STR);
        }

        return e;
    }

    private Element createTypeElement(String name, Xtype type,
                                      String ... attrs) {
        Element e = createElement(name, attrs);
        return addChildNodes(e, trans(type.getGccAttributes()));
    }

    /* Copied from XmcXobjectToXmObjTranslator.java */
    private XobjList getDeclForNotDeclared(XobjList identList) {
        if (identList == null) {
            return null;
        }

        XobjList declList = Xcons.List();
        for (Xobject a : identList) {
            Ident id = (Ident)a;
            if (id.isDeclared() || !id.getStorageClass().isVarOrFunc()) {
                continue;
            }
            Xtype t = id.Type();
            Xcode declCode = t.isFunction() ? Xcode.FUNCTION_DECL
                                            : Xcode.VAR_DECL;
            declList.add(Xcons.List(declCode, Xcons.Symbol(Xcode.IDENT,
                                                           id.getName()),
                                    null));
        }

        if (declList.isEmpty()) {
            return null;
        }
        return declList;
    }
    
    private Node transACCPragmaVarOrArray(Xobject x){
	if(x == null) return null;

	if(x.Opcode() != Xcode.LIST){
	    return trans(x);
	}else{
	    //array
	    Element array = createElement("list");
	    for (Xobject dim : (XobjList)x){
		if(dim.Opcode() != Xcode.LIST){
		    addChildNode(array, trans(dim));
		}else{
		    Element range = createElement("list");
		    for(Xobject j : (XobjList)dim){
			addChildNode(range, trans(j));
		    }
		    addChildNode(array, range);
		}
	    }
	    return array;
	}
    }
}
