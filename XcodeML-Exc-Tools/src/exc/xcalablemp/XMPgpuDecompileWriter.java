/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.io.*;
import java.util.*;

public class XMPgpuDecompileWriter extends PrintWriter {
  private XobjectFile _env = null;

  public XMPgpuDecompileWriter(Writer out, XobjectFile env) {
    super(out);
    _env = env;
  }

  public void print(XobjectDef def) {
//    printWithIdentList(def.getDef(), _env.getGlobalIdentList());
    System.out.println(def.getDef().toString());
  }

  private void fatal(String msg) {
    System.err.println("Fatal XobjectDecompileStream: " + msg);
    System.exit(1);
  }

  private void printWithIdentList(Xobject v, Xobject id_list) {
    Ident id = null, arg_id = null;
    String func_args = null;
    if (v == null) {
      return;
    }

    switch (v.Opcode()) {
      case LIST:
        {
          for (XobjArgs a = v.getArgs(); a != null; a = a.nextArgs()) {
            printWithIdentList(a.getArg(), id_list);
          }
        } break;
      case FUNCTION_DEFINITION:
        {
          if ((id = findIdent(id_list,v.getArg(0))) == null) {
            fatal("Function id is not found");
          }

          if (id.Type().isFuncProto() && id.Type().getFuncParam() != null) {
            XobjArgs a = id.Type().getFuncParam().getArgs();
            XobjArgs n = v.getArg(1).getArgs();

            if (n == null) {
              if (a != null  &&  a.getArg() == null) {
                func_args = "(...)";
              } else {
                func_args = "(void)";
              }
            } else {
              func_args = "(";
              for (;  n != null;  a = a.nextArgs(), n = n.nextArgs()) {
                arg_id = (Ident)n.getArg();
                func_args += makeDeclType(arg_id.Type(), arg_id.getName());
                if (n.nextArgs() != null) {
                  func_args += ", ";
                }
              }
              if (a != null  &&  a.getArg() == null) {
                func_args += ", ...";
              }
              func_args += ")";
            }
            printStorageClass(id);
            printDeclType(id.Type().getRef(), id.getName() + func_args);
          } else {
            func_args = "(";
            if (v.getArg(1) != null) {
              for (XobjArgs a = v.getArg(1).getArgs(); a != null; a = a.nextArgs()) {
                arg_id = (Ident)a.getArg();
                func_args += arg_id.getName();
                if (a.nextArgs() != null) {
                  func_args += ",";
                }
              }
            }
            func_args += ")";
            printStorageClass(id);
            printDeclType(id.Type().getRef(), id.getName() + func_args);

            printIdentList(v.getArg(1));
            printDeclList(v.getArg(2), v.getArg(1));
          }

          printBody(v.getArg(3), 0);
          println();
        } break;
      case VAR_DECL:
        {
          printLineNo(v);
          printDeclList(v,id_list);
        } break;
      case PRAGMA_LINE:
        {
          printLineNo(v);
          String ptn = "__ompc_output";
          String str = v.getArg(0).getString();
          String arg0 = str.substring(0, ptn.length());
          char spr = str.charAt(ptn.length());
          if ((arg0.compareTo(ptn) == 0) && ((spr == ' ') || (spr == '\t'))) {
            print(str.substring(ptn.length() + 1));
          } else {
            print("#pragma " + str);
          }
        } break;
      default:
        print(v, 0);
        break;
    }
  }

  private void print(Xobject v,int l){
    String op;
    if(v == null) return;

    /* indent */
    //  for(int i = 0; i < l; i++) print("  ");
    //  l++;

    switch(v.Opcode()){
    case LIST:    /* nested */
      for(XobjArgs a = v.getArgs(); a != null; a = a.nextArgs())
        print(a.getArg(),l);
      break;
      // 
      // Statement
      //
    case COMPOUND_STATEMENT:
      /* (COMPOUND_STATEMENT id-list decl statement-list) */
      printLineNo(v);
      println("{");
      printIdentList(v.getArg(0));
      printDeclList(v.getArg(1),v.getArg(0));
      print(v.getArg(2),l);
      println(); print("}");
      break;

    case IF_STATEMENT: /* (IF_STATMENT cond then-part else-part) */
      printLineNo(v);
      print("if(");
      print(v.getArg(0));
      print(")");
      printBody(v.getArg(1),l);
      if(v.getArg(2) != null){
        print(" else ");
        printBody(v.getArg(2),l);
      }
      break;

    case WHILE_STATEMENT: /* (WHILE_STATEMENT cond body) */
      printLineNo(v);
      print("while(");
      print(v.getArg(0));
      print(")");
      printBody(v.getArg(1),l);
      break;

    case FOR_STATEMENT:  /* (FOR init cond iter body) */
      printLineNo(v);
      print("for(");
      print(v.getArg(0));       /* init */
      print(";");
      print(v.getArg(1)); /* cond */
      print(";");
      print(v.getArg(2));  /* iter */
      print(")");
      printBody(v.getArg(3),l);  /* body */
      break;

    case DO_STATEMENT: /* (DO_STATEMENT body cond) */
      printLineNo(v);
      print("do ");
      printBody(v.getArg(0),l);
      print(" while(");
      print(v.getArg(1));
      print(");");
      break;
    case BREAK_STATEMENT:  /* (BREAK_STATEMENT) */
      printLineNo(v);
      print("break; ");
      break;
    case CONTINUE_STATEMENT: /* (CONTINUE_STATEMENT) */
      printLineNo(v);
      print("continue; ");
      break;
    case GOTO_STATEMENT:  /* (GOTO_STATEMENT label) */
      printLineNo(v);
      print("goto ");
      print(v.getArg(0).getSym());
      print(";");
      break;
    case STATEMENT_LABEL: /* (STATEMENT_LABEL label_ident) */
      printLineNo(v);
      print(v.getArg(0).getSym());
      print(":;");
      break;
    case SWITCH_STATEMENT: /* (SWITCH_STATEMENT value body) */
      printLineNo(v);
      print("switch(");
      print(v.getArg(0));
      print(")");
      printBody(v.getArg(1),l);
      break;
    case CASE_LABEL:              /* (CASE_LABEL value) */
      printLineNo(v);
      print("case ");
      print(v.getArg(0));
      print(":");
      break;
    case DEFAULT_LABEL: /* (DEFAULT_LABEL) */
      printLineNo(v);
      print("default:");
      break;
    case RETURN_STATEMENT: /* (RETURN_STATEMENT value) */
      printLineNo(v);
      print("return ");
      if(v.getArg(0) != null) print(v.getArg(0));
      print(";");
      break;
    case EXPR_STATEMENT:
      printLineNo(v);
      print(v.getArg(0));
      print(";");
      break;

      //
      // Expression
      //
    case STRING_CONSTANT:
      print("\"");
      String s = v.getString();
      for(int i = 0; i < s.length(); i++){
        char c = s.charAt(i);
        if(c < ' ' || c == '"') {
            /* less than 0100 */
            if (c < 8) {
                print("\\00"+Integer.toOctalString((int)c));
            } else {
                print("\\0"+Integer.toOctalString((int)c));
            }
        }
        else if(c == '\\') print("\\\\");
        else print(c);
      }
      print("\"");
      break;

    case INT_CONSTANT:
      if(v.Type().isUnsigned())
        print("((unsigned)0x"+Integer.toHexString(v.getInt())+")");
      else
        print(v.getInt());
      break;

    case FLOAT_CONSTANT:
      print(v.getFloat());
      break;

    case LONGLONG_CONSTANT:
      print("0x"+Long.toHexString(v.getLong()));
      break;

      // variable reference
    case VAR_ADDR:
      print("&"+v.getSym());
      break;
    case VAR:
    case ARRAY_ADDR:
    case FUNC_ADDR:
    case MOE_CONSTANT:
      print(v.getSym());
      break;
    case REG:
      print(v.getName());
      break;

      // primary expr
    case POINTER_REF:
      switch(v.left().Opcode()){
      case VAR_ADDR:
        print(v.left().getSym());
        break;
      case VAR:
        print("*");
        print(v.left());
        break;
      case MEMBER_ADDR:
        print("(");
        print(v.left().left());
        print(")->");
        print(v.left().getArg(1).getSym());
        break;
        /*
          case .PLUS_EXPR:
          print(v.left().left());
          print("[");
          print(v.left().right());
          print("]");
          break;
          */
      default:
        print("*(");
        print(v.left());
        print(")");
      }
      break;
    case ARRAY_AREF:
      print("*(");
      print(v.left());
      print(")");
      break;

    case MEMBER_REF:      /* (MEMBER_REF v member), v.member */
      print("((");
      print(v.left());
      print(").");
      print(v.getArg(1).getSym());
      print(")");
      break;

    case MEMBER_ADDR:
      print("&((");
      print(v.left());
      print(")->");
      print(v.getArg(1).getSym());
      print(")");
      break;

    case MEMBER_ARRAY_ADDR:
      print("(");
      print(v.left());
      print(")->");
      print(v.getArg(1).getSym());
      break;

    case ARRAY_REF:       /* (ARRAY_REF x n) */
      print(v.getArg(0));
      if(v.getArg(1).Opcode() == Xcode.LIST){
        for(XobjArgs a = v.getArg(1).getArgs();
            a != null; a = a.nextArgs()){
          print("[");
          print(a.getArg());
          print("]");
        }
      } else {
        print("[");
        print(v.getArg(1));
        print("]");
      }
      break;

    case ADDR_OF: /* & operator */
      print("&(");
      print(v.left());
      print(")");
      break;

    case COMMA_EXPR:      // extended
      print("(");
      for(XobjArgs a = v.getArgs(); a != null; a = a.nextArgs()){
        print(a.getArg());
        if(a.nextArgs() != null) print(",");
      }
      print(")");
      break;

    case FUNCTION_CALL: /* (FUNCTION_CALL function args-list) */
      // add for va_list
      if (v.left().Opcode() == Xcode.FUNC_ADDR) {
        if (v.left().getSym() == "__ompc_output") {
          if (v.right() != null) {
            Xobject  arg0 = v.right().getArgs().getArg();
            XobjArgs args = v.right().getArgs().nextArgs();
            String   fmt  = arg0.getString();

            for (int from = 0; from < fmt.length(); ) {
              int index = fmt.indexOf("%", from);
              if (index == -1) {
                print(fmt.substring(from));
                break;
              } else {
                print(fmt.substring(from,index));
                from = index + 2;
                if (fmt.charAt(index+1) == 's') {
                  print (args.getArg());
                } else if (fmt.charAt(index+1) == 't') {
                  printDeclType (args.getArg().Type(), "");
                }
                args = args.nextArgs();
              }
            }
          }
          break;
        }
      }
      // add for va_list : end

      if(v.left().Opcode() == Xcode.FUNC_ADDR){
        print(v.left().getSym());
      } else {
        print("(");  // bug fix
        print(v.left());
        print(")");
      }
      printArgList(v.right());
      break;

    case POST_INCR_EXPR:
      print("(");
      print(v.left());
      print(")++");
      break;
    case POST_DECR_EXPR:
      print("(");
      print(v.left());
      print(")--");
      break;

    case SIZE_OF_EXPR:    /* (SIZE_OF_EXPR type-or-expr) */
      print("sizeof(");
      printDeclType(v.Type(),null);
      print(")");
      break;

    case CAST_EXPR:               /* (CAST type_name expr ) */
      print("((");
      printDeclType(v.Type(),null);
      print(")(");
      print(v.left());
      print("))");
      break;

    case CONDITIONAL_EXPR:
      /* (CONDITIONAL_EXPR condition true-expr false-expr) */
      print("(");
      print(v.left());
      print(")?(");
      print(v.right().left());
      print("):(");
      print(v.right().right());
      print(")");
      break;

    case ASSIGN_EXPR:
      op = "=";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_PLUS_EXPR:
      op = "+=";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_MINUS_EXPR:
      op = "-=";  printBinaryExpr(op,v.left(),v.right()); break;
    case PLUS_EXPR:
      op = "+";  printBinaryExpr(op,v.left(),v.right()); break;
    case MINUS_EXPR:
      op = "-";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_DIV_EXPR:
      op = "/=";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_MUL_EXPR:
      op = "*=";  printBinaryExpr(op,v.left(),v.right()); break;
    case DIV_EXPR:
      op = "/";  printBinaryExpr(op,v.left(),v.right()); break;
    case MUL_EXPR:
      op = "*";  printBinaryExpr(op,v.left(),v.right()); break;
    case UNARY_MINUS_EXPR:
      op = "-";  printUnaryExpr(op,v.left()); break;
    case ASG_MOD_EXPR:
      op = "%=";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_BIT_AND_EXPR:
      op = "&=";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_BIT_OR_EXPR:
      op = "|=";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_BIT_XOR_EXPR:
      op = "^=";  printBinaryExpr(op,v.left(),v.right()); break;
    case MOD_EXPR:
      op = "%";  printBinaryExpr(op,v.left(),v.right()); break;
    case BIT_AND_EXPR:
      op = "&";  printBinaryExpr(op,v.left(),v.right()); break;
    case BIT_OR_EXPR:
      op = "|";  printBinaryExpr(op,v.left(),v.right()); break;
    case BIT_XOR_EXPR:
      op = "^";  printBinaryExpr(op,v.left(),v.right()); break;
    case BIT_NOT_EXPR:
      op = "~";  printUnaryExpr(op,v.left()); break;
    case ASG_LSHIFT_EXPR:
      op = "<<=";  printBinaryExpr(op,v.left(),v.right()); break;
    case ASG_RSHIFT_EXPR:
      op = ">>=";  printBinaryExpr(op,v.left(),v.right()); break;
    case LSHIFT_EXPR:
      op = "<<";  printBinaryExpr(op,v.left(),v.right()); break;
    case RSHIFT_EXPR:
      op = ">>";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_GE_EXPR:
      op = ">=";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_GT_EXPR:
      op = ">";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_LE_EXPR:
      op = "<=";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_LT_EXPR:
      op = "<";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_EQ_EXPR:
      op = "==";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_NEQ_EXPR:
      op = "!=";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_AND_EXPR:
      op = "&&";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_OR_EXPR:
      op = "||";  printBinaryExpr(op,v.left(),v.right()); break;
    case LOG_NOT_EXPR:
      op = "!";  printUnaryExpr(op,v.left()); break;
    case OMP_PRAGMA:
    case XMP_PRAGMA:
      // System.err.println("OpenMP/XcalableMP pragma is ignored!!!");
      print(v.getArg(2));
      break;

    default:
      /* fatal("print: unknown decopmile = "+v); */
      printUserCode(v);
    }
  }

  private void printStorageClass(Ident id) {
    switch(id.getStorageClass()){
      case AUTO:
        print("auto ");
        break;
      case SNULL: // member
        {
          printDeclType(id.Type(),id.getName());
          if (id.getBitField() != 0) {
            print(":"+id.getBitField());
          }
          return;
        }
      case PARAM:
      case EXTDEF:
      case MEMBER:
        break;
      case EXTERN:
        print("extern ");
        break;
      case STATIC:
        print("static ");
        break;
      case REGISTER:
        print("register ");
        break;
      case REG:
        printDeclType(id.Type(),id.getName());
        return;
      case TAGNAME:
      default:
        fatal("printStorageClass: bad class," + id.getStorageClass().toXcodeString());
        return;
    }
  }

  private void printTypeName(Xtype type) {
    print(makeTypeName(type));
  }

  // FIXME
  private String makeTypeName(Xtype type){
    return new String("void");
  }

  private Ident findIdent(Xobject id_list, Xobject s) {
    if (id_list == null || s == null) {
      return null;
    }

    if (s.Opcode() != Xcode.IDENT) {
      fatal("findIdent: not IDENT");
    }

    for (XobjArgs a = id_list.getArgs(); a != null; a = a.nextArgs()) {
      Ident id = (Ident)a.getArg();
      if (id.getStorageClass() == StorageClass.TAGNAME) {
        continue;
      }

      if (s.getSym().equals(id.getName())) {
        return id;
      }
    }
    return null;
  }

  private void printDeclType(Xtype type, String name) {
    print(makeDeclType(type, name));
  }

  private String makeDeclType(Xtype type, String name) {
    String decltype = "";
    Xtype t;
    Stack nested_decls = new Stack();

    for (t = type; t != null; t = t.getRef()) {
      if (t.getKind() == Xtype.POINTER ||
          t.getKind() == Xtype.FUNCTION ||
          t.getKind() == Xtype.ARRAY) {
        nested_decls.push(t);
      } else {
        break;
      }
    }

    if (t == null) {
      fatal("makeDeclType: no base type");
    } else {
      if (t.isConst()) {
        decltype += "const ";
      }

      if (t.isVolatile()) {
        decltype += "volatile ";
      }

      decltype += makeTypeName(t);
    }

    decltype += " " + makeDeclaratorRec(nested_decls, name);

    return decltype;
  }

  private String makeDeclaratorRec(Stack decls, String name) {
    String decltype = "";
    boolean pred = false;

    if (decls.empty()) {
      if (name != null) {
        return name;
      } else {
        return "";
      }
    }

    Xtype t = (Xtype) decls.pop();
    if (!decls.empty() && t.getKind() != Xtype.POINTER && ((Xtype)decls.peek()).getKind() == Xtype.POINTER) {
      pred = true;
    }

    switch (t.getKind()) {
      case Xtype.POINTER:
        {
          decltype += "*";
          if (t.isConst()) {
            decltype += "const ";
          }

          if (t.isVolatile()) {
            decltype += "volatile ";
          }

          decltype += makeDeclaratorRec(decls, name);
        } break;
    case Xtype.FUNCTION:
      {
        if (pred) {
          decltype += "(";
        }

        decltype += makeDeclaratorRec(decls, name);

        if (pred) {
          decltype += ")";
        }

        if (t.isFuncProto() && t.getFuncParam() != null) {
          decltype += makeFuncProtoArgs(t.getFuncParam());
        } else {
          decltype += "()";
        }
      } break;
    case Xtype.ARRAY:
      {
        if (!decls.empty() && ((Xtype)decls.peek()).getKind() == Xtype.FUNCTION) {
          decltype += "*";
          decltype += makeDeclaratorRec(decls, name);
          break;
        }

        if (t.getArrayDim() < 0) {
          decltype += "(";
          decltype += "*";
          decltype += makeDeclaratorRec(decls, name);
          decltype += ")";
          break;
        }

        if (pred) {
          decltype += "(";
        }

        decltype += makeDeclaratorRec(decls, name);

        if (pred) {
          decltype += ")";
        }

        if (t.getArrayDim() <= 0) {
          decltype += "[]";
        } else {
          decltype += "[" + t.getArrayDim() + "]";
        }
      } break;
    default:
      fatal("printDeclaratorRec");
    }

    return decltype;
  }

  private String makeFuncProtoArgs(Xobject args) {
    String func_args;

    if (args.getArgs() == null) {
      func_args = "(void)";
    } else {
      func_args = "(";
      for (XobjArgs a = args.getArgs(); a != null; a = a.nextArgs()) {
        if (a.getArg() == null) {
          func_args += "...";
          break;
        } else {
          if (a.getArg().Opcode() == Xcode.IDENT) {
            func_args += makeDeclType(a.getArg().Type(), a.getArg().getName());
          } else {
            func_args += makeDeclType(a.getArg().Type(), null);
          }
        }

        if (a.nextArgs() != null) {
          func_args += ",";
        }
      }
      func_args += ")";
    }

    return func_args;
  }

  private void printInitializer(Xobject init) {
    if (init.Opcode() == Xcode.LIST) {
      int n = 0;
      print("{");
      for (XobjArgs a = init.getArgs(); a != null; a = a.nextArgs()) {
        printInitializer(a.getArg());
        print(",");
        n++;
        if (n % 16 == 0) {
          print("\n");
        }
      }
      print("}");
    } else {
      print(init);
    }
  }

  private void printBinaryExpr(String op, Xobject left, Xobject right) {
    print("(");
    print(left);
    print(")");
    print(op);
    print("(");
    print(right);
    print(")");
  }

  private void printUnaryExpr(String op, Xobject opd) {
    print(op);
    print("(");
    print(opd);
    print(")");
  }

  private void printIdentDecl(Ident id) {
    switch (id.getStorageClass()) {
      case AUTO:
        print("auto ");
        break;
      case SNULL:
        printDeclType(id.Type(), id.getName());
        if (id.getBitField() != 0) {
          print(":" + id.getBitField());
        }
        return;
      case PARAM:
      case EXTDEF:
      case MEMBER:
        break;
      case EXTERN:
        print("extern ");
        break;
      case STATIC:
        print("static ");
        break;
      case REGISTER:
        print("register ");
        break;
      case REG:
        printDeclType(id.Type(), id.getName());
        return;
      case TYPEDEF_NAME:
        return;
      case TAGNAME:
      default:
        fatal("printIdentDecl: bad class," + id.getStorageClass().toXcodeString());
        return;
    }

    printDeclType(id.Type(), id.getName());
  }

  private void printIdentList(Xobject id_list) {
    Ident id;
    Xtype type;

    if (id_list == null) {
      return;
    }

    printStructTAGNAME(id_list);
  }

  private void printStructTAGNAME(Xobject id_list) {
    Ident id;
    Xtype type;

    for (XobjArgs a = id_list.getArgs(); a != null; a = a.nextArgs()) {
      id = (Ident)a.getArg();

      if (id.getStorageClass() != StorageClass.TAGNAME) {
        continue;
      }

      type = id.Type();

      switch (type.getKind()) {
        case Xtype.STRUCT:
        case Xtype.UNION:
          printTypeName(type);
          println(";");
          break;
        default:
          break;
      }
    }

    for (XobjArgs a = id_list.getArgs(); a != null; a = a.nextArgs()) {
      id = (Ident)a.getArg();

      if (id.getStorageClass() != StorageClass.TAGNAME) {
        continue;
      }

      type = id.Type();

      switch (type.getKind()) {
        case Xtype.STRUCT:
        case Xtype.UNION:
          {
            if (type.getMemberList().getArgs() == null) {
              break;
            }
            printTypeName(type);
            println(" {");
            for (XobjArgs aa = type.getMemberList().getArgs(); aa != null; aa = aa.nextArgs()) {
              printIdentDecl((Ident)aa.getArg());
              println(";");
            }
            println(" };");
          } break;
        case Xtype.ENUM:
          {
            if(type.getMoeList() == null) {
              break;
            }

            printTypeName(type);
            println(" {");

            for (XobjArgs a1 = type.getMoeList().getArgs(); a1 != null; a1=a1.nextArgs()) {
              if ((id = findIdent(id_list, a1.getArg())) == null) {
                fatal("Enum ID is not found");
              }

              print(a1.getArg().getSym());

              if (id.getValue() != null) {
                print("=");
                print(id.getAddr());
              }

              println(",");
            }

            println( " };");
          } break;
        default:
          break;
      }
    }
  }

  private void printUserCode(Xobject v){
    print("_user_xcode(/* " + v.toString() +" */)");
  }

  private void printArgList(Xobject v) {
    print("(");

    if (v != null) {
      for (XobjArgs a = v.getArgs(); a != null; a = a.nextArgs()) {
        print(a.getArg());
        if (a.nextArgs() != null) {
          print(",");
        }
      }
    }

    print(")");
  }

  private void printBody(Xobject v, int l) {
    if (v == null) {
      print("{}");  // null body
      return;
    }

    if (v.Opcode() == Xcode.LIST) {
      print("{");
      print(v, l);
      println();
      print("}");
    } else {
      print(v, l);
    }
  }

  private void printDeclList(Xobject v, Xobject id_list) {
    Ident id;
    if (v == null) {
      return;
    }

    switch (v.Opcode()) {
      case LIST:
        {
          for(XobjArgs a = v.getArgs(); a != null; a = a.nextArgs()) {
            printDeclList(a.getArg(),id_list);
          }
        } break;
      case VAR_DECL:
        {
          Xobject s = v.getArg(0);
          if ((id = findIdent(id_list,s)) == null &&
              (id = findIdent(_env.getGlobalIdentList(), s)) == null) {
            fatal("Variable id is not found, "+s);
          }

          if (id.getStorageClass() == StorageClass.SNULL) {
            break;
          }

          if (id.getStorageClass() == StorageClass.EXTDEF) {
            print("extern ");
          }

          printIdentDecl(id);

          if (v.getArg(1) != null) {
            print("=");
            printInitializer(v.getArg(1));
          }

          println(";");
        } break;
      default:
        break;
    }
  }

  private void printLineNo(Xobject v) {
    LineNo ln = v.getLineNo();
    if (ln == null) {
      return;
    }

    println();
    println("# " + ln.lineNo() + " \"" + ln.fileName() + "\"");
  }
}
