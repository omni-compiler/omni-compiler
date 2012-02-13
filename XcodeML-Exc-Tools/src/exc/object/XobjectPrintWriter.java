/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.io.*;
import java.util.List;

import xcodeml.util.XmLog;

public class XobjectPrintWriter extends PrintWriter
{
    // constructor
    public XobjectPrintWriter(OutputStream out)
    {
        super(out, true);
    }

    public XobjectPrintWriter(Writer out)
    {
        super(out, true);
    }

    public void print(Xobject v)
    {
        printObject(v);
    }

    public void printInt(long i)
    {
        print("0x");
        print(Long.toHexString(i));
    }

    public void printBool(boolean b)
    {
        if(b)
            print("1");
        else
            print("0");
    }

    public void printObject(Xobject v)
    {
        printObjectRec(v, 0);
        println();
        flush();
    }

    private void printDefs(List<XobjectDef> defs, int i)
    {
        for(XobjectDef def : defs) {
            if(def.getDef() != null) {
                printObjectRec(def.getDef(), i);
                println();
                flush();
            }
            if(def.hasChildren()) {
                indent(i + 1);
                print("(F_CONTAINS_STATEMENT");
                println();
                printDefs(def.getChildren(), i + 2);
                indent(i + 1);
                print(")");
                println();
                flush();
            }
        }
    }
    
    public void printDefs(List<XobjectDef> defs)
    {
        printDefs(defs, 0);
    }

    public void printObjectNoIndent(Xobject v)
    {
        printObjectRec(v, -1);
        flush();
    }
    
    private void indent(int l)
    {
        /* indent */
        for(int i = 0; i < l; i++)
            print("  ");
    }

    void printObjectRec(Xobject v, int l)
    {
        Xcode opcode;
        Xtype type;

        /* indent */
        indent(l);

        if(v == null) { /* special case */
            print("()");
            return;
        }

        opcode = v.Opcode();
        /* XCODE NAME */
        if(opcode != null && opcode.isBuiltinCode()) {
            print("(");
            print(opcode.toXcodeString());
            if((type = v.Type()) != null) {
                print(":");
                printType(type);
            }
        } else {
            print(v.toString());
            return;
        }

        if(v.isTerminal()) {
            switch(opcode) {
            case ID_LIST:
                printIdentList(v, l + 1);
                indent(l);
                print(")");
                return;
            case IDENT: /* NAME */
            case VAR_ADDR: /* ICON */
            case VAR:
            case ARRAY_ADDR:
            case FUNC_ADDR:
            case MOE_CONSTANT:
                print(" " + v.getSym() + ")");
                return;
            case INT_CONSTANT:
            case REG:
                print(" 0x" + Integer.toHexString(v.getInt()) + ")");
                return;
            case LONGLONG_CONSTANT:
                print(" 0x" + Long.toHexString(v.getLongHigh()) + " 0x"
                    + Long.toHexString(v.getLongLow()) + ")");
                return;
            case FLOAT_CONSTANT:
                print(" " + ((XobjFloat)v).getFloatString() + ")");
                return;
            case F_COMPLEX_CONSTATNT:
                print(" " + ((XobjComplex)v).getReValue() + " " + ((XobjComplex)v).getImValue() + ")");
                return;
            case F_LOGICAL_CONSTATNT:
                print(" " + ((XobjBool)v).getBoolValue() + ")");
                return;
            case STRING:
            case STRING_CONSTANT:
            case F_CHARACTER_CONSTATNT:
	      // case TEXT:
	      // case PRAGMA_LINE:
                print(" \"" + v.getString() + "\")");
                return;
            }
            XmLog.fatal("unknown terminal : " + v.OpcodeName());
        }

        LineNo ln = v.getLineNo();
        if(ln != null)
            print("@" + ln.lineNo() + "." + ln.fileName());

        if(v.getArgs() != null) {
            if(l < 0)
                print(" ");
            else {
                println();
                l++;
            }
            for(XobjArgs a = v.getArgs(); a != null; a = a.nextArgs()) {
                printObjectRec(a.getArg(), l);
                if(a.nextArgs() != null) {
                    if(l < 0)
                        print(" ");
                    else
                        println();
                }
            }
        }
        print(")");
    }

    public void printIdentList(Xobject o, int l)
    {
        println();
        int n = o.Nargs();
        for(Xobject a : (XobjList)o) {
            indent(l);
            if(--n == 0)
                print(a);
            else
                println(a);
        }
    }

    public void printType(Xtype type)
    {
        String tid = type.getXcodeCId();
        if(tid == null)
            tid = type.getXcodeFId();
        print(tid != null ? tid : "<NA>");
    }
}
