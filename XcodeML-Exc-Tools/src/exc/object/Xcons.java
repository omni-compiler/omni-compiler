package exc.object;

import java.math.BigInteger;
import exc.block.Block;
import xcodeml.util.XmOption;
import static xcodeml.util.XmLog.fatal;

//
// class for static constructor
// 
public class Xcons
{
    public static XobjString Symbol(Xcode code, String value)
    {
        return new XobjString(code, value);
    }

    public static XobjString Symbol(Xcode code, Xtype type, String value)
    {
        return new XobjString(code, type, value);
    }

    public static XobjString Symbol(Xcode code, Xtype type, String value, VarScope scope)
    {
        XobjString x = new XobjString(code, type, value);
        x.setScope(scope);
        return x;
    }

    public static XobjString StringConstant(String value)
    {
        return StringConstant(Xtype.stringType, value);
    }

    public static XobjString StringConstant(Xtype type, String value)
    {
        return new XobjString(Xcode.STRING_CONSTANT, type, value, null);
    }

    public static XobjString FcharacterConstant(Xtype type, String value, String fkind)
    {
        return new XobjString(Xcode.F_CHARACTER_CONSTATNT, type, value, fkind);
    }

    public static XobjString String(String value)
    {
        return new XobjString(Xcode.STRING, value);
    }

    public static XobjInt Int(Xcode code, Xtype type, int value)
    {
        return new XobjInt(code, type, value, null);
    }

    public static XobjInt Int(Xcode code, int value)
    {
        return new XobjInt(code, value);
    }

    public static XobjInt IntConstant(int value)
    {
        return new XobjInt(Xcode.INT_CONSTANT, Xtype.intType, value, null);
    }

    public static XobjInt IntConstant(int value, Xtype type, String kind)
    {
        return new XobjInt(Xcode.INT_CONSTANT, type, value, kind);
    }

    public static XobjLong Long(Xcode code, Xtype type, long low)
    {
        return Long(code, type, 0, low);
    }

    public static XobjLong Long(Xcode code, Xtype type, long high, long low)
    {
        return new XobjLong(code, type, high, low);
    }

    public static XobjLong LongLongConstant(long high, long low)
    {
        return new XobjLong(high, low);
    }

    public static XobjLong LongLongConstant(BigInteger bi, Xtype type, String fkind)
    {
        return new XobjLong(Xcode.LONGLONG_CONSTANT, type, bi, fkind);
    }

    public static XobjFloat Float(Xcode code, Xtype type, double d)
    {
        return new XobjFloat(code, type, d);
    }

    public static XobjFloat Float(Xcode code, Xtype type, String floatStr)
    {
        return new XobjFloat(code, type, floatStr);
    }

    public static XobjFloat FloatConstant(double d)
    {
        return new XobjFloat(Xcode.FLOAT_CONSTANT, Xtype.doubleType, d);
    }

    public static XobjFloat FloatConstant(Xtype type, String floatStr, String fkind)
    {
        return new XobjFloat(Xcode.FLOAT_CONSTANT, Xtype.floatType, floatStr, fkind);
    }
    
    public static XobjBool FlogicalConstant(Xtype type, boolean value, String fkind)
    {
        return new XobjBool(Xcode.F_LOGICAL_CONSTATNT, type, value, fkind);
    }
    
    public static XobjBool FlogicalConstant(boolean value)
    {
        return FlogicalConstant(Xtype.FlogicalType, value, null);
    }
    
    public static XobjComplex ComplexConstant(Xtype type, Xobject re, Xobject im)
    {
        return new XobjComplex(Xcode.F_COMPLEX_CONSTATNT, type, re, im);
    }
    
    public static XobjList List()
    {
        return new XobjList();
    }

    public static XobjList List(Xcode code)
    {
        return new XobjList(code);
    }

    public static XobjList List(Xcode code, Xtype type)
    {
        return new XobjList(code, type);
    }

    public static XobjList List(Xcode code, Xtype type, XobjArgs a)
    {
        return new XobjList(code, type, a);
    }

    public static XobjList List(Xcode code, Xtype type, Xobject ... a)
    {
        return new XobjList(code, type, a);
    }
    
    public static XobjList List(Xcode code, XobjArgs a)
    {
        return new XobjList(code, a);
    }

    public static XobjList List(Xcode code, Xobject ... a)
    {
        return new XobjList(code, a);
    }

    public static XobjList List(XobjArgs a)
    {
        return new XobjList(Xcode.LIST, a);
    }

    public static XobjList List(Xobject ... a)
    {
        return new XobjList(Xcode.LIST, a);
    }
    
    public static XobjList IDList()
    {
        return new XobjList(Xcode.ID_LIST);
    }

    public static Ident Ident(String name, StorageClass stg_class, Xtype type,
                              Xobject v, VarScope scope)
    {
        return new Ident(name, stg_class, type, v, scope, null);
    }
    public static Ident Ident(String name, StorageClass stg_class, Xtype type,
                              Xobject v, VarScope scope, Xobject codimensions)
    {
        return new Ident(name, stg_class, type, v, scope, codimensions);
    }
    
    //
    // Constructor for expression
    // making constant folding (not implmented)
    //
    // if type == null, don't care data type!
    public static Xobject PointerRef(Xobject x)
    {
        if(!XmOption.isLanguageC())
            return x;
        
        Xtype type = x.Type();
        if(type != null && type.isPointer())
            type = type.getRef();
        switch(x.Opcode()) {
        case ARRAY_ADDR:
            return Xcons.Symbol(Xcode.ARRAY_ADDR, type, x.getSym());
        case VAR_ADDR:
            return Xcons.Symbol(Xcode.VAR, type, x.getSym());
        case FUNC_ADDR:
            fatal("PointerRef(): FUNC_ADDR");
            break;
        case ADDR_OF:
            return x.operand();
        default:
            if(type == null || x.Type().isPointer())
                return Xcons.List(Xcode.POINTER_REF, type, x);
            else {
                fatal("PointerRef(): not Pointer");
            }
        }
        return null;
    }

    public static Xobject memberAddr(Xobject x, String member_name)
    {
        Xtype type = x.Type();
        if(!type.isPointer())
            fatal("memberRef: not Pointer");
        type = type.getRef();
        if(!type.isStruct() && !type.isUnion())
            fatal("memberAddr: not struct/union");
        
        Ident mid = type.getMemberList().getMember(member_name); // id_list
        if(mid == null)
            fatal("memberAddr: member name is not found: " + member_name);
        
        type = mid.Type();
        return List(Xcode.MEMBER_ADDR,
            Xtype.Pointer(mid.Type()), x,
            Symbol(Xcode.IDENT, mid.Type(), mid.getName()));
    }

    public static Xobject memberRef(Xobject x, String member_name)
    {
        Xtype type = x.Type();
        if(!type.isPointer())
            fatal("memberRef: not Pointer");
        type = type.getRef();
        if(!type.isStruct() && !type.isUnion())
            fatal("memberAddr: not struct/union");
        
        Ident mid = type.getMemberList().getMember(member_name); // id_list
        if(mid == null)
            fatal("memberAddr: member name is not found: " + member_name);
        
        type = mid.Type();
        return List(Xcode.MEMBER_REF,
            mid.Type(), x,
            Symbol(Xcode.IDENT, mid.Type(), mid.getName()));
    }

    public static Xobject memberAddr(Xobject x, Ident mid)
    {
        Xtype type = x.Type();
        if(!type.isPointer())
            fatal("memberAddr: not Pointer");
        type = type.getRef();
        if(!type.isStruct() && !type.isUnion())
            fatal("memberAddr: not struct/union");
        type = mid.Type();
        
        if(XmOption.isLanguageC())
            return List(Xcode.MEMBER_ADDR, Xtype.Pointer(type), x,
                Symbol(Xcode.IDENT, type, mid.getName()));
        
        return List(Xcode.MEMBER_REF, type, x,
            Symbol(Xcode.IDENT, type, mid.getName()));
    }

    // don't reduce
    public static Xobject AddrOf(Xobject v)
    {
        if(v == null)
            throw new NullPointerException("v");
        if(!XmOption.isLanguageC())
            return v;
        Xtype type = v.Type();
        if(type != null)
            type = Xtype.Pointer(type);
        return Xcons.List(Xcode.ADDR_OF, type, v);
    }

    public static Xobject AddrOfVar(Xobject x)
    {
        if(x == null)
            throw new NullPointerException("x");
        
        if(x instanceof Ident) {
            return Xcons.Symbol(Xcode.VAR_ADDR, Xtype.Pointer(x.Type()), x.getName());
        }
        
        if(!XmOption.isLanguageC())
            return x;

        switch(x.Opcode()) {
        case VAR:
            return Xcons.Symbol(Xcode.VAR_ADDR, Xtype.Pointer(x.Type()), x.getSym());
        case ARRAY_REF:
            return Xcons.Symbol(Xcode.ARRAY_ADDR, Xtype.Pointer(x.Type()), x.getSym());
        case MEMBER_REF:
            return Xcons.Symbol(Xcode.MEMBER_ADDR, Xtype.Pointer(x.Type()), x.getSym());
        default:
            fatal("AddrOfVar: not Variable");
            return null;
        }
    }

    public static Xobject Set(Xobject lv, Xobject rv)
    {
        if(lv == null)
            throw new NullPointerException("lv");
        if(rv == null)
            throw new NullPointerException("rv");
        
        if(XmOption.isLanguageC())
            return Xcons.List(Xcode.ASSIGN_EXPR, lv.Type(), lv, rv);
        return Xcons.List(Xcode.F_ASSIGN_STATEMENT, lv, rv);
    }

    public static Xobject Cast(Xtype t, Xobject v)
    {
        if(v == null)
            throw new NullPointerException("v");
        return Xcons.List(Xcode.CAST_EXPR, t, v);
    }
    
    public static Xobject functionCall(Xobject f, Xobject args)
    {
        if(f == null)
            throw new NullPointerException("f");
        Xtype t;
        if(f.Opcode() == Xcode.FUNC_ADDR)
            t = f.Type().getRef();
        else
            t = f.Type();
        if(!t.isFunction())
            fatal("fuctionCall: not Function : " + f.toString());
        return Xcons.List(Xcode.FUNCTION_CALL, t.getRef(), f, args);
    }

    public static Xobject binaryOp(Xcode code, Xobject x, Xobject y)
    {
        if(x == null)
            throw new NullPointerException("x");
        if(y == null)
            throw new NullPointerException("y");
        Xtype t, lt, rt;
        t = null;
        lt = x.Type();
        rt = y.Type();
        
        if(lt.isFarray())
            lt = lt.getRef();
        if(rt.isFarray())
            rt = rt.getRef();
        
        switch(code) {
        case PLUS_EXPR:
        case MINUS_EXPR:
            if(lt.isPointer() && rt.isIntegral()) {
                t = lt;
                break;
            }
            if(rt.isPointer() && lt.isIntegral()) {
                t = rt;
                break;
            }
            if(lt.isIntegral() && rt.isFloating()) {
                t = lt;
                break;
            }
            if(rt.isIntegral() && lt.isFloating()) {
                t = rt;
                break;
            }
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            //if(lt.isFloating() && rt.isFloating()) {
            if(lt.isNumeric() && rt.isNumeric()) {           // #357
                t = BasicType.Conversion(lt, rt);
                break;
            }
            fatal("BinaryOp: bad type");

        case MUL_EXPR:
            if(y.isZeroConstant())
                return IntConstant(0); /* x*0 = 0 */
            if(x.isOneConstant())
                return y; /* 1*y = y */
        case DIV_EXPR:
            if(x.isZeroConstant())
                return IntConstant(0); /* 0*y = 0, 0/y = 0 */
            if(y.isOneConstant())
                return x; /* x*1 = x, x/1 = x */

            if(lt.isIntegral() && rt.isFloating()) {
                t = lt;
                break;
            }
            if(rt.isIntegral() && lt.isFloating()) {
                t = rt;
                break;
            }
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            //if(lt.isFloating() && rt.isFloating()) {
            if(lt.isNumeric() && rt.isNumeric()) {           // #357
                t = BasicType.Conversion(lt, rt);
                break;
            }
            fatal("BinaryOp: bad type");

        case MOD_EXPR:
            if(y.isOneConstant())
                return x;
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            if(lt.isNumeric() && rt.isNumeric()) {           // #357
                t = BasicType.Conversion(lt, rt);
                break;
            }
            fatal("BinaryOp: bad type");

        case BIT_OR_EXPR:
            if(y.isZeroConstant())
                return x;
            if(x.isZeroConstant())
                return y;
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            fatal("BinaryOp: bad type");

        case BIT_AND_EXPR:
            if(y.isZeroConstant() || x.isZeroConstant())
                return IntConstant(0);
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            fatal("BinaryOp: bad type");

        case BIT_XOR_EXPR:
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            fatal("BinaryOp: bad type");

        case LOG_GE_EXPR:
        case LOG_GT_EXPR:
        case LOG_LE_EXPR:
        case LOG_LT_EXPR:
        case LOG_EQ_EXPR:
        case LOG_NEQ_EXPR:
            if(lt.isIntegral() && rt.isFloating()) {
                t = Xtype.intType;
                break;
            }
            if(rt.isIntegral() && lt.isFloating()) {
                t = Xtype.intType;
                break;
            }
            if(lt.isIntegral() && rt.isIntegral()) {
                t = Xtype.intType;
                break;
            }
            if(lt.isFloating() && rt.isFloating()) {
                t = Xtype.intType;
                break;
            }
            fatal("BinaryOp: bad type");

        case LOG_AND_EXPR:
        case LOG_OR_EXPR:
            t = Xtype.intType; /* ok ? */
            break;

        case LSHIFT_EXPR:
        case RSHIFT_EXPR:
            if(lt.isIntegral() && rt.isIntegral()) {
                t = lt;
                break;
            }
            fatal("BinaryOp: bad type");
        default:
            fatal("BinaryOp: bad code");
        }
        // reduce phase
        switch(code) {
        case PLUS_EXPR:
            if(x.isIntConstant() && y.isIntConstant())
                return Int(Xcode.INT_CONSTANT, t, x.getInt() + y.getInt());
            if(x.isZeroConstant()) {
                if (rt.equals(t))
                    return y;
                else
                    return Xcons.Cast(t, y);
            }
            // go through
        case MINUS_EXPR:
            if(code == Xcode.MINUS_EXPR) {
                if(x.isIntConstant() && y.isIntConstant())
                    return Int(Xcode.INT_CONSTANT, t, x.getInt() - y.getInt());
                if(x.isZeroConstant())
                    return Xcons.List(Xcode.UNARY_MINUS_EXPR, t, y);
            }
            if(y.isZeroConstant()) {
                if (lt.equals(t))
                    return x;
                else
                    return Xcons.Cast(t, x);
            }
        }
        return Xcons.List(code, t, x, y);
    }

    public static Xobject asgOp(Xcode code, Xobject x, Xobject y)
    {
        if(x == null)
            throw new NullPointerException("x");
        if(y == null)
            throw new NullPointerException("y");
        Xtype t, lt, rt;
        t = null;
        lt = x.Type();
        rt = y.Type();
        switch(code) {
        case ASG_PLUS_EXPR:
        case ASG_MINUS_EXPR:
            if(lt.isPointer() && rt.isIntegral()) {
                t = lt;
                break;
            }
            if(rt.isPointer() && lt.isIntegral()) {
                t = rt;
                break;
            }
        case ASG_MUL_EXPR:
        case ASG_DIV_EXPR:
            if(lt.isIntegral() && rt.isFloating()) {
                t = lt;
                break;
            }
            if(rt.isIntegral() && lt.isFloating()) {
                t = rt;
                break;
            }
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            if(lt.isFloating() && rt.isFloating()) {
                t = BasicType.Conversion(lt, rt);
                break;
            }
            fatal("AgnOp: bad type");

        case ASG_MOD_EXPR:
        case ASG_BIT_OR_EXPR:
        case ASG_BIT_AND_EXPR:
        case ASG_BIT_XOR_EXPR:
            if(lt.isIntegral() && rt.isIntegral()) {
                t = ConversionIntegral(lt, rt);
                break;
            }
            fatal("AsgOp: bad type");

        case ASG_LSHIFT_EXPR:
        case ASG_RSHIFT_EXPR:
            if(lt.isIntegral() && rt.isIntegral()) {
                t = lt;
                break;
            }
            fatal("AsgOp: bad type");
        default:
            fatal("AsgOp: bad code");
        }
        return Xcons.List(code, t, x, y);
    }

    public static Xcode unAsgOpcode(Xcode code)
    {
        switch(code) {
        case POST_INCR_EXPR:
        case PRE_INCR_EXPR:
        case ASG_PLUS_EXPR:
            return Xcode.PLUS_EXPR;
        case POST_DECR_EXPR:
        case PRE_DECR_EXPR:
        case ASG_MINUS_EXPR:
            return Xcode.MINUS_EXPR;
        case ASG_MUL_EXPR:
            return Xcode.MUL_EXPR;
        case ASG_DIV_EXPR:
            return Xcode.DIV_EXPR;
        case ASG_MOD_EXPR:
            return Xcode.MOD_EXPR;
        case ASG_BIT_OR_EXPR:
            return Xcode.BIT_OR_EXPR;
        case ASG_BIT_AND_EXPR:
            return Xcode.BIT_AND_EXPR;
        case ASG_BIT_XOR_EXPR:
            return Xcode.BIT_XOR_EXPR;
        case ASG_LSHIFT_EXPR:
            return Xcode.LSHIFT_EXPR;
        case ASG_RSHIFT_EXPR:
            return Xcode.RSHIFT_EXPR;
        default:
            fatal("unAsgOpcode: bad code, " + code);
            return null;
        }
    }

    public static Xcode revLogicalOpcode(Xcode code)
    {
        switch(code) {
        case LOG_EQ_EXPR:
            return Xcode.LOG_NEQ_EXPR;
        case LOG_NEQ_EXPR:
            return Xcode.LOG_EQ_EXPR;
        case LOG_GE_EXPR:
            return Xcode.LOG_LT_EXPR;
        case LOG_GT_EXPR:
            return Xcode.LOG_LE_EXPR;
        case LOG_LE_EXPR:
            return Xcode.LOG_GT_EXPR;
        case LOG_LT_EXPR:
            return Xcode.LOG_GT_EXPR;
        default:
            fatal("revLogicalOpcode: bad code, " + code);
        }
        return null;
    }

    public static Xobject unaryOp(Xcode code, Xobject x)
    {
        if(x == null)
            throw new NullPointerException("x");
        Xtype t = x.Type();
        switch(code) {
        case UNARY_MINUS_EXPR:
            /* if(x.isZeroConstant()) return IntConstant(0); */
            if(x.Opcode() == Xcode.INT_CONSTANT)
                return Int(Xcode.INT_CONSTANT, x.Type(), -x.getInt());
            if(t.isIntegral() || t.isFloating())
                break;
            fatal("UnaryOp: bad type");
        case LOG_NOT_EXPR:
            if(t.isIntegral() || t.isFloating()) {
                t = Xtype.intType;
                break;
            } else if(t.isBool()){
                break;
            }
            fatal("UnaryOp: bad type");
        case BIT_NOT_EXPR:
            if(t.isIntegral())
                break;
            fatal("UnaryOp: bad type");
        default:
            fatal("BinaryOp: bad code");
        }
        return Xcons.List(code, t, x);
    }
    
    public static Xobject SizeOf(Xtype type)
    {
        return Xcons.List(Xcode.SIZE_OF_EXPR, Xtype.intType, Xcons.List(Xcode.TYPE_NAME, type));
    }
    
    public static Xtype ConversionIntegral(Xtype lt, Xtype rt)
    {
        if(lt.isEnum() && rt.isEnum()) {
            return lt;
        }
        if((lt.isEnum() && !rt.isEnum()) || (!lt.isEnum() && rt.isEnum())) {
            Xtype et = lt.isEnum() ? lt : rt;
            Xtype at = lt.isEnum() ? rt : lt;
            
            if(at.getKind() == Xtype.BASIC) {
                switch(at.getBasicType()) {
                case BasicType.CHAR:
                case BasicType.UNSIGNED_CHAR:
                case BasicType.SHORT:
                case BasicType.UNSIGNED_SHORT:
                    return et;
                }
            }
            return at;
        }
        
        return BasicType.Conversion(lt, rt);
    }

    /* constant folding. only simple & integer case is supported */
    public static Xobject Reduce(Xobject x)
    {
        return Reduce(x, null);
    }
    
    /* constant folding. only simple & integer case is supported */
    public static Xobject Reduce(Xobject x, Block b)
    {
        if(x == null)
            return null;
        
        Xobject r, l;
        
        if(b != null && XmOption.isLanguageF()) {
            Ident id = null;
            if(x.isVarRef())
                id = b.findVarIdent(x.getName());
            else if(x instanceof Ident)
                id = (Ident)x;
            
            if(id != null && id.getFparamValue() != null) {
                return Reduce(id.getFparamValue().getArg(0), b);
            }
        }
        
        if(!x.Type().isIntegral())
            return x; // only for integer

        switch(x.Opcode()) {
        case UNARY_MINUS_EXPR:
            l = Reduce(x, b);
            if(l.isIntConstant())
                return Int(Xcode.INT_CONSTANT, x.Type(), -l.getInt());
            return l;
        case PLUS_EXPR:
            l = Reduce(x.left(), b);
            r = Reduce(x.right(), b);
            if(l.isIntConstant() && r.isIntConstant())
                return Int(Xcode.INT_CONSTANT, x.Type(), l.getInt() + r.getInt());
            if(r.isZeroConstant())
                return l; /* l+0 = l */
            if(l.isZeroConstant())
                return r; /* 0+r=r */
            break;
        case MINUS_EXPR:
            l = Reduce(x.left(), b);
            r = Reduce(x.right(), b);
            if(l.isIntConstant() && r.isIntConstant())
                return Int(Xcode.INT_CONSTANT, x.Type(), l.getInt() - r.getInt());
            if(r.isZeroConstant())
                return l; /* l-0 = l */
            if(l.isZeroConstant())
                return Xcons.List(Xcode.UNARY_MINUS_EXPR, x.Type(), r); /*
                                                                         * 0-r=-r
                                                                         */
            break;
        case MUL_EXPR:
            l = Reduce(x.left(), b);
            r = Reduce(x.right(), b);
            if(l.isIntConstant() && r.isIntConstant())
                return Int(Xcode.INT_CONSTANT, x.Type(), l.getInt() * r.getInt());
            if(l.isZeroConstant() || r.isZeroConstant())
                return IntConstant(0);
            if(l.isOneConstant())
                return r;
            if(r.isOneConstant())
                return l;
            break;
        case DIV_EXPR:
            l = Reduce(x.left(), b);
            r = Reduce(x.right(), b);
            if(r.isZeroConstant())
                break; // avoid error
            if(l.isIntConstant() && r.isIntConstant())
                return Int(Xcode.INT_CONSTANT, x.Type(), l.getInt() / r.getInt());
            if(l.isZeroConstant())
                return IntConstant(0); /* 0/r = 0 */
            if(r.isOneConstant())
                return l; /* l/1 = l */
            break;

        case F_POWER_EXPR:
            l = Reduce(x.left(), b);
            r = Reduce(x.right(), b);
            if(l.isIntConstant() && r.isIntConstant())
                return Int(Xcode.INT_CONSTANT, x.Type(), (int)Math.pow(l.getInt(), r.getInt()));
            if(l.isZeroConstant() || r.isZeroConstant())
                return IntConstant(0);
            if(l.isOneConstant() || r.isOneConstant())
                return l;
            break;
            
        default:
            return x; // not reduce
        }
        if(l != x.left() || r != x.right())
            return Xcons.List(x.Opcode(), x.Type(), l, r);
        else
            return x;
    }

    public static Xobject SymbolRef(Ident ident)
    {
        Xtype t = ident.Type();
        
        if(t == null) {
            throw new IllegalArgumentException("type of '" + ident.getName() + "' is not set.");
        }
        
        Xcode code;
        switch(t.getKind()) {
        case Xtype.ARRAY:
            code = Xcode.ARRAY_REF;
            break;
        case Xtype.FUNCTION:
            code = Xcode.FUNC_ADDR;
            t = Xtype.Pointer(t);
            break;
        default:
            code = Xcode.VAR;
            break;
        }
        return Xcons.Symbol(code, t, ident.getName(), ident.getScope());
    }

    public static Xobject arrayRef(Xtype type, Xobject arrayAddr, XobjList arrayRefList)
    {
        return Xcons.List(Xcode.ARRAY_REF, type, arrayAddr, arrayRefList);
    }
    
    public static Xobject CompoundStatement(Xobject id_list, Xobject decl_list, Xobject body)
    {
        return List(Xcode.COMPOUND_STATEMENT, id_list, decl_list, body);
    }

    public static Xobject CompoundStatement(Xobject body)
    {
        return CompoundStatement(IDList(), List(), body);
    }
    
    public static Xobject statementList()
    {
        return List(XmOption.isLanguageC() ? Xcode.LIST : Xcode.F_STATEMENT_LIST);
    }
    
    public static Xobject FstatementList()
    {
        return List(Xcode.F_STATEMENT_LIST);
    }
    
    public static Xobject FstatementList(Xobject ... body)
    {
        return List(Xcode.F_STATEMENT_LIST, body);
    }
    
    public static Xobject FvarRef(Ident id)
    {
        return List(Xcode.F_VAR_REF, id.getAddr());
    }
    
    public static Xobject FindexRange(Xobject lb, Xobject ub, Xobject step)
    {
        return Xcons.List(Xcode.F_INDEX_RANGE, lb, ub, step);
    }
    
    public static Xobject FindexRange(Xobject lb, Xobject ub)
    {
        return FindexRange(lb, ub, null);
    }
    
    public static Xobject FindexRangeOfAssumedShape()
    {
        return Xcons.List(Xcode.F_INDEX_RANGE, (Xobject)null, null, null, IntConstant(1));
    }

    public static Xobject FindexRangeOfAssumedShape(Xobject lb)
    {
        return FindexRange(lb, (Xobject)null);
    }
    
    public static Xobject FindexRangeOfAssumedSize()
    {
        return Xcons.List(Xcode.F_INDEX_RANGE, (Xobject)null, null, null, null, IntConstant(1));
    }
    
    public static Xobject FarrayIndex(Xobject idx)
    {
        return Xcons.List(Xcode.F_ARRAY_INDEX, idx);
    }
    
    public static Xobject FarrayRef(Xobject var, Xobject ... indices)
    {
        Xobject x = Xcons.List(Xcode.F_ARRAY_REF, var.Type().getRef(),
            Xcons.List(Xcode.F_VAR_REF, var));
        Xobject l = Xcons.List();
        
        if(var.Type().isFarray()) {
            int n = var.Type().getNumDimensions() - indices.length;
            for(int i = 0; i < n; ++i)
                l.add(Xcons.FindexRangeOfAssumedShape());
        }
        
        for(Xobject i : indices) {
            l.add(Xcons.List(Xcode.F_ARRAY_INDEX, i));
        }
        
        x.add(l);
        return x;
    }
    
    public static Xobject Fallocate(Xobject var, Xobject ... indices)
    {
        XobjList l = Xcons.List(indices);
        return Xcons.List(Xcode.F_ALLOCATE_STATEMENT, 
			  (Xobject)null, // no stat
			  Xcons.List(Xcons.List(Xcode.F_ALLOC, var, l)));
    }

    public static Xobject FallocateByList(Xobject var, XobjList l)
    {
      return Xcons.List(Xcode.F_ALLOCATE_STATEMENT,  (Xobject)null, // no stat
			Xcons.List(Xcons.List(Xcode.F_ALLOC, var, l)));
    }
    
    public static Xobject FpointerAssignment(Xobject ptr, Xobject tgt)
    {
        return Xcons.List(Xcode.F_POINTER_ASSIGN_STATEMENT, ptr, tgt);
    }
    
    public static Xobject FinterfaceFunctionDecl(Ident name, XobjList paramDecls)
    {
        return Xcons.List(Xcode.F_INTERFACE_DECL, (Xobject)null,
            IntConstant(0), IntConstant(0), Xcons.List(
                Xcons.List(Xcode.FUNCTION_DECL, name, null, null, paramDecls)));
    }

    public static Xobject StatementLabel(String label)
    {
	return Xcons.List(Xcode.STATEMENT_LABEL, Xcons.StringConstant(label));
    }
}
