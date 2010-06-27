/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import exc.xcodeml.XmSymbol;
import exc.xcodeml.XmSymbolUtil;
import xcodeml.XmException;
import xcodeml.XmObj;
import xcodeml.util.XmLog;
import xcodeml.util.XmXmObjToXobjectTranslator;

/**
 * Base pragma lexer for Fortran.
 */
public class FpragmaLexer extends PragmaLexer
{
    protected FpragmaLexer()
    {
        super();
    }
    
    public FpragmaLexer(XmXmObjToXobjectTranslator xm2xobjTranslator, XmObj context)
    {
        super(xm2xobjTranslator, context);
    }

    protected FpragmaLexer(PragmaLexer lexer)
    {
        super(lexer);
    }

    @Override
    protected Xcode pg_term_op(int pre)
    {
        switch(pre) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            if(pg_tok == PG_FEQ)
                return Xcode.LOG_EQ_EXPR;
            if(pg_tok == PG_FNEQ)
                return Xcode.LOG_NEQ_EXPR;
            if(pg_tok == PG_FEQV)
                return Xcode.F_LOG_EQV_EXPR;
            if(pg_tok == PG_FNEQV)
                return Xcode.F_LOG_NEQV_EXPR;
            if(pg_tok == PG_FAND)
                return Xcode.LOG_AND_EXPR;
            if(pg_tok == PG_FOR)
                return Xcode.LOG_OR_EXPR;
            break;
        case 5:
            if(pg_tok == '>')
                return Xcode.LOG_GT_EXPR;
            if(pg_tok == '<')
                return Xcode.LOG_LT_EXPR;
            if(pg_tok == PG_GTEQ)
                return Xcode.LOG_GE_EXPR;
            if(pg_tok == PG_LTEQ)
                return Xcode.LOG_LE_EXPR;
            break;
        case 6:
            if(pg_tok == PG_CONCAT)
                return Xcode.F_CONCAT_EXPR;
            break;
        case 7:
            if(pg_tok == '+')
                return Xcode.PLUS_EXPR;
            if(pg_tok == '-')
                return Xcode.MINUS_EXPR;
            break;
        case 8:
            if(pg_tok == '*')
                return Xcode.MUL_EXPR;
            if(pg_tok == '/')
                return Xcode.DIV_EXPR;
            break;
        case 10:
            if(pg_tok == PG_POW)
                return Xcode.F_POWER_EXPR;
            break;
        }
        
        return null;
    }

    @Override
    protected Xtype getBinaryOpType(Xcode op, Xtype t1, Xtype t2)
    {
        switch(op) {
        case LOG_OR_EXPR:
        case LOG_AND_EXPR:
        case LOG_EQ_EXPR:
        case LOG_NEQ_EXPR:
        case LOG_GT_EXPR:
        case LOG_LT_EXPR:
        case LOG_GE_EXPR:
        case LOG_LE_EXPR:
        case F_LOG_EQV_EXPR:
        case F_LOG_NEQV_EXPR:
            return Xtype.FlogicalType;
        case PLUS_EXPR:
        case MINUS_EXPR:
        case MUL_EXPR:
        case DIV_EXPR:
            return Xcons.ConversionIntegral(t1, t2);
        case F_POWER_EXPR:
            return t1;
        case F_CONCAT_EXPR:
            return Xtype.FcharacterWithLenType;
        }
        
        XmLog.fatal(op.toString());
        return null;
    }

    protected Xobject pg_unary_expr() throws XmException
    {
        Xobject e;
        Xcode op;

        switch(pg_tok) {
        case '-':
            pg_get_token();
            if((e = pg_unary_expr()) == null) {
                return null;
            }
            op = Xcode.UNARY_MINUS_EXPR;
            break;

        case PG_FNOT:
            pg_get_token();
            if((e = pg_unary_expr()) == null) {
                return null;
            }
            op = Xcode.LOG_NOT_EXPR;
            break;
            
        default:
            return pg_factor_expr();
        }
        
        Xtype t = getUnaryOpType(op, e.Type());
        return Xcons.List(op, t, e);
    }
    
    private Xtype getUnaryOpType(Xcode op, Xtype t)
    {
        switch(op) {
        case UNARY_MINUS_EXPR:
            return t;
        case LOG_NOT_EXPR:
            return Xtype.FlogicalType;
        }
        
        XmLog.fatal(op.toString());
        return null;
    }

    // process postfix expression
    private Xobject pg_factor_expr() throws XmException
    {
        Xobject e, ee, args;

        e = pg_primary_expr();
        
        if(has_error())
            return null;

        while(true) {
            switch(pg_tok) {
            case '(':
                pg_get_token();
                args = Xcons.List();
                if(pg_tok != ')') {
                    while(true) {
                        if((ee = pg_term_expr(0)) == null) {
                            return null;
                        }
                        args.add(ee);
                        if(pg_tok != ',') {
                            break;
                        }
                        pg_get_token();
                    }
                }

                if(pg_tok == ')') {
                    pg_get_token();
                    e = Xcons.functionCall(e, args);
                    break;
                }
                error("syntax error in pragma expression");
                return null;

            case '%':
                pg_get_token();
                if(pg_tok != PG_IDENT) {
                    error("syntax error in pragma expression");
                    return null;
                }
                ee = pg_primary_expr();
                e = Xcons.List(Xcode.MEMBER_REF,
                    e.Type().getMemberType(ee.getName()), e, ee);
                break;

            default:
                return e;
            }
        }
    }
    
    private Xobject pg_primary_expr() throws XmException
    {
        Xobject e = null;
        XmSymbol sym;

        switch(pg_tok) {
        case PG_IDENT:
            pg_get_token();
            sym = XmSymbolUtil.lookupSymbol(context, pg_tok_buf);
            if(sym != null && sym.isIdent()) {
                e = (Ident)xm2xobjTranslator.translate(sym.getXmObj());
                e = Xcons.SymbolRef((Ident)e);
            } else if(!is_lang_c() && pg_tok == '(') {
                //may be intrinsic call
                e = Xcons.Ident(pg_tok_buf, null, Xtype.Function(Xtype.FnumericAllType),
                    Xcons.Symbol(Xcode.FUNC_ADDR, Xtype.Function(Xtype.FnumericAllType), pg_tok_buf), null);
            } else {
                error("bad symbol in expression of pragma args");
                return null;
            }
            return e;

        case '(':
            pg_get_token();
            if((e = pg_term_expr(0)) == null) {
                return null;
            }
            if(pg_tok != ')') {
                error("mismatch paren in pragma expression");
                return null;
            }
            pg_get_token();
            return e;

        case PG_CONST:
            e = pg_tok_val;
            pg_get_token();
            return e;

        default:
            error("syntax error in pragma expression");
            return null;
        }
    }

    public void pg_get_token()
    {
        while(true) {
            // skip white space
            skipSpace();
            char c = pg_cp_char();
            if(c == '_' || Character.isLetter(c)) {
                StringBuilder cp = new StringBuilder(128);
                cp.append(c);
                pg_cp++;
                while(true) {
                    c = pg_cp_char();
                    if(c == '_' || Character.isLetterOrDigit(c)) {
                        cp.append(c);
                        ++pg_cp;
                    } else {
                        break;
                    }
                }
                pg_tok_buf = cp.toString().toLowerCase();
                pg_tok = PG_IDENT; // identifier
                return;
            }

            if(Character.isDigit(c)) {
                pg_tok_val = pg_parse_number();
                pg_tok = PG_CONST; // const
                return;
            }

            // single charactor
            c = pg_cp_char();
            switch(c) {
            case 0:
            case '+':
            case '-':
            case '%':
            case ')':
            case '(':
            case ',':
            case ':':
                pg_tok = pg_cp_char_incr();
                return;
            case '.':
                pg_tok = pg_parse_dot_kw_dot();
                return;
            
            case '*':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '*') { // "**"
                    pg_cp++;
                    pg_tok = PG_POW;
                }
                return;
                
            case '/':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '=') { // "/="
                    pg_cp++;
                    pg_tok = PG_NEQ;
                } else if(pg_cp_char() == '/') { // "//"
                    pg_cp++;
                    pg_tok = PG_CONCAT;
                }
                return;

            case '=':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '=') { // "=="
                    pg_cp++;
                    pg_tok = PG_EQEQ;
                }
                return;

            case '<':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '=') { // "<="
                    pg_cp++;
                    pg_tok = PG_LTEQ;
                }
                return;

            case '>':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '=') { // ">="
                    pg_cp++;
                    pg_tok = PG_GTEQ;
                }
                return;

            case '"':
            case '\'':
                pg_cp++;
                pg_tok_val = pg_parse_string_constant(c);
                pg_tok = PG_CONST;
                return;
            }

            error_unknown_char();
            return;
        }
    }
    
    private char pg_parse_dot_kw_dot()
    {
        StringBuilder buf = new StringBuilder(5);
        boolean toolong = false;
        ++pg_cp;
        
        while((pg_tok = pg_cp_char_incr()) != '.' && pg_tok != 0) {
            buf.append(pg_tok);
            if(buf.length() > 5) {
                toolong = true;
                break;
            }
        }
        
        if(toolong || pg_tok == 0) {
            pg_cp -= buf.length() + 2;
            return pg_cp_char();
        }
        
        String op = buf.toString().toLowerCase();
        if(op.equals("and")) {
            return PG_FAND;
        } else if(op.equals("or")) {
            return PG_FOR;
        } else if(op.equals("eq")) {
            return PG_FEQ;
        } else if(op.equals("neq")) {
            return PG_FNEQ;
        } else if(op.equals("eqv")) {
            return PG_FEQV;
        } else if(op.equals("neqv")) {
            return PG_FNEQV;
        } else if(op.equals("lt")) {
            return PG_FLT;
        } else if(op.equals("le")) {
            return PG_FLEQ;
        } else if(op.equals("gt")) {
            return PG_FGT;
        } else if(op.equals("ge")) {
            return PG_FGEQ;
        } else if(op.equals("not")) {
            return PG_FNOT;
        } else if(op.equals("true")) {
            pg_tok_val = Xcons.FlogicalConstant(true);
            return PG_CONST;
        } else if(op.equals("false")) {
            pg_tok_val = Xcons.FlogicalConstant(false);
            return PG_CONST;
        }
        
        pg_cp -= buf.length() + 2;
        return pg_cp_char();
    }

    private Xobject pg_parse_string_constant(char quote)
    {
        char ch = 0, ch2;
        StringBuilder cp = new StringBuilder(32);

        while(true) {
            ch = pg_cp_char_incr();
            if(ch == quote) {
                ch2 = pg_cp_char_incr();
                if(ch2 == quote)
                    cp.append(ch2);
                else {
                    --pg_cp;
                    break;
                }
            } else if(ch == 0) {
                break;
            } else {
                cp.append(ch);
            }
            
            if(cp.length() >= MAX_TOKEN_LEN) {
                error("too long string");
                break;
            }
        }
        
        if(ch == 0) {
            error("unexpected end of line in pragma");
        }
        
        pg_tok_buf = cp.toString();

        return Xcons.FcharacterConstant(Xtype.FcharacterType, pg_tok_buf, null);
    }

    private Xobject pg_parse_number()
    {
        StringBuilder cp = new StringBuilder(16);
        char ch;
        int value, h_value;
        int radix = 10;
        boolean intOrFloat = true;
    
        ch = pg_cp_char_incr();

        do {
            if(ch != '0') {
                intOrFloat = false;
                break;
            }
            
            ch = pg_cp_char_incr();

            if(ch == '.') {
                intOrFloat = false;
                break;
            }
           
        } while(false);
        
        while(true) {
            if(intOrFloat) {
                pg_tok_buf = cp.toString();
                int[] hlvalues = string_to_integer(pg_tok_buf, radix);
                if(hlvalues == null)
                    return null;
                h_value = hlvalues[0];
                value = hlvalues[1];
                --pg_cp;
                if(h_value != 0)
                    return Xcons.LongLongConstant(value, h_value);
                return Xcons.IntConstant(value);
            } else {
                /* else decimal or floating */
                while(Character.isDigit(ch)) {
                    cp.append(ch);
                    ch = pg_cp_char_incr();
                }
                switch(ch) {
                case '.':
                    cp.append(ch);
                    /* reading floating */
                    ch = pg_cp_char_incr();
                    while(Character.isDigit(ch)) {
                        cp.append(ch);
                        ch = pg_cp_char_incr();
                    }
                    break;
                case 'e': case 'E':
                case 'd': case 'D':
                    cp.append(ch);
                    ch = pg_cp_char_incr();
                    if(ch == '+' || ch == '-') {
                        cp.append(ch);
                        ch = pg_cp_char_incr();
                    }
                    while(Character.isDigit(ch)){
                        cp.append(ch);
                        ch = pg_cp_char_incr();
                    }
                    break;
                default:
                    intOrFloat = true;
                    continue;
                }
            
                --pg_cp;
                pg_tok_buf = cp.toString();
                return Xcons.FloatConstant(Xtype.floatType, pg_tok_buf, null);
            }
        }
    }
}
