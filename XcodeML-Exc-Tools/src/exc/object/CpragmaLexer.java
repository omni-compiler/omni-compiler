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
 * Base pragma lexer for C.
 */
public class CpragmaLexer extends PragmaLexer
{
    protected CpragmaLexer()
    {
        super();
    }
    
    public CpragmaLexer(XmXmObjToXobjectTranslator xm2xobjTranslator, XmObj context)
    {
        super(xm2xobjTranslator, context);
    }

    protected CpragmaLexer(PragmaLexer lexer)
    {
        super(lexer);
    }

    @Override
    protected Xcode pg_term_op(int pre)
    {
        switch(pre) {
        case 0:
            if(pg_tok == PG_OROR)
                return Xcode.LOG_OR_EXPR;
            break;
        case 1:
            if(pg_tok == PG_ANDAND)
                return Xcode.LOG_AND_EXPR;
            break;
        case 2:
            if(pg_tok == '|')
                return Xcode.BIT_OR_EXPR;
            break;
        case 3:
            if(pg_tok == '^')
                return Xcode.BIT_XOR_EXPR;
            break;
        case 4:
            if(pg_tok == '&')
                return Xcode.BIT_AND_EXPR;
            break;
        case 5:
            if(pg_tok == PG_EQEQ)
                return Xcode.LOG_EQ_EXPR;
            if(pg_tok == PG_NEQ)
                return Xcode.LOG_NEQ_EXPR;
            break;
        case 6:
            if(pg_tok == '>')
                return Xcode.LOG_GT_EXPR;
            if(pg_tok == '<')
                return Xcode.LOG_LT_EXPR;
            if(pg_tok == PG_GTEQ)
                return Xcode.LOG_GE_EXPR;
            if(pg_tok == PG_LTEQ)
                return Xcode.LOG_LE_EXPR;
            break;
        case 7:
            if(pg_tok == PG_LTLT)
                return Xcode.LSHIFT_EXPR;
            if(pg_tok == PG_GTGT)
                return Xcode.RSHIFT_EXPR;
            break;
        case 8:
            if(pg_tok == '+')
                return Xcode.PLUS_EXPR;
            if(pg_tok == '-')
                return Xcode.MINUS_EXPR;
            break;
        case 10:
            if(pg_tok == '*')
                return Xcode.MUL_EXPR;
            if(pg_tok == '/')
                return Xcode.DIV_EXPR;
            if(pg_tok == '%')
                return Xcode.MOD_EXPR;
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
            return Xtype.intType;
        case PLUS_EXPR:
        case MINUS_EXPR:
        case MUL_EXPR:
        case DIV_EXPR:
        case MOD_EXPR:
        case BIT_OR_EXPR:
        case BIT_XOR_EXPR:
        case BIT_AND_EXPR:
            return Xcons.ConversionIntegral(t1, t2);
        case LSHIFT_EXPR:
        case RSHIFT_EXPR:
            return t1;
        }
        
        XmLog.fatal(op.toString());
        return null;
    }

    protected Xobject pg_unary_expr() throws XmException
    {
        Xobject e;
        Xcode op;

        switch(pg_tok) {
        case '*':
            pg_get_token();
            if((e = pg_unary_expr()) == null) {
                return null;
            }
            op = Xcode.POINTER_REF;
            break;

        case '-':
            pg_get_token();
            if((e = pg_unary_expr()) == null) {
                return null;
            }
            op = Xcode.UNARY_MINUS_EXPR;
            break;

        case '!':
            pg_get_token();
            if((e = pg_unary_expr()) == null) {
                return null;
            }
            op = Xcode.LOG_NOT_EXPR;
            break;

        case '~':
            pg_get_token();
            if((e = pg_unary_expr()) == null) {
                return null;
            }
            op = Xcode.BIT_NOT_EXPR;
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
        case POINTER_REF:
            return t.getRef();
        case UNARY_MINUS_EXPR:
        case BIT_NOT_EXPR:
            return t;
        case LOG_NOT_EXPR:
            return Xtype.intType;
        }
        
        XmLog.fatal(op.toString());
        return null;
    }

    // process postfix expression
    private Xobject pg_factor_expr() throws XmException
    {
        Xobject e, ee, args;

        e = pg_primary_expr();

        while(true) {
            switch(pg_tok) {
            case '[':
                pg_get_token();
                if((ee = pg_term_expr(0)) == null) {
                    return null;
                }
                if(pg_tok != ']') {
                    error("syntax error near ']' in pragma expression");
                    return null; // syntax error
                }
                e = Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, e, ee));
                pg_get_token();
                break;

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

            case '.':
                pg_get_token();
                if(pg_tok != PG_IDENT) {
                    error("syntax error in pragma expression");
                    return null;
                }
                ee = pg_primary_expr();
                e = Xcons.List(Xcode.MEMBER_REF,
                    e.Type().getMemberType(ee.getName()), Xcons.AddrOf(e), ee);
                break;

            case PG_STREF:
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
                pg_tok_buf = cp.toString();
                pg_tok = PG_IDENT; // identifier
                return;
            }

            if(Character.isDigit(pg_cp_char())) {
                pg_tok_val = pg_parse_number();
                pg_tok = PG_CONST; // const
                return;
            }

            // single charactor
            c = pg_cp_char();
            switch(c) {
            case 0:
            case '+':
            case '*':
            case '^':
            case '%':
            case ')':
            case '(':
            case ',':
            case '[':
            case ']':
            case ':':
                pg_tok = pg_cp_char_incr();
                return;
            case '-':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '>') { // "->"
                    pg_cp++;
                    pg_tok = PG_STREF;
                }
                return;
            case '/':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '*') { // comment
                    while(pg_cp < pg_cp_str.length()) {
                        if(pg_cp_char_incr() == '*' && pg_cp_char() == '/') {
                            pg_cp++;
                            continue;
                        }
                    }
                    error("bad comment in pragma");
                    pg_tok = PG_NONE;
                }
                return;

            case '|':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '|') { // "||"
                    pg_cp++;
                    pg_tok = PG_OROR;
                }
                return;

            case '&':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '&') { // "&&"
                    pg_cp++;
                    pg_tok = PG_ANDAND;
                }
                return;

            case '!':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '=') { // "!="
                    pg_cp++;
                    pg_tok = PG_NEQ;
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
                } else if(pg_cp_char() == '<') { // "<<"
                    pg_cp++;
                    pg_tok = PG_LTLT;
                }
                return;

            case '>':
                pg_tok = pg_cp_char_incr();
                if(pg_cp_char() == '=') { // ">="
                    pg_cp++;
                    pg_tok = PG_GTEQ;
                } else if(pg_cp_char() == '>') { // ">>"
                    pg_cp++;
                    pg_tok = PG_GTGT;
                }
                return;

            case '"':
                pg_cp++;
                pg_tok_val = pg_parse_string_constant();
                pg_tok = PG_CONST;
                error_unknown_char();
                return;

            case '\'':
                pg_cp++;
                pg_tok_val = pg_parse_char_constant();
                pg_tok = PG_CONST;
                error_unknown_char();
                return;
            }

            error_unknown_char();
            return;
        }
    }

    private Xobject pg_parse_string_constant()
    {
        int i;
        char ch, val;
        StringBuilder cp = new StringBuilder(32);

        do {
            ch = pg_cp_char_incr();
            
          loop:
            while(ch != '"') {
                switch(ch) {
                case '\\': /* escape */
                    if(cp.length() >= MAX_TOKEN_LEN) {
                        XmLog.fatal("too long string in pragma");
                        break;
                    }
                    switch((ch = pg_cp_char_incr())) {
                    /* escaped char(n,r,...) */
                    case '\0':
                        XmLog.error("unexpected end of line in pragma");
                        break loop;
                    case 't':
                        ch = '\t';
                        break;
                    case 'b':
                        ch = '\b';
                        break;
                    case 'f':
                        ch = '\f';
                        break;
                    case 'n':
                        ch = '\n';
                        break;
                    case 'r':
                        ch = '\r';
                        break;
                    // case 'v': ch = '\v'; break;
                    // case 'a': ch = '\a'; break;
                    case '\\':
                        ch = '\\';
                        break;
                    case '0':
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                        val = 0;
                        for(i = 0; i < 3; i++) {
                            if(!(ch >= '0' && ch <= '7')) {
                                --pg_cp;
                                break;
                            }
                            val = (char)(val * 8 + ch - '0');
                            ch = pg_cp_char_incr();
                        }
                        ch = val;
                    }
                    cp.append(ch);
                    break;

                default:
                    cp.append(ch);
                    break;
                }

                if(cp.length() >= MAX_TOKEN_LEN) {
                    XmLog.fatal("too long string");
                    break;
                }
                ch = pg_cp_char_incr();
            }

            do {
                ch = pg_cp_char_incr();
            } while(Character.isSpaceChar(ch));
        } while(ch == '"');

        --pg_cp;
        pg_tok_buf = cp.toString();

        return Xcons.StringConstant(pg_tok_buf);
    }

    private Xobject pg_parse_char_constant()
    {
        char ch;
        int value = 0;
        StringBuilder cp = new StringBuilder(32);

        ch = pg_cp_char_incr();

        switch(ch) {
        case '\0':
            error("unexpected end of line in pragma");
            break;

        case '\n':
            error("newline in char constant");
            break;

        case '\\': /* escape sequence */
            /* '\': \nnn and \xNN are default except top 2 chars */
            ch = pg_cp_char_incr();
            switch(ch) {
            case 'x': /* hex '\xhh', at most 2 */
                ch = pg_cp_char_incr();
                if(!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'))) {
                    XmLog.warning("\\x must follow hex digit");
                    break;
                }
                cp.append(ch);
                value = 0xf & ch;
                ch = pg_cp_char_incr();
                if(!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'))) {
                    break;
                }
                cp.append(ch);
                value = (value << 4) | (0xf & ch);
                break;

            case '0': /* oct '\ooo', at most 3 */
                ch = pg_cp_char_incr();
                if(ch == '"') { /* '\0' */
                    --pg_cp;
                    break;
                }
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
                value = 0x7 & ch;
                ch = pg_cp_char_incr();
                if(!(ch >= '0' && ch <= '7'))
                    break;
                cp.append(ch);
                value = (value << 3) | (0x7 & ch);
                ch = pg_cp_char_incr();
                if(!(ch >= '0' && ch <= '7'))
                    break;
                cp.append(ch);
                value = (value << 3) | (0x7 & ch);
                break;

            case 'b':
                value = '\b';
                break;
            case 'f':
                value = '\f';
                break;
            case 'n':
                value = '\n';
                break;
            case 'r':
                value = '\r';
                break;
            case 't':
                value = '\t';
                break;
            case '\\':
                value = '\\';
                break;
            case '\'':
                value = '\'';
                break;
            case '"':
                value = '\"';
                break;
            /*
            case 'v': value = '\v'; break;
            case 'a': value = '\a'; break;
            case '?': value = '\?'; break;
             */
            default:
                XmLog.warning("unknown escape sequence");
                break;
            }
            cp.append(ch);
            break;

        default:
            cp.append(ch);
            value = ch;
            break;
        }

        ch = pg_cp_char_incr();
        if(ch != '\'')
            XmLog.fatal("too many characters");

        if(cp.length() == 0)
            XmLog.error("empty character constant");
        
        pg_tok_buf = cp.toString();

        return Xcons.IntConstant(value);
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
            boolean makeConst = false;
            if ( ch == 'x' || ch == 'X' ) { /* HEX */
                radix = 16;
                for(;;) {
                    ch = pg_cp_char_incr();
                    if(!isxdigit(ch)) {
                        makeConst = true;
                        break;
                    }
                    cp.append(ch);
                }
            }
            
            if(makeConst)
                break;

            if(ch == '.') {
                intOrFloat = false;
                break;
            }
            
            if(!(ch >= '0' && ch <= '7'))
                break;
            
            /* octal */
            radix = 8;
            for(;;) {
                cp.append(ch);
                ch = pg_cp_char_incr();
                if(!(ch >= '0' && ch <= '7'))
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
                if(ch == 'L') {
                    ch = pg_cp_char_incr();
                    if(ch == 'L') {
                        return Xcons.LongLongConstant(value, h_value);
                    }
                }
                --pg_cp;
                if(h_value != 0)
                    XmLog.warning("integer constant out of range");
                return Xcons.IntConstant(value);
            } else {
                /* else decimal or floating */
                while(Character.isDigit(ch)) {
                    cp.append(ch);
                    ch = pg_cp_char_incr();
                }
                if(ch != '.' && ch != 'e' && ch != 'E') {
                    intOrFloat = true;
                    continue;
                }
                /* floating */
                if(ch == '.') {
                    cp.append(ch);
                    /* reading floating */
                    ch = pg_cp_char_incr();
                    while(Character.isDigit(ch)) {
                        cp.append(ch);
                        ch = pg_cp_char_incr();
                    }
                }
            
                if(ch == 'e' || ch == 'E') {
                    cp.append('e');
                    ch = pg_cp_char_incr();
                    if(ch == '+' || ch == '-') {
                        cp.append(ch);
                        ch = pg_cp_char_incr();
                    }
                    while(Character.isDigit(ch)){
                        cp.append(ch);
                        ch = pg_cp_char_incr();
                    }
                }
            
                --pg_cp;
                pg_tok_buf = cp.toString();

                return Xcons.Float(Xcode.FLOAT_CONSTANT,
                    Xtype.floatType,
                    Double.parseDouble(pg_tok_buf));
            }
        }
    }
}
