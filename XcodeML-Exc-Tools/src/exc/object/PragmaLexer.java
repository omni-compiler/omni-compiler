/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import xcodeml.XmException;
import xcodeml.XmObj;
import xcodeml.util.XmLog;
import xcodeml.util.XmOption;
import xcodeml.util.XmXmObjToXobjectTranslator;

import exc.openmp.OMPpragmaLexer;
import exc.xcalablemp.XMPpragmaSyntaxAnalyzer;

/**
 * Base pragma lexer.
 */
public abstract class PragmaLexer
{
    protected static final int MAX_TOKEN_LEN = 128;
    
    /* token value */
    public static final char PG_NONE      = 0;
    public static final char PG_ANDAND    = 'A';
    public static final char PG_FAND      = PG_ANDAND;
    public static final char PG_CONCAT    = 'c';
    public static final char PG_ERR       = 'E';
    public static final char PG_IDENT     = 'I';
    public static final char PG_CONST     = 'N';
    public static final char PG_FNEQV     = 'n';
    public static final char PG_OROR      = 'O';
    public static final char PG_FOR       = PG_OROR;
    public static final char PG_EQEQ      = 'P';
    public static final char PG_FEQ       = PG_EQEQ;
    public static final char PG_POW       = 'p';
    public static final char PG_NEQ       = 'Q';
    public static final char PG_FNEQ      = PG_NEQ;
    public static final char PG_FEQV      = 'q';
    public static final char PG_STREF     = 'S';
    public static final char PG_LTEQ      = 'T';
    public static final char PG_FLEQ      = PG_LTEQ;
    public static final char PG_FNOT      = 't';
    public static final char PG_GTEQ      = 'U';
    public static final char PG_FGEQ      = PG_GTEQ;
    public static final char PG_FLT       = '<';
    public static final char PG_FGT       = '>';
    public static final char PG_LTLT      = 'V';
    public static final char PG_GTGT      = 'W';

    /** character index in pg_cp_str */
    protected int pg_cp;
    /** parsing target string */
    protected String pg_cp_str;
    /** current token code */
    protected char pg_tok;
    /** current token of keyword */
    protected String pg_tok_buf;
    /** current token of constant */
    protected Xobject pg_tok_val;
    /** error message */
    private String error;
    /** parsing XmObj */
    protected XmObj context;
    /** translator */
    protected XmXmObjToXobjectTranslator xm2xobjTranslator;

    protected PragmaLexer()
    {
    }
    
    public PragmaLexer(XmXmObjToXobjectTranslator xm2xobjTranslator, XmObj context)
    {
        this.xm2xobjTranslator = xm2xobjTranslator;
        this.context = context;
    }

    protected PragmaLexer(PragmaLexer lexer)
    {
        pg_cp               = lexer.pg_cp;
        pg_cp_str           = lexer.pg_cp_str;
        context             = lexer.context;
        xm2xobjTranslator   = lexer.xm2xobjTranslator;
    }
    
    /** get node reprsents current context. */
    public final XmObj getContext()
    {
        return context;
    }
    
    public final XmXmObjToXobjectTranslator getXmObjToXobjectTranslator()
    {
        return xm2xobjTranslator;
    }
    
    public final Xobject pg_parse_expr() throws XmException
    {
        return pg_term_expr(0);
    }
    
    /** get current token. */
    public final char pg_tok()
    {
        return pg_tok;
    }
    
    /** get current token string. */
    public final String pg_tok_buf()
    {
        return pg_tok_buf;
    }

    /** process terminal node. */
    protected abstract Xcode pg_term_op(int pre);
    
    /** process binary operator node. */
    protected abstract Xtype getBinaryOpType(Xcode op, Xtype t1, Xtype t2);
    
    /** process unary operator node. */
    protected abstract Xobject pg_unary_expr() throws XmException;

    /** process and set token string */
    public abstract void pg_get_token();

    /** process expression. */
    protected final Xobject pg_term_expr(int pre) throws XmException
    {
        Xcode op = Xcode.LIST;
        Xobject e, ee;

        if(pre > 10) {
            return pg_unary_expr();
        }

        if((e = pg_term_expr(pre + 1)) == null) {
            return null;
        }

        while(true) {
            Xcode top = pg_term_op(pre);
            if(top == null)
                return e;
            op = top;
            pg_get_token();
            if((ee = pg_term_expr(pre + 1)) == null) {
                return null;
            }
            Xtype t = getBinaryOpType(op, e.Type(), ee.Type());
            e = Xcons.List(op, t, e, ee);
        }
    }
    
    /** return if the character is hexical number. */
    protected final boolean isxdigit(char c)
    {
        return ('0' <= c && c <= '9') || ('a' <= c && c <='f') || ('A' <= c && c <= 'F');
    }
    
    /** get int value from string. */
    protected final int[] string_to_integer(String s, int radix)
    {
        long l;
        
        if(s.length() == 0)
            return new int[] { 0, 0 };
        
        try {
            l = Long.parseLong(s, radix);
        } catch(NumberFormatException e) {
            XmLog.error("invalid number constant in pragma : " + s);
            return null;
        }
        return new int[] { (int)((l >> 32) & 0xFFFFFFFF), (int)(l & 0xFFFFFFFF) };
    }

    /** set error. */
    public final void error(String msg)
    {
        error = msg;
    }
    
    /** return if error set. */
    public final boolean has_error()
    {
        return (error != null);
    }

    /** get character in current position. */
    protected final char pg_cp_char()
    {
        return (pg_cp < pg_cp_str.length()) ? pg_cp_str.charAt(pg_cp) : '\0';
    }

    /** get character in current position and increment position. */
    protected final char pg_cp_char_incr()
    {
        char c = pg_cp_char();
        ++pg_cp;
        return c;
    }

    /** move current position after space characters. */
    protected final void skipSpace()
    {
        while(Character.isSpaceChar(pg_cp_char())) {
            ++pg_cp;
        }
    }

    /** set error of 'unknown char' */
    protected final void error_unknown_char()
    {
        pg_tok = PG_ERR;
        error("unknown character '" + pg_cp_char() + "' in pragma");
    }

    public static class Result
    {
        public final Xobject xobject;
        public final String error_message;

        public Result(PragmaLexer lexer, Xobject obj)
        {
            this.xobject = obj;
            this.error_message = lexer.error;
        }
    };

    public final boolean pg_is_ident()
    {
        return (pg_tok == PG_IDENT);
    }

    /** return if current token is specified identifier. */
    public final boolean pg_is_ident(String name)
    {
        return (pg_tok == PG_IDENT && name.equalsIgnoreCase(pg_tok_buf));
    }
    
    /** do lexical analyze and return tokens. */
    public final Result lex(String line) throws XmException
    {
        pg_cp_str = line;
        pg_cp = 0;
        pg_get_token();
        
        Result r = null;
        
        if(pg_tok == PG_IDENT) {
            if(pg_is_ident("omp")) {
                if(XmOption.isOpenMP())
                    r = new OMPpragmaLexer(this).continueLex();
            }
            else if(pg_is_ident("xmp")) {
                if(XmOption.isXcalableMP())
                    r = new XMPpragmaSyntaxAnalyzer(this).continueLex();
            }
        }

        // external pramga
        if(r == null)
            r = new Result(this, Xcons.List(Xcode.PRAGMA_LINE, Xcons.String(line)));
        
        return r;
    }

    /** return if target is C language. */
    public boolean is_lang_c()
    {
        return XmOption.isLanguageC();
    }
}
