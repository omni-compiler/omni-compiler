/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-pragma.c
 */

#include <sys/param.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

#include "c-pragma.h"
#include "c-parser.h"
#include "c-const.h"
#include "c-option.h"

#include "c-omp.h"
#include "c-xmp.h"
#include "c-acc.h"

//! Pointer to parsing string
char    *pg_cp;
//! current token
char    pg_tok;
//! token buffer
char    pg_tok_buf[MAX_NAME_SIZ];

//! has error at parsing
int     pg_hasError;
//! token value
CExpr*  pg_tok_val = NULL;

PRIVATE_STATIC CExpr* pg_parse_number(void);
PRIVATE_STATIC CExpr* pg_parse_string_constant(void);
PRIVATE_STATIC CExpr* pg_parse_char_constant(void);

PRIVATE_STATIC CExpr* pg_term_expr(int pre);
PRIVATE_STATIC CExpr* pg_factor_expr(void);
PRIVATE_STATIC CExpr* pg_unary_expr(void);
PRIVATE_STATIC CExpr* pg_primary_expr(void);

/**
 * \brief
 * judge c is token separator
 *
 * @param c
 *      character
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
is_token_separator(char c)
{
    return(c == 0 || c == ' ' || c == '\t' || c == '\r' || c == '\n');
}


/**
 * \brief
 * judge c is token separator.
 * treat '(' as token separator.
 *
 * @param c
 *      character
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
is_token_separatorP(char c)
{
    return is_token_separator(c) || c == '(';
}


/**
 * \brief
 * skip token separator and return char pointer
 *
 * @param p
 *      char pointer
 * @return
 *      skipped char pointer
 */
char*
lexSkipSpace(char *p)
{
    while(is_token_separator(*p) && *p) ++p;
    return p;
}


/**
 * \brief
 * skip token separator and return char pointer.
 * treat '(' as token separator.
 *
 * @param p
 *      char pointer
 * @return
 *      skipped char pointer
 */
char*
lexSkipSpaceP(char *p)
{
    while(is_token_separatorP(*p) && *p) ++p;
    return p;
}


/**
 * \brief
 * skip token and return char pointer
 *
 * @param p
 *      char pointer
 * @return
 *      skipped char pointer
 */
char*
lexSkipWord(char *p)
{
    while(is_token_separator(*p) == 0) ++p;
    return p;
}


/**
 * \brief
 * skip token and return char pointer.
 * treat '(' as token separator.
 *
 * @param p
 *      char pointer
 * @return
 *      skipped char pointer
 */
char*
lexSkipWordP(char *p)
{
    while(is_token_separatorP(*p) == 0) ++p;
    return p;
}


/**
 * \brief
 * judge s1 and s2 equals as token
 *
 * @param s1
 *      token 1
 * @param s2
 *      token 2
 * @param sepByParenthesis
 *      set to 1 for treating '(' as token separator
 * @return
 *      0:no, 1:yes
 *
 */
PRIVATE_STATIC int
equals_token0(const char *s1, const char *s2, int sepByParenthesis)
{
    int len = 0;
    const char *p = s1;
    while(p) {
        if((sepByParenthesis ? is_token_separatorP(*p) : is_token_separator(*p)))
            break;
        ++p;
        ++len;
    }

    if(strncmp(s1, s2, len) != 0)
        return 0;

    return (sepByParenthesis ?
        is_token_separatorP(s2[len]) : is_token_separator(s2[len]));
}


/**
 * \brief
 * judge s1 and s2 equals as token.
 * treat '(' as token separator.
 *
 * @param s1
 *      token 1
 * @param s2
 *      token 2
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
equals_tokenP(const char *s1, const char *s2)
{
    return equals_token0(s1, s2, 1);
}


/**
 * \brief
 * get pragma kind
 *
 * @param p
 *      char pointer
 * @return
 *      pragma kind
 */
CPragmaKind
getPragmaKind(char *p)
{
    extern int s_useXMP;
    extern int s_useACC;

    /* after '#directive[space]' */
    p = lexSkipSpace(p);
    p = lexSkipWord(p);
    p = lexSkipSpace(p);

    if(equals_tokenP("pack", p))
        return PK_PACK;
    else if(equals_tokenP("omp", p)) /* OpenMP */
        return PK_OMP;
    else if(s_useXMP && equals_tokenP("xmp", p)) /* XcalableMP */
        return PK_XMP;
    else if(s_useACC && equals_tokenP("acc", p)) /* OpenACC */
        return PK_ACC;
    else
        return PK_NOT_PARSABLE;
}


/**
 * \brief
 * add error for syntax error
 */
PRIVATE_STATIC void 
addSyntaxErrorInExpression()
{
    addError(NULL, CERR_052);
}


/**
 * \brief
 * add error for syntax error
 *
 * @param s
 *      message argument
 */
PRIVATE_STATIC void 
addSyntaxErrorNearInExpression(const char *s)
{
    addError(NULL, CERR_053, s);
}


/**
 * \brief
 * parse token
 */
void
pg_get_token()
{
    char *cp;

  again:
    cp = pg_tok_buf;

    while(isspace((int)*pg_cp)) pg_cp++;

    if(*pg_cp == '_' || isalpha((int)*pg_cp)){
        *cp++ = *pg_cp++;
        while(isalnum((int)*pg_cp) || *pg_cp == '_') {
            *cp++= *pg_cp++;
        }
	*cp = '\0';
        pg_tok_val = (CExpr*)
	    allocExprOfSymbol(EC_IDENT,ccol_strdup(pg_tok_buf, MAX_NAME_SIZ));
        pg_tok = PG_IDENT;        /* identifier */
        return;
    }

    if(isdigit((int)*pg_cp)){
        pg_tok_val = pg_parse_number();
        pg_tok = PG_CONST;                /* number */
        return;
    }

    /* single charactor */
    switch(*pg_cp){
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
        pg_tok = *pg_cp++;
        return;
    case '-':
        pg_tok = *pg_cp++;
        if(*pg_cp == '>'){
            pg_cp++;
            pg_tok = PG_STREF;
        } 
        return;
    case '/':
        pg_tok = *pg_cp++;
        if(*pg_cp == '*'){   /* comment */
            while(*pg_cp != 0){
                if(*pg_cp++ == '*'){
                    if(*pg_cp == '/'){
                        pg_cp++;
                        goto again;
                    }
                }
            }
            addError(NULL, CERR_054);
            pg_tok = 0;
        }
        return;
    case ':':
        pg_tok = *pg_cp++;
        if(*pg_cp == ':'){
            pg_cp++;
            pg_tok = PG_COL2;
        } 
        return;
    case '|':
        pg_tok = *pg_cp++;
        if(*pg_cp == '|'){
            pg_cp++;
            pg_tok = PG_OROR;
        } 
        return;
    case '&':
        pg_tok = *pg_cp++;
        if(*pg_cp == '&'){
            pg_cp++;
            pg_tok = PG_ANDAND;
        } 
        return;
    case '!':
        pg_tok = *pg_cp++;
        if(*pg_cp == '='){
            pg_cp++;
            pg_tok = PG_NEQ;
            return;
        }
        return;
    case '=':
        pg_tok = *pg_cp++;
        if(*pg_cp == '='){
            pg_cp++;
            pg_tok = PG_EQEQ;
            return;
        } 
        return;
    case '<':
        pg_tok = *pg_cp++;
        if(*pg_cp == '='){
            pg_cp++;
            pg_tok = PG_LTEQ;
        }
        else if(*pg_cp == '<'){
            pg_cp++;
            pg_tok = PG_LTLT;
        }
        return;
    case '>':
        pg_tok = *pg_cp++;
        if(*pg_cp == '='){
            pg_cp++;
            pg_tok = PG_GTEQ;
        }
        else if(*pg_cp == '>'){
            pg_cp++;
            pg_tok = PG_GTGT;
        }
        return;

    case '"':
        pg_cp++;
        pg_tok_val = pg_parse_string_constant();
        pg_tok = PG_CONST;
        return;

    case '\'':
        pg_cp++;
        pg_tok_val = pg_parse_char_constant();
        pg_tok = PG_CONST;
        return;
    }

    pg_tok = PG_ERR;
    addError(NULL, CERR_055, *pg_cp);
    pg_hasError = 1;
}


/**
 * \brief
 * parse expression
 */
CExpr*
pg_parse_expr()
{
    return pg_term_expr(0);
}

extern CExprOfTypeDesc s_voidTypeDesc;
extern CExprOfTypeDesc s_numTypeDescs[BT_END];

/**
 * \brief
 * parse terminal node
 */
PRIVATE_STATIC CExpr*
pg_term_expr(int pre)
{
    CExprCodeEnum code;
    CExpr *e = NULL, *ee = NULL;

    if(pre > 10)
        return pg_unary_expr();

    if((e = pg_term_expr(pre + 1)) == NULL)
        return NULL;

  again:
    switch(pre) {
    case 0:
        if(pg_tok == PG_OROR) { code = EC_LOG_OR; goto next; }
        break;
    case 1:
        if(pg_tok == PG_ANDAND) { code = EC_LOG_AND; goto next; }
        break;
    case 2:
        if(pg_tok == '|') { code = EC_BIT_OR; goto next; }
        break;
    case 3:
        if(pg_tok == '^') { code = EC_BIT_XOR;  goto next; }
        break;
    case 4:
        if(pg_tok == '&') { code = EC_BIT_AND; goto next; }
        break;
    case 5:
        if(pg_tok == PG_EQEQ) { code = EC_ARITH_EQ; goto next; }
        if(pg_tok == PG_NEQ) { code = EC_ARITH_NE; goto next; }
        break;
    case 6:
        if(pg_tok == '>') { code = EC_ARITH_GT; goto next; }
        if(pg_tok == '<') { code = EC_ARITH_LT; goto next; }
        if(pg_tok == PG_GTEQ) { code = EC_ARITH_GE; goto next; }
        if(pg_tok == PG_LTEQ) { code = EC_ARITH_LE; goto next; }
        break;
    case 7:
        if(pg_tok == PG_LTLT) { code = EC_LSHIFT; goto next; }
        if(pg_tok == PG_GTGT) { code = EC_RSHIFT; goto next; }
        break;
    case 8:
        if(pg_tok == '+') { code = EC_PLUS; goto next; }
        if(pg_tok == '-') { code = EC_MINUS; goto next; }
        break;
    case 10:
        if(pg_tok == '*') { code = EC_MUL; goto next; }
        if(pg_tok == '/') { code = EC_DIV; goto next; }
        if(pg_tok == '%') { code = EC_MOD; goto next; }
        break;
    }

    return e;

  next:

    pg_tok_val = NULL;
    pg_get_token();

    if((ee = pg_term_expr(pre + 1)) == NULL) {
        if(e)
            freeExpr(e);
        return NULL;
    }

    e = exprBinary(code, e, ee);
    //exprSetExprsType(e, &s_voidTypeDesc);
    exprSetExprsType(e, &s_numTypeDescs[BT_INT]);

    goto again;
}


/**
 * \brief
 * parse unary expression
 */
PRIVATE_STATIC CExpr*
pg_unary_expr()
{
    CExpr *e = NULL;
    CExprCodeEnum code;

    switch(pg_tok){
    case '-':
        pg_get_token();
        if((e = pg_factor_expr()) == NULL)
            goto error;
        code = EC_UNARY_MINUS;
        break;

    case '!':
        pg_get_token();
        if((e = pg_factor_expr()) == NULL)
            goto error;
        code = EC_LOG_NOT;
        break;

    case '~':
        pg_get_token();
        if((e = pg_factor_expr()) == NULL)
            goto error;
        code = EC_BIT_NOT;
        break;

    default:
        return pg_factor_expr();
    }

    return exprUnary(code, e);

  error:
    if(e)
        freeExpr(e);
    return NULL;
}


/**
 * \brief
 * add error for 'expected ...'
 *
 * @param expected
 *      message argument
 */
PRIVATE_STATIC void
addExpectedCharError(const char *expected)
{
    addError(NULL, CERR_056, expected);
}


/**
 * \brief
 * parse  postfix expression
 */
PRIVATE_STATIC CExpr*
pg_factor_expr()
{
    CExpr *e, *ee = NULL, *args;

    e = pg_primary_expr();

    if(e == NULL)
        goto error;

  next:

    switch(pg_tok) {

    case '[':
        pg_get_token();

        if((ee = pg_term_expr(0)) == NULL)
            goto error;

        if(pg_tok != ']') {
            addExpectedCharError("]");
            goto error;
        }

        e = exprList2(EC_ARRAY_REF, e, ee);
        pg_get_token();
        break;

    case '(':
      pg_get_token();
      args = (CExpr *)allocExprOfList(EC_UNDEF);
      if (pg_tok != ')'){
	while (1){
	  if ((ee = pg_term_expr(0)) == NULL) {
	    goto error;
	  }
	  args = exprListAdd(args, ee);
	  if (pg_tok != ','){
	    break;
	  }
	  pg_get_token();
	}
      }

      if (pg_tok == ')'){
	pg_get_token();
	e = exprBinary(EC_FUNCTION_CALL, e, args);
	//exprSetExprsType(e, &s_voidTypeDesc);
	exprSetExprsType(e, &s_numTypeDescs[BT_INT]);
	break;
      }
      goto error;

    case '.':
        pg_get_token();
        if(pg_tok != PG_IDENT){
            addSyntaxErrorNearInExpression(".");
            goto error;
        }
        if((ee = pg_primary_expr()) == NULL)
            goto error;
        e = exprBinary(EC_MEMBER_REF, e, ee);
        break;

    case PG_STREF:
        pg_get_token();
        if(pg_tok != PG_IDENT){
            addSyntaxErrorNearInExpression("->");
            goto error;
        }

        if((ee = pg_primary_expr()) == NULL)
            goto error;

        e = exprBinary(EC_POINTS_AT, e, ee);
        break;

    default:
        return e;
    }

    goto next;

  error:

    if(e)
        freeExpr(e);
    if(ee)
        freeExpr(ee);
    return NULL;
}


/**
 * \brief
 * parse primary expression
 */
PRIVATE_STATIC CExpr*
pg_primary_expr()
{
    CExpr *e;

    switch(pg_tok){
    case '*':
        e = (CExpr*)allocExprOfGeneralCode(EC_FLEXIBLE_STAR, 0);
        pg_get_token();
        return e;

    case PG_IDENT:
        pg_get_token();
        assert(pg_tok_val);
        assertExprCode((CExpr*)pg_tok_val, EC_IDENT);
#ifdef not /* not needed to intern */
        e = (CExpr*)findSymbol(EXPR_SYMBOL(pg_tok_val)->e_symName);
        if(e == NULL) {
            addError(NULL, CERR_060, EXPR_SYMBOL(pg_tok_val)->e_symName);
            goto error;
        }
#endif
        return pg_tok_val;

    case '(':
        pg_get_token();
        if((e = pg_term_expr(0)) == NULL)
            goto error;

        if(pg_tok != ')'){
            addSyntaxErrorNearInExpression("(");
            goto error;
        }
        pg_get_token();
        return e;

    case PG_CONST:
        e = pg_tok_val;
        pg_get_token();
        return e;

    default:
        addSyntaxErrorInExpression();
        break;
    }

  error:
    return NULL;
}


/**
 * \brief
 * convert string to integer 
 */
PRIVATE_STATIC void
string_to_integer(long long int *p, char *cp, int radix)
{
    char    ch;
#ifdef NO_LONGLONG
    int     value;
#endif
    int            x;
    unsigned int v0, v1, v2, v3;

#ifdef NO_LONGLONG
    value = 0;
    for( ; (ch = *cp) != 0 ; cp++ ){
        if ( isdigit(ch) )
            x = ch - '0';
        else if ( isupper(ch) )
            x = ch - 'A' + 10;
        else
            x = ch - 'a' + 10;
        value = value * radix + x;
    }
    *p = value;
#endif
    v0 = v1 = v2 = v3 = 0;        /* clear */
    for( ; (ch = *cp) != 0 ; cp++ ) {
        if (isdigit((int)ch))
            x = ch - '0';
        else if (isupper((int)ch))
            x = ch - 'A' + 10;
        else
            x = ch - 'a' + 10;
        v0 = v0 * radix + x;
        v1 = v1 * radix + ((v0 >> 16) & 0xFFFF);
        v2 = v2 * radix + ((v1 >> 16) & 0xFFFF);
        v3 = v3 * radix + ((v2 >> 16) & 0xFFFF);
        v0 &= 0xFFFF;
        v1 &= 0xFFFF;
        v2 &= 0xFFFF;
    }

    *p = ((long long)v3 << 48) | ((long long)v2 << 32) | ((long long)v1 << 16) | (long long)v0;
}


/**
 * \brief
 * parse number constant
 */
PRIVATE_STATIC CExpr*
pg_parse_number()
{
    char            ch, *cp;
    long long int   value;
    int             radix;
    CCardinalEnum   cd;
    char            *orgToken;

    cp = pg_tok_buf;  /* used as buffer */

    radix = 10;
    cd = CD_DEC;
    ch = *pg_cp++;

    if( ch == '0' ) {
        ch = *pg_cp++;
        if(ch == 'x' || ch == 'X') {    /* HEX */
            radix = 16;
            cd = CD_HEX;
            for(;;) {
                ch = *pg_cp++;
                if(!isxdigit((int)ch))
                    goto ret_INT;
                *cp++ = ch;
            }
        }

        if( ch == '.' )
            goto read_floating;
        if(!(ch >= '0' && ch <= '7'))
            goto ret_INT;

        /* octal */
        radix = 8;
        cd = CD_OCT;
        for(;;) {
            *cp++ = ch;
            ch = *pg_cp++;
            if ( !(ch >= '0' && ch <= '7') )
                goto ret_INT;
        }
        /* NOT REACHED */
    }

    /* else decimal or floating */

  read_floating:

    while(isdigit((int)ch)){
        *cp++ = ch;
        ch = *pg_cp++;
    }
    if (ch != '.' && ch != 'e' && ch != 'E')
        goto ret_INT;
    /* floating */
    if( ch == '.' ) {
        *cp++ = ch;
        /* reading floating */
        ch = *pg_cp++;
        while(isdigit((int)ch)) {
            *cp++ = ch;
            ch = *pg_cp++;
        }
    }

    if(ch == 'e' || ch == 'E'){
        *cp++ = 'e';
        ch = *pg_cp++;
        if(ch == '+' || ch == '-') {
            *cp++ = ch;
            ch = *pg_cp++;
        }
        while(isdigit((int)ch)) {
            *cp++ = ch;
            ch = *pg_cp++;
        }
    }

    --pg_cp;
    *cp = '\0';
    orgToken = ccol_strdup(pg_tok_buf, MAX_NAME_SIZ);
    return (CExpr*)allocExprOfNumberConst(EC_NUMBER_CONST, BT_DOUBLE, cd, orgToken);

  ret_INT:

    *cp = '\0';
    orgToken = ccol_strdup(pg_tok_buf, MAX_NAME_SIZ);
    string_to_integer(&value, pg_tok_buf, radix);

    if(ch == 'L'){
        ch = *pg_cp++;
        if(ch == 'L')
            return (CExpr*)allocExprOfNumberConst(EC_NUMBER_CONST, BT_LONGLONG, cd, orgToken);

        --pg_cp;
        if(value > LONG_MAX || value < LONG_MIN)
            addError(NULL, CERR_061);

        return (CExpr*)allocExprOfNumberConst(EC_NUMBER_CONST, BT_LONG, cd, orgToken);
    }

    --pg_cp;
        if(value > INT_MAX || value < INT_MIN)
            addError(NULL, CERR_061);

    return (CExpr*)allocExprOfNumberConst(EC_NUMBER_CONST, BT_INT, cd, orgToken);
} 


/**
 * \brief
 * parse string constant
 */
PRIVATE_STATIC CExpr*
pg_parse_string_constant()
{
    int     ch;
    char   *cp,*end;
    int i,val;

    cp = pg_tok_buf;
    end = &pg_tok_buf[MAX_TOKEN_LEN];

  cont:

    ch = *pg_cp++;
    while(ch != '"') {
        switch (ch) {
        case '\\':        /* escape */
            if (cp >= end){
                addFatal(NULL, CFTL_003);
                break;
            }
            switch(ch = *pg_cp++){ /* escaped char(n,r,...) */
            case '\0':
                addError(NULL, CERR_062);
                goto exit;
            case 't': ch = '\t'; break;
            case 'b': ch = '\b'; break;
            case 'f': ch = '\f'; break;
            case 'n': ch = '\n'; break;
            case 'a': ch = '\a'; break;
            case 'r': ch = '\r'; break;
            case 'v': ch = '\v'; break;
            case '\\': ch = '\\'; break;
            case '0': 
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
                val = 0;
                for(i = 0; i < 3; i++){
                    if(!(ch >= '0' && ch <= '7')){
                        --pg_cp;
                        break;
                    }
                    val = val*8 + ch - '0';
                    ch = *pg_cp++;
                }
                ch = val;
            }
            *cp++ = ch;
            break;

        default:
            *cp++ = ch;
        }

        if (cp >= end) {
            addFatal(NULL, CFTL_004);
            break;
        }
        ch = *pg_cp++;
    }

  exit:

    do {
        ch = *pg_cp++;
    } while(isspace(ch));

    if(ch == '"') goto cont;
    --pg_cp;
    *cp = '\0';

    /* end of string */
    return (CExpr*)allocExprOfStringConst(EC_STRING_CONST,
        ccol_strdup(pg_tok_buf, MAX_NAME_SIZ), CT_MB);
}


/**
 * \brief
 * parse character constant
 */
PRIVATE_STATIC CExpr*
pg_parse_char_constant()
{
    int     ch, value;
    char   *cp;

    value = 0;
    cp = pg_tok_buf;
    ch = *pg_cp++;

    switch (ch) {
    case '\0':
        addError(NULL, CERR_063);
        break;

    case '\n':
        addError(NULL, CERR_064);
        break;

    case '\\':        /* escape sequence */
                /* '\': \nnn and \xNN are default except top 2 chars */
        ch = *pg_cp++;
        switch (ch) {
        case 'x':        /* hex '\xhh', at most 2 */
            ch = *pg_cp++;
            if ( !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
                   (ch >= 'A' && ch <= 'F')) ){
                addWarn(NULL, CWRN_007);
                break;
            }
            *cp++ = ch;
            value = 0xf & ch;
            ch = *pg_cp++;
            if ( !((ch >= '0' && ch <= '9') ||(ch >= 'a' && ch <= 'f') ||
                   (ch >= 'A' && ch <= 'F'))){
                break;
            }
            *cp++ = ch;
            value = (value << 4) | (0xf & ch);
            break;

        case '0':        /* oct '\ooo', at most 3 */
            ch = *pg_cp++;
            if ( ch == '"'){        /* '\0' */
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
            ch = *pg_cp++;
            if ( !(ch >= '0' && ch <= '7') )
                break;
            *cp++ = ch;
            value = (value << 3) | (0x7 & ch);
            ch = *pg_cp++;
            if ( !(ch >= '0' && ch <= '7') )
                break;
            *cp++ = ch;
            value = (value << 3) | (0x7 & ch);
            break;

        case 'a':
            value = '\a';
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
        case 'v':
            value = '\v';
            break;
        case '\\':
            value = '\\';
            break;
        case '?':
            value = '\?';
            break;
        case '\'':
            value = '\'';
            break;
        case '"':
            value = '\"';
            break;
        default:
            addError(NULL, CERR_065);
            break;
        }
        *cp++ = ch;
        break;

    default:
        *cp++ = ch;
        value = ch;
        break;
    }
    *cp = '\0';

    ch = *pg_cp++;
    if(ch != '\'')
        addError(NULL, CERR_066);

    if(cp == pg_tok_buf)
        addError(NULL, CERR_067);

    return (CExpr*)allocExprOfCharConst(EC_CHAR_CONST, NULL, CT_MB);
}


/**
 * \brief
 * parse any directive
 *
 * @param name
 *      pragma name
 * @param type
 *      directive type
 * @return
 *      allocated node
 */
CExpr*
lexAllocDirective(const char *name, CDirectiveTypeEnum type)
{
    char *p = yytext;
    /* after '#directive[space]' */
    while(isspace(*p)) ++p;
    while(isspace(*p) == 0) ++p;
    while(isspace(*p)) ++p;

    int len = strlen(p);
    char *str = XALLOCSZ(char, len + 1);
    strncpy(str, p, len);
    /* remove tailing lf */
    str[len] = 0;
    if(len > 0 && (str[len - 1] == '\n' || str[len - 1] == '\r'))
        str[len - 1] = 0;
    if(len > 1 && (str[len - 2] == '\n' || str[len - 2] == '\r'))
        str[len - 2] = 0;

    return (CExpr*)allocExprOfDirective(
        type, ccol_strdup(name, MAX_NAME_SIZ), str);
}


/**
 * \brief
 * parse pragma pack
 *
 * @param p
 *      char pointer
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExpr*
lexParsePragmaPack(char *p)
{
    //skip pragma[space]pack[space]*
    p = lexSkipSpace(lexSkipWordP(lexSkipSpace(lexSkipWord(lexSkipSpace(p)))));
    if(*p != '(') {
        addWarn(NULL, CWRN_010);
        return exprNull();
    }

    pg_cp = ++p;
    char *p1 = lexSkipSpace(p);
    CExpr *e = NULL;
    if(*p1 != ')') {
        pg_get_token();
        e = pg_parse_expr(')');
        if(EXPR_ISNULL(e) == 0) {
            CNumValueWithType nvt;
            if(getConstNumValue(e, &nvt) == 0 ||
                (nvt.nvt_numKind != NK_LL && nvt.nvt_numKind != NK_ULL) ||
                (nvt.nvt_numKind == NK_LL ?
                    nvt.nvt_numValue.ll < 0 : nvt.nvt_numValue.ull < 0)) {
                addWarn(NULL, CWRN_025);
                return exprNull();
            }
            freeExpr(e);
            e = (CExpr*)allocExprOfNumberConst1(&nvt);
        }
    }

    CExpr *packDir = exprUnary(EC_PRAGMA_PACK, e);
    pg_tok_val = NULL;

    return packDir;
}


/**
 * \brief
 * parse pragma
 *
 * @param p
 *      char pointer
 * @param[out] token
 *      token code
 * @return
 *      allocated node
 */
CExpr*
lexParsePragma(char *p, int *token)
{
    CPragmaKind pk = getPragmaKind(p);

    if(pk == PK_PACK) {
        *token = PRAGMA_PACK;
        return lexParsePragmaPack(p);
    }
    else if(pk == PK_OMP) {
      return lexParsePragmaOMP(p,token);
    }
    else if(pk == PK_XMP) {
      return lexParsePragmaXMP(p,token);
    }
    else if(pk == PK_ACC) {
      return lexParsePragmaACC(p,token);
    }
    else if(pk == PK_NOT_PARSABLE) {
        *token = DIRECTIVE;
        return lexAllocDirective("#pragma", DT_PRAGMA);
    }
    else {
        ABORT();
        return 0;
    }
}
