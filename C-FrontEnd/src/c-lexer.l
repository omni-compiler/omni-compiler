%{
/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include <sys/param.h>
#include <ctype.h>
#include "c-expr.h"
#include "c-lexyacc.h"
#include "c-parser.h"
#include "c-token.h"
#include "c-pragma.h"
#include "c-option.h"

#define YY_ALWAYS_INTERACTIVE 1
#define YY_NO_UNPUT 1

int             s_hasError = 0;
int             s_hasWarn = 0;
CLineNumInfo    s_lineNumInfo = { 0 };
int             s_saveToken = 0;
char            *s_saveTokenBuf = NULL;
int             s_saveTokenLen = 0;
int             s_saveTokenMaxLen = 0;

//! line number
PRIVATE_STATIC int              s_lineno = 1;
//! column number
PRIVATE_STATIC int              s_column = 0;
//! file ID sequence
PRIVATE_STATIC int              s_fileIdSeq = 1;
//! filename specified by line directive
PRIVATE_STATIC char             s_fileName[MAXPATHLEN];
//! is source filename set
PRIVATE_STATIC int              s_isSourceSet;
//! yylex return value of pre-called yylex
PRIVATE_STATIC int              s_preLexCode = 0;
//! symbol type of pre-called yylex
PRIVATE_STATIC int              s_preSymType = 0;

PRIVATE_STATIC void         lexError();
PRIVATE_STATIC CExpr*       lexAllocTypeQual(CTypeQualEnum typeQual);
PRIVATE_STATIC CExpr*       lexAllocTypeSpec(CTypeSpecEnum typeQual);
PRIVATE_STATIC CExpr*       lexAllocSCSpec(CSCSpecEnum scspec);
PRIVATE_STATIC CExpr*       lexAllocFuncID();
PRIVATE_STATIC CExpr*       lexAllocNumberConst(CBasicTypeEnum bt, CCardinalEnum c);
PRIVATE_STATIC CExpr*       lexAllocCharConst(CCharTypeEnum t);
PRIVATE_STATIC CExpr*       lexAllocStringConst(CCharTypeEnum t);
PRIVATE_STATIC void         lexParseLineDirective(int skipDirectiveKeyword);
PRIVATE_STATIC void         comment();
PRIVATE_STATIC void         count();
PRIVATE_STATIC void         setFileName();

//lex function
//extern int fileno(FILE*);

#define LEX_EXPRCODE(c, t) \
    yylval.expr = lexAllocExprCode(c, t); return t;

#define LEX_NUMCONST(bt, cd) \
    yylval.expr = lexAllocNumberConst(bt, cd); return CONSTANT

#define LEX_GCC_NUMCONST(bt, cd) \
    if(s_supportGcc) { yylval.expr = lexAllocNumberConst(bt, cd); return CONSTANT; }\
    else { lexError(); yylval.expr = lexAllocSymbol(); return IDENTIFIER; }

#define LEX_GCC_TYPESPEC(x) \
    if(s_supportGcc) { yylval.expr = lexAllocTypeSpec(x); return TYPESPEC; }\
    yylval.expr = lexAllocSymbol(); return IDENTIFIER

#define LEX_GCC_SCSPEC(x) \
    if(s_supportGcc) { yylval.expr = lexAllocSCSpec(x); return SCSPEC; }\
    yylval.expr = lexAllocSymbol(); return IDENTIFIER

#define LEX_GCC_TYPEQUAL(x) \
    if(s_supportGcc) { yylval.expr = lexAllocTypeQual(x); return TYPEQUAL; }\
    yylval.expr = lexAllocSymbol(); return IDENTIFIER

#define LEX_GCC_TYPENAME() \
    yylval.expr = lexAllocSymbol(); return s_supportGcc ? TYPENAME : IDENTIFIER

#define LEX_GCC_IDENT(c) \
    if(s_supportGcc) return c;\
    yylval.expr = lexAllocSymbol(); return IDENTIFIER

#define LEX_GCC_FUNCID() \
    if(s_supportGcc) { yylval.expr = lexAllocFuncID(); }\
    else { yylval.expr = lexAllocSymbol(); } return IDENTIFIER

#define LEX_GCC_DIRECTIVE(name, t) \
    if(s_supportGcc) { yylval.expr = lexAllocDirective(name, t); return DIRECTIVE; }\
    else { lexError(); yylval.expr = lexAllocSymbol(); return IDENTIFIER; }

#define YY_DECL int raw_yylex(void)

#define dup_yytext() (ccol_strdup(yytext, MAX_NAME_SIZ))

%}

%option yylineno
%option nounput

Alph        [a-zA-Z_]
Word        {Alph}({Alph}|{Dig})*
Bin         [01]
Dig         [0-9]
DigNZ       [1-9]
Oct         [0-7]
Hex         [a-fA-F0-9]
BinPfx      0[bB]
OctPfx      0
HexPfx      0[xX]
Exp         [eE][+-]?{Dig}+
BExp        [pP][+-]?{Dig}+
BinNum      {BinPfx}{Bin}+
OctNum      {OctPfx}{Oct}+
DecNum      ({DigNZ}{Dig}*|0)
FDecNum     (({Dig}+"."{Dig}*|"."{Dig}+|{Dig}+){Exp})|({Dig}+"."{Dig}*|"."{Dig}+)
HexNum      {HexPfx}{Hex}+
FHexNum     {HexPfx}({Hex}+|{Hex}+"."{Hex}*|"."{Hex}+){BExp}
FSfx        [fF]
USfx        [uU]
LFSfx       [lL]
LSfx        [lL]
ULSfx       {USfx}{LSfx}|{LSfx}{USfx}
LLSfx       {LSfx}{LSfx}
ULLSfx      {USfx}{LLSfx}|{LLSfx}{USfx}
ISfx        i
IFSfx       {ISfx}{FSfx}|{FSfx}{ISfx}

%%
                        /* ------ comment ------ */

"/*"           { comment(); }
"//"[^\n]*     {}

                        /* ------ gcc ------ */

{BinNum}                { count(); LEX_GCC_NUMCONST(BT_INT,               CD_BIN); }
{BinNum}{USfx}          { count(); LEX_GCC_NUMCONST(BT_UNSIGNED_INT,      CD_BIN); }
{BinNum}{LSfx}          { count(); LEX_GCC_NUMCONST(BT_LONG,              CD_BIN); }
{BinNum}{ULSfx}         { count(); LEX_GCC_NUMCONST(BT_UNSIGNED_LONG,     CD_BIN); }
{BinNum}{LLSfx}         { count(); LEX_GCC_NUMCONST(BT_LONGLONG,          CD_BIN); }
{BinNum}{ULLSfx}        { count(); LEX_GCC_NUMCONST(BT_UNSIGNED_LONGLONG, CD_BIN); }
"__thread"              { count(); LEX_GCC_SCSPEC(SS_THREAD); }
"__signed"              { count(); LEX_GCC_TYPESPEC(TS_SIGNED); }
"__signed__"            { count(); LEX_GCC_TYPESPEC(TS_SIGNED); }
"__complex"             { count(); LEX_GCC_TYPESPEC(TS_COMPLEX); }
"__complex__"           { count(); LEX_GCC_TYPESPEC(TS_COMPLEX); }
"__const"               { count(); LEX_GCC_TYPEQUAL(TQ_CONST); }
"__const__"             { count(); LEX_GCC_TYPEQUAL(TQ_CONST); }
"__volatile"            { count(); LEX_GCC_TYPEQUAL(TQ_VOLATILE); }
"__volatile__"          { count(); LEX_GCC_TYPEQUAL(TQ_VOLATILE); }
"__restrict"            { count(); LEX_GCC_TYPEQUAL(TQ_RESTRICT); }
"__restrict__"          { count(); LEX_GCC_TYPEQUAL(TQ_RESTRICT); }
"__inline"              { count(); LEX_GCC_TYPEQUAL(TQ_INLINE); }
"__inline__"            { count(); LEX_GCC_TYPEQUAL(TQ_INLINE); }
"__FUNCTION__"          { count(); LEX_GCC_FUNCID(); }
"__PRETTY_FUNCTION__"   { count(); LEX_GCC_FUNCID(); }
"__alignof"             { count(); LEX_GCC_IDENT(ALIGNOF); }
"__alignof__"           { count(); LEX_GCC_IDENT(ALIGNOF); }
"asm"                   { count(); LEX_GCC_IDENT(ASSEMBLER); }
"__asm"                 { count(); LEX_GCC_IDENT(ASSEMBLER); }
"__asm__"               { count(); LEX_GCC_IDENT(ASSEMBLER); }
"__attribute"           { count(); LEX_GCC_IDENT(ATTRIBUTE); }
"__attribute__"         { count(); LEX_GCC_IDENT(ATTRIBUTE); }
"__real"                { count(); LEX_GCC_IDENT(REALPART); }
"__real__"              { count(); LEX_GCC_IDENT(REALPART); }
"__imag__"              { count(); LEX_GCC_IDENT(IMAGPART); }
"__imag"                { count(); LEX_GCC_IDENT(IMAGPART); }
"__extension__"         { count(); LEX_GCC_IDENT(EXTENSION); }
"typeof"                { count(); LEX_GCC_IDENT(TYPEOF); }
"__typeof"              { count(); LEX_GCC_IDENT(TYPEOF); }
"__typeof__"            { count(); LEX_GCC_IDENT(TYPEOF); }
"__label__"             { count(); LEX_GCC_IDENT(LABEL); }

"__builtin_va_arg"             { count(); if(s_useBuiltinVaArg){
                                              LEX_GCC_IDENT(BLTIN_VA_ARG); 
                                          } else {
                                              yylval.expr = lexAllocSymbol(); return IDENTIFIER;
                                          }
                               }
"__builtin_offsetof"           { count(); LEX_GCC_IDENT(BLTIN_OFFSETOF); }
"__builtin_types_compatible_p" { count(); LEX_GCC_IDENT(BLTIN_TYPES_COMPATIBLE_P); }


                        /* ------ c99 reserved words ------ */

"auto"         { count(); yylval.expr = lexAllocSCSpec(SS_AUTO);         return SCSPEC; }
"extern"       { count(); yylval.expr = lexAllocSCSpec(SS_EXTERN);       return SCSPEC; }
"register"     { count(); yylval.expr = lexAllocSCSpec(SS_REGISTER);     return SCSPEC; }
"typedef"      { count(); yylval.expr = lexAllocSCSpec(SS_TYPEDEF);      return SCSPEC; }

"void"         { count(); yylval.expr = lexAllocTypeSpec(TS_VOID);       return TYPESPEC; }
"unsigned"     { count(); yylval.expr = lexAllocTypeSpec(TS_UNSIGNED);   return TYPESPEC; }
"signed"       { count(); yylval.expr = lexAllocTypeSpec(TS_SIGNED);     return TYPESPEC; }
"int"          { count(); yylval.expr = lexAllocTypeSpec(TS_INT);        return TYPESPEC; }
"char"         { count(); yylval.expr = lexAllocTypeSpec(TS_CHAR);       return TYPESPEC; }
"wchar_t"      { count(); if(s_useBuiltinWchar) {
                        yylval.expr = lexAllocTypeSpec(TS_WCHAR); return TYPESPEC;
                    } else {
                        yylval.expr = lexAllocSymbol(); return IDENTIFIER;
                    }
               }
"short"        { count(); yylval.expr = lexAllocTypeSpec(TS_SHORT);      return TYPESPEC; }
"long"         { count(); yylval.expr = lexAllocTypeSpec(TS_LONG);       return TYPESPEC; }
"float"        { count(); yylval.expr = lexAllocTypeSpec(TS_FLOAT);      return TYPESPEC; }
"double"       { count(); yylval.expr = lexAllocTypeSpec(TS_DOUBLE);     return TYPESPEC; }
"_Complex"     { count(); yylval.expr = lexAllocTypeSpec(TS_COMPLEX);    return TYPESPEC; }
"_Bool"        { count(); yylval.expr = lexAllocTypeSpec(TS_BOOL);       return TYPESPEC; }
"_Imaginary"   { count(); yylval.expr = lexAllocTypeSpec(TS_IMAGINARY);  return TYPESPEC; }

"const"        { count(); yylval.expr = lexAllocTypeQual(TQ_CONST);      return TYPEQUAL; }
"volatile"     { count(); yylval.expr = lexAllocTypeQual(TQ_VOLATILE);   return TYPEQUAL; }
"restrict"     { count(); yylval.expr = lexAllocTypeQual(TQ_RESTRICT);   return TYPEQUAL; }
"inline"       { count(); yylval.expr = lexAllocTypeQual(TQ_INLINE);     return TYPEQUAL; }

"__func__"     { count(); yylval.expr = lexAllocFuncID();                return IDENTIFIER; }

"static"       { count(); yylval.expr = lexAllocSCSpec(SS_STATIC);       return STATIC; }
"struct"       { count(); return STRUCT; }
"union"        { count(); return UNION; }
"enum"         { count(); return ENUM; }
"break"        { count(); return BREAK; }
"case"         { count(); return CASE; }
"continue"     { count(); return CONTINUE; }
"default"      { count(); LEX_EXPRCODE(EC_DEFAULT_LABEL, DEFAULT); }
"do"           { count(); return DO; }
"else"         { count(); return ELSE; }
"for"          { count(); return FOR; }
"goto"         { count(); return GOTO; }
"if"           { count(); return IF; }
"return"       { count(); return RETURN; }
"sizeof"       { count(); return SIZEOF; }
"switch"       { count(); return SWITCH; }
"while"        { count(); return WHILE; }

"xmp_desc_of"       { count(); return XMP_DESC_OF; }

                        /* ------ constant literals ------ */

{Word}              { count(); yylval.expr = lexAllocSymbol(); return IDENTIFIER; }
{OctNum}            { count(); LEX_NUMCONST(BT_INT,                   CD_OCT); }
{OctNum}{USfx}      { count(); LEX_NUMCONST(BT_UNSIGNED_INT,          CD_OCT); }
{OctNum}{LSfx}      { count(); LEX_NUMCONST(BT_LONG,                  CD_OCT); }
{OctNum}{ULSfx}     { count(); LEX_NUMCONST(BT_UNSIGNED_LONG,         CD_OCT); }
{OctNum}{LLSfx}     { count(); LEX_NUMCONST(BT_LONGLONG,              CD_OCT); }
{OctNum}{ULLSfx}    { count(); LEX_NUMCONST(BT_UNSIGNED_LONGLONG,     CD_OCT); }
{DecNum}            { count(); LEX_NUMCONST(BT_INT,                   CD_DEC); }
{DecNum}{USfx}      { count(); LEX_NUMCONST(BT_UNSIGNED_INT,          CD_DEC); }
{DecNum}{LSfx}      { count(); LEX_NUMCONST(BT_LONG,                  CD_DEC); }
{DecNum}{ULSfx}     { count(); LEX_NUMCONST(BT_UNSIGNED_LONG,         CD_DEC); }
{DecNum}{LLSfx}     { count(); LEX_NUMCONST(BT_LONGLONG,              CD_DEC); }
{DecNum}{ULLSfx}    { count(); LEX_NUMCONST(BT_UNSIGNED_LONGLONG,     CD_DEC); }
{HexNum}            { count(); LEX_NUMCONST(BT_INT,                   CD_HEX); }
{HexNum}{USfx}      { count(); LEX_NUMCONST(BT_UNSIGNED_INT,          CD_HEX); }
{HexNum}{LSfx}      { count(); LEX_NUMCONST(BT_LONG,                  CD_HEX); }
{HexNum}{ULSfx}     { count(); LEX_NUMCONST(BT_UNSIGNED_LONG,         CD_HEX); }
{HexNum}{LLSfx}     { count(); LEX_NUMCONST(BT_LONGLONG,              CD_HEX); }
{HexNum}{ULLSfx}    { count(); LEX_NUMCONST(BT_UNSIGNED_LONGLONG,     CD_HEX); }
{FDecNum}{FSfx}     { count(); LEX_NUMCONST(BT_FLOAT,                 CD_DEC); }
{FDecNum}           { count(); LEX_NUMCONST(BT_DOUBLE,                CD_DEC); }
{FDecNum}{LFSfx}    { count(); LEX_NUMCONST(BT_LONGDOUBLE,            CD_DEC); }
{FDecNum}{IFSfx}    { count(); LEX_NUMCONST(BT_FLOAT_IMAGINARY,       CD_DEC); }
{FDecNum}{ISfx}     { count(); LEX_NUMCONST(BT_DOUBLE_IMAGINARY,      CD_DEC); }
{FHexNum}{FSfx}     { count(); LEX_NUMCONST(BT_FLOAT,                 CD_HEX); }
{FHexNum}           { count(); LEX_NUMCONST(BT_DOUBLE,                CD_HEX); }
{FHexNum}{LFSfx}    { count(); LEX_NUMCONST(BT_LONGDOUBLE,            CD_HEX); }

L'(\\.|[^\\'])+'        { count(); yylval.expr = lexAllocCharConst(CT_WIDE);    return CONSTANT; }
'(\\.|[^\\'])+'         { count(); yylval.expr = lexAllocCharConst(CT_MB);      return CONSTANT; }
L\"(\\.|[^\\"])*\"      { count(); yylval.expr = lexAllocStringConst(CT_WIDE);  return STRING;}
\"(\\.|[^\\"])*\"       { count(); yylval.expr = lexAllocStringConst(CT_MB);    return STRING; }

                        /* ------ symbols ------ */

"..."        { count(); return ELLIPSIS; }

">>="        { count(); yylval.assigntype = AS_RSHIFT_EQ;   return ASSIGN; }
"<<="        { count(); yylval.assigntype = AS_LSHIFT_EQ;   return ASSIGN; }
"+="         { count(); yylval.assigntype = AS_PLUS_EQ;     return ASSIGN; }
"-="         { count(); yylval.assigntype = AS_MINUS_EQ;    return ASSIGN; }
"*="         { count(); yylval.assigntype = AS_MUL_EQ;      return ASSIGN; }
"/="         { count(); yylval.assigntype = AS_DIV_EQ;      return ASSIGN; }
"%="         { count(); yylval.assigntype = AS_MOD_EQ;      return ASSIGN; }
"&="         { count(); yylval.assigntype = AS_AND_EQ;      return ASSIGN; }
"^="         { count(); yylval.assigntype = AS_XOR_EQ;      return ASSIGN; }
"|="         { count(); yylval.assigntype = AS_OR_EQ;       return ASSIGN; }

">>"         { count(); return RSHIFT; }
"<<"         { count(); return LSHIFT; }
"++"         { count(); return PLUSPLUS; }
"--"         { count(); return MINUSMINUS; }
"&&"         { count(); return ANDAND; }
"||"         { count(); return OROR; }
"->"         { count(); return POINTSAT; }
 
"<="         { count(); return ARITH_LE; }
">="         { count(); return ARITH_GE; }
"=="         { count(); return ARITH_EQ; }
"!="         { count(); return ARITH_NE; }
"<"          { count(); return '<'; }
">"          { count(); return '>'; }

":"[ \t\n]*"["  { count(); return COLON_SQBRACKET; }

"="          { count(); return '='; }
"."          { count(); return '.'; }
","          { count(); return ','; }
":"          { count(); return ':'; }
";"          { count(); return ';'; }
("{"|"<%")   { count(); return '{'; }
("}"|"%>")   { count(); return '}'; }
"("          { count(); return '('; }
")"          { count(); return ')'; }
("["|"<:")   { count(); return '['; }
("]"|":>")   { count(); return ']'; }
"&"          { count(); return '&'; }
"!"          { count(); return '!'; }
"~"          { count(); return '~'; }
"-"          { count(); return '-'; }
"+"          { count(); return '+'; }
"*"          { count(); return '*'; }
"/"          { count(); return '/'; }
"%"          { count(); return '%'; }
"^"          { count(); return '^'; }
"|"          { count(); return '|'; }
"?"          { count(); return '?'; }

                        /* ------ directive ------ */
 /*
    Note: Since a directive include "\n", count() should be invoked after 
          processing the directive. (Hitoshi Murai)
 */

^[ \t]*"#ident"([ \t]+[^\n]+)?[\n]      { LEX_GCC_DIRECTIVE("#ident", DT_IDENT); count(); }
^[ \t]*"#warning"([ \t]+[^\n]+)?[\n]    { LEX_GCC_DIRECTIVE("#warning", DT_WARNING); count(); }
^[ \t]*"#error"([ \t]+[^\n]+)[\n]       { LEX_GCC_DIRECTIVE("#error", DT_ERROR); count(); }
^[ \t]*"#"[ \t]*"pragma"([ \t]+[^\n]+)?[\n]     { int token;
                                            yylval.expr = lexParsePragma(yytext, &token); count();
                                            return token; }
^[ \t]*"_Pragma"[ \t]*"("[ \t]*"L"?"\""[^\n]*"\""[ \t]*")"[ \t]*[\n] { int token;
                                            char *converted_yytext = lexConvertUnderscorePragma(yytext);
                                            yylval.expr = lexParsePragma(converted_yytext, &token); count();
                                            free(converted_yytext);
                                            return token; }
^[ \t]*"#"[ \t]*[0-9]+([ \t]+[^\r\n]*)?\r?\n  { lexParseLineDirective(0); }
^[ \t]*"#"[ \t]*"line"[ \t]+[0-9]+([ \t]+[^\r\n]*)?\r?\n  { lexParseLineDirective(1); }

                        /* ------ ignoring character ------ */

[ \t\v\r\n\f]  { count(); }
\x0c           { count(); /* skip ^L */ }
.              { count(); lexError(); }

%%

int
yywrap()
{
    return 1;
}


PRIVATE_STATIC void
lexParseLineDirective(int skipDirectiveKeyword)
{
    #define WHILE_SPC(p) \
        while(*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
    #define WHILE_WORD(p) \
        while(*p && (*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n'))

    char *p = yytext;
    // skip to '#'
    WHILE_SPC(p) ++p;
    // skip '#[space]'
    while(*p && *p == '#') ++p;
    WHILE_SPC(p) ++p;

    if(skipDirectiveKeyword) {
        // skip 'line[space]'
        WHILE_WORD(p) ++p;
        WHILE_SPC(p) ++p;
    }

    // read line num
    s_charBuf[0][0] = 0;
    int i = 0, dq = 0;
    WHILE_WORD(p) s_charBuf[0][i++] = *(p++);

    s_charBuf[0][i] = 0;
    s_lineno = atoi(s_charBuf[0]);
    setLineNumInfo();

    WHILE_SPC(p) ++p;
    if(*p == 0)
        return;

    // read file name
    s_charBuf[0][0] = 0;
    i = 0;
    while(*p && (*p != '\r' && *p != '\n')) {
        if(*p == '"') {
            if(++dq == 2)
                break;
        } else {
            s_charBuf[0][i++] = *p;
        }
        ++p;
    }

    s_charBuf[0][i] = 0;
    strcpy(s_fileName, s_charBuf[0]);
    setFileName();

    if(s_isSourceSet == 0) {
        s_isSourceSet = 1;
        strcpy(s_sourceFileName, s_fileName);
    }
}


PRIVATE_STATIC CExpr*
lexAllocTypeQual(CTypeQualEnum typeQual)
{
    return (CExpr*)allocExprOfGeneralCode(EC_TYPEQUAL, typeQual);
}


PRIVATE_STATIC CExpr*
lexAllocTypeSpec(CTypeSpecEnum typeSpec)
{
    return (CExpr*)allocExprOfGeneralCode(EC_TYPESPEC, typeSpec);
}


PRIVATE_STATIC CExpr*
lexAllocSCSpec(CSCSpecEnum scspec)
{
    return (CExpr*)allocExprOfGeneralCode(EC_SCSPEC, scspec);
}


CExpr*
lexAllocSymbol()
{
    return (CExpr*)allocExprOfSymbol(EC_IDENT, dup_yytext());
}


PRIVATE_STATIC CExpr*
lexAllocFuncID()
{
    CExpr *expr = lexAllocSymbol();
    EXPR_SYMBOL(expr)->e_symType = ST_FUNCID;
    return expr;
}


PRIVATE_STATIC CExpr*
lexAllocNumberConst(CBasicTypeEnum bt, CCardinalEnum c)
{
    return (CExpr*)allocExprOfNumberConst(EC_NUMBER_CONST, bt, c, dup_yytext());
}


PRIVATE_STATIC char*
dup_yytext_for_const(CCharTypeEnum t)
{
    int len = strlen(yytext);
    int ctw = (t == CT_WIDE);
    assertYYLineno(len >= 2);
    /* CT_MB: "xx" -> xx, CT_WIDE: L"xx" -> xx */
    int fixlen = len - (ctw + 2) + 1;
    char *str = XALLOCSZ(char, fixlen);
    memcpy(str, &yytext[ctw + 1], fixlen - 1);
    str[fixlen - 1] = 0;
    return str;
}


PRIVATE_STATIC CExpr*
lexAllocCharConst(CCharTypeEnum t)
{
    return (CExpr*)allocExprOfCharConst(EC_CHAR_CONST, dup_yytext_for_const(t), t);
}


PRIVATE_STATIC CExpr*
lexAllocStringConst(CCharTypeEnum t)
{
    return (CExpr*)allocExprOfStringConst(EC_STRING_CONST, dup_yytext_for_const(t), t);
}



CExpr*
lexAllocExprCode(CExprCodeEnum exprCode, int token)
{
    return (CExpr*)allocExprOfGeneralCode(exprCode, token);
}


PRIVATE_STATIC void
setFileName()
{
    CCOL_HashEntry *he = CCOL_HT_FIND_STR(&s_fileIdTab, s_fileName);
    CFileIdEntry *fie;

    if(he == NULL) {
        /* Create FileID Entry */
        fie = XALLOC(CFileIdEntry);
        fie->fie_id = s_fileIdSeq++;
        CCOL_HT_PUT_STR(&s_fileIdTab, s_fileName, fie);
    } else {
        fie = (CFileIdEntry*)CCOL_HT_DATA(he);
    }

    s_lineNumInfo.ln_fileId = fie->fie_id;
}


const char *
getFileName(int fileId)
{
    CCOL_HashEntry *ite;
    CCOL_HashSearch hs;

    CCOL_HT_FOREACH(ite, hs, &s_fileIdTab) {
        CFileIdEntry *entry = (CFileIdEntry*)CCOL_HT_DATA(ite);
        if(entry->fie_id == fileId) {
            return (const char*)CCOL_HT_KEY(&s_fileIdTab, ite);
        }
    }

    return "";
}


void
setLineNumInfo()
{
    s_lineNumInfo.ln_lineNum = s_lineno;
    s_lineNumInfo.ln_column = s_column;
    s_lineNumInfo.ln_rawLineNum = yylineno;
}


PRIVATE_STATIC void
comment()
{
    char c, prev = 0;
    while ((c = input()) != 0) /* (EOF maps to 0) */ {
        if (c == '/' && prev == '*')
            return;
        prev = c;
    }
    yyerror(CERR_002);
}


PRIVATE_STATIC void
count()
{
    int i;

    for (i = 0; yytext[i] != '\0'; i++) {
        if (yytext[i] == '\n') {
            s_column = 0;
            s_lineno++;
            setLineNumInfo();
        } else if (yytext[i] == '\t') {
            s_column += 8 - (s_column % 8);
            s_lineNumInfo.ln_column = s_column;
        } else {
            s_column++;
            s_lineNumInfo.ln_column = s_column;
        }
    }
}


void
copy_yytext(char *p)
{
    strncpy(p, yytext, MAX_NAME_SIZ - 1);
    p[MAX_NAME_SIZ - 1] = 0;
}


void
initLexer(FILE *fp)
{
    strcpy(s_fileName, s_sourceFileName);
    setFileName();
    setLineNumInfo();
    yyin = fp;
}


#ifdef MTRACE
void
freeLexer()
{
#if(YY_FLEX_MAJOR_VERSION >= 2 && YY_FLEX_MINOR_VERSION >= 5 && !defined(YY_FLEX_SUBMINOR_VERSION))
    yy_delete_buffer(yy_current_buffer);
#elif defined(YY_FLEX_SUBMINOR_VERSION)
    yy_delete_buffer(YY_CURRENT_BUFFER);
#endif
}
#endif


PRIVATE_STATIC void
lexError()
{
    yyerror(CERR_001);
}

void
yyerror(const char *msg)
{
    setLineNumInfo();

#ifdef CEXPR_DEBUG_STAT
    DBGPRINTC(ESQ_RED, ("<<*** %s *** %s:%d-%d<<", msg,
        getFileName(s_lineNumInfo.ln_fileId), s_lineNumInfo.ln_lineNum,
        s_lineNumInfo.ln_rawLineNum));
#endif

    if(strcmp(msg, "syntax error") == 0)
        msg = CERR_001;
    if(isSkipErrorOutput(NULL) == 0)
        addError(NULL, msg);
}


#ifdef CEXPR_DEBUG_STAT
PRIVATE_STATIC const char*
lexGetTokenName(int code)
{
    for(int i = 0; i < CTOKENINFO_SIZE; ++i) {
        if(s_CTokenInfos[i].code == code)
            return s_CTokenInfos[i].name;
    }

    return NULL;
}
#endif


int
yylex(void)
{
    CExprOfSymbol *tsym = NULL;
    CSymbolTypeEnum symType = ST_UNDEF;
    int code = raw_yylex();

#ifdef CEXPR_DEBUG_STAT
    const char *tokenName;
#endif

    if(code == IDENTIFIER) {

        switch(s_preSymType) {
        /* symbol after non-reserved type or tag may be variable or member field */
        case ST_TYPE:
        case ST_TAG:
            goto end;
        }

        switch(s_preLexCode) {
        /* symbol after 'struct'/'union'/'enum' may be tag */
        case STRUCT:
        case UNION:
        case ENUM:
        /* symbol after type spec. may be variable or member field */
        case TYPESPEC:
        /* symbol after '->'/'.' may be member field */
        case POINTSAT:
        case '.':
        /* symbol after 'goto' may be label */
        case GOTO:
        /* symbol after '&&' may be variable or label */
        case ANDAND:
        /* symbol after '*' in param list may be variable or function */
        case '*':
            goto end;
        }

        if(isGccBuiltinType(yytext))
            code = TYPENAME;

        if(code != TYPENAME) {
            tsym = findSymbolByGroup(yytext, STB_IDENT);
            if(tsym != NULL)
                symType = tsym->e_symType;

            if(symType == ST_TYPE) {
                code = TYPENAME;
            }
        }
    }

end:

#ifdef CEXPR_DEBUG_STAT
    tokenName = lexGetTokenName(code);

    if(tokenName != NULL) {
        STAT_TRACE(("{yylex:%s}", tokenName));
    } else if(code < 255 && code >= 32) {
        STAT_TRACE(("{yylex:'%c'}", (char)code));
    } else {
        STAT_TRACE(("{yylex:%d}", code));
    }
#endif

    s_preLexCode = code;
    s_preSymType = symType;

    return code;
}

