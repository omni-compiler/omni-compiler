/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-xcodeml.c
 */

#include <stdlib.h>
#include <stdarg.h> 
#include <wchar.h>
#include "c-comp.h"
#include "c-option.h"

#include "c-xcodeml.h"
#include "c-omp.h"
#include "c-xmp.h"
#include "c-acc.h"

void outxTag1(FILE *fp, int indent, CExpr *expr, const char *tag,
                        int xattrFlag);

#define MAX_TYPEID_SIZE     256

#define XATTR_NORETURN          (1 << 0)
#define XATTR_NOCONTENT         (1 << 1)
#define XATTR_TYPE              (1 << 2)
#define XATTR_IS_MODIFIED       (1 << 3)
#define XATTR_IS_GCCSYNTAX      (1 << 4)
#define XATTR_LINENO            (1 << 5)
#define XATTR_IS_GCCEXTENSION   (1 << 6)
#define XATTR_COMMON            (XATTR_IS_MODIFIED|XATTR_IS_GCCSYNTAX)

static const char   *s_CBasicTypeEnumXcodes[] = CBasicTypeEnumXcodeDef;
static int          s_tidSeqBasicType = 0;
static int          s_tidSeqStruct = 0;
static int          s_tidSeqUnion = 0;
static int          s_tidSeqEnum = 0;
static int          s_tidSeqPointer = 0;
static int          s_tidSeqArray = 0;
static int          s_tidSeqFunc = 0;
static int          s_tidSeqCoarray = 0;

#define CSYMBOLS_GLOBAL (1 << 0)
#define CSYMBOLS_PARAM  (1 << 1)

void outx_ARRAY_REF2(FILE*, int, CExprOfBinaryNode*);
void outx_COARRAY_REF(FILE *fp, int indent, CExprOfBinaryNode *expr);
void outx_GCC_BLTIN_OFFSETOF(FILE *fp, int indent, CExprOfBinaryNode *expr);
void outx_GCC_REALPART(FILE *fp, int indent, CExpr *expr);
void outx_GCC_IMAGPART(FILE *fp, int indent, CExpr *expr);
void outx_GCC_ASM_STMT(FILE *fp, int indent, CExprOfBinaryNode *expr);
void outx_GCC_ASM_EXPR(FILE *fp, int indent, CExpr *expr);
void outx_GCC_ASM_EXPR_asAsmDef(FILE *fp, int indent, CExpr *expr);
void outx_GCC_ASM_OPE(FILE *fp, int indent, CExpr *expr);

void out_PRAGMA_COMP_STMT(FILE *fp, int indent, CExpr* expr);

void
xstrcat(char **p, const char *s)
{
    int len = strlen(s);
    strcpy(*p, s);
    *p += len;
}


char*
allocXmlEscapedStr(const char *s)
{
    const char *p = s;
    char c;
    int len = 0;

    while((c = *p++)) {
        switch(c) {
        case '<': case '>': case '&': case '"': case '\'':
            len += 6;
            break;
        default:
            if(c <= 31 || c >= 127)
                len += 6;
            else
                ++len;
        }
    }

    p = s;
    char *x = malloc(len + 1);
    char *px = x;
    char buf[16];

    while((c = *p++)) {
        switch(c) {
        case '<':  xstrcat(&px, "&lt;"); break;
        case '>':  xstrcat(&px, "&gt;"); break;
        case '&':  xstrcat(&px, "&amp;"); break;
        case '"':  xstrcat(&px, "&quot;"); break;
        case '\'': xstrcat(&px, "&apos;"); break;
        default:
            if(c <= 31 || c >= 127) {
                switch(c) {
                case '\a': sprintf(buf, "\\a"); break;
                case '\b': sprintf(buf, "\\b"); break;
                case '\f': sprintf(buf, "\\f"); break;
                case '\n': sprintf(buf, "\\n"); break;
                case '\r': sprintf(buf, "\\r"); break;
                case '\t': sprintf(buf, "\\t"); break;
                case '\v': sprintf(buf, "\\v"); break;
                default:
                    sprintf(buf, "\\x%02x", (unsigned int)(c & 0xFF));
                }
                xstrcat(&px, buf); 
            } else {
                *px++ = c;
            }
            break;
        }
    }

    *px = 0;

    return x;
}


void
outxEscapedStr(FILE *fp, const char *s)
{
    char *xs = allocXmlEscapedStr(s);
    fputs(xs, fp);
    free(xs);
}


const char*
getScope(CExprOfSymbol *sym)
{
    const char *scope;

    if(sym->e_symType == ST_PARAM) {
        scope = "param";
    } else if(sym->e_symType == ST_VAR) {
        if(sym->e_isGlobal)
            scope = "global";
        else
            scope = "local";
    } else {
        scope = NULL;
    }

    return scope;
}


void
outxIndent(FILE *fp, int indent)
{
    for(int i = 0; i < indent; ++i)
        fputs(s_xmlIndent, fp);
}


void
outxPrint(FILE *fp, int indent, const char *fmt, ...)
{
    outxIndent(fp, indent);
    va_list args;
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
}


void
outxContextWithTag(FILE *fp, int indent, CExpr *expr, const char *tag)
{
    if(EXPR_ISNULL(expr)) {
        outxTag1(fp, indent, expr, tag, XATTR_NOCONTENT);
    } else {
        outxTag1(fp, indent, expr, tag, 0);
        outxContext(fp, indent + 1, expr);
        outxTagClose(fp, indent, tag);
    }
}


void
voutxTag(FILE *fp, int indent, CExpr *expr, const char *tag, int xattrFlag,
    const char *attrFmt, va_list args)
{
    outxPrint(fp, indent, "<%s", tag);

    if(attrFmt)
        vfprintf(fp, attrFmt, args);

    if(expr && (xattrFlag & XATTR_TYPE) > 0) {
        const char *typeId;

        if(EXPR_CODE(expr) == EC_TYPE_DESC) {
            typeId = EXPR_T(expr)->e_typeId;
        } else {
            assertExpr(expr, EXPRS_TYPE(expr));
            typeId = EXPR_TYPEID(expr);
        }
        assertExpr(expr, typeId);
        fprintf(fp, " type=\"%s\"", typeId);
    }
    if(expr && (xattrFlag & XATTR_IS_MODIFIED) > 0 && EXPR_ISCONVERTED(expr))
        fprintf(fp, " is_modified=\"1\"");
    if(expr && (xattrFlag & XATTR_IS_GCCSYNTAX) > 0 && EXPR_ISGCCSYNTAX(expr))
        fprintf(fp, " is_gccSyntax=\"1\"");
    if(expr && (xattrFlag & XATTR_IS_GCCEXTENSION) > 0 && EXPR_GCCEXTENSION(expr))
        fprintf(fp, " is_gccExtension=\"1\"");

    if(s_xoutputInfo && expr && (xattrFlag & XATTR_LINENO) > 0) {
        CLineNumInfo *lni = &EXPR_C(expr)->e_lineNumInfo;
        const char *file = getFileNameByFileId(lni->ln_fileId);

        fprintf(fp, " lineno=\"%d\"", lni->ln_lineNum);
        fputs(" file=\"", fp);
        outxEscapedStr(fp, file);
        fputs("\"", fp);

        if(s_rawlineNo)
            fprintf(fp, " rawlineno=\"%d\"", lni->ln_rawLineNum);
#ifdef XCODEML_DEBUG
        fprintf(fp, " p=\"" ADDR_PRINT_FMT "\"", (uintptr_t)expr);
#endif
    }

    if((xattrFlag & XATTR_NOCONTENT) > 0)
        fprintf(fp, "/>");
    else
        fprintf(fp, ">");

    if((xattrFlag & XATTR_NORETURN) == 0)
        fprintf(fp, "\n");
}


void
outxTag(FILE *fp, int indent, CExpr *expr, const char *tag, int xattrFlag,
    const char *attrFmt, ...)
{
    va_list args;
    if(attrFmt) {
        va_start(args, attrFmt);
        voutxTag(fp, indent, expr, tag, xattrFlag, attrFmt, args);
        va_end(args);
    } else {
        voutxTag(fp, indent, expr, tag, xattrFlag, attrFmt, NULL);
    }
}


void
outxTagOnly(FILE *fp, int indent, const char *tag, int xattrFlag)
{
    outxTag(fp, indent, NULL, tag, xattrFlag, NULL);
}


void
outxTag1(FILE *fp, int indent, CExpr *expr, const char *tag, int xattrFlag)
{
    outxTag(fp, indent, expr, tag, xattrFlag, NULL);
}


void
outxTagForStmt(FILE *fp, int indent, CExpr *expr, const char *tag, int addXattrFlag,
    const char *attrFmt, ...)
{
    if(attrFmt) {
        va_list args;
        va_start(args, attrFmt);
        voutxTag(fp, indent, expr, tag,
            XATTR_LINENO|XATTR_COMMON|addXattrFlag,
            attrFmt, args);
        va_end(args);
    } else {
        voutxTag(fp, indent, expr, tag,
            XATTR_LINENO|XATTR_COMMON|addXattrFlag,
            attrFmt, NULL);
    }
}


void
outxTagForExpr(FILE *fp, int indent, CExpr *expr, const char *tag, int addXattrFlag,
    const char *attrFmt, ...)
{
    if(attrFmt) {
        va_list args;
        va_start(args, attrFmt);
        voutxTag(fp, indent, expr, tag,
            XATTR_TYPE|XATTR_IS_GCCEXTENSION|XATTR_COMMON|addXattrFlag,
            attrFmt, args);
        va_end(args);
    } else {
        voutxTag(fp, indent, expr, tag,
            XATTR_TYPE|XATTR_IS_GCCEXTENSION|XATTR_COMMON|addXattrFlag,
            attrFmt, NULL);
    }
}


void
outxTagClose(FILE *fp, int indent, const char *tag)
{
    outxPrint(fp, indent, "</%s>\n", tag);
}


void
outxTagCloseNoIndent(FILE *fp, const char *tag)
{
    fprintf(fp, "</%s>\n", tag);
}


void
outxChildren(FILE *fp, int indent, CExpr *expr)
{
    if(expr == NULL)
        return;

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr)
        if(ite.node)
            outxContext(fp, indent, ite.node);
}


void
outxChildrenForExpr(FILE *fp, int indent, CExpr *expr, const char *tag)
{
    if(hasChildren(expr)) {
        outxTagForExpr(fp, indent, expr, tag, 0, NULL);
        outxChildren(fp, indent + 1, expr);
        outxTagClose(fp, indent, tag);
    } else {
        outxTagForExpr(fp, indent, expr, tag, XATTR_NOCONTENT, NULL);
    }
}


void
outxChildrenForStmt(FILE *fp, int indent, CExpr *expr, const char *tag)
{
    if(hasChildren(expr)) {
        outxTagForStmt(fp, indent, expr, tag, 0, NULL);
        outxChildren(fp, indent + 1, expr);
        outxTagClose(fp, indent, tag);
    } else {
        outxTagForStmt(fp, indent, expr, tag, XATTR_NOCONTENT, NULL);
    }
}


void
outxChildrenWithTag(FILE *fp, int indent, CExpr *expr, const char *tag)
{
    if(hasChildren(expr)) {
        outxTag1(fp, indent, expr, tag, XATTR_COMMON);
        outxChildren(fp, indent + 1, expr);
        outxTagClose(fp, indent, tag);
    } else {
        outxTag1(fp, indent, expr, tag, XATTR_COMMON|XATTR_NOCONTENT);
    }
}


void
outxContext(FILE *fp, int indent, CExpr *expr)
{
    if(expr == NULL)
        return;

    assertExpr(expr, EXPR_ISERROR(expr) == 0);

    switch(EXPR_CODE(expr)) {

    // Skip
    case EC_NULL_NODE:
    case EC_DATA_DEF:
    case EC_DECL:
    case EC_XMP_COARRAY_DECLARATIONS:
    case EC_GCC_LABEL_IDENTS:
    case EC_GCC_LABEL_DECLS:
    case EC_XMP_CRITICAL:
        break;

    // Output only children
    case EC_INIT_DECLS:
    case EC_XMP_COARRAY_DIMENSIONS:
    case EC_GCC_LABEL_ADDR:
    case EC_LABELS:
    case EC_ADDR_OF:
    case EC_BRACED_EXPR:
    case EC_STMTS_AND_DECLS:
        outxChildren(fp, indent, expr);
        break;

    // Constants, Literals
    case EC_CHAR_CONST:
        outx_CHAR_CONST(fp, indent, EXPR_CHARCONST(expr)); break;
    case EC_STRING_CONST:
        outx_STRING_CONST(fp, indent, EXPR_STRINGCONST(expr)); break;
    case EC_NUMBER_CONST:
        outx_NUMBER_CONST(fp, indent, EXPR_NUMBERCONST(expr)); break;
    case EC_COMPOUND_LITERAL:
        outx_COMPOUND_LITERAL(fp, indent, expr); break;

    // Primary Expression
    #define OUTX_OP(tag) outxChildrenForExpr(fp, indent, expr, tag); break;

    case EC_IDENT:
        outx_IDENT(fp, indent, EXPR_SYMBOL(expr)); break;
    case EC_FUNCTION_CALL:
        outx_FUNCTION_CALL(fp, indent, EXPR_B(expr)); break;
    case EC_POINTS_AT:
        outx_POINTS_AT(fp, indent, EXPR_B(expr)); break;
    case EC_ARRAY_REF:
        outx_ARRAY_REF(fp, indent, EXPR_B(expr)); break;
    case EC_CAST:
        outx_CAST(fp, indent, EXPR_B(expr)); break;
    case EC_POINTER_REF:
        OUTX_OP("pointerRef"); break;
    case EC_SIZE_OF:
        OUTX_OP("sizeOfExpr"); break;
    case EC_INITIALIZERS:
        outx_INITIALIZERS(fp, indent, expr); break;
    case EC_INITIALIZER:
        outxContext(fp, indent, EXPR_B(expr)->e_nodes[1]); break;
    case EC_TYPE_DESC:
        outx_TYPE_DESC(fp, indent, EXPR_T(expr)); break;
    case EC_XMP_COARRAY_REF:
        outx_COARRAY_REF(fp, indent, EXPR_B(expr)); break;
    case EC_XMP_DESC_OF:
        // OUTX_OP("xmpDescOf"); 
        outxTagForExpr(fp, indent, expr, "xmpDescOf", 0, NULL);
	outxPrint(fp, indent+1, 
		  "<Var type=\"int\" scope=\"global\">%s</Var>\n",
		  EXPR_SYMBOL(EXPR_L_AT(expr,0))->e_symName);
        outxTagClose(fp, indent, "xmpDescOf");
	break;

    // Operators
    case EC_UNARY_MINUS:    OUTX_OP("unaryMinusExpr"); break;
    case EC_BIT_NOT:        OUTX_OP("bitNotExpr"); break;
    case EC_LOG_NOT:        OUTX_OP("logNotExpr"); break;
    case EC_PRE_INCR:       OUTX_OP("preIncrExpr"); break;
    case EC_PRE_DECR:       OUTX_OP("preDecrExpr"); break;
    case EC_POST_INCR:      OUTX_OP("postIncrExpr"); break;
    case EC_POST_DECR:      OUTX_OP("postDecrExpr"); break;
    case EC_LSHIFT:         OUTX_OP("LshiftExpr"); break;
    case EC_RSHIFT:         OUTX_OP("RshiftExpr"); break;
    case EC_PLUS:           OUTX_OP("plusExpr"); break;
    case EC_MINUS:          OUTX_OP("minusExpr"); break;
    case EC_MUL:            OUTX_OP("mulExpr"); break;
    case EC_DIV:            OUTX_OP("divExpr"); break;
    case EC_MOD:            OUTX_OP("modExpr"); break;
    case EC_ARITH_EQ:       OUTX_OP("logEQExpr"); break;
    case EC_ARITH_NE:       OUTX_OP("logNEQExpr"); break;
    case EC_ARITH_GE:       OUTX_OP("logGEExpr"); break;
    case EC_ARITH_GT:       OUTX_OP("logGTExpr"); break;
    case EC_ARITH_LE:       OUTX_OP("logLEExpr"); break;
    case EC_ARITH_LT:       OUTX_OP("logLTExpr"); break;
    case EC_LOG_AND:        OUTX_OP("logAndExpr"); break;
    case EC_LOG_OR:         OUTX_OP("logOrExpr"); break;
    case EC_BIT_AND:        OUTX_OP("bitAndExpr"); break;
    case EC_BIT_OR:         OUTX_OP("bitOrExpr"); break;
    case EC_BIT_XOR:        OUTX_OP("bitXorExpr"); break;
    case EC_ASSIGN:         outx_ASSIGN(fp, indent, expr); break;
    case EC_ASSIGN_PLUS:    OUTX_OP("asgPlusExpr"); break;
    case EC_ASSIGN_MINUS:   OUTX_OP("asgMinusExpr"); break;
    case EC_ASSIGN_MUL:     OUTX_OP("asgMulExpr"); break;
    case EC_ASSIGN_DIV:     OUTX_OP("asgDivExpr"); break;
    case EC_ASSIGN_MOD:     OUTX_OP("asgModExpr"); break;
    case EC_ASSIGN_LSHIFT:  OUTX_OP("asgLshiftExpr"); break;
    case EC_ASSIGN_RSHIFT:  OUTX_OP("asgRshiftExpr"); break;
    case EC_ASSIGN_BIT_AND: OUTX_OP("asgBitAndExpr"); break;
    case EC_ASSIGN_BIT_OR:  OUTX_OP("asgBitOrExpr"); break;
    case EC_ASSIGN_BIT_XOR: OUTX_OP("asgBitXorExpr"); break;
    case EC_CONDEXPR:       OUTX_OP("condExpr"); break;
    case EC_EXPRS:          outx_EXPRS(fp, indent, expr); break;

    // Statements
    #define OUTX_SIMPLE_STMT(tag) outxChildrenForStmt(fp, indent, expr, tag);
    case EC_COMP_STMT:
	if(((CExprOfList *)expr)->e_aux_info != NULL){
	    out_PRAGMA_COMP_STMT(fp, indent, expr);
	    break;
	}
	outx_COMP_STMT(fp, indent, expr); break;
    case EC_IF_STMT:
        outx_IF_STMT(fp, indent, expr); break;
    case EC_WHILE_STMT:
        outx_WHILE_STMT(fp, indent, EXPR_B(expr)); break;
    case EC_DO_STMT:
        outx_DO_STMT(fp, indent, EXPR_B(expr)); break;
    case EC_FOR_STMT:
        outx_FOR_STMT(fp, indent, expr); break;
    case EC_SWITCH_STMT:
        outx_SWITCH_STMT(fp, indent, EXPR_B(expr)); break;
    case EC_CASE_LABEL:
        outx_CASE_LABEL(fp, indent, EXPR_B(expr)); break;
    case EC_DEFAULT_LABEL:
        OUTX_SIMPLE_STMT("defaultLabel"); break;
    case EC_EXPR_STMT:
        if(EXPR_ISNULL(EXPR_U(expr)->e_node) == 0)
            OUTX_SIMPLE_STMT("exprStatement"); break;
    case EC_GOTO_STMT:
        OUTX_SIMPLE_STMT("gotoStatement"); break;
    case EC_CONTINUE_STMT:
        OUTX_SIMPLE_STMT("continueStatement"); break;
    case EC_RETURN_STMT:
        OUTX_SIMPLE_STMT("returnStatement"); break;
    case EC_BREAK_STMT:
        OUTX_SIMPLE_STMT("breakStatement"); break;
    case EC_LABEL:
        OUTX_SIMPLE_STMT("statementLabel"); break;
    case EC_DIRECTIVE:
        outx_DIRECTIVE(fp, indent, EXPR_DIRECTIVE(expr)); break;

    // Definitions / Declarations
    case EC_EXT_DEFS:
        outx_EXT_DEFS(fp, indent, expr); break;
    case EC_FUNC_DEF:
        outx_FUNC_DEF(fp, indent, expr); break;
    case EC_INIT_DECL:
        outx_INIT_DECL(fp, indent, expr); break;

    // GCC specific
    #define OUTX_BUILTIN_OP(name) outxBuiltinOpCall(fp, indent, name, expr, expr)
    case EC_GCC_COMP_STMT_EXPR:
        OUTX_OP("gccCompoundExpr"); break;
    case EC_GCC_ALIGN_OF:
        OUTX_OP("gccAlignOfExpr"); break;
    case EC_GCC_BLTIN_VA_ARG:
        OUTX_BUILTIN_OP("__builtin_va_arg"); break;
    case EC_GCC_BLTIN_TYPES_COMPATIBLE_P:
        OUTX_BUILTIN_OP("__builtin_types_compatible_p"); break;
    case EC_GCC_BLTIN_OFFSET_OF:
        outx_GCC_BLTIN_OFFSETOF(fp, indent, EXPR_B(expr)); break;
    case EC_GCC_ASM_STMT:
        outx_GCC_ASM_STMT(fp, indent, EXPR_B(expr)); break;
    case EC_GCC_ASM_EXPR:
        outx_GCC_ASM_EXPR(fp, indent, expr); break;
    case EC_GCC_ASM_OPE:
        outx_GCC_ASM_OPE(fp, indent, expr); break;
    case EC_GCC_REALPART:
        outx_GCC_REALPART(fp, indent, expr); break;
    case EC_GCC_IMAGPART:
        outx_GCC_IMAGPART(fp, indent, expr); break;

    // Illegal elements
    case EC_DECLARATOR:
    case EC_DATA_DECL:
    case EC_DATA_DECLS:
    case EC_DESIGNATORS:
    case EC_ARRAY_DESIGNATOR:
    case EC_INIT:
    case EC_IDENTS:
    case EC_STRUCT_TYPE:
    case EC_UNION_TYPE:
    case EC_ENUM_TYPE:
    case EC_MEMBER_REF:
    case EC_MEMBERS:
    case EC_MEMBER_DECL:
    case EC_MEMBER_DECLS:
    case EC_MEMBER_DECLARATOR:
    case EC_ENUMERATORS:
    case EC_PARAMS:
    case EC_ELLIPSIS:
    case EC_STRINGS:
    case EC_DECL_SPECS:
    case EC_LDECLARATOR:
    case EC_ARRAY_DECL:
    case EC_COARRAY_DECL:
    case EC_POINTER_DECL:
    case EC_PARAM:
    case EC_TYPENAME:
    case EC_SCSPEC:
    case EC_TYPEQUAL:
    case EC_TYPESPEC:
    case EC_ERROR_NODE:
    case EC_ARRAY_DIMENSION:
    case EC_GCC_OFS_MEMBER_REF:
    case EC_GCC_OFS_ARRAY_REF:
    case EC_GCC_EXTENSION:
    case EC_GCC_ATTRS:
    case EC_GCC_ATTR_ARG:
    case EC_GCC_ATTR_ARGS:
    case EC_GCC_TYPEOF:
    case EC_GCC_ASM_ARG:
    case EC_GCC_ASM_OPES:
    case EC_GCC_ASM_CLOBS:
    case EC_XMP_COARRAY_DECLARATION:
    case EC_XMP_COARRAY_DIM_DEFS:
    case EC_FLEXIBLE_STAR:
    case EC_PRAGMA_PACK:
    case EC_UNDEF:
    case EC_END:
        assertExpr(expr, 0);
        ABORT();
    }
}


void
outxGccAttr(FILE *fp, int indent, CExpr *expr, CGccAttrKindEnum gak)
{
    CExpr *attrs = NULL;
    if(EXPR_CODE(expr) == EC_GCC_ATTRS)
        attrs = expr;
    else
        attrs = EXPR_C(expr)->e_gccAttrPre;

    if(EXPR_ISNULL(attrs) || EXPR_L_SIZE(attrs) == 0)
        return;

    CCOL_DListNode *ite;
    const char *tag = "gccAttribute";
    EXPR_FOREACH(ite, attrs) {
        CExprOfBinaryNode *arg = EXPR_B(exprListHeadData(EXPR_L_DATA(ite)));
        assertExpr(attrs, (CExpr*)arg);

        if((arg->e_gccAttrKind & gak) > 0) {

            arg->e_gccAttrOutput = 1;
            const char *symName = arg->e_gccAttrInfo->ga_symbol;
            CExpr *argExprs = arg->e_nodes[1];

            if(EXPR_L_ISNULL(argExprs)) {
                outxTag(fp, indent, (CExpr*)arg, tag, XATTR_NOCONTENT,
                        " name=\"%s\"", symName);
            } else {
                outxTag(fp, indent, (CExpr*)arg, tag, 0,
                        " name=\"%s\"", symName);
                outxChildren(fp, indent + 1, argExprs);
                outxTagClose(fp, indent, tag);
            }
        }
    }
}


void
outxGccAttrVarOrFunc(FILE *fp, int indent, CExprOfTypeDesc *td, CGccAttrKindEnum gak)
{
    CExpr *attrs = exprList(EC_GCC_ATTRS);
    getGccAttrRecurse(attrs, td, gak);

    if(EXPR_L_SIZE(attrs) == 0)
        goto end;

    const char *tag = "gccAttributes";
    outxTagOnly(fp, indent, tag, 0);
    outxGccAttr(fp, indent + 1, attrs, gak);
    outxTagClose(fp, indent, tag);

  end:
    freeExpr(attrs);
}


#define NO_CLOSE        0
#define CLOSE_NO_CHILD  1
#define CLOSE_CHILD     2

void
outxTypeDescClosureAndGccAttr(
    FILE *fp, int indent, CExprOfTypeDesc *td, CGccAttrKindEnum gak,
    const char *tag, int hasGA, int needsClosure)
{
    if(hasGA) {
        if(needsClosure == CLOSE_NO_CHILD)
            fprintf(fp, ">\n");

        const char *gccAttrsTag = "gccAttributes";
        outxTagOnly(fp, indent, gccAttrsTag, 0);
        outxGccAttr(fp, indent + 1, (CExpr*)td, gak);
        outxTagClose(fp, indent, gccAttrsTag);
        if(needsClosure != NO_CLOSE)
            outxTagClose(fp, indent - 1, tag);

    } else {
        if(needsClosure == CLOSE_NO_CHILD)
            fprintf(fp, "/>\n");
        else if(needsClosure == CLOSE_CHILD) {
            outxTagClose(fp, indent - 1, tag);
        }
    }
}


void
outxStructOrUnionType(FILE *fp, int indent, CExprOfTypeDesc *typeDesc, const char *tag, int hasGA)
{
    CExpr *memDecls = getMemberDeclsExpr(typeDesc);

    if(EXPR_ISNULL(memDecls)) {
        //null node represents no member declaration block (used for pointer)
        outxTypeDescClosureAndGccAttr(fp, indent + 1, typeDesc,
            GAK_TYPE, tag, hasGA, CLOSE_NO_CHILD);
        return;
    }

    fprintf(fp, ">\n");
    assertExprCode(memDecls, EC_MEMBER_DECLS);
    CCOL_DListNode *ite1, *ite2;

    outxTypeDescClosureAndGccAttr(fp, indent + 1, typeDesc,
        GAK_TYPE, tag, hasGA, NO_CLOSE);

    const char *symbolsTag = "symbols";
    if(hasChildren(memDecls) == 0) {
        outxTagOnly(fp, indent + 1, symbolsTag, XATTR_NOCONTENT);
    } else {
        outxTagOnly(fp, indent + 1, symbolsTag, 0);

        EXPR_FOREACH(ite1, memDecls) {
            CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
            if(EXPR_IS_MEMDECL(memDecl) ==0)
                continue;
            assertExprCode((CExpr*)memDecl, EC_MEMBER_DECL);
            CExprOfList *mems = EXPR_L(memDecl->e_nodes[1]);
            assertExprCode((CExpr*)mems, EC_MEMBERS);

            EXPR_FOREACH(ite2, mems) {
                CExprOfBinaryNode *memDeclr = EXPR_B(EXPR_L_DATA(ite2));
                assertExprCode((CExpr*)memDeclr, EC_MEMBER_DECLARATOR);
                CExprOfBinaryNode *declr = EXPR_B(memDeclr->e_nodes[0]);
                CExprOfTypeDesc *memType;
                assertExprCode((CExpr*)declr, EC_DECLARATOR);
                memType = EXPR_T(declr->e_nodes[0]);
                assertExpr((CExpr*)declr, memType);
                assertExprCode((CExpr*)memType, EC_TYPE_DESC);
                CExprOfSymbol *sym = EXPR_SYMBOL(declr->e_nodes[1]);

                s_charBuf[0][0] = 0;
                s_charBuf[1][0] = 0;
                int bl = memType->e_bitLen;
                int isBfAttr = (memType->e_bitLenExpr && bl > 0);
                int isBfTag = (memType->e_bitLenExpr && bl <= 0);
                if(isBfAttr)
                    sprintf(s_charBuf[0], " bit_field=\"%d\"", bl);
                else if(isBfTag)
                    sprintf(s_charBuf[0], " bit_field=\"*\"");

                if(EXPR_C(memDecl)->e_gccExtension)
                    sprintf(s_charBuf[1], " is_gccExtension=\"1\"");

                const char *idTag = "id";
                outxTag(fp, indent + 2, (CExpr*)memType, idTag,
                    XATTR_TYPE|XATTR_COMMON, "%s%s",
                   s_charBuf[0], s_charBuf[1]);

                if(sym) {
                    outxPrint(fp, indent + 3, "<name>%s</name>\n", sym->e_symName);
                } else {
                    outxPrint(fp, indent + 3, "<name/>\n");
                }

                if(isBfTag)
                    outxContextWithTag(fp, indent + 3, memType->e_bitLenExpr, "bitField");

                outxGccAttrVarOrFunc(fp, indent + 3, memType, GAK_VAR);

                outxTagClose(fp, indent + 2, idTag);
            }
        }
        outxTagClose(fp, indent + 1, symbolsTag);
    }
    outxTagClose(fp, indent, tag);
}


void
outxEnumType(FILE *fp, int indent, CExprOfTypeDesc *typeDesc, const char *tag, int hasGA)
{
    CExpr *enums = exprListNextNData(typeDesc->e_typeExpr, 2);
    int indent1 = indent + 1;

    if(EXPR_ISNULL(enums)) {
        outxTypeDescClosureAndGccAttr(fp, indent1, typeDesc,
            GAK_TYPE, tag, hasGA, CLOSE_NO_CHILD);
        return;
    }

    fprintf(fp, ">\n");
    outxTypeDescClosureAndGccAttr(fp, indent + 1, typeDesc,
        GAK_TYPE, tag, hasGA, NO_CLOSE);

    const char *symbolsTag = "symbols";
    const char *idTag = "id";
    const char *valueTag = "value";
    outxTagOnly(fp, indent1, symbolsTag, 0);

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, enums) {
        CExprOfSymbol *sym = EXPR_SYMBOL(EXPR_L_DATA(ite));
        outxTagOnly(fp, indent1 + 1, idTag, 0);
        outxPrint(fp, indent1 + 2, "<name>%s</name>\n", sym->e_symName);
        CExpr *val = sym->e_valueExpr;
        if(sym->e_isEnumInited && EXPR_ISNULL(val) == 0)
            outxContextWithTag(fp, indent1 + 2, val, valueTag);
        outxTagClose(fp, indent1 + 1, idTag);
    }

    outxTagClose(fp, indent1, symbolsTag);
    outxTagClose(fp, indent, tag);
}


void
outxArrayType(FILE *fp, int indent, CExprOfTypeDesc *td, const char *tag, int hasGA, const char *refId)
{
    if(td->e_len.eln_isStatic)
        fprintf(fp, " is_static=\"1\"");
    if(td->e_len.eln_isConst)
        fprintf(fp, " is_const=\"1\"");
    if(td->e_len.eln_isVolatile)
        fprintf(fp, " is_volatile=\"1\"");
    if(td->e_len.eln_isRestrict)
        fprintf(fp, " is_restrict=\"1\"");

    fprintf(fp, " element_type=\"%s\"", refId);
    CExpr *lenExpr = td->e_len.eln_lenExpr;

    int needsClosure = CLOSE_NO_CHILD;
    const char *arraySizeTag = "arraySize";
    int hasArraySizeTag = 0;

    if(EXPR_ISNULL(lenExpr) == 0) {
        if(EXPR_CODE(lenExpr) == EC_NUMBER_CONST) {
            long sz = getNumberConstAsLong(lenExpr);
            fprintf(fp, " array_size=\"%ld\"", sz);
        } else {
            fprintf(fp, " array_size=\"*\">\n");
            hasArraySizeTag = 1;
            needsClosure = NO_CLOSE;
        }
    } else if(td->e_len.eln_isVariable) {
        needsClosure = CLOSE_NO_CHILD;
    } else if(td->e_tdKind == TD_COARRAY) {   // co-index [*] 
        fprintf(fp, " array_size=\"*\"");
    } else {
        fprintf(fp, " array_size=\"0\"");
    }

    outxTypeDescClosureAndGccAttr(fp, indent + 1, td,
        GAK_TYPE, tag, hasGA, needsClosure);

    if(hasArraySizeTag) {
        outxContextWithTag(fp, indent + 1, lenExpr, arraySizeTag);
        outxTagClose(fp, indent, tag);
    }
}


void
outxFuncParams(FILE *fp, int indent, CExpr *params)
{
    const char *paramTag = "params";
    const char *ellipsisTag = "ellipsis";

    if(hasChildren(params) == 0) {
        outxTagOnly(fp, indent, paramTag, XATTR_NOCONTENT);
    } else {
        outxTagOnly(fp, indent, paramTag, 0);

        CCOL_DListNode *ite;
        EXPR_FOREACH(ite, params) {
            CExprOfBinaryNode *declr = EXPR_B(EXPR_L_DATA(ite));
            if(EXPR_CODE(declr) == EC_ELLIPSIS) {
                outxTagOnly(fp, indent + 1, ellipsisTag, XATTR_NOCONTENT);
            } else {
                CExprOfTypeDesc *paramType = EXPR_T(declr->e_nodes[0]);
                CExprOfSymbol *paramSym = EXPR_SYMBOL(declr->e_nodes[1]);

                if(EXPR_ISNULL(paramSym) || paramSym->e_symName == NULL)
                    outxPrint(fp, indent + 1, "<name type=\"%s\"/>\n", paramType->e_typeId);
                else 
                    outxPrint(fp, indent + 1, "<name type=\"%s\">%s</name>\n",
                        paramType->e_typeId, paramSym->e_symName);
            }
        }
        outxTagClose(fp, indent, paramTag);
    }
}


void
outxFuncType(FILE *fp, int indent, CExprOfTypeDesc *typeDesc, const char *tag, int hasGA)
{
    fprintf(fp, " return_type=\"%s\"", EXPR_T(typeDesc->e_typeExpr)->e_typeId);
    CExpr *params = typeDesc->e_paramExpr;

    if(EXPR_ISNULL(params)) {
        outxTypeDescClosureAndGccAttr(fp, indent + 1, typeDesc,
            GAK_TYPE, tag, hasGA, CLOSE_NO_CHILD);
        return;
    }

    fprintf(fp, ">\n");
    outxTypeDescClosureAndGccAttr(fp, indent + 1, typeDesc,
        GAK_TYPE, tag, hasGA, NO_CLOSE);
    outxFuncParams(fp, indent + 1, params);
    outxTagClose(fp, indent, tag);
}


void
outxTypeDesc(FILE *fp, int indent, CExprOfTypeDesc *td)
{
    if(td->e_isDuplicated || td->e_isMarked ||
        td->e_tdKind == TD_GCC_BUILTIN_ANY)
        return;

    td->e_isMarked = 1;
    const char *tag = NULL;
    CTypeDescKindEnum tdKind = td->e_tdKind;
    CExprOfTypeDesc *refType = NULL;
    const CGccAttrKindEnum gak = GAK_TYPE;
    int hasGA = hasGccAttr(td, gak);

    switch(tdKind) {
    case TD_GCC_BUILTIN:
    case TD_BASICTYPE:
        if(isTypeQualOrExtensionSet(td) == 0 && hasGA == 0)
            return;
        tag = "basicType";
        break;
    case TD_STRUCT:
    case TD_UNION:
    case TD_ENUM:
        if(td->e_refType) {
            if(td->e_isDifferentQaulifierFromRefType == 0)
                return;
            tdKind = TD_DERIVED;
            tag = "basicType";
            refType = td->e_refType;
        } else {
            tag = (tdKind == TD_STRUCT) ? "structType":
                ((tdKind == TD_UNION) ? "unionType" : "enumType");
        }
        break;
    case TD_POINTER:
        tag = "pointerType";
        refType = EXPR_T(td->e_typeExpr);
        break;
    case TD_ARRAY:
        tag = "arrayType";
        refType = EXPR_T(td->e_typeExpr);
        break;
    case TD_FUNC:
        tag = "functionType";
        break;
    case TD_COARRAY:
        tag = "coArrayType";
        refType = EXPR_T(td->e_typeExpr);
        break;
    case TD_DERIVED:
        if(td->e_isDifferentQaulifierFromRefType == 0)
            return;
        tag = "basicType";
        refType = td->e_refType;
        break;
    case TD_GCC_BUILTIN_ANY:
    case TD_TYPEREF:
    case TD_GCC_TYPEOF:
    case TD_FUNC_OLDSTYLE:
    case TD_UNDEF:
    case TD_END:
        assertExpr((CExpr*)td, 0);
        ABORT();
    }

    outxPrint(fp, indent, "<%s type=\"%s\"", tag, td->e_typeId);

    /* no size/align
    int tsize = getTypeSize(td), talign = getTypeAlign(td);
    if(tsize > 0)
        fprintf(fp, " size=\"%d\"", tsize);

    if(talign > 0)
        fprintf(fp, " align=\"%d\"", talign);
    */

    if(td->e_tq.etq_isConst)
        fprintf(fp, " is_const=\"1\"");
    if(td->e_tq.etq_isVolatile)
        fprintf(fp, " is_volatile=\"1\"");
    if(td->e_tq.etq_isRestrict)
        fprintf(fp, " is_restrict=\"1\"");
    if(tdKind == TD_FUNC && td->e_tq.etq_isInline)
        fprintf(fp, " is_inline=\"1\"");

#ifdef XCODEML_DEBUG
    fprintf(fp, " p=\"" ADDR_PRINT_FMT "\"", (uintptr_t)td);
    fprintf(fp, " rawlineno=\"%d\"", EXPR_C(td)->e_lineNumInfo.ln_rawLineNum);
#endif

    switch(tdKind) {
    case TD_BASICTYPE:
        fprintf(fp, " name=\"%s\"", s_CBasicTypeEnumXcodes[td->e_basicType]);
        outxTypeDescClosureAndGccAttr(fp, indent + 1, td,
            gak, tag, hasGA, CLOSE_NO_CHILD);
        break;
    case TD_GCC_BUILTIN:
        fprintf(fp, " name=\"%s\"", EXPR_SYMBOL(td->e_typeExpr)->e_symName);
        outxTypeDescClosureAndGccAttr(fp, indent + 1, td,
            gak, tag, hasGA, CLOSE_NO_CHILD);
        break;
    case TD_DERIVED: {
            fprintf(fp, " name=\"%s\"", refType->e_typeId);
            outxTypeDescClosureAndGccAttr(fp, indent + 1, td,
                gak, tag, hasGA, CLOSE_NO_CHILD);
        }
        break;
    case TD_POINTER:
        fprintf(fp, " ref=\"%s\"", refType->e_typeId);
#ifdef XCODEML_DEBUG
        fprintf(fp, " ref_p=\"" ADDR_PRINT_FMT "\"", (uintptr_t)EXPR_T(td->e_typeExpr));
#endif
        outxTypeDescClosureAndGccAttr(fp, indent + 1, td,
            gak, tag, hasGA, CLOSE_NO_CHILD);
        break;
    case TD_STRUCT:
    case TD_UNION:
        outxStructOrUnionType(fp, indent, td, tag, hasGA);
        break;
    case TD_ENUM:
        outxEnumType(fp, indent, td, tag, hasGA);
        break;
    case TD_COARRAY:
    case TD_ARRAY:
        outxArrayType(fp, indent, td, tag, hasGA, refType->e_typeId);
        break;
    case TD_FUNC:
        outxFuncType(fp, indent, td, tag, hasGA);
        break;
    case TD_TYPEREF:
    case TD_GCC_TYPEOF:
    case TD_GCC_BUILTIN_ANY:
    case TD_FUNC_OLDSTYLE:
    case TD_UNDEF:
    case TD_END:
        ABORT();
    }
}


CExprOfTypeDesc*
getDuplicatedType(CExprOfTypeDesc *td)
{
    if(td == NULL)
        return NULL;

    int tdKind = td->e_tdKind;

    switch(tdKind) {
    case TD_BASICTYPE:
    case TD_POINTER:
    case TD_ARRAY:
    case TD_COARRAY:
        break;
    default:
        return NULL;
    }

    if(EXPR_ISNULL(EXPR_C(td)->e_gccAttrPre) == 0)
        return NULL;

    CCOL_DListNode *ite;
    CCOL_DL_FOREACH(ite, &s_typeDescList) {
        CExprOfTypeDesc *td1 = EXPR_T(CCOL_DL_DATA(ite));

        if(td != td1 && td1->e_isMarked &&
            tdKind == td1->e_tdKind &&
            isTypeQualEquals(td, td1) &&
            EXPR_ISNULL(EXPR_C(td1)->e_gccAttrPre)) {

            switch(tdKind) {
            case TD_BASICTYPE:
                if(td->e_basicType == td1->e_basicType)
                    return td1;
                break;
            case TD_POINTER: {
                    assertExpr((CExpr*)td, td->e_typeExpr);
                    assertExpr((CExpr*)td1, td1->e_typeExpr);
                    const char *rid = EXPR_T(td->e_typeExpr)->e_typeId;
                    const char *rid1 = EXPR_T(td1->e_typeExpr)->e_typeId;
                    if((td->e_typeExpr == td1->e_typeExpr) || (rid && rid1 && strcmp(rid, rid1) == 0))
                        return td1;
                }
                break;
            case TD_ARRAY:
            case TD_COARRAY: {
                    if(compareType(td, td1) == CMT_EQUAL)
                        return td1;
                }
                break;
            default:
                break;
            }
        }
    }

    return NULL;
}


void
setTypeId(CExprOfTypeDesc *td)
{
    if(td->e_typeId || td->e_isMarked)
        return;

    td->e_isMarked = 1;

    CTypeDescKindEnum tdKind = td->e_tdKind;
    CExprOfTypeDesc *tdo = getRefType(td);
    if(ETYP_IS_FUNC(tdo) == 0)
        td->e_tq.etq_isInline = 0;
    const CGccAttrKindEnum gak = GAK_TYPE;
    int qualified = (isTypeQualOrExtensionSet(td) || hasGccAttr(td, gak));
    char h = 0;
    int *seq = NULL;

    switch(tdKind) {
    case TD_BASICTYPE:
        if(qualified == 0) {
            CBasicTypeEnum btEnum = td->e_basicType;
            td->e_typeId = ccol_strdup(s_CBasicTypeEnumXcodes[btEnum], MAX_TYPEID_SIZE);
            return;
        }
        break;
    case TD_GCC_BUILTIN:
        if(qualified == 0) {
            td->e_typeId = ccol_strdup(EXPR_SYMBOL(td->e_typeExpr)->e_symName, MAX_TYPEID_SIZE);
            return;
        }
        break;
    case TD_GCC_BUILTIN_ANY:
        return;
    case TD_POINTER:
    case TD_FUNC:
    case TD_ARRAY:
    case TD_COARRAY:
        if(td->e_typeExpr)
            setTypeId(EXPR_T(td->e_typeExpr));
        break;
    case TD_STRUCT:
    case TD_UNION:
    case TD_ENUM:
        if(td->e_refType) {
            if(qualified) {
                td->e_isDifferentQaulifierFromRefType = 1;
            } else {
                CExprOfTypeDesc *refType = getRefType(td->e_refType);
                const char *refTypeId = refType->e_typeId;
                if(refTypeId == NULL) {
                    setTypeId(refType);
                    refTypeId = refType->e_typeId;
                }

                td->e_typeId = ccol_strdup(refTypeId, MAX_TYPEID_SIZE);
                return;
            }
        }
        break;
    case TD_DERIVED: {
            CExprOfTypeDesc *refTd = td->e_refType;
            setTypeId(refTd);
            if(qualified) {
                tdKind = getRefType(refTd)->e_tdKind;
                td->e_isDifferentQaulifierFromRefType = 1;
            } else {
                td->e_typeId = ccol_strdup(refTd->e_typeId, MAX_TYPEID_SIZE);
                return;
            }
        }
        break;
    case TD_TYPEREF:
    case TD_GCC_TYPEOF:
    case TD_FUNC_OLDSTYLE:
    case TD_UNDEF:
    case TD_END:
        ABORT();
    }

    if(s_suppressSameTypes) {
        CExprOfTypeDesc *td0 = getDuplicatedType(td);
        if(td0) {
            td->e_typeId = ccol_strdup(td0->e_typeId, MAX_TYPEID_SIZE);
            td->e_isDuplicated = 1;
            return;
        }
    }

    switch(tdKind) {
    case TD_GCC_BUILTIN:
    case TD_BASICTYPE:
        h = 'B';
        seq = &s_tidSeqBasicType;
        break;
    case TD_STRUCT:
        h = 'S';
        seq = &s_tidSeqStruct;
        break;
    case TD_UNION:
        h = 'U';
        seq = &s_tidSeqUnion;
        break;
    case TD_ENUM:
        h = 'E';
        seq = &s_tidSeqEnum;
        break;
    case TD_POINTER:
        h = 'P';
        seq = &s_tidSeqPointer;
        break;
    case TD_ARRAY:
        h = 'A';
        seq = &s_tidSeqArray;
        break;
    case TD_FUNC:
        h = 'F';
        seq = &s_tidSeqFunc;
        break;
    case TD_COARRAY:
        h = 'C';
        seq = &s_tidSeqCoarray;
        break;
    case TD_TYPEREF:
    case TD_DERIVED:
    case TD_GCC_TYPEOF:
    case TD_GCC_BUILTIN_ANY:
    case TD_FUNC_OLDSTYLE:
    case TD_UNDEF:
    case TD_END:
        assertExpr((CExpr*)td, 0);
        ABORT();
    }

    int tid = (*seq)++;
    int tmp = tid, col = 0;
    if(tmp) {
        while(tmp > 0) {
            ++col;
            tmp /= 10;
        }
    } else {
        col = 1;
    }

    assertExpr((CExpr*)td, td->e_typeId == NULL);
    char *buf = (char*)malloc(col + 2);
    sprintf(buf, "%c%d", h, tid);
    td->e_typeId = buf;
}


void
setTypeIds()
{
    CCOL_DListNode *ite;

    CCOL_DL_FOREACH(ite, &s_typeDescList) {
        (EXPR_T(CCOL_DL_DATA(ite)))->e_isMarked = 0;
    }

    CCOL_DL_FOREACH(ite, &s_typeDescList) {
        CExprOfTypeDesc *td = EXPR_T(CCOL_DL_DATA(ite));
        if(ETYP_IS_BASICTYPE(td))
            setTypeId(td);
    }

    CCOL_DL_FOREACH(ite, &s_typeDescList) {
        CExprOfTypeDesc *td = EXPR_T(CCOL_DL_DATA(ite));
        if(ETYP_IS_BASICTYPE(td) == 0)
            setTypeId(td);
    }

    CCOL_DL_FOREACH(ite, &s_typeDescList) {
        (EXPR_T(CCOL_DL_DATA(ite)))->e_isMarked = 0;
    }
}


void
outxTypeDescByKind(FILE *fp, int indent, CTypeDescKindEnum tdKind)
{
    CCOL_DListNode *ite;
    CCOL_DL_FOREACH(ite, &s_typeDescList) {
        CExprOfTypeDesc *td = EXPR_T(CCOL_DL_DATA(ite));
        if(td->e_isNoTypeId == 0 && (tdKind == TD_UNDEF || (td->e_tdKind == tdKind)))
            outxTypeDesc(fp, indent, td);
    }
}


void
outxTypeTable(FILE *fp, int indent)
{
    outxPrint(fp, indent, "<typeTable>\n");

    int nindent = indent + 1;
    outxTypeDescByKind(fp, nindent, TD_BASICTYPE);
    outxTypeDescByKind(fp, nindent, TD_GCC_BUILTIN);
    outxTypeDescByKind(fp, nindent, TD_POINTER);
    outxTypeDescByKind(fp, nindent, TD_ARRAY);
    outxTypeDescByKind(fp, nindent, TD_COARRAY);
    outxTypeDescByKind(fp, nindent, TD_STRUCT);
    outxTypeDescByKind(fp, nindent, TD_UNION);
    outxTypeDescByKind(fp, nindent, TD_ENUM);
    outxTypeDescByKind(fp, nindent, TD_FUNC);
    outxTypeDescByKind(fp, nindent, TD_UNDEF);

    outxPrint(fp, indent, "</typeTable>\n");
}


const char*
getSclassOfFunc(CExprOfSymbol* sym, int isGlobal)
{
    CExprOfTypeDesc *td = EXPRS_TYPE(sym);
    assertExpr((CExpr*)sym, td);
    assertExpr((CExpr*)sym, ETYP_IS_FUNC(getRefType(td)));

    if(td->e_sc.esc_isExtern)
        return "extern";
    if(td->e_sc.esc_isStatic)
        return "static";

    //auto means nested function
    return isGlobal ? "extern_def" : "auto";
}


const char*
getSclassOfVar(CExprOfSymbol* sym, int isGlobal)
{
    CExprOfTypeDesc *td = EXPRS_TYPE(sym);
    assertExpr((CExpr*)sym, td);
    assertExpr((CExpr*)sym, ETYP_IS_FUNC(getRefType(td)) == 0);

    if(td->e_sc.esc_isExtern)
        return "extern";
    if(td->e_sc.esc_isStatic)
        return "static";
    if(td->e_sc.esc_isRegister)
        return "register";

    return isGlobal ? "extern_def" : "auto";
}


int
compareSymbolOrder(const void *v1, const void *v2)
{
    CExprOfSymbol *s1 = *(CExprOfSymbol**)v1;
    CExprOfSymbol *s2 = *(CExprOfSymbol**)v2;
    int o1 = s1->e_putOrder;
    int o2 = s2->e_putOrder;
    return (o1 == o2) ? 0 : ((o1 < o2) ? -1 : 1);
}


void
collectSymbols(CExprOfSymbol **syms, CCOL_HashTable *ht, int *idx)
{
    CCOL_HashEntry *he;
    CCOL_HashSearch hs;
    int i = *idx;

    CCOL_HT_FOREACH(he, hs, ht) {
        CExprOfSymbol *sym = CCOL_HT_DATA(he);
        syms[i++] = sym;
    }

    *idx = i;
}


void
outxSymbols0(FILE *fp, int indent, CSymbolTable *symTab, int symbolsFlags)
{
    int sz = CCOL_HT_SIZE(&symTab->stb_identGroup) +
        CCOL_HT_SIZE(&symTab->stb_tagGroup) +
        CCOL_HT_SIZE(&symTab->stb_labelGroup);

    if(sz == 0)
        return;

    CExprOfSymbol **syms = malloc(sizeof(CExprOfSymbol*) * sz);
    int idx = 0;
    collectSymbols(syms, &symTab->stb_identGroup, &idx);
    collectSymbols(syms, &symTab->stb_tagGroup,   &idx);
    collectSymbols(syms, &symTab->stb_labelGroup, &idx);
    qsort(syms, sz, sizeof(CExprOfSymbol*), compareSymbolOrder);

    int isGlobal = ((symbolsFlags & CSYMBOLS_GLOBAL) > 0);
    int isParam = ((symbolsFlags & CSYMBOLS_PARAM) > 0);

    for(idx = 0; idx < sz; ++idx) {
        CExprOfSymbol *sym = syms[idx];
        const char *sclass = NULL;
        CGccAttrKindEnum gak = GAK_UNDEF;

        if((isParam && sym->e_symType != ST_PARAM) ||
            (isParam == 0 && sym->e_symType == ST_PARAM))
            continue;

        switch(sym->e_symType) {
        case ST_TYPE:
            sclass = "typedef_name";
            break;
        case ST_ENUM:
            sclass = "moe";
            break;
        case ST_TAG:
            sclass = "tagname";
            break;
        case ST_LABEL:
            sclass = "label";
            break;
        case ST_PARAM:
            sclass = "param";
            gak = GAK_VAR;
            break;
        case ST_FUNC:
            sclass = getSclassOfFunc(sym, isGlobal);
            gak = GAK_FUNC;
            break;
        case ST_VAR:
            sclass = getSclassOfVar(sym, isGlobal);
            gak = GAK_VAR;
            break;
        case ST_MEMBER:
            sclass = NULL;
            break;
        case ST_GCC_BUILTIN:
            sclass = "extern";
            break;
        case ST_GCC_LABEL:
        case ST_FUNCID:
        case ST_GCC_ASM_IDENT:
        case ST_UNDEF:
        case ST_END:
            ABORT();
        }

        int isGccExtension = 0;
        if(sym->e_symType == ST_TYPE || sym->e_symType == ST_VAR ||
            sym->e_symType == ST_FUNC) {
            CExpr *symP1 = EXPR_PARENT(sym);
            if(symP1) {
                CExpr *symP2 = EXPR_PARENT(symP1);
                if(symP2) {
                    CExpr *symP3 = EXPR_PARENT(symP2);
                    if(symP3) {
                        CExpr *symP4 = EXPR_PARENT(symP3);
                        if(symP4 && (EXPR_CODE(symP4) == EC_DATA_DEF ||
                            EXPR_CODE(symP4) == EC_DECL)) {
                            isGccExtension = EXPR_C(symP4)->e_gccExtension;
                        }
                    }
                }
            }
        }

        const char *type;
        CExprOfTypeDesc *td = EXPRS_TYPE(sym);
        int isGccThread = 0;

        if(td) {
            type = td->e_typeId;
            isGccThread = td->e_sc.esc_isGccThread;
        } else
            type = NULL;

        if(type && sclass)
            outxPrint(fp, indent, "<id type=\"%s\" sclass=\"%s\"", type, sclass);
        else if(type)
            outxPrint(fp, indent, "<id type=\"%s\"", type);
        else
            outxPrint(fp, indent, "<id sclass=\"%s\"", sclass);

        if(isGccExtension)
            fprintf(fp, " is_gccExtension=\"1\"");
        if(isGccThread)
            fprintf(fp, " is_gccThread=\"1\"");

#ifdef XCODEML_DEBUG
        if(type)
            fprintf(fp, " type_p=\"" ADDR_PRINT_FMT "\"", (uintptr_t)td);
#endif

        fprintf(fp, ">\n");

        outxPrint(fp, indent + 1, "<name>%s</name>\n", sym->e_symName);

        if(gak != GAK_UNDEF)
            outxGccAttrVarOrFunc(fp, indent + 1, td, gak);

        outxPrint(fp, indent, "</id>\n");
    }

    free(syms);
}


void
outxSymbols(FILE *fp, int indent, CSymbolTable *symTab, int symbolsFlags)
{
    const char *tag = ((symbolsFlags & CSYMBOLS_GLOBAL) > 0) ?
        "globalSymbols" : "symbols";

    if(symTab) {
        int symSize = CCOL_HT_SIZE(&symTab->stb_identGroup) +
                    CCOL_HT_SIZE(&symTab->stb_tagGroup) +
                    CCOL_HT_SIZE(&symTab->stb_labelGroup);

        if(symSize == 0) {
            outxTagOnly(fp, indent, tag, XATTR_NOCONTENT);
        } else {
            outxTagOnly(fp, indent, tag, 0);
            outxSymbols0(fp, indent + 1, symTab, symbolsFlags);
            outxTagClose(fp, indent, tag);
        }
    } else {
        outxTagOnly(fp, indent, tag, XATTR_NOCONTENT);
    }
}


void
outxDeclarations(FILE *fp, int indent, CExpr *stmts)
{
    outxPrint(fp, indent, "<declarations");

    int hasChild = 0;
    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, stmts) {
        CExpr *expr = EXPR_L_DATA(ite);

        if(EXPR_ISNULL(expr))
            continue;

        switch(EXPR_CODE(expr)) {
        case EC_DECL:
            if(EXPR_B(expr)->e_nodes[1]) {
                if(hasChild == 0) {
                    fprintf(fp, ">\n");
                    hasChild = 1;
                }
                outxContext(fp, indent + 1, EXPR_B(expr)->e_nodes[1]); //INIT_DECLS
            }
            break;
        default:
            break;
        }
    }

    if(hasChild)
        outxTagClose(fp, indent, "declarations");
    else
        fprintf(fp, "/>\n");
}


void
outxGlobalDeclarations(FILE *fp, int indent, CExpr *extDefs)
{
    if(EXPR_L_SIZE(extDefs) == 0)
        return;

    outxPrint(fp, indent, "<globalDeclarations");
    int hasChild = 0, indent1 = indent + 1;
    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, extDefs) {
        CExpr *expr = EXPR_L_DATA(ite);

        if(EXPR_ISNULL(expr))
            continue;

        switch(EXPR_CODE(expr)) {
        case EC_DATA_DEF:
            if(EXPR_B(expr)->e_nodes[1]) {
                if(hasChild == 0) {
                    fprintf(fp, ">\n");
                    hasChild = 1;
                }
                outxContext(fp, indent1, EXPR_B(expr)->e_nodes[1]); //INIT_DECLS
            }
            break;
        case EC_GCC_ASM_EXPR:
            if(hasChild == 0) {
                fprintf(fp, ">\n");
                hasChild = 1;
            }
            outx_GCC_ASM_EXPR_asAsmDef(fp, indent1, expr);
            break;
        default:
            if(hasChild == 0) {
                fprintf(fp, ">\n");
                hasChild = 1;
            }
            outxContext(fp, indent1, expr);
            break;
        }
    }

    if(hasChild)
        outxTagClose(fp, indent, "globalDeclarations");
    else
        fprintf(fp, "/>\n");
}


void
outxSymbolsAndDecls(FILE *fp, int indent, CExpr *extDefsOrStmts, int symbolsFlags)
{
    outxSymbols(fp, indent, EXPR_C(extDefsOrStmts)->e_symTab, symbolsFlags);

    if((symbolsFlags & CSYMBOLS_GLOBAL) > 0)
        outxGlobalDeclarations(fp, indent, extDefsOrStmts);
    else
        outxDeclarations(fp, indent, extDefsOrStmts);
}


void
outxBody(FILE *fp, int indent, CExpr *stmts)
{
    const char *bodyTag = "body";
    if(hasSymbols(stmts)) {
        outxContextWithTag(fp, indent, stmts, bodyTag);
    } else {
        outxChildrenWithTag(fp, indent, stmts, bodyTag);
    }
}


void
outxTagForConstant(FILE *fp, int indent, CExpr *e, const char *tag,
    const char *typeId, const char *val)
{
    outxTag(fp, indent, e, tag, XATTR_NORETURN|XATTR_COMMON,
        " type=\"%s\"", typeId);
    fputs(val, fp);
    outxTagCloseNoIndent(fp, tag);
}


void
outx_CHAR_CONST(FILE *fp, int indent, CExprOfCharConst *cc)
{
    assertExprCode((CExpr*)cc, EC_CHAR_CONST);
    char *token = cc->e_token;
    CCharTypeEnum ct = cc->e_charType;

    switch(ct) {
    case CT_MB:
        sprintf(s_charBuf[0], "0x%X", (unsigned int)token[0]);
        break;
    case CT_WIDE:
        sprintf(s_charBuf[0], "0x%X", (unsigned int)((token[0] << 8) + token[1]));
        break;
    default:
        ABORT();
    }

    outxTagForConstant(fp, indent, (CExpr*)cc, "intConstant",
        s_CBasicTypeEnumXcodes[ct == CT_MB ? BT_CHAR : BT_WCHAR], s_charBuf[0]);
}


void
outx_STRING_CONST(FILE *fp, int indent, CExprOfStringConst *sc)
{
    assertExprCode((CExpr*)sc, EC_STRING_CONST);
    s_charBuf[0][0] = 0;
    if(sc->e_charType == CT_WIDE)
        sprintf(s_charBuf[0], " is_wide=\"1\"");
    const char *stringConstTag = "stringConstant";
    outxTag(fp, indent, (CExpr*)sc, stringConstTag,
        XATTR_NORETURN|XATTR_COMMON, "%s", s_charBuf[0]);
    outxEscapedStr(fp, sc->e_orgToken);
    outxTagCloseNoIndent(fp, stringConstTag);
}


void
outx_NUMBER_CONST(FILE *fp, int indent, CExprOfNumberConst *nc)
{
    assertExprCode((CExpr*)nc, EC_NUMBER_CONST);
    const char *tag = NULL;
    int bt = nc->e_basicType;
    int dec = (nc->e_cardinal == CD_DEC);
    CNumValue *num = &nc->e_numValue;
    s_charBuf[0][0] = 0;

    switch(bt) {
    case BT_SHORT:
    case BT_INT:
        if(dec && num->ll < 0xFFFF)
            sprintf(s_charBuf[0], "%d", (int)num->ll);
        else
            sprintf(s_charBuf[0], "0x%X", (int)num->ll);
        break;
    case BT_UNSIGNED_SHORT:
    case BT_BOOL:
        if(dec && num->ull < 0xFFFF)
            sprintf(s_charBuf[0], "%d", (unsigned int)num->ull);
        else
            sprintf(s_charBuf[0], "0x%X", (unsigned int)num->ull);
        break;
    case BT_LONG:
        if(dec && num->ll < 0xFFFF)
            sprintf(s_charBuf[0], "%ld", (long)num->ll);
        else
            sprintf(s_charBuf[0], "0x%lX", (long)num->ll);
        break;
    case BT_UNSIGNED_INT:
    case BT_UNSIGNED_LONG:
        if(dec && num->ull < 0xFFFF)
            sprintf(s_charBuf[0], "%lld", (long long)num->ull);
        else
            sprintf(s_charBuf[0], "0x%llX", (long long)num->ull);
        break;
    case BT_LONGLONG:
        {
            long h = (long)((num->ll & 0xFFFFFFFF00000000LL) >> 32);
            long l = (long)(num->ll & 0xFFFFFFFFLL);
            sprintf(s_charBuf[0], "0x%lX 0x%lX", h, l);
        }
    case BT_UNSIGNED_LONGLONG:
        {
            long h = (long)((num->ull & 0xFFFFFFFF00000000ULL) >> 32);
            long l = (long)(num->ull & 0xFFFFFFFFULL);
            sprintf(s_charBuf[0], "0x%lX 0x%lX", h, l);
        }
        break;
    case BT_FLOAT:
    case BT_FLOAT_IMAGINARY:
    case BT_DOUBLE:
    case BT_DOUBLE_IMAGINARY:
    case BT_LONGDOUBLE:
        sprintf(s_charBuf[0], "%s", nc->e_orgToken);
        break;
    default:
        addError((CExpr*)nc, CERR_051,
            s_CBasicTypeEnumNames[bt]);
        return;
    }

    switch(bt) {
    case BT_SHORT:
    case BT_INT:
    case BT_LONG:
    case BT_UNSIGNED_SHORT:
    case BT_UNSIGNED_INT:
    case BT_UNSIGNED_LONG:
    case BT_BOOL:
        tag = "intConstant";
        break;
    case BT_LONGLONG:
    case BT_UNSIGNED_LONGLONG:
        tag = "longlongConstant";
        break;
    case BT_FLOAT:
    case BT_FLOAT_IMAGINARY:
    case BT_DOUBLE:
    case BT_DOUBLE_IMAGINARY:
    case BT_LONGDOUBLE:
        tag = "floatConstant";
        break;
    default:
        ABORT();
    }

    const char *type = s_CBasicTypeEnumXcodes[bt];
    outxTagForConstant(fp, indent, (CExpr*)nc, tag, type, s_charBuf[0]);
}


int
isAddrOfChild(CExpr *expr, CExpr *parentExpr, CExpr **outParentExpr)
{
    assertExpr(expr, parentExpr);
    CExprCodeEnum pec = EXPR_CODE(parentExpr);
    //if(pec == EC_EXPRS || pec == EC_ARRAY_REF)
    if (pec == EC_EXPRS)
        return isAddrOfChild(parentExpr, EXPR_PARENT(parentExpr), outParentExpr);
    if(outParentExpr)
        *outParentExpr = parentExpr;
    //return (pec == EC_ADDR_OF) || (pec == EC_GCC_LABEL_ADDR) ||
    //(pec == EC_ARRAY_REF);
    return (pec == EC_ADDR_OF) || (pec == EC_GCC_LABEL_ADDR);
}


void
outx_IDENT(FILE *fp, int indent, CExprOfSymbol *sym)
{
    CExprOfTypeDesc *td = EXPRS_TYPE(sym);
    CExpr *parentExpr = NULL;
    int isAddr = isAddrOfChild((CExpr*)sym, EXPR_PARENT(sym), &parentExpr);
    const char *tag = NULL;
    const char *scope = getScope(sym), *name = sym->e_symName;
    const char *typeId = NULL;
    int isConverted = 0;

    if(isAddr)
        typeId = EXPR_TYPEID(parentExpr);
    else if(td)
        typeId = td->e_typeId;

    switch(sym->e_symType) {
    case ST_PARAM:
    case ST_VAR:
        assertExpr((CExpr*)sym, td);
        isConverted = EXPR_ISCONVERTED(sym);
        switch(td->e_tdKind) {
        case TD_ARRAY:
	  //            tag = (isAddr ? "arrayAddr" : "arrayRef"); break;
            tag = "arrayAddr"; break;
        case TD_COARRAY:
            tag = "name"; scope = NULL; break;
        default:
            tag = (isAddr ? "varAddr" : "Var"); break;
        }
        break;
    case ST_LABEL:
    case ST_GCC_LABEL:
        if(isAddr)
            tag = "gccLabelAddr";
        else {
            tag = "name";
            td = NULL;
        }
        break;
    case ST_ENUM:
        tag = "moeConstant"; break;
    case ST_FUNCID:
        outxTagForExpr(fp, indent, (CExpr*)sym, "builtin_op",
            XATTR_NOCONTENT, " is_id=\"1\" name=\"%s\"", name);
        return;
    case ST_FUNC:
        tag = "funcAddr"; break;
    case ST_TYPE:
        outxPrint(fp, indent, "<typeName type=\"%s\"/>\n", typeId);
        return;
    case ST_GCC_BUILTIN:
        outxPrint(fp, indent, "<builtin_op");
        if(typeId)
            fprintf(fp, " type=\"%s\"", typeId);
        fprintf(fp, " name=\"%s\"", name);
        fprintf(fp, " is_gccSyntax=\"1\" is_id=\"1\"");
        if(isAddr)
            fprintf(fp, " is_addrOf=\"1\"");
        fprintf(fp, "/>\n");
        return;
    case ST_MEMBER:
    case ST_TAG:
    case ST_GCC_ASM_IDENT:
    case ST_END:
        assertExpr((CExpr*)sym, 0);
        ABORT();
    case ST_UNDEF: /* don't care */
	tag = "Var"; // default
	break;
    }

    outxPrint(fp, indent, "<%s", tag);
    if(td)
        fprintf(fp, " type=\"%s\"", typeId);
    if(scope)
        fprintf(fp, " scope=\"%s\"", scope);
    if(isConverted)
        fprintf(fp, " is_modified=\"1\"");
    fprintf(fp, ">%s</%s>\n", name, tag);
}


void
outx_EXT_DEFS(FILE *fp, int indent, CExpr *expr)
{
    fprintf(fp, "<XcodeProgram");

    if(s_xoutputInfo) {
        fprintf(fp, " source=\"");
        outxEscapedStr(fp, s_sourceFileName);

        fprintf(fp, "\" language=\"%s\" time=\"%s\"\n"
                "              compiler-info=\"%s\" version=\"%s\"",
            C_TARGET_LANG, s_timeStamp, C_FRONTEND_NAME, C_FRONTEND_VER);
    }

    fprintf(fp, ">\n");

    int indent1 = indent + 1;

    if(EXPR_ISNULL(expr)) {
        outxTagOnly(fp, indent1, "typeTable", XATTR_NOCONTENT);
        outxTagOnly(fp, indent1, "globalSymbols", XATTR_NOCONTENT);
        outxTagOnly(fp, indent1, "globalDeclarations", XATTR_NOCONTENT);
    } else {
        outxTypeTable(fp, indent1);
        outxSymbolsAndDecls(fp, indent1, expr, CSYMBOLS_GLOBAL);
    }
    fprintf(fp, "</XcodeProgram>\n");
}


void
outx_FUNC_DEF(FILE *fp, int indent, CExpr *funcDef)
{
    CExprOfBinaryNode *declr = EXPR_B(exprListHeadData(funcDef));
    CExprOfTypeDesc *td = EXPR_T(declr->e_nodes[0]);
    CExprOfSymbol *sym = EXPR_SYMBOL(declr->e_nodes[1]);
    CExpr *body = exprListNextNData(funcDef, 2);

    const char *funcDefTag = "functionDefinition";
    outxTagForStmt(fp, indent, funcDef, funcDefTag, 0, NULL);
    int indent1 = indent + 1;
    outxPrint(fp, indent1, "<name>%s</name>\n", sym->e_symName);
    outxGccAttrVarOrFunc(fp, indent1, td, GAK_FUNC_DEF);
    outxSymbols(fp, indent1, EXPR_C(funcDef)->e_symTab, CSYMBOLS_PARAM);
    outxFuncParams(fp, indent1, td->e_paramExpr);
    outxBody(fp, indent1, body);
    outxTagClose(fp, indent, funcDefTag);
}

void
outx_DIRECTIVE(FILE *fp, int indent, CExprOfDirective *directive)
{
    const char *textTag = "text";
    const char *pragmaTag = "pragma";

    if(strcmp(directive->e_direcName, "#pragma") == 0) {
        outxTagForStmt(fp, indent, (CExpr*)directive, pragmaTag,
            XATTR_NORETURN, NULL);
        outxEscapedStr(fp, directive->e_direcArgs);
        outxTagCloseNoIndent(fp, pragmaTag);
    } 
    else if(strcmp(directive->e_direcName, "omp") == 0){
	abort(); /* not yet */
    } else {
        outxTagForStmt(fp, indent, (CExpr*)directive, textTag,
            XATTR_NORETURN, NULL);
        outxEscapedStr(fp, directive->e_direcName);
        fprintf(fp, " ");
        outxEscapedStr(fp, directive->e_direcArgs);
        outxTagCloseNoIndent(fp, textTag);
    }
}

void
out_PRAGMA_COMP_STMT(FILE *fp, int indent, CExpr* expr)
{
    CExprOfList *body = (CExprOfList *)expr;
    CExprOfList *clauseList = (CExprOfList *)body->e_aux_info;
    int code = clauseList->e_aux;
    
    if(IS_OMP_PRAGMA_CODE(code)) 
	out_OMP_PRAGMA(fp,indent,code,expr);
    else if(IS_XMP_PRAGMA_CODE(code))
	out_XMP_PRAGMA(fp,indent,code,expr);
    else if(IS_ACC_PRAGMA_CODE(code))
	out_ACC_PRAGMA(fp,indent,code,expr);
    else
	addFatal(NULL, "unknown PRAGMA CODE");
}


void
outx_INIT_DECL(FILE *fp, int indent, CExpr *initDecl)
{
    CExprOfBinaryNode *declr = EXPR_B(exprListNextNData(initDecl, 0));
    CExprOfUnaryNode *init = NULL;
    if(EXPR_L_SIZE(initDecl) > 2)
        init = EXPR_U(exprListNextNData(initDecl, 2));

    CExprOfTypeDesc *td = EXPR_T(declr->e_nodes[0]);
    assertExprCode((CExpr*)td, EC_TYPE_DESC);
    CExprOfSymbol *sym = EXPR_SYMBOL(declr->e_nodes[1]);

    if(td == NULL || sym == NULL ||
        sym->e_symType == ST_TYPE || td->e_isTypeDef)
        return;

    CExpr *asmExpr = exprListNextNData(initDecl, 1);
    CExprOfTypeDesc *tdo = getRefType(td);
    int isFunc = ETYP_IS_FUNC(tdo);

    if(isFunc) {
        const char *funcTag = "functionDecl";
        outxTagOnly(fp, indent, funcTag, 0);
        outxPrint(fp, indent + 1, "<name>%s</name>\n", sym->e_symName);
        if(EXPR_ISNULL(asmExpr) == 0)
            outx_GCC_ASM_EXPR(fp, indent + 1, asmExpr);
        outxTagClose(fp, indent, funcTag);
    } else {
        const char *varDeclTag = "varDecl";
        outxTagForStmt(fp, indent, initDecl, varDeclTag, 0, NULL);
        outxPrint(fp, indent + 1, "<name>%s</name>\n", sym->e_symName);
        if(EXPR_ISNULL(init) == 0)
            outxContextWithTag(fp, indent + 1, EXPR_U(init)->e_node, "value");
        if(EXPR_ISNULL(asmExpr) == 0)
            outx_GCC_ASM_EXPR(fp, indent + 1, asmExpr);
        outxTagClose(fp, indent, varDeclTag);
    }
}


void
outx_COMPOUND_LITERAL(FILE *fp, int indent, CExpr *expr)
{
    CExprOfTypeDesc *td = EXPRS_TYPE(expr);
    CExpr *parent = NULL;
    int isAddr = isAddrOfChild((CExpr*)expr, EXPR_PARENT(expr), &parent);
    if(isAddr)
        td = EXPRS_TYPE(parent);
    const char *compLtrTag = isAddr ? "compoundValueAddr" : "compoundValue";
    outxTag(fp, indent, expr, compLtrTag, XATTR_COMMON,
        " type=\"%s\"", td->e_typeId);
    outxContextWithTag(fp, indent + 1, EXPR_B(expr)->e_nodes[1], "value");
    outxTagClose(fp, indent, compLtrTag);
}


void
outx_ASSIGN(FILE *fp, int indent, CExpr *expr)
{
  //    const char *assignTag = isCoArrayAssign(expr) ? "coArrayAssignExpr" : "assignExpr";
    const char *assignTag = "assignExpr";
    outxChildrenForExpr(fp, indent, expr, assignTag);
}


void
outx_EXPRS(FILE *fp, int indent, CExpr *expr)
{
    CCOL_DListNode *ite;
    int sz = 0;
    EXPR_FOREACH(ite, expr) {
        CExpr *node = EXPR_L_DATA(ite);
        if(EXPR_ISNULL(node) == 0)
            ++sz;
    }

    switch(sz) {
    case 0:
        return;
    case 1:
        if(EXPR_ISNULL(exprListHeadData(expr)))
            return;
        outxChildren(fp, indent, expr);
        break;
    default:
        outxChildrenForExpr(fp, indent, expr, "commaExpr");
        break;
    }
}


void
outx_COMP_STMT(FILE *fp, int indent, CExpr *expr)
{
    const char *compStmtTag = "compoundStatement";
    outxTagForStmt(fp, indent, expr, compStmtTag, 0, NULL);
    int indent1 = indent + 1;
    outxSymbolsAndDecls(fp, indent1, expr, 0);
    outxChildrenWithTag(fp, indent1, expr, "body");
    outxTagClose(fp, indent, compStmtTag);
}


void
outx_IF_STMT(FILE *fp, int indent, CExpr *stmt)
{
    CExpr *cond     = exprListNextNData(stmt, 0);
    CExpr *thenBody = exprListNextNData(stmt, 1);
    CExpr *elseBody = NULL;
    if(EXPR_L_SIZE(stmt) > 2)
        elseBody = exprListNextNData(stmt, 2);

    int indent1 = indent + 1;
    const char *ifStmtTag = "ifStatement";
    outxTagForStmt(fp, indent, stmt, ifStmtTag, 0, NULL);
    outxContextWithTag(fp, indent1, cond, "condition");
    outxContextWithTag(fp, indent1, thenBody, "then");
    if(elseBody)
        outxContextWithTag(fp, indent1, elseBody, "else");
    outxTagClose(fp, indent, ifStmtTag);
}


void
outx_WHILE_STMT(FILE *fp, int indent, CExprOfBinaryNode *stmt)
{
    CExpr *cond = stmt->e_nodes[0];
    CExpr *body = stmt->e_nodes[1];
    int indent1 = indent + 1;
    const char *whileStmtTag = "whileStatement";
    outxTagForStmt(fp, indent, (CExpr*)stmt, whileStmtTag, 0, NULL);
    outxContextWithTag(fp, indent1, cond, "condition");
    outxBody(fp, indent1, body);
    outxTagClose(fp, indent, whileStmtTag);
}


void
outx_DO_STMT(FILE *fp, int indent, CExprOfBinaryNode *stmt)
{
    CExpr *body = stmt->e_nodes[0];
    CExpr *cond = stmt->e_nodes[1];
    int indent1 = indent + 1;
    const char *doStmtTag = "doStatement";
    outxTagForStmt(fp, indent, (CExpr*)stmt, doStmtTag, 0, NULL);
    outxBody(fp, indent1, body);
    outxContextWithTag(fp, indent1, cond, "condition");
    outxTagClose(fp, indent, doStmtTag);
}


void
outx_FOR_STMT(FILE *fp, int indent, CExpr *stmt)
{
    CExpr *init = exprListNextNData(stmt, 0);
    CExpr *cond = exprListNextNData(stmt, 1);
    CExpr *iter = exprListNextNData(stmt, 2);
    CExpr *body = exprListNextNData(stmt, 3);
    int indent1 = indent + 1;
    const char *forStmtTag = "forStatement";
    outxTagForStmt(fp, indent, stmt, forStmtTag, 0, NULL);
    outxContextWithTag(fp, indent1, init, "init");
    outxContextWithTag(fp, indent1, cond, "condition");
    outxContextWithTag(fp, indent1, iter, "iter");
    outxBody(fp, indent1, body);
    outxTagClose(fp, indent, forStmtTag);
}


void
outx_SWITCH_STMT(FILE *fp, int indent, CExprOfBinaryNode *stmt)
{
    CExpr *val = stmt->e_nodes[0];
    CExpr *body = stmt->e_nodes[1];
    const char *switchStmtTag = "switchStatement";

    outxTagForStmt(fp, indent, (CExpr*)stmt, switchStmtTag, 0, NULL);
    outxContextWithTag(fp, indent + 1, val, "value");
    outxBody(fp, indent + 1, body);
    outxTagClose(fp, indent, switchStmtTag);
}


void
outx_CASE_LABEL(FILE *fp, int indent, CExprOfBinaryNode *stmt)
{
    CExpr *case1 = stmt->e_nodes[0];
    CExpr *case2 = stmt->e_nodes[1];

    if(EXPR_ISNULL(case2)) {
        const char *caseLabelTag = "caseLabel";
        outxTagForStmt(fp, indent, (CExpr*)stmt, caseLabelTag, 0, NULL);
        outxContextWithTag(fp, indent + 1, case1, "value");
        outxTagClose(fp, indent, caseLabelTag);
    } else {
        const char *gccRangedCaseLabelTag = "gccRangedCaseLabel";
        outxTagForStmt(fp, indent, (CExpr*)stmt, gccRangedCaseLabelTag, 0, NULL);
        outxContextWithTag(fp, indent + 1, case1, "value");
        outxContextWithTag(fp, indent + 1, case2, "value");
        outxTagClose(fp, indent, gccRangedCaseLabelTag);
    }
}


void
outxBuiltinOpCall(FILE *fp, int indent, const char *name, CExpr *typeExpr, CExpr *args)
{
    const char *builtinOpTag = "builtin_op";
    outxTagForExpr(fp, indent, typeExpr, builtinOpTag, 0,
        " name=\"%s\"", name);
    outxChildren(fp, indent + 1, args);
    outxTagClose(fp, indent, builtinOpTag);
}


void
outx_FUNCTION_CALL(FILE *fp, int indent, CExprOfBinaryNode *funcCall)
{
    CExpr *funcExpr = funcCall->e_nodes[0];
    CExpr *args = funcCall->e_nodes[1];

    if(EXPR_CODE(funcExpr) == EC_IDENT &&
        EXPR_SYMBOL(funcExpr)->e_symType == ST_GCC_BUILTIN) {

        outxBuiltinOpCall(fp, indent, EXPR_SYMBOL(funcExpr)->e_symName,
            (CExpr*)funcCall, args);
    } else {
        const char *funcCallTag = "functionCall";
        outxTagForExpr(fp, indent, (CExpr*)funcCall, funcCallTag, 0, NULL);
        outxContextWithTag(fp, indent + 1, funcExpr, "function");
        outxChildrenWithTag(fp, indent + 1, args, "arguments");
        outxTagClose(fp, indent, funcCallTag);
    }
}


void
outx_TYPE_DESC(FILE *fp, int indent, CExprOfTypeDesc *td)
{
    outxPrint(fp, indent, "<typeName type=\"%s\"/>\n", td->e_typeId);
}


void
outx_INITIALIZERS(FILE *fp, int indent, CExpr *expr)
{
    int isDesignated = (EXPR_ISNULL(EXPR_L(expr)->e_symbol) == 0);
    const char *valueTag = isDesignated ? "designatedValue" : "value";
    if(isDesignated) {
        //member designator
        sprintf(s_charBuf[0], " member=\"%s\"", EXPR_L(expr)->e_symbol->e_symName);
    } else {
        s_charBuf[0][0] = 0;
    }
    outxTag(fp, indent, expr, valueTag, 0, s_charBuf[0]);
    outxChildren(fp, indent + 1, expr);
    outxTagClose(fp, indent, valueTag);
}


void
outx_CAST(FILE *fp, int indent, CExprOfBinaryNode *castExpr)
{
    CExpr *expr = castExpr->e_nodes[1];
    const char *castExprTag = "castExpr";
    outxTagForExpr(fp, indent, (CExpr*)castExpr, castExprTag, 0, NULL);
    outxContext(fp, indent + 1, expr);
    outxTagClose(fp, indent, castExprTag);
}


void
outx_POINTS_AT(FILE *fp, int indent, CExprOfBinaryNode *pointsAt)
{
    CExpr *composExpr = pointsAt->e_nodes[0];
    CExprOfSymbol *memSym = EXPR_SYMBOL(pointsAt->e_nodes[1]);
    CExprOfTypeDesc *memTd = EXPRS_TYPE(memSym);
    CExprOfTypeDesc *memTdo = getRefType(memTd);
    CExpr *parentExpr = NULL;
    int isAddr = isAddrOfChild((CExpr*)pointsAt, EXPR_PARENT(pointsAt), &parentExpr);
    const char *tag;

    if(isAddr) {
        if(ETYP_IS_ARRAY(memTdo))
            tag = "memberArrayAddr";
        else
            tag = "memberAddr";
        memTd = EXPRS_TYPE(parentExpr);
    } else {
        if(ETYP_IS_ARRAY(memTdo))
            tag = "memberArrayRef";
        else
            tag = "memberRef";
    }

    outxTag(fp, indent, (CExpr*)memSym, tag, XATTR_COMMON,
        " type=\"%s\" member=\"%s\"", memTd->e_typeId, memSym->e_symName);

    if(EXPR_CODE(composExpr) == EC_POINTER_REF)
        outxContext(fp, indent + 1, EXPR_U(composExpr)->e_node);
    else
        outxContext(fp, indent + 1, composExpr);

    outxTagClose(fp, indent, tag);
}


/* void */
/* outx_ARRAY_REF(FILE *fp, int indent, CExprOfBinaryNode *aryRef) */
/* { */
/*     CExpr *aryExpr = aryRef->e_nodes[0]; */
/*     CExprOfTypeDesc *aryTd = EXPRS_TYPE(aryExpr); */
/*     CExprOfTypeDesc *aryTdo = getRefType(aryTd); */
/*     CExpr *dim = aryRef->e_nodes[1]; */
/*     CExpr *dimLwr = exprListHeadData(dim); */

/*     if(EXPR_L_SIZE(dim) > 1) { */
/*         //sub array */
/*         CExpr *dimUpr = exprListNextNData(dim, 1); */
/*         CExpr *dimStp = exprListNextNData(dim, 2); */
/*         CExprOfTypeDesc *dimTd = EXPR_T(exprListNextNData(dim, 3)); */

/*         outxPrint(fp, indent, "<subArrayRef type=\"%s\">\n", dimTd->e_typeId); */
/*         outxContext(fp, indent + 1, aryExpr); */
/*         outxContextWithTag(fp, indent + 1, dimLwr, "lowerBound"); */

/*         CExpr *tmpUpr = NULL; */
/*         if(EXPR_ISNULL(dimUpr)) { */
/*             dimUpr = aryTdo->e_len.eln_lenExpr; */
/*             if(dimUpr) { */
/*                 // complete subarray ref upper bound */
/*                 if(dimUpr && EXPR_CODE(dimUpr) == EC_NUMBER_CONST) { */
/*                     tmpUpr = (CExpr*)allocExprOfNumberConst2( */
/*                         getNumberConstAsLong(dimUpr) - 1, BT_INT); */
/*                 } else { */
/*                     tmpUpr = exprBinary(EC_MINUS, dimUpr, */
/*                         (CExpr*)allocExprOfNumberConst2(1, BT_INT)); */
/*                 } */
/*             } */
/*             dimUpr = tmpUpr; */
/*         } */
/*         if(dimUpr) */
/*             outxContextWithTag(fp, indent + 1, dimUpr, "upperBound"); */
/*         outxContextWithTag(fp, indent + 1, dimStp, "step"); */
/*         outxPrint(fp, indent, "</subArrayRef>\n"); */

/*         if(tmpUpr) */
/*             freeExpr(tmpUpr); */
/*     } else { */
/*         //normal array */
/*         CExpr *parent = EXPR_PARENT(aryRef); */
/*         CExprCodeEnum pec = EXPR_CODE(parent); */
/*         int pref = (pec != EC_ADDR_OF); */
/*         int indent1 = pref ? indent: indent - 1; */
/*         const char *pointerRefTag = "pointerRef"; */
/*         const char *plusExprTag = "plusExpr"; */

/*         if(pref) */
/*             outxTag1(fp, indent, (CExpr*)aryRef, pointerRefTag, XATTR_TYPE); */

/*         CExprOfTypeDesc *plusExprTd = ETYP_IS_ARRAY(aryTdo) ? */
/*             EXPR_T(aryTdo->e_paramExpr) : aryTd; */

/*         outxTag(fp, indent1 + 1, (CExpr*)aryRef, plusExprTag, XATTR_COMMON, */
/*             " type=\"%s\"", plusExprTd->e_typeId); */
/*         outxContext(fp, indent1 + 2, aryExpr); */
/*         outxContext(fp, indent1 + 2, dimLwr); */
/*         outxTagClose(fp, indent1 + 1, plusExprTag); */

/*         if(pref) */
/*             outxTagClose(fp, indent, pointerRefTag); */
/*     } */
/* } */

extern unsigned int s_arrayToPointer;

void
outx_ARRAY_REF(FILE *fp, int indent, CExprOfBinaryNode *aryRef)
{
  CExpr *aryExpr = aryRef->e_nodes[0];
  CExprOfTypeDesc *aryTd = EXPRS_TYPE(aryExpr);
  CExprOfTypeDesc *aryTdo = getRefType(aryTd);

  CExpr *dim = aryRef->e_nodes[1];
  CExpr *dimLwr = exprListHeadData(dim);

  if (isSubArrayRef2((CExpr*)aryRef)){
    //sub array
/*     CExprOfTypeDesc *dimTd = EXPR_T(exprListNextNData(dim, 3)); */
/*     outxPrint(fp, indent, "<subArrayRef type=\"%s\">\n", dimTd->e_typeId); */
    outxPrint(fp, indent, "<subArrayRef type=\"%s\">\n", EXPR_TYPEID(aryRef));
    outx_SUBARRAY_REF(fp, indent + 1, aryRef);
    outxPrint(fp, indent, "</subArrayRef>\n");
  }
  else {

    //normal array

    CExpr *parent = EXPR_PARENT(aryRef);
    CExprCodeEnum pec = EXPR_CODE(parent);

    CExprOfBinaryNode *tmp_aryRef = EXPR_B(aryRef->e_nodes[0]);
    while (EXPR_CODE(tmp_aryRef) == EC_ARRAY_REF){
      tmp_aryRef = EXPR_B(tmp_aryRef->e_nodes[0]);
    }

    if (s_arrayToPointer || EXPR_CODE(tmp_aryRef) != EC_IDENT ||
	EXPRS_TYPE(EXPR_SYMBOL(tmp_aryRef))->e_tdKind != TD_ARRAY){

      int pref = (pec != EC_ADDR_OF);
      int indent1 = pref ? indent: indent - 1;
      const char *pointerRefTag = "pointerRef";
      const char *plusExprTag = "plusExpr";
    
      if (pref)
	outxTag1(fp, indent, (CExpr*)aryRef, pointerRefTag, XATTR_TYPE);

      CExprOfTypeDesc *plusExprTd = ETYP_IS_ARRAY(aryTdo) ?
	EXPR_T(aryTdo->e_paramExpr) : aryTd;

      outxTag(fp, indent1 + 1, (CExpr*)aryRef, plusExprTag, XATTR_COMMON,
	      " type=\"%s\"", plusExprTd->e_typeId);
      outxContext(fp, indent1 + 2, aryExpr);
      outxContext(fp, indent1 + 2, dimLwr);
      outxTagClose(fp, indent1 + 1, plusExprTag);
    
      if (pref)
	outxTagClose(fp, indent, pointerRefTag);
    }
    else {
      int pref = (pec == EC_ADDR_OF);

      if (pref){
	CExprOfTypeDesc *addrTd = ETYP_IS_ARRAY(aryTdo) ?
	  EXPR_T(aryTdo->e_paramExpr) : aryTd;
	outxPrint(fp, indent++, "<addrOfExpr type=\"%s\">\n", addrTd->e_typeId);
      }

      outxPrint(fp, indent, "<arrayRef type=\"%s\">\n", EXPR_TYPEID(aryRef));
      outx_ARRAY_REF2(fp, indent + 1, aryRef);
      outxPrint(fp, indent, "</arrayRef>\n");

      if (pref)
	outxPrint(fp, --indent, "</addrOfExpr>\n");
    }
  }
}


void
outx_SUBARRAY_REF(FILE *fp, int indent, CExprOfBinaryNode *aryRef)
{
  if (EXPR_CODE((CExpr*)aryRef) != EC_ARRAY_REF){
    outxContext(fp, indent, (CExpr*)aryRef);
    return;
  }

  CExpr *aryExpr = aryRef->e_nodes[0];

  outx_SUBARRAY_REF(fp, indent, EXPR_B(aryExpr));

  CExpr *dim = aryRef->e_nodes[1];
  CExpr *dimLwr = exprListHeadData(dim);

  if (EXPR_L_SIZE(dim) <= 1){
    outxContext(fp, indent, dimLwr);
  }
  else {

    outxPrint(fp, indent, "<indexRange>\n");

    CExprOfTypeDesc *aryTd = EXPRS_TYPE(aryExpr);
    CExprOfTypeDesc *aryTdo = getRefType(aryTd);

    CExpr *dimUpr = exprListNextNData(dim, 1);
    CExpr *dimStp = exprListNextNData(dim, 2);

    outxContextWithTag(fp, indent + 1, dimLwr, "lowerBound");

    CExpr *tmpUpr = NULL;

    if (EXPR_ISNULL(dimUpr)) {
      dimUpr = aryTdo->e_len.eln_lenExpr;
      if (dimUpr) {
	// complete subarray ref upper bound
	if (dimUpr && EXPR_CODE(dimUpr) == EC_NUMBER_CONST) {
	  tmpUpr = (CExpr*)allocExprOfNumberConst2(
			     getNumberConstAsLong(dimUpr) - 1, BT_INT);
	}
	else {
	  tmpUpr = exprBinary(EC_MINUS, dimUpr,
			      (CExpr*)allocExprOfNumberConst2(1, BT_INT));
	}
      }
      dimUpr = tmpUpr;
    }
    
    if (dimUpr)
      outxContextWithTag(fp, indent + 1, dimUpr, "upperBound");

    outxContextWithTag(fp, indent + 1, dimStp, "step");
    
    outxPrint(fp, indent, "</indexRange>\n");

    if (tmpUpr)
      freeExpr(tmpUpr);
  }
}


void
outx_ARRAY_REF2(FILE *fp, int indent, CExprOfBinaryNode *aryRef)
{
  if (EXPR_CODE((CExpr*)aryRef) != EC_ARRAY_REF){
    outxContext(fp, indent, (CExpr*)aryRef);
    return;
  }

  CExpr *aryExpr = aryRef->e_nodes[0];
  outx_ARRAY_REF2(fp, indent, EXPR_B(aryExpr));

  CExpr *dim = aryRef->e_nodes[1];
  CExpr *dimLwr = exprListHeadData(dim);
  outxContext(fp, indent, dimLwr);
}


void
outx_COARRAY_REF(FILE *fp, int indent, CExprOfBinaryNode *aryRef)
{
    CExpr *sym = EXPR_B(aryRef)->e_nodes[0];
    CExpr *coDims = EXPR_B(aryRef)->e_nodes[1];
    //const char *scope = getScope(EXPR_SYMBOL(sym));

    CExprOfTypeDesc *td = EXPRS_TYPE(aryRef);

    //outxPrint(fp, indent, "<coArrayRef type=\"%s\" scope=\"%s\">\n", td->e_typeId, scope);
    outxPrint(fp, indent, "<coArrayRef type=\"%s\">\n", td->e_typeId);
    outxContext(fp, indent + 1, sym);
    outxChildren(fp, indent + 1, coDims);
    outxPrint(fp, indent, "</coArrayRef>\n");
}


void
outxOfsMemberDesignator(FILE *fp, int indent, CExprOfBinaryNode *expr)
{
    const char *gccMemDesigTag = "gccMemberDesignator";
    CExpr *parent = expr->e_nodes[0];
    CExpr *identOrAry = expr->e_nodes[1];
    int noParent = EXPR_ISNULL(parent);
    s_charBuf[0][0] = 0;

    if(EXPR_CODE(identOrAry) == EC_IDENT)
        sprintf(s_charBuf[0], " member=\"%s\"", EXPR_SYMBOL(identOrAry)->e_symName);

    outxTag(fp, indent, (CExpr*)expr, gccMemDesigTag,
        XATTR_COMMON | (noParent ? XATTR_NOCONTENT : 0),
        " ref=\"%s\"%s", EXPR_TYPEID(expr), s_charBuf[0]);

    if(noParent)
        return;

    outxOfsMemberDesignator(fp, indent + 1, EXPR_B(parent));
    if(EXPR_CODE(identOrAry) != EC_IDENT) 
        outxContext(fp, indent + 1, identOrAry);
    outxTagClose(fp, indent, gccMemDesigTag);
}


void
outx_GCC_BLTIN_OFFSETOF(FILE *fp, int indent, CExprOfBinaryNode *expr)
{
    const char *builtinOpTag = "builtin_op";
    outxTagForExpr(fp, indent, (CExpr*)expr, builtinOpTag, 0,
        " name=\"__builtin_offsetof\"");
    int indent1 = indent + 1;
    outxContext(fp, indent1, expr->e_nodes[0]);
    outxOfsMemberDesignator(fp, indent1, EXPR_B(expr->e_nodes[1]));
    outxTagClose(fp, indent, builtinOpTag);
}


void
outx_GCC_REALPART(FILE *fp, int indent, CExpr *expr)
{
    const char *builtinOpTag = "builtin_op";
    outxTagForExpr(fp, indent, expr, builtinOpTag, XATTR_NOCONTENT,
        " name=\"__real__\"");
}


void
outx_GCC_IMAGPART(FILE *fp, int indent, CExpr *expr)
{
    const char *builtinOpTag = "builtin_op";
    outxTagForExpr(fp, indent, expr, builtinOpTag, XATTR_NOCONTENT,
        " name=\"__imag__\"");
}


void
outx_GCC_ASM_EXPR(FILE *fp, int indent, CExpr *asmExpr)
{
    outxChildrenWithTag(fp, indent, asmExpr, "gccAsm");
}


void
outx_GCC_ASM_EXPR_asAsmDef(FILE *fp, int indent, CExpr *asmExpr)
{
    const char *gccAsmDefTag = "gccAsmDefinition";
    outxTagForStmt(fp, indent, asmExpr, gccAsmDefTag, 0, NULL);
    outxChildren(fp, indent + 1, asmExpr);
    outxTagClose(fp, indent, gccAsmDefTag);
}


void
outx_GCC_ASM_STMT(FILE *fp, int indent, CExprOfBinaryNode *stmt)
{
    CExpr *volatileExpr = stmt->e_nodes[0];
    CExpr *arg = stmt->e_nodes[1];
    CExpr *asmStr = exprListNextNData(arg, 0); 
    CExpr *opes1 = NULL, *opes2 = NULL, *clobs = NULL;
    int argSz = EXPR_L_SIZE(arg);
    if(argSz > 1)
        opes1 = exprListNextNData(arg, 1); 
    if(argSz > 2)
        opes2 = exprListNextNData(arg, 2); 
    if(argSz > 3)
        clobs = exprListNextNData(arg, 3); 

    const char *gccAsmStmtTag = "gccAsmStatement";
    const char *gccAsmOperandsTag = "gccAsmOperands";
    const char *gccAsmClobbersTag = "gccAsmClobbers";
    s_charBuf[0][0] = 0;

    if(EXPR_ISNULL(volatileExpr) == 0)
        sprintf(s_charBuf[0], " is_volatile=\"1\"");

    outxTagForStmt(fp, indent, (CExpr*)stmt, gccAsmStmtTag, 0, "%s", s_charBuf[0]);

    int indent1 = indent + 1;
    outxContext(fp, indent1, asmStr);
    if(opes1) {
        outxChildrenWithTag(fp, indent1, opes1, gccAsmOperandsTag);
        if(opes2) {
            outxChildrenWithTag(fp, indent1, opes2, gccAsmOperandsTag);
            if(clobs) {
                outxChildrenWithTag(fp, indent1, clobs, gccAsmClobbersTag);
            }
        }
    }

    outxTagClose(fp, indent, gccAsmStmtTag);
}


void
outx_GCC_ASM_OPE(FILE *fp, int indent, CExpr *expr)
{
    CExpr *match = exprListNextNData(expr, 0);
    CExpr *constraint = exprListNextNData(expr, 1);
    CExpr *arg = exprListNextNData(expr, 2);
    const char *gccAsmOperandTag = "gccAsmOperand";

    outxPrint(fp, indent, "<");
    fputs(gccAsmOperandTag, fp);

    if(EXPR_ISNULL(match) == 0) {
        fprintf(fp, " match=\"");
        outxEscapedStr(fp, EXPR_SYMBOL(match)->e_symName);
        fprintf(fp, "\"");
    }
    if(EXPR_ISNULL(constraint) == 0) {
        fprintf(fp, " constraint=\"");
        outxEscapedStr(fp, EXPR_STRINGCONST(constraint)->e_orgToken);
        fprintf(fp, "\"");
    }

    if(EXPR_ISNULL(arg)) {
        fprintf(fp, "/>\n");
    } else {
        fprintf(fp, ">\n");
        outxContext(fp, indent + 1, arg);
        outxTagClose(fp, indent, gccAsmOperandTag);
    }
}


/**
 * \brief
 * output XcodeML
 *
 * @param fp
 *      output file pointer
 * @param expr
 *      output nodes
 */
void
outputXcodeML(FILE *fp, CExpr *expr)
{
    if(s_verbose)
        printf("generating XcodeML ...\n");

    setTypeIds();
    setExprParent(expr, NULL);
    //fprintf(fp, "<?xml version=\"1.0\" encoding=\"%s\"?>\n", s_xmlEncoding);
    fprintf(fp, "<?xml version=\"1.0/1.2\" encoding=\"%s\"?>\n", s_xmlEncoding);
    if(EXPR_ISNULL(expr))
        outx_EXT_DEFS(fp, 0, NULL);
    else
        outxContext(fp, 0, expr);

#ifdef CEXPR_DEBUG
    checkGccAttrOutput();
#endif
    CCOL_DL_CLEAR(&s_typeDescList);
}

