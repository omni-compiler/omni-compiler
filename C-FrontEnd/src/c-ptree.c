/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-ptree.c
 * - display parse tree (PT) for debugging and enhancement
 * These functions were made from freeExpr() and innderFreeExprOfXxx() functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>
#include <limits.h>
#include <float.h>

#include "c-expr.h"
#include "c-option.h"
#include "c-pragma.h"
#include "c-expr.h"

#define DISP_STRING(str)    if (expr->str) fprintf(fp, " %s=\"%s\"", #str, expr->str);
#define DISP_EXPR_HEAD(str)   _printExprHead(fp, #str, (CExprCommon*)expr, indent)

const char *NAME_e_struct[] = CExprStructEnumNamesDef;
const char *NAME_e_symType[] = CSymbolTypeEnumNamesDef;
const char *NAME_e_tdKind[] = CTypeDescKindEnumNamesDef;
const char *NAME_e_basicType[] = CBasicTypeEnumCcodeDef;

static void _dispExpr(FILE *fp, CExpr *expr, int indent);
static void _dispExprBlock(FILE *fp, CExpr *expr, int indent, char *terminal);

static void _dispInnerExprOfSymbol(FILE *fp, CExprOfSymbol *expr, int indent);
static void _dispInnerExprOfNumberConst(FILE *fp, CExprOfNumberConst *expr, int indent);
static void _dispInnerExprOfCharConst(FILE *fp, CExprOfCharConst *expr, int indent);
static void _dispInnerExprOfStringConst(FILE *fp, CExprOfStringConst *expr, int indent);
static void _dispInnerExprOfList(FILE *fp, CExprOfList *expr, int indent);
static void _dispInnerExprOfGeneralCode(FILE *fp, CExprOfGeneralCode *expr, int indent);
static void _dispInnerExprOfUnaryNode(FILE *fp, CExprOfUnaryNode *expr, int indent);
static void _dispInnerExprOfBinaryNode(FILE *fp, CExprOfBinaryNode *expr, int indent);
static void _dispInnerExprOfArrayDecl(FILE *fp, CExprOfArrayDecl *expr, int indent);
static void _dispInnerExprOfDirective(FILE *fp, CExprOfDirective *expr, int indent);
static void _dispInnerExprOfTypeDesc(FILE *fp, CExprOfTypeDesc *expr, int indent);
static void _dispInnerExprOfErrorNode(FILE *fp, CExprOfErrorNode *expr, int indent);
static void _dispInnerExprOfNull(FILE *fp, CExprOfNull *expr, int indent);

static char *_getCExprCodeEnumString(CExprCodeEnum code);
static void _printExprHead(FILE *fp, char *str, CExprCommon *expr, int indent);
static void _printNewlineAndIndent(FILE *fp, int indent);



/**
 * \brief
 * display struct CExpr as a parse tree (PT)
 *
 * @param expr
 *      root node of parse tree
 * @param fp
 *      output file pointer
 */
void
dispParseTree(FILE *fp, CExpr *expr, char *title)
{
    char *msg;

    msg = title ? title : "PARSE TREE";

    fprintf(fp, "[%s] start", msg);
    _dispExpr(fp, expr, 0);
    fprintf(fp, "\n");
    fprintf(fp, "[%s] end.\n", msg);
}

/**
   for debugging
**/
int _ptree(CExpr *expr)
{
  if (expr) {
    dispParseTree(stdout, expr, "parse tree");
    return 1;
  }
  return 0;
}



static void 
_dispExpr(FILE *fp, CExpr *expr, int indent)
{
    if (expr == NULL)
        return;

    if (EXPR_C(expr)->e_gccAttrPre) {
        _dispExprBlock(fp, EXPR_C(expr)->e_gccAttrPre, indent, "<pre>");
    }

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfSymbol:
        _dispInnerExprOfSymbol(fp, EXPR_SYMBOL(expr), indent);
        break;
    case STRUCT_CExprOfList:
        _dispInnerExprOfList(fp, EXPR_L(expr), indent);
        break;
    case STRUCT_CExprOfNumberConst:
        _dispInnerExprOfNumberConst(fp, EXPR_NUMBERCONST(expr), indent);
        break;
    case STRUCT_CExprOfCharConst:
        _dispInnerExprOfCharConst(fp, EXPR_CHARCONST(expr), indent);
        break;
    case STRUCT_CExprOfStringConst:
        _dispInnerExprOfStringConst(fp, EXPR_STRINGCONST(expr), indent);
        break;
    case STRUCT_CExprOfGeneralCode:
        _dispInnerExprOfGeneralCode(fp, EXPR_GENERALCODE(expr), indent);
        break;
    case STRUCT_CExprOfUnaryNode:
        _dispInnerExprOfUnaryNode(fp, EXPR_U(expr), indent);
        break;
    case STRUCT_CExprOfBinaryNode:
        _dispInnerExprOfBinaryNode(fp, EXPR_B(expr), indent);
        break;
    case STRUCT_CExprOfArrayDecl:
        _dispInnerExprOfArrayDecl(fp, EXPR_ARRAYDECL(expr), indent);
        break;
    case STRUCT_CExprOfDirective:
        _dispInnerExprOfDirective(fp, EXPR_DIRECTIVE(expr), indent);
        break;
    case STRUCT_CExprOfTypeDesc:
        _dispInnerExprOfTypeDesc(fp, EXPR_T(expr), indent);
        break;
    case STRUCT_CExprOfErrorNode:
        _dispInnerExprOfErrorNode(fp, EXPR_ERRORNODE(expr), indent);
        break;
    case STRUCT_CExprOfNull:
        _dispInnerExprOfNull(fp, EXPR_NULL(expr), indent);
        break;
    case STRUCT_CExpr_UNDEF:
    case STRUCT_CExpr_END:
        ABORT();
    }

    if (EXPR_C(expr)->e_gccAttrPost) {
        _dispExprBlock(fp, EXPR_C(expr)->e_gccAttrPost, indent, "<post>");
    }
}


static void 
_dispExprBlock(FILE *fp, CExpr *expr, int indent, char *terminal)
{
    _dispExpr(fp, expr, indent);

    if (expr && terminal) {
        _printNewlineAndIndent(fp, indent);
        fprintf(fp, "+end of %s", terminal);
    }
}


static void
_printExprHead(FILE *fp, char *str, CExprCommon *expr, int indent)
{
    _printNewlineAndIndent(fp, indent);
    fprintf(fp, "%s", str);
    fprintf(fp, "(line=%d)", expr->e_lineNumInfo.ln_rawLineNum);
    fprintf(fp, " e_refCount=%d", expr->e_refCount);
    fprintf(fp, " e_struct=%s", NAME_e_struct[expr->e_struct]);
    fprintf(fp, " e_exprCode=%s", _getCExprCodeEnumString(EXPR_CODE(expr)));

    if (expr->e_symTab) fprintf(fp, " e_symTab");
    if (expr->e_exprsType) fprintf(fp, " e_exprsType");
    if (expr->e_parentExpr) fprintf(fp, " e_parentExpr");
    if (expr->e_gccAttrPre) fprintf(fp, " e_gccAttrPre");
    if (expr->e_gccAttrPost) fprintf(fp, " e_gccAttrPost");

    if (expr->e_isError) fprintf(fp, " e_isError");
    if (expr->e_isConstValue) fprintf(fp, " e_isConstValue");
    if (expr->e_isConstValueChcked) fprintf(fp, " e_isConstValueChcked");
    if (expr->e_gccExtension) fprintf(fp, " e_gccExtension");
    if (expr->e_isCompleted ) fprintf(fp, " e_isCompleted");
    if (expr->e_hasInit     ) fprintf(fp, " e_hasInit");
    if (expr->e_isCompiled  ) fprintf(fp, " e_isCompiled");
    if (expr->e_isConverted ) fprintf(fp, " e_isConverted");
    if (expr->e_isGenerated ) fprintf(fp, " e_isGenerated");
    if (expr->e_isGccSyntax ) fprintf(fp, " e_isGccSyntax");
    if (expr->e_isDeleting  ) fprintf(fp, " e_isDeleting");
}

static void
_printNewlineAndIndent(FILE *fp, int indent)
{
    fputc('\n', fp);
    for (int i = 0; i < indent; i++)
        fprintf(fp, "| ");
}


static void
_dispInnerExprOfSymbol(FILE *fp, CExprOfSymbol *expr, int indent)
{
    DISP_EXPR_HEAD(Symbol);
    DISP_STRING(e_symName);
    if (expr->e_symType   ) fprintf(fp, " e_symType=%s", NAME_e_symType[expr->e_symType]);
    if (expr->e_headType  ) fprintf(fp, " e_headType");
    if (expr->e_declrExpr ) fprintf(fp, " e_declrExpr");
    if (expr->e_putOrder  ) fprintf(fp, " e_putOrder=%d", expr->e_putOrder);
    if (expr->e_isGlobal  ) fprintf(fp, " e_isGlobal");
    if (expr->e_isEnumInited) fprintf(fp, " e_isEnumInited");
    if (expr->e_isGccLabelDecl) fprintf(fp, " e_isGccLabelDecl");
    if (expr->e_isConstButUnreducable) fprintf(fp, " e_isConstButUnreducable");
    _dispExprBlock(fp, expr->e_valueExpr, indent + 1, "e_valueExpr");
    _dispExprBlock(fp, expr->e_codimensions, indent + 1, "e_codimensions");
}


static void
_dispInnerExprOfNumberConst(FILE *fp, CExprOfNumberConst *expr, int indent)
{
    DISP_EXPR_HEAD(NumberConst);
    DISP_STRING(e_orgToken);
}


static void
_dispInnerExprOfCharConst(FILE *fp, CExprOfCharConst *expr, int indent)
{
    DISP_EXPR_HEAD(CharConst);
    DISP_STRING(e_orgToken);
    DISP_STRING(e_token);
}


static void
_dispInnerExprOfStringConst(FILE *fp, CExprOfStringConst *expr, int indent)
{
    DISP_EXPR_HEAD(StringConst);
    DISP_STRING(e_orgToken);
}


static void
_dispInnerExprOfList(FILE *fp, CExprOfList *expr, int indent)
{
    DISP_EXPR_HEAD(List);

    CCOL_DListNode *ite, *iten;
    EXPR_FOREACH_SAFE(ite, iten, expr) {
        _dispExpr(fp, CCOL_DL_DATA(ite), indent + 1);
    }

    _dispExprBlock(fp, (CExpr*)expr->e_symbol, indent + 1, "e_symbol");
    _dispExprBlock(fp, expr->e_aux_info, indent + 1, "e_aux_info");
}


static void
_dispInnerExprOfGeneralCode(FILE *fp, CExprOfGeneralCode *expr, int indent)
{
    DISP_EXPR_HEAD(GeneralCode);
}


static void
_dispInnerExprOfUnaryNode(FILE *fp, CExprOfUnaryNode *expr, int indent)
{
    DISP_EXPR_HEAD(UnaryNode);
    _dispExprBlock(fp, expr->e_node, indent + 1, "operand e_node");
}


static void
_dispInnerExprOfBinaryNode(FILE *fp, CExprOfBinaryNode *expr, int indent)
{
    DISP_EXPR_HEAD(BinaryNode);
    _dispExprBlock(fp, expr->e_nodes[0], indent + 1, "first operand e_nodes[0]");
    _dispExprBlock(fp, expr->e_nodes[1], indent + 1, "second operand e_nodes[1]");
}


static void
_dispInnerExprOfArrayDecl(FILE *fp, CExprOfArrayDecl *expr, int indent)
{
    DISP_EXPR_HEAD(ArrayDecl);
    if (expr->e_isVariable) fprintf(fp, " e_isVariable");
    if (expr->e_isStatic) fprintf(fp, " e_isStatic");
    _dispExprBlock(fp, expr->e_typeQualExpr, indent + 1, "e_typeQualExpr");
    _dispExprBlock(fp, expr->e_lenExpr, indent + 1, "e_lenExpr");
}


static void
_dispInnerExprOfDirective(FILE *fp, CExprOfDirective *expr, int indent)
{
    DISP_EXPR_HEAD(Directive);
    DISP_STRING(e_direcArgs);
    DISP_STRING(e_direcName);
}


static void
_dispInnerExprOfTypeDesc(FILE *fp, CExprOfTypeDesc *expr, int indent)
{
    DISP_EXPR_HEAD(TypeDesc);
    if (expr->e_tdKind ) fprintf(fp, " e_tdKind=%s", NAME_e_tdKind[expr->e_tdKind]);
    if (expr->e_basicType) fprintf(fp, " e_basicType=%s", NAME_e_basicType[expr->e_basicType]);

    if (expr->e_bitLen ) fprintf(fp, " e_bitLen=%d", expr->e_bitLen);
    if (expr->e_refType ) fprintf(fp, " e_refType");
    if (expr->e_preDeclType ) fprintf(fp, " e_preDeclType");
    if (expr->e_typeId ) fprintf(fp, " e_typeId");
    if (expr->e_align ) fprintf(fp, " e_align=%d", expr->e_align);
    if (expr->e_size ) fprintf(fp, " e_size=%d", expr->e_size);
    if (expr->e_symbol) fprintf(fp, " e_symbol");
    if (expr->e_len.eln_len ) fprintf(fp, " e_len.eln_len=%u", expr->e_len.eln_len);

    if (expr->e_len.eln_isVariable ) fprintf(fp, " e_len.eln_isVariable");
    if (expr->e_len.eln_isFlexible ) fprintf(fp, " e_len.eln_isFlexible");
    if (expr->e_len.eln_isStatic   ) fprintf(fp, " e_len.eln_isStatic");
    if (expr->e_len.eln_isConst    ) fprintf(fp, " e_len.eln_isConst");
    if (expr->e_len.eln_isVolatile ) fprintf(fp, " e_len.eln_isVolatile");
    if (expr->e_len.eln_isRestrict ) fprintf(fp, " e_len.eln_isRestrict");

    if (expr->e_tq.etq_isConst) fprintf(fp, " e_tq.etq_isConst");
    if (expr->e_tq.etq_isInline) fprintf(fp, " e_tq.etq_isInline");
    if (expr->e_tq.etq_isVolatile) fprintf(fp, " e_tq.etq_isVolatile");
    if (expr->e_tq.etq_isRestrict) fprintf(fp, " e_tq.etq_isRestrict");

    if (expr->e_sc.esc_isAuto) fprintf(fp, " e_sc.esc_isAuto");
    if (expr->e_sc.esc_isStatic) fprintf(fp, " e_sc.esc_isStatic");
    if (expr->e_sc.esc_isExtern) fprintf(fp, " e_sc.esc_isExtern");
    if (expr->e_sc.esc_isRegister) fprintf(fp, " e_sc.esc_isRegister");
    if (expr->e_sc.esc_isGccThread) fprintf(fp, " e_sc.esc_isGccThread");

    if (expr->e_isTypeDef    ) fprintf(fp, " e_isTypeDef");
    if (expr->e_isExist      ) fprintf(fp, " e_isExist");
    if (expr->e_isNoMemDecl  ) fprintf(fp, " e_isNoMemDecl");
    if (expr->e_isAnonTag    ) fprintf(fp, " e_isAnonTag");
    if (expr->e_isAnonMember ) fprintf(fp, " e_isAnonMember");
    if (expr->e_isDefined    ) fprintf(fp, " e_isDefined");
    if (expr->e_isUsed       ) fprintf(fp, " e_isUsed");
    if (expr->e_isExprsType  ) fprintf(fp, " e_isExprsType");
    if (expr->e_isCollected  ) fprintf(fp, " e_isCollected");
    if (expr->e_isTemporary  ) fprintf(fp, " e_isTemporary");
    if (expr->e_isFixed      ) fprintf(fp, " e_isFixed");
    if (expr->e_isCompiling  ) fprintf(fp, " e_isCompiling");
    if (expr->e_isGccConst   ) fprintf(fp, " e_isGccConst");
    if (expr->e_isGccAttrDuplicated) fprintf(fp, " e_isGccAttrDuplicated");
    if (expr->e_isSizeUnreducable) fprintf(fp, " e_isSizeUnreducable");
    if (expr->e_isSizeZero   ) fprintf(fp, " e_isSizeZero");
    if (expr->e_isNoTypeId   ) fprintf(fp, " e_isNoTypeId");
    if (expr->e_isMarked     ) fprintf(fp, " e_isMarked");
    if (expr->e_isDuplicated ) fprintf(fp, " e_isDuplicated");
    if (expr->e_isDifferentQaulifierFromRefType) fprintf(fp, " e_isDifferentQaulifierFromRefType");

    _dispExprBlock(fp, expr->e_typeExpr, indent + 1, "e_typeExpr");
    _dispExprBlock(fp, expr->e_paramExpr, indent + 1, "e_paramExpr");
    _dispExprBlock(fp, expr->e_bitLenExpr, indent + 1, "e_bitLenExpr");
    _dispExprBlock(fp, expr->e_len.eln_lenExpr, indent + 1, "e_len.eln_lenExpr");
    _dispExprBlock(fp, expr->e_len.eln_orgLenExpr, indent + 1, "e_len.eln_orgLenExpr");
    DISP_STRING(e_typeId);

    //refType is only for reference
}


static void
_dispInnerExprOfErrorNode(FILE *fp, CExprOfErrorNode *expr, int indent)
{
    DISP_EXPR_HEAD(ErrorNode);
    _dispExprBlock(fp, expr->e_nearExpr, indent + 1, "e_nearExpr");
}


static void
_dispInnerExprOfNull(FILE *fp, CExprOfNull *expr, int indent)
{
    DISP_EXPR_HEAD(Null);
}



static char *
_getCExprCodeEnumString(CExprCodeEnum code)
{
    switch(code) {
    case EC_UNDEF                        : return "EC_UNDEF";
    case EC_CHAR_CONST                   : return "EC_CHAR_CONST";
    case EC_STRING_CONST                 : return "EC_STRING_CONST";
    case EC_NUMBER_CONST                 : return "EC_NUMBER_CONST";
    case EC_DEFAULT_LABEL                : return "EC_DEFAULT_LABEL";
    case EC_ELLIPSIS                     : return "EC_ELLIPSIS";
    case EC_IDENT                        : return "EC_IDENT";
    case EC_NULL_NODE                    : return "EC_NULL_NODE";
    case EC_EXT_DEFS                     : return "EC_EXT_DEFS";
    case EC_FUNC_DEF                     : return "EC_FUNC_DEF";
    case EC_DECL                         : return "EC_DECL";
    case EC_DATA_DEF                     : return "EC_DATA_DEF";
    case EC_INIT                         : return "EC_INIT";
    case EC_INITIALIZER                  : return "EC_INITIALIZER";
    case EC_INITIALIZERS                 : return "EC_INITIALIZERS";
    case EC_INIT_DECL                    : return "EC_INIT_DECL";
    case EC_INIT_DECLS                   : return "EC_INIT_DECLS";
    case EC_DESIGNATORS                  : return "EC_DESIGNATORS";
    case EC_ARRAY_DESIGNATOR             : return "EC_ARRAY_DESIGNATOR";
    case EC_DECLARATOR                   : return "EC_DECLARATOR";
    case EC_STRUCT_TYPE                  : return "EC_STRUCT_TYPE";
    case EC_UNION_TYPE                   : return "EC_UNION_TYPE";
    case EC_ENUM_TYPE                    : return "EC_ENUM_TYPE";
    case EC_MEMBERS                      : return "EC_MEMBERS";
    case EC_MEMBER_DECL                  : return "EC_MEMBER_DECL";
    case EC_MEMBER_DECLS                 : return "EC_MEMBER_DECLS";
    case EC_MEMBER_DECLARATOR            : return "EC_MEMBER_DECLARATOR";
    case EC_ENUMERATORS                  : return "EC_ENUMERATORS";
    case EC_GCC_LABEL_DECLS              : return "EC_GCC_LABEL_DECLS";
    case EC_COMPOUND_LITERAL             : return "EC_COMPOUND_LITERAL";
    case EC_PARAMS                       : return "EC_PARAMS";
    case EC_IDENTS                       : return "EC_IDENTS";
    case EC_GCC_LABEL_IDENTS             : return "EC_GCC_LABEL_IDENTS";
    case EC_STMTS_AND_DECLS              : return "EC_STMTS_AND_DECLS";
    case EC_COMP_STMT                    : return "EC_COMP_STMT";
    case EC_EXPRS                        : return "EC_EXPRS";
    case EC_GCC_COMP_STMT_EXPR           : return "EC_GCC_COMP_STMT_EXPR";
    case EC_EXPR_STMT                    : return "EC_EXPR_STMT";
    case EC_IF_STMT                      : return "EC_IF_STMT";
    case EC_WHILE_STMT                   : return "EC_WHILE_STMT";
    case EC_DO_STMT                      : return "EC_DO_STMT";
    case EC_FOR_STMT                     : return "EC_FOR_STMT";
    case EC_SWITCH_STMT                  : return "EC_SWITCH_STMT";
    case EC_BREAK_STMT                   : return "EC_BREAK_STMT";
    case EC_CONTINUE_STMT                : return "EC_CONTINUE_STMT";
    case EC_RETURN_STMT                  : return "EC_RETURN_STMT";
    case EC_GOTO_STMT                    : return "EC_GOTO_STMT";
    case EC_CASE_LABEL                   : return "EC_CASE_LABEL";
    case EC_LABEL                        : return "EC_LABEL";
    case EC_LABELS                       : return "EC_LABELS";
    case EC_FUNCTION_CALL                : return "EC_FUNCTION_CALL";
    case EC_TYPE_DESC                    : return "EC_TYPE_DESC";
    case EC_BRACED_EXPR                  : return "EC_BRACED_EXPR";
    case EC_UNARY_MINUS                  : return "EC_UNARY_MINUS";
    case EC_BIT_NOT                      : return "EC_BIT_NOT";
    case EC_POINTER_REF                  : return "EC_POINTER_REF";
    case EC_ADDR_OF                      : return "EC_ADDR_OF";
    case EC_PRE_INCR                     : return "EC_PRE_INCR";
    case EC_PRE_DECR                     : return "EC_PRE_DECR";
    case EC_LOG_NOT                      : return "EC_LOG_NOT";
    case EC_SIZE_OF                      : return "EC_SIZE_OF";
    case EC_CAST                         : return "EC_CAST";
    case EC_POST_INCR                    : return "EC_POST_INCR";
    case EC_POST_DECR                    : return "EC_POST_DECR";
    case EC_LSHIFT                       : return "EC_LSHIFT";
    case EC_RSHIFT                       : return "EC_RSHIFT";
    case EC_PLUS                         : return "EC_PLUS";
    case EC_MINUS                        : return "EC_MINUS";
    case EC_MUL                          : return "EC_MUL";
    case EC_DIV                          : return "EC_DIV";
    case EC_MOD                          : return "EC_MOD";
    case EC_ARITH_EQ                     : return "EC_ARITH_EQ";
    case EC_ARITH_NE                     : return "EC_ARITH_NE";
    case EC_ARITH_GE                     : return "EC_ARITH_GE";
    case EC_ARITH_GT                     : return "EC_ARITH_GT";
    case EC_ARITH_LE                     : return "EC_ARITH_LE";
    case EC_ARITH_LT                     : return "EC_ARITH_LT";
    case EC_LOG_AND                      : return "EC_LOG_AND";
    case EC_LOG_OR                       : return "EC_LOG_OR";
    case EC_BIT_AND                      : return "EC_BIT_AND";
    case EC_BIT_OR                       : return "EC_BIT_OR";
    case EC_BIT_XOR                      : return "EC_BIT_XOR";
    case EC_ASSIGN                       : return "EC_ASSIGN";
    case EC_ASSIGN_PLUS                  : return "EC_ASSIGN_PLUS";
    case EC_ASSIGN_MINUS                 : return "EC_ASSIGN_MINUS";
    case EC_ASSIGN_MUL                   : return "EC_ASSIGN_MUL";
    case EC_ASSIGN_DIV                   : return "EC_ASSIGN_DIV";
    case EC_ASSIGN_MOD                   : return "EC_ASSIGN_MOD";
    case EC_ASSIGN_LSHIFT                : return "EC_ASSIGN_LSHIFT";
    case EC_ASSIGN_RSHIFT                : return "EC_ASSIGN_RSHIFT";
    case EC_ASSIGN_BIT_AND               : return "EC_ASSIGN_BIT_AND";
    case EC_ASSIGN_BIT_OR                : return "EC_ASSIGN_BIT_OR";
    case EC_ASSIGN_BIT_XOR               : return "EC_ASSIGN_BIT_XOR";
    case EC_CONDEXPR                     : return "EC_CONDEXPR";
    case EC_POINTS_AT                    : return "EC_POINTS_AT";
    case EC_ARRAY_REF                    : return "EC_ARRAY_REF";
    case EC_ARRAY_DIMENSION              : return "EC_ARRAY_DIMENSION";
    case EC_MEMBER_REF                   : return "EC_MEMBER_REF";
    case EC_FLEXIBLE_STAR                : return "EC_FLEXIBLE_STAR";
    case EC_XMP_COARRAY_DECLARATION      : return "EC_XMP_COARRAY_DECLARATION";
    case EC_XMP_COARRAY_DECLARATIONS     : return "EC_XMP_COARRAY_DECLARATIONS";
    case EC_XMP_COARRAY_DIM_DEFS         : return "EC_XMP_COARRAY_DIM_DEFS";
    case EC_XMP_COARRAY_DIMENSIONS       : return "EC_XMP_COARRAY_DIMENSIONS";
    case EC_XMP_COARRAY_REF              : return "EC_XMP_COARRAY_REF";
    case EC_XMP_CRITICAL                 : return "EC_XMP_CRITICAL";
    case EC_GCC_EXTENSION                : return "EC_GCC_EXTENSION";
    case EC_GCC_LABEL_ADDR               : return "EC_GCC_LABEL_ADDR";
    case EC_GCC_ALIGN_OF                 : return "EC_GCC_ALIGN_OF";
    case EC_GCC_TYPEOF                   : return "EC_GCC_TYPEOF";
    case EC_GCC_REALPART                 : return "EC_GCC_REALPART";
    case EC_GCC_IMAGPART                 : return "EC_GCC_IMAGPART";
    case EC_GCC_BLTIN_VA_ARG             : return "EC_GCC_BLTIN_VA_ARG";
    case EC_GCC_BLTIN_OFFSET_OF          : return "EC_GCC_BLTIN_OFFSET_OF";
    case EC_GCC_BLTIN_TYPES_COMPATIBLE_P : return "EC_GCC_BLTIN_TYPES_COMPATIBLE_P";
    case EC_GCC_OFS_MEMBER_REF           : return "EC_GCC_OFS_MEMBER_REF";
    case EC_GCC_OFS_ARRAY_REF            : return "EC_GCC_OFS_ARRAY_REF";
    case EC_GCC_ATTRS                    : return "EC_GCC_ATTRS";
    case EC_GCC_ATTR_ARG                 : return "EC_GCC_ATTR_ARG";
    case EC_GCC_ATTR_ARGS                : return "EC_GCC_ATTR_ARGS";
    case EC_GCC_ASM_STMT                 : return "EC_GCC_ASM_STMT";
    case EC_GCC_ASM_EXPR                 : return "EC_GCC_ASM_EXPR";
    case EC_GCC_ASM_ARG                  : return "EC_GCC_ASM_ARG";
    case EC_GCC_ASM_OPE                  : return "EC_GCC_ASM_OPE";
    case EC_GCC_ASM_OPES                 : return "EC_GCC_ASM_OPES";
    case EC_GCC_ASM_CLOBS                : return "EC_GCC_ASM_CLOBS";
    case EC_DIRECTIVE                    : return "EC_DIRECTIVE";
    case EC_STRINGS                      : return "EC_STRINGS";
    case EC_DECL_SPECS                   : return "EC_DECL_SPECS";
    case EC_LDECLARATOR                  : return "EC_LDECLARATOR";
    case EC_DATA_DECL                    : return "EC_DATA_DECL";
    case EC_DATA_DECLS                   : return "EC_DATA_DECLS";
    case EC_ARRAY_DECL                   : return "EC_ARRAY_DECL";
    case EC_COARRAY_DECL                 : return "EC_COARRAY_DECL";
    case EC_POINTER_DECL                 : return "EC_POINTER_DECL";
    case EC_PARAM                        : return "EC_PARAM";
    case EC_TYPENAME                     : return "EC_TYPENAME";
    case EC_SCSPEC                       : return "EC_SCSPEC";
    case EC_TYPESPEC                     : return "EC_TYPESPEC";
    case EC_ERROR_NODE                   : return "EC_ERROR_NODE";
    case EC_TYPEQUAL                     : return "EC_TYPEQUAL";
    case EC_PRAGMA_PACK                  : return "EC_PRAGMA_PACK";
    case EC_XMP_DESC_OF                  : return "EC_XMP_DESC_OF";
    case EC_END                          : return "EC_END";
    default: abort();
    }
}



