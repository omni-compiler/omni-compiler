/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-dump.c
 */

#include "c-expr.h"
#include "c-const.h"

#define STR_BUFSIZE 256

const char *s_CSCSpecEnumNames[]            = CSCSpecEnumNamesDef;
const char *s_CTypeSpecEnumNames[]          = CTypeSpecEnumNamesDef;
const char *s_CTypeQualEnumNames[]          = CTypeQualEnumNamesDef;
const char *s_CAssignEnumNames[]            = CAssignEnumNamesDef;
const char *s_CCharTypeEnumNames[]          = CCharTypeEnumNamesDef;
const char *s_CExprStructEnumNames[]        = CExprStructEnumNamesDef;
const char *s_CSymbolTypeEnumNames[]        = CSymbolTypeEnumNamesDef;
const char *s_CBasicTypeEnumNames[]         = CBasicTypeEnumNamesDef;
const char *s_CCardinalEnumNames[]          = CCardinalEnumNamesDef;
const char *s_CDirectiveTypeEnumNames[]     = CDirectiveTypeEnumNamesDef;
const char *s_CTypeDescKindEnumNames[]      = CTypeDescKindEnumNamesDef;
const char *s_CGccAttrKindEnumNames[]       = CGccAttrKindEnumNamesDef;

PRIVATE_STATIC void dumpExpr1(FILE *fp, CExpr *expr, int indent, int child);


PRIVATE_STATIC void
printIndent(FILE *fp, int n)
{
    for(int i = 0; i < n; ++i) {
        fprintf(fp, " ");
    }
}


PRIVATE_STATIC void
dumpExprOfSymbol(FILE *fp, CExprOfSymbol *expr, int indent, int child)
{
    assert(expr->e_symName != NULL);
    assert(expr->e_symType >= ST_UNDEF && expr->e_symType < ST_END);

    printIndent(fp, indent);
    fprintf(fp, "name='%s', symType=%s",
        expr->e_symName, s_CSymbolTypeEnumNames[expr->e_symType]);

    if(expr->e_isEnumInited)
        fprintf(fp, ", isEnumInited");
    if(expr->e_isConstButUnreducable)
        fprintf(fp, ", isConstButUnreducable");
    if(expr->e_isGlobal)
        fprintf(fp, ", isGlobal");

    if(expr->e_declrExpr)
        fprintf(fp, ", declrExpr=" ADDR_PRINT_FMT, (uintptr_t)expr->e_declrExpr);

    fprintf(fp, "\n");

    if(child == 0)
        return;

    if(expr->e_valueExpr) {
        printIndent(fp, indent);
        fprintf(fp, "valueExpr:\n");
        dumpExpr1(fp, expr->e_valueExpr, indent + 1, 1);
    }
}


PRIVATE_STATIC void
dumpExprOfList(FILE *fp, CExprOfList *expr, int indent, int child)
{
    if(child == 0)
        return;

    int i = 0, nindent = indent + 1;

    printIndent(fp, indent);
    fprintf(fp, "nodes=%d", EXPR_L_SIZE(expr));

    if(expr->e_symbol)
        fprintf(fp, ", symbol='%s' (" ADDR_PRINT_FMT ")",
            expr->e_symbol->e_symName, (uintptr_t)expr->e_symbol);

    fprintf(fp, "\n");

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, expr) {
        printIndent(fp, indent);
        fprintf(fp, "next[%d]:\n", i);
        ++i;
        dumpExpr1(fp, EXPR_L_DATA(ite), nindent, 1);
    }
}

PRIVATE_STATIC void
dumpExprOfNumberConst(FILE *fp, CExprOfNumberConst *expr, int indent, int child)
{
    printIndent(fp, indent);
    fprintf(fp, "basicType=%s, cardinal=%s, token='%s', numValue=",
        s_CBasicTypeEnumNames[expr->e_basicType],
        s_CCardinalEnumNames[expr->e_cardinal],
        expr->e_orgToken);

    CNumValue *v = &expr->e_numValue;
    switch(getNumValueKind(expr->e_basicType)) {
    case NK_LL:
        fprintf(fp, "0x%08llx", v->ll); break;
    case NK_ULL:
        fprintf(fp, "0x%08llx", v->ull); break;
    case NK_LD:
        fprintf(fp, "%-3.5Le", v->ld); break;
    }
    fprintf(fp, "\n");
}


PRIVATE_STATIC void
dumpExprOfCharConst(FILE *fp, CExprOfCharConst *expr, int indent, int child)
{
    printIndent(fp, indent);
    fprintf(fp, "charType=%s, token='%s'\n",
        s_CCharTypeEnumNames[expr->e_charType], expr->e_orgToken);
}


PRIVATE_STATIC void
dumpExprOfStringConst(FILE *fp, CExprOfStringConst *expr, int indent, int child)
{
    printIndent(fp, indent);
    fprintf(fp, "charType=%s, token='%s', numChars=%d\n",
        s_CCharTypeEnumNames[expr->e_charType], expr->e_orgToken, expr->e_numChars);
}


PRIVATE_STATIC void
dumpExprOfGeneralCode(FILE *fp, CExprOfGeneralCode *expr, int indent, int child)
{
    printIndent(fp, indent);
    const char *codeStr;
    int exprCode = EXPR_CODE(expr);

    switch(exprCode) {

    case EC_TYPEQUAL:
        codeStr = s_CTypeQualEnumNames[expr->e_code];
        break;
    case EC_TYPESPEC:
        codeStr = s_CTypeSpecEnumNames[expr->e_code];
        break;
    case EC_SCSPEC:
        codeStr = s_CSCSpecEnumNames[expr->e_code];
        break;
    case EC_ELLIPSIS:
        codeStr = "...";
        break;
    case EC_DEFAULT_LABEL:
    case EC_XMP_CRITICAL:
        codeStr = "";
        break;
    case EC_POINTER_DECL:
    case EC_FLEXIBLE_STAR:
        codeStr = "*";
        break;
    default:
        if(exprCode >= EC_ERROR_NODE && exprCode < EC_END) {
            DBGPRINT(("Invalid exprCode : %s", s_CExprCodeInfos[exprCode].ec_name));
        } else {
            DBGPRINT(("Invalid exprCode : %d", exprCode));
        }
        abort();
        break;
    }

    fprintf(fp, "code=%s\n", codeStr);
}


PRIVATE_STATIC void
dumpExprOfUnaryNode(FILE *fp, CExprOfUnaryNode *expr, int indent, int child)
{
    if(child == 0)
        return;

    printIndent(fp, indent);
    fprintf(fp, "node:\n");
    dumpExpr1(fp, expr->e_node, indent + 1, 1);
}


PRIVATE_STATIC void
dumpExprOfBinaryNode(FILE *fp, CExprOfBinaryNode *expr, int indent, int child)
{
    if(EXPR_CODE(expr) == EC_GCC_ATTR_ARG) {
        printIndent(fp, indent);
        fprintf(fp, "gccAttrKind=%s",
            s_CGccAttrKindEnumNames[expr->e_gccAttrKind]);
        if(expr->e_gccAttrIgnored)
            fprintf(fp, ", gccAttrIgnored");
        fprintf(fp, "\n");
    }

    if(child == 0)
        return;

    for(int i = 0; i < 2; ++i) {
        printIndent(fp, indent);
        fprintf(fp, "nodes[%d]:\n", i);
        dumpExpr1(fp, expr->e_nodes[i], indent + 1, 1);
    }
}


PRIVATE_STATIC void
dumpExprOfArrayDecl(FILE *fp, CExprOfArrayDecl *expr, int indent, int child)
{
    char buf[STR_BUFSIZE];
    buf[0] = 0;
    if(expr->e_isVariable)
        strcat(buf, "isVariable=1");
    if(expr->e_isStatic) {
        if(buf[0])
            strcat(buf, ",");
        strcat(buf, "isStatic=1");
    }

    if(buf[0]) {
        printIndent(fp, indent);
        fprintf(fp, "%s\n", buf);
    }

    if(child == 0)
        return;

    if(expr->e_typeQualExpr) {
        printIndent(fp, indent);
        fprintf(fp, "typeQualExpr:\n");
        dumpExpr1(fp, expr->e_typeQualExpr, indent + 1, 1);
    }

    if(expr->e_lenExpr) {
        printIndent(fp, indent);
        fprintf(fp, "lenExpr:\n");
        dumpExpr1(fp, expr->e_lenExpr, indent + 1, 1);
    }
}


PRIVATE_STATIC void
dumpExprOfDirective(FILE *fp, CExprOfDirective *expr, int indent, int child)
{
    printIndent(fp, indent);
    fprintf(fp, "direcType=%s, direcName='%s', direcArgs='%s'\n",
        s_CDirectiveTypeEnumNames[expr->e_direcType],
        expr->e_direcName, expr->e_direcArgs);
}


PRIVATE_STATIC void
dumpExprOfTypeDesc(FILE *fp, CExprOfTypeDesc *expr, int indent, int child)
{
    printIndent(fp, indent);

    fprintf(fp, "tdKind=%s", s_CTypeDescKindEnumNames[expr->e_tdKind]);

    if(expr->e_tdKind == TD_BASICTYPE)
        fprintf(fp, ", basicType=%s", s_CBasicTypeEnumNames[expr->e_basicType]);

    if(expr->e_bitLen > 0)          fprintf(fp, ", bitLen=%d", expr->e_bitLen);
    if(expr->e_isTypeDef)           fprintf(fp, ", isTypeDef");
    if(expr->e_isExist)             fprintf(fp, ", isExist");
    if(expr->e_isNoMemDecl)         fprintf(fp, ", isNoMemDecl");
    if(expr->e_isUsed)              fprintf(fp, ", isUsed");
    if(expr->e_isFixed)             fprintf(fp, ", isFixed");
    if(expr->e_isMarked)            fprintf(fp, ", isMarked");
    if(expr->e_isDuplicated)        fprintf(fp, ", isDuplicated");
    if(expr->e_isAnonTag)           fprintf(fp, ", isAnonTag");
    if(expr->e_isAnonMember)        fprintf(fp, ", isAnonMember");
    if(expr->e_isCollected)         fprintf(fp, ", isCollected");
    if(expr->e_isGccConst)          fprintf(fp, ", isGccConst");
    if(expr->e_isGccAttrDuplicated) fprintf(fp, ", isGccAttrDuplicated");
    if(expr->e_tq.etq_isConst)      fprintf(fp, ", tq.isConst");
    if(expr->e_tq.etq_isInline)     fprintf(fp, ", tq.isInline");
    if(expr->e_tq.etq_isVolatile)   fprintf(fp, ", tq.isVolatile");
    if(expr->e_tq.etq_isRestrict)   fprintf(fp, ", tq.isRestrict");
    if(expr->e_sc.esc_isAuto)       fprintf(fp, ", sc.isAuto");
    if(expr->e_sc.esc_isStatic)     fprintf(fp, ", sc.isStatic");
    if(expr->e_sc.esc_isExtern)     fprintf(fp, ", sc.isExtern");
    if(expr->e_sc.esc_isRegister)   fprintf(fp, ", sc.isRegister");
    if(expr->e_sc.esc_isGccThread)  fprintf(fp, ", sc.isGccThread");
    if(expr->e_len.eln_isVariable)  fprintf(fp, ", len.isVariable");
    if(expr->e_len.eln_isStatic)    fprintf(fp, ", len.isStatic");
    if(expr->e_len.eln_isConst)     fprintf(fp, ", len.isConst");
    if(expr->e_len.eln_isVolatile)  fprintf(fp, ", len.isVolatile");
    if(expr->e_len.eln_isRestrict)  fprintf(fp, ", len.isRestrict");

    if(expr->e_size > 0)
        fprintf(fp, ", size=%d", expr->e_size);
    if(expr->e_align > 0)
        fprintf(fp, ", align=%d", expr->e_align);

    fprintf(fp, "\n");

    if(child == 0)
        return;

    if(expr->e_typeExpr) {
        printIndent(fp, indent);
        fprintf(fp, "typeExpr:\n");
        dumpExpr1(fp, expr->e_typeExpr, indent + 1, 1);
    }

    if(expr->e_refType) {
        printIndent(fp, indent);
        fprintf(fp, "refType:\n");
        dumpExpr1(fp, (CExpr*)expr->e_refType, indent + 1, 0);
    }

    if(expr->e_paramExpr) {
        printIndent(fp, indent);
        fprintf(fp, "paramExpr:\n");
        dumpExpr1(fp, expr->e_paramExpr, indent + 1, 1);
    }

    if(expr->e_len.eln_lenExpr) {
        printIndent(fp, indent);
        fprintf(fp, "lenExpr:\n");
        dumpExpr1(fp, expr->e_len.eln_lenExpr, indent + 1, 1);
    }

    if(expr->e_bitLenExpr) {
        printIndent(fp, indent);
        fprintf(fp, "bitLenExpr:\n");
        dumpExpr1(fp, expr->e_bitLenExpr, indent + 1, 1);
    }
}


PRIVATE_STATIC void
dumpExprOfErrorNode(FILE *fp, CExprOfErrorNode *expr, int indent, int child)
{
    if(child == 0)
        return;

    if(expr->e_nearExpr) {
        printIndent(fp, indent);
        fprintf(fp, "nearExpr:\n");
        dumpExpr1(fp, expr->e_nearExpr, indent + 1, 1);
    }
}


PRIVATE_STATIC void
dumpExprOfNull(FILE *fp, CExprOfNull *expr, int indent, int child)
{
}


PRIVATE_STATIC void
dumpExpr1(FILE *fp, CExpr *expr, int indent, int child)
{
    printIndent(fp, indent);

    if(expr == NULL) {
        fprintf(fp, "(NULL)\n");
        return;
    }

    CExprCommon *cmn = (CExprCommon*)expr;
    const CExprCodeInfo *ec = &s_CExprCodeInfos[cmn->e_exprCode];
    CLineNumInfo *li = &cmn->e_lineNumInfo;

    fprintf(fp, "%s <%s@" ADDR_PRINT_FMT "> at FID=%d,%d-%d/%d ref=%d", ec->ec_name,
        s_CExprStructEnumNames[cmn->e_struct], (uintptr_t)expr,
        li->ln_fileId, li->ln_column, li->ln_lineNum, li->ln_rawLineNum, cmn->e_refCount);

    if(ec->ec_opeName != NULL)
        fprintf(fp, " ('%s')", ec->ec_opeName);

    if(cmn->e_symTab)
        fprintf(fp, ", symTab=" ADDR_PRINT_FMT, (uintptr_t)cmn->e_symTab);
    if(cmn->e_exprsType)
        fprintf(fp, ", exprsType=" ADDR_PRINT_FMT, (uintptr_t)cmn->e_exprsType);
    if(cmn->e_parentExpr)
        fprintf(fp, ", parentExpr=" ADDR_PRINT_FMT, (uintptr_t)cmn->e_parentExpr);
    if(cmn->e_isError)
        fprintf(fp, ", isError");
    if(cmn->e_isCompleted)
        fprintf(fp, ", isCompleted");
    if(cmn->e_isCompiled)
        fprintf(fp, ", isCompiled");
    if(cmn->e_hasInit)
        fprintf(fp, ", hasInit");

    fprintf(fp, "\n");

    int nindent = indent + 1;

    if(child) {

        if(cmn->e_gccExtension) {
            printIndent(fp, nindent);
            fprintf(fp, "gccExtension=1\n");
        }

        if(EXPR_ISNULL(cmn->e_gccAttrPre) == 0) {
            printIndent(fp, nindent);
            fprintf(fp, "gccAttrPre:\n");
            dumpExpr1(fp, cmn->e_gccAttrPre, nindent + 1, 1);
        }

        if(EXPR_ISNULL(cmn->e_gccAttrPost) == 0) {
            printIndent(fp, nindent);
            fprintf(fp, "gccAttrPost:\n");
            dumpExpr1(fp, cmn->e_gccAttrPost, nindent + 1, 1);
        }

        fflush(fp);
    }

    switch(cmn->e_struct) {

    case STRUCT_CExprOfSymbol:
        dumpExprOfSymbol(fp, EXPR_SYMBOL(expr), nindent, child);
        break;
    case STRUCT_CExprOfList:
        dumpExprOfList(fp, EXPR_L(expr), nindent, child);
        break;
    case STRUCT_CExprOfNumberConst:
        dumpExprOfNumberConst(fp, EXPR_NUMBERCONST(expr), nindent, child);
        break;
    case STRUCT_CExprOfCharConst:
        dumpExprOfCharConst(fp, EXPR_CHARCONST(expr), nindent, child);
        break;
    case STRUCT_CExprOfStringConst:
        dumpExprOfStringConst(fp, EXPR_STRINGCONST(expr), nindent, child);
        break;
    case STRUCT_CExprOfGeneralCode:
        dumpExprOfGeneralCode(fp, EXPR_GENERALCODE(expr), nindent, child);
        break;
    case STRUCT_CExprOfUnaryNode:
        dumpExprOfUnaryNode(fp, EXPR_U(expr), nindent, child);
        break;
    case STRUCT_CExprOfBinaryNode:
        dumpExprOfBinaryNode(fp, EXPR_B(expr), nindent, child);
        break;
    case STRUCT_CExprOfArrayDecl:
        dumpExprOfArrayDecl(fp, EXPR_ARRAYDECL(expr), nindent, child);
        break;
    case STRUCT_CExprOfDirective:
        dumpExprOfDirective(fp, EXPR_DIRECTIVE(expr), nindent, child);
        break;
    case STRUCT_CExprOfTypeDesc:
        dumpExprOfTypeDesc(fp, EXPR_T(expr), nindent, child);
        break;
    case STRUCT_CExprOfErrorNode:
        dumpExprOfErrorNode(fp, EXPR_ERRORNODE(expr), nindent, child);
        break;
    case STRUCT_CExprOfNull:
        dumpExprOfNull(fp, EXPR_NULL(expr), nindent, child);
        break;
    case STRUCT_CExpr_UNDEF:
    case STRUCT_CExpr_END:
        abort();
    }

    fflush(fp);
}


/**
 * \brief
 * dump node
 *
 * @param fp
 *      output file pointer
 */
void
dumpExpr(FILE *fp, CExpr *expr)
{
    dumpExpr1(fp, expr, 0, 1);
}


/**
 * \brief
 * dump node without detail
 *
 * @param fp
 *      output file pointer
 */
void
dumpExprSingle(FILE *fp, CExpr *expr)
{
    dumpExpr1(fp, expr, 0, 0);
}


PRIVATE_STATIC void
dumpSymbolTableGroup(const char *title, FILE *fp, CCOL_HashTable *ht)
{
    if(CCOL_HT_SIZE(ht) == 0)
        return;

    CCOL_HashEntry *he;
    CCOL_HashSearch hs;
    int c = 0;

    fprintf(fp, "[");
    fputs(title, fp);
    fprintf(fp, "]");

    CCOL_HT_FOREACH(he, hs, ht) {

        CExprOfSymbol *sym = (CExprOfSymbol*)CCOL_HT_DATA(he);
        fprintf(fp, " %s(%s:%d)", sym->e_symName,
            s_CSymbolTypeEnumNames[sym->e_symType], sym->e_putOrder);
        if((++c % 8) == 0)
            fprintf(fp, "\n");
    }

    if(c == 1 || ((c - 1) % 8) != 0)
        fprintf(fp, "\n");

    fflush(fp);
}


/**
 * \brief
 * dump symbol tables
 *
 * @param fp
 *      output file pointer
 */
void
dumpSymbolTable(FILE *fp)
{
    CCOL_DListNode *n;
    int i = CCOL_DL_SIZE(&s_symTabStack);

    CCOL_DL_FOREACH(n, &s_symTabStack) {

        CSymbolTable *symTab = (CSymbolTable*)CCOL_DL_DATA(n);
        fprintf(fp, "level:%d ------\n", i--);
        dumpSymbolTableGroup("ident", fp, &symTab->stb_identGroup);
        dumpSymbolTableGroup("tag",   fp, &symTab->stb_tagGroup);
        dumpSymbolTableGroup("label", fp, &symTab->stb_labelGroup);
    }
}


/**
 * \brief
 * dump errors
 *
 * @param fp
 *      output file pointer
 */
void
dumpError(FILE *fp)
{
    CCOL_SListNode *nd;
    CCOL_SL_FOREACH(nd, &s_errorList) {
        CError *pe = (CError*)CCOL_SL_DATA(nd);
        CExpr *expr0 = pe->pe_expr;
        CExprCommon *cmn = EXPR_C(expr0);
        CLineNumInfo *li = &pe->pe_lineNumInfo;

        switch(pe->pe_errorKind) {
        case EK_WARN:
            fprintf(fp, "WARN : ");
            break;
        case EK_ERROR:
            fprintf(fp, "ERROR : ");
            break;
        case EK_FATAL:
            fprintf(fp, "FATAL : ");
            break;
        }

        fprintf(fp, "FID=%d:%d/%d : ",
            li->ln_fileId, li->ln_lineNum, li->ln_rawLineNum);

        if(pe->pe_expr != NULL) {
            const CExprCodeInfo *ec = &s_CExprCodeInfos[cmn->e_exprCode];
            if(pe->pe_msg == NULL) {
                fprintf(fp, "%s <%s@" ADDR_PRINT_FMT ">\n", ec->ec_name,
                    s_CExprStructEnumNames[cmn->e_struct], (uintptr_t)pe->pe_expr);
            } else {
                fprintf(fp, "%s <%s@" ADDR_PRINT_FMT "> '%s'\n", ec->ec_name,
                    s_CExprStructEnumNames[cmn->e_struct], (uintptr_t)pe->pe_expr, pe->pe_msg);
            }
        } else {
            fprintf(fp, "'%s'\n", pe->pe_msg);
        }
    }
}

//! dump file info
struct dumpFileInfo {
    int id;
    const char *name;
};


PRIVATE_STATIC int
dumpFileInfoComp(const void *a, const void *b)
{
    const struct dumpFileInfo *d1 = (struct dumpFileInfo*)a;
    const struct dumpFileInfo *d2 = (struct dumpFileInfo*)b;

    if(d1->id == d2->id)
        return 0;
    return d1->id < d2->id ? -1 : 1;
}

/**
 * \brief
 * dump file ID table
 *
 * @param fp
 *      output file pointer
 */
void
dumpFileIdTable(FILE *fp)
{
    CCOL_HashEntry *he;
    CCOL_HashSearch hs;
    int sz, c = 0;
    struct dumpFileInfo *dfi, *pdfi;


    if(s_freedFileIdTab) {
        sz = CCOL_HT_SIZE(&s_fileIdToNameTab);
        if(sz == 0)
            return;

        dfi = (struct dumpFileInfo*)malloc(sz * sizeof(struct dumpFileInfo));
        CCOL_HT_FOREACH(he, hs, &s_fileIdToNameTab) {
            int id = (int)(uintptr_t)CCOL_HT_KEY(&s_fileIdToNameTab, he);
            char *file = CCOL_HT_DATA(he);
            pdfi = &dfi[c++];
            pdfi->id = id;
            pdfi->name = file;
        }
    } else {
        sz = CCOL_HT_SIZE(&s_fileIdTab);
        if(sz == 0)
            return;

        dfi = (struct dumpFileInfo*)malloc(sz * sizeof(struct dumpFileInfo));
        CCOL_HT_FOREACH(he, hs, &s_fileIdTab) {
            CFileIdEntry *fie = (CFileIdEntry*)CCOL_HT_DATA(he);
            const char *file = CCOL_HT_KEY(&s_fileIdTab, he);
            pdfi = &dfi[c++];
            pdfi->id = fie->fie_id;
            pdfi->name = file;
        }
    }

    qsort((void*)dfi, sz, sizeof(struct dumpFileInfo), dumpFileInfoComp);

    for(int i = 0; i < sz; ++i) {
        fprintf(fp, "%3d : %s\n", dfi[i].id, dfi[i].name);
    }

    free(dfi);
}

