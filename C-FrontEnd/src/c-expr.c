/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-expr.c
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


CExpr                   *s_exprStart = NULL;
CCOL_DList              s_symTabStack;
CSymbolTable            *s_defaultSymTab;
CCOL_SList              s_exprsTypeDescList;
CCOL_DList              s_typeDescList;
CCOL_SList              s_errorList;
CCOL_HashTable          s_fileIdTab;
CCOL_HashTable          s_fileIdToNameTab;
int                     s_freedFileIdTab = 0;
static CCOL_SList       s_masterSymTabList;
static const char       *s_CBasicTypeEnumCcodes[] = CBasicTypeEnumCcodeDef;
char                    s_charBuf[2][MAX_NAME_SIZ];
int                     s_pragmaPackEnabled = 0;
int                     s_pragmaPackAlign = 0;


#ifdef CEXPR_DEBUG_GCCATTR
/**
 * \brief
 * gcc attribute info for debugging
 */
typedef struct GccAttrCheckInfo {
    CCOL_SList  gac_attrArgs;
    int         gac_extCount;
} GccAttrCheckInfo;


CCOL_SList *s_gccAttrCheckInfos = NULL;
#endif

extern void setLineNumInfo();

PRIVATE_STATIC void
innerCopyExprCommon(CExpr *dst, CExpr *src);

PRIVATE_STATIC void
innerDuplicateExprCommon(CExpr *dst, CExpr *src);

#ifndef MTRACE
/**
 * \brief
 * call malloc and initialize by zero
 */
void*
xalloc(size_t sz)
{
    void *p = malloc(sz);
    assert(p != NULL);
    if(p == NULL) {
        fputs("Out of memory error\n", stderr);
        ABORT();
    }
    memset(p, 0, sz);

    ALLOC_TRACE(p);

    return p;
}
#endif


/**
 * \brief
 * initialize static data
 */
PRIVATE_STATIC void
initStaticExprData()
{
    memset(&s_masterSymTabList, 0, sizeof(s_masterSymTabList));
    memset(&s_symTabStack, 0, sizeof(s_symTabStack));
    memset(&s_errorList, 0, sizeof(s_errorList));
    CCOL_HT_INIT(&s_fileIdTab, CCOL_HT_STRING_KEYS);
    CCOL_HT_INIT(&s_fileIdToNameTab, CCOL_HT_ONE_WORD_KEYS);

    //Default Symbol Table
    s_defaultSymTab = allocSymbolTable();
}


/**
 * \brief
 * free filename to file ID table
 */
PRIVATE_STATIC void
freeFileIdTab()
{
    if(s_freedFileIdTab == 0) {
        CCOL_HashEntry *he;
        CCOL_HashSearch hs;

        CCOL_HT_FOREACH(he, hs, &s_fileIdTab) {
            CFileIdEntry *fie = (CFileIdEntry*)CCOL_HT_DATA(he);
            XFREE(fie);
        }

        CCOL_HT_DESTROY(&s_fileIdTab);
    }
    s_freedFileIdTab = 1;
}


/**
 * \brief
 * free file ID to filename Table
 */
PRIVATE_STATIC void
freeFileIdToNameTab()
{
    CCOL_HashEntry *he;
    CCOL_HashSearch hs;

    CCOL_HT_FOREACH(he, hs, &s_fileIdToNameTab) {
        char *name = (char*)CCOL_HT_DATA(he);
        free(name);
    }

    CCOL_HT_DESTROY(&s_fileIdToNameTab);
}


/**
 * \brief
 * free static data related parsing, symbol table
 */
PRIVATE_STATIC void
freeStaticExprData()
{
    CCOL_SListNode *site;

    // File ID Table
    freeFileIdTab();
    freeFileIdToNameTab();

    // Error List
    CCOL_SL_FOREACH(site, &s_errorList) {
        freeError((CError*)CCOL_SL_DATA(site));
    }

    CCOL_SL_CLEAR(&s_errorList);

    //Symbol Table List, Symbol Table Stack
    freeSymbolTableList();

    freeSymbolTable(s_defaultSymTab);
}


/**
 * \brief
 * initialize static data
 */
void
initStaticData()
{
    initStaticExprData();
    initStaticTypeDescData();
}


/**
 * \brief
 * free static data
 */
void
freeStaticData()
{
    freeStaticTypeDescData();
    freeStaticExprData();
}


/**
 * \brief
 * free errors
 */
void
freeError(CError *err)
{
    if(err->pe_expr)
        freeExpr(err->pe_expr);
    if(err->pe_msg)
        XFREE(err->pe_msg);
    XFREE(err);
}


/**
 * \brief
 * alloc CExprOfErrorNode for syntax error
 *
 * @return
 *      allocated node
 */
CExpr*
exprError()
{
    return exprError1(NULL);
}


/**
 * \brief
 * judge it is necessary to skip error of expr
 *
 * @return
 *      0:no, 1:yes
 */
int
isSkipErrorOutput(CExpr *expr)
{
    if(CCOL_SL_SIZE(&s_errorList) == 0)
        return 0;

    CError *er = (CError*)CCOL_SL_DATA(CCOL_SL_HEAD(&s_errorList));
    int ln = expr ? EXPR_C(expr)->e_lineNumInfo.ln_rawLineNum :
        s_lineNumInfo.ln_rawLineNum;
    return (er->pe_errorKind == EK_ERROR &&
        er->pe_lineNumInfo.ln_rawLineNum == ln);
}


/**
 * \brief
 * alloc CExprOfErrorNode for syntax error
 *
 * @return
 *      allocated node
 */
CExpr*
exprError1(CExpr *nearExpr)
{
    CExprOfErrorNode *expr = allocExprOfErrorNode(nearExpr);
    if(isSkipErrorOutput(nearExpr) == 0)
        addError(nearExpr, CERR_001);

    return (CExpr*)expr;
}


/**
 * \brief
 * alloc CError
 *
 * @param ek
 *      error kind
 * @param expr
 *      target node 
 * @param fmt
 *      error message
 * @param args
 *      arguments of error message
 */
PRIVATE_STATIC CError*
addError0(CErrorKind ek, CExpr *expr, const char *fmt, va_list args)
{
    static int syntax_errs = 0;
    char *buf = s_charBuf[0];

    if(strcmp(fmt, CERR_001) == 0 && syntax_errs++ > 10) {
        return NULL;
    }

    CError *pe = XALLOC(CError);
    CCOL_SL_CONS(&s_errorList, pe);
    if(ek == EK_WARN)
        s_hasWarn = 1;
    else
        s_hasError = 1;
    pe->pe_errorKind = ek;

    if(expr) {
        pe->pe_lineNumInfo = EXPR_C(expr)->e_lineNumInfo;
        pe->pe_expr = expr;
        if(ek != EK_WARN)
            EXPR_ISERROR(expr) = 1;
        EXPR_REF(expr);
    } else if(s_isParsing)  {
        setLineNumInfo();
        pe->pe_lineNumInfo = s_lineNumInfo;
    }

    vsprintf(buf, fmt, args);

    pe->pe_msg = ccol_strdup(buf, MAX_NAME_SIZ);

    return pe;
}


/**
 * \brief
 * add CError for warning
 *
 * @param expr
 *      taget node
 * @param fmt
 *      error message      
 * @param ...
 *      argument for error message
 * @return
 *      allocated node
 */
CError*
addWarn(CExpr *expr, const char *fmt, ...)
{
    va_list args;
    CError *pe;
    va_start(args, fmt);
    pe = addError0(EK_WARN, expr, fmt, args);
    va_end(args);
    return pe;
}


/**
 * \brief
 * add CError
 *
 * @param expr
 *      taget node
 * @param fmt
 *      error message      
 * @param ...
 *      argument for error message
 * @return
 *      allocated node
 */
CError*
addError(CExpr *expr, const char *fmt, ...)
{
    va_list args;
    CError *pe;
    va_start(args, fmt);
    pe = addError0(EK_ERROR, expr, fmt, args);
    va_end(args);
    return pe;
}


/**
 * \brief
 * add CError for fatal error and abort.
 *
 * @param expr
 *      taget node
 * @param fmt
 *      error message      
 * @param ...
 *      argument for error message
 */
void
addFatal(CExpr *expr, const char *fmt, ...)
{
    va_list args;
    CError *pe;
    va_start(args, fmt);
    pe = addError0(EK_FATAL, expr, fmt, args);
    fputs(pe->pe_msg, stderr);
    fputs("\n", stderr);
    va_end(args);
    DBGDUMPEXPR(expr);
    ABORT();
}


/**
 * \brief
 * add symbol at parsing
 *
 * @param sym
 *      target symbol
 * @param symType
 *      symbol type
 */
void
addSymbolInParser(CExpr *sym, CSymbolTypeEnum symType)
{
    const int fn = 0, rc = 0;
    addSymbolAt(EXPR_SYMBOL(sym), NULL, NULL, EXPR_T(NULL), symType, fn, rc);
}


/**
 * \brief
 * add function name symbol at parsing
 *
 * @param sym
 *      symbol type
 */
PRIVATE_STATIC void
addSymbolForFuncInParser(CExprOfSymbol *sym, CExpr *parent)
{
    const int fn = 1, rc = 0;
    addSymbolAt(sym, parent, NULL, NULL, ST_FUNC, fn, rc);
}


/**
 * \brief
 * add redeclaration error
 *
 * @param sym
 *      symbol type
 */
PRIVATE_STATIC void
addRedeclError(CExprOfSymbol *sym)
{
    addError((CExpr*)sym, CERR_098, sym->e_symName);
}


/**
 * \brief
 * merge declarations
 *
 * @param sym1
 *      declared symbol 1
 * @param sym2
 *      declared symbol 2
 */
PRIVATE_STATIC void
checkAndMergeIdent(CExprOfSymbol *sym1, CExprOfSymbol *sym2,
    CCOL_HashTable *ht)
{
    CExprOfTypeDesc *td1 = EXPRS_TYPE(sym1);
    if(td1 == NULL)
        return;
    int hasInit1 = EXPR_C(sym1)->e_hasInit;

    CExprOfTypeDesc *td2 = EXPRS_TYPE(sym2);
    if(td2 == NULL)
        return;
    int hasInit2 = EXPR_C(sym2)->e_hasInit;

    int setInit = 0;

    if(hasInit1) {
        if(hasInit2) {
            addError((CExpr*)sym2, CERR_030, sym2->e_symName);
            return;
        }
    } else {
        hasInit1 = hasInit2;
        if(hasInit2)
            setInit = 1;
    }

    CCompareTypeEnum ct = compareTypeExcludeGccAttr(td1, td2);

    if(ct != CMT_EQUAL) {
        const char *msg = NULL;
        switch(ct) {
        case CMT_DIFF_TYPEQUAL:
            msg = CERR_032;
            break;
        case CMT_DIFF_ARRAYLEN:
            msg = CERR_033;
            break;
        case CMT_DIFF_FUNCPARAM:
            msg = CERR_034;
            break;
        case CMT_DIFF_FUNCRETURN:
            msg = CERR_035;
            break;
        case CMT_DIFF_TYPE:
            msg = CERR_036;
            break;
        default:
            ABORT();
        }
        addError((CExpr*)sym2, msg, sym2->e_symName);
        addError((CExpr*)sym1, CERR_031, sym2->e_symName);
        return;
    }

    CExprOfTypeDesc *ntd = td1;
    CExprOfTypeDesc *tdo1 = getRefType(td1);
    CExprOfTypeDesc *tdo2 = getRefType(td2);
    CExpr *declr2 = sym2->e_declrExpr;
    int toExternDef = (td1->e_sc.esc_isExtern && td2->e_sc.esc_isExtern == 0);
    int funcReplace = ((ETYP_IS_FUNC(tdo1) &&
        (EXPR_L_ISNULL(tdo1->e_paramExpr) &&
        EXPR_L_ISNULL(tdo2->e_paramExpr) == 0)) ||
        (EXPR_CODE(EXPR_PARENT(declr2)) == EC_FUNC_DEF));

    if(toExternDef)
        td1->e_sc.esc_isExtern = 0;

    if(setInit || funcReplace ||
        (toExternDef && (ETYP_IS_FUNC(tdo1) == 0))) {
        //move td2 attribute to ntd
        exprJoinAttrToPre((CExpr*)td2, (CExpr*)td1);
        td2->e_tq.etq_isInline |= td1->e_tq.etq_isInline;
        CCOL_HT_PUT_STR(ht, sym1->e_symName, sym2);
        sym2->e_putOrder = sym1->e_putOrder;
        ntd = td2;
    } else {
        td1->e_tq.etq_isInline |= td2->e_tq.etq_isInline;

        do {
            if(declr2 == NULL)
                break;
            //function definition's attribute will be
            //output to <functionDefinition> - <gccAttributes>
            if(EXPR_CODE(EXPR_PARENT(declr2)) == EC_FUNC_DEF) {
                CCOL_DListNode *ite;
                CExpr *attrs = EXPR_C(td2)->e_gccAttrPre;
                if(attrs == NULL)
                    break;
                EXPR_FOREACH(ite, attrs) {
                    CExprOfBinaryNode *arg = EXPR_B(
                        exprListHeadData(EXPR_L_DATA(ite)));
                    // inline and attribute((gnu_inline)) must be use together.
                    if(arg->e_gccAttrInfo &&
                        arg->e_gccAttrInfo->ga_symbolId == GA_GNU_INLINE) {
                        exprAddAttrToPre(td1, arg);
                    }
                }
            } else {
                exprJoinAttrToPre((CExpr*)td2, (CExpr*)td1);
                exprJoinAttrToPre((CExpr*)td1, (CExpr*)td2);

                if(ETYP_IS_FUNC(tdo1) == 0 || tdo1->e_typeExpr == NULL ||
                    tdo2->e_typeExpr == NULL)
                    break;

                //merge return type's attribute
                CExprOfTypeDesc *rtd1 = EXPR_T(tdo1->e_typeExpr);
                CExprOfTypeDesc *rtd2 = EXPR_T(tdo2->e_typeExpr);
                exprJoinAttrToPre((CExpr*)rtd2, (CExpr*)rtd1);
                exprJoinAttrToPre((CExpr*)rtd1, (CExpr*)rtd2);
                exprChoiceAttr((CExpr*)EXPR_C(rtd1)->e_gccAttrPre);
            }
        } while(0);
    }

    exprChoiceAttr((CExpr*)EXPR_C(ntd)->e_gccAttrPre);
}


/**
 * \brief
 * add symbol to symbol table
 *
 * @param sym
 *      symbol
 * @param parent
 *      parent node
 * @param symTab
 *      symbol table. if NULL, use current symbol table
 * @param td
 *      type descriptor
 * @param symType
 *      symbol type
 */
void
addSymbolAt(CExprOfSymbol *sym, CExpr *parent, CSymbolTable *symTab,
    CExprOfTypeDesc *td, CSymbolTypeEnum symType, int forwardNum,
    int redeclCheck)
{
    /*
    when called in c-parser, td == NULL
    when called in c-comp, td != NULL except symType == ST_LABEL
    */

    assertYYLineno(sym != NULL);
    assertExprStruct((CExpr*)sym, STRUCT_CExprOfSymbol);

    if(td && symType != ST_LABEL && EXPRS_TYPE(sym) == NULL) {
        exprSetExprsType((CExpr*)sym, td);
    }

    EXPR_SYMBOL(sym)->e_symType = symType;

    if(symTab == NULL)
        symTab = (CSymbolTable*)CCOL_DL_DATA(CCOL_DL_AT(&s_symTabStack, forwardNum));
    sym->e_isGlobal = symTab->stb_isGlobal;

    CCOL_HashTable *ht = NULL;
    CSymbolTableGroupEnum group = 0;

    assertExpr((CExpr*)sym, symTab != NULL);

    switch(symType) {
    case ST_TYPE:
    case ST_FUNC:
    case ST_VAR:
    case ST_PARAM:
    case ST_ENUM:
        ht = &symTab->stb_identGroup;
        group = STB_IDENT;
        break;
    case ST_TAG:
        ht = &symTab->stb_tagGroup;
        group = STB_TAG;
        break;
    case ST_LABEL:
    case ST_GCC_LABEL:
        ht = &symTab->stb_labelGroup;
        group = STB_LABEL;
        break;
    case ST_FUNCID:
    case ST_MEMBER:
    case ST_GCC_BUILTIN:
    case ST_GCC_ASM_IDENT:
    case ST_UNDEF:
    case ST_END:
        ABORT();
        break;
    }

    const char *key = EXPR_SYMBOL(sym)->e_symName;
    CCOL_HashEntry *he = CCOL_HT_FIND_STR(ht, key);

    if(he == NULL) {
        if(td && group == STB_TAG && td->e_isNoMemDecl) {
            CExprOfSymbol *tagSym = findSymbolByGroup(key, STB_TAG);
            CExprOfTypeDesc *refTd;

            if(tagSym && (refTd = EXPRS_TYPE(tagSym))) {
                exprSetExprsType((CExpr*)tagSym, refTd);
                if(td->e_refType == NULL && td != refTd)
                    td->e_refType = refTd;
                return;
            }
        }

        sym->e_putOrder = symTab->stb_putCount++;
        CCOL_HT_PUT_STR(ht, key, sym);
    } else if(redeclCheck) {
        CExprOfSymbol *hsym = EXPR_SYMBOL(CCOL_HT_DATA(he));

        if(hsym == sym)
            return;

        CExprOfTypeDesc *htd = NULL;
        if(symType != ST_LABEL && symType != ST_GCC_LABEL) {
            assertExpr((CExpr*)td, EXPRS_TYPE(hsym) != td);
            htd = EXPRS_TYPE(hsym);
        }

        switch(group) {
        case STB_IDENT:
            if(hsym->e_symType != symType) {
                addError((CExpr*)sym, CERR_099, sym->e_symName);
            } else {
                switch(symType) {
                case ST_FUNC:
                    break;
                case ST_VAR:
                    //not allow extern/extern_def -> static
                    //allow static -> extern/extern_def
                    if(htd->e_sc.esc_isStatic == 0 && td->e_sc.esc_isStatic) {
                        addError((CExpr*)sym, CERR_100, sym->e_symName);
                        return;
                    } else {
                        td->e_preDeclType = htd;
                    }
                    break;
                default:
                    addRedeclError(sym);
                    return;
                }
                checkAndMergeIdent(hsym, sym, ht);
            }
            break;
        case STB_TAG:
            if(EXPRS_TYPE(hsym) != td) {
                int nm1 = EXPRS_TYPE(hsym)->e_isNoMemDecl;
                int nm2 = td->e_isNoMemDecl;

                if(nm1 == 0 && nm2 == 0) {
                    addRedeclError(sym);
                } else if(nm1 == 1 && nm2 == 0) {
                    /* replace declaration with new type which has MEMBER_DECLS */
                    EXPRS_TYPE(hsym)->e_refType = td;
                    sym->e_putOrder = symTab->stb_putCount++;
                    CCOL_HT_SET_DATA(he, sym);
                } else if(nm2 == 1) {
                    td->e_refType = EXPRS_TYPE(hsym);
                }
            }
            break;
        case STB_LABEL:
            if(symType == ST_GCC_LABEL) {
                EXPR_ISGCCSYNTAX(sym) = 1;
                if(hsym->e_symType != ST_GCC_LABEL)
                    addRedeclError(sym);
                else if(hsym->e_isGccLabelDecl)
                    hsym->e_isGccLabelDecl = 0;
            } else {
                addRedeclError(sym);
            }
            break;
        default:
            break;
        }
    }

    //check parameter hiding
    if(redeclCheck && group == STB_IDENT && symTab->stb_isGlobal == 0) {
        CSymbolTable *symTab1 = symTab->stb_parentTab;
        CExprOfSymbol *hsym = findSymbolByGroup1(symTab1, sym->e_symName, group);
        if(hsym && hsym->e_symType == ST_PARAM) {
            addWarn((CExpr*)sym, CWRN_009, sym->e_symName);
        }
    }
}


/**
 * \brief
 * get current symbol table
 *
 * @return
 *      symbol table
 */
CSymbolTable*
getCurrentSymbolTable()
{
    return (CSymbolTable*)CCOL_DL_DATA(CCOL_DL_HEAD(&s_symTabStack));
}


/**
 * \brief
 * get global scope symbol table
 *
 * @return
 *      symbol table
 */
CSymbolTable*
getGlobalSymbolTable()
{
    CCOL_DListNode *ite;
    CCOL_DL_FOREACH_REVERSE(ite, &s_symTabStack) {
        CSymbolTable *symTab = CCOL_DL_DATA(ite);
        if(symTab->stb_isGlobal)
            return symTab;
    }

    return NULL;
}


/**
 * \brief
 * add symbol table
 *
 * @param expr
 *      node which has block scope
 */
void
pushSymbolTableToExpr(CExpr *expr)
{
    int isGlobal = 0;
    if(CCOL_DL_SIZE(&s_symTabStack) == 0) {
        CCOL_DL_CONS(&s_symTabStack, s_defaultSymTab);
        isGlobal = 1;
    }
    
    CSymbolTable *symTab = NULL;

    if(expr && EXPR_C(expr)->e_symTab) {
        symTab = EXPR_C(expr)->e_symTab;
    } else {
        symTab = allocSymbolTable();
        CSymbolTable *parentTab = getCurrentSymbolTable();
        symTab->stb_parentTab = parentTab;
        symTab->stb_isGlobal = isGlobal;

        if(expr && EXPR_CODE(expr) == EC_COMP_STMT) {
            CExpr *pexpr = EXPR_PARENT(expr);
            if(pexpr && EXPR_CODE(pexpr) == EC_FUNC_DEF)
                symTab->stb_isFuncDefBody = 1;
        }

        if(parentTab)
            parentTab->stb_childTab = symTab;
        CCOL_SL_CONS(&s_masterSymTabList, symTab);
    }

    CCOL_DL_CONS(&s_symTabStack, symTab);

    if(expr)
        EXPR_C(expr)->e_symTab = symTab;
    if(s_debugSymbol) {
        if(expr) {
            DBGPRINTC(ESQ_RED, (">pushSymTab>"));
        } else {
            DBGPRINTC(ESQ_RED, (">pushSymTab@%d>", s_rawlineNo));
        }
    }
}


/**
 * \brief
 * add symbol table
 */
void
pushSymbolTable()
{
    pushSymbolTableToExpr(NULL);
}


/**
 * \brief
 * remove current symbol table
 *
 * @return
 *      line number where symbol table add
 */
CLineNumInfo
popSymbolTable()
{
    if(s_debugSymbol) {
        DBGPRINTC(ESQ_RED, ("\n--- Before popSymbolTable ---\n"));
        dumpSymbolTable(stdout);
    }

    CSymbolTable *symTab = CCOL_DL_REMOVE_HEAD(&s_symTabStack);

    if(s_debugSymbol) {
        DBGPRINTC(ESQ_RED, ("<popSymTab@%d<", s_rawlineNo));
    }

    if(symTab)
        return symTab->stb_lineNumInfo;

    CLineNumInfo ln;
    memset(&ln, 0, sizeof(ln));
    return ln;
}


/**
 * \brief
 * alloc symbol table
 *
 * @param parentSymTab
 *      previous symbol table
 * @return
 *      allocated symbol table
 */
CSymbolTable*
allocSymbolTable1(CSymbolTable *parentSymTab)
{
    CSymbolTable *symTab = allocSymbolTable();
    CSymbolTable *childTab = parentSymTab->stb_childTab;
    symTab->stb_parentTab = parentSymTab;
    symTab->stb_childTab = childTab;
    parentSymTab->stb_childTab = symTab;
    if(childTab) {
        childTab->stb_parentTab = symTab;
    }

    CCOL_SL_CONS(&s_masterSymTabList, symTab);

    return symTab;
}


/**
 * \brief
 * free symbol table
 */
void
freeSymbolTableList()
{
    while(CCOL_DL_REMOVE_HEAD(&s_symTabStack) != NULL);

    CCOL_SListNode *ite;
    CCOL_SL_FOREACH(ite, &s_masterSymTabList) {
        CSymbolTable *symTab = (CSymbolTable*)CCOL_SL_DATA(ite);
        if(symTab != s_defaultSymTab)
            freeSymbolTable(symTab);
    }

    CCOL_SL_CLEAR(&s_masterSymTabList);
}


/**
 * \brief
 * find symbol in symbol table
 *
 * @param symbol
 *      target symbol
 * @return
 *      NULL for not found, or symbol in symbol table
 */
CExprOfSymbol*
findSymbol(const char *symbol)
{
    CExprOfSymbol *sym = findSymbolByGroup(symbol, STB_IDENT);
    if(sym == NULL) {
        sym = findSymbolByGroup(symbol, STB_TAG);
        if(sym == NULL)
            sym = findSymbolByGroup(symbol, STB_LABEL);
    }
    return sym;
}


/**
 * \brief
 * find symbol in symbol table
 *
 * @param symbol
 *      target symbol
 * @param group
 *      symbol table group
 * @return
 *      NULL for not found, or symbol in symbol table
 */
CExprOfSymbol*
findSymbolByGroup(const char *symbol, CSymbolTableGroupEnum group)
{
    return findSymbolByGroup1(NULL, symbol, group);
}


/**
 * \brief
 * get raw symbol table of specified symbol table group
 *
 * @param symTab
 *      symbol table
 * @param group
 *      symbol table group
 * @return
 *      raw symbol table 
 */
CCOL_HashTable*
getSymbolHashTable(CSymbolTable *symTab, CSymbolTableGroupEnum group)
{
    switch(group) {
    case STB_IDENT:
        return &symTab->stb_identGroup;
    case STB_TAG:
        return &symTab->stb_tagGroup;
    case STB_LABEL:
        return &symTab->stb_labelGroup;
    default:
        ABORT();
        return NULL;
    }
}


/**
 * \brief
 * find symbol in symbol table
 *
 * @param symTab
 *      symbol table
 * @param symbol
 *      target symbol
 * @param group
 *      symbol table group
 * @return
 *      NULL for not found, or symbol in symbol table
 */
PRIVATE_STATIC CExprOfSymbol*
findSymbolByGroup0(CSymbolTable *symTab, const char *symbol, CSymbolTableGroupEnum group)
{
    CCOL_HashTable *ht = getSymbolHashTable(symTab, group);
    CCOL_HashEntry *he;
    
    he = CCOL_HT_FIND_STR(ht, symbol);

    if(he != NULL) {
        CExprOfSymbol *sym = EXPR_SYMBOL(CCOL_HT_DATA(he));
        return sym;
    }

    return NULL;
}


/**
 * \brief
 * find symbol in symbol table. if not found in
 * specified symbol table, find parent symbol table.
 *
 * @param symTab
 *      symbol table
 * @param symbol
 *      target symbol
 * @param group
 *      symbol table group
 * @return
 *      NULL for not found, or symbol in symbol table
 */
CExprOfSymbol*
findSymbolByGroup1(CSymbolTable *symTab, const char *symbol,
    CSymbolTableGroupEnum group)
{
    if(symTab == NULL)
        symTab = (CSymbolTable*)CCOL_DL_DATA(CCOL_DL_HEAD(&s_symTabStack));

    while(symTab) {
        CExprOfSymbol *sym = findSymbolByGroup0(symTab, symbol, group);
        if(sym)
            return sym;
        symTab = symTab->stb_parentTab;
    }

    return NULL;
}


/**
 * \brief
 * remove symbol in symbol table
 *
 * @param symTab
 *      symbol table
 * @param symbol
 *      target symbol
 * @param group
 *      symbol table group
 * @return
 *      removed symbol
 */
CExprOfSymbol*
removeSymbolByGroup(CSymbolTable *symTab, const char *symbol,
    CSymbolTableGroupEnum group)
{
    CCOL_HashTable *ht = getSymbolHashTable(symTab, group);
    CCOL_HashEntry *he = CCOL_HT_FIND_STR(ht, symbol);

    if(he) {
        CExprOfSymbol *sym = EXPR_SYMBOL(CCOL_HT_DATA(he));
        CCOL_HT_REMOVE_STR(ht, symbol);
        return sym;
    }

    return NULL;
}


/**
 * \brief
 * add symbol to symbol table
 *
 * @param symTab
 *      symbol table
 * @param sym
 *      target symbol
 * @param group
 *      symbol table group
 */
void
addSymbolDirect(CSymbolTable *symTab, CExprOfSymbol *sym,
    CSymbolTableGroupEnum group)
{
    CCOL_HashTable *ht = getSymbolHashTable(symTab, group);
    CCOL_HashEntry *he = CCOL_HT_FIND_STR(ht, sym->e_symName);
    if(he) {
        addFatal(NULL, CFTL_005);
        return;
    }

    sym->e_putOrder = symTab->stb_putCount++;
    CCOL_HT_PUT_STR(ht, sym->e_symName, sym);
}


/**
 * \brief
 * alloc CExprOfSymbol
 *
 * @param exprCode
 *      expression code
 * @param
 *      symbol token
 * @return
 *      allocated node
 */
CExprOfSymbol*
allocExprOfSymbol(CExprCodeEnum exprCode, char *token)
{
    assert(token != NULL);
    EXPR_ALLOC(CExprOfSymbol, expr, exprCode);
    expr->e_symName = token;
    expr->e_symType = ST_UNDEF;
    return expr;
}


/**
 * \brief
 * copy CExprOfSymbol
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfSymbol*
copyExprOfSymbol(CExprOfSymbol *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfSymbol, dst, src);
    dst->e_symName = ccol_strdup(src->e_symName, MAX_NAME_SIZ);
    EXPR_SET0(dst->e_valueExpr, copyExpr(src->e_valueExpr));
    return dst;
}


/**
 * \brief
 * shallow copy CExprOfSymbol
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
CExprOfSymbol*
duplicateExprOfSymbol(CExprOfSymbol *src)
{
    assert(src);
    CExprOfSymbol *dst = allocExprOfSymbol(EXPR_CODE(src),
        ccol_strdup(src->e_symName, strlen(src->e_symName)));
    innerDuplicateExprCommon((CExpr*)dst, (CExpr*)src);
    dst->e_symType = src->e_symType;
    dst->e_valueExpr = NULL;

    return dst;
}


/**
 * \brief
 * alloc CExprOfSymbol
 *
 * @param
 *      symbol token
 * @return
 *      allocated node
 */
CExprOfSymbol*
allocExprOfSymbol2(const char *token)
{
    return allocExprOfSymbol(EC_IDENT, ccol_strdup(token, MAX_NAME_SIZ));
}


/**
 * \brief
 * copy CExprCommon src to dst
 *
 * @param dst
 *      destination node
 * @param src
 *      source node
 */
PRIVATE_STATIC void
innerCopyExprCommon(CExpr *dst, CExpr *src)
{
    CExprCommon *cdst = EXPR_C(dst);
    CExprCommon *csrc = EXPR_C(src);

    memcpy(dst, src, sizeof(dst));
    EXPR_C(dst)->e_refCount = 0;

    EXPR_SET0(cdst->e_gccAttrPre, copyExpr(csrc->e_gccAttrPre));
    EXPR_SET0(cdst->e_gccAttrPost, copyExpr(csrc->e_gccAttrPost));

#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("@CExpr-copy:" ADDR_PRINT_FMT " -> " ADDR_PRINT_FMT "\n", (uintptr_t)src, (uintptr_t)dst));
#endif
}


/**
 * \brief
 * shallow copy CExprCommon src to dst
 *
 * @param dst
 *      destination node
 * @param src
 *      source node
 */
PRIVATE_STATIC void
innerDuplicateExprCommon(CExpr *dst, CExpr *src)
{
    CExprCommon *cdst = EXPR_C(dst);
    CExprCommon *csrc = EXPR_C(src);

    CExprOfList *attrPre = duplicateExprOfList(EXPR_L(csrc->e_gccAttrPre));
    cdst->e_gccAttrPre = (CExpr*)attrPre;
    if(attrPre)
        EXPR_REF(attrPre);

    CExprOfList *attrPost = duplicateExprOfList(EXPR_L(csrc->e_gccAttrPost));
    cdst->e_gccAttrPost = (CExpr*)attrPost;
    if(attrPost)
        EXPR_REF(attrPost);

    EXPRS_TYPE(cdst) = EXPRS_TYPE(csrc);
    cdst->e_refCount = 0;
    EXPR_ISCONVERTED(cdst) = EXPR_ISCONVERTED(csrc);
    EXPR_ISGCCSYNTAX(cdst) = EXPR_ISGCCSYNTAX(csrc);
    EXPR_GCCEXTENSION(cdst) = EXPR_GCCEXTENSION(csrc);
#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("@CExpr-dup:" ADDR_PRINT_FMT " -> " ADDR_PRINT_FMT "\n", (uintptr_t)src, (uintptr_t)dst));
#endif
}


/**
 * \brief
 * free CExprOfSymbol
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfSymbol(CExprOfSymbol *expr)
{
    if(expr == NULL)
        return;
    if(expr->e_symName)
        free(expr->e_symName);
    if(expr->e_valueExpr)
        freeExpr(expr->e_valueExpr);
}


/**
 * \brief
 * alloc CExprOfNumberConst
 *
 * @param exprCode
 *      expression code
 * @param bt
 *      basic type
 * @param c
 *      cardinal number
 * @param orgToken
 *      original token
 * @return
 *      allocated node
 */
CExprOfNumberConst*
allocExprOfNumberConst(CExprCodeEnum exprCode, CBasicTypeEnum bt,
    CCardinalEnum c, char *orgToken)
{
    EXPR_ALLOC(CExprOfNumberConst, expr, exprCode);
    expr->e_orgToken = orgToken;
    expr->e_cardinal = c;

    int base = 0;

    switch(c) {
    case CD_BIN: base = 2; break;
    case CD_OCT: base = 8; break;
    case CD_DEC: base = 10; break;
    case CD_HEX: base = 16; break;
    default: ABORT();
    }

    int overRange1 = 0, overRange2 = 0;

    if(orgToken) {
        int len = strlen(orgToken);
        char digits[len + 1];

        if(orgToken[0] == '0' && len > 2) {
            char c = orgToken[1];
            if(c == 'x' || c == 'X' || c == 'b' || c == 'B')
                strcpy(digits, orgToken + 2); // hex, bin
            else
                strcpy(digits, orgToken + 1); // oct
        } else
            strcpy(digits, orgToken); // dec

        errno = 0;

        switch(bt) {
        case BT_CHAR:
        case BT_SHORT:
        case BT_INT:
        case BT_LONG:
        case BT_LONGLONG: {
                long long n = expr->e_numValue.ll = strtoll(digits, NULL, base);
                if(errno) {
                    overRange1 = 1;
                    if(bt == BT_LONGLONG) {
                        bt = BT_UNSIGNED_LONGLONG;
                        errno = 0;
                        n = expr->e_numValue.ull = strtoull(digits, NULL, base);
                        overRange1 = (errno != 0);
                    }
                } else {
                    switch(bt) {
                    case BT_CHAR:
                        if(n > UCHAR_MAX) overRange1 = 1;
                        else if(n > SCHAR_MAX) bt = BT_UNSIGNED_CHAR;
                        overRange1 = 1;
                    case BT_SHORT:
                        if(n > USHRT_MAX) overRange1 = 1;
                        else if(n > SHRT_MAX) bt = BT_UNSIGNED_SHORT;
                    case BT_INT:
                        if(n > UINT_MAX) overRange1 = 1;
                        else if(n > INT_MAX) bt = BT_UNSIGNED_INT;
                    case BT_LONG:
                        if(n > ULONG_MAX) overRange1 = 1;
                        else if(n > LONG_MAX) bt = BT_UNSIGNED_LONG;
                    default:
                        break;
                    }
                }
            }
            break;
        case BT_UNSIGNED_CHAR:
        case BT_UNSIGNED_SHORT:
        case BT_UNSIGNED_INT:
        case BT_UNSIGNED_LONG:
        case BT_UNSIGNED_LONGLONG: {
                unsigned long long n = expr->e_numValue.ull = strtoull(digits, NULL, base);
                if(errno) {
                    overRange1 = 1;
                } else {
                    switch(bt) {
                    case BT_CHAR:
                        overRange1 = (n > UCHAR_MAX); break;
                    case BT_SHORT:
                        overRange1 = (n > USHRT_MAX); break;
                    case BT_INT:
                        overRange1 = (n > UINT_MAX); break;
                    case BT_LONG:
                        overRange1 = (n > ULONG_MAX); break;
                    default:
                        break;
                    }
                }
            }
            break;
        case BT_FLOAT:
        case BT_FLOAT_IMAGINARY:
          {
            long double n = expr->e_numValue.ld = strtof(digits, NULL);
            if(errno) {
              overRange2 = 1;
            }
            else {
              overRange2 = (n > FLT_MAX);
            }
          } break;
        case BT_DOUBLE:
        case BT_DOUBLE_IMAGINARY:
          {
            long double n = expr->e_numValue.ld = strtod(digits, NULL);
            if(errno) {
              overRange2 = 1;
            }
            else {
              overRange2 = (n > DBL_MAX);
            }
          } break;
        case BT_LONGDOUBLE:
          {
            long double n = expr->e_numValue.ld = strtold(digits, NULL);
            if(errno) {
              overRange2 = 1;
            }
            else {
              overRange2 = (n > DBL_MAX);
            }
          } break;
        default:
            ABORT();
        }
    }

    expr->e_basicType = bt;
    
    const char *cerr = (overRange1 ? CERR_121 : (overRange2 ? CERR_122 : 0));

    if(cerr)
        addError((CExpr*)expr, cerr, s_CBasicTypeEnumCcodes[bt]);

    return expr;
}


/**
 * \brief
 * copy CExprOfNumberConst
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfNumberConst*
copyExprOfNumberConst(CExprOfNumberConst *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfNumberConst, dst, src);
    EXPR_SET0(dst->e_orgToken, ccol_strdup(src->e_orgToken, MAX_NAME_SIZ));

    return dst;
}


/**
 * \brief
 * alloc CExprOfNumberConst
 *
 * @param nvt
 *      number value
 * @return
 *      allocated node
 */
CExprOfNumberConst*
allocExprOfNumberConst1(CNumValueWithType *nvt)
{
    char *token = s_charBuf[0];
    switch(nvt->nvt_numKind) {
    case NK_LL:
        sprintf(token, "%lld", nvt->nvt_numValue.ll);
        break;
    case NK_ULL:
        sprintf(token, "%llu", nvt->nvt_numValue.ull);
        break;
    case NK_LD:
        sprintf(token, "%Lf", nvt->nvt_numValue.ld);
        break;
    }

    CExprOfNumberConst *e = allocExprOfNumberConst(EC_NUMBER_CONST,
        nvt->nvt_basicType, CD_DEC, ccol_strdup(token, MAX_NAME_SIZ));

    e->e_numValue = nvt->nvt_numValue;

    return e;
}


/**
 * \brief
 * alloc CExprOfNumberConst
 *
 * @param n
 *      number value
 * @param bt
 *      basic type
 * @return
 *      allocated node
 */
CExprOfNumberConst*
allocExprOfNumberConst2(long long n, CBasicTypeEnum bt)
{
    CNumValueWithType nvt;
    memset(&nvt, 0, sizeof(nvt));
    nvt.nvt_basicType = bt;
    nvt.nvt_numKind = NK_LL;
    nvt.nvt_numValue.ll = n;
    return allocExprOfNumberConst1(&nvt);
}


/**
 * \brief
 * free CExprOfNumberConst
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfNumberConst(CExprOfNumberConst *expr)
{
    if(expr == NULL)
        return;
    if(expr->e_orgToken)
        free(expr->e_orgToken);
}


/**
 * \brief
 * alloc CExprOfCharConst
 *
 * @param exprCode
 *      expression code
 * @param orgToken
 *      original token
 * @param charType
 *      character type
 * @return
 *      allocated node
 */
CExprOfCharConst*
allocExprOfCharConst(CExprCodeEnum exprCode, char *orgToken,
    CCharTypeEnum charType)
{
    assert(orgToken != NULL);
    EXPR_ALLOC(CExprOfCharConst, expr, exprCode);
    expr->e_orgToken = orgToken;
    expr->e_charType = charType;

    char *s = malloc(8);
    int numChar = 0;
    if(unescChar(s, orgToken, &numChar, (charType == CT_WIDE)) == 0) {
        addError((CExpr*)expr, CERR_101);
        s[0] = 0;
    } else if((charType == CT_MB && numChar > 1) || numChar > 2) {
        addError((CExpr*)expr, CERR_102);
        s[0] = 0;
    }
    expr->e_token = s;

    return expr;
}


/**
 * \brief
 * copy CExprOfCharConst
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfCharConst*
copyExprOfCharConst(CExprOfCharConst *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfCharConst, dst, src);
    dst->e_token = ccol_strdup(src->e_token, MAX_NAME_SIZ);
    dst->e_orgToken = ccol_strdup(src->e_orgToken, MAX_NAME_SIZ);

    return dst;
}


/**
 * \brief
 * free CExprOfCharConst
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfCharConst(CExprOfCharConst *expr)
{
    if(expr == NULL)
        return;
    if(expr->e_orgToken)
        free(expr->e_orgToken);
    if(expr->e_token)
        free(expr->e_token);
}


/**
 * \brief
 * alloc CExprOfStringConst
 *
 * @param exprCode
 *      expression code
 * @param orgToken
 *      original token
 * @param charType
 *      character type
 * @return
 *      allocated node
 */
CExprOfStringConst*
allocExprOfStringConst(CExprCodeEnum exprCode, char *orgToken,
    CCharTypeEnum charType)
{
    assert(orgToken != NULL);
    EXPR_ALLOC(CExprOfStringConst, expr, exprCode);
    expr->e_orgToken = orgToken;
    expr->e_charType = charType;

    int numChar = 0;
    if(unescChar(NULL, orgToken, &numChar, 0) == 0) {
        addError((CExpr*)expr, CERR_103);
    }

    expr->e_numChars = numChar;

    return expr;
}


/**
 * \brief
 * copy CExprOfStringConst
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfStringConst*
copyExprOfStringConst(CExprOfStringConst *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfStringConst, dst, src);
    dst->e_orgToken = ccol_strdup(src->e_orgToken, MAX_NAME_SIZ);

    return dst;
}


/**
 * \brief
 * alloc CExprOfStringConst
 *
 * @param orgToken
 *      original token
 * @return
 *      allocated node
 */
CExprOfStringConst*
allocExprOfStringConst2(const char *orgToken)
{
    return allocExprOfStringConst(EC_IDENT,
        ccol_strdup(orgToken, MAX_NAME_SIZ), CT_MB);
}


/**
 * \brief
 * free CExprOfStringConst
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfStringConst(CExprOfStringConst *expr)
{
    if(expr == NULL)
        return;
    if(expr->e_orgToken)
        free(expr->e_orgToken);
}


/**
 * \brief
 * alloc CExprOfList
 *
 * @param exprCode
 *      expression code
 * @return
 *      allocated node
 */
CExprOfList*
allocExprOfList(CExprCodeEnum exprCode)
{
    EXPR_ALLOC(CExprOfList, expr, exprCode);
    expr->e_aux = 0; // clear
    expr->e_aux_info = NULL;
    return expr;
}

/**
 * \brief
 * copy CExprOfList
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfList*
copyExprOfList(CExprOfList *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfList, dst, src);
    EXPR_SET0(dst->e_symbol, copyExpr((CExpr*)src->e_symbol));

    memset(&dst->e_dlist, 0, sizeof(dst->e_dlist));
    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, src) {
        CExpr *e = EXPR_L_DATA(ite);
        CCOL_DL_ADD(&dst->e_dlist, copyExpr(e));
    }

    return dst;
}


/**
 * \brief
 * shallow copy CExprOfList
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
CExprOfList*
duplicateExprOfList(CExprOfList *src)
{
    if(EXPR_ISNULL(src))
        return NULL;

    CExprOfList *dst = allocExprOfList(EXPR_CODE(src));
    innerDuplicateExprCommon((CExpr*)dst, (CExpr*)src);

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, src) {
        CExpr *e = EXPR_L_DATA(ite);
        exprListAdd((CExpr*)dst, e);
    }

    return dst;
}


/**
 * \brief
 * alloc CExprOfList and 1 child
 *
 * @param exprCode
 *      expression code
 * @param expr1
 *      child node
 * @return
 *      allocated node
 */
CExprOfList*
allocExprOfList1(CExprCodeEnum exprCode, CExpr *expr1)
{
    if(expr1 == NULL)
        expr1 = exprNull();

    CExprOfList *expr = allocExprOfList(exprCode);
    exprListAdd((CExpr*)expr, expr1);
    return expr;
}


/**
 * \brief
 * alloc CExprOfList and 2 children
 *
 * @param exprCode
 *      expression code
 * @param expr1
 *      child node
 * @param expr2
 *      child node
 * @return
 *      allocated node
 */
CExprOfList*
allocExprOfList2(CExprCodeEnum exprCode, CExpr *expr1, CExpr *expr2)
{
    CExprOfList *expr = allocExprOfList1(exprCode, expr1);
    exprListAdd((CExpr*)expr, expr2);
    return expr;
}


/**
 * \brief
 * alloc CExprOfList and 3 children
 *
 * @param exprCode
 *      expression code
 * @param expr1
 *      child node
 * @param expr2
 *      child node
 * @param expr3
 *      child node
 * @return
 *      allocated node
 */
CExprOfList*
allocExprOfList3(CExprCodeEnum exprCode, CExpr *expr1, CExpr *expr2,
    CExpr *expr3)
{
    CExprOfList *expr = allocExprOfList2(exprCode, expr1, expr2);
    exprListAdd((CExpr*)expr, expr3);
    return expr;
}


/**
 * \brief
 * alloc CExprOfList and 4 children
 *
 * @param exprCode
 *      expression code
 * @param expr1
 *      child node
 * @param expr2
 *      child node
 * @param expr3
 *      child node
 * @param expr4
 *      child node
 * @return
 *      allocated node
 */
CExprOfList*
allocExprOfList4(CExprCodeEnum exprCode, CExpr *expr1, CExpr *expr2,
    CExpr *expr3, CExpr *expr4)
{
    CExprOfList *expr = allocExprOfList3(exprCode, expr1, expr2, expr3);
    exprListAdd((CExpr*)expr, expr4);
    return expr;
}


/**
 * \brief
 * alloc CExprOfList and 5 children
 *
 * @param exprCode
 *      expression code
 * @param expr1
 *      child node
 * @param expr2
 *      child node
 * @param expr3
 *      child node
 * @param expr4
 *      child node
 * @param expr5
 *      child node
 * @return
 *      allocated node
 */
CExprOfList*
allocExprOfList5(CExprCodeEnum exprCode, CExpr *expr1, CExpr *expr2,
		 CExpr *expr3, CExpr *expr4, CExpr *expr5)
{
  CExprOfList *expr = allocExprOfList4(exprCode, expr1, expr2, expr3, expr4);
  exprListAdd((CExpr*)expr, expr5);
  return expr;
}


/**
 * \brief
 * free CExprOfList
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfList(CExprOfList *expr)
{
    if(expr == NULL)
        return;

    CCOL_DListNode *ite, *iten;
    EXPR_FOREACH_SAFE(ite, iten, expr) {
        freeExpr(CCOL_DL_DATA(ite));
    }

    if(expr->e_symbol)
        freeExpr((CExpr*)expr->e_symbol);

    CCOL_DL_CLEAR(&expr->e_dlist);
}


/**
 * \brief
 * alloc CExprOfGeneralCode
 *
 * @param exprCode
 *      expression code
 * @return
 *      allocated node
 */
CExprOfGeneralCode*
allocExprOfGeneralCode(CExprCodeEnum exprCode, int code)
{
    EXPR_ALLOC(CExprOfGeneralCode, expr, exprCode);
    expr->e_code = code;
    return expr;
}


/**
 * \brief
 * copy CExprOfGeneralCode
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfGeneralCode*
copyExprOfGeneralCode(CExprOfGeneralCode *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfGeneralCode, dst, src);

    return dst;
}


#define innerFreeExprOfGeneralCode(expr)


/**
 * \brief
 * alloc CExprOfUnaryNode
 *
 * @param exprCode
 *      expression code
 * @param node
 *      child node
 * @return
 *      allocated node
 */
CExprOfUnaryNode*
allocExprOfUnaryNode(CExprCodeEnum exprCode, CExpr *node)
{
    EXPR_ALLOC(CExprOfUnaryNode, expr, exprCode);
    if(node == NULL)
        node = (CExpr*)allocExprOfNull();
    else
        exprCopyLineNum((CExpr*)expr, node);

    if(node) {
        expr->e_node = node;
        EXPR_REF(node);
        EXPR_PARENT(node) = (CExpr*)expr;
    }

    return expr;
}


/**
 * \brief
 * copy CExprOfUnaryNode
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfUnaryNode*
copyExprOfUnaryNode(CExprOfUnaryNode *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfUnaryNode, dst, src);
    EXPR_SET0(dst->e_node, copyExpr(src->e_node));

    return dst;
}


/**
 * \brief
 * free CExprOfUnaryNode
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfUnaryNode(CExprOfUnaryNode *expr)
{
    if(expr == NULL)
        return;
    if(expr->e_node)
        freeExpr(expr->e_node);
}


/**
 * \brief
 * alloc CExprOfBinaryNode
 *
 * @param exprCode
 *      expression code
 * @param node1
 *      child node
 * @param node2
 *      child node
 * @return
 *      allocated node
 */
CExprOfBinaryNode*
allocExprOfBinaryNode1(CExprCodeEnum exprCode, CExpr *node1, CExpr *node2)
{
    EXPR_ALLOC(CExprOfBinaryNode, expr, exprCode);

    if(node1) {
        exprCopyLineNum((CExpr*)expr, node1);
    } else if(node2) {
        exprCopyLineNum((CExpr*)expr, node2);
    }

    if(node1) {
        expr->e_nodes[0] = node1;
        EXPR_REF(node1);
        EXPR_PARENT(node1) = (CExpr*)expr;
    }

    if(node2) {
        expr->e_nodes[1] = node2;
        EXPR_REF(node2);
        EXPR_PARENT(node2) = (CExpr*)expr;
    }

    return expr;
}


/**
 * \brief
 * copy CExprOfBinaryNode
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfBinaryNode*
copyExprOfBinaryNode(CExprOfBinaryNode *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfBinaryNode, dst, src);
    EXPR_SET0(dst->e_nodes[0], copyExpr(src->e_nodes[0]));
    EXPR_SET0(dst->e_nodes[1], copyExpr(src->e_nodes[1]));

    return dst;
}


/**
 * \brief
 * alloc CExprOfBinaryNode with completiton null node
 *
 * @param exprCode
 *      expression code
 * @param node1
 *      child node
 * @param node2
 *      child node
 * @return
 *      allocated node
 */
CExprOfBinaryNode*
allocExprOfBinaryNode2(CExprCodeEnum exprCode, CExpr *node1, CExpr *node2)
{
    EXPR_ALLOC(CExprOfBinaryNode, expr, exprCode);

    if(node1 != NULL) {
        exprCopyLineNum((CExpr*)expr, node1);
    } else if(node2 != NULL) {
        exprCopyLineNum((CExpr*)expr, node2);
    }

    if(node1 == NULL)
        node1 = (CExpr*)allocExprOfNull();
    if(node2 == NULL)
        node2 = (CExpr*)allocExprOfNull();

    expr->e_nodes[0] = node1;
    EXPR_REF(node1);
    EXPR_PARENT(node1) = (CExpr*)expr;
    expr->e_nodes[1] = node2;
    EXPR_REF(node2);
    EXPR_PARENT(node2) = (CExpr*)expr;

    return expr;
}


/**
 * \brief
 * free CExprOfBinaryNode
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfBinaryNode(CExprOfBinaryNode *expr)
{
    if(expr == NULL)
        return;
    for(int i = 0; i < 2; ++i)
        if(expr->e_nodes[i])
            freeExpr(expr->e_nodes[i]);
}


/**
 * \brief
 * alloc CExprOfArrayDecl
 *
 * @param typeQualExpr
 *      type qualifer node for array length
 * @param lenExpr
 *      array length node
 * @return
 *      allocated node
 */
CExprOfArrayDecl*
allocExprOfArrayDecl(CExpr *typeQualExpr, CExpr *lenExpr)
{
    EXPR_ALLOC(CExprOfArrayDecl, expr, EC_ARRAY_DECL);
    expr->e_typeQualExpr = typeQualExpr;
    expr->e_lenExpr = lenExpr;
    if(typeQualExpr)
        EXPR_REF(typeQualExpr);
    if(lenExpr) {
        EXPR_REF(lenExpr);
        EXPR_PARENT(lenExpr) = (CExpr*)expr;
    }
    return expr;
}


/**
 * \brief
 * free CExprOfArrayDecl
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfArrayDecl(CExprOfArrayDecl *expr)
{
    if(expr == NULL)
        return;
    if(expr->e_typeQualExpr)
        freeExpr(expr->e_typeQualExpr);
    if(expr->e_lenExpr)
        freeExpr(expr->e_lenExpr);
}


/**
 * \brief
 * alloc CExprOfDirective
 *
 * @param type
 *      directive type
 * @param name
 *      pragma name
 * @param args
 *      pragma arguments
 * @return
 *      allocated node
 */
CExprOfDirective*
allocExprOfDirective(CDirectiveTypeEnum type, char *name, char *args)
{
    EXPR_ALLOC(CExprOfDirective, expr, EC_DIRECTIVE);
    expr->e_direcName = name;
    expr->e_direcType = type;
    expr->e_direcArgs = args;
    return expr;
}


/**
 * \brief
 * copy CExprOfDirective
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfDirective*
copyExprOfDirective(CExprOfDirective *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfDirective, dst, src);
    dst->e_direcName = ccol_strdup(src->e_direcName, MAX_NAME_SIZ);
    dst->e_direcArgs = ccol_strdup(src->e_direcArgs, MAX_NAME_SIZ);

    return dst;
}


/**
 * \brief
 * free CExprOfDirective
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfDirective(CExprOfDirective *expr)
{
    if(expr->e_direcArgs)
        free(expr->e_direcArgs);
    if(expr->e_direcName)
        free(expr->e_direcName);
}


/**
 * \brief
 * alloc CExprOfTypeDesc
 *
 * @return
 *      allocated node
 */
CExprOfTypeDesc*
allocExprOfTypeDesc()
{
    EXPR_ALLOC(CExprOfTypeDesc, expr, EC_TYPE_DESC);
    return expr;
}


/**
 * \brief
 * copy CExprOfTypeDesc
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfTypeDesc*
copyExprOfTypeDesc(CExprOfTypeDesc *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfTypeDesc, dst, src);
    EXPR_SET0(dst->e_typeExpr, copyExpr(src->e_typeExpr));
    EXPR_SET0(dst->e_paramExpr, copyExpr(src->e_paramExpr));
    EXPR_SET0(dst->e_bitLenExpr, copyExpr(src->e_bitLenExpr));
    EXPR_SET0(dst->e_len.eln_lenExpr, copyExpr(src->e_len.eln_lenExpr));
    EXPR_SET0(dst->e_len.eln_orgLenExpr, copyExpr(src->e_len.eln_orgLenExpr));
    dst->e_typeId = ccol_strdup(src->e_typeId, MAX_NAME_SIZ);

    return dst;
}


/**
 * \brief
 * shallow copy CExprOfTypeDesc
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
CExprOfTypeDesc*
duplicateExprOfTypeDesc(CExprOfTypeDesc *src)
{
    EXPR_ALLOC(CExprOfTypeDesc, dst, EC_TYPE_DESC);
    memcpy(dst, src, sizeof(CExprOfTypeDesc));
    innerDuplicateExprCommon((CExpr*)dst, (CExpr*)src);
    dst->e_isExprsType = 0;

    if(dst->e_typeExpr)
        EXPR_REF(dst->e_typeExpr);
    if(dst->e_paramExpr)
        EXPR_REF(dst->e_paramExpr);
    if(dst->e_bitLenExpr)
        EXPR_REF(dst->e_bitLenExpr);
    if(dst->e_len.eln_lenExpr)
        EXPR_REF(dst->e_len.eln_lenExpr);
    if(dst->e_len.eln_orgLenExpr)
        EXPR_REF(dst->e_len.eln_orgLenExpr);
    if(dst->e_typeId)
        dst->e_typeId = ccol_strdup(dst->e_typeId, MAX_NAME_SIZ);

    return dst;
}


/**
 * \brief
 * free CExprOfTypeDesc
 *
 * @param expr
 *      target node
 */
void
innerFreeExprOfTypeDesc(CExprOfTypeDesc *expr)
{
    if(expr->e_typeExpr)
        freeExpr(expr->e_typeExpr);
    if(expr->e_paramExpr)
        freeExpr(expr->e_paramExpr);
    if(expr->e_bitLenExpr)
        freeExpr(expr->e_bitLenExpr);
    if(expr->e_len.eln_lenExpr)
        freeExpr(expr->e_len.eln_lenExpr);
    if(expr->e_len.eln_orgLenExpr)
        freeExpr(expr->e_len.eln_orgLenExpr);
    if(expr->e_typeId)
        free(expr->e_typeId);

    //refType is only for reference
}


/**
 * \brief
 * alloc CExprOfErrorNode
 *
 * @param nearExpr
 *      error node
 * @return
 *      allocated node
 */
CExprOfErrorNode*
allocExprOfErrorNode(CExpr *nearExpr)
{
    EXPR_ALLOC(CExprOfErrorNode, expr, EC_ERROR_NODE);
    expr->e_nearExpr = nearExpr;
    EXPR_REF(nearExpr);
    return expr;
}


/**
 * \brief
 * copy CExprOfErrorNode
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfErrorNode*
copyExprOfErrorNode(CExprOfErrorNode *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfErrorNode, dst, src);
    if(dst->e_nearExpr)
        EXPR_REF(dst->e_nearExpr);

    return dst;
}


/**
 * \brief
 * free CExprOfErrorNode
 *
 * @param expr
 *      target node
 */
PRIVATE_STATIC void
innerFreeExprOfErrorNode(CExprOfErrorNode *expr)
{
    if(expr == NULL)
        return;
    if(expr->e_nearExpr)
        freeExpr(expr->e_nearExpr);
}


/**
 * \brief
 * alloc CExprOfNull
 *
 * @return
 *      allocated node
 */
CExprOfNull*
allocExprOfNull()
{
    EXPR_ALLOC(CExprOfNull, expr, EC_NULL_NODE);
    return expr;
}


/**
 * \brief
 * copy CExprOfNull
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfNull*
copyExprOfNull(CExprOfNull *src)
{
    assert(src);
    EXPR_ALLOC_COPY(CExprOfNull, dst, src);

    return dst;
}


#define innerFreeExprOfNull(expr)


/**
 * \brief
 * set child node
 *
 * @param plexpr
 *      pointer to child node field
 * @param rexpr
 *      child node
 * @return
 */
CExpr*
exprSet0(CExpr **plexpr, CExpr *rexpr)
{
    if(rexpr != NULL)
        EXPR_REF(rexpr);
    *plexpr = rexpr;

    return rexpr;
}


/**
 * \brief
 * set and replace child node
 *
 * @param plexpr
 *      pointer to child node field
 * @param rexpr
 *      child node
 * @return
 */
CExpr*
exprSet(CExpr **plexpr, CExpr *rexpr)
{
    CExpr *lexpr = *plexpr;
    if(lexpr == rexpr)
        return rexpr;

    if(lexpr != NULL) {
        EXPR_UNREF(lexpr);
        if(EXPR_C(lexpr)->e_refCount <= 0)
            freeExpr(lexpr);
    }

    if(rexpr != NULL)
        EXPR_REF(rexpr);
    *plexpr = rexpr;

    return rexpr;
}


/**
 * \brief
 * increment reference counter
 *
 * @param expr
 *      target node
 * @return
 *      reference count
 */
int
exprRef(CExpr *expr)
{
    if(expr)
        return ++(((CExprCommon*)(expr))->e_refCount);
    return 0;
}


/**
 * \brief
 * decrement reference counter
 *
 * @param expr
 *      target node
 * @return
 *      reference count
 */
int
exprUnref(CExpr *expr)
{
    if(expr == NULL)
        return 0;
    int ref = --(((CExprCommon*)(expr))->e_refCount);
#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("@CExpr-ur/%d:" ADDR_PRINT_FMT "@\n", ref, (uintptr_t)expr));
    if(EXPR_CODE(expr) == EC_TYPE_DESC) {
        DBGPRINT(("@^CExprOfTypeDesc:%s\n", s_CTypeDescKindEnumNames[EXPR_T(expr)->e_tdKind]));
    }
#endif
    return ref;
}


/**
 * \brief
 * set expression's type descriptor
 *
 * @param expr
 *      target node
 * @param td
 *      type descriptor
 */
void
exprSetExprsType(CExpr *expr, CExprOfTypeDesc *td)
{
    assertExprCode((CExpr*)td, EC_TYPE_DESC);
    if(expr) {
        assertExpr(expr, expr != (CExpr*)td);
        EXPRS_TYPE(expr) = td;
    }

    if(td->e_isExprsType == 0) {
        td->e_isExprsType = 1;
        EXPR_REF(td);
        CCOL_SL_CONS(&s_exprsTypeDescList, td);
    }
    
    if(td->e_isUsed == 0)
        td->e_isUsed = 1;
}


/**
 * \brief
 * free CExpr
 *
 * @param expr
 *      target node
 */
void
freeExpr(CExpr *expr)
{
    if(expr == NULL)
        return;

    int ref = EXPR_UNREF(expr);
    if(ref > 0) {
#if 0
        DBGPRINT(("@CExpr-ur/%d:" ADDR_PRINT_FMT "@\n", ref, (uintptr_t)expr));
        if(EXPR_CODE(expr) == EC_TYPE_DESC) {
            DBGPRINT(("@^CExprOfTypeDesc:%s\n", s_CTypeDescKindEnumNames[EXPR_T(expr)->e_tdKind]));
        }
#endif
        return;
    }

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfSymbol:
        innerFreeExprOfSymbol(EXPR_SYMBOL(expr));
        break;
    case STRUCT_CExprOfList:
        innerFreeExprOfList(EXPR_L(expr));
        break;
    case STRUCT_CExprOfNumberConst:
        innerFreeExprOfNumberConst(EXPR_NUMBERCONST(expr));
        break;
    case STRUCT_CExprOfCharConst:
        innerFreeExprOfCharConst(EXPR_CHARCONST(expr));
        break;
    case STRUCT_CExprOfStringConst:
        innerFreeExprOfStringConst(EXPR_STRINGCONST(expr));
        break;
    case STRUCT_CExprOfGeneralCode:
        innerFreeExprOfGeneralCode(EXPR_GENERALCODE(expr));
        break;
    case STRUCT_CExprOfUnaryNode:
        innerFreeExprOfUnaryNode(EXPR_U(expr));
        break;
    case STRUCT_CExprOfBinaryNode:
        innerFreeExprOfBinaryNode(EXPR_B(expr));
        break;
    case STRUCT_CExprOfArrayDecl:
        innerFreeExprOfArrayDecl(EXPR_ARRAYDECL(expr));
        break;
    case STRUCT_CExprOfDirective:
        innerFreeExprOfDirective(EXPR_DIRECTIVE(expr));
        break;
    case STRUCT_CExprOfTypeDesc:
        innerFreeExprOfTypeDesc(EXPR_T(expr));
        break;
    case STRUCT_CExprOfErrorNode:
        innerFreeExprOfErrorNode(EXPR_ERRORNODE(expr));
        break;
    case STRUCT_CExprOfNull:
        innerFreeExprOfNull(EXPR_NULL(expr));
        break;
    case STRUCT_CExpr_UNDEF:
    case STRUCT_CExpr_END:
        ABORT();
    }

    freeExpr(EXPR_C(expr)->e_gccAttrPre);
    freeExpr(EXPR_C(expr)->e_gccAttrPost);

    XFREE(expr);
}


/**
 * \brief
 * deep copy CExpr
 *
 * @param src
 *      source node
 * @return
 *      allocated node
 */
CExpr*
copyExpr(CExpr *src)
{
    if(src == NULL)
        return NULL;

    CExpr *dst = NULL;

    switch(EXPR_STRUCT(src)) {
    case STRUCT_CExprOfSymbol:
        dst = (CExpr*)copyExprOfSymbol(EXPR_SYMBOL(src));
        break;
    case STRUCT_CExprOfList:
        dst = (CExpr*)copyExprOfList(EXPR_L(src));
        break;
    case STRUCT_CExprOfNumberConst:
        dst = (CExpr*)copyExprOfNumberConst(EXPR_NUMBERCONST(src));
        break;
    case STRUCT_CExprOfCharConst:
        dst = (CExpr*)copyExprOfCharConst(EXPR_CHARCONST(src));
        break;
    case STRUCT_CExprOfStringConst:
        dst = (CExpr*)copyExprOfStringConst(EXPR_STRINGCONST(src));
        break;
    case STRUCT_CExprOfGeneralCode:
        dst = (CExpr*)copyExprOfGeneralCode(EXPR_GENERALCODE(src));
        break;
    case STRUCT_CExprOfUnaryNode:
        dst = (CExpr*)copyExprOfUnaryNode(EXPR_U(src));
        break;
    case STRUCT_CExprOfBinaryNode:
        dst = (CExpr*)copyExprOfBinaryNode(EXPR_B(src));
        break;
    case STRUCT_CExprOfDirective:
        dst = (CExpr*)copyExprOfDirective(EXPR_DIRECTIVE(src));
        break;
    case STRUCT_CExprOfTypeDesc:
        dst = (CExpr*)copyExprOfTypeDesc(EXPR_T(src));
        break;
    case STRUCT_CExprOfErrorNode:
        dst = (CExpr*)copyExprOfErrorNode(EXPR_ERRORNODE(src));
        break;
    case STRUCT_CExprOfNull:
        dst = (CExpr*)copyExprOfNull(EXPR_NULL(src));
        break;
    case STRUCT_CExprOfArrayDecl:
    case STRUCT_CExpr_UNDEF:
    case STRUCT_CExpr_END:
        ABORT();
    }

    return dst;
}


/**
 * \brief
 * alloc CSymbolTable
 *
 * @return
 *      allocated symbo table
 */
CSymbolTable*
allocSymbolTable()
{
    CSymbolTable *symTab = XALLOC(CSymbolTable);
    CCOL_HT_INIT(&symTab->stb_identGroup, CCOL_HT_STRING_KEYS);
    CCOL_HT_INIT(&symTab->stb_tagGroup,   CCOL_HT_STRING_KEYS);
    CCOL_HT_INIT(&symTab->stb_labelGroup, CCOL_HT_STRING_KEYS);

    if(s_isParsing)
        symTab->stb_lineNumInfo = s_lineNumInfo;

#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("@CSymbolTable:" ADDR_PRINT_FMT "@\n", (uintptr_t)symTab));
#endif

    return symTab;
}


/**
 * \brief
 * free CSymbolTable
 *
 * @param symTab
 *      target symbol table
 */
void
freeSymbolTable(CSymbolTable *symTab)
{
    if(symTab == NULL)
        return;

#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("@CSymbolTable-free:" ADDR_PRINT_FMT "@\n", (uintptr_t)symTab));
#endif
    CCOL_HT_DESTROY(&symTab->stb_identGroup);
    CCOL_HT_DESTROY(&symTab->stb_tagGroup);
    CCOL_HT_DESTROY(&symTab->stb_labelGroup);

    XFREE(symTab);
}


/**
 * \brief
 * join CExprOfList
 *
 * @param exprHead
 *      head node list
 * @param exprTail
 *      tail node list
 * @return
 *      head node list
 */
CExpr*
exprListJoin(CExpr *exprHead, CExpr *exprTail)
{
    assert(exprHead != NULL);
    assert(exprTail != NULL);
    assertExprStruct(exprHead, STRUCT_CExprOfList);

    CCOL_DList *exprHeadList = EXPR_DLIST(exprHead);

    if(EXPR_STRUCT(exprTail) == STRUCT_CExprOfList &&
        EXPR_CODE(exprHead) == EXPR_CODE(exprTail)) {

        /* nodes in exprTail move into exprHeadList */
        CCOL_DL_JOIN(exprHeadList, EXPR_DLIST(exprTail));
        CCOL_DListNode *ite;
        EXPR_FOREACH(ite, exprTail) {
            CExpr *node = EXPR_L_DATA(ite);
            EXPR_REF(node);
            EXPR_PARENT(node) = exprHead;
        }
        freeExpr(exprTail);
    } else {
        exprListAdd(exprHead, exprTail);
    }

    return exprHead;
}


/**
 * \brief
 * copy line number info
 *
 * @param to
 *      destination
 * @param from
 *      source
 */
void
exprCopyLineNum(CExpr *to, CExpr *from)
{
    EXPR_C(to)->e_lineNumInfo = EXPR_C(from)->e_lineNumInfo;
}


/**
 * \brief
 * add list node
 *
 * @param exprHead
 *      head node list
 * @param exprTail
 *      tail node
 * @return
 *      head node list
 */
CExpr*
exprListAdd(CExpr *exprHead, CExpr *exprTail)
{
    assert(exprHead != NULL);
    assertExprStruct(exprHead, STRUCT_CExprOfList);

    if(exprTail == NULL)
        exprTail = exprNull();

    EXPR_REF(exprTail);
    CCOL_DL_ADD(EXPR_DLIST(exprHead), exprTail);

    if(EXPR_L_SIZE(exprHead) == 1 && EXPR_ISNULL(exprTail) == 0) {
        exprCopyLineNum(exprHead, exprTail);
    }

    EXPR_PARENT(exprTail) = exprHead;

    return exprHead;
}


/**
 * \brief
 * cons list node
 *
 * @param exprHead
 *      head node
 * @param exprTail
 *      tail node list
 * @return
 *      tail node list
 */
CExpr*
exprListCons(CExpr *exprHead, CExpr *exprTail)
{
    assert(exprHead != NULL);
    assert(exprTail != NULL);
    assertExprStruct(exprTail, STRUCT_CExprOfList);

    EXPR_REF(exprHead);
    CCOL_DL_CONS(EXPR_DLIST(exprTail), exprHead);
    EXPR_PARENT(exprHead) = exprTail;
    return exprTail;
}


/**
 * \brief
 * remove head node from node list
 *
 * @param exprList
 *      target node list
 * @return
 *      removed node
 */
CExpr*
exprListRemoveHead(CExpr *exprList)
{
    CCOL_DListNode *head = exprListHead(exprList);
    CExpr *expr = EXPR_L_DATA(head);
    CCOL_DL_REMOVE_HEAD(EXPR_DLIST(exprList));
    EXPR_UNREF(expr);
    EXPR_PARENT(expr) = NULL;
    return expr;
}


/**
 * \brief
 * remove tail node from node list
 *
 * @param exprList
 *      target node list
 * @return
 *      removed node
 */
CExpr*
exprListRemoveTail(CExpr *exprList)
{
    CCOL_DListNode *tail = exprListTail(exprList);
    CExpr *expr = EXPR_L_DATA(tail);
    CCOL_DL_REMOVE_TAIL(EXPR_DLIST(exprList));
    EXPR_UNREF(expr);
    EXPR_PARENT(expr) = NULL;
    return expr;
}


/**
 * \brief
 * remove specified node from node list
 *
 * @param exprList
 *      target node List
 * @param ite
 *      node iterator to remove
 * @return
 *      removed node
 */
CExpr*
exprListRemove(CExpr *exprList, CCOL_DListNode *ite)
{
    CExpr *expr = (CExpr*)CCOL_DL_REMOVE(EXPR_DLIST(exprList), ite);
    EXPR_UNREF(expr);
    EXPR_PARENT(expr) = NULL;
    return expr;
}


/**
 * \brief
 * remove all nodes in node list
 *
 * @param exprList
 *      target node list
 * @return
 *      exprList
 */
CExpr*
exprListClear(CExpr *exprList)
{
    CCOL_DListNode *ite, *iten;
    EXPR_FOREACH_SAFE(ite, iten, exprList) {
        CExpr *expr = exprListRemove(exprList, ite);
        freeExpr(expr);
    }

    return exprList;
}



/**
 * \brief
 * get head node iterator in node list
 *
 * @param listExpr
 *      target node list
 * @return
 *      head node iterator
 */
CCOL_DListNode*
exprListHead(CExpr *listExpr)
{
    if(listExpr == NULL)
        return NULL;

    assertExprStruct(listExpr, STRUCT_CExprOfList);
    return CCOL_DL_HEAD(EXPR_DLIST(listExpr));
}


/**
 * \brief
 * get head node in node list
 *
 * @param listExpr
 *      target node list
 * @return
 *      head node
 */
CExpr*
exprListHeadData(CExpr *listExpr)
{
    CCOL_DListNode *nd = exprListHead(listExpr);
    if(nd == NULL)
        return NULL;
    return EXPR_L_DATA(nd);
}


/**
 * \brief
 * get tail node iterator in node list
 *
 * @param listExpr
 *      target node list
 * @return
 *      tail node iterator
 */
CCOL_DListNode*
exprListTail(CExpr *listExpr)
{
    if(listExpr == NULL)
        return NULL;

    assertExprStruct(listExpr, STRUCT_CExprOfList);
    return CCOL_DL_TAIL(EXPR_DLIST(listExpr));
}


/**
 * \brief
 * get tail node in node list
 *
 * @param listExpr
 *      target node list
 * @return
 *      tail node
 */
CExpr*
exprListTailData(CExpr *listExpr)
{
    CCOL_DListNode *nd = exprListTail(listExpr);
    if(nd == NULL)
        return NULL;
    return EXPR_L_DATA(nd);
}


/**
 * \brief
 * get next node iterator in node list
 *
 * @param listExpr
 *      target node list
 * @return
 *      next node iterator
 */
CCOL_DListNode*
exprListNext(CExpr *listExpr)
{
    if(listExpr == NULL)
        return NULL;

    assertExprStruct(listExpr, STRUCT_CExprOfList);
    return CCOL_DL_NEXT(exprListHead(listExpr));
}


/**
 * \brief
 * get next node in node list
 *
 * @param listExpr
 *      target node list
 * @return
 *      next node
 */
CExpr*
exprListNextData(CExpr *listExpr)
{
    CCOL_DListNode *nd = exprListNext(listExpr);
    if(nd == NULL)
        return NULL;
    return EXPR_L_DATA(nd);
}


/**
 * \brief
 * get next n th node iterator in node list
 *
 * @param listExpr
 *      target node list
 * @param n
 *      index of node
 * @return
 *      next n th node iterator
 */
CCOL_DListNode*
exprListNextN(CExpr *listExpr, unsigned int n)
{
    if(listExpr == NULL)
        return NULL;

    assertExprStruct(listExpr, STRUCT_CExprOfList);
    return CCOL_DL_NEXTN(exprListHead(listExpr), n);
}


/**
 * \brief
 * get next n th node in node list
 *
 * @param listExpr
 *      target node list
 * @param n
 *      index of node
 * @return
 *      next n th node
 */
CExpr*
exprListNextNData(CExpr *listExpr, unsigned int n)
{
    CCOL_DListNode *nd = exprListNextN(listExpr, n);
    if(nd == NULL)
        return NULL;
    return EXPR_L_DATA(nd);
}


/**
 * \brief
 * set gcc extension flag
 *
 * @param expr
 *      target node
 * @return
 *      expr
 */
CExpr*
exprSetExtension(CExpr *expr)
{
    assertYYLineno(expr != NULL);
    assertYYLineno(EXPR_C(expr)->e_gccExtension == 0);
    EXPR_C(expr)->e_gccExtension = 1;
    return expr;
}


/**
 * \brief
 * get symbol node in specified node
 *
 * @param expr
 *      target node
 * @return
 *      NULL or symbol node
 */
PRIVATE_STATIC CExpr*
getExprOfSymbol(CExpr *expr)
{
    if(EXPR_ISNULL(expr))
        return NULL;

    CCOL_DListNode *ite;

    switch(EXPR_CODE(expr)) {
    case EC_IDENT:
        return expr;

    case EC_LDECLARATOR:
        EXPR_FOREACH(ite, expr) {
            CExpr *tmpExpr = EXPR_L_DATA(ite);
            switch(EXPR_CODE(tmpExpr)) {
            case EC_IDENT:
                return tmpExpr;
            case EC_DECL_SPECS:
                /* skip '*' before identifier */
                if(EXPR_CODE(exprListHeadData(tmpExpr)) == EC_POINTER_DECL)
                    continue;
                return getExprOfSymbol(tmpExpr);
            case EC_LDECLARATOR:
                return getExprOfSymbol(tmpExpr);
            default:
                break;
            }
        }
        break;
 
    default:
        break;
    }

    return NULL;
}


/**
 * \brief
 * process 'typedef' node at parsing
 *
 * @param dataDefExprOrDecl
 *      EC_DATA_DEF or EC_DECL node
 */
void
procTypeDefInParser(CExpr *dataDefExprOrDecl)
{
    if(dataDefExprOrDecl == NULL)
        return;

    switch(EXPR_CODE(dataDefExprOrDecl)) {
    case EC_DATA_DEF:
    case EC_DECL:
        break;
    default:
        return;
    }

    CExpr *declSpecs = EXPR_B(dataDefExprOrDecl)->e_nodes[0];

    if(declSpecs == NULL || EXPR_CODE(declSpecs) != EC_DECL_SPECS)
        return;

    CExpr *initDecls = EXPR_B(dataDefExprOrDecl)->e_nodes[1];

    if(initDecls == NULL || EXPR_CODE(initDecls) != EC_INIT_DECLS)
        return;

    CExpr *scspec = exprListHeadData(declSpecs);

    if(scspec ==NULL || EXPR_CODE(scspec) != EC_SCSPEC)
        return;

    CExprOfGeneralCode *typeDef = EXPR_GENERALCODE(scspec);

    if(typeDef->e_code != SS_TYPEDEF)
        return;

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, initDecls) {

        CExpr *initDecl = EXPR_L_DATA(ite);

        if(EXPR_CODE(initDecl) != EC_INIT_DECL)
            continue;

        CExpr *declarator = exprListHeadData(initDecl);

        if(declarator == NULL || EXPR_CODE(declarator) != EC_LDECLARATOR)
            continue;

        CExpr *ident = getExprOfSymbol(declarator);

        if(ident != NULL) {
            addSymbolInParser(ident, ST_TYPE);
        }
    }
}


/**
 * \brief
 * replace child node of CExprOfUnaryNode, CExprOfBinaryNode, CExprOfTypeDesc,
 * CExprOfTypeDesc, CExprOfSymbol
 *
 * @param parent
 *      target parent node
 * @param oldChild
 *      old child node to remove
 * @param newChild
 *      new child node to add
 */
void
exprReplace(CExpr *parent, CExpr *oldChild, CExpr *newChild)
{
    assert(parent != NULL);
    assertExpr(parent, oldChild != NULL);
    assertExpr(parent, newChild != NULL);

    if(oldChild == newChild)
        return;

    EXPR_REF(newChild);

    switch(EXPR_STRUCT(parent)) {

    case STRUCT_CExprOfUnaryNode:
        assertExpr(parent, EXPR_U(parent)->e_node == oldChild);
        EXPR_U(parent)->e_node = newChild;
        freeExpr(oldChild);
        break;

    case STRUCT_CExprOfBinaryNode:
        for(int i = 0; i < 2; ++i) {
            CExpr * node = EXPR_B(parent)->e_nodes[i];
            if(node == oldChild) {
                EXPR_B(parent)->e_nodes[i] = newChild;
                freeExpr(oldChild);
                goto end;
            }
        }
        ABORT();

    case STRUCT_CExprOfList: {
            CCOL_DListNode *ite;
            EXPR_FOREACH(ite, parent) {
                CExpr *node = EXPR_L_DATA(ite);
                if(node == oldChild) {
                    CCOL_DL_SET_DATA(ite, newChild);
                    freeExpr(oldChild);
                    goto end;
                }
            }
        }
        ABORT();

    case STRUCT_CExprOfSymbol:
        assertExpr(parent, EXPR_SYMBOL(parent)->e_valueExpr == oldChild);
        EXPR_SYMBOL(parent)->e_valueExpr = newChild;
        freeExpr(oldChild);
        break;

    case STRUCT_CExprOfTypeDesc: {
            CExprOfTypeDesc *td = EXPR_T(parent);
            if(td->e_typeExpr == oldChild)
                td->e_typeExpr = newChild;
            else if(td->e_len.eln_lenExpr == oldChild)
                td->e_len.eln_lenExpr = newChild;
            else if(td->e_bitLenExpr == oldChild)
                td->e_bitLenExpr = newChild;
            else if(td->e_paramExpr == oldChild)
                td->e_paramExpr = newChild;
            else
                ABORT();
            freeExpr(oldChild);
        }
        break;

    default:
        ABORT();
    }

end:
    EXPR_UNREF(newChild);
    EXPR_PARENT(newChild) = parent;
}


/**
 * \brief
 * remove node from parent of CExprOfUnaryNode, CExprOfBinaryNode, CExprOfList
 *
 * @param parent
 *      target parent node
 * @param child
 *      child node to remove
 */
void
exprRemove(CExpr *parent, CExpr *child)
{
    assert(parent);
    assertExpr(parent, child);
    assert(parent != child);

    switch(EXPR_STRUCT(parent)) {

    case STRUCT_CExprOfUnaryNode: {
            assertExpr(parent, EXPR_U(parent)->e_node == child);
            EXPR_U(parent)->e_node = NULL;
            freeExpr(child);
        }
        break;

    case STRUCT_CExprOfBinaryNode:
        for(int i = 0; i < 2; ++i) {
            CExpr * node = EXPR_B(parent)->e_nodes[i];
            if(node == child) {
                EXPR_B(parent)->e_nodes[i] = NULL;
                freeExpr(child);
                return;
            }
        }
        ABORT();

    case STRUCT_CExprOfList: {
            CCOL_DListNode *ite;
            EXPR_FOREACH(ite, parent) {
                CExpr *node = EXPR_L_DATA(ite);
                if(node == child) {
                    exprListRemove(parent, ite);
                    freeExpr(node);
                    return;
                }
            }
        }
        ABORT();

    default:
        ABORT();
    }
}


/**
 * \brief
 *
 *
 * @param expr
 *
 * @return
 */
CExpr*
allocPointerDecl(CExpr *typeQuals)
{
    CExpr *declSpecs = typeQuals;

    if(EXPR_ISNULL(declSpecs)) {
        freeExpr(declSpecs);
        declSpecs = exprList(EC_DECL_SPECS);
        if(typeQuals != NULL)
            exprCopyAttr(declSpecs, typeQuals);
        freeExpr(typeQuals);
    }

    declSpecs = exprListCons((CExpr*)allocExprOfGeneralCode(EC_POINTER_DECL, 1), declSpecs);

    return declSpecs;
}


/**
 * \brief
 * proceed iterator till points at non-EC_POINTER_DECL node
 *
 * @param declrChildNode
 *      declarator's child node
 * @param declarator
 *      declarator node
 * @param numPtrDecl
 *      number of skipped pointer decl
 * @return
 *      node iterator
 */
CCOL_DListNode*
skipPointerDeclNode(CCOL_DListNode *declrChildNode, CExpr *declarator, int *numPtrDecl)
{
    assertYYLineno(declrChildNode);

    if(numPtrDecl)
        *numPtrDecl = 0;

    CCOL_DListNode *ite = declrChildNode;

    EXPR_FOREACH_FROM(ite, declarator) {
        CExpr *next = EXPR_L_DATA(ite); 
        if(EXPR_CODE(next) != EC_DECL_SPECS)
            return ite;

        if(EXPR_CODE(next) == EC_DECL_SPECS) {
            if(EXPR_L_SIZE(next) == 0 || EXPR_CODE(exprListHeadData(next)) != EC_POINTER_DECL)
                return ite;
            if(numPtrDecl != NULL)
                ++(*numPtrDecl);
        }
    }

    /* not found a node which does not have POINTER_DECL */

    return NULL;
}


/**
 * \brief
 * add function name and parameters' name to Symbol Table
 * in func_def-declarator.
 *
 * @param declarator
 *
 * @return
 */
void
addFuncDefDeclaratorSymbolInParser(CExpr *declarator)
{
    assertExprCode(declarator, EC_LDECLARATOR);
    CCOL_DListNode *head = exprListHead(declarator);

    if(head == NULL)
        return;

    CExprOfSymbol *funcName = EXPR_SYMBOL(getExprOfSymbol(EXPR_L_DATA(head)));

    if(funcName == NULL)
        return;

    addSymbolForFuncInParser(funcName, declarator);

    if(EXPR_L_SIZE(declarator) < 2)
        return;

    CCOL_DListNode *next = CCOL_DL_NEXT(head);
    CExpr *params = (CExpr*)CCOL_DL_DATA(next);

    if(EXPR_ISNULL(params) || EXPR_CODE(params) != EC_PARAMS) {
        next = CCOL_DL_NEXT(next);
        params = (CExpr*)CCOL_DL_DATA(next);
        if(EXPR_ISNULL(params) || EXPR_CODE(params) != EC_PARAMS)
            return;
    }

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, params) {
        CExpr *param = EXPR_L_DATA(ite);

        if(EXPR_CODE(param) == EC_ELLIPSIS)
            continue;

        if(EXPR_CODE(param) != EC_PARAM) {
            addError(param, CERR_001);
            continue;
        }

        CExpr *paramDeclr = EXPR_B(param)->e_nodes[1];

        if(EXPR_ISNULL(paramDeclr))
            continue;

        //EC_IDENT means tagged type paramter
        //which name is overridden by type name
        //ex)
        //  typedef struct s { int a; } s;
        //  void f(struct s s)
        if(EXPR_CODE(paramDeclr) == EC_IDENT) {
        } else if(EXPR_CODE(paramDeclr) != EC_LDECLARATOR) {
            addError(paramDeclr, CERR_104);
            continue;
        }

        CExpr *paramName = getExprOfSymbol(paramDeclr);

        if(paramName) {
            addSymbolInParser(paramName, ST_PARAM);
        }
    }
}


/**
 * \brief
 * add symbol in EC_INIT_DECL node at parsing
 *
 * @param initDecl
 *      EC_INIT_DECL node
 */
void
addInitDeclSymbolInParser(CExpr *initDecl)
{
    assertExprCode(initDecl, EC_INIT_DECL);    

    CExpr *declr = exprListHeadData(initDecl);
    CExpr *varName = getExprOfSymbol(declr);

    if(varName == NULL)
        return;

    addSymbolInParser(varName, ST_VAR);
}


/**
 * \brief
 * get tag symbol in struct/union/enum
 *
 * @param td
 *      type descriptor of TD_STRUCT/TD_UNION/TD_ENUM
 * @return
 *      tag symbol
 */
CExprOfSymbol*
getTagSymbol(CExprOfTypeDesc *td)
{
    return (CExprOfSymbol*)exprListNextNData(td->e_typeExpr, EXPR_L_TAG_INDEX);
}


/**
 * \brief
 * get EC_MEMBER_DECLS node in struct/union
 *
 * @param td
 *      type descriptor of TD_STRUCT/TD_UNION
 * @return
 *      EC_MEMBER_DECLS node
 */
CExpr*
getMemberDeclsExpr(CExprOfTypeDesc *td)
{
    return exprListNextNData(td->e_typeExpr, EXPR_L_MEMBER_DECLS_INDEX);
}


/**
 * \brief
 * judge expr is related to symbol table and
 * symbol table has symbols.
 *
 * @param expr
 *      node which has block scope
 * @return
 *      0:no, 1:yes
 */
int
hasSymbols(CExpr *expr)
{
    CSymbolTable *symTab = EXPR_C(expr)->e_symTab;
    if(symTab == NULL)
        return 0;

    if(CCOL_HT_SIZE(&symTab->stb_identGroup) > 0)
        return 1;
    if(CCOL_HT_SIZE(&symTab->stb_tagGroup) > 0)
        return 1;
    if(CCOL_HT_SIZE(&symTab->stb_labelGroup) > 0)
        return 1;
    return 0;
}


/**
 * \brief
 * judge expr has EC_DECL or EC_DATA_DEF
 *
 * @param expr 
 *      target node
 * @return
 *      0:no, 1:yes
 */
int
hasDeclarationsCurLevel(CExpr *expr)
{
    if(expr == NULL)
        return 0;

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, expr) {
        CExpr *def = EXPR_L_DATA(ite);

        if(EXPR_ISNULL(def))
            continue;

        switch(EXPR_CODE(def)) {
        case EC_DATA_DEF:
        case EC_DECL:
            return 1;
            break;
        default:
            break;
        }
    }

    return 0;
}


/**
 * \brief
 * alloc subarray dimension
 *
 * @param el
 *      lower bound
 * @param eu
 *      upper bound
 * @param es
 *      step
 * @return
 *      allocated node
 */
CExpr*
exprSubArrayDimension(CExpr *el, CExpr *eu, CExpr *es)
{
    if(EXPR_ISNULL(el)) {
        freeExpr(el);
        el = (CExpr*)allocExprOfNumberConst2(0, BT_INT);
    }

    if(EXPR_ISNULL(es)) {
        freeExpr(es);
        es = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
    }

    return exprList3(EC_ARRAY_DIMENSION, el, eu, es);
}


/**
 * \brief
 * judege expr is subarray-ref
 *
 * @param expr
 *      target node
 * @return
 *      0:no, 1:yes
 */
int
isSubArrayRef(CExpr *expr)
{
    if(EXPR_CODE(expr) != EC_ARRAY_REF)
        return 0;

    CExpr *dim = EXPR_B(expr)->e_nodes[1];

    if(EXPR_CODE(dim) != EC_ARRAY_DIMENSION)
        return 0;

    if(EXPR_L_SIZE(dim) <= 1)
        return 0;

    return 1;
}


int
isSubArrayRef2(CExpr *expr)
{
  if (EXPR_CODE(expr) != EC_ARRAY_REF)
    return 0;

  CExpr *aryExpr = EXPR_B(expr)->e_nodes[0];
  CExpr *dim = EXPR_B(expr)->e_nodes[1];

  if (EXPR_CODE(dim) != EC_ARRAY_DIMENSION)
    return 0;

  if (EXPR_L_SIZE(dim) <= 1)
    return isSubArrayRef2(aryExpr);

  return 1;
}


PRIVATE_STATIC CExpr*
exprCoarrayRef0(CExpr *prim, CExpr *dims, int addPtrRef)
{
    switch(EXPR_CODE(prim)) {
    case EC_IDENT: {
            if(addPtrRef == 0)
                EXPR_UNREF(prim);
            if(addPtrRef && EXPR_PARENT(prim))
                addPtrRef = (EXPR_CODE(EXPR_PARENT(prim)) != EC_ARRAY_REF);
            CExpr *coExpr = exprBinary(EC_XMP_COARRAY_REF, prim, dims);
            if(addPtrRef) {
                CExpr *ptrRef = exprUnary(EC_POINTER_REF, coExpr);
                if(EXPR_PARENT(prim) == NULL)
                    EXPR_REF(ptrRef);
                return ptrRef;
            } else {
                return coExpr;
            }
        }
      /*    case EC_ARRAY_REF: {
            CExpr *aryExpr = EXPR_B(prim)->e_nodes[0];
            CExpr *coExpr = exprCoarrayRef0(aryExpr, dims, 0);
            EXPR_B(prim)->e_nodes[0] = coExpr;
            return prim;
	    }*/
    case EC_ARRAY_REF:
    case EC_MEMBER_REF: {
      //EXPR_UNREF(prim);
      CExpr *coExpr = exprBinary(EC_XMP_COARRAY_REF, prim, dims);
      return coExpr;
    }
    default:
        addError(prim, CERR_105);
        return prim;
    }
}


/**
 * \brief
 * alloc EC_XMP_COARRAY_REF node
 *
 * @param prim
 *      var node
 * @param dims
 *      coarray dimension node
 * @return
 *      allocated node
 */
CExpr*
exprCoarrayRef(CExpr *prim, CExpr *dims)
{
    return exprCoarrayRef0(prim, dims, 0);
}


/**
 * \brief
 * convert "filename to file ID table" to "file ID to filename table"
 */
void
convertFileIdToNameTab()
{
    CCOL_HashEntry *he;
    CCOL_HashSearch hs;

    CCOL_HT_FOREACH(he, hs, &s_fileIdTab) {
        char *key = CCOL_HT_KEY(&s_fileIdTab, he);
        CFileIdEntry *fie = (CFileIdEntry*)CCOL_HT_DATA(he);
        uintptr_t id = (uintptr_t)fie->fie_id;
        CCOL_HT_PUT_WORD(&s_fileIdToNameTab, id,
            ccol_strdup(key, MAX_NAME_SIZ));
    }

    freeFileIdTab();
}


/**
 * \brief
 * get filename by specified file ID
 *
 * @param fileId
 *      file ID
 * @return
 *      NULL or filename
 */
const char*
getFileNameByFileId(int fileId)
{
    CCOL_HashEntry *he = CCOL_HT_FIND_WORD(&s_fileIdToNameTab, (uintptr_t)fileId);
    return he ? (const char*)CCOL_HT_DATA(he) : NULL;
}


/**
 * \brief
 * judge expr has children nodes
 *
 * @param expr
 *      taget node
 * @return
 *      0:no, 1:yes
 */
int
hasChildren(CExpr *expr)
{
    if(expr == NULL)
        return 0;

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfUnaryNode:
        return EXPR_ISNULL(EXPR_U(expr)->e_node) == 0;
    case STRUCT_CExprOfBinaryNode:
        for(int i = 0; i < 2; ++i)
            if(EXPR_ISNULL(EXPR_B(expr)->e_nodes[i]) == 0)
                return 1;
        break;
    case STRUCT_CExprOfList: {
            CCOL_DListNode *ite;
            EXPR_FOREACH(ite, expr)
                if(EXPR_ISNULL(EXPR_L_DATA(ite)) == 0)
                    return 1;
        }
        break;
    default:
        break;
    }

    return 0;
}


/**
 * \brief
 * judge expr is statement node exclude label statement
 *
 * @param expr
 *      taget node
 * @return
 *      0:no, 1:yes
 */
int
isStatement(CExpr *expr)
{
    switch(EXPR_CODE(expr)) {
    case EC_STMTS_AND_DECLS:
    case EC_COMP_STMT:
    case EC_EXPR_STMT:
    case EC_IF_STMT:
    case EC_SWITCH_STMT:
    case EC_FOR_STMT:
    case EC_DO_STMT:
    case EC_WHILE_STMT:
    case EC_GOTO_STMT:
    case EC_BREAK_STMT:
    case EC_CONTINUE_STMT:
    case EC_RETURN_STMT:
    case EC_GCC_ASM_STMT:
        return 1;
    default:
        return 0;
    }
}


/**
 * \brief
 * judge expr is statement node or external definition/declaration node
 *
 * @param expr
 *      taget node
 * @return
 *      0:no, 1:yes
 */
int
isStatementOrLabelOrDeclOrDef(CExpr *expr)
{
    switch(EXPR_CODE(expr)) {
    case EC_STMTS_AND_DECLS:
    case EC_COMP_STMT:
    case EC_EXPR_STMT:
    case EC_IF_STMT:
    case EC_SWITCH_STMT:
    case EC_FOR_STMT:
    case EC_DO_STMT:
    case EC_WHILE_STMT:
    case EC_GOTO_STMT:
    case EC_BREAK_STMT:
    case EC_CONTINUE_STMT:
    case EC_RETURN_STMT:
    case EC_GCC_ASM_STMT:
    case EC_LABEL:
    case EC_CASE_LABEL:
    case EC_DEFAULT_LABEL:
    case EC_EXT_DEFS:
    case EC_DATA_DEF:
    case EC_FUNC_DEF:
    case EC_DECL:
    case EC_DECLARATOR:
    case EC_INIT:
    case EC_INIT_DECL:
    case EC_MEMBER_DECL:
    case EC_MEMBER_DECLS:
    case EC_MEMBER_DECLARATOR:
        return 1;
    default:
        return 0;
    }
}


/**
 * \brief
 * set parent node to children nodes of CExprOfUnaryNode, CExprOfBinaryNode, 
 * CExprOfList, CExprOfTypeDesc, CExprOfSymbol
 *
 * @param expr
 *      target node
 * @param parent
 *      new parent node
 */
void
setExprParent(CExpr *expr, CExpr *parent)
{
    if(expr == NULL)
        return;
    EXPR_PARENT(expr) = parent;

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfUnaryNode:
        setExprParent(EXPR_U(expr)->e_node, expr);
        break;
    case STRUCT_CExprOfBinaryNode:
        for(int i = 0; i < 2; ++i)
            setExprParent(EXPR_B(expr)->e_nodes[i], expr);
        break;
    case STRUCT_CExprOfList: {
            CCOL_DListNode *ite;
            EXPR_FOREACH(ite, expr)
                setExprParent(EXPR_L_DATA(ite), expr);
        }
        break;
    case STRUCT_CExprOfSymbol:
        if(EXPR_SYMBOL(expr)->e_valueExpr)
            setExprParent(EXPR_SYMBOL(expr)->e_valueExpr, expr);
        break;
    case STRUCT_CExprOfTypeDesc: {
            CExprOfTypeDesc *td = EXPR_T(expr);
            switch(td->e_tdKind) {
            case TD_STRUCT:
            case TD_UNION:
            case TD_ENUM:
                if(td->e_typeExpr)
                    setExprParent(td->e_typeExpr, expr);
                break;
            case TD_FUNC:
                if(td->e_paramExpr)
                    setExprParent(td->e_paramExpr, expr);
                break;
            case TD_ARRAY:
                if(td->e_len.eln_lenExpr)
                    setExprParent(td->e_len.eln_lenExpr, expr);
            default:
                if(td->e_bitLenExpr)
                    setExprParent(td->e_bitLenExpr, expr);
                break;
            }
        }
        break;
    default:
        break;
    }
}


/**
 * \brief
 * get last child EC_EXPR_STMT node
 *
 * @param expr
 *      target node
 * @return
 *      NULL or EC_EXPR_STMT node
 */
CExpr*
getLastExprStmt(CExpr *stmts)
{
    CCOL_DListNode *ite;
    EXPR_FOREACH_REVERSE(ite, stmts) {
        CExpr *expr = EXPR_L_DATA(ite);
        if(EXPR_ISNULL(expr))
            continue;
        if(EXPR_CODE(expr) == EC_EXPR_STMT)
            return expr;
    }

    return NULL;
}


/**
 * \brief
 * judge expr is child node of ec node
 *
 * @param expr
 *      target node
 * @param ec
 *      expression code
 * @param parentExpr
 *      parent node of expr
 * @param outParentExpr
 *      if found ec node, this will be set to ec node's parent node
 * @return
 *      0:not found, 1:found
 */
int
isExprCodeChildOf(CExpr *expr, CExprCodeEnum ec, CExpr *parentExpr,
    CExpr **outParentExpr)
{
    assertExpr(expr, parentExpr);
    if(outParentExpr)
        *outParentExpr = parentExpr;
    CExprCodeEnum pec = EXPR_CODE(parentExpr);
    if(pec == ec)
        return 1;

    switch(pec) {
    case EC_EXPRS:
    case EC_TYPENAME:
    case EC_BRACED_EXPR:
    case EC_MEMBER_REF:
    case EC_POINTS_AT:
    case EC_ARRAY_REF:
        // do not add other operators code for case label check
        return isExprCodeChildOf(parentExpr, ec, EXPR_PARENT(parentExpr),
            outParentExpr);
    default:
        break;
    }
    return 0;
}


/**
 * \brief
 * judge expr is child node of ec node
 *
 * @param expr
 *      target node
 * @param ec
 *      expression code
 * @param parentExpr
 *      parent node of expr
 * @param outParentExpr
 *      if found ec node, this will be set to ec node's parent node
 * @return
 *      0:not found, 1:found
 */
int
isExprCodeChildStmtOf(CExpr *expr, CExprCodeEnum ec, CExpr *parentExpr,
    CExpr **outParentExpr)
{
    assertExpr(expr, parentExpr);
    if(outParentExpr)
        *outParentExpr = parentExpr;
    CExprCodeEnum pec = EXPR_CODE(parentExpr);
    if(pec == ec)
        return 1;

    switch(pec) {
    case EC_EXT_DEFS:
    case EC_FUNC_DEF:
    case EC_GCC_COMP_STMT_EXPR:
        break;
    default:
        return isExprCodeChildStmtOf(parentExpr, ec, EXPR_PARENT(parentExpr),
            outParentExpr);
    }
    return 0;
}


/**
 * \brief
 * get function call node in expr children
 *
 * @param expr
 *      target node
 * @return
 *      NULL or EC_FUNCTION_CALL node
 */
CExpr*
getFuncCall(CExpr *expr)
{
    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr) {
        CExpr *e = getFuncCallAbsolutelyCalling(ite.node);
        if(e)
            return e;
    }

    return NULL;
}


/**
 * \brief
 * get function call node in expr
 *
 * @param expr
 *      target node
 * @return
 *      NULL or EC_FUNCTION_CALL node
 */
CExpr*
getFuncCallAbsolutelyCalling(CExpr *expr)
{
    if(expr == NULL)
        return NULL;

    switch(EXPR_CODE(expr)) {
    case EC_CONDEXPR:
        return getFuncCallAbsolutelyCalling(exprListHeadData(expr));
    case EC_LOG_AND:
    case EC_LOG_OR:
        return getFuncCallAbsolutelyCalling(EXPR_B(expr)->e_nodes[0]);
    case EC_GCC_COMP_STMT_EXPR:
        return NULL;
    case EC_FUNCTION_CALL:
        return expr;
    default:
        break;
    }

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr) {
        CExpr *e = getFuncCallAbsolutelyCalling(ite.node);
        if(e)
            return e;
    }

    return NULL;
}


/**
 * \brief
 * set new symbol order number
 *
 * @param sym
 *      target symbol
 */
void
reorderSymbol(CExprOfSymbol *sym)
{
    sym->e_putOrder = getCurrentSymbolTable()->stb_putCount++;
}


/**
 * \brief
 * judge e can be logical expression
 *
 * @param e
 *      target node
 * @return
 *      0:no, 1:yes
 */
int
isLogicalExpr(CExpr *e)
{
    assert(e);
    CExprOfTypeDesc *td = (EXPR_CODE(e) == EC_TYPE_DESC) ?
        EXPR_T(e) : resolveType(e);
    if(td == NULL)
        return 0;

    td = getRefType(td);

    switch(td->e_tdKind) {
    case TD_BASICTYPE:
    case TD_POINTER:
    case TD_ARRAY:
    case TD_ENUM:
        return 1;
    default:
        return 0;
    }
}


/**
 * \brief
 * jugde expr is coarray assign node
 *
 * @param expr
 *      target node
 * @return
 *      0:no, 1:yes
 */
int isCoArrayAssign(CExpr *expr)
{
    switch(EXPR_CODE(expr)) {
    case EC_ASSIGN:
    case EC_ASSIGN_PLUS:
    case EC_ASSIGN_MINUS:
    case EC_ASSIGN_MUL:
    case EC_ASSIGN_DIV:
    case EC_ASSIGN_MOD:
    case EC_ASSIGN_LSHIFT:
    case EC_ASSIGN_RSHIFT:
    case EC_ASSIGN_BIT_AND:
    case EC_ASSIGN_BIT_OR:
    case EC_ASSIGN_BIT_XOR:
        break;
    default:
        return 0;
    }

    CExpr *expr1 = EXPR_B(expr)->e_nodes[0];
    CExprCodeEnum ec;

    while((ec = EXPR_CODE(expr1)) == EC_ARRAY_REF ||
        ec == EC_POINTS_AT || ec == EC_ADDR_OF || ec == EC_POINTER_REF) {
        expr1 = (EXPR_STRUCT(expr1) == STRUCT_CExprOfBinaryNode) ?
            EXPR_B(expr1)->e_nodes[0] : EXPR_U(expr1)->e_node;
    }

    return (ec == EC_XMP_COARRAY_REF);
}


/**
 * \brief
 * print errors and warnings.
 *
 * @param fp
 *      output file pointer
 */
void 
printErrors(FILE *fp)
{
    CCOL_SListNode *nd;
    CCOL_SL_FOREACH(nd, &s_errorList) {
        CError *pe = (CError*)CCOL_SL_DATA(nd);

        if(pe->pe_errorKind == EK_WARN && isWarnableId(pe->pe_msg) == 0)
            continue;

        CLineNumInfo *li = &pe->pe_lineNumInfo;
        const char *file;
        int lineNum;

        if(s_rawlineNo) {
            file = s_inFile ? s_inFile : "<stdin>";
            lineNum = li->ln_rawLineNum;
        } else {
            file = getFileNameByFileId(li->ln_fileId);
            lineNum = li->ln_lineNum;
        }

        if(lineNum > 0)
            fprintf(fp, "%s:%d: ", file, lineNum);

        switch(pe->pe_errorKind) {
        case EK_WARN:
            fprintf(fp, "warn:");
            break;
        case EK_ERROR:
            fprintf(fp, "error:");
            break;
        case EK_FATAL:
            fprintf(fp, "fatal:");
            break;
        }

        fprintf(fp, "%s\n", pe->pe_msg);
    }
}


