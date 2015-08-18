/**
 * \file c-convert.c
 * implementations for conversion AST before output XcodeML.
 */

#include "c-comp.h"
#include "c-option.h"

//! sequence number for temporary variable
PRIVATE_STATIC int  s_tmpVarSeq = 1;
//! sequence number for gcc __label___
PRIVATE_STATIC int  s_gccLabelSeq = 1;
//! conversion count for c99 style declaration block
PRIVATE_STATIC int  s_numConvsC99StyleDeclBlock = 0;
//! conversion count for declaration in for statement init block
PRIVATE_STATIC int  s_numConvsDeclInForStmt = 0;
//! conversion count for x?:y
PRIVATE_STATIC int  s_numConvsCond2ndExprNull = 0;
//! conversion count for function call in initializer
PRIVATE_STATIC int  s_numConvsFuncCallInInitializer = 0;
//! conversion count for function call which returns composite type
PRIVATE_STATIC int  s_numConvsFuncCallReturnComposType = 0;
//! conversion count for member-ref to points-at
PRIVATE_STATIC int  s_numConvsMemRef = 0;
//! conversion count for addr of condition expression
PRIVATE_STATIC int  s_numConvsAddrOfCondExpr = 0;
//! conversion count for &*
PRIVATE_STATIC int  s_numConvsAddrOfPtrRef = 0;
//! conversion count for *&
PRIVATE_STATIC int  s_numConvsPtrRefAddrOf = 0;
//! conversion count for gcc __label__
PRIVATE_STATIC int  s_numConvsGccLabel = 0;
//! conversion count for addr of compound statement expression
PRIVATE_STATIC int  s_numConvsGccCompExprAddr = 0;
//! conversion count for anonymous member access
int                 s_numConvsAnonMemberAccess = 0;
//! conversion count for pragma pack
int                 s_numConvsPragmaPack = 0;

//! conversion result
enum {
    CONVS_STAY,
    CONVS_REPLACE,
    CONVS_DELETE,
    CONVS_RESET
};


//! temporary variable kinds
typedef enum {
    TVK_CONDEXPR,
    TVK_GCC_COMP_STMT_EXPR,
    TVK_FUNCTION_CALL,
    TVK_INITIALIZER,
    TVK_END
} CTempVarKind;


//! temporary variable suffixes
PRIVATE_STATIC const char *s_varSuffixes[TVK_END] = {
    "cond",
    "compexpr",
    "call",
    "init",
};


/**
 * \brief
 * get parent node of EC_COMP_STMT/EC_EXT_DEFS and
 * skip iterator to expr
 *
 * @param expr
 *      target node
 * @param[out] ite
 *      iterator which points at expr
 * @return
 *      parent node
 */
PRIVATE_STATIC CExpr*
getParentBlockStmt(CExpr *expr, CCOL_DListNode **ite)
{
    assert(expr);
    CExpr *parent = EXPR_PARENT(expr);
    if(parent == NULL)
        ABORT();
    assertExpr(expr, parent);

    if(EXPR_CODE(parent) == EC_COMP_STMT ||
        EXPR_CODE(parent) == EC_EXT_DEFS) {
        if(ite) {
            EXPR_FOREACH(*ite, parent) {
                if(EXPR_L_DATA(*ite) == expr)
                    break;
            }
        }
        return parent;
    }

    return getParentBlockStmt(parent, ite);
}


/**
 * \brief
 * add generated node near cur node
 *
 * @param gen
 *      generated node
 * @param cur
 *      current position node
 * @return
 *      parent node of generated node
 */
PRIVATE_STATIC CExpr*
addGeneratedStmt(CExpr *gen, CExpr *cur)
{
    CCOL_DListNode *ite;
    CExpr *stmts = getParentBlockStmt(cur, &ite);
    assertExpr(cur, stmts);
    CCOL_DL_INSERT_PREV(EXPR_DLIST(stmts), gen, ite);

    EXPR_REF(gen);
    EXPR_PARENT(gen) = stmts;
    EXPR_C(gen)->e_isGenerated = 1;
    return stmts;
}


/**
 * \brief
 * add declarations of temporary variable
 *
 * @param cur
 *      current position node
 * @param td
 *      variable type descriptor
 * @param tvk
 *      temporary variable kind
 * @return
 *      declared symbol
 */
PRIVATE_STATIC CExprOfSymbol*
addTempVarDecl(CExpr *cur, CExprOfTypeDesc *td, CTempVarKind tvk)
{
    char buf[128];
    sprintf(buf, "%s%d_%s", s_tmpVarPrefix, s_tmpVarSeq++, s_varSuffixes[tvk]);
    CExprOfSymbol *sym = allocExprOfSymbol(EC_IDENT, ccol_strdup(buf, 128));
    exprCopyLineNum((CExpr*)sym, cur);
    sym->e_symType = ST_VAR;

    if(td->e_tq.etq_isConst ||
        td->e_tq.etq_isRestrict ||
        td->e_sc.esc_isExtern ||
        td->e_sc.esc_isStatic ||
        td->e_isTypeDef) {

        td = duplicateExprOfTypeDesc(td);
        td->e_tq.etq_isConst = 0;
        td->e_tq.etq_isRestrict = 0;
        td->e_sc.esc_isExtern = 0;
        td->e_sc.esc_isStatic = 0;
        td->e_isTypeDef = 0;
        exprSetExprsType(NULL, td);
        addTypeDesc(td);
    }

    exprSetExprsType((CExpr*)sym, td);

    CCOL_DListNode *curIte;
    CExpr *stmtsOrDefs = getParentBlockStmt(cur, &curIte);
    CExpr *declr = exprBinary(EC_DECLARATOR, (CExpr*)td, (CExpr*)sym);
    exprCopyLineNum(declr, cur);
    sym->e_declrExpr = declr;
    CExpr *initDecl = exprList1(EC_INIT_DECL, declr); 
    exprCopyLineNum(initDecl, cur);
    CExpr *initDecls = exprList1(EC_INIT_DECLS, initDecl); 
    exprCopyLineNum(initDecls, cur);
    int isStmts = (EXPR_CODE(stmtsOrDefs) == EC_COMP_STMT);
    CExpr *declOrDataDef = exprBinary(isStmts ? EC_DECL : EC_DATA_DEF, 
        NULL, initDecls);
    exprCopyLineNum(declOrDataDef, cur);
    if(isStmts)
        exprListCons(declOrDataDef, stmtsOrDefs);
    else {
        CCOL_DL_INSERT_PREV(EXPR_DLIST(stmtsOrDefs), declOrDataDef, curIte);
        EXPR_REF(declOrDataDef);
    }

    EXPR_ISCONVERTED(sym) = 1;
    EXPR_ISCONVERTED(declr) = 1;
    EXPR_ISCONVERTED(initDecl) = 1;
    EXPR_ISCONVERTED(initDecls) = 1;

    addSymbolDirect(EXPR_C(stmtsOrDefs)->e_symTab,
        sym, STB_IDENT);

    return sym;
}


/**
 * \brief
 * convert EC_MEMBER_REF
 *
 * - before: s.a
 * - after:  &(s)->a
 *
 * @param expr
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_memberRef(CExpr *expr, CSymbolTable *symTab, int *converted)
{
    assertExprCode(expr, EC_MEMBER_REF);
    EXPR_C(expr)->e_exprCode = EC_POINTS_AT;
    CExpr *composExpr = EXPR_B(expr)->e_nodes[0];
    CExprOfTypeDesc *composTd = EXPRS_TYPE(composExpr);
    CExprOfTypeDesc *td = allocPointerTypeDesc(composTd);
    CExpr *addrExpr = exprUnary(EC_ADDR_OF, composExpr);
    freeExpr(composExpr);
    EXPR_B(expr)->e_nodes[0] = addrExpr;
    EXPR_PARENT(addrExpr) = expr;
    EXPR_REF(addrExpr);

    exprSetExprsType(addrExpr, td);
    addTypeDesc(td);

    *converted = CONVS_REPLACE; 
    ++s_numConvsMemRef;
    return NULL;
}


/**
 * \brief
 * move symbol in symbol table to another symbol table
 *
 * @param sym
 *      target symbol
 * @param stb
 *      symbol group
 * @param dstSymTab
 *      destination symbol table
 * @param srcSymTab
 *      source symbol table
 */
PRIVATE_STATIC void
moveSymbol(CExprOfSymbol *sym, CSymbolTableGroupEnum stb,
    CSymbolTable *dstSymTab, CSymbolTable *srcSymTab)
{
    if(EXPR_ISNULL(sym))
        return;
    CExprOfSymbol *hsym = removeSymbolByGroup(
        srcSymTab, sym->e_symName, stb);
    if(hsym)
        addSymbolDirect(dstSymTab, hsym, stb);
}


/**
 * \brief
 * move tag symbol in symbol table to another symbol table
 *
 * @param td
 *      target type descriptor
 * @param dstSymTab
 *      destination symbll table
 * @param srcSymTab
 *      source symbll table
 */
PRIVATE_STATIC void
moveTaggedTypeSymbol(CExprOfTypeDesc *td,
    CSymbolTable *dstSymTab, CSymbolTable *srcSymTab)
{
    if(EXPR_ISNULL(td) || ETYP_IS_TAGGED(td) == 0)
        return;

    CExprOfSymbol *tag = getTagSymbol(td);
    moveSymbol(tag, STB_TAG, dstSymTab, srcSymTab);

    if(ETYP_IS_ENUM(td) == 0)
        return;

    //move enumerator symbols
    CExpr *enums = getMemberDeclsExpr(td);
    if(EXPR_ISNULL(enums))
        return;

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, enums) {
        CExprOfSymbol *sym = EXPR_SYMBOL(EXPR_L_DATA(ite));
        assertExprCode((CExpr*)sym, EC_IDENT);
        moveSymbol(sym, STB_IDENT, dstSymTab, srcSymTab);
    }
}


/**
 * \brief
 * move declared symbol in symbol table to another symbol table
 *
 * @param symTab
 *      destination symbol table
 * @param parentSymTab
 *      source symbol table
 * @param decl
 *      declaration
 */
PRIVATE_STATIC void
moveDeclSymbolFromParent(CSymbolTable *symTab,
    CSymbolTable *parentSymTab, CExprOfBinaryNode *decl)
{
    CCOL_DListNode *ite;
    CExpr *initDecls = decl->e_nodes[1];
    if(EXPR_ISNULL(initDecls)) {
        //tagged type definition
        CExprOfTypeDesc *td = EXPR_T(decl->e_nodes[0]);
        moveTaggedTypeSymbol(td, symTab, parentSymTab);
    } else {
        EXPR_FOREACH(ite, initDecls) {
            CExpr *initDecl = EXPR_L_DATA(ite);
            CExprOfBinaryNode *declr = EXPR_B(exprListHeadData(initDecl));
            CExprOfTypeDesc *td = declr ? EXPR_T(declr->e_nodes[0]) : NULL;
            CExprOfSymbol *sym = declr ? EXPR_SYMBOL(declr->e_nodes[1]) : NULL;
            moveSymbol(sym, STB_IDENT, symTab, parentSymTab);
            moveTaggedTypeSymbol(td, symTab, parentSymTab);
        }
    }
}


/**
 * \brief
 * convert EC_COMP_STMT
 *
 * - before:  { int x; ++x; int y; ++y }
 * - after:   { int x; ++x; { int y; ++y; } }
 *
 * @param stmtAndDecls
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_compStmt(CExpr *stmtAndDecls, CSymbolTable *symTab, int *converted)
{
    assertExprCode(stmtAndDecls,EC_COMP_STMT);

    CCOL_DListNode *ite;
    int declExists = 0, noDeclExists = 0;
    CExpr *firstDecl = NULL;
    EXPR_FOREACH(ite, stmtAndDecls) {
        CExpr *expr = EXPR_L_DATA(ite);
        if(expr && EXPR_CODE(expr) == EC_DECL) {
            if(noDeclExists) {
                declExists = 1;
                firstDecl = expr;
                break;
            }
        } else if(EXPR_ISNULL(expr) == 0) {
            noDeclExists = 1;
        }
    }

    if(declExists == 0)
        return NULL;

    CExpr *childStmts = exprList(EC_COMP_STMT);
    exprCopyLineNum(childStmts, firstDecl);
    CExpr *expr;
    CCOL_SList decls;
    memset(&decls, 0, sizeof(decls));

    do {
        expr = exprListRemoveTail(stmtAndDecls);
        if(expr) {
            exprListCons(expr, childStmts);
            if(EXPR_CODE(expr) == EC_DECL)
                CCOL_SL_CONS(&decls, expr);
        }
    } while(expr != firstDecl);

    if(EXPR_CODE(EXPR_PARENT(stmtAndDecls)) == EC_GCC_COMP_STMT_EXPR) {
        CExpr *compExpr = exprUnary(EC_GCC_COMP_STMT_EXPR, childStmts);
        exprSetExprsType(compExpr, &s_voidPtrTypeDesc);
        CExpr *exprStmt = exprUnary(EC_EXPR_STMT, compExpr);
        exprListAdd(stmtAndDecls, exprStmt);
    } else {
        exprListAdd(stmtAndDecls, childStmts);
    }

    *converted = CONVS_REPLACE;

    // move symbol in symbol table
    CSymbolTable *parentSymTab = EXPR_C(stmtAndDecls)->e_symTab;

    if(parentSymTab == NULL)
        goto end;

    CSymbolTable *symTab1 = allocSymbolTable1(parentSymTab);
    EXPR_C(childStmts)->e_symTab = symTab1;

    CCOL_SListNode *site;

    CCOL_SL_FOREACH(site, &decls) {
        CExprOfBinaryNode *decl = EXPR_B(CCOL_SL_DATA(site));
        moveDeclSymbolFromParent(symTab1, parentSymTab, decl);
    }

  end:

    CCOL_SL_CLEAR(&decls);
    ++s_numConvsC99StyleDeclBlock;

    return NULL;
}


/**
 * \brief
 * convert EC_FOR_STMT
 *
 * - before: for(int x;...
 * - after:  { int x; for(;... }
 *
 * @param forStmt
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_forStmt(CExpr *forStmt, CSymbolTable *symTab, int *converted)
{
    CExpr *init = exprListHeadData(forStmt);
    if(EXPR_ISNULL(init) || EXPR_CODE(init) != EC_DECL)
        return NULL;

    CExpr *x = exprListNextNData(EXPR_B(init)->e_nodes[1], 0);
    CExpr *ctlVar = EXPR_B(exprListNextNData(x, 0))->e_nodes[1];
    CExpr *initValue = EXPR_U(exprListNextNData(x, 2))->e_node;
    EXPR_U(exprListNextNData(x, 2))->e_node = NULL;

    CExpr *compStmt = exprList2(EC_COMP_STMT, init, forStmt);

    exprCopyLineNum(compStmt, forStmt);
    exprListRemoveHead(forStmt); // remove init

    CExpr *newInit = exprList1(EC_EXPRS, exprBinary(EC_ASSIGN, ctlVar, initValue));
    resolveType(newInit);
    exprListCons(newInit, forStmt); // add new init
    //exprListCons(exprNull(), forStmt);

    EXPR_REF(compStmt);

    EXPR_ISCONVERTED(init) = 1;
    EXPR_ISCONVERTED(forStmt) = 1;

    *converted = CONVS_REPLACE;
    ++s_numConvsDeclInForStmt;
    return compStmt;
}


/**
 * \brief
 * sub function of moveAllSymbolToParent()
 */
PRIVATE_STATIC void
moveAllSymbolToParent0(CCOL_HashTable *ht, CCOL_HashTable *pht)
{
    CCOL_HashEntry *he;
    CCOL_HashSearch hs;

    CCOL_HT_FOREACH(he, hs, ht) {
        char *key = CCOL_HT_KEY(ht, he);
        CCOL_Data data = CCOL_HT_DATA(he);
        CCOL_HT_REMOVE_STR(ht, key);
        CCOL_HT_PUT_STR(pht, key, data);
    }
}


/**
 * \brief
 * move symbol in symbol table related to expr
 * to symbol table related to parent
 *
 * @param expr
 *      source node
 * @param parent
 *      destination node
 */
PRIVATE_STATIC void
moveAllSymbolToParent(CExpr *expr, CExpr *parent)
{
    CSymbolTable *symTab = EXPR_C(expr)->e_symTab;

    if(symTab == NULL)
        return;

    CSymbolTable *parentSymTab = EXPR_C(parent)->e_symTab;

    if(parentSymTab) {
        const int stbs[] = { STB_IDENT, STB_TAG, STB_LABEL };

        for(int i = 0; i < sizeof(stbs) / sizeof(CSymbolTableGroupEnum); ++i) {
            CSymbolTableGroupEnum stb = stbs[i];
            moveAllSymbolToParent0(
                getSymbolHashTable(symTab, stb),
                getSymbolHashTable(parentSymTab, stb));
        }
    } else {
        EXPR_C(parent)->e_symTab = symTab;
	EXPR_C(expr)->e_symTab = NULL;
    }
}


/**
 * \brief
 * convert EC_CONDEXPR
 *
 * - before:  ... x ? : y
 * - after:   int r; ... (r=x) ? r : y
 *
 * @param condExpr
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_condExpr(CExpr *condExpr, CSymbolTable *symTab, int *converted)
{
    assertExprCode(condExpr, EC_CONDEXPR);
    CExpr *e2 = exprListNextNData(condExpr, 1);
    if(EXPR_ISNULL(e2) == 0)
        return NULL;

    CExpr *e1 = exprListHeadData(condExpr);
    CExprOfTypeDesc *rtd = EXPRS_TYPE(e1);

    if(symTab->stb_isGlobal) {
        // in data def (constant values)
        exprListRemoveHead(condExpr);
        freeExpr(exprListRemoveHead(condExpr));
        exprListCons(e1, condExpr);
        exprListCons(e1, condExpr);
    } else {
        // in expression statements 
        CExprOfSymbol *varSym = addTempVarDecl(condExpr, rtd, TVK_CONDEXPR);
        CExprOfSymbol *varSym1 = duplicateExprOfSymbol(varSym);
        CExprOfSymbol *varSym2 = duplicateExprOfSymbol(varSym);
        CExpr *asg = exprBinary(EC_ASSIGN, (CExpr*)varSym1, e1);
        rtd = EXPRS_TYPE(varSym);
        exprSetExprsType(asg, rtd);
        exprListRemoveHead(condExpr);
        freeExpr(exprListRemoveHead(condExpr));
        exprListCons((CExpr*)varSym2, condExpr);
        exprListCons(asg, condExpr);
        EXPR_PARENT(e1) = asg;
    }

    EXPR_ISCONVERTED(condExpr) = 1;

    *converted = CONVS_REPLACE;
    ++s_numConvsCond2ndExprNull;
    return NULL;
}


/**
 * \brief
 * convert EC_FUNCTION_CALL under EC_INIT
 *
 * - before:  ... <func() in initializer>
 * - after:   int r; ... r = func(); <r in initializer>
 *
 * @param initrsOrInit
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_initalizersOrInit(CExpr *initrsOrInit, CSymbolTable *symTab, int *converted,
    int allowCompLtr)
{
    if(s_transFuncInInit == 0)
        return NULL;

    CExpr *val = initrsOrInit;
    if(EXPR_CODE(initrsOrInit) == EC_INIT) {
        val = EXPR_U(initrsOrInit)->e_node;
        if(EXPR_CODE(val) == EC_INITIALIZERS)
            return NULL;
    }

    if(allowCompLtr == 0 &&
        isExprCodeChildStmtOf(initrsOrInit, EC_COMPOUND_LITERAL,
        EXPR_PARENT(initrsOrInit), NULL)) {
        return NULL;
    }

    CExpr *fc = getFuncCallAbsolutelyCalling(val);

    if(fc == NULL)
        return NULL;

    //treat static+inline+always_inline+const func as constant.
    //this code needs to compile the linux kernel
    if(isConstExpr(fc, 1))
        return NULL;

    CExprOfTypeDesc *rtd = EXPRS_TYPE(fc);
    CExpr *parent = EXPR_PARENT(fc);
    CExpr *valParent = EXPR_PARENT(val);
    CExprOfSymbol *varSym = addTempVarDecl(val, rtd, TVK_INITIALIZER);
    CExprOfSymbol *varSym1 = duplicateExprOfSymbol(varSym);
    CExprOfSymbol *varSym2 = duplicateExprOfSymbol(varSym);
    EXPR_REF(varSym2);
    CExpr *asg = exprBinary(EC_ASSIGN, (CExpr*)varSym1, fc);
    exprCopyLineNum(asg, initrsOrInit);
    rtd = EXPRS_TYPE(varSym);
    exprSetExprsType(asg, rtd);
    CExpr *stmt = exprUnary(EC_EXPR_STMT, asg);
    exprCopyLineNum(stmt, initrsOrInit);
    addGeneratedStmt(stmt, valParent);
    exprReplace(parent, fc, (CExpr*)varSym2);
    
    EXPR_ISCONVERTED(asg) = 1;
    EXPR_ISCONVERTED(stmt) = 1;
    EXPR_ISCONVERTED(val) = 1;
    EXPR_ISCONVERTED(fc) = 1;

    *converted = CONVS_REPLACE;
    ++s_numConvsFuncCallInInitializer;
    
    return NULL;
}


/**
 * \brief
 * convert EC_BRACED_EXPR
 *
 * - before: (x)
 * - after:   x
 *
 * @param br
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_bracedExpr(CExpr *br, CSymbolTable *symTab, int *converted)
{
    CExpr *e = EXPR_U(br)->e_node;
    if(e == NULL)
        return NULL;
    *converted = CONVS_REPLACE;
    EXPR_U(br)->e_node = NULL;
    return e;
}


/**
 * \brief
 * convert EC_ADDR_OF with EC_CONDEXPR
 *
 * - before:  &(x?y:z)
 * - after:   (x?&y:&z)
 *
 * @param addrOf
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_addrOf1(CExpr *addrOf, CSymbolTable *symTab, int *converted)
{
    assertExprCode(addrOf, EC_ADDR_OF);
    CExpr *condExpr = EXPR_U(addrOf)->e_node;
    CExpr *e1 = exprListNextNData(condExpr, 1);
    CExpr *e2 = exprListNextNData(condExpr, 2);
    CExpr *a1 = exprUnary(EC_ADDR_OF, e1);
    CExpr *a2 = exprUnary(EC_ADDR_OF, e2);
    exprCopyLineNum(a1, e1);
    exprCopyLineNum(a2, e2);
    CExprOfTypeDesc *td1 = allocPointerTypeDesc(EXPRS_TYPE(e1));
    CExprOfTypeDesc *td2 = allocPointerTypeDesc(EXPRS_TYPE(e2));
    exprSetExprsType(a1, td1);
    exprSetExprsType(a2, td2);
    exprSetExprsType(condExpr, td1);
    EXPR_REF(a1);
    EXPR_REF(a2);
    exprReplace(condExpr, e1, a1);
    exprReplace(condExpr, e2, a2);
    EXPR_U(addrOf)->e_node = NULL;

    *converted = CONVS_REPLACE;
    ++s_numConvsAddrOfCondExpr;
    return condExpr;
}


/**
 * \brief
 * convert EC_ADDR_OF
 *
 * - before:  &*x
 * - after:   x
 *
 * @param addrOf
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_addrOf(CExpr *addrOf, CSymbolTable *symTab, int *converted)
{
    assertExprCode(addrOf, EC_ADDR_OF);
    CExpr *pref = EXPR_U(addrOf)->e_node;
    switch(EXPR_CODE(pref)) {
    case EC_POINTER_REF:
        break;
    case EC_CONDEXPR:
        return convs_addrOf1(addrOf, symTab, converted);
    default:
        return NULL;
    }

    CExpr *e = EXPR_U(pref)->e_node;
    EXPR_U(pref)->e_node = NULL;
    *converted = CONVS_REPLACE;
    ++s_numConvsAddrOfPtrRef;
    return e;
}


/**
 * \brief
 * convert EC_POINTER_REF
 *
 * - before:  *&x
 * - after:   x
 *
 * @param ptrRef
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_pointerRef(CExpr *ptrRef, CSymbolTable *symTab, int *converted)
{
    CExpr *addrOf = EXPR_U(ptrRef)->e_node;
    if(EXPR_CODE(addrOf) != EC_ADDR_OF)
        return NULL;
    CExpr *e = EXPR_U(addrOf)->e_node;
    EXPR_U(addrOf)->e_node = NULL;
    *converted = CONVS_REPLACE;
    ++s_numConvsPtrRefAddrOf;
    return e;
}


/**
 * \brief
 * convert EC_FUNCTION_CALL
 *
 * - condition: typedef int f_t; f_t *fp;
 * - before: (*fp)()
 * - after:  fp()
 *
 * @param funcCall
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_functionCall1(CExpr *funcCall, CSymbolTable *symTab, int *converted)
{
    assertExprCode(funcCall, EC_FUNCTION_CALL);
    CExpr *exprs = EXPR_B(funcCall)->e_nodes[0];
    assertExprCode(exprs, EC_EXPRS);
    CExpr *ptrRef = exprListHeadData(exprs);
    assertExprCode(ptrRef, EC_POINTER_REF);
    CExpr *var = EXPR_U(ptrRef)->e_node;
    if(ETYP_IS_FUNC(getRefType(EXPRS_TYPE(var))) == 0)
        return NULL;
    assertExprCode(var, EC_IDENT);

    EXPR_U(ptrRef)->e_node = NULL;
    exprReplace(funcCall, exprs, var);

    *converted = 1;
    return NULL;
}


/**
 * \brief
 * convert EC_FUNCTION_CALL
 *
 * - condition: struct s func();
 * - before: func()
 * - after:  struct s r; r = func(); r;
 *
 * @param funcCall
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_functionCall(CExpr *funcCall, CSymbolTable *symTab, int *converted)
{
    assertExprCode(funcCall, EC_FUNCTION_CALL);

    CExpr *funcExpr = EXPR_B(funcCall)->e_nodes[0];
    if(EXPR_CODE(funcExpr) == EC_EXPRS &&
        EXPR_L_SIZE(funcExpr) == 1) {
        CExpr *funcExpr1 = exprListHeadData(funcExpr);
        if(EXPR_CODE(funcExpr1) == EC_POINTER_REF) {
            CExpr *funcExpr2 = EXPR_U(funcExpr1)->e_node;
            if(EXPR_CODE(funcExpr2) == EC_IDENT &&
                EXPR_SYMBOL(funcExpr2)->e_symType == ST_VAR) {
                return convs_functionCall1(funcCall, symTab, converted);
            }
        }
    }

    CExprOfTypeDesc *rtd = EXPRS_TYPE(funcCall);
    if(ETYP_IS_COMPOSITE(getRefType(rtd)) == 0)
        return NULL;

    CExpr *parent = NULL;
    int isAddr = isExprCodeChildOf(
        funcCall, EC_ADDR_OF, EXPR_PARENT(funcCall), &parent);
    if(isAddr == 0)
        return NULL;

    CExprOfSymbol *varSym = addTempVarDecl(funcCall, rtd, TVK_FUNCTION_CALL);
    CExprOfSymbol *varSym1 = duplicateExprOfSymbol(varSym);
    CExprOfSymbol *varSym2 = duplicateExprOfSymbol(varSym);

    CExpr *asg = exprBinary(EC_ASSIGN, (CExpr*)varSym1, funcCall);
    exprCopyLineNum(asg, funcCall);
    CExpr *exprStmt = exprUnary(EC_EXPR_STMT, asg);
    exprSetExprsType(asg, rtd);
    exprSetExprsType((CExpr*)varSym1, rtd);
    exprSetExprsType((CExpr*)varSym2, rtd);
    addGeneratedStmt(exprStmt, parent);
    EXPR_REF(varSym2);

    EXPR_ISCONVERTED(exprStmt) = 1;
    EXPR_ISCONVERTED(asg) = 1;

    *converted = CONVS_REPLACE;
    ++s_numConvsFuncCallReturnComposType;
    return (CExpr*)varSym2;
}


/**
 * \brief
 * convert EC_EXPRS under EC_ADDR_OF
 *
 * - before: <addr_of|pointer_ref <exprs <expr>>>
 * - after:  <addr_of|pointer_ref element expr>
 *
 * @param exprs
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_exprs(CExpr *exprs, CSymbolTable *symTab, int *converted)
{
    assertExprCode(exprs, EC_EXPRS);

    CExpr *parent = EXPR_PARENT(exprs);
    if(EXPR_CODE(parent) != EC_ADDR_OF &&
        EXPR_CODE(parent) != EC_POINTER_REF)
        return NULL;

    int sz = EXPR_L_SIZE(exprs);
    if(sz != 1)
        return NULL;

    CExpr *e = exprListHeadData(exprs);

    if(EXPR_ISNULL(e))
        return NULL;

    e = exprListRemoveHead(exprs);
    EXPR_REF(e);
    *converted = CONVS_REPLACE;
    return e;
}


/**
 * \brief
 * convert EC_GCC_COMP_STMT_EXPR
 *
 * - before: &{(...)}
 * - after: int r; (r = {(...)}, &r)
 *
 * &{(...)} does not exist original syntax, but generated by
 * member ref conversion.
 * ex) struct st { int a; } s; ({s;}).a;
 *    -> (&({s;}))->a
 *
 * @param compExpr
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_gccCompExpr(CExpr *compExpr, CSymbolTable *symTab, int *converted)
{
    assertExprCode(compExpr, EC_GCC_COMP_STMT_EXPR);
    CExpr *parent = EXPR_PARENT(compExpr);

    if(EXPR_CODE(parent) != EC_ADDR_OF)
        return NULL;

    CExprOfTypeDesc *td = EXPRS_TYPE(compExpr);
    CExprOfSymbol *varSym = addTempVarDecl(compExpr, td, TVK_GCC_COMP_STMT_EXPR);
    CExprOfSymbol *varSym1 = duplicateExprOfSymbol(varSym);
    CExprOfSymbol *varSym2 = duplicateExprOfSymbol(varSym);
    td = EXPRS_TYPE(varSym);
    CExpr *asg = exprBinary(EC_ASSIGN, (CExpr*)varSym1, compExpr);
    exprCopyLineNum(asg, compExpr);
    exprSetExprsType(asg, td);
    CExpr *varAddr = exprUnary(EC_ADDR_OF, (CExpr*)varSym2);
    exprCopyLineNum(varAddr, compExpr);
    exprSetExprsType(varAddr, EXPRS_TYPE(parent));
    CExpr *exprs = exprList2(EC_EXPRS, asg, varAddr);
    exprCopyLineNum(exprs, compExpr);
    exprSetExprsType(exprs, EXPRS_TYPE(parent));

    EXPR_ISCONVERTED(compExpr) = 1;
    EXPR_ISCONVERTED(asg) = 1;
    EXPR_ISCONVERTED(varAddr) = 1;
    EXPR_ISCONVERTED(exprs) = 1;
    *converted = CONVS_REPLACE;
    ++s_numConvsGccCompExprAddr;

    return exprs;
}


/**
 * \brief
 * replace label symbol name
 *
 * @param expr
 *      target node
 * @param oldName
 *      old label name
 * @param newName
 *      new label name
 */
PRIVATE_STATIC void
replaceGccLabelSym(CExprOfSymbol *sym, const char *oldName, const char *newName)
{
    if(sym->e_symType != ST_GCC_LABEL ||
        (oldName && strcmp(sym->e_symName, oldName) != 0))
        return;

    free(sym->e_symName);
    sym->e_symName = ccol_strdup(newName, MAX_NAME_SIZ);
    sym->e_symType = ST_LABEL;
    EXPR_ISCONVERTED(sym) = 1;
}


/**
 * \brief
 * replace label declared with __label__ to function scope label
 *
 * @param expr
 *      target node
 * @param oldName
 *      old label name
 * @param newName
 *      new label name
 * @param isTop
 *      0:expr is child node of node which is started conversion,
 *      1:expr is node which is started conversion
 */
PRIVATE_STATIC void
replaceGccLabel(CExpr *expr, const char *oldName, const char *newName, int isTop)
{
    if(expr == NULL || (isTop == 0 && isScopedStmt(expr)))
        return;

    switch(EXPR_CODE(expr)) {
    case EC_GOTO_STMT:
    case EC_LABEL:
    case EC_GCC_LABEL_ADDR: {
            CExpr *node = EXPR_U(expr)->e_node;
            if(EXPR_CODE(node) == EC_IDENT) {
                replaceGccLabelSym(EXPR_SYMBOL(node), oldName, newName);
                EXPR_ISCONVERTED(expr) = 1;
                EXPR_ISGCCSYNTAX(expr) = (EXPR_CODE(expr) != EC_GCC_LABEL_ADDR);
                return;
            }
        }
        break;
    default:
        break;
    }

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr)
        replaceGccLabel(ite.node, oldName, newName, 0);
}


/**
 * \brief
 * convert gcc __label__
 *
 * - before: __label__ l;  l:
 * - after:  local_l:
 *
 * @param stmt
 *      target node
 * @param symTab
 *      symbol table
 */
PRIVATE_STATIC void
convs_gccLabel(CExpr *stmts, CSymbolTable *symTab)
{
    assertExpr(stmts, EXPR_CODE(stmts) == EC_COMP_STMT);
    CCOL_HashEntry *he;
    CCOL_HashSearch hs;
    CCOL_HashTable *ht = &symTab->stb_labelGroup;

    CCOL_HT_FOREACH(he, hs, ht) {
        CExprOfSymbol *sym = CCOL_HT_DATA(he);
        if(sym->e_symType == ST_GCC_LABEL) {
            char *labelName = s_charBuf[0];
            sprintf(labelName, "%s%d_%s",
                s_gccLocalLabelPrefix, s_gccLabelSeq++, sym->e_symName);
            replaceGccLabel(stmts, sym->e_symName, labelName, 1);
            removeSymbolByGroup(symTab, sym->e_symName, STB_LABEL);
            replaceGccLabelSym(sym, NULL, labelName);
            
            CSymbolTable *symTab1 = symTab;
            while(symTab1 && symTab1->stb_isFuncDefBody == 0)
                symTab1 = symTab1->stb_parentTab;
            if(symTab1 == NULL)
                symTab1 = symTab;
            addSymbolDirect(symTab1, sym, STB_LABEL);
            EXPR_ISCONVERTED(sym) = 1;
            ++s_numConvsGccLabel;
        }
    }
}


/**
 * \brief
 * convert sizeof/alignof operator 
 *
 * - condition: enum e { a }
 * - before: sizeof(typeof(a))
 * - after:  sizeof(int)
 *
 * @param sizeAlignOf
 *      target node
 * @param symTab
 *      symbol table
 * @param[out] converted 
 *      if target node is converted, set to non-CONVS_STAY
 * @return
 *      NULL or converted node
 */
PRIVATE_STATIC CExpr*
convs_sizeAlignOf(CExpr *sizeAlignOf, CSymbolTable *symTab, int *converted)
{
    assertExpr(sizeAlignOf, EXPR_CODE(sizeAlignOf) == EC_SIZE_OF ||
        EXPR_CODE(sizeAlignOf) == EC_GCC_ALIGN_OF);

    CExpr *arg = EXPR_U(sizeAlignOf)->e_node;
    if(EXPR_CODE(arg) != EC_TYPE_DESC)
        return NULL;

    CExprOfTypeDesc *argTd = EXPR_T(arg);
    if(ETYP_IS_DERIVED(argTd) == 0)
        return NULL;

    CExprOfTypeDesc *argTdo = getRefType(argTd);
    if(ETYP_IS_ENUM(argTdo) == 0)
        return NULL;

    EXPR_U(sizeAlignOf)->e_node = (CExpr*)&s_numTypeDescs[BT_INT];
    EXPR_REF(EXPR_U(sizeAlignOf)->e_node);
    freeExpr(arg);

    *converted = CONVS_REPLACE;

    return NULL;
}


//! conversion mode
typedef enum {
    CONVS_MODE_STMT,
    CONVS_MODE_EXPR,
    CONVS_MODE_STMT2,
    CONVS_MODE_DECL,
} CConvsModeEnum;


/**
 * \brief
 * convert syntax
 *
 * @param mode
 *      conversion mode
 * @param expr
 *      target node
 * @param symTab
 *      symbol table
 */
PRIVATE_STATIC int
convertSyntax0(CConvsModeEnum mode, CExpr *expr, CSymbolTable *symTab)
{
    if(expr == NULL || EXPR_ISERROR(expr))
        return 1;

    CSymbolTable *symTab1 = EXPR_C(expr)->e_symTab;

    if(symTab1) {
        symTab = symTab1;
        if(EXPR_CODE(expr) == EC_COMP_STMT)
            convs_gccLabel(expr, symTab);
    }
        
    CExpr *newExpr, *parent;
    int converted;

  again:

    parent = EXPR_PARENT(expr);
    newExpr = NULL;
    converted = CONVS_STAY;

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr)
        if(convertSyntax0(mode, ite.node, symTab) == 0)
            return 0;

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfSymbol:
        if(EXPR_SYMBOL(expr)->e_valueExpr &&
            convertSyntax0(mode, EXPR_SYMBOL(expr)->e_valueExpr, symTab) == 0)
            return 0;
        break;

    case STRUCT_CExprOfTypeDesc: {
            CExprOfTypeDesc *td = EXPR_T(expr);
            if(ETYP_IS_TAGGED(td) && td->e_typeExpr &&
                convertSyntax0(mode, td->e_typeExpr, symTab) == 0)
                return 0;
            if(td->e_len.eln_lenExpr &&
                convertSyntax0(mode, td->e_len.eln_lenExpr, symTab) == 0)
                return 0;
            if(td->e_bitLenExpr &&
                convertSyntax0(mode, td->e_bitLenExpr, symTab) == 0)
                return 0;
        }
        break;
    default:
        break;
    }

    if(EXPR_ISDELETING(expr)) {
        converted = CONVS_REPLACE;
        newExpr = exprNull();
    } else {
        switch(mode) {
        case CONVS_MODE_STMT:
            switch(EXPR_CODE(expr)) {
            case EC_FOR_STMT:
                newExpr = convs_forStmt(expr, symTab, &converted);
                break;
            case EC_INIT:
            case EC_INITIALIZERS:
                newExpr = convs_initalizersOrInit(expr, symTab, &converted, 0);
                break;
            default:
                break;
            }
            break;

        case CONVS_MODE_EXPR:
            switch(EXPR_CODE(expr)) {
            case EC_MEMBER_REF:
                newExpr = convs_memberRef(expr, symTab, &converted);
                break;
            case EC_CONDEXPR:
                newExpr = convs_condExpr(expr, symTab, &converted);
                break;
            case EC_BRACED_EXPR:
                newExpr = convs_bracedExpr(expr, symTab, &converted);
                break;
            case EC_ADDR_OF:
                newExpr = convs_addrOf(expr, symTab, &converted);
                break;
            case EC_POINTER_REF:
                newExpr = convs_pointerRef(expr, symTab, &converted);
                break;
            case EC_FUNCTION_CALL:
                newExpr = convs_functionCall(expr, symTab, &converted);
                break;
            case EC_EXPRS:
                newExpr = convs_exprs(expr, symTab, &converted);
                break;
            case EC_GCC_COMP_STMT_EXPR:
                newExpr = convs_gccCompExpr(expr, symTab, &converted);
                break;
            case EC_SIZE_OF:
            case EC_GCC_ALIGN_OF:
                newExpr = convs_sizeAlignOf(expr, symTab, &converted);
                break;
            default:
                break;
            }
            break;

        case CONVS_MODE_STMT2:
            switch(EXPR_CODE(expr)) {
            case EC_INIT:
            case EC_INITIALIZERS:
                //for comound lieteral
                newExpr = convs_initalizersOrInit(expr, symTab, &converted, 1);
                break;
            default:
                break;
            }
            break;

        case CONVS_MODE_DECL:
            if(EXPR_CODE(expr) == EC_COMP_STMT) {
                newExpr = convs_compStmt(expr, symTab, &converted);
            }

            //move symbol table in control statements
            switch(EXPR_CODE(expr)) {
            case EC_IF_STMT:
            case EC_SWITCH_STMT:
            case EC_FOR_STMT:
            case EC_WHILE_STMT:
            case EC_DO_STMT: {
                    CExprCodeEnum pec = EXPR_CODE(EXPR_PARENT(expr));
                    if(pec == EC_COMP_STMT || pec == EC_STMTS_AND_DECLS)
                        moveAllSymbolToParent(expr, EXPR_PARENT(expr));
                }
            default:
                break;
            }
            break;
        }
    }

    switch(converted) {
    case CONVS_REPLACE:
        //modified or replaced
        if(newExpr) {
            exprReplace(parent, expr, newExpr);
            expr = newExpr;
        }
        goto again;

    case CONVS_RESET:
        //deleted
        return 0;

    case CONVS_DELETE:
        exprRemove(parent, expr);
        break;
    }

    return 1;
}


/**
 * \brief
 * print conversion count
 *
 * @param msg
 *      message
 * @param numConvs
 *      conversion count
 */
PRIVATE_STATIC void
printNumConvs(const char *msg, int numConvs)
{
    if(numConvs == 0)
        return;
    printf("transformed %4d location(s) : ", numConvs);
    fputs(msg, stdout);
    fputs("\n", stdout);
}


/**
 * \brief
 * convert syntax to XcodeML's syntax
 *
 * @param expr
 *      target node
 */
void
convertSyntax(CExpr *expr)
{
    if(s_verbose) {
        printf("transforming syntax ...\n");
    }

    for(CConvsModeEnum mode = CONVS_MODE_STMT;
        mode <= CONVS_MODE_DECL; ++mode) {

        setExprParent(expr, NULL);
        while(convertSyntax0(mode, expr, NULL) == 0);
    }

    if(s_verbose) {
        printNumConvs(
            "c99 style declaration block",
            s_numConvsC99StyleDeclBlock);

        printNumConvs(
            "declarator in for statment's init block",
            s_numConvsDeclInForStmt);

        printNumConvs(
            "condition expression that 2nd expression is null",
            s_numConvsCond2ndExprNull);

        printNumConvs(
            "function call in initializer",
            s_numConvsFuncCallInInitializer);

        printNumConvs(
            "function call which returns struct/union value",
            s_numConvsFuncCallReturnComposType);

        printNumConvs(
            "member reference of compound statement expression",
            s_numConvsGccCompExprAddr);

        printNumConvs(
            "anonymous member reference",
            s_numConvsAnonMemberAccess);

        printNumConvs(
            "label declared by __label__ to function scope label",
            s_numConvsGccLabel);

        printNumConvs(
            "'.' to '->'",
            s_numConvsMemRef);

        printNumConvs(
            "delete '&' '*'",
            s_numConvsAddrOfPtrRef);

        printNumConvs(
            "delete '*' '&'",
            s_numConvsPtrRefAddrOf);

        printNumConvs(
            "'#pragma pack' to packed/aligned attribute",
            s_numConvsPragmaPack);

        if(s_tmpVarSeq > 1) {
            printf("add %d variable(s) to transform\n",
                s_tmpVarSeq - 1);
        }
    }
}

