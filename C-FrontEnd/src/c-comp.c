/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-comp.c
 */

#include <limits.h>

#include "c-comp.h"
#include "c-option.h"
#include "c-pragma.h"

PRIVATE_STATIC void
compile_typeDesc(CExprOfTypeDesc *td, CDeclaratorContext declrContext);

PRIVATE_STATIC void
compile_typeDesc1(CExprOfTypeDesc *td, CDeclaratorContext declrContext,
    CExprOfSymbol *sym);

PRIVATE_STATIC void
compile_params(CExpr *params, int addSym);

PRIVATE_STATIC void
compile_declarator(CExprOfTypeDesc *headType,
    CExprOfTypeDesc *attrTd, CExprOfBinaryNode *declr,
    CSymbolTypeEnum varType, int addSym);

PRIVATE_STATIC void
compile_declaratorInMemberDecl(CExprOfTypeDesc *headType,
    CExprOfTypeDesc *attrTd, CExprOfBinaryNode *declr);

PRIVATE_STATIC void
compile_declaratorInParams(CExprOfTypeDesc *headType,
    CExprOfBinaryNode *declr, int addSym);

PRIVATE_STATIC void
compile_declaratorInTypeName(CExprOfTypeDesc *headType, CExprOfBinaryNode *declr);

PRIVATE_STATIC void
compile_funcDef(CExpr *funcDef, CExpr *parent);

PRIVATE_STATIC void
collectTypeDesc0(CExpr *expr);

PRIVATE_STATIC CExpr*
compile_typeName(CExpr *expr, CExpr *parent);

//! sequence number for anonymous tag 
PRIVATE_STATIC int s_anonymousTagNameSeq = 1;
//! sequence number for anonymous member 
PRIVATE_STATIC int s_anonymousMemberNameSeq = 1;


/**
 * \brief
 * compilation context
 */
struct CCompileContext {
    CCOL_SList      cc_funcDefTypes;
} s_compileContext;


/**
 * \brief
 * set type descriptor to bottom reference type
 *
 * @param parentTd
 *      parent type descriptor
 * @param td
 *      bottom type descriptor
 */
PRIVATE_STATIC void
setTypeToBottom(CExprOfTypeDesc *parentTd, CExprOfTypeDesc *td)
{
    if(parentTd == td || ETYP_IS_DERIVED(parentTd)) {
        /* already set */
        return;
    }

    if(parentTd->e_typeExpr) {
        if(EXPR_STRUCT(parentTd->e_typeExpr) == STRUCT_CExprOfTypeDesc)
            setTypeToBottom(EXPR_T(parentTd->e_typeExpr), td);
    } else {
        assertExpr((CExpr*)parentTd, ETYP_IS_BASICTYPE(parentTd) == 0);
        parentTd->e_typeExpr = (CExpr*)td;
        EXPR_REF(td);
    }
}


/**
 * \brief
 * compile EC_PARAMS
 *
 * @param params
 *      target node
 * @param addSym
 *      set to 1 for adding symbol to symbol table
 */
PRIVATE_STATIC void
compile_params(CExpr *params, int addSym)
{
    assert(params);
    assertExprCode(params, EC_PARAMS);

    CCOL_DListNode *ite, *iten;
    CExpr *nparams = exprList(EC_PARAMS);
    CExprOfTypeDesc *paramType;
    CExprOfBinaryNode *paramDeclr;
    int resetParams = 1;

    EXPR_FOREACH_SAFE(ite, iten, params) {
        CExprOfBinaryNode *param = EXPR_B(EXPR_L_DATA(ite));

        switch(EXPR_CODE(param)) {
        case EC_DECLARATOR:
            /* already resolved */
            if(addSym) {
                CExprOfTypeDesc *ptd = EXPR_T(param->e_nodes[0]);
                CExprOfSymbol *sym = EXPR_SYMBOL(param->e_nodes[1]);
                if(EXPR_ISNULL(sym) == 0) {
                    addSymbolAt(sym, (CExpr*)param, NULL, ptd, ST_PARAM, 0, addSym);
                    EXPRS_TYPE(sym) = ptd;
                }
            }
            resetParams = 0;
            break;
        case EC_ELLIPSIS:
            if(resetParams) {
                EXPR_REF(param);
                exprListAdd(nparams, (CExpr*)param);
            }
            break;
        case EC_PARAM:
            paramType = EXPR_T(param->e_nodes[0]);
            assertExpr((CExpr*)param, paramType);
            paramDeclr = EXPR_B(param->e_nodes[1]);

            if(paramDeclr) {
                exprJoinAttrToPre((CExpr*)paramType, (CExpr*)param);
                compile_declaratorInParams(paramType, paramDeclr, addSym);
            } else {
                paramDeclr = allocExprOfBinaryNode1(
                    EC_DECLARATOR, (CExpr*)paramType, (CExpr*)NULL);
                compile_typeDesc(paramType, DC_IN_PARAMS);
            }
            exprListAdd(nparams, (CExpr*)paramDeclr);
            EXPR_SET(param->e_nodes[0], NULL);
            break;
        default:
            ABORT();
        }
    }

    if(resetParams) {
        exprListClear(params);

        EXPR_FOREACH_SAFE(ite, iten, nparams) {
            CExpr *declr = exprListRemove(nparams, ite);
            exprListAdd(params, declr);
        }
    }

    freeExpr(nparams);
}


/**
 * \brief
 * move flags in return type descriptor to parent type descriptor
 *
 * @param dstTd
 *      destination
 * @param srcTd
 *      source
 */
PRIVATE_STATIC void
transferScspecFlag(CExprOfTypeDesc *dstTd, CExprOfTypeDesc *srcTd)
{
    int isInline = 0, isExtern = 0, isStatic = 0, isAuto = 0, isGccThread = 0;

    for(CExprOfTypeDesc *td = srcTd; td != NULL; ) {

        if(ETYP_IS_FUNC(td) == 0 && td->e_tq.etq_isInline)
            isInline = 1;

        if(td->e_sc.esc_isExtern)
            isExtern = 1;

        if(td->e_sc.esc_isStatic)
            isStatic = 1;

        if(td->e_sc.esc_isAuto)
            isAuto = 1;

        if(td->e_sc.esc_isGccThread)
            isGccThread = 1;

        if(td->e_typeExpr && EXPR_CODE(td->e_typeExpr) == EC_TYPE_DESC) {
            td = EXPR_T(td->e_typeExpr);
        } else if(ETYP_IS_DERIVED(td)) {
            td = td->e_refType;
        } else {
            td = NULL;
        }
    }

    if(isInline)
        dstTd->e_tq.etq_isInline = 1;
    if(isExtern)
        dstTd->e_sc.esc_isExtern = 1;
    if(isStatic)
        dstTd->e_sc.esc_isStatic = 1;
    if(isAuto)
        dstTd->e_sc.esc_isAuto = 1;
    if(isGccThread)
        dstTd->e_sc.esc_isGccThread = 1;
}


/**
 * \brief
 * set alternative symbol name to anoymous tagged type
 *
 * @param body
 *      struct/union/enum's definition
 * @param td
 *      type descriptor
 * @param declrSym
 *      declared symbol (var/type)
 */
PRIVATE_STATIC void
setAnonymousTagSymbol(CExpr *body, CExprOfTypeDesc *td, CExprOfSymbol *declrSym)
{
    char buf[128];

    for(;;) {
        if(declrSym) {
            sprintf(buf, "%s%d_%s", s_anonymousCompositePrefix,
                s_anonymousTagNameSeq++, declrSym->e_symName);
        } else {
            sprintf(buf, "%s%d", s_anonymousCompositePrefix, s_anonymousTagNameSeq++);
        }
        CExprOfSymbol *sym = findSymbolByGroup(buf, STB_TAG);
        if(sym == NULL)
            break;
    }

    char *token = ccol_strdup(buf, 128);
    CExprOfSymbol *tag = allocExprOfSymbol(EC_IDENT, token);
    EXPR_REF(tag);
    exprCopyLineNum((CExpr*)tag, body);
    addSymbolAt(tag, body, NULL, td, ST_TAG, 0, 1);

    CExprOfSymbol *nullSym = getTagSymbol(td);
    assertExpr(body, nullSym);
    assertExprCode((CExpr*)nullSym, EC_NULL_NODE);
    exprReplace(body, (CExpr*)nullSym, (CExpr*)tag);
}


/**
 * \brief
 * compile and add enumerator symbol to symbol table
 *
 * @param td
 *      enum type descriptor
 */
PRIVATE_STATIC void
addEnumeratorSymbols(CExprOfTypeDesc *td)
{
    assert(td);
    assertExpr((CExpr*)td, ETYP_IS_ENUM(td));
    assertExpr((CExpr*)td, td->e_typeExpr);

    CExpr *enums = getMemberDeclsExpr(td);

    if(EXPR_ISNULL(enums))
        return;

    compile1(enums, NULL);

    // set enum constant values
    assertExprCode(enums, EC_ENUMERATORS);
    CExpr *preValExpr = NULL;
    td->e_isUsed = 1;
    int val = 0, err = 0;
    const int fn = 0, rc = 1;

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, enums) {
        int isPureConst = 0;
        CExprOfSymbol *esym = EXPR_SYMBOL(EXPR_L_DATA(ite));
        exprSetExprsType((CExpr*)esym, td);
        addSymbolAt(esym, enums, NULL, &s_enumeratorTypeDesc,
            ST_ENUM, fn, rc);

        if(EXPR_ISNULL(esym->e_valueExpr)) {
            isPureConst = (preValExpr == NULL);
        } else {
            compile1(esym->e_valueExpr, (CExpr*)esym);
            CNumValueWithType n;
            if(getConstNumValue(esym->e_valueExpr, &n) == 0 ||
                (n.nvt_numKind != NK_LL && n.nvt_numKind != NK_ULL)) {
                if(n.nvt_isConstButUnreducable == 0) {
                    addError((CExpr*)esym, CERR_017, esym->e_symName);
                    err = 1;
                }
            }

            if(err == 0) {
                esym->e_isEnumInited = 1;

                if(n.nvt_isConstButMutable || n.nvt_isConstButUnreducable) {
                    preValExpr = esym->e_valueExpr;
                } else {
                    isPureConst = 1;
                    freeExpr(esym->e_valueExpr);
                    esym->e_valueExpr = NULL;
                    val = (int)getCastedLongValue(&n);
                    preValExpr = NULL;
                }
            }
        }

        if(err)
            continue;

        if(isPureConst) {
            CExpr *valueExpr = (CExpr*)allocExprOfNumberConst(
                EC_NUMBER_CONST, BT_INT, CD_DEC, NULL);
            EXPR_NUMBERCONST(valueExpr)->e_numValue.ll = val;
            EXPR_REF(valueExpr);
            esym->e_valueExpr = valueExpr;
        } else {
            esym->e_isConstButUnreducable = 1;
        }

        ++val;
    }
}


/**
 * \brief
 * compile type descriptor
 *
 * @param td
 *      target type descriptor
 * @param declrContext
 *      declarator context
 * @param declrSym
 *      declared symbol
 */
PRIVATE_STATIC void
compile_typeDesc1(CExprOfTypeDesc *td, CDeclaratorContext declrContext,
    CExprOfSymbol *declrSym)
{
    if(EXPR_ISCOMPILED(td))
        return;

    EXPR_ISCOMPILED(td) = 1;
    td->e_isCompiling = 1;

    if(ETYP_IS_TAGGED(td)) {

        CExprOfSymbol *sym = getTagSymbol(td);
        CExpr *memDecls = getMemberDeclsExpr(td);
        if(td->e_isNoMemDecl == 0)
            td->e_isNoMemDecl = EXPR_ISNULL(memDecls);

        if(td->e_isAnonTag == 0 && td->e_isNoMemDecl == 0 &&
            EXPR_ISNULL(sym)) {
            td->e_isAnonTag = 1;
            setAnonymousTagSymbol(td->e_typeExpr, td, declrSym);
        } else if(EXPR_ISNULL(sym) == 0) {
            CExprOfSymbol *tsym = findSymbolByGroup(sym->e_symName, STB_TAG);
            if(tsym == NULL && declrContext == DC_IN_PARAMS) {
                addError((CExpr*)td, CERR_018,
                    ETYP_IS_STRUCT(td) ? "struct" :
                    (ETYP_IS_UNION(td) ? "union" : "enum"), sym->e_symName);
            } else {
                const int fn = (declrContext == DC_IN_FUNC_DEF), rc = 1;
                addSymbolAt(sym, td->e_typeExpr, NULL, td, ST_TAG, fn, rc);
            }
        }

        if(td->e_isNoMemDecl == 0) {
            CExpr *attrHolder = exprListHeadData(td->e_typeExpr);
            exprJoinAttrToPre((CExpr*)td, attrHolder);
            exprFixAttr(td, NULL, DC_IN_NO_MEMBER_DECL, 1);
        }

        if(ETYP_IS_ENUM(td)) {
            td->e_isCompiling = 0; // allow use self type in initial value
                                   // ex) enum e { a = sizeof(e) }
            resolveType((CExpr*)td);
            addEnumeratorSymbols(td);
        } else {
            // compile for nested struct/union in memDecls
            compile1(memDecls, td->e_typeExpr);
        }
    } else {
        if(td->e_typeExpr) {
            if(EXPR_CODE(td->e_typeExpr) == EC_TYPE_DESC)
                compile_typeDesc((CExprOfTypeDesc*)td->e_typeExpr, declrContext);
            else
                compile1(td->e_typeExpr, (CExpr*)td);
        }

        compile1(td->e_bitLenExpr, (CExpr*)td);
        compile1(td->e_len.eln_lenExpr, (CExpr*)td);

        if(ETYP_IS_FUNC(td) && td->e_paramExpr) {
            int as = (declrContext == DC_IN_FUNC_DEF);
            compile_params(td->e_paramExpr, as);
        }
    }

    td->e_isCompiling = 0;

    resolveType((CExpr*)td);
}


/**
 * \brief
 * compile type descriptor
 *
 * @param td
 *      target type descriptor
 * @param declrContext
 *      declarator context
 */
PRIVATE_STATIC void
compile_typeDesc(CExprOfTypeDesc *td, CDeclaratorContext declrContext)
{
    compile_typeDesc1(td, declrContext, NULL);
}


/**
 * \brief
 * get gcc attributes of specified kind in gcc attributes list
 *
 * @param attrs
 *      gcc attributes list
 * @param gak
 *      gcc attribute kind
 * @return
 *      allocated gcc attributes list
 */
PRIVATE_STATIC CExpr*
getGccAttrs(CExpr *attrs, CGccAttrKindEnum gak)
{
    if(EXPR_L_ISNULL(attrs))
        return NULL;

    CCOL_DListNode *ite;
    CExpr *attrs1 = exprList(EC_GCC_ATTRS);
    EXPR_FOREACH(ite, attrs) {
        CExpr *args = EXPR_L_DATA(ite);
        CExprOfBinaryNode *arg = EXPR_B(exprListHeadData(args));
        if((arg->e_gccAttrKind & gak) > 0) {
            exprListAdd(attrs1, (CExpr*)args);
        }
    }

    if(EXPR_L_SIZE(attrs1) == 0) {
        freeExpr(attrs1);
        attrs1 = NULL;
    }

    return attrs1;
}


/**
 * \brief
 * compile declarator
 *
 * @param headType
 *      head type descriptor
 * @param attrTd
 *      type descriptor which has gcc attributes
 * @param declr
 *      declarator
 * @param varType
 *      symbol type to set to symbols in declarator
 * @param addSym
 *      set to 1 for adding symbol to symbol table
 * @param declrContext
 *      declarator context
 */
PRIVATE_STATIC void
compile_declarator0(
    CExprOfTypeDesc *headType, CExprOfTypeDesc *attrTd,
    CExprOfBinaryNode *declr,
    CSymbolTypeEnum varType, int addSym, CDeclaratorContext declrContext)
{
    assert(declr);
    assertExprCode((CExpr*)declr, EC_DECLARATOR);
    assert(headType);
    assertExprCode((CExpr*)headType, EC_TYPE_DESC);

    if(EXPR_ISCOMPILED(declr))
        return;
    EXPR_ISCOMPILED(declr) = 1;

    CExprOfTypeDesc *declrType = EXPR_T(declr->e_nodes[0]);

    if(EXPR_ISNULL(declrType)) {
        freeExpr((CExpr*)declrType);
        declrType = NULL;
    }

    CExprOfSymbol *sym = EXPR_SYMBOL(declr->e_nodes[1]);
    const char *symName = "(anonymous)";
    if(sym && EXPR_CODE(sym) == EC_IDENT)
        symName = sym->e_symName;
    CExprOfTypeDesc *regType;
    int isTypeDef = headType->e_isTypeDef;

    if(declrType) {
        regType = declrType;
        transferScspecFlag(declrType, headType);
        setTypeToBottom(declrType, headType);
        //duplicate head type to add attributes to
        //each declarator type
        //ex) <attr1> int(headType) *(declrType)a <attr2>, *(declrType)b <attr3>
        //  => <attr2> <attr1> int *a, <attr3> <attr1> int *b
        exprJoinDuplicatedAttr(regType, headType);
    } else {
        if(EXPR_HAS_GCCATTR(declr) || EXPR_HAS_GCCATTR(headType)) {
            compile_typeDesc1(headType, declrContext, NULL);
            regType = allocDerivedTypeDesc(headType);
            exprJoinDuplicatedAttr(regType, headType);
        } else {
            regType = headType;
        }
        EXPR_SET0(declr->e_nodes[0], regType);
    }

    exprJoinAttrToPre((CExpr*)regType, (CExpr*)declr);

    CExprOfTypeDesc *parentTd = NULL, *parentTdo;
    CExprOfTypeDesc *funcTd = getFuncType(regType, &parentTd);

    if(funcTd) {
        CExprOfTypeDesc *rtd = getRefTypeWithoutFix(EXPR_T(
            getRefTypeWithoutFix(funcTd)->e_typeExpr));
        if(rtd && (ETYP_IS_FUNC(rtd) || ETYP_IS_ARRAY(rtd)))
            addError((CExpr*)declr, CERR_123);

        if(declrContext == DC_IN_PARAMS &&
            ((parentTd == NULL) ||
            ((parentTdo = getRefTypeWithoutFix(parentTd)) &&
            ETYP_IS_POINTER(parentTdo) == 0))) {
            //complete function pointer
            //ex) f(int g(int)) -> f(int (*g)(int))
            CExprOfTypeDesc *ptrTd = allocPointerTypeDesc(funcTd);
            EXPR_REF(ptrTd);
            freeExpr((CExpr*)funcTd);
            exprCopyLineNum((CExpr*)ptrTd, (CExpr*)regType);
            if(parentTd == regType || parentTd == NULL) {
                declr->e_nodes[0] = (CExpr*)ptrTd;
                declrType = ptrTd;
            } else {
                parentTd->e_typeExpr = (CExpr*)ptrTd;
            }
        }
        // function of typedef name has no paramExpr
        if(ETYP_IS_FUNC(funcTd) && funcTd->e_paramExpr) {
            int as = (declrContext == DC_IN_FUNC_DEF);
            compile_params(funcTd->e_paramExpr, as);
        }
    }

    compile_typeDesc1(headType, declrContext, sym);

    if(regType != headType) {
        compile_typeDesc(regType, declrContext);
    }

    CExprOfTypeDesc *regType0 = getRefType(regType);

    if(regType0 == NULL || EXPR_ISERROR(regType0))
        return;

    int isVoid = (regType0->e_basicType == BT_VOID);
    int isNoMemDecl = (ETYP_IS_COMPOSITE(regType0) && regType0->e_isNoMemDecl);
    int isGlobal = getCurrentSymbolTable()->stb_isGlobal;

    if(isTypeDef == 0 && headType->e_sc.esc_isExtern == 0 &&
        regType->e_sc.esc_isExtern == 0 &&
        (isVoid || (isGlobal == 0 && isNoMemDecl))) {
        //incomplete struct/union, void
        addError((CExpr*)declr, CERR_107, symName);
    } else if(declrContext == DC_IN_MEMBER_DECL &&
        ETYP_IS_FUNC(regType0)) {
        //function in member
        addError((CExpr*)declr, CERR_109, symName);
    }

    exprFixAttr(attrTd ? attrTd : regType, declr, declrContext, isTypeDef);

    if(attrTd && hasGccAttr(attrTd, GAK_VAR|GAK_FUNC)) {
        //move GAK_VAR/GAK_FUNC attrs to regType.
        //when attrTd != NULL, regType is derived type.
        CExpr *attrs = getGccAttrs(
            EXPR_C(attrTd)->e_gccAttrPre, GAK_VAR|GAK_FUNC);
        if(attrs) {
            exprJoinAttrToPre((CExpr*)regType, attrs);
        }
    }

    if(regType != headType)
        exprFixAttr(regType, declr, declrContext, isTypeDef);

    if(EXPR_ISNULL(sym) == 0) {
        CSymbolTypeEnum st = ST_UNDEF;
        if(ETYP_IS_FUNC(regType0))
            regType->e_symbol = sym;

        if(isTypeDef) {
            st = ST_TYPE;
        } else if(declrType == NULL ||
            (ETYP_IS_FUNC(declrType) == 0 &&
            ETYP_IS_FUNC_OLDSTYLE(declrType) == 0)) {
            if(declrType && EXPR_ISNULL(declrType->e_paramExpr) == 0 &&
                EXPR_CODE(declrType->e_paramExpr) != EC_TYPE_DESC) {
                st = ST_FUNC;
            } else {
                if(varType == ST_VAR && declrType == NULL) {
                    CExprOfTypeDesc *td1 = resolveType((CExpr*)regType);
                    td1 = td1 ? getRefType(td1) : NULL;

                    if(td1 && ETYP_IS_FUNC(td1))
                        st = ST_FUNC; // typeof(func) f;
                }
                if(st == ST_UNDEF)
                    st = varType;
            }
        } else {
            st = ST_FUNC;
        }

        if(addSym && EXPR_ISERROR(regType) == 0) {
            const int fn = (declrContext == DC_IN_FUNC_DEF), rc = 1;
            EXPR_C(sym)->e_hasInit = EXPR_C(declr)->e_hasInit;

            if(EXPR_C(sym)->e_hasInit) {
                if(isTypeDef)
                    addError((CExpr*)declr, CERR_108, symName);
                else if(ETYP_IS_FUNC(regType0))
                    addError((CExpr*)declr, CERR_110, symName);
                else if(headType->e_sc.esc_isExtern)
                    addError((CExpr*)declr, CERR_111, symName);
                else if(ETYP_IS_COMPOSITE(regType0) && regType0->e_isNoMemDecl)
                    addError((CExpr*)declr, CERR_029, symName);
            }

            addSymbolAt(sym, (CExpr*)declr, NULL, regType, st, fn, rc);

            if(varType == ST_PARAM)
                sym->e_symType = varType;

            // set size/align to pre-declared symbol type
            // ex) extern int a[]; int a[2];
            //   ... set 2*sizeof(int) to array type of pre-declaraed a
            CExprOfTypeDesc *preDeclTd = regType->e_preDeclType;
            if(preDeclTd && preDeclTd->e_size == 0)
                ETYP_COPY_SIZE(preDeclTd, regType);
        }

        if(addSym == 0 && (varType == ST_PARAM || varType == ST_MEMBER)) {
            // st == ST_FUNC (&& varType == ST_PARAM)
            // means abbreviated pointer of function in parameter list
            assertExpr((CExpr*)sym, st == varType || st == ST_FUNC);
            sym->e_symType = varType;
        }
    }

    if(isTypeDef) {
        //renumbering order number to output sym after declrType
        reorderSymbol(sym);
    }
}


/**
 * \brief
 * compile declarator
 *
 * @param headType
 *      head type descriptor
 * @param attrTd
 *      type descriptor which has gcc attributes
 * @param declr
 *      declarator
 * @param varType
 *      symbol type to set to symbols in declarator
 * @param addSym
 *      set to 1 for adding symbol to symbol table
 */
PRIVATE_STATIC void
compile_declarator(CExprOfTypeDesc *headType,
    CExprOfTypeDesc *attrTd, CExprOfBinaryNode *declr,
    CSymbolTypeEnum varType, int addSym)
{
    compile_declarator0(headType, attrTd, declr,
        varType, addSym, DC_IN_ANY);
}


/**
 * \brief
 * compile declarator of function definition
 *
 * @param headType
 *      head type descriptor
 * @param declr
 *      declarator
 * @param addSym
 *      set to 1 for adding symbol to symbol table
 */
PRIVATE_STATIC void
compile_declaratorInFuncDef(CExprOfTypeDesc *headType,
    CExprOfBinaryNode *declr, int addSym)
{
    compile_declarator0(headType, NULL, declr,
        ST_UNDEF, addSym, DC_IN_FUNC_DEF);
}


/**
 * \brief
 * compile declarator of struct/union member declaration
 *
 * @param headType
 *      head type descriptor
 * @param attrTd
 *      type descriptor which has gcc attributes
 * @param declr
 *      declarator
 */
PRIVATE_STATIC void
compile_declaratorInMemberDecl(CExprOfTypeDesc *headType,
    CExprOfTypeDesc *attrTd, CExprOfBinaryNode *declr)
{
    compile_declarator0(headType, attrTd, declr,
        ST_MEMBER, 0, DC_IN_MEMBER_DECL);
}


/**
 * \brief
 * compile declarator of function parameter
 *
 * @param headType
 *      head type descriptor
 * @param declr
 *      declarator
 * @param addSym
 *      set to 1 for adding symbol to symbol table
 */
PRIVATE_STATIC void
compile_declaratorInParams(CExprOfTypeDesc *headType, CExprOfBinaryNode *declr,
    int addSym)
{
    compile_declarator0(headType, NULL, declr,
        ST_PARAM, addSym, DC_IN_PARAMS);
}


/**
 * \brief
 * compile declarator in EC_TYPENAME such as cast expression
 *
 * @param headType
 *      head type descriptor
 * @param declr
 *      declarator
 */
PRIVATE_STATIC void
compile_declaratorInTypeName(CExprOfTypeDesc *headType, CExprOfBinaryNode *declr)
{
    compile_declarator0(headType, NULL, declr,
        ST_UNDEF, 0, DC_IN_TYPENAME);
}


/**
 * \brief
 * sub function of existsTypeDesc()
 */
PRIVATE_STATIC int
existsTypeDesc0(CExprOfTypeDesc *td, CExprOfTypeDesc *stopTd)
{
    assert(td);

    if(td->e_isExist)
        return 1;
    if(stopTd == td || EXPR_ISERROR(td))
        return 0;

    if(stopTd == NULL)
        stopTd = td;
    CSymbolTableGroupEnum group = 0, exists = 0;
    CExprOfSymbol *sym = NULL;

    switch(td->e_tdKind) {
    case TD_BASICTYPE:
    case TD_GCC_BUILTIN:
    case TD_GCC_BUILTIN_ANY:
        exists = 1;
        break;
    case TD_POINTER:
    case TD_ARRAY:
    case TD_COARRAY:
    case TD_FUNC:
    case TD_FUNC_OLDSTYLE:
        if(td->e_typeExpr == NULL) {
            EXPR_ISERROR(td) = 1;
            return 0;
        } else {
            return existsTypeDesc0(EXPR_T(td->e_typeExpr), stopTd);
        }
    case TD_GCC_TYPEOF:
        return resolveType(td->e_typeExpr) != NULL;
    case TD_STRUCT:
    case TD_UNION:
    case TD_ENUM: {
            if(td->e_isNoMemDecl || td->e_isTypeDef)
                exists = 1;
            else {
                CExpr *body = getMemberDeclsExpr(td);
                if(body)
                    exists = 1;
                else {
                    group = STB_TAG;
                    sym = getTagSymbol(td);
                    exists = (EXPR_ISNULL(sym) == 0);
                }
            }
        }
        break;
    case TD_TYPEREF:
        group = STB_IDENT;
        sym = EXPR_SYMBOL(td->e_typeExpr);

        if(isGccBuiltinType(sym->e_symName)) {
            td->e_tdKind = TD_GCC_BUILTIN;
            exists = 1;
        }
        break;
    case TD_DERIVED:
        return existsTypeDesc0(td->e_refType, stopTd);
    case TD_UNDEF:
    case TD_END:
        ABORT();
    }

    if(exists) {
        td->e_isExist = 1;
        return 1;
    }

    if(EXPR_ISNULL(sym)) {
        EXPR_ISERROR(td) = 1;
        return 0;
    }

    CExprOfSymbol *tsym = findSymbolByGroup(sym->e_symName, group);

    if(tsym == NULL || EXPRS_TYPE(tsym) == NULL ||
        (group == STB_IDENT && tsym->e_symType != ST_TYPE)) {

        EXPR_ISERROR(td) = 1;
        return 0;
    }

    return existsTypeDesc0(EXPRS_TYPE(tsym), stopTd);
}


/**
 * \brief
 * judge td is already defined or builtin type
 *
 * @param td
 *      target type descriptor
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
existsTypeDesc(CExprOfTypeDesc *td)
{
    return existsTypeDesc0(td, NULL);
}


/**
 * \brief
 * get declarator type descriptor
 *
 * @param decl
 *      declarator
 * @return
 *      NULL or type descriptor
 */
PRIVATE_STATIC CExprOfTypeDesc*
getDeclrType(CExpr *decl)
{
    CExprOfTypeDesc *td = EXPR_T(EXPR_B(decl)->e_nodes[0]);
    if(td)
        return td;
    CExpr *initDecls = EXPR_B(decl)->e_nodes[1];
    if(EXPR_ISNULL(initDecls))
        return NULL;
    CExpr *initDecl = exprListHeadData(initDecls);
    if(EXPR_ISNULL(initDecl))
        return NULL;
    CExpr *declr = exprListHeadData(initDecl);
    td = EXPR_T(EXPR_B(declr)->e_nodes[0]);
    EXPR_REF(td);

    return td;
}


/**
 * \brief
 * if necessary, complete type descriptor of EC_DATA_DEF, EC_DECL, EC_FUNC_DEF node
 *
 * @param expr
 *      target node
 * @return
 *      completed type descriptor
 */
PRIVATE_STATIC CExprOfTypeDesc*
compile_noTypeCompletion(CExpr *expr)
{
    CExpr *etd = NULL;

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfBinaryNode:
        etd = (CExpr*)getDeclrType(expr);
        break;
    case STRUCT_CExprOfList:
        etd = exprListHeadData(expr);
        break;
    default:
        ABORT();
    }

    if(EXPR_ISNULL(etd) == 0) {
        return (CExprOfTypeDesc*)etd;
    }

    CExprOfTypeDesc *compType = allocIntTypeDesc();
    EXPR_REF(compType);

    if(etd) {
        exprJoinAttr((CExpr*)compType, etd);
        freeExpr(etd);
    }

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfBinaryNode:
        EXPR_B(expr)->e_nodes[0] = (CExpr*)compType;
        break;
    case STRUCT_CExprOfList:
        CCOL_DL_SET_DATA(exprListHead(expr), compType);
        break;
    default:
        ABORT();
    }

    addWarn((CExpr*)expr, CWRN_002);
    
    return compType;
}


/**
 * \brief
 * if td is struct/union/enum type and qualifed, copy td 
 *
 * @param td
 *      type descriptor
 * @param dataDefOrDecl
 *      EC_DATA_DEF/EC_DECL node which has td
 * @return
 *      NULL or duplicated node
 */
PRIVATE_STATIC CExprOfTypeDesc*
duplicateQualifiedTaggedType(CExprOfTypeDesc *td, CExprOfBinaryNode *dataDefOrDecl)
{
    if(ETYP_IS_TAGGED(td) == 0 ||
        (isTypeQualSet(td) == 0 && isScspecSet(td) == 0 &&
        (EXPR_L_ISNULL(td->e_typeExpr) ||
        EXPR_HAS_GCCATTR(exprListHeadData(td->e_typeExpr)) == 0)))
        return td;

    EXPR_UNREF(td);
    CExprOfTypeDesc *dtd = allocDerivedTypeDesc(td);
    dataDefOrDecl->e_nodes[0] = (CExpr*)dtd;
    EXPR_REF(dtd);
    compile_typeDesc(td, DC_IN_ANY);
    compile_typeDesc(dtd, DC_IN_ANY);

    return dtd;
}


/**
 * \brief
 * compile EC_DATA_DEF, EC_DECL node
 *
 * @param dataDef
 *      target node
 * @param parent
 *      parent node
 */
PRIVATE_STATIC void
compile_dataDefOrDecl(CExpr *dataDef, CExpr *parent)
{
    assert(dataDef);
    assertExpr(dataDef, EXPR_CODE(dataDef) == EC_DATA_DEF || EXPR_CODE(dataDef) == EC_DECL);

    CExpr *initDecls = EXPR_B(dataDef)->e_nodes[1];
    CExprOfTypeDesc *td = compile_noTypeCompletion(dataDef);
    if(EXPR_ISERROR(td) || EXPR_ISERROR(dataDef))
        return;

    if(EXPR_CODE(td) != EC_TYPE_DESC ||
        (initDecls && EXPR_CODE(initDecls) == EC_FUNC_DEF)) {

        /* EC_FUNC_DEF means nested func */
        EXPR_ISGCCSYNTAX(initDecls) = 1;
        CExpr *head = exprListHeadData(initDecls);
        assertExpr(head, EXPR_ISNULL(head));
        freeExpr(head);
        CCOL_DL_SET_DATA(exprListHead(initDecls), td);
        EXPR_B(dataDef)->e_nodes[0] = NULL;

        compile_funcDef(initDecls, parent);
        return;
    }

    if(ETYP_IS_GCC_TYPEOF(td)) {
        compile1(td->e_typeExpr, (CExpr*)td);
    }

    int isDerivedCompos = 0;
    CExprOfTypeDesc *td0 = td;
    td = duplicateQualifiedTaggedType(td, EXPR_B(dataDef));
    isDerivedCompos = (td != td0);

    if(initDecls == NULL) {
        if(td->e_isTypeDef) {
            addError((CExpr*)td, CERR_019);
        } else {
            compile_typeDesc(td, DC_IN_ANY);
            exprFixAttr(td0, NULL, DC_IN_ANY, 1);
            if(isDerivedCompos == 0 && ETYP_IS_TAGGED(td) == 0) {
                addError((CExpr*)dataDef, CERR_020);
                EXPR_ISERROR(td) = 1;
            }
        }
    } else {

        CCOL_DListNode *ite;
        CExprOfTypeDesc *attrTd = ((td == td0) ? NULL : td0);

        EXPR_FOREACH(ite, initDecls) {
            CExpr *initDecl = EXPR_L_DATA(ite);
            assertExprCode(initDecl, EC_INIT_DECL);
            
            CExpr *declr = exprListHeadData(initDecl);
            assertExpr(initDecl, declr);
            assertExprCode(declr, EC_DECLARATOR);

            if(existsTypeDesc(td0) == 0) {
                if(td0->e_typeExpr && EXPR_CODE(td0->e_typeExpr) == EC_IDENT)
                    addError((CExpr*)td0, CERR_021,
                        EXPR_SYMBOL(td0->e_typeExpr)->e_symName);
                else
                    addError((CExpr*)td0, CERR_022);
                EXPR_ISERROR(initDecl) = 1;
                break;
            } else {
                const int addSym = 1;
                CSymbolTypeEnum st = td0->e_isTypeDef ? ST_TYPE : ST_VAR;
                compile_declarator(td, attrTd, EXPR_B(declr), st, addSym);
                CExpr *declType = EXPR_B(declr)->e_nodes[0];
                if(EXPR_ISNULL(declType) == 0) {
                    assertExprCode(declType, EC_TYPE_DESC);
                    compile1(EXPR_T(declType)->e_typeExpr, declType);
                }

                compile1(initDecl, initDecls);
            }
        }

        if(EXPR_ISERROR(td) == 0 && td->e_isTypeDef == 0) {
            freeExpr((CExpr*)td);
            EXPR_B(dataDef)->e_nodes[0] = NULL;
        }
    }
}


/**
 * \brief
 * judge expr is statement which has block scope
 *
 * @param expr
 *      target node
 * @return 
 *      0:no, 1:yes
 */
int
isScopedStmt(CExpr *expr)
{
    if(expr == NULL)
        return 0;

    switch(EXPR_CODE(expr)) {
    case EC_COMP_STMT:
    case EC_IF_STMT:
    case EC_FOR_STMT:
    case EC_WHILE_STMT:
    case EC_DO_STMT:
    case EC_SWITCH_STMT:
        return 1;
    default:
        return 0;
    }
}


/**
 * \brief
 * add label symbols to symbol table in function body
 *
 * @param expr
 *      target node
 * @param funcBodySymTab
 *      function body's symbol table
 */
PRIVATE_STATIC void
addLabelSymbols(CExpr *expr, CSymbolTable *funcBodySymTab)
{
    if(expr == NULL)
        return;

    const int fn = 0, rc = 1;
    int scoped = 0;

    switch(EXPR_CODE(expr)) {
    case EC_LABEL: {
            CExprOfSymbol *sym = EXPR_SYMBOL(EXPR_U(expr)->e_node);
            CExprOfSymbol *hsym = findSymbolByGroup(sym->e_symName, STB_LABEL);
            int isGccLabel = (hsym && hsym->e_isGccLabelDecl);
            CSymbolTypeEnum st = isGccLabel ? ST_GCC_LABEL : ST_LABEL;
            CSymbolTable *symTab = isGccLabel ? NULL : funcBodySymTab;
            addSymbolAt(sym, NULL, symTab, NULL, st, fn, rc);
            return;
        }
    case EC_GCC_LABEL_DECLS: {
            CCOL_DListNode *ite, *ite1;
            EXPR_FOREACH(ite, expr) {
                CExpr *syms = EXPR_L_DATA(ite);
                EXPR_FOREACH(ite1, syms) {
                    CExprOfSymbol *sym = EXPR_SYMBOL(EXPR_L_DATA(ite1));
                    sym->e_isGccLabelDecl = 1;
                    EXPR_ISGCCSYNTAX(sym) = 1;
                    addSymbolAt(sym, NULL, NULL, NULL, ST_GCC_LABEL, fn, rc);
                }
            }
            return;
        }
    case EC_FUNC_DEF: //nested function is out of scope
        return;
    default:
        if(isScopedStmt(expr) && EXPR_C(expr)->e_symTab == NULL) {
            scoped = 1;
            pushSymbolTableToExpr(expr);
        }
        break;
    }

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr)
        addLabelSymbols(ite.node, funcBodySymTab);

    if(scoped)
        popSymbolTable();
}


/**
 * \brief
 * compile EC_FUNC_DEF node
 *
 * @param funcDef
 *      target node
 * @param parent
 *      parent node
 */
PRIVATE_STATIC void
compile_funcDef(CExpr *funcDef, CExpr *parent)
{
    assertExpr(parent, funcDef);
    assertExprCode(funcDef, EC_FUNC_DEF);

    if(EXPR_ISCOMPILED(funcDef))
        return;
    EXPR_ISCOMPILED(funcDef) = 1;

    CExprOfTypeDesc *td;
    int nested = (EXPR_CODE(parent) == EC_DECL);
    
    if(nested)
        td = EXPR_T(EXPR_B(parent)->e_nodes[0]);
    else
        td = compile_noTypeCompletion(funcDef);

    assertExprCode((CExpr*)td, EC_TYPE_DESC);

    CExprOfBinaryNode *declr = EXPR_B(exprListNextData(funcDef));
    const int addSym = 1;
    pushSymbolTableToExpr(funcDef);
    compile_declaratorInFuncDef(EXPR_T(td), declr, addSym);

    CExprOfTypeDesc *declrType = EXPR_T(declr->e_nodes[0]);
    assertExpr(funcDef, declrType);
    CExpr *params = declrType->e_paramExpr;

    compile_params(params, addSym);

    CExpr *head;
    EXPR_SET0(head, exprListRemoveHead(funcDef));
    freeExpr(head);

    if(nested)
        EXPR_SET(EXPR_B(parent)->e_nodes[0], NULL);

    CCOL_DListNode *ite = exprListNext(funcDef);
    EXPR_FOREACH_FROM(ite, funcDef) {
        CExpr *node = EXPR_L_DATA(ite);
        compile1(node, funcDef);
    }

    popSymbolTable();
}


/**
 * \brief
 * alloc symbol for anonymous member
 *
 * @param memDecls
 *      EC_MEMBER_DECLS node
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfSymbol*
createAnonymousMemberSymbol(CExpr *memDecls)
{
    int anonCount = s_anonymousMemberNameSeq++;
    char buf[128], *name;
    sprintf(buf, "%s%d", s_anonymousMemberPrefix, anonCount);
    name = ccol_strdup(buf, 128);
    CExprOfSymbol *sym = allocExprOfSymbol(EC_IDENT, name);
    exprCopyLineNum((CExpr*)sym, memDecls);
    sym->e_symType = ST_MEMBER;
    return sym;
}


/**
 * \brief
 * compile EC_MEMBER_DECL node
 *
 * @param memDecl
 *      target node
 * @param memDecls
 *      parent node
 */
PRIVATE_STATIC void
compile_memberDecl(CExpr *memDecl, CExpr *memDecls)
{
    assertExpr(memDecls, memDecl);
    assertExprCode(memDecl, EC_MEMBER_DECL);

    if(EXPR_ISCOMPILED(memDecl))
        return;
    EXPR_ISCOMPILED(memDecl) = 1;

    //td is not compiled yet
    CExprOfTypeDesc *td = EXPR_T(EXPR_B(memDecl)->e_nodes[0]);
    assertExprCode((CExpr*)td, EC_TYPE_DESC);
    CExprOfTypeDesc *tdo = getRefTypeWithoutFix(td);

    if(EXPR_ISERROR(td)) {
        EXPR_ISERROR(memDecl) = 1;
        return;
    }

    CExpr *mems = EXPR_B(memDecl)->e_nodes[1];

    if(EXPR_ISNULL(mems)) {
        //complete declarator for anonymous struct or union member
        td->e_isAnonMember = 1;
        CExpr *sym = (CExpr*)createAnonymousMemberSymbol(memDecls);
        CExpr *declr = (CExpr*)allocExprOfBinaryNode1(
            EC_DECLARATOR, (CExpr*)td, sym);
        exprCopyLineNum(declr, memDecl);
        CExpr *memDeclr = (CExpr*)allocExprOfBinaryNode1(
            EC_MEMBER_DECLARATOR, declr, NULL);
        exprCopyLineNum(memDeclr, memDecl);
        mems = exprList1(EC_MEMBERS, memDeclr);
        exprCopyLineNum(mems, memDecl);
        EXPR_REF(mems);
        freeExpr(EXPR_B(memDecl)->e_nodes[1]);
        EXPR_B(memDecl)->e_nodes[1] = mems;

    } else {
        CCOL_DListNode *ite;
        int multiMems = (EXPR_L_SIZE(mems) > 1);
        int dupCount = 0;

        EXPR_FOREACH(ite, mems) {
            CExprOfBinaryNode *memDeclr = EXPR_B(EXPR_L_DATA(ite));
            CExprOfBinaryNode *declr = EXPR_B(memDeclr->e_nodes[0]);
            CExpr *bitLenExpr = memDeclr->e_nodes[1];
            compile1(bitLenExpr, (CExpr*)memDeclr);
            int anon = 0, duplicated = 0;

            if(EXPR_ISNULL(declr)) {
                //complete declarator for anonymous member
                anon = 1;
                declr = allocExprOfBinaryNode1(EC_DECLARATOR, (CExpr*)td, NULL);
                exprCopyLineNum((CExpr*)declr, (CExpr*)memDeclr);
                EXPR_REF(declr);
                freeExpr(memDeclr->e_nodes[0]);
                memDeclr->e_nodes[0] = (CExpr*)declr;
            }

            CExprOfTypeDesc *td0 = td;
            td = duplicateQualifiedTaggedType(td, EXPR_B(memDecl));

            compile_declaratorInMemberDecl(td, (td == td0) ? NULL : td0, declr);

            CExprOfTypeDesc *fieldTd = EXPR_T(declr->e_nodes[0]);
            compile_typeDesc(fieldTd, DC_IN_MEMBER_DECL);

            if(multiMems && exprHasGccAttr((CExpr*)td)) {
                CExprOfTypeDesc *dupFieldTd = NULL;
                if(ETYP_IS_TAGGED(tdo)) {
                    dupFieldTd = allocExprOfTypeDesc();
                    dupFieldTd->e_tdKind = TD_DERIVED;
                    dupFieldTd->e_refType = fieldTd;
                } else {
                    //gcc attr list is duplicated
                    dupFieldTd = duplicateExprOfTypeDesc(fieldTd);
                }
                EXPR_REF(dupFieldTd);
                declr->e_nodes[0] = (CExpr*)dupFieldTd;
                freeExpr((CExpr*)fieldTd);
                fieldTd = dupFieldTd;
                duplicated = 1;
                compile_typeDesc(fieldTd, DC_IN_MEMBER_DECL);
            }

            // check bit field
            if(EXPR_ISNULL(bitLenExpr) == 0) {
                if(multiMems && duplicated == 0) {
                    CExprOfTypeDesc *dupFieldTd =
                        duplicateExprOfTypeDesc(fieldTd);
                    EXPR_REF(dupFieldTd);
                    freeExpr(dupFieldTd->e_bitLenExpr);
                    dupFieldTd->e_bitLenExpr = NULL;
                    declr->e_nodes[0] = (CExpr*)dupFieldTd;
                    freeExpr((CExpr*)fieldTd);
                    fieldTd = dupFieldTd;
                    ++dupCount;
                }

                EXPR_REF(bitLenExpr);
                fieldTd->e_bitLenExpr = bitLenExpr;
                int l = -1;
                CExprOfTypeDesc *fieldTdo = getRefType(fieldTd);

                if(isIntegerType(fieldTdo) == 0) {
                    addError((CExpr*)td, CERR_106);
                    EXPR_ISERROR(fieldTd) = 1;
                } else {
                    CNumValueWithType n;
                    if(getConstNumValue(bitLenExpr, &n)) {
                        if(n.nvt_isConstButMutable)
                            l = -1;
                        else
                            l = (int)n.nvt_numValue.ll;
                    } else {
                        l = -1;
                        if(n.nvt_isConstButUnreducable == 0) {
                            addError(bitLenExpr, CERR_093);
                            EXPR_ISERROR(fieldTd) = 1;
                        }
                    }

                    if(l > getTypeSize(fieldTdo) * CHAR_BIT) {
                        addError((CExpr*)td, CERR_048);
                        EXPR_ISERROR(td) = 1;
                    }
                }

                fieldTd->e_bitLen = l;

                freeExpr(bitLenExpr);
                memDeclr->e_nodes[1] = NULL;
            }

            if(anon) {
                // create name for anonymous member
                assert(EXPR_ISNULL(declr->e_nodes[1]));
                freeExpr(declr->e_nodes[1]);
                fieldTd->e_isAnonMember = 1;
                CExpr *sym = (CExpr*)createAnonymousMemberSymbol(memDecls);
                EXPR_REF(sym);
                declr->e_nodes[1] = sym;
                exprJoinAttrToPre((CExpr*)fieldTd, (CExpr*)memDeclr);
                exprFixAttr(fieldTd, declr, DC_IN_MEMBER_DECL, 0);
            }
        }

        if(dupCount >= EXPR_L_SIZE(mems)) {
            td->e_isUsed = 0;
        }
    }

    //nested struct/union declaration
    if(ETYP_IS_COMPOSITE(td)) {
        CExprOfSymbol *tag = getTagSymbol(td);
        if(EXPR_ISNULL(tag) == 0) {
            CExprOfSymbol *tsym = findSymbolByGroup(tag->e_symName, STB_TAG);
            if(tsym == NULL) {
                const int fn = 0, rc = 1;
                addSymbolAt(tag, td->e_typeExpr, NULL, td, ST_TAG, fn, rc);
            }
        }
    }

    compile_typeDesc(td, DC_IN_MEMBER_DECL);
}


/**
 * \brief
 * compile EC_TYPENAME
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 */
PRIVATE_STATIC CExpr*
compile_typeName(CExpr *expr, CExpr *parent)
{
    assert(expr);
    assertExprCode(expr, EC_TYPENAME);

    CExprOfTypeDesc *td = EXPR_T(EXPR_B(expr)->e_nodes[0]);
    //avoid free memory at constant folding
    assertExpr(expr, td);
    CExprOfBinaryNode *declr = EXPR_B(EXPR_B(expr)->e_nodes[1]);
    CExpr *ntd;

    if(declr == NULL) {
        ntd = (CExpr*)td;
        compile_typeDesc(td, DC_IN_TYPENAME);
        exprFixAttr(td, NULL, DC_IN_TYPENAME, 0);
    } else {
        compile_declaratorInTypeName(td, declr);
        ntd = declr->e_nodes[0];
        assertExpr((CExpr*)declr, ntd);
    }

    EXPR_REF(ntd);
    exprReplace(parent, expr, ntd);

    return ntd;
}


/**
 * \brief
 * compile EC_XMP_COARRAY_DECLARATION node
 *
 * @param caDeclr
 *      target node
 */
PRIVATE_STATIC void
compile_coarrayDeclaration(CExprOfBinaryNode *caDeclr)
{
    CExpr *dims = caDeclr->e_nodes[1];
    CExprOfSymbol *sym = EXPR_SYMBOL(caDeclr->e_nodes[0]);
    CExprOfSymbol *sym1 = findSymbolByGroup(sym->e_symName, STB_IDENT);

    if(sym1 == NULL) {
        addError((CExpr*)sym, CERR_023, sym->e_symName);
        return;
    }

    int isGlobalVar = (sym1->e_symType == ST_VAR && sym1->e_isGlobal);
    int isParam = (sym1->e_symType == ST_PARAM);

    if(isGlobalVar == 0 && isParam == 0) {
        addError((CExpr*)sym, CERR_024, sym->e_symName);
        return;
    }

    CExprOfBinaryNode *declr = EXPR_B(sym1->e_declrExpr);

    if(declr == NULL) {
        addFatal((CExpr*)sym, CFTL_001, sym->e_symName);
        return;
    }

    CExpr *typeExpr = declr->e_nodes[0];
    CCOL_DListNode *ite;

    EXPR_FOREACH_REVERSE(ite, dims) {
        CExprOfTypeDesc *td = allocExprOfTypeDesc();
        EXPR_REF(td);
        td->e_tdKind = TD_COARRAY;
        td->e_typeExpr = typeExpr;
        CExpr *dim = EXPR_L_DATA(ite);
        if(EXPR_CODE(dim) == EC_FLEXIBLE_STAR)
            td->e_len.eln_isVariable = 1;
        else {
            td->e_len.eln_lenExpr = dim;
            EXPR_REF(dim);
        }
        typeExpr = (CExpr*)td;
    }

    declr->e_nodes[0] = typeExpr;
    exprSetExprsType((CExpr*)sym1, EXPR_T(typeExpr));
}


/**
 * \brief
 * compile EC_XMP_COARRAY_DECLARATIONS node
 *
 * @param caDeclrs
 *      target node
 * @param parent
 *      parent node
 */
PRIVATE_STATIC void
compile_coarrayDeclarations(CExpr *caDeclrs, CExpr *parent)
{
    if(EXPR_ISCOMPILED(caDeclrs))
        return;
    EXPR_ISCOMPILED(caDeclrs) = 1;

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, caDeclrs) {
        CExpr *caDeclr = EXPR_L_DATA(ite);
        compile_coarrayDeclaration(EXPR_B(caDeclr));
    }
}


/**
 * \brief
 * compile EC_IDENT node
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @return
 *      EC_XMP_COARRAY_REF node to replace expr
 *
 */
PRIVATE_STATIC CExpr*
compile_completeIdent(CExpr *ident, CExpr *parent)
{
    if(parent && EXPR_CODE(parent) == EC_XMP_COARRAY_REF)
        return ident;

    if(EXPR_ISCOMPILED(ident))
        return ident;

    EXPR_ISCOMPILED(ident) = 1;
    CExprOfSymbol *sym = findSymbolByGroup(
        EXPR_SYMBOL(ident)->e_symName, STB_IDENT);

    if(sym == NULL)
        return ident;

    CExprOfTypeDesc *td = EXPRS_TYPE(sym);
    if(td == NULL || ETYP_IS_COARRAY(td) == 0)
        return ident;

    assert(parent);
    CExpr *coDims = exprList(EC_XMP_COARRAY_DIMENSIONS);
    CExpr *coRef = exprCoarrayRef(ident, coDims);
    EXPR_REF(coRef);

    exprReplace(parent, ident, coRef);

    return coRef;
}


/**
 * \brief
 * compile EC_LABEL node's parent
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 *
 * @return
 *
 */
PRIVATE_STATIC void
compile_labelParent(CExpr *expr, CExpr *parent)
{
    CExpr *node = EXPR_U(expr)->e_node;
    if(EXPR_CODE(node) == EC_IDENT) {
        CExprOfSymbol *sym = EXPR_SYMBOL(node);
        CExprOfSymbol *hsym = findSymbolByGroup(sym->e_symName, STB_LABEL);
        if(hsym == NULL || (hsym->e_symType != ST_LABEL &&
            hsym->e_symType != ST_GCC_LABEL)) {
            addError((CExpr*)sym, CERR_025, sym->e_symName);
        } else
            sym->e_symType = hsym->e_symType;
    } else {
        compile1(node, expr);
    }
}


/**
 * \brief
 * compile #pragma pack
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @return
 *      null node to replace expr
 */
PRIVATE_STATIC CExpr*
compile_pragmaPack(CExpr *expr, CExpr *parent)
{
    CExpr *e = EXPR_U(expr)->e_node;

    if(EXPR_ISNULL(e)) {
        s_pragmaPackEnabled = 0;
    } else {
        s_pragmaPackAlign = (int)getNumberConstAsLong(e);
        s_pragmaPackEnabled = (s_pragmaPackAlign > 0);
    }

    CExpr *nullExpr = exprNull();
    exprReplace(parent, expr, nullExpr);

    return nullExpr;
}


/**
 * \brief
 * validate EC_DECLARATOR node or add error
 *
 * @param declr
 *      target node
 */
PRIVATE_STATIC void
checkDeclarator(CExpr *declr)
{
    CExprOfSymbol *sym = EXPR_SYMBOL(EXPR_B(declr)->e_nodes[1]);
    if(sym && sym->e_symType == ST_VAR) {
        CExprOfTypeDesc *td = EXPR_T(EXPR_B(declr)->e_nodes[0]);
        CExprOfTypeDesc *tdo =
            getRefType(EXPR_T(EXPR_B(declr)->e_nodes[0]));

        if(td->e_sc.esc_isExtern == 0 &&
            (ETYP_IS_VOID(tdo) ||
            (ETYP_IS_TAGGED(tdo) && tdo->e_isNoMemDecl))) {
            addError((CExpr*)declr, CERR_028, sym->e_symName);
            return;
        }
    }
}


/**
 * \brief
 * validate EC_SWITCH_STMT node or add error
 *
 * @param switchStmt
 *      target node
 */
PRIVATE_STATIC void
checkSwitchStmt(CExpr *switchStmt)
{
    assertExprCode(switchStmt, EC_SWITCH_STMT);

    CExpr *cond = EXPR_B(switchStmt)->e_nodes[0];
    CExprOfTypeDesc *td = EXPRS_TYPE(cond);
    if(td == NULL)
        return;
    if(isIntegerType(td) == 0) {
        addError(cond, CERR_113);
    }

    CCOL_SList values;
    CCOL_SListNode *site;
    memset(&values, 0, sizeof(values));
    CExpr *body = EXPR_B(switchStmt)->e_nodes[1];
    if(EXPR_ISNULL(body))
        return;

    int hasDefault = 0;
    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, body) {
        CExpr *e = EXPR_L_DATA(ite);
        switch(EXPR_CODE(e)) {
        case EC_DEFAULT_LABEL:
            if(hasDefault) {
                addError(e, CERR_115);
            } else {
                hasDefault = 1;
            }
            break;
        case EC_CASE_LABEL: {
                CExpr *cv1 = EXPR_B(e)->e_nodes[0];
                CExpr *cv2 = EXPR_B(e)->e_nodes[1];
                int has2nd = (EXPR_ISNULL(cv2) == 0);

                if(isConstExpr(cv1, 0) == 0 ||
                    (has2nd && isConstExpr(cv2, 0) == 0)) {
                    addError(e, CERR_026);
                    continue;
                }

                CNumValueWithType n1, n2;
                long long i1, i2;
                int isConst1 = getConstNumValue(cv1, &n1);
                int isConst2 = 0;
                if(has2nd)
                    isConst2 = getConstNumValue(cv2, &n2);

                if((isConst1 == 0 && n1.nvt_isConstButUnreducable) ||
                    (has2nd && isConst2 == 0 && n2.nvt_isConstButUnreducable))
                    continue;

                if(isConst1 == 0 || (has2nd && isConst2 == 0)) {
                    //const check must be done
                    addFatal(cv1, CFTL_008);
                    return;
                }

                i1 = getCastedLongLongValue(&n1);

                if(has2nd) {
                    i2 = getCastedLongLongValue(&n2);

                    if(i2 < i1) {
                        addError(e, CERR_027);
                        continue;
                    }
                }

                CCOL_SListNode *site;
                int err = 0;
                CCOL_SL_FOREACH(site, &values) {
                    long long *p = (long long*)CCOL_SL_DATA(site);
                    if(*p == i1) {
                        addError(cv1, CERR_114);
                        err = 1;
                        break;
                    }
                }
                if(err)
                    continue;

                long long *p = malloc(sizeof(i1));
                *p = i1;
                CCOL_SL_CONS(&values, p);
            }
            break;
        default:
            break;
        }
    }

    CCOL_SL_FOREACH(site, &values) {
        free(CCOL_SL_DATA(site));
    }

    CCOL_SL_CLEAR(&values);
}


/**
 * \brief
 * validate EC_DEFAULT_LABEL, EC_CASE_LABEL or add error
 *
 * @param lbl
 *      target node
 */
PRIVATE_STATIC void
checkDefaultCaseLabel(CExpr *lbl)
{
    CExprCodeEnum ec = EXPR_CODE(lbl);
    assertExpr(lbl, ec == EC_DEFAULT_LABEL || ec == EC_CASE_LABEL);

    if(isExprCodeChildStmtOf(lbl, EC_SWITCH_STMT, EXPR_PARENT(lbl), NULL) == 0) {
        addWarn(lbl, (ec == EC_DEFAULT_LABEL ? CWRN_013 : CWRN_014));
    }
}


/**
 * \brief
 * judge lbl is under loop statement or switch statement
 *
 * @param lbl
 *      target node
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isUnderLoopOrSwitch(CExpr *lbl)
{
    CExpr *parent = EXPR_PARENT(lbl);
    if(parent == NULL)
        return 0;
    CExprCodeEnum pec = EXPR_CODE(parent);

    switch(pec) {
    case EC_FOR_STMT:
    case EC_DO_STMT:
    case EC_WHILE_STMT:
    case EC_SWITCH_STMT:
        return 1;
    case EC_EXT_DEFS:
    case EC_FUNC_DEF:
    case EC_GCC_COMP_STMT_EXPR:
        break;
    default:
        return isUnderLoopOrSwitch(parent);
    }
    return 0;
}


/**
 * \brief
 * judge lbl is under loop statement
 *
 * @param lbl
 *      target node
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isUnderLoop(CExpr *lbl)
{
    CExpr *parent = EXPR_PARENT(lbl);
    if(parent == NULL)
        return 0;
    CExprCodeEnum pec = EXPR_CODE(parent);

    switch(pec) {
    case EC_FOR_STMT:
    case EC_DO_STMT:
    case EC_WHILE_STMT:
        return 1;
    case EC_EXT_DEFS:
    case EC_FUNC_DEF:
    case EC_GCC_COMP_STMT_EXPR:
        break;
    default:
        return isUnderLoop(parent);
    }
    return 0;
}


/**
 * \brief
 * validate EC_BREAK_STMT node or add error
 *
 * @param stmt
 *      target node
 */
PRIVATE_STATIC void
checkBreakStmt(CExpr *stmt)
{
    assertExprCode(stmt, EC_BREAK_STMT);

    if(isUnderLoopOrSwitch(stmt) == 0)
        addError(stmt, CERR_117);
}


/**
 * \brief
 * validate EC_CONTINUE_STMT node or add error
 *
 * @param stmt
 *      target node
 */
PRIVATE_STATIC void
checkContinueStmt(CExpr *stmt)
{
    assertExprCode(stmt, EC_CONTINUE_STMT);

    if(isUnderLoop(stmt) == 0)
        addError(stmt, CERR_125);
}


/**
 * \brief
 * validate EC_RETURN_STMT node or add error
 *
 * @param stmt
 *      target node
 */
PRIVATE_STATIC void
checkReturnStmt(CExpr *stmt)
{
    assertExprCode(stmt, EC_RETURN_STMT);

    CExprOfTypeDesc *ftd = EXPR_T(CCOL_SL_DATA(CCOL_SL_HEAD(
                        &s_compileContext.cc_funcDefTypes)));

    if(ftd == NULL)
        return;
    CExprOfTypeDesc *td1 = EXPR_T(ftd->e_typeExpr);
    CExpr *val = EXPR_U(stmt)->e_node;

    CExprOfTypeDesc *td1o = getRefType(td1);
    if(ETYP_IS_VOID(td1o)) {
        if(EXPR_ISNULL(val) == 0) {
            CExprOfTypeDesc *td2o = EXPRS_TYPE(val) ?
                getRefType(EXPRS_TYPE(val)) : NULL;
            if(td2o && ETYP_IS_VOID(td2o) == 0)
                addWarn(stmt, CWRN_017);
        }
        return;
    } else if(EXPR_ISNULL(val)) {
        addWarn(stmt, CWRN_018);
    }

    if(EXPR_ISNULL(val))
        return;

    CExprOfTypeDesc *td2 = EXPRS_TYPE(val);
    if(td2 == NULL)
        return;

    if(getPriorityTypeForAssign(td1, td2) == NULL) {
        addError(stmt, CERR_128);
    } else if(ETYP_IS_ARRAY(td2) == 0 && //treat array as pointer
        ETYP_IS_UNKNOWN_SIZE(td2)) {
        addError(stmt, CERR_130);
    }
}


/**
 * \brief
 * check that e can be logical expression or add error
 *
 * @param e
 *      target node
 */
PRIVATE_STATIC void
checkLogicalExpr(CExpr *e)
{
    if(isLogicalExpr(e) == 0)
        addError(e, CERR_126);
}


/**
 * \brief
 * compile node
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 */
void
compile1(CExpr *expr, CExpr *parent)
{
    if(EXPR_ISNULL(expr) ||
        EXPR_ISCOMPILED(expr) ||
        EXPR_ISERROR(expr))
        return;

    int scoped = 0;

    if(parent) {
        switch(EXPR_CODE(parent)) {
        case EC_FUNC_DEF:
            switch(EXPR_CODE(expr)) {
                case EC_DECLARATOR:
                    goto end;
                default:
                    break;
            }
            break;
        default:
            break;
        }
    }

#ifdef CEXPR_DEBUG_GCCATTR
    startCheckGccAttr(expr);
#endif

    switch(EXPR_CODE(expr)) {
    case EC_TYPENAME:
        expr = compile_typeName(expr, parent);
        goto end;
    case EC_DATA_DEF:
    case EC_DECL:
        compile_dataDefOrDecl(expr, parent);
        goto end;
    case EC_FUNC_DEF:
        compile_funcDef(expr, parent);
        goto end;
    case EC_MEMBER_DECL:
        compile_memberDecl(expr, parent);
        goto end;
    case EC_MEMBER_REF:
    case EC_POINTS_AT:
        compile1(EXPR_B(expr)->e_nodes[0], expr);
        goto end;
    case EC_XMP_COARRAY_DECLARATIONS:
        compile_coarrayDeclarations(expr, parent);
        goto end;
    case EC_IDENT:
        expr = compile_completeIdent(expr, parent);
        goto end;
    case EC_GCC_LABEL_ADDR:
    case EC_GOTO_STMT:
        compile_labelParent(expr, parent);
        goto end;
    case EC_ARRAY_REF:
        if(isSubArrayRef(expr))
            goto end; // subarray must not be compiled children firstly
        break;
    case EC_PRAGMA_PACK:
        expr = compile_pragmaPack(expr, parent);
        goto end;

    default:
        if(isScopedStmt(expr)) {
            if(parent && EXPR_CODE(parent) == EC_FUNC_DEF) {
                CSymbolTable *symTab = EXPR_C(parent)->e_symTab;
                assert(symTab);
                symTab->stb_isFuncDefBody = 1;

                CExprOfBinaryNode *funcDeclr = EXPR_B(exprListHeadData(parent));
                CCOL_SL_CONS(&s_compileContext.cc_funcDefTypes,
                    funcDeclr->e_nodes[0]);

                EXPR_C(expr)->e_symTab = symTab;
                addLabelSymbols(expr, symTab);
            } else {
                scoped = 1;
                pushSymbolTableToExpr(expr);
            }
        }
        break;
    }

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr)
        compile1(ite.node, expr);

 end:

    if(EXPR_ISERROR(expr) == 0)
        resolveType(expr);

    if(EXPR_ISERROR(expr) == 0) {
        //miscellaneous check
        switch(EXPR_CODE(expr)) {
        case EC_DECLARATOR:
            checkDeclarator(expr);
            break;
        case EC_IF_STMT:
            checkLogicalExpr(exprListHeadData(expr));
            break;
        case EC_WHILE_STMT:
            checkLogicalExpr(EXPR_B(expr)->e_nodes[0]);
            break;
        case EC_DO_STMT:
            checkLogicalExpr(EXPR_B(expr)->e_nodes[1]);
            break;
        case EC_FOR_STMT: {
                CExpr *cond = exprListNextNData(expr, 1);
                if(EXPR_ISNULL(cond) == 0)
                    checkLogicalExpr(cond);
            }
            break;
        case EC_SWITCH_STMT:
            checkSwitchStmt(expr);
            break;
        case EC_DEFAULT_LABEL:
        case EC_CASE_LABEL:
            checkDefaultCaseLabel(expr);
            break;
        case EC_BREAK_STMT:
            checkBreakStmt(expr);
            break;
        case EC_CONTINUE_STMT:
            checkContinueStmt(expr);
            break;
        case EC_RETURN_STMT:
            checkReturnStmt(expr);
            break;
        default:
            break;
        }
    }

    if(scoped) {
        popSymbolTable();
    }

    CSymbolTable *symTab = EXPR_C(expr)->e_symTab;
    if(symTab && symTab->stb_isFuncDefBody) {
        CCOL_SL_REMOVE_HEAD(&s_compileContext.cc_funcDefTypes);
    }

#ifdef CEXPR_DEBUG_GCCATTR
    endCheckGccAttr(expr);
#endif
    EXPR_ISCOMPILED(expr) = 1;
}


/**
 * \brief
 * sub function of collectTypeDesc()
 */
PRIVATE_STATIC void
collectTypeDesc0(CExpr *expr)
{
    if(expr == NULL) {
        return;
    } else if(EXPR_CODE(expr) == EC_TYPE_DESC) {
        if(EXPR_T(expr)->e_isMarked)
            return;
    } else {
        if(EXPR_CODE(expr) != EC_TYPE_DESC) {
            collectTypeDesc0((CExpr*)EXPRS_TYPE(expr));
            if(EXPR_CODE(expr) == EC_IDENT) {
                collectTypeDesc0(EXPR_SYMBOL(expr)->e_valueExpr);
            } else {
                CExprIterator ite;
                EXPR_FOREACH_MULTI(ite, expr)
                    collectTypeDesc0(ite.node);
            }
            return;
        }
    }

    //defined types which are not used in expression have
    //not been called resolveType.
    CExprOfTypeDesc *td = resolveType(expr);
    if(td == NULL || EXPR_ISERROR(td))
        return;
    td->e_isMarked = 1;

    addTypeDesc(td);

    if(ETYP_IS_GCC_BUILTIN(td) == 0)
        collectTypeDesc0(td->e_typeExpr);
    collectTypeDesc0(td->e_paramExpr);
    collectTypeDesc0(td->e_bitLenExpr);
    collectTypeDesc0(td->e_len.eln_lenExpr);
    collectTypeDesc0((CExpr*)td->e_refType);
    collectTypeDesc0(EXPR_C(td)->e_gccAttrPre);

    if(ETYP_IS_COMPOSITE(td)) {
        CCOL_DListNode *ite1, *ite2;
        CExpr *memDecls = getMemberDeclsExpr(td);

        if(EXPR_ISNULL(memDecls) == 0) {
            EXPR_FOREACH(ite1, memDecls) {
                CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
                if(EXPR_IS_MEMDECL(memDecl) == 0)
                    continue;
                CExpr *mems = memDecl->e_nodes[1];
                if(EXPR_ISNULL(mems))
                    continue;
                EXPR_FOREACH(ite2, mems) {
                    CExprOfBinaryNode *memDeclr = EXPR_B(EXPR_L_DATA(ite2));
                    CExprOfBinaryNode *declr = EXPR_B(memDeclr->e_nodes[0]);
                    if(EXPR_ISNULL(declr))
                        continue;
                    CExprOfTypeDesc *memTd = EXPR_T(declr->e_nodes[0]);
                    collectTypeDesc0((CExpr*)memTd);
                }
            }
        }
    }
}


/**
 * \brief
 * collect type descriptos in nodes
 *
 * @param node
 *      target node
 */
void
collectTypeDesc(CExpr *expr)
{
    collectTypeDesc0(expr);

    CCOL_SListNode *ite;
    CCOL_SL_FOREACH(ite, &s_staticTypeDescs) {
        CExprOfTypeDesc *td = (CExprOfTypeDesc*)CCOL_SL_DATA(ite);
        if(td->e_isMarked == 0 && td->e_isUsed) {
            addTypeDesc(td);
            td->e_isMarked = 0;
        }
    }
}


/**
 * \brief
 * compile node
 *
 * @param
 *      target node
 */
void
compile(CExpr *expr)
{
    freeSymbolTableList();
    memset(&s_compileContext, 0, sizeof(s_compileContext));
    pushSymbolTableToExpr(expr);

    //set CExprCommon.exprsType
    //convert TD_TYPEREF, TD_GCC_TYPEOF to TD_DERIVED
    compile1(expr, NULL);
    if(s_hasError)
        return;
}

