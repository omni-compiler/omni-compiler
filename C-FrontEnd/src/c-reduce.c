/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-reduce.c
 * implementations for reducing AST before main compilation.
 */

#include "c-comp.h"
#include "c-pragma.h"
#include "c-option.h"

PRIVATE_STATIC CExpr* reduceExpr1(CExpr *expr, CExpr *parent, int *isReduced);
PRIVATE_STATIC CExprOfTypeDesc *
reduce_declarator_1(CExpr *declr, CExpr *parent, int *isReduced, CCOL_DListNode *ite,
                    CExprOfTypeDesc *htd, CExprOfTypeDesc *ptd);


/**
 * \brief
 * reduce nodes of EC_STRINGS.
 * concat to one EC_STRING.
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduce_concatStrings(CExpr *strings, CExpr *parent, int *isReduced)
{
    assert(EXPR_L_SIZE(strings) > 0);
    int numChars = 0;

    if(EXPR_L_SIZE(strings) > 1) {

        CCOL_DListNode *ite;
        char *newToken;
        int len = 0, alen = 0, offset = 0;
        int charType = CT_MB;

        EXPR_FOREACH(ite, strings) {
            CExprOfStringConst *str = EXPR_STRINGCONST(EXPR_L_DATA(ite));
            alen += strlen(str->e_orgToken);
            if(str->e_charType == CT_WIDE)
                charType = CT_WIDE;
        }

        ++alen;
        newToken = XALLOCSZ(char, alen);

        EXPR_FOREACH(ite, strings) {
            CExprOfStringConst *str = EXPR_STRINGCONST(EXPR_L_DATA(ite));
            len = strlen(str->e_orgToken);
            if(len > 0) {
                memcpy(&newToken[offset], str->e_orgToken, len);
                offset += len;
            }

            numChars += str->e_numChars;
        }

        newToken[offset] = 0;
        CExprOfStringConst *head = EXPR_STRINGCONST(exprListHeadData(strings));
        head->e_charType = charType;
        free(head->e_orgToken);
        head->e_orgToken = newToken;
        head->e_numChars = numChars;
    }

    CExprOfStringConst *head = EXPR_STRINGCONST(exprListRemoveHead(strings));
    freeExpr(strings);
    *isReduced = 1;
    EXPR_REF(head);

    return (CExpr*)head;
}


/**
 * Basic Types Info
 */
PRIVATE_STATIC struct validTypeSpecs {
    CTypeSpecEnum   typeSpecs[3];
    int             signable;
    CBasicTypeEnum  sgnType, unsType;
} s_validTypeSpecs[] = {

    { { TS_VOID,      TS_UNDEF,       TS_UNDEF     }, 0, BT_VOID, BT_UNDEF },
    { { TS_CHAR,      TS_UNDEF,       TS_UNDEF     }, 1, BT_CHAR, BT_UNSIGNED_CHAR },
    { { TS_WCHAR,     TS_UNDEF,       TS_UNDEF     }, 0, BT_WCHAR, BT_UNDEF },
    { { TS_SHORT,     TS_UNDEF,       TS_UNDEF     }, 1, BT_SHORT, BT_UNSIGNED_SHORT },
    { { TS_SHORT,     TS_INT,         TS_UNDEF     }, 1, BT_SHORT, BT_UNSIGNED_SHORT },
    { { TS_INT  ,     TS_SHORT,       TS_UNDEF     }, 1, BT_SHORT, BT_UNSIGNED_SHORT },
    { { TS_INT,       TS_UNDEF,       TS_UNDEF     }, 1, BT_INT, BT_UNSIGNED_INT },
    { { TS_LONG,      TS_UNDEF,       TS_UNDEF     }, 1, BT_LONG, BT_UNSIGNED_LONG },
    { { TS_LONG,      TS_INT,         TS_UNDEF     }, 1, BT_LONG, BT_UNSIGNED_LONG },
    { { TS_INT,       TS_LONG,        TS_UNDEF     }, 1, BT_LONG, BT_UNSIGNED_LONG },
    { { TS_LONG,      TS_LONG,        TS_UNDEF     }, 1, BT_LONGLONG, BT_UNSIGNED_LONGLONG },
    { { TS_LONG,      TS_LONG,        TS_INT       }, 1, BT_LONGLONG, BT_UNSIGNED_LONGLONG },
    { { TS_INT ,      TS_LONG,        TS_LONG      }, 1, BT_LONGLONG, BT_UNSIGNED_LONGLONG },
    { { TS_LONG,      TS_INT,         TS_LONG      }, 1, BT_LONGLONG, BT_UNSIGNED_LONGLONG },
    { { TS_FLOAT,     TS_UNDEF,       TS_UNDEF     }, 0, BT_FLOAT, BT_UNDEF },
    { { TS_DOUBLE,    TS_UNDEF,       TS_UNDEF     }, 0, BT_DOUBLE, BT_UNDEF },
    { { TS_LONG,      TS_DOUBLE,      TS_UNDEF     }, 0, BT_LONGDOUBLE, BT_UNDEF },
    { { TS_BOOL,      TS_UNDEF,       TS_UNDEF     }, 0, BT_BOOL, BT_UNDEF },
    { { TS_FLOAT,     TS_COMPLEX,     TS_UNDEF     }, 0, BT_FLOAT_COMPLEX, BT_UNDEF },
    { { TS_DOUBLE,    TS_COMPLEX,     TS_UNDEF     }, 0, BT_DOUBLE_COMPLEX, BT_UNDEF },
    { { TS_LONG,      TS_DOUBLE,      TS_COMPLEX   }, 0, BT_LONGDOUBLE_COMPLEX, BT_UNDEF },
    { { TS_FLOAT,     TS_IMAGINARY,   TS_UNDEF     }, 0, BT_FLOAT_IMAGINARY, BT_UNDEF },
    { { TS_DOUBLE,    TS_IMAGINARY,   TS_UNDEF     }, 0, BT_DOUBLE_IMAGINARY, BT_UNDEF },
    { { TS_LONG,      TS_DOUBLE,      TS_IMAGINARY }, 0, BT_LONGDOUBLE_IMAGINARY, BT_UNDEF }
};

#define VALIDTYPESPECS_SIZE (sizeof(s_validTypeSpecs) / sizeof(struct validTypeSpecs))


/**
 * \brief
 * add error for multi type qualifiers
 *
 * @param expr
 *      error node
 * @param keyword
 *      type qualifier name
 */
PRIVATE_STATIC void addDuplicateError(CExpr *expr, const char *keyword)
{
    addError(expr, CERR_004, keyword);
}


/**
 * \brief
 * reduce nodes of EC_DECL_SPECS.
 * convert to EC_TYPE_DESC.
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduce_declSpecs(CExpr *declSpecs, CExpr *parent, int *isReduced)
{
    assert(declSpecs);
    assertExprCode(declSpecs, EC_DECL_SPECS);
    CCOL_DListNode *ite, *iten;

    /* skip DECL_SPECS which has POINTER_DECL
       (DECL_SPECS under DECLARATOR) */
    EXPR_FOREACH(ite, declSpecs) {
        CExpr *expr = EXPR_L_DATA(ite);

        switch(EXPR_CODE(expr)) {

        case EC_TYPENAME:
        case EC_POINTER_DECL:
            return NULL;
        default:
            break;
        }
    }

    int ti = 0, err = 0, fsgn = 0, funs = 0;
    CExpr *typeExpr = NULL;
    CTypeSpecEnum tstack[3];
    CExprOfTypeDesc *td = allocExprOfTypeDesc();
    exprCopyLineNum((CExpr*)td, declSpecs);
    CCOL_DList attrList;

    memset(&attrList, 0, sizeof(attrList));
    memset(tstack, 0, sizeof(tstack));

    EXPR_FOREACH_SAFE(ite, iten, declSpecs) {
        CExpr *expr = EXPR_L_DATA(ite);
        int fe = 1;

        switch(EXPR_CODE(expr)) {

        case EC_SCSPEC:
            switch(EXPR_GENERALCODE(expr)->e_code) {
            case SS_STATIC:
                if(td->e_sc.esc_isStatic) {
                    if(err == 0) addDuplicateError(expr, "static");
                    err = 1;
                } else
                    td->e_sc.esc_isStatic = 1;
                break;
            case SS_AUTO:
                if(td->e_sc.esc_isAuto) {
                    if(err == 0) addDuplicateError(expr, "auto");
                    err = 1;
                } else
                    td->e_sc.esc_isAuto = 1;
                break;
            case SS_EXTERN:
                if(td->e_sc.esc_isExtern) {
                    if(err == 0) addDuplicateError(expr, "extern");
                    err = 1;
                } else
                    td->e_sc.esc_isExtern = 1;
                break;
            case SS_REGISTER:
                if(td->e_sc.esc_isRegister) {
                    if(err == 0) addDuplicateError(expr, "register");
                    err = 1;
                }
                else
                    td->e_sc.esc_isRegister = 1;
                break;
            case SS_THREAD:
                if(td->e_sc.esc_isGccThread) {
                    if(err == 0) addDuplicateError(expr, "__thread");
                    err = 1;
                } else
                    td->e_sc.esc_isGccThread = 1;
                break;
            case SS_TYPEDEF:
                if(td->e_isTypeDef) {
                    if(err == 0) addDuplicateError(expr, "typedef");
                    err = 1;
                } else
                    td->e_isTypeDef = 1;
                break;
            default:
                ABORT();
            }
            break;

        case EC_TYPEQUAL:
            switch(EXPR_GENERALCODE(expr)->e_code) {
            case TQ_CONST:
                if(td->e_tq.etq_isConst) {
                    if(err == 0) addDuplicateError(expr, "const");
                    err = 1;
                } else
                    td->e_tq.etq_isConst = 1;
                break;
            case TQ_VOLATILE:
                if(td->e_tq.etq_isVolatile) {
                    if(err == 0) addDuplicateError(expr, "volatile");
                    err = 1;
                } else
                    td->e_tq.etq_isVolatile = 1;
                break;
            case TQ_RESTRICT:
                if(td->e_tq.etq_isRestrict) {
                    if(err == 0) addDuplicateError(expr, "restrict");
                    err = 1;
                } else
                    td->e_tq.etq_isRestrict = 1;
                break;
            case TQ_INLINE:
                if(td->e_tq.etq_isInline) {
                    if(err == 0) addDuplicateError(expr, "inline");
                    err = 1;
                } else
                    td->e_tq.etq_isInline = 1;
                break;
            default:
                ABORT();
            }
            break;

        case EC_TYPESPEC:
            switch(EXPR_GENERALCODE(expr)->e_code) {
            case TS_SIGNED:
                if(fsgn) {
                    if(err == 0) addDuplicateError(expr, "signed");
                    err = 1;
                } else if(funs) {
                    if(err == 0) addDuplicateError(expr, "signed/unsigned");
                    err = 1;
                } else
                    fsgn = 1;
                break;
            case TS_UNSIGNED:
                if(funs) {
                    if(err == 0) addDuplicateError(expr, "unsigned");
                    err = 1;
                } else if(fsgn) {
                    if(err == 0) addDuplicateError(expr, "signed/unsigned");
                    err = 1;
                } else
                    funs = 1;
                break;
            case TS_UNDEF:
                ABORT();
            default:
                if(ti > 3) {
                    if(err == 0) addError(expr, CERR_005);
                    err = 1;
                } else
                    tstack[ti++] = EXPR_GENERALCODE(expr)->e_code;
                break;
            }
            break;

        case EC_STRUCT_TYPE:
        case EC_UNION_TYPE:
        case EC_ENUM_TYPE:

            if(typeExpr) {
                if(err == 0) addError(expr, CERR_006);
                err = 1;
            } else {
                typeExpr = expr;
                switch(EXPR_CODE(expr)) {
                case EC_STRUCT_TYPE:
                    td->e_tdKind = TD_STRUCT;
                    break;
                case EC_UNION_TYPE:
                    td->e_tdKind = TD_UNION;
                    break;
                case EC_ENUM_TYPE:
                    td->e_tdKind = TD_ENUM;
                    break;
                default:
                    break;
                }
                fe = 0;
            }
            break;

        case EC_GCC_TYPEOF:
            if(typeExpr) {
                if(err == 0) addError(expr, CERR_006);
                err = 1;
            } else {
                typeExpr = EXPR_U(expr)->e_node;
                td->e_tdKind = TD_GCC_TYPEOF;
                EXPR_U(expr)->e_node = NULL;
                exprJoinAttr((CExpr*)td, expr);
            }
            break;

        case EC_IDENT:
            if(typeExpr) {
                if(err == 0) addError(expr, CERR_006);
                err = 1;
            } else {
                assertExprStruct(expr, STRUCT_CExprOfSymbol);
                typeExpr = expr;
                if(isGccBuiltinType(EXPR_SYMBOL(expr)->e_symName))
                    td->e_tdKind = TD_GCC_BUILTIN;
                else
                    td->e_tdKind = TD_TYPEREF;
                EXPR_SYMBOL(expr)->e_symType = ST_TYPE;
                fe = 0;
            }
            break;

        case EC_NULL_NODE:
            break;

        case EC_GCC_ATTRS:
            CCOL_DL_CONS(&attrList, expr);
            break;

        default:
            assertExpr(expr, 0);
            ABORT();
        }

        CExprCommon *ce = EXPR_C(expr);
        int isReduced1 = 0;

        if(EXPR_ISNULL(ce->e_gccAttrPre) == 0) {
            CExpr *attr = reduceExpr1(ce->e_gccAttrPre, expr, &isReduced1);
            if(attr)
                CCOL_DL_CONS(&attrList, attr);
            ce->e_gccAttrPre = NULL;
            EXPR_UNREF(attr);
        }
        if(EXPR_ISNULL(ce->e_gccAttrPost) == 0) {
            CExpr *attr = reduceExpr1(ce->e_gccAttrPost, expr, &isReduced1);
            if(attr)
                CCOL_DL_CONS(&attrList, attr);
            ce->e_gccAttrPost = NULL;
            EXPR_UNREF(attr);
        }

        if(fe) {
            /* free DECL_SPECS, SCSPEC, TYPEQUAL, TYPESPEC,
               GCC_TYPEOF, NULL_NODE  */

            CCOL_DL_REMOVE(EXPR_DLIST(declSpecs), ite);

            if(ce->e_gccExtension) {
                EXPR_C(td)->e_gccExtension = 1;
            }

            if(EXPR_CODE(expr) == EC_GCC_ATTRS) {
                EXPR_UNREF(expr);
            } else {
                freeExpr(expr);
            }
        } else {
            EXPR_REF(expr);
        }
    }
    int bt = BT_UNDEF;

    if(err == 0) {
        if(ti == 0) {
            if((fsgn || funs) && typeExpr == NULL) {
                bt = fsgn ? BT_INT : BT_UNSIGNED_INT;
            } else if(typeExpr == NULL) {
                addWarn(declSpecs, CWRN_001);
                bt = BT_INT;
            }
        } else {
            for(int i = 0; i < VALIDTYPESPECS_SIZE; ++i) {
                struct validTypeSpecs *vt = &s_validTypeSpecs[i];
                if(vt->typeSpecs[0] == tstack[0] &&
                    vt->typeSpecs[1] == tstack[1] &&
                    vt->typeSpecs[2] == tstack[2]) {

                    if(fsgn || funs) {
                        if(vt->signable == 0) {
                            addError(declSpecs, CERR_007);
                            err = 1;
                        } else {
                            bt = fsgn ? vt->sgnType : vt->unsType;
                        }
                    } else {
                        bt = vt->sgnType;
                    }
                    break;
                }
            }
        }

        if(CCOL_DL_SIZE(&attrList) > 0) {
            CExpr *head = NULL;
            CCOL_DListNode *ite;
            CCOL_DL_FOREACH(ite, &attrList) {
                CExpr *attrExpr = (CExpr*)CCOL_DL_DATA(ite);
                if(head == NULL)
                    head = attrExpr;
                else
                    exprListJoin(head, attrExpr);
            }
            EXPR_REF(head);
            reduceExpr1(head, NULL, isReduced);
            if(EXPR_C(td)->e_gccAttrPre == NULL) {
                EXPR_C(td)->e_gccAttrPre = head;
            } else {
                exprListJoin(EXPR_C(td)->e_gccAttrPre, head);
            }
        }
    }

    CCOL_DL_CLEAR(&attrList);

    if((bt == BT_UNDEF && typeExpr == NULL) ||
        (bt != BT_UNDEF && typeExpr)) {
        err = 1;
    } else if(typeExpr) {
        int isReduced1 = 0;
        td->e_typeExpr = reduceExpr1(typeExpr, (CExpr*)td, &isReduced1);
        if(EXPR_CODE(td->e_typeExpr) != EC_TYPE_DESC)
            EXPR_PARENT(td->e_typeExpr) = (CExpr*)td;
    } else if(bt != BT_UNDEF) {
        td->e_tdKind = TD_BASICTYPE;
        td->e_basicType = bt;
    }

    if(err) {
        addError(declSpecs, CERR_008);
        freeExpr(declSpecs);
        EXPR_ISERROR(td) = 1;
        return (CExpr*)td;
    }

    exprJoinAttr((CExpr*)td, declSpecs);
    reduceExpr1((CExpr*)td, NULL, isReduced); // for reduce gcc attributes

    freeExpr(declSpecs);
    EXPR_REF(td);

    *isReduced = 1;
    return (CExpr*)td;
}


/**
 * \brief
 * collect type specifiers
 *
 * @param declr
 *      declarator
 * @param[out] stack
 *      type specifier stack
 * @param[out] sym
 *      declared symbol
 */
PRIVATE_STATIC void
collectSpecs(CExpr *declr, CExpr *stack, CExpr *costack, CExpr **sym)
{
    CCOL_DListNode *ite, *iten;
    CExpr *innerDeclr = NULL;

    EXPR_FOREACH(ite, declr) {
        CExpr *expr = EXPR_L_DATA(ite);
        if(EXPR_CODE(expr) == EC_LDECLARATOR) {
            innerDeclr = expr;
            break;
        }
    }

    /** 1. braced declarator */
    if(innerDeclr) {
        collectSpecs(innerDeclr, stack, costack, sym);
    }

    int isReduced1 = 0;

    /** 2. array declarator */
    EXPR_FOREACH_SAFE(ite, iten, declr) {
        CExpr *expr = EXPR_L_DATA(ite);

        switch(EXPR_CODE(expr)) {
        case EC_ARRAY_DECL:
            exprListAdd(stack, reduceExpr1(expr, declr, &isReduced1));
            exprListRemove(declr, ite);
            break;
        case EC_COARRAY_DECL:
            exprListAdd(costack, reduceExpr1(((CExprOfArrayDecl*)expr)->e_lenExpr,
                                             declr, &isReduced1));
            exprListRemove(declr, ite);
            break;
        case EC_PARAMS:
        case EC_IDENTS:
        case EC_IDENT:
        case EC_LDECLARATOR:
        case EC_DECL_SPECS:
        case EC_NULL_NODE:
            break;
        default:
            assertExpr(expr, 0);
            ABORT();
        }
    }

    /** 3. function params */
    CExpr *tailExpr = exprListTailData(declr);

    if(tailExpr) {
        switch(EXPR_CODE(tailExpr)) {
        case EC_PARAMS:
        case EC_IDENTS: /* old style function declaration */
            exprListAdd(stack, reduceExpr1(tailExpr, declr, &isReduced1));
            break;
        case EC_IDENT:
        case EC_LDECLARATOR:
        case EC_DECL_SPECS:
        case EC_NULL_NODE:
            break;
        default:
            assertExpr(tailExpr, 0);
            ABORT();
        }
    }

    /** 4. pointer */
    EXPR_FOREACH_SAFE(ite, iten, declr) {
        CExpr *expr = exprListRemove(declr, ite);

        switch(EXPR_CODE(expr)) {
        case EC_IDENT:
            assert(*sym == NULL);
            *sym = expr;
            EXPR_REF(expr);
            break;
        case EC_DECL_SPECS: {
                CCOL_DListNode *dsite;
                CExpr *ptrDecl = NULL;
                int tq = 0, attr = 0;

                EXPR_FOREACH(dsite, expr) {
                    CExpr *dsexpr = EXPR_L_DATA(dsite);
                    switch(EXPR_CODE(dsexpr)) {
                    case EC_POINTER_DECL:
                        assertExpr(dsexpr, ptrDecl == NULL);
                        ptrDecl = dsexpr;
                        exprListAdd(stack, dsexpr);
                        break;
                    case EC_TYPEQUAL:
                        tq = 1;
                        break;
                    case EC_GCC_ATTRS:
                        attr = 1;
                        break;
                    default:
                        break;
                    }
                }

                if(ptrDecl && (tq || attr)) {
                    CExprOfTypeDesc *td = allocExprOfTypeDesc();
                    exprCopyLineNum((CExpr*)td, expr);

                    EXPR_FOREACH(dsite, expr) {
                        CExpr *dsexpr = EXPR_L_DATA(dsite);
                        switch(EXPR_CODE(dsexpr)) {
                        case EC_TYPEQUAL:
                            switch(EXPR_GENERALCODE(dsexpr)->e_code) {
                            case TQ_CONST:
                                td->e_tq.etq_isConst = 1;
                                break;
                            case TQ_VOLATILE:
                                td->e_tq.etq_isVolatile = 1;
                                break;
                            case TQ_RESTRICT:
                                td->e_tq.etq_isRestrict = 1;
                                break;
                            case TQ_INLINE:
                                addError(dsexpr, CERR_009);
                                break;
                            default:
                                assertExpr(expr, 0);
                                break;
                            }
                            break;
                        case EC_GCC_ATTRS: {
                                EXPR_REF(dsexpr);
                                int isReduced = 0;
                                reduceExpr1(dsexpr, NULL, &isReduced);
                                exprJoinAttr((CExpr*)td, dsexpr);
                            }
                            break;
                        default:
                            break;
                        }
                    }

                    exprListAdd(stack, (CExpr*)td);
                }
                freeExpr(expr);
            }
            break;
        case EC_NULL_NODE:
            exprListAdd(stack, expr);
            break;
        case EC_LDECLARATOR:
            freeExpr(expr);
            break;
        case EC_PARAMS:
        case EC_IDENTS:
            break;
        default:
            assertExpr(expr, 0);
            ABORT();
        }
    }
}


/**
 * \brief
 * reduce nodes of EC_LDECLARATOR.
 * convert to EC_DECLARATOR.
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduce_declarator(CExpr *declr, CExpr *parent, int *isReduced)
{
    assertExprCode(declr, EC_LDECLARATOR);
    *isReduced = 1;
    CExpr *sym = NULL, *stack, *costack;
    stack = exprList(EC_LDECLARATOR);
    costack = exprList(EC_XMP_COARRAY_DIMENSIONS);   // (ID=284,TRY5)
    collectSpecs(declr, stack, costack, &sym);
    CCOL_DListNode *ite;

    CExprOfTypeDesc *htd = NULL, *ptd = NULL;
    EXPR_FOREACH(ite, stack) {
        if(htd == NULL) {
            htd = allocExprOfTypeDesc();
            exprCopyLineNum((CExpr*)htd, declr);
        }
        ptd = reduce_declarator_1(declr, parent, isReduced, ite, htd, ptd);
    }

#define TRY5

#ifdef TRY4
    EXPR_FOREACH(ite, costack) {
        if(htd == NULL) {
            htd = allocExprOfTypeDesc();
            exprCopyLineNum((CExpr*)htd, declr);
        }
        ptd = reduce_declarator_1(declr, parent, isReduced, ite, htd, ptd);
    }
#endif

    EXPR_UNREF(sym);
    CExpr *newDeclr = (CExpr*)allocExprOfBinaryNode1(
        EC_DECLARATOR, (CExpr*)htd, (CExpr*)sym);
    EXPR_C(newDeclr)->e_hasInit = EXPR_C(declr)->e_hasInit;
    exprCopyLineNum(newDeclr, declr);
    if(sym)
        EXPR_SYMBOL(sym)->e_declrExpr = newDeclr;
    EXPR_REF(newDeclr);
    exprJoinAttr(newDeclr, declr);
    freeExpr(declr);
    freeExpr(stack);

#ifdef TRY4A
    /*   set e_codimensions (ID=284)
     */
    CExprOfTypeDesc *htd2 = NULL, *ptd2 = NULL;
    EXPR_FOREACH(ite, costack) {
        if(htd2 == NULL) {
            htd2 = allocExprOfTypeDesc();
            exprCopyLineNum((CExpr*)htd2, declr);
        }
        ptd2 = reduce_declarator_1(declr, parent, isReduced, ite, htd2, ptd2);
    }
    if (htd2) {
      EXPR_SYMBOL(sym)->e_codimensions = (CExpr*)htd2;
      EXPR_REF(htd2);
    }
#endif

#ifdef TRY5
    if (EXPR_L_SIZE(costack) > 0) {
        EXPR_SYMBOL(sym)->e_codimensions = costack;
        EXPR_REF(costack);
    } else {
      freeExpr(costack);
    }
#endif

    return newDeclr;
}


PRIVATE_STATIC CExprOfTypeDesc *
reduce_declarator_1(CExpr *declr, CExpr *parent, int *isReduced,
                    CCOL_DListNode *ite, CExprOfTypeDesc *htd, 
                    CExprOfTypeDesc *ptd)
{
    CExpr *expr = EXPR_L_DATA(ite);
    exprJoinAttr((CExpr*)htd, expr);

    if(EXPR_ISNULL(expr)) {
        return ptd;
    }

    if(EXPR_CODE(expr) == EC_TYPE_DESC) {
        assert(ptd);
        assertExpr((CExpr*)ptd, ETYP_IS_POINTER(ptd));
        ptd->e_tq = EXPR_T(expr)->e_tq;
        exprJoinAttrToPre((CExpr*)ptd, expr);
        return ptd;
    }

    CExprOfTypeDesc *td;
    if(ptd == NULL) {
        td = htd;
    } else {
        td = allocExprOfTypeDesc();
        exprCopyLineNum((CExpr*)td, expr);
    }

    if(ptd) {
        ptd->e_typeExpr = (CExpr*)td;
        EXPR_REF(td);
    }

    switch(EXPR_CODE(expr)) {
    case EC_POINTER_DECL:
        td->e_tdKind = TD_POINTER;
        break;
    case EC_PARAMS:
        td->e_tdKind = TD_FUNC;
        td->e_paramExpr = expr;
        EXPR_REF(expr);
        break;
    case EC_IDENTS:
        td->e_tdKind = TD_FUNC_OLDSTYLE;
        td->e_paramExpr = expr;
        EXPR_REF(expr);
        break;
    case EC_GCC_ATTRS:
        reduceExpr1(expr, NULL, isReduced);
        exprJoinAttr((CExpr*)td, expr);
        break;
    case EC_ARRAY_DECL:
    case EC_COARRAY_DECL:
        {
            CExprOfArrayDecl *ary = EXPR_ARRAYDECL(expr);
            td->e_tdKind = TD_ARRAY;

            if(EXPR_ISNULL(ary->e_lenExpr) == 0) {
                int isReduced1 = 0;
                td->e_len.eln_lenExpr = reduceExpr1(ary->e_lenExpr, expr, &isReduced1);
                ary->e_lenExpr = NULL;
            }

            td->e_len.eln_isVariable = ary->e_isVariable;
            td->e_len.eln_isStatic = ary->e_isStatic;

            if(EXPR_ISNULL(ary->e_typeQualExpr) == 0) {
                CCOL_DListNode *tqite;
                EXPR_FOREACH(tqite, ary->e_typeQualExpr) {
                    CExpr *tq = EXPR_L_DATA(tqite);
                    assertExprCode(tq, EC_TYPEQUAL);
                    switch(EXPR_GENERALCODE(tq)->e_code) {
                    case TQ_CONST:
                        td->e_len.eln_isConst = 1;
                        break;
                    case TQ_VOLATILE:
                        td->e_len.eln_isVolatile = 1;
                        break;
                    case TQ_RESTRICT:
                        td->e_len.eln_isRestrict = 1;
                        break;
                    case TQ_INLINE:
                        addError(tq, CERR_010);
                        break;
                    default:
                      ABORT();
                    }
                }
            }
        }
        break;
    default:
      ABORT();
    }

    if(ptd && ptd->e_typeExpr && ETYP_IS_ARRAY(ptd)) {
        CExprOfTypeDesc *rtd = getRefTypeWithoutFix(td);
        if(rtd && ETYP_IS_FUNC(rtd)) {
            addError((CExpr*)ptd, CERR_112);
        }
    }

    return td;
}



/**
 * \brief
 * reduce nodes of EC_STMTS_AND_DECLS.
 * join continuous EC_STMTS_AND_DECLS. 
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduce_reduceStmtAndDecls(CExpr *stmt, CExpr *parent, int *isReduced)
{
    assert(stmt);
    assertExprCode(stmt, EC_STMTS_AND_DECLS);

    CCOL_DListNode *ite, *iten;

    EXPR_FOREACH_SAFE(ite, iten, stmt) {
        CExpr *node = EXPR_L_DATA(ite);
        if(node && EXPR_CODE(node) == EC_STMTS_AND_DECLS) {
            reduce_reduceStmtAndDecls(node, stmt, isReduced);
            *isReduced = 1;
            exprListRemove(stmt, ite);
            exprJoinAttrToPre(stmt, node);
            exprListJoin(stmt, node);
        }
    }

    return stmt;
}


/**
 * \brief
 * reduce nodes of EC_LABELS.
 * delete empty EC_LABELS. 
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduce_deleteEmptyLabels(CExpr *expr, CExpr *parent, int *isReduced)
{
    assert(expr);
    assertExprCode(expr, EC_LABELS);

    if(EXPR_L_SIZE(expr) > 0)
        return expr;

    *isReduced = 1;
    freeExpr(expr);

    return NULL;
}


/**
 * \brief
 * reduce nodes of EC_FUNC_DEF.
 * convert old style parameter declaration to new style.
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduce_funcDefOldStyle(CExpr *expr, CExpr *parent, int *isReduced)
{
    assert(expr);
    assertExprCode(expr, EC_FUNC_DEF);

    CExpr *declr = exprListNextNData(expr, 0); /* nested func */
    if(EXPR_CODE(declr) != EC_DECLARATOR)
        declr = exprListNextNData(expr, 1);

    assertExprCode(declr, EC_DECLARATOR);

    CExprOfTypeDesc *td = EXPR_T(EXPR_B(declr)->e_nodes[0]);

    assertExpr(declr, td);
    assertExprCode((CExpr*)td, EC_TYPE_DESC);

    CExpr *idents = td->e_paramExpr;

    if(ETYP_IS_FUNC_OLDSTYLE(td) == 0 || idents == NULL)
        return expr;

    if(EXPR_CODE(idents) != EC_IDENTS)
        return expr;

    if(EXPR_L_SIZE(expr) < 3) {
        addError(expr, CERR_001);
        return expr;
    }

    CCOL_DListNode *dataDeclsNd = exprListNextN(expr, 2);
    CExpr *dataDecls = EXPR_L_DATA(dataDeclsNd);

    if(dataDecls && EXPR_CODE(dataDecls) != EC_DATA_DECLS) {
        addError(expr, CERR_001);
        return expr;
    }

    CCOL_HashTable ht;
    CCOL_DListNode *ite;
    CCOL_HT_INIT(&ht, CCOL_HT_STRING_KEYS);
    CExpr *params = NULL;

    EXPR_FOREACH(ite, idents) {
        CExprOfSymbol *ident = EXPR_SYMBOL(EXPR_L_DATA(ite));
        CCOL_HashEntry *he = CCOL_HT_FIND_STR(&ht, ident->e_symName);

        if(he == NULL) {
            CCOL_HT_PUT_STR(&ht, ident->e_symName, ident);
        } else {
            addError((CExpr*)ident, CERR_011);
            EXPR_ISERROR(td) = 1;
            goto end;
        }
    }

    if(dataDecls) {

        CCOL_DListNode *ite1, *ite2;

        EXPR_FOREACH(ite1, dataDecls) {
            CExpr *dataDecl = EXPR_L_DATA(ite1);
            assertExpr(dataDecls, dataDecl);
            assertExprCode(dataDecl, EC_DATA_DECL);
            CExpr *td1 = EXPR_B(dataDecl)->e_nodes[0];
            CExpr *initDecls = EXPR_B(dataDecl)->e_nodes[1];
            assertExpr(dataDecl, initDecls);
            assertExprCode(initDecls, EC_INIT_DECLS);

            EXPR_FOREACH(ite2, initDecls) {
                CExpr *initDecl = EXPR_L_DATA(ite2);
                assertExpr(initDecls, initDecl);
                CExprOfBinaryNode *declr =
                    EXPR_B(exprListHeadData(initDecl));
                assertExpr(initDecl, declr);
                CExprOfSymbol *td2 = EXPR_SYMBOL(declr->e_nodes[0]);
                CExprOfSymbol *sym = EXPR_SYMBOL(declr->e_nodes[1]);
                char *key = sym->e_symName;
                CCOL_HashEntry *he = CCOL_HT_FIND_STR(&ht, key);

                if(he == NULL) {
                    addError((CExpr*)sym, CERR_012, key);
                    EXPR_ISERROR(td) = 1;
                    goto end;
                }

                CExprOfSymbol *hsym = EXPR_SYMBOL(CCOL_HT_DATA(he));
                if(EXPRS_TYPE(hsym)) {
                    addError((CExpr*)sym, CERR_013);
                    EXPR_ISERROR(td) = 1;
                    goto end;
                }

                hsym->e_headType = EXPR_T(td1);
                EXPRS_TYPE(hsym) = EXPR_T(td2);
            }
        }
    }

    params = exprList(EC_PARAMS);
    EXPR_FOREACH(ite, idents) {
        CExprOfSymbol *ident = EXPR_SYMBOL(EXPR_L_DATA(ite));
        CCOL_HashEntry *he = CCOL_HT_FIND_STR(&ht, ident->e_symName);
        assert(he);
        CExprOfSymbol *hsym = EXPR_SYMBOL(CCOL_HT_DATA(he));
        CExprOfTypeDesc *td1 = hsym->e_headType;
        hsym->e_headType = NULL;
        CExprOfTypeDesc *td2 = EXPRS_TYPE(hsym);
        if(td1 == NULL && td2) {
            addError((CExpr*)hsym, CERR_014);
            EXPR_ISERROR(td) = 1;
            goto end;
        } else if(td1 == NULL && td2 == NULL) {
            /* no type -> int */
            td1 = allocIntTypeDesc();
            exprCopyLineNum((CExpr*)td1, expr);
        }

        CExpr *declr = (CExpr*)allocExprOfBinaryNode1(
            EC_DECLARATOR, (CExpr*)td2, (CExpr*)hsym);
        CExpr *param = (CExpr*)allocExprOfBinaryNode1(
            EC_PARAM, (CExpr*)td1, declr);
        exprListAdd(params, param);

        hsym->e_declrExpr = declr;
        EXPRS_TYPE(hsym) = NULL;
    }

    EXPR_REF(params);

    td->e_tdKind = TD_FUNC;
    td->e_paramExpr = params;

end:
    CCOL_HT_DESTROY(&ht);

    if(EXPR_ISERROR(td)) {
        if(params)
            freeExpr(params);
    }

    CCOL_DL_SET_DATA(dataDeclsNd, NULL);
    freeExpr(dataDecls);
    freeExpr(idents);
    *isReduced = 1;

    return expr;
}


/**
 * \brief
 * reduce node of EC_MEMBER_DECLS
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduce_memberDecls(CExpr *expr, CExpr *parent, int *isReduced)
{
    assertExprCode(expr, EC_MEMBER_DECLS);
    CCOL_DListNode *ite, *iten;
    EXPR_FOREACH_SAFE(ite, iten, expr) {
        CExpr *node = EXPR_L_DATA(ite);
        if(EXPR_CODE(node) != EC_DIRECTIVE)
            continue;
        exprListRemove(expr, ite);
        addWarn(expr, CWRN_010);
        *isReduced = 1;
    }

    return expr;
}


/**
 * \brief
 * reduce nodes
 *
 * @param expr
 *      target node
 * @param parent
 *      parent node
 * @param[out] isReduced
 *      if expr is reduced, set to 1
 * @return
 *      NULL or reduced node
 */
PRIVATE_STATIC CExpr*
reduceExpr1(CExpr *expr, CExpr *parent, int *isReduced)
{
    if(expr == NULL)
        return NULL;

#ifdef CEXPR_DEBUG_GCCATTR
    startCheckGccAttr(expr);
#endif

    int isReduced1 = 0;
    EXPR_C(expr)->e_gccAttrPre =
        reduceExpr1(EXPR_C(expr)->e_gccAttrPre, expr, &isReduced1);
    EXPR_C(expr)->e_gccAttrPost =
        reduceExpr1(EXPR_C(expr)->e_gccAttrPost, expr, &isReduced1);

    switch(EXPR_CODE(expr)) {

    case EC_DECL_SPECS:
        /* convert EC_DECLS_SPECS to EC_TYPE_DESC */
        expr = reduce_declSpecs(expr, parent, isReduced);
        goto end;
    case EC_LDECLARATOR:
        /* convert EC_LDECLARATOR to EC_DECLARATOR */
        expr = reduce_declarator(expr, parent, isReduced);
        goto end;
    case EC_UNDEF:
        assertExpr(parent, 0);
        ABORT();
    default:
        break;
    }

    switch(EXPR_STRUCT(expr)) {

    case STRUCT_CExprOfSymbol: {
            CExpr *node = EXPR_SYMBOL(expr)->e_valueExpr;
            if(node && EXPR_CODE(node) == EC_NULL_NODE) {
                freeExpr(node);
                EXPR_SYMBOL(expr)->e_valueExpr = NULL;
            } else {
                int isReduced1 = 0;
                EXPR_SYMBOL(expr)->e_valueExpr =
                    reduceExpr1(node, expr, &isReduced1);
            }
        }
        break;

    case STRUCT_CExprOfUnaryNode: {
            CExpr *node = EXPR_U(expr)->e_node;
            if(node && EXPR_CODE(node) == EC_NULL_NODE) {
                freeExpr(node);
                EXPR_U(expr)->e_node = NULL;
            } else {
                int isReduced1 = 0;
                EXPR_U(expr)->e_node = reduceExpr1(node, expr, &isReduced1);
            }
        }
        break;

    case STRUCT_CExprOfBinaryNode:
        for(int i = 0; i < 2; ++i) {
            CExpr * node = EXPR_B(expr)->e_nodes[i];
            if(i == 1 && node && EXPR_CODE(node) == EC_NULL_NODE) {
                /* remove tail null node */
                freeExpr(node);
                EXPR_B(expr)->e_nodes[i] = NULL;
            } else {
                int isReduced1 = 0;
                EXPR_B(expr)->e_nodes[i] =
                    reduceExpr1(node, expr, &isReduced1);
            }
        }
        break;

    case STRUCT_CExprOfList: {
            CCOL_DListNode *ite, *iten, *tail;
            tail = exprListTail(expr);
            EXPR_FOREACH_SAFE(ite, iten, expr) {
                CExpr *node = EXPR_L_DATA(ite);
                if(ite == tail && node && EXPR_CODE(node) == EC_NULL_NODE) {
                    /* remove tail null node */
                    exprListRemoveTail(expr);
                    freeExpr(node);
                } else {
                    int isReduced1 = 0;
                    CCOL_DL_SET_DATA(ite, reduceExpr1(node, expr, &isReduced1));
                }
            }
        }
        break;

    default:
        break;
    }

    switch(EXPR_CODE(expr)) {
    case EC_STRINGS:
        /* concatenate EC_STRING under EC_STRINGS to EC_STRING */
        expr = reduce_concatStrings(expr, parent, isReduced);
        goto end;
    case EC_LABELS:
        /* delete empty EC_LABELS */
        expr = reduce_deleteEmptyLabels(expr, parent, isReduced);
        goto end;
    case EC_STMTS_AND_DECLS:
        // reduce duplicated EC_STMTS_AND_DECLS
        expr = reduce_reduceStmtAndDecls(expr, parent, isReduced);
        goto end;
    case EC_FUNC_DEF:
        /* convert TD_FUNC_OLDSTYLE in EC_FUNC_DEF to TD_FUNC */
        expr = reduce_funcDefOldStyle(expr, parent, isReduced);
        goto end;
    case EC_MEMBER_DECLS:
        /* delete EC_DIRECTIVE in EC_MEMBER_DECLS */
        expr = reduce_memberDecls(expr, parent, isReduced);
        goto end;
    default:
        break;
    }

end:
#ifdef CEXPR_DEBUG_GCCATTR
    endCheckGccAttr(expr);
#endif

    if(expr)
        EXPR_PARENT(expr) = parent;

    return expr;
}


/**
 * \brief
 * reduce nodes before compiling
 *
 * @param expr
 *      target nodes
 */
void
reduceExpr(CExpr *expr)
{
    int isReduced = 0;
    reduceExpr1(expr, NULL, &isReduced);
}

