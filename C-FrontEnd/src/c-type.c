/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-type.c
 * implementations related to type descriptor.
 */

#include <stdio.h>
#include <limits.h>

#include "c-expr.h"
#include "c-option.h"
#include "c-const.h"
#include "c-pragma.h"


#define MAX(x, y)       (((x) > (y)) ? (x) : (y))
#define ROUND(x, y)     ((((x)+(y)-1)/(y))*(y))

//! return value for getImplicitCastResult()
typedef enum CImplicitCastResultEnum {
    ICR_OK,
    ICR_ERR_INCOMPAT_TYPE,
    ICR_WRN_INCOMPAT_PTR,
    ICR_WRN_INT_FROM_PTR,
    ICR_WRN_PTR_FROM_INT,
} CImplicitCastResultEnum;


CExprOfTypeDesc     s_numTypeDescs[BT_END];
CExprOfTypeDesc     s_enumeratorTypeDesc;
CExprOfTypeDesc     s_stringTypeDesc;
CExprOfTypeDesc     s_wideStringTypeDesc;
CExprOfTypeDesc     s_charTypeDesc;
CExprOfTypeDesc     s_wideCharTypeDesc;
CExprOfTypeDesc     s_charPtrTypeDesc;
CExprOfTypeDesc     s_voidTypeDesc;
CExprOfTypeDesc     s_voidPtrTypeDesc;
CExprOfTypeDesc     s_addrIntTypeDesc;
CExprOfTypeDesc     s_implicitDeclFuncPtrTypeDesc;
CCOL_SList          s_staticTypeDescs;
CBasicTypeEnum      s_int64Type;
CExprOfTypeDesc     s_undefTypeDesc;
CBasicTypeEnum      s_wcharType = BT_LONG;


extern void
compile1(CExpr *expr, CExpr *parent);

PRIVATE_STATIC CCompareTypeEnum
compareType0(CExprOfTypeDesc *td1, CExpr *e1,
    CExprOfTypeDesc *td2, CExpr *e2,
    CTypeRestrictLevel rlevel, int checkGccAttr);

PRIVATE_STATIC CExprOfTypeDesc* getPriorityTypeForAssignExpr(
    CExprOfTypeDesc *td1, CExpr *e1, CExprOfTypeDesc *td2, CExpr *e2);

PRIVATE_STATIC CExprOfTypeDesc*
fixInitVal(CExprOfTypeDesc *lvalTd, CExpr *initVal, CExpr **fixVal);

PRIVATE_STATIC int
fixInitValByDesignator(CCOL_DListNode *desigIte,
    CExpr *designators, CExprOfTypeDesc *lvalTd, CExpr *initVal,
    CExpr **fixVal, int *index);

PRIVATE_STATIC void
fixTypeDesc(CExprOfTypeDesc *td);

PRIVATE_STATIC int
checkExprsTypeDescOfChildren(CExpr *expr);

PRIVATE_STATIC CExprOfTypeDesc*
getFirstMemberRefType(CExprOfTypeDesc *td);

/**
 * \brief
 * add type descriptor to master type descritor list
 *
 * @param td
 *      type descriptor
 */
void
addTypeDesc(CExprOfTypeDesc *td)
{
    assert(td);
    assert(EXPR_STRUCT(td) == STRUCT_CExprOfTypeDesc);
    assert(ETYP_IS_TYPEREF(td) == 0 && ETYP_IS_GCC_TYPEOF(td) == 0);
    if(td->e_isCollected)
        return;
    td->e_isCollected = 1;
    CCOL_DL_ADD(&s_typeDescList, td);
}


/**
 * \brief
 * initialize td as basic type 
 *
 * @param td
 *      type descriptor
 * @param bt
 *      basic type
 * @param isConst
 *      set 1 to set const type qualifier
 * @param isRestrict
 *      set 1 to set restrict type qualifier
 */
PRIVATE_STATIC void
setBasicTypeDescFields(CExprOfTypeDesc *td, CBasicTypeEnum bt, int isConst,
    int isRestrict)
{

#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("setBasicTypeDescFields:" ADDR_PRINT_FMT "\n", (uintptr_t)td));
#endif

    memset(td, 0, sizeof(CExprOfTypeDesc));
    EXPR_REF(td);
    EXPR_C(td)->e_exprCode = EC_TYPE_DESC;
    EXPR_C(td)->e_struct = STRUCT_CExprOfTypeDesc;
    td->e_tdKind = TD_BASICTYPE;
    td->e_basicType = bt;
    td->e_tq.etq_isConst = isConst;
    td->e_tq.etq_isRestrict = isRestrict;
    td->e_size = getBasicTypeSize(bt);
    td->e_align = getBasicTypeAlign(bt);
    td->e_isFixed = 1;
}


/**
 * \brief
 * alloc basic type 
 *
 * @param bt
 *      basic type
 * @param isConst
 *      set 1 to set const type qualifier
 * @param isRestrict
 *      set 1 to set restrict type qualifier
 * @return
 *      allocated type descriptor
 */
PRIVATE_STATIC CExprOfTypeDesc*
createBasicTypeDescFields(CBasicTypeEnum bt, int isConst, int isRestrict)
{
    CExprOfTypeDesc *td = allocExprOfTypeDesc();
    setBasicTypeDescFields(td, bt, isConst, isRestrict);
    EXPR_UNREF(td);

#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("createBasicTypeDescFields:" ADDR_PRINT_FMT "\n", (uintptr_t)td));
#endif

    return td;
}


/**
 * \brief
 * initialize td as pointer type 
 *
 * @param td
 *      type descriptors
 * @param isConst
 *      set 1 to set const type qualifier
 * @param isRestrict
 *      set 1 to set restrict type qualifier
 * @param typeExpr
 *      reference type
 */
PRIVATE_STATIC void
setPointerTypeDescFields(CExprOfTypeDesc *td, int isConst, int isRestrict,
    CExprOfTypeDesc *typeExpr)
{

#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("setPointerTypeDescFields:" ADDR_PRINT_FMT "\n", (uintptr_t)td));
#endif

    memset(td, 0, sizeof(CExprOfTypeDesc));
    EXPR_REF(td);
    EXPR_REF(typeExpr);
    EXPR_C(td)->e_exprCode = EC_TYPE_DESC;
    EXPR_C(td)->e_struct = STRUCT_CExprOfTypeDesc;
    td->e_tdKind = TD_POINTER;
    td->e_tq.etq_isConst = isConst;
    td->e_tq.etq_isRestrict = isRestrict;
    td->e_size = s_sizeAddr;
    td->e_align = s_alignAddr;
    td->e_typeExpr = (CExpr*)typeExpr;
    td->e_isFixed = 1;
}


/**
 * \brief
 * initialize td as function type 
 *
 * @param td
 *      type descriptors
 * @param isConst
 *      set 1 to set const type qualifier
 * @param isRestrict
 *      set 1 to set restrict type qualifier
 */
PRIVATE_STATIC void
setFuncTypeDescFields(CExprOfTypeDesc *td,
    int isConst, int isRestrict, CExprOfTypeDesc *rtd)
{

#ifdef CEXPR_DEBUG_MEM
    DBGPRINT(("setFuncTypeDescFields:" ADDR_PRINT_FMT "\n", (uintptr_t)td));
#endif

    setPointerTypeDescFields(td, isConst, isRestrict, rtd);
    EXPR_UNREF(td);
    td->e_tdKind = TD_FUNC;
    CExpr *paramExpr = exprList(EC_PARAMS);
    EXPR_REF(paramExpr);
    td->e_paramExpr = paramExpr;
    td->e_isFixed = 1;
}


/**
 * \brief
 * initialize static data for type descriptors
 */
void
initStaticTypeDescData()
{
    memset(&s_exprsTypeDescList, 0, sizeof(s_exprsTypeDescList));
    memset(&s_typeDescList, 0, sizeof(s_typeDescList));

    //constant TypeDesc list
    memset(&s_staticTypeDescs, 0, sizeof(s_staticTypeDescs));
    CCOL_SL_CONS(&s_staticTypeDescs, &s_enumeratorTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_stringTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_wideStringTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_charTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_wideCharTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_charPtrTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_voidTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_voidPtrTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_addrIntTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_implicitDeclFuncPtrTypeDesc);
    CCOL_SL_CONS(&s_staticTypeDescs, &s_undefTypeDesc);

    //TypeDesc: Enumerator
    setBasicTypeDescFields(&s_enumeratorTypeDesc, BT_INT, 0, 0);
 
    //TypeDesc: String (const char pointer)
    setPointerTypeDescFields(&s_stringTypeDesc, 0, 1,
        createBasicTypeDescFields(BT_CHAR, 0, 0));

    //TypeDesc: Wide String (const wchar pointer)
    int wcharBt = BT_LONG;
    if(s_useBuiltinWchar)
        wcharBt = BT_WCHAR;
    else if(s_useShortWchar)
        wcharBt = BT_UNSIGNED_SHORT;

    setPointerTypeDescFields(&s_wideStringTypeDesc, 0, 0,
        createBasicTypeDescFields(wcharBt, 0, 0));

    //TypeDesc: Char
    setBasicTypeDescFields(&s_charTypeDesc, BT_CHAR, 0, 0);

    //TypeDesc: Wide Char
    setBasicTypeDescFields(&s_wideCharTypeDesc, wcharBt, 0, 0);

    //TypeDesc: Char Pointer
    setPointerTypeDescFields(&s_charPtrTypeDesc, 0, 0,
        createBasicTypeDescFields(BT_CHAR, 0, 0));

    //TypeDesc: void
    setBasicTypeDescFields(&s_voidTypeDesc, BT_VOID, 0, 0);

    //TypeDesc: void pointer
    setPointerTypeDescFields(&s_voidPtrTypeDesc, 0, 0,
        createBasicTypeDescFields(BT_VOID, 0, 0));

    //TypeDesc: integer of address width
    if(s_sizeAddr <= 4)
        setBasicTypeDescFields(&s_addrIntTypeDesc, BT_INT, 0, 0);
    else
        setBasicTypeDescFields(&s_addrIntTypeDesc, BT_LONG, 0, 0);

    //TypeDesc: Implicit Declarared Function
    CExprOfTypeDesc *implicitFuncTd = allocExprOfTypeDesc();
    setFuncTypeDescFields(implicitFuncTd, 0, 0,
        createBasicTypeDescFields(BT_INT, 0, 0));

    //TypeDesc: Implicit Declarared Function Pointer
    setPointerTypeDescFields(&s_implicitDeclFuncPtrTypeDesc, 0, 0,
        implicitFuncTd);

    //TypeDesc: undef (for gcc builtin identifiers)
    EXPR_C(&s_undefTypeDesc)->e_exprCode = EC_TYPE_DESC;
    EXPR_C(&s_undefTypeDesc)->e_struct = STRUCT_CExprOfTypeDesc;
    s_undefTypeDesc.e_tdKind = TD_GCC_BUILTIN_ANY;
    s_undefTypeDesc.e_isFixed = 1;
    EXPR_REF(&s_undefTypeDesc);

    //TypeDesc: Number
    memset(&s_numTypeDescs, 0, sizeof(s_numTypeDescs));
    for(CBasicTypeEnum bt = BT_UNDEF + 1; bt < BT_END; ++bt) {
        CExprOfTypeDesc *td = &s_numTypeDescs[bt];
        setBasicTypeDescFields(td, bt, 0, 0);
        CCOL_SL_CONS(&s_staticTypeDescs, td);
    }

    //BasicTypeEnum: int64
    if(s_sizeLong == 8)
        s_int64Type = BT_LONG;
    else
        s_int64Type = BT_LONGLONG;

    //Default Symbol Table
    CCOL_HashTable *identGroup = &s_defaultSymTab->stb_identGroup;

    for(const char **p = s_gccBuiltinTypes; *p != NULL; ++p) {
        char *token = ccol_strdup(*p, MAX_NAME_SIZ);
        CExprOfSymbol *sym = allocExprOfSymbol(EC_IDENT, token);
        EXPR_REF(sym);
        sym->e_symType = ST_GCC_BUILTIN;
        CCOL_HT_PUT_STR(identGroup, token, sym);
    }
}


/**
 * \brief
 * free static data for type descriptors
 */
void
freeStaticTypeDescData()
{
    // TypeDesc List
    CCOL_SListNode *site;

    CCOL_SL_FOREACH(site, &s_exprsTypeDescList) {
        freeExpr((CExpr*)CCOL_SL_DATA(site));
    }

    CCOL_SL_CLEAR(&s_exprsTypeDescList);

    CCOL_SL_FOREACH(site, &s_staticTypeDescs) {
        CExprOfTypeDesc *td = EXPR_T(CCOL_SL_DATA(site));
        innerFreeExprOfTypeDesc(td);
    }

    CCOL_SL_CLEAR(&s_staticTypeDescs);

    //Default Symbol Table
    CCOL_HashEntry *he;
    CCOL_HashSearch hs;

    CCOL_HT_FOREACH(he, hs, &s_defaultSymTab->stb_identGroup) {
        CExpr *sym = (CExpr*)CCOL_HT_DATA(he);
        freeExpr(sym);
    }
}


/**
 * \brief
 * judge td is integer type 
 *
 * @param td
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
int
isIntegerType(CExprOfTypeDesc *td)
{
    td = getRefType(td);

    if(ETYP_IS_BASICTYPE(td)) {
        switch(td->e_basicType) {
        case BT_BOOL:
        case BT_CHAR:
        case BT_UNSIGNED_CHAR:
        case BT_SHORT:
        case BT_UNSIGNED_SHORT:
        case BT_INT:
        case BT_UNSIGNED_INT:
        case BT_LONG:
        case BT_UNSIGNED_LONG:
        case BT_LONGLONG:
        case BT_UNSIGNED_LONGLONG:
            return 1;
        default:
            break;
        }
    } else if(ETYP_IS_ENUM(td)) {
        return 1;
    }

    return 0;
}


/**
 * \brief
 * judge td is scalar or pointer type
 *
 * @param td
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
int
isScalarOrPointerType(CExprOfTypeDesc *td)
{
    td = getRefType(td);
    
    switch(td->e_tdKind) {
    case TD_BASICTYPE:
    case TD_ENUM:
    case TD_POINTER:
    case TD_ARRAY:
    case TD_GCC_BUILTIN:
        return 1;
    default:
        break;
    }

    return 0;
}


/**
 * \brief
 * judge td has storage specifier
 *
 * @param td
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
int
isScspecSet(CExprOfTypeDesc *td)
{
    return
        td->e_sc.esc_isStatic ||
        td->e_sc.esc_isAuto ||
        td->e_sc.esc_isExtern ||
        td->e_sc.esc_isRegister ||
        td->e_sc.esc_isGccThread;
}


/**
 * \brief
 * judge td has type qualifer
 *
 * @param td
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
int
isTypeQualSet(CExprOfTypeDesc *td)
{
    return (td->e_tq.etq_isConst ||
        td->e_tq.etq_isRestrict ||
        td->e_tq.etq_isVolatile ||
        td->e_tq.etq_isInline);
}


/**
 * \brief
 * judge td has type qualifer or gcc __extension__ keyword
 *
 * @param
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
int
isTypeQualOrExtensionSet(CExprOfTypeDesc *td)
{
    return isTypeQualSet(td) ||
        EXPR_GCCEXTENSION(td);
}


/**
 * \brief
 * judge type qualifiers of t1 and t2 equal
 *
 * @param t1
 *      type descriptor 1
 * @param t2
 *      type descriptor 2
 * @return
 *      0:no, 1:yes
 */
int
isTypeQualEquals(CExprOfTypeDesc *t1, CExprOfTypeDesc *t2)
{
    return (
        t1->e_tq.etq_isConst    == t2->e_tq.etq_isConst    &&
        t1->e_tq.etq_isRestrict == t2->e_tq.etq_isRestrict &&
        t1->e_tq.etq_isVolatile == t2->e_tq.etq_isVolatile &&
        t1->e_tq.etq_isInline   == t2->e_tq.etq_isInline); 
}


/**
 * \brief
 * judge type qualifiers of t1 and t2 equal exclude 'inline'
 *
 * @param t1
 *      type descriptor 1
 * @param t2
 *      type descriptor 2
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isTypeQualEqualsExcludeInline(CExprOfTypeDesc *t1, CExprOfTypeDesc *t2)
{
    return (
        t1->e_tq.etq_isConst    == t2->e_tq.etq_isConst    &&
        t1->e_tq.etq_isRestrict == t2->e_tq.etq_isRestrict &&
        t1->e_tq.etq_isVolatile == t2->e_tq.etq_isVolatile);
}


/**
 * \brief
 * sub function of getRefType
 */
PRIVATE_STATIC CExprOfTypeDesc*
getRefType0(CExprOfTypeDesc *td, CExprOfTypeDesc *orgTd)
{
    if(td->e_refType) {
        if(orgTd == td->e_refType)
            return orgTd;
        else
            return getRefType0(td->e_refType, orgTd);
    }

    if(td->e_typeExpr && EXPR_CODE(td->e_typeExpr) == EC_IDENT &&
        ETYP_IS_GCC_BUILTIN(td) == 0) {
        CExpr *sym = td->e_typeExpr;
        CExprOfTypeDesc *rtd = resolveType(sym);
        if(rtd)
            return getRefType0(rtd, orgTd);
    }

    resolveType((CExpr*)td);

    return td;
}


/**
 * \brief
 * get reference type with resolving type
 *
 * @param td
 *      type descriptor
 * @return
 *      NULL or reference type
 */
CExprOfTypeDesc*
getRefType(CExprOfTypeDesc *td)
{
    return getRefType0(td, td);
}


/**
 * \brief
 * sub function of getRefTypeWithoutFix
 */
PRIVATE_STATIC CExprOfTypeDesc*
getRefTypeWithoutFix0(CExprOfTypeDesc *td, CExprOfTypeDesc *orgTd)
{
    if(td->e_refType) {
        if(orgTd == td->e_refType)
            return orgTd;
        else
            return getRefTypeWithoutFix0(td->e_refType, orgTd);
    }

    if(td->e_typeExpr && EXPR_CODE(td->e_typeExpr) == EC_IDENT &&
        ETYP_IS_GCC_BUILTIN(td) == 0) {
        CExprOfSymbol *sym = EXPR_SYMBOL(td->e_typeExpr);
        CExprOfTypeDesc *rtd = EXPRS_TYPE(sym);
        if(rtd == NULL) {
            CExprOfSymbol *tsym = findSymbolByGroup(sym->e_symName, STB_IDENT);
            if(tsym)
                rtd = EXPRS_TYPE(tsym);
        }
        if(rtd)
            return getRefTypeWithoutFix0(rtd, orgTd);
    }

    return td;
}


/**
 * \brief
 * get reference type without resolving type
 *
 * @param td
 *      type descriptor
 * @return
 *      NULL or reference type
 */
CExprOfTypeDesc*
getRefTypeWithoutFix(CExprOfTypeDesc *td)
{
    return getRefTypeWithoutFix0(td, td);
}


/**
 * \brief
 * get member type descriptor by member symbol
 *
 * @param parentTd
 *      composite type descriptor
 * @param memberName
 *      mamber name
 * @param[out] outParentTd
 *      if member is anonymous, parent composite type will be set
 * @param[out] outParentSym
 *      if member is anonymous, parent composite member's symbol will be set
 * @return
 *      NULL or member type descriptor
 */
CExprOfTypeDesc*
getMemberType(CExprOfTypeDesc *parentTd, const char *memberName,
    CExprOfTypeDesc **outParentTd, CExprOfSymbol **outParentSym)
{
    assert(parentTd);
    assertExprCode((CExpr*)parentTd, EC_TYPE_DESC);

    if(outParentTd)
        *outParentTd = NULL;
    if(outParentSym)
        *outParentSym = NULL;

    CExprOfTypeDesc *composTd = getRefType(parentTd);
    if(composTd == NULL)
        return NULL;
    CExpr *memDecls = getMemberDeclsExpr(composTd);
    if(memDecls == NULL)
        return NULL;

    CCOL_DListNode *ite1, *ite2;
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
            CExprOfTypeDesc *td = EXPR_T(declr->e_nodes[0]);
            CExprOfSymbol *sym = EXPR_SYMBOL(declr->e_nodes[1]);
            if(EXPR_ISNULL(sym) == 0 && strcmp(sym->e_symName, memberName) == 0) {
                if(outParentTd && *outParentTd == NULL)
                    *outParentTd = parentTd;
                return td;
            }
        }
    }

    //gcc allows getting nested anonymous member of struct/union type
    EXPR_FOREACH(ite1, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExprOfTypeDesc *anonMemTd = EXPR_T(memDecl->e_nodes[0]);

        if(anonMemTd != parentTd && anonMemTd->e_isAnonMember &&
            anonMemTd->e_isNoMemDecl == 0 &&
            ETYP_IS_COMPOSITE(anonMemTd)) {

            CExprOfTypeDesc *memTd = getMemberType(
                anonMemTd, memberName, outParentTd, outParentSym);
            if(memTd) {
                if(outParentSym && *outParentSym == NULL) {
                    CExpr *mems = memDecl->e_nodes[1];
                    CExprOfBinaryNode *memDeclr = EXPR_B(exprListHeadData(mems));
                    CExprOfBinaryNode *delcr = EXPR_B(memDeclr->e_nodes[0]);
                    *outParentSym = EXPR_SYMBOL(delcr->e_nodes[1]);
                }
                return memTd;
            }
        }
    }

    return NULL; 
}


/**
 * \brief
 * set type to member designator for gcc __builtin_offsetof
 *
 * @param composTd
 *      composit type (1st argument of __builtin_offsetof)
 * @param dsg
 *      member designator
 * @return
 *      NULL or expression's type of member designator
 */
PRIVATE_STATIC CExprOfTypeDesc*
setGccOfsMemberDesignatorType(CExprOfTypeDesc *composTd,
    CExprOfBinaryNode *dsg)
{
    CExprOfBinaryNode *innerDsg = EXPR_B(dsg->e_nodes[0]);
    CExpr *identOrAry = dsg->e_nodes[1];
    CExprOfTypeDesc *innerTd = composTd;
    
    if(EXPR_ISNULL(innerDsg) == 0) {
        innerTd = setGccOfsMemberDesignatorType(composTd, innerDsg);
        if(innerTd == NULL)
            return NULL;
    }

    CExprOfTypeDesc *innerTd0 = getRefType(innerTd);

    if(EXPR_CODE(identOrAry) != EC_IDENT) {
        assertExpr(identOrAry, innerDsg);
        if(ETYP_IS_ARRAY(innerTd0) == 0) {
            addError(identOrAry, CERR_037);
            return NULL;
        }
        CExprOfTypeDesc *td = EXPR_T(innerTd0->e_typeExpr);
        exprSetExprsType((CExpr*)dsg, td);
        return td;
    }

    CExprOfTypeDesc *parentTd = NULL;
    CExprOfSymbol *parentSym = NULL;
    CExprOfSymbol *ident = EXPR_SYMBOL(identOrAry);

    if(ETYP_IS_TAGGED(innerTd0) == 0) {
        addError(identOrAry, CERR_038);
        return NULL;
    }

    CExprOfTypeDesc *memTd = getMemberType(
        innerTd, ident->e_symName, &parentTd, &parentSym);

    if(memTd == NULL) {
        addError(identOrAry, CERR_039, ident->e_symName);
        return NULL;
    }

    CExprOfTypeDesc *td;

    if(parentTd == innerTd || parentTd == NULL) {
        td = innerTd;
    } else {
        // complete anonymous member access
        CExprOfSymbol *parentSym1 = allocExprOfSymbol(
            EC_IDENT, ccol_strdup(parentSym->e_symName, MAX_NAME_SIZ));
        parentSym1->e_symType = ST_MEMBER;
        CExprOfBinaryNode *fixDsg = allocExprOfBinaryNode1(
            EC_GCC_OFS_MEMBER_REF, (CExpr*)innerDsg, (CExpr*)parentSym1);
        EXPR_REF(fixDsg);
        freeExpr((CExpr*)innerDsg);
        dsg->e_nodes[0] = (CExpr*)fixDsg;
        CExprOfTypeDesc *tmpTd = setGccOfsMemberDesignatorType(composTd, fixDsg);
        if(tmpTd == NULL)
            return NULL;
        td = parentTd;
        //ISGCCSYNTAX is not needed.
        //because offsetof member designator's tag starts with "gcc".
        ++s_numConvsAnonMemberAccess;
    }

    exprSetExprsType((CExpr*)dsg, td);

    return memTd;
}


/**
 * \brief
 * resolve type for EC_GCC_BLTIN_OFFSET_OF
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_gccBuiltinOffsetOf(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;
    CExprOfTypeDesc *composTd = resolveType(EXPR_B(expr)->e_nodes[0]);
    if(composTd)
        setGccOfsMemberDesignatorType(composTd,
            EXPR_B(EXPR_B(expr)->e_nodes[1]));

    CExprOfTypeDesc *td = &s_numTypeDescs[BT_INT];

    return td;
}


/**
 * \brief
 * judge expr is child of specified kind of type descriptor
 *
 * @param expr
 *      target node
 * @param tk
 *      type descriptor kind
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isTypeDescKindChildOf(CExpr *expr, CTypeDescKindEnum tk)
{
    if(expr == NULL)
        return 0;
    CExpr *parent = EXPR_PARENT(expr);
    if(parent == NULL)
        return 0;

    switch(EXPR_CODE(parent)) {
    case EC_TYPE_DESC:
        return tk == TD_UNDEF || (EXPR_T(parent)->e_tdKind == tk);
    case EC_SIZE_OF:
    case EC_GCC_ALIGN_OF:
    case EC_GCC_BLTIN_OFFSET_OF:
        return 0;
    default:
        break;
    }

    if(isStatementOrLabelOrDeclOrDef(parent)) {
        return 0;
    } else {
        return isTypeDescKindChildOf(parent, tk);
    }
}


/**
 * \brief
 * convert type descriptor of TD_TYPEREF/TD_GCC_TYPEOF to TD_DERIVED
 *
 * @param td
 *      type descriptor
 */
PRIVATE_STATIC void
convertTypeRefToDerived(CExprOfTypeDesc *td)
{
    assertExpr((CExpr*)td, ETYP_IS_TYPEREF(td) || ETYP_IS_GCC_TYPEOF(td));
    CExprOfTypeDesc *etd = resolveType(td->e_typeExpr);

    if(etd == NULL) {
        EXPR_ISERROR(td) = 1;
        return;
    }

    assert(etd != td);
    CExprOfTypeDesc *etdo = getRefType(etd);

    if(ETYP_IS_ARRAY(etdo) && etdo->e_len.eln_isFlexible) {
        //convert flexible array to pointer
        etd = setPointerTypeOfArrayElem(etdo);
    } else if(ETYP_IS_GCC_TYPEOF(td) && ETYP_IS_ENUM(etdo) &&
        EXPR_CODE(td->e_typeExpr) != EC_TYPE_DESC) {
        //enum { E }; typeof(E) a; -> int a;
        etd = &s_numTypeDescs[BT_INT];
    }

    td->e_tdKind = TD_DERIVED;
    td->e_paramExpr = td->e_typeExpr;
    td->e_typeExpr = NULL;
    td->e_refType = etd;
}


/**
 * \brief
 * judge td1 and td2 are able to cast each other implicitly
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @return
 *      cast result code
 */
PRIVATE_STATIC CImplicitCastResultEnum
getImplicitCastResult(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2)
{
    CExprOfTypeDesc *tdo1 = getRefType(td1);
    CExprOfTypeDesc *tdo2 = getRefType(td2);
    int b1 = isBasicTypeOrEnum(tdo1);
    int b2 = isBasicTypeOrEnum(tdo2);

    if(b1 && b2)
        return ICR_OK;

    int i1 = isIntegerType(tdo1);
    int i2 = isIntegerType(tdo2);
    int p1 = ETYP_IS_PTR_OR_ARRAY(tdo1);
    int p2 = ETYP_IS_PTR_OR_ARRAY(tdo2);

    if(i1) {
        if(p2)
            return ICR_WRN_INT_FROM_PTR;
        else
            return ICR_ERR_INCOMPAT_TYPE;
    }

    if(i2) {
        if(p1)
            return ICR_WRN_PTR_FROM_INT;
        else
            return ICR_ERR_INCOMPAT_TYPE;
    }

    if(p1 && p2) {
        if(isCompatiblePointerType(tdo1, tdo2, NULL, 0) == 0)
            return ICR_WRN_INCOMPAT_PTR;
        else
            return ICR_OK;
    }

    if(compareTypeForAssign(tdo1, tdo2) == CMT_EQUAL)
        return ICR_OK;

    return ICR_ERR_INCOMPAT_TYPE;
}


/**
 * \brief
 * get gcc builtin function type by function name
 *
 * @param sym
 *      function name symbol
 * @return
 *      NULL or function type
 */
PRIVATE_STATIC CExprOfTypeDesc*
getGccBuiltinFuncType(CExprOfSymbol *sym)
{
    CExprOfTypeDesc *argTd = NULL;

    CExpr *parent = EXPR_PARENT(sym);
    if(EXPR_CODE(parent) == EC_FUNCTION_CALL) {
        CExpr *args = EXPR_B(parent)->e_nodes[1];
        if(EXPR_ISNULL(args) == 0 && EXPR_L_SIZE(args) > 0) {
            compile1(args, parent);
            argTd = resolveType(exprListHeadData(args));
            if(argTd)
                argTd = getRefType(argTd);
        }
    }

    const char *symName = sym->e_symName;
    int isConstFunc = 0;
    CExprOfTypeDesc *rtd = getGccBuiltinFuncReturnType(
        symName, argTd, (CExpr*)sym, &isConstFunc);
    CExprOfTypeDesc *ftd = NULL;

    if(rtd) {
        //GCC Builtin Func
        sym->e_symType = ST_GCC_BUILTIN;
        rtd->e_isUsed = 1;
        //dummy function type
        ftd = allocExprOfTypeDesc();
        if(EXPR_CODE(parent) == EC_FUNCTION_CALL)
            ftd->e_isNoTypeId = 1;
        ftd->e_tdKind = TD_FUNC;
        ftd->e_typeExpr = (CExpr*)rtd;
        if(isConstFunc) {
            ftd->e_isGccConst = 1;
            ftd->e_sc.esc_isStatic = 1;
            ftd->e_tq.etq_isInline = 1;
        }
        EXPR_REF(rtd);
        ftd->e_paramExpr = exprList(EC_PARAMS);
        EXPR_REF(ftd);
        CCOL_SL_CONS(&s_exprsTypeDescList, ftd);
    }

    return ftd;
}


/**
 * \brief
 * resolve type for EC_IDENT
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
CExprOfTypeDesc*
resolveType_ident(CExpr *expr, int occurErrorIfNotFound, int *ignore)
{
    if(EXPRS_TYPE(expr))
        return EXPRS_TYPE(expr);
    if(EXPR_ISERROR(expr))
        return NULL;
    CExprOfSymbol *sym = EXPR_SYMBOL(expr);
    CExpr *parent = EXPR_PARENT(sym);
    CExprCodeEnum pec = parent ? EXPR_CODE(parent) : EC_NULL_NODE;
    int isMemRefKind = (pec == EC_MEMBER_REF ||
        pec == EC_POINTS_AT || pec == EC_GCC_OFS_MEMBER_REF);
    CSymbolTypeEnum st = sym->e_symType;

    if(st == ST_UNDEF &&
        ((isMemRefKind && EXPR_B(parent)->e_nodes[1] == expr) ||
        pec == EC_DESIGNATORS)) {

        sym->e_symType = ST_MEMBER;
        *ignore = 1;
        return NULL;
    }

    switch(st) {
    case ST_TAG:
    case ST_ENUM:
    case ST_GCC_ASM_IDENT:
    case ST_MEMBER:
        *ignore = 1;
        return NULL;
    case ST_GCC_LABEL:
        EXPR_ISGCCSYNTAX(sym) = 1;
        //nobreak
    case ST_LABEL:
        //label address -> void *
        return &s_voidPtrTypeDesc;
    default:
        break;
    }

    CExprOfSymbol *tsym = findSymbolByGroup(sym->e_symName, STB_IDENT);
    
    if(tsym && tsym->e_symType == ST_GCC_BUILTIN) {
        sym->e_symType = tsym->e_symType;
        *ignore = 1;
        return NULL;
    }

    if(tsym == sym) {
        return NULL;
    }

    CExprOfTypeDesc *td = NULL;

    if(tsym == NULL) {
        switch(st) {
        case ST_UNDEF:
        case ST_FUNC:
            td = getGccBuiltinFuncType(sym);
            if(td == NULL) {
                if(EXPR_CODE(EXPR_PARENT(sym)) != EC_FUNCTION_CALL) {
                    if(occurErrorIfNotFound)
                        addError((CExpr*)sym, CERR_040, sym->e_symName);
                } else {
                    sym->e_symType = ST_FUNC;
                    *ignore = 1; // implicit declared function call
                }
                return NULL;
            }
            break;
        case ST_TYPE:
            addError((CExpr*)sym, CERR_040, sym->e_symName);
            return NULL;
        case ST_FUNCID:
            td = &s_stringTypeDesc;
            break;
        case ST_LABEL:
        case ST_GCC_LABEL:
            // label address -> void *
            td = &s_voidPtrTypeDesc;
            break;
        default:
            addError((CExpr*)sym, CERR_041, sym->e_symName);
            return NULL;
        }

        return td;
    }

    sym->e_isGlobal = tsym->e_isGlobal;
    CExprOfTypeDesc *tsymTd = resolveType((CExpr*)tsym);

    if(tsymTd == NULL)
        return NULL;

    td = tsymTd;

    if(tsym->e_symType == ST_FUNC && pec != EC_PARAMS) {
        //convert function type to function pointer type
        CExprOfTypeDesc *tsymTdo = getRefType(td); 
        if(ETYP_IS_FUNC(tsymTdo) &&
            isTypeDescKindChildOf(expr, TD_GCC_TYPEOF) == 0) {

            //isTypeDescKindChildOf() == 1 means the declaration
            //such as 'typeof(func) f;'
            td = allocPointerTypeDesc(tsymTd);
        }
    }

    if(sym->e_symType == ST_UNDEF) {
        sym->e_symType = tsym->e_symType;
    }

    return td;
}


/**
 * \brief
 * resolve type for EC_CAST
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_cast(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;

    CExprOfTypeDesc *td1 = EXPR_T(EXPR_B(expr)->e_nodes[0]);
    CExpr *v = EXPR_B(expr)->e_nodes[1];
    CExprOfTypeDesc *td2 = EXPRS_TYPE(v);
    assert(td1);
    assert(td2);
    td1 = getRefType(td1);

    if(ETYP_IS_BOOL(td1))
        return td1;

    td2 = getRefType(td2);
    int p1 = ETYP_IS_PTR_OR_ARRAY(td1);
    int p2 = ETYP_IS_PTR_OR_ARRAY(td2);
    int i1 = isIntegerType(td1);
    int i2 = isIntegerType(td2);

    if((p1 && i2) || (i1 && p2)) {
        int sz1 = ETYP_IS_ARRAY(td1) ? s_sizeAddr : getTypeSize(td1);
        int sz2 = ETYP_IS_ARRAY(td2) ? s_sizeAddr : getTypeSize(td2);
        if(sz1 > 0 && sz2 > 0 && sz1 < sz2 && isConstZero(v) == 0) {
            addWarn(expr, p1 ? CWRN_015 : CWRN_016);
        }
    } else if(ETYP_IS_VOID(td1) == 0 && ETYP_IS_VOID(td2) == 0 &&
        ((p1 == 0 && isBasicTypeOrEnum(td1) == 0) ||
        (p2 == 0 && isBasicTypeOrEnum(td2) == 0))) {
        addError(expr, CERR_124);
    }

    return td1;
}


/**
 * \brief
 * convert  float number type to complex/imaginary type
 *
 * @param bt
 *      basic type
 * @return
 *      complex/imaginary type
 */
PRIVATE_STATIC CBasicTypeEnum
toComplexType(CBasicTypeEnum bt)
{
    switch(bt) {
    case BT_FLOAT:
    case BT_FLOAT_IMAGINARY:
        return BT_FLOAT_COMPLEX;
    case BT_LONGDOUBLE:
    case BT_LONGDOUBLE_IMAGINARY:
        return BT_LONGDOUBLE_COMPLEX;
    default:
        return BT_DOUBLE_COMPLEX;
    }
}


/**
 * \brief
 * convert complex/imaginary type to float number type
 *
 * @param bt
 *      basic type
 * @return
 *      float number type
 */
PRIVATE_STATIC CBasicTypeEnum
toRealType(CBasicTypeEnum bt)
{
    switch(bt) {
    case BT_FLOAT_IMAGINARY:
    case BT_FLOAT_COMPLEX:
        return BT_FLOAT;
    case BT_DOUBLE_IMAGINARY:
    case BT_DOUBLE_COMPLEX:
        return BT_DOUBLE;
    case BT_LONGDOUBLE_IMAGINARY:
    case BT_LONGDOUBLE_COMPLEX:
        return BT_LONGDOUBLE;
    default:
        return bt;
    }
}


/**
 * \brief
 * judge td is basic type or enum type
 *
 * @param td
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
int
isBasicTypeOrEnum(CExprOfTypeDesc *td)
{
    td = getRefType(td);
    return (ETYP_IS_BASICTYPE(td) || ETYP_IS_ENUM(td));
}


/**
 * \brief
 * get td's basic type.
 * if td is enum type, return BT_INT.
 *
 * @param td
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC CBasicTypeEnum
getBasicTypeWithEnum(CExprOfTypeDesc *td)
{
    td = getRefType(td);
    if(ETYP_IS_BASICTYPE(td))
        return td->e_basicType;
    else if(ETYP_IS_ENUM(td))
        return BT_INT;
    else
        ABORT();
}


/**
 * \brief
 * judge bt is unsigned integer type
 *
 * @param bt
 *      basic type
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isUnsignedIntegerType(CBasicTypeEnum bt)
{
    if(bt == BT_WCHAR)
        bt = s_wcharType;

    switch(bt) {
    case BT_UNSIGNED_CHAR:
    case BT_UNSIGNED_SHORT:
    case BT_BOOL:
    case BT_UNSIGNED_INT:
    case BT_UNSIGNED_LONG:
    case BT_UNSIGNED_LONGLONG:
        return 1;
    default:
        return 0;
    }
}


/**
 * \brief
 * get unsigned integer type of basic type
 *
 * @param bt
 *      basic type
 * @return
 *      unsigned integer type descriptor
 */
PRIVATE_STATIC CExprOfTypeDesc*
getUnsignedIntegerType(CBasicTypeEnum bt)
{
    if(bt == BT_WCHAR)
        bt = s_wcharType;

    switch(bt) {
    case BT_CHAR:
        return &s_numTypeDescs[BT_UNSIGNED_CHAR];
    case BT_SHORT:
        return &s_numTypeDescs[BT_UNSIGNED_SHORT];
    case BT_BOOL:
    case BT_INT:
        return &s_numTypeDescs[BT_UNSIGNED_INT];
    case BT_LONG:
        return &s_numTypeDescs[BT_UNSIGNED_LONG];
    case BT_LONGLONG:
        return &s_numTypeDescs[BT_UNSIGNED_LONGLONG];
    default:
        ABORT();
    }
    return NULL;
}


/**
 * \brief
 * get prior number type
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @param ec
 *      operator's expression code
 * @return
 *      prior type descriptor
 */
PRIVATE_STATIC CExprOfTypeDesc*
getPriorNumberType(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2,
    CExprCodeEnum ec)
{
    assertExpr((CExpr*)td1, ETYP_IS_BASICTYPE(td1) || ETYP_IS_ENUM(td1));
    assertExpr((CExpr*)td2, ETYP_IS_BASICTYPE(td2) || ETYP_IS_ENUM(td2));

    int i1 = isIntegerType(td1);
    int i2 = isIntegerType(td2);

    if(i1 == 0 && i2)
        return td1;

    if(i1 && i2 == 0)
        return td2;

    CBasicTypeEnum b1 = ETYP_IS_ENUM(td1) ? BT_INT : td1->e_basicType;
    CBasicTypeEnum b2 = ETYP_IS_ENUM(td2) ? BT_INT : td2->e_basicType;

    if(i1 == 0 && i2 == 0)
        return (b1 >= b2) ? td1 : td2;

    int s1 = getBasicTypeSize(b1);
    int s2 = getBasicTypeSize(b2);

    if(s1 > s2)
        return td1;

    if(s1 < s2)
        return td2;

    //   [unsigned|signed][char|short|_Bool] op [unsigned|signed][char|short|_Bool]
    //-> int
    if(BTYP_IS_SMALLER_INT(b1) && BTYP_IS_SMALLER_INT(b2))
        return &s_numTypeDescs[BT_INT];

    CBasicTypeEnum c1 = (b1 >= b2) ? b1 : b2;
    CBasicTypeEnum c2 = (b1 >= b2) ? b2 : b1;
    CExprOfTypeDesc *ctd1 = (b1 >= b2) ? td1 : td2;

    // ex) long(32bit) + unsigned int(32bit) => unsigned long
    return (isUnsignedIntegerType(c2)) ?
            (isUnsignedIntegerType(c1) ? ctd1 : getUnsignedIntegerType(c1)) : ctd1;
}


/**
 * \brief
 * resolve type for binary arithmetic operators
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_binaryArithOp(CExprOfBinaryNode *expr)
{
    if(checkExprsTypeDescOfChildren((CExpr*)expr) == 0)
        return NULL;
    CExprOfTypeDesc *td = NULL;
    CExpr *e1 = expr->e_nodes[0];
    CExpr *e2 = expr->e_nodes[1];

    CExprOfTypeDesc *td1 = EXPRS_TYPE(e1);
    if(td1 == NULL)
        return NULL;
    CExprOfTypeDesc *td2 = EXPRS_TYPE(e2);
    if(td2 == NULL)
        return NULL;

    CExprCodeEnum ec = EXPR_CODE(expr);
    CExprOfTypeDesc *tdo1 = getRefType(td1);
    CExprOfTypeDesc *tdo2 = getRefType(td2);
    int b1 = isBasicTypeOrEnum(tdo1);
    int b2 = isBasicTypeOrEnum(tdo2);

    if(b1 == 0 || b2 == 0) {
        // allow 'pointer - pointer'
        isCompatiblePointerType(td1, td2, &td, (ec != EC_MINUS));
        if(td == NULL) {
            addError((CExpr*)expr, CERR_042,
                s_CExprCodeInfos[EXPR_CODE(expr)].ec_opeName);
        } else if(ec == EC_MINUS && b1 == 0 && b2 == 0) {
            td = &s_addrIntTypeDesc;
        }
    } else {
        CBasicTypeEnum b1 = getBasicTypeWithEnum(tdo1);
        CBasicTypeEnum b2 = getBasicTypeWithEnum(tdo2);
        int isReal1 = (b1 < BT_FLOAT_IMAGINARY);
        int isReal2 = (b2 < BT_FLOAT_IMAGINARY);
        int isImag1 = (b1 >= BT_FLOAT_IMAGINARY && b1 < BT_FLOAT_COMPLEX);
        int isImag2 = (b2 >= BT_FLOAT_IMAGINARY && b2 < BT_FLOAT_COMPLEX);
        int isPlusMinus = (ec == EC_PLUS || ec == EC_MINUS);

        if(isReal1) {
            if(isReal2)
                td = getPriorNumberType(tdo1, tdo2, ec); // real op real : real
            else {
                if(isPlusMinus) {
                    if(isImag2)
                        td = &s_numTypeDescs[toComplexType(b2)]; //real +- imag : comp
                    else
                        td = td2; //real +- comp : comp
                } else {
                    td = td2; // real */ imag : imag
                              // real */ comp : comp
                }
            }
        } else if(isImag1) {
            if(isPlusMinus) {
                if(isReal2)
                    td = &s_numTypeDescs[toComplexType(b1)]; //imag +- real : comp
                else if(isImag2)
                    td = (b1 >= b2) ? td1 : td2; // imag +- imag : imag
                else
                    td = td2; // imag +- comp : comp
            } else {
                if(isReal2)
                    td = td1; // imag */ real : imag
                else if(isImag2)
                    td = &s_numTypeDescs[toRealType((b1 >= b2 ? b1 : b2))];
                             // imag */ imag : real
                else
                    td = td2; // imag */ comp : comp
            }
        } else {
            if(isReal2 || isImag2)
                td = td1; // comp +-*/ real : comp
                          // comp +-*/ imag : comp
            else
                td = (b1 >= b2) ? td1 : td2; // comp +-*/ comp : comp
        }
    }

    if(td && ETYP_IS_BASICTYPE(td) == 0 &&
        (ETYP_IS_ARRAY(tdo1) || ETYP_IS_ARRAY(tdo2))) {
        // get pointer type of array elem type
        td = ETYP_IS_ARRAY(tdo1) ? tdo1 : tdo2;
        td = setPointerTypeOfArrayElem(td);
    }
    return td;
}


/**
 * \brief
 * judge td is qualified with const
 *
 * @param td
 *      type descriptor
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isConstType(CExprOfTypeDesc *td)
{
    if(td->e_tq.etq_isConst)
        return 1;

    if(ETYP_IS_DERIVED(td))
        return isConstType(td->e_refType);

    return 0;
}


/**
 * \brief
 * judge composTd has const member
 *
 * @param composTd
 *      composite type descriptor
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isMemberConst(CExprOfTypeDesc *composTd)
{
    CExpr *memDecls = getMemberDeclsExpr(composTd);
    if(EXPR_ISNULL(memDecls))
        return 0;

    CCOL_DListNode *ite1, *ite2;
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
            CExprOfTypeDesc *td = EXPR_T(declr->e_nodes[0]);
            if(td->e_tq.etq_isConst)
                return 1;
        }
    }

    return 0;
}


/**
 * \brief
 * get child node of EC_BRACED_EXPR
 *
 * @param e
 *      target node
 * @return
 *      e or child node
 */
PRIVATE_STATIC CExpr*
stripBracedExpr(CExpr *e)
{
    CExprCodeEnum ec = EXPR_CODE(e);
    if(ec == EC_BRACED_EXPR)
        return stripBracedExpr(EXPR_U(e)->e_node);
    else if(ec == EC_EXPRS && EXPR_L_SIZE(e) == 1)
        return stripBracedExpr(exprListHeadData(e));
    return e;
}


/**
 * \brief
 * judge e is pointer or array ref.
 * skip EC_BRACED_EXPR.
 *
 * @param e
 *      target node
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isPointerOrArrayRef(CExpr *e)
{
    e = stripBracedExpr(e);
    CExprCodeEnum ec = EXPR_CODE(e);
    return (ec == EC_POINTER_REF || ec == EC_ARRAY_REF);
}


/**
 * \brief
 * judge e can be lvalue
 *
 * @param e
 *      target node
 * @param td
 *      target node's expression's type
 * @param modifiable
 *      set to 1 if lvalue is modifiable
 * @param ope
 *      operator
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isLValue(CExpr *e, CExprOfTypeDesc *td, int modifiable, CExpr *ope)
{
    if(td == NULL)
        return 0;
    
    e = stripBracedExpr(e);
    td = getRefType(td);
    CExprCodeEnum ec = EXPR_CODE(e);

    if(ec == EC_GCC_COMP_STMT_EXPR)
        return 0;

    if(modifiable) {
        if(td->e_tq.etq_isConst)
            return 0;
        if(ETYP_IS_FUNC(td))
            return 0;
        if(ec == EC_EXPRS && EXPR_L_SIZE(e) > 1) // comma expr
            return 0;
        if(ec == EC_GCC_COMP_STMT_EXPR)
            return 0;
        if(ETYP_IS_VOID(td))
            return 0;
        if(ETYP_IS_COMPOSITE(td) && isMemberConst(td))
            return 0;
        if(ec == EC_COMPOUND_LITERAL)
            return 0;
    }

    if(ETYP_IS_PTR_OR_ARRAY(td) == 0 &&
        isPointerOrArrayRef(e) == 0 &&
        isConstExpr(e, 1) &&
        (modifiable || (ec == EC_COMPOUND_LITERAL)))
        return 0;

    return 1;
}


/**
 * \brief
 * resolve type for assignment operators
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_assignOp(CExprOfBinaryNode *expr)
{
    if(checkExprsTypeDescOfChildren((CExpr*)expr) == 0)
        return NULL;
    CExpr *e1 = expr->e_nodes[0];
    CExpr *e2 = expr->e_nodes[1];
    CExprOfTypeDesc *td1 = EXPRS_TYPE(e1);
    if(td1 == NULL)
        return NULL;
    CExprOfTypeDesc *td2 = EXPRS_TYPE(e2);
    if(td2 == NULL)
        return NULL;

    if(isConstType(td1) && isConstType(td2) == 0) {
        if(EXPR_CODE(e1) == EC_IDENT) {
            addError((CExpr*)expr,
                CERR_043, EXPR_SYMBOL(e1)->e_symName);
        } else {
            addError((CExpr*)expr, CERR_044);
        }

        return NULL;
    }

    if(isLValue(e1, td1, 1, (CExpr*)expr) == 0) {
        addError((CExpr*)expr, CERR_045);
        return NULL;
    }

    CExprOfTypeDesc *td;

    if(EXPR_CODE(expr) == EC_ASSIGN) {
        td = getPriorityTypeForAssignExpr(td1, e1, td2, e2);
        if(td)
            td = td1;
        else {
            // FIXME modified by xcalablemp
            if(isSubArrayRef(e1) || isSubArrayRef(e2)) {
                td = td1;
            }
            else {
                addError((CExpr*)expr, CERR_046);
                DBGDUMPEXPR(e1);
                DBGDUMPEXPR(e2);
                DBGDUMPEXPR(td1);
                DBGDUMPEXPR(td2);
            }
        }
    } else {
        td = resolveType_binaryArithOp(expr);
        if(td)
            td = td1;
    }

    if(isSubArrayRef(e1) || isSubArrayRef(e2)) {
        // subarray assignment cannot be used as expression
        CExpr *p1 = EXPR_PARENT(expr);
        CExprCodeEnum pc1 = p1 ? EXPR_CODE(p1) : EC_UNDEF;
        CExpr *p2 = EXPR_PARENT(p1);
        CExprCodeEnum pc2 = p2 ? EXPR_CODE(p2) : EC_UNDEF;

        if(pc1 != EC_EXPRS || pc2 != EC_EXPR_STMT) {
            addError((CExpr*)expr, CERR_135);
            return NULL;
        }
    }

    return td;
}


/**
 * \brief
 * resolve type for unary arithmetic operators
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_unaryArithOp(CExprOfUnaryNode *expr)
{
    if(checkExprsTypeDescOfChildren((CExpr*)expr) == 0)
        return 0;

    CExprOfTypeDesc *td = EXPRS_TYPE(expr->e_node);
    CExprOfTypeDesc *tdo = getRefType(td);

    if(isScalarOrPointerType(tdo) == 0) {
        addError((CExpr*)expr, CERR_042,
            s_CExprCodeInfos[EXPR_CODE(expr)].ec_opeName);
        return NULL;
    }

    return td;
}


/**
 * \brief
 * resolve type for unary bit operators
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_unaryBitOp(CExprOfUnaryNode *expr)
{
    if(checkExprsTypeDescOfChildren((CExpr*)expr) == 0)
        return 0;

    CExprOfTypeDesc *td = EXPRS_TYPE(expr->e_node);
    CExprOfTypeDesc *tdo = getRefType(td);

    if(isIntegerType(tdo) == 0) {
        addError((CExpr*)expr, CERR_042,
            s_CExprCodeInfos[EXPR_CODE(expr)].ec_opeName);
        return NULL;
    }

    return td;
}


/**
 * \brief
 * resolve type for binary bit operators
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_binaryBitOp(CExprOfBinaryNode *expr)
{
    if(checkExprsTypeDescOfChildren((CExpr*)expr) == 0)
        return 0;

    CExprOfTypeDesc *td1 = EXPRS_TYPE(expr->e_nodes[0]);
    CExprOfTypeDesc *td1o = getRefType(td1);
    CExprOfTypeDesc *td2o = getRefType(EXPRS_TYPE(expr->e_nodes[1]));

    if(isIntegerType(td1o) == 0 || isIntegerType(td2o) == 0) {
        addError((CExpr*)expr, CERR_042,
            s_CExprCodeInfos[EXPR_CODE(expr)].ec_opeName);
        return NULL;
    }

    //   [unsigned|signed][char|short|_Bool] op [unsigned|signed][char|short|_Bool]
    //-> int
    if(BTYP_IS_SMALLER_INT(td1o->e_basicType))
        return &s_numTypeDescs[BT_INT];

    return td1;
}


/**
 * \brief
 * resolve type for funtion arguments
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC int
checkFuncArgs(CExprOfTypeDesc *funcTd, CExpr *funcExpr, CExpr *args)
{
    CExprOfTypeDesc * funcTdo = getRefType(funcTd);
    CExpr *params = funcTdo->e_paramExpr;
    CExprOfSymbol *funcSym = funcTd->e_symbol;
    if(EXPR_ISNULL(funcSym) && EXPR_CODE(funcExpr) == EC_IDENT)
        funcSym = EXPR_SYMBOL(funcExpr);
    const char *funcName = (funcSym ? funcSym->e_symName : "(anonymous)");

    if(funcSym && EXPR_L_ISNULL(params)) {
        if(funcSym == NULL)
            return 1;
        
        funcSym = findSymbolByGroup(funcSym->e_symName, STB_IDENT);
        if(funcSym == NULL)
            return 1;
        CExprOfTypeDesc * rsymTd = EXPRS_TYPE(funcSym);
        if(rsymTd && EXPR_ISERROR(rsymTd) == 0) {
            rsymTd = getRefType(rsymTd);
            if(ETYP_IS_FUNC(rsymTd) && EXPR_ISERROR(rsymTd) == 0) {
                CExpr *rparams = rsymTd->e_paramExpr;
                if(EXPR_L_ISNULL(rparams) == 0) {
                    params = rparams;
                }
            }
        }
    }

    if(EXPR_L_ISNULL(params)) // parameters not defined
        return 1;

    int paramSize = EXPR_L_SIZE(params);
    int argSize = EXPR_L_ISNULL(args) ? 0 : EXPR_L_SIZE(args);

    if(paramSize == 1) {
        CExpr *declr1 = exprListHeadData(params);
        if(EXPR_CODE(declr1) != EC_DECLARATOR)
            return 1;
        CExprOfTypeDesc *ptd = EXPR_T(EXPR_B(declr1)->e_nodes[0]);
        if(ptd == NULL)
            return 1;
        ptd = getRefType(ptd);
        if(ETYP_IS_VOID(ptd) && argSize > 0) {
            addError(funcExpr, CERR_132, funcName);
            return 0;
        }
        return 1;
    }

    CExpr *ptail = exprListTailData(params);
    int hasEllipsis = (EXPR_CODE(ptail) == EC_ELLIPSIS);
    
    if(hasEllipsis == 0) {
        CExprOfTypeDesc *ptd = EXPR_T(EXPR_B(ptail)->e_nodes[0]);
        if(ptd == NULL)
            return 1;
        ptd = getRefType(ptd);
        if(ETYP_IS_GCC_BUILTIN(ptd))
            return 1;
    }

    if((hasEllipsis == 0 && argSize < paramSize) ||
        (hasEllipsis && argSize < paramSize - 1)) {
        addError(funcExpr, CERR_131, funcName);
        return 0;
    }

    if(hasEllipsis == 0 && argSize > paramSize) {
        addError(funcExpr, CERR_132, funcName);
        return 0;
    }

    CCOL_DListNode *ite1, *ite2 = exprListHead(args);
    int idx = 1;
    EXPR_FOREACH(ite1, params) {
        CExpr *declr1 = EXPR_L_DATA(ite1);

        if(EXPR_CODE(declr1) == EC_ELLIPSIS)
            return 1;

        CExpr *arg2 = EXPR_L_DATA(ite2);

        if(EXPR_ISNULL(arg2) || EXPR_CODE(arg2) == EC_TYPE_DESC)
            goto next;

        CExprOfTypeDesc *td1 = resolveType(EXPR_B(declr1)->e_nodes[0]);
        CExprOfTypeDesc *td2 = resolveType(arg2);

        if(td1 == NULL || td2 == NULL ||
            EXPR_ISERROR(td1) || EXPR_ISERROR(td2))
            goto next;

        switch(getImplicitCastResult(td1, td2)) {
        case ICR_OK:
            break;
        case ICR_ERR_INCOMPAT_TYPE:
            addError(funcExpr, CERR_133, idx, funcName);
            return 0;
        case ICR_WRN_INCOMPAT_PTR:
            addWarn(funcExpr, CWRN_023, idx, funcName);
            break;
        case ICR_WRN_INT_FROM_PTR:
            if(isConstZero(arg2) == 0)
                addWarn(funcExpr, CWRN_022, idx, funcName);
            break;
        case ICR_WRN_PTR_FROM_INT:
            if(isConstZero(arg2) == 0)
                addWarn(funcExpr, CWRN_021, idx, funcName);
            break;
        }

      next:
        ++idx;
        ite2 = CCOL_DL_NEXT(ite2);
    }

    return 1;
}


/**
 * \brief
 * resolve type for EC_FUNCTION_CALL
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_functionCall(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;
    CExpr *funcExpr = EXPR_B(expr)->e_nodes[0];
    CExprOfTypeDesc *td = NULL;
    CExprOfTypeDesc *funcTd = resolveType(funcExpr);

    if(funcTd) {
        CExprOfTypeDesc *funcTdo = getRefType(funcTd);
        if(ETYP_IS_FUNC(funcTdo) == 0) {
            CExprOfTypeDesc *td1 = funcTdo->e_typeExpr ?
                getRefType(EXPR_T(funcTdo->e_typeExpr)) : NULL;
            if(ETYP_IS_FUNC(td1) == 0) {
                EXPR_ISERROR(td) = 1;
                addError((CExpr*)expr, CERR_047);
                return NULL;
            } else {
                funcTdo = getRefType(td1); // func pointer -> func
            }
        }
        td = (CExprOfTypeDesc*)funcTdo->e_typeExpr; // func -> return type
    } else {
        if(EXPR_CODE(funcExpr) == EC_IDENT) {
            //function call without declaration
            const char *funcName = EXPR_SYMBOL(funcExpr)->e_symName;
            addWarn((CExpr*)expr, CWRN_003, funcName);
            exprSetExprsType(funcExpr, &s_implicitDeclFuncPtrTypeDesc);
            s_implicitDeclFuncPtrTypeDesc.e_isUsed = 1;
            funcTd = EXPR_T(s_implicitDeclFuncPtrTypeDesc.e_typeExpr);
            funcTd->e_isUsed = 1;
            td = EXPR_T(funcTd->e_typeExpr);
            addSymbolAt(EXPR_SYMBOL(funcExpr), EXPR_PARENT(funcExpr),
                getGlobalSymbolTable(), funcTd, ST_FUNC, 0, 0);
            exprSetExprsType(funcExpr, funcTd);
        } else {
            addError((CExpr*)expr, CERR_047);
            return NULL;
        }
    }

    if(td == NULL)
        return NULL;

    CExpr *args = EXPR_B(expr)->e_nodes[1];
    if(EXPR_ISNULL(args) == 0) {
        CExprOfTypeDesc *argsTd = resolveType(args);
        if(argsTd == NULL)
            return NULL;
    }

    if(checkFuncArgs(funcTd, funcExpr, args) == 0)
        return NULL;

    return td;
}


/**
 * \brief
 * resolve type for comparison operators
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_comparatorOp(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;

    CExpr *e1 = EXPR_B(expr)->e_nodes[0];
    CExpr *e2 = EXPR_B(expr)->e_nodes[1];

    CExprOfTypeDesc *td1 = getRefType(EXPRS_TYPE(e1));
    CExprOfTypeDesc *td2 = getRefType(EXPRS_TYPE(e2));
    int p1 = ETYP_IS_PTR_OR_ARRAY(td1);
    int p2 = ETYP_IS_PTR_OR_ARRAY(td2);
    int i1 = isIntegerType(td1);
    int i2 = isIntegerType(td2);

    if((p1 && i2) || (i1 && p2)) {
        if(isConstZero(e1) == 0 && isConstZero(e2) == 0)
            addWarn(expr, CWRN_019);
    } else if(p1 && p2) {
        if(isCompatiblePointerType(td1, td2, NULL, 0) == 0)
            addWarn(expr, CWRN_020);
    } else if(isScalarOrPointerType(td1) == 0 ||
        isScalarOrPointerType(td2) == 0) {
        addError((CExpr*)expr, CERR_042,
            s_CExprCodeInfos[EXPR_CODE(expr)].ec_opeName);
        return NULL;
    }

    return &s_numTypeDescs[BT_INT];
}


/**
 * \brief
 * resolve type for logical expression
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_logOp(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;

    int err = 0;

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfUnaryNode:
        err = (isLogicalExpr(EXPR_U(expr)->e_node) == 0);
        break;
    case STRUCT_CExprOfBinaryNode:
        err = (isLogicalExpr(EXPR_B(expr)->e_nodes[0]) == 0) ||
                (isLogicalExpr(EXPR_B(expr)->e_nodes[1]) == 0);
        break;
    default:
        break;
    }

    if(err) {
        addError(expr, CERR_127, s_CExprCodeInfos[EXPR_CODE(expr)].ec_opeName);
        return NULL;
    }

    return &s_numTypeDescs[BT_INT];
}


/**
 * \brief
 * resolve type for EC_MEMBER_REF
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_memberRef(CExprOfBinaryNode *memRef)
{
    assertExpr((CExpr*)memRef, EXPR_CODE(memRef) == EC_MEMBER_REF ||
        EXPR_CODE(memRef) == EC_POINTS_AT);
    CExprOfTypeDesc *td = NULL;
    CExpr *composExpr = memRef->e_nodes[0];
    CExprOfTypeDesc *composTd0 = resolveType(composExpr);
    CExprOfTypeDesc *composTd = composTd0;

    if(composTd == NULL) {
        addError((CExpr*)memRef, CERR_049);
        return NULL;
    }

    if(EXPR_ISERROR(composTd))
        return NULL;

  again:

    composTd = getRefType(composTd);

    if(composTd) {
        if(EXPR_ISERROR(composTd))
            return NULL;

        if(EXPR_CODE(memRef) == EC_POINTS_AT) {
            //chkPtr = 1 : first call
            //chkPtr = 0 : recursive call for anoymous member
            int chkPtr = (composTd->e_isAnonMember == 0);
            CExprOfTypeDesc *composTd1 = composTd;

            if(chkPtr) {
                if(ETYP_IS_POINTER(composTd) == 0) {

                    if(ETYP_IS_ARRAY(composTd)) {
                        // 's->a' => '(s + 0)->a'
                        CExprOfTypeDesc *rtd = EXPR_T(composTd->e_typeExpr);
                        CExprOfTypeDesc *rtdo = getRefType(rtd);
                        if(ETYP_IS_COMPOSITE(rtdo)) {
                            CExpr *refExpr = exprBinary(EC_PLUS, composExpr,
                                (CExpr*)allocExprOfNumberConst2(0, BT_INT));
                            memRef->e_nodes[0] = refExpr;
                            EXPR_REF(refExpr);
                            freeExpr(composExpr);
                            composTd = resolveType(refExpr);
                            goto again;
                        }
                    }

                    addError((CExpr*)memRef, CERR_050);
                    return NULL;
                }

                composTd1 = resolveType(composTd->e_typeExpr);

                if(composTd1 == NULL) {
                    addError((CExpr*)memRef, CERR_050);
                    return NULL;
                }
            }

            CExprOfTypeDesc *composTd2 = getRefType(composTd1);
            composTd = composTd2;
        }
    }

    if(composTd == NULL || ETYP_IS_COMPOSITE(composTd) == 0) {
        addError((CExpr*)memRef, CERR_049);
        if(composTd)
            EXPR_ISERROR(composTd) = 1;
        return NULL;
    }

    CExprOfSymbol *sym = (CExprOfSymbol*)memRef->e_nodes[1];
    td = resolveType((CExpr*)sym);

    if(td == NULL) {
        CExprOfTypeDesc *parentTd = NULL;
        CExprOfSymbol *parentSym = NULL;
        td = getMemberType(composTd, sym->e_symName, &parentTd, &parentSym);

        if(td == NULL) {
            addError((CExpr*)sym, CERR_039, sym->e_symName);
            EXPR_ISERROR(composTd) = 1;
            return NULL;
        }

        if(parentTd && parentTd != composTd) {
            // complete anonymous member access
            CExprCodeEnum ec = EXPR_CODE(memRef);
            EXPR_CODE(memRef) = EC_MEMBER_REF;
            CExprOfSymbol *parentSym1 = allocExprOfSymbol(
                EC_IDENT, ccol_strdup(parentSym->e_symName, MAX_NAME_SIZ));
            parentSym1->e_symType = ST_MEMBER;
            CExprOfBinaryNode *parentMemRef = allocExprOfBinaryNode1(
                ec, composExpr, (CExpr*)parentSym1);
            EXPR_REF(parentMemRef);
            freeExpr(composExpr);
            memRef->e_nodes[0] = (CExpr*)parentMemRef;
            resolveType((CExpr*)memRef);
            EXPR_ISCONVERTED(memRef) = 1;
            EXPR_ISGCCSYNTAX(memRef) = 1;
            ++s_numConvsAnonMemberAccess;
        }

        exprSetExprsType((CExpr*)sym, td);
    }

    return td;
}


/**
 * \brief
 * complete subarray bounds and create type of subarray
 *
 * @param aryRef
 *      node of EC_ARRAY_REF
 * @param orgTd
 *      original array type descriptor
 * @param elemTd
 *      array element type descriptor
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
fixSubArrayRef(CExprOfBinaryNode *aryRef, CExprOfTypeDesc *orgTd,
    CExprOfTypeDesc *elemTd)
{
    CExpr *dim = aryRef->e_nodes[1];

    //create variable array type for subarray

    // create array size expr
    //   step=1 : upper - lower + 1
    //   step>1 : (upper - lower) / step + 1
    CExpr *lowerExpr = exprListNextNData(dim, 0);
    CExpr *upperExpr = exprListNextNData(dim, 1);
    CExpr *stepExpr  = exprListNextNData(dim, 2);

    if(resolveType(lowerExpr) == NULL)
        return NULL;
    if(resolveType(stepExpr) == NULL)
        return NULL;

    if(EXPR_ISNULL(upperExpr)) {
        CExpr *upperExpr1 = orgTd->e_len.eln_lenExpr;
        if(EXPR_ISNULL(upperExpr1)) {
            addError((CExpr*)aryRef, CERR_134);
            return NULL;
        }
        upperExpr1 = copyExpr(upperExpr1);
        upperExpr1 = exprBinary(EC_MINUS, upperExpr1,
            (CExpr*)allocExprOfNumberConst2(1, BT_INT));
        EXPR_REF(upperExpr1);
        exprReplace(dim, upperExpr, upperExpr1);
        upperExpr = upperExpr1;
    }

    if(resolveType(upperExpr) == NULL)
        return NULL;

    CExprOfTypeDesc *td = allocExprOfTypeDesc();
    addTypeDesc(td);
    exprCopyLineNum((CExpr*)td, (CExpr*)aryRef);
    td->e_tdKind = TD_ARRAY;
    EXPR_REF(elemTd);
    elemTd->e_isUsed = 1;
    td->e_typeExpr = (CExpr*)elemTd;
    addTypeDesc(elemTd);
    td->e_len.eln_isVariable = 1;

    long step;
    CExpr *len1 = exprBinary(EC_MINUS, upperExpr, lowerExpr);
    exprCopyLineNum(len1, (CExpr*)aryRef);
    CExpr *len2;
    if(getCastedLongValueOfExpr(stepExpr, &step) && step == 1) {
        len2 = len1;
    } else {
        len2 = exprBinary(EC_DIV, len1, stepExpr);
    }
    exprCopyLineNum(len2, (CExpr*)aryRef);
    CExpr *lenExpr = exprBinary(EC_PLUS, len2, (CExpr*)allocExprOfNumberConst2(1, BT_INT));
    exprCopyLineNum(lenExpr, (CExpr*)aryRef);

    EXPR_REF(lenExpr);
    td->e_len.eln_lenExpr = lenExpr;
    td->e_isSizeUnreducable = 1;
    exprListAdd(dim, (CExpr*)td);

    exprSetExprsType((CExpr*)aryRef, td);

    return td;
}


PRIVATE_STATIC void
completeCoarrayRefOfSubArray(CExpr *expr)
{
    CExpr *parent = NULL;
    CExpr *aryExpr = (CExpr*)expr;
    while(aryExpr && EXPR_CODE(aryExpr) == EC_ARRAY_REF) {
        parent = aryExpr;
        aryExpr = EXPR_B(aryExpr)->e_nodes[0];
    }

    if(EXPR_CODE(aryExpr) != EC_IDENT)
        return;

    CExprOfTypeDesc *td = resolveType(aryExpr);
    if(td == NULL)
        return;

    td = getRefType(td);
    if(ETYP_IS_COARRAY(td) == 0)
        return;

    CExpr *coDims = exprList(EC_XMP_COARRAY_DIMENSIONS);
    CExpr *coRef = exprCoarrayRef(aryExpr, coDims);

    exprReplace(parent, aryExpr, coRef);
}


/**
 * \brief
 * resolve type for EC_ARRAY_REF which represents subarray
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
CExprOfTypeDesc*
resolveType_subArrayRef(CExprOfBinaryNode *expr)
{
    assertExpr(expr->e_nodes[1], EXPR_L_SIZE(expr->e_nodes[1]) == 3);

    completeCoarrayRefOfSubArray((CExpr*)expr);

    CCOL_DList aryTds, aryRefs;
    memset(&aryTds, 0, sizeof(aryTds));
    memset(&aryRefs, 0, sizeof(aryRefs));

    // get array of highest dimension in expr
    // and get number of subarray dimensions
    int subAryDims = 0;
    CExpr *aryExpr = (CExpr*)expr;
    while(aryExpr && isSubArrayRef(aryExpr)) {
        ++subAryDims;
        CCOL_DL_ADD(&aryRefs, aryExpr);
        aryExpr = EXPR_B(aryExpr)->e_nodes[0];
    }

    if(aryExpr == NULL) {
        addError((CExpr*)expr, CERR_136);
        return NULL;
    }
    
    assertExpr((CExpr*)expr, aryExpr);
    CExprOfTypeDesc *td0 = resolveType(aryExpr);
    if(td0 == NULL)
        return NULL;

    CExprOfTypeDesc *aryTd = getRefType(td0);

    if(aryTd == NULL || ETYP_IS_ARRAY(aryTd) == 0) {
        addError((CExpr*)expr, CERR_136);
        return NULL;
    }

    // get minimum subarray array element
    int aryDims = 0;
    CExprOfTypeDesc *td = NULL;
    CExprOfTypeDesc *tmpTd = aryTd, *elemTd0 = NULL;

    while(ETYP_IS_ARRAY(tmpTd)) {
        ++aryDims;
        // order of aryTds is reverse to that of aryRefs
        CCOL_DL_CONS(&aryTds, tmpTd);

        if(aryDims == subAryDims) {
            elemTd0 = EXPR_T(tmpTd->e_typeExpr);
            break;
        }

        tmpTd = EXPR_T(tmpTd->e_typeExpr);
        tmpTd = getRefType(tmpTd);
    }

    if(elemTd0 == NULL) {
        addError((CExpr*)expr, CERR_136);
        goto end;
    }

    CCOL_DListNode *ite1, *ite2 = CCOL_DL_HEAD(&aryRefs);
    CExprOfTypeDesc *elemTd = elemTd0;
    int i = 0;

    CCOL_DL_FOREACH(ite1, &aryTds) {
        if(i++ >= subAryDims)
            break;
        CExprOfTypeDesc *orgTd = EXPR_T(CCOL_DL_DATA(ite1));
        CExprOfBinaryNode *aryRef = EXPR_B(CCOL_DL_DATA(ite2));
        elemTd = fixSubArrayRef(aryRef, orgTd, elemTd);
        
        if(elemTd == NULL)
            goto end;

        ite2 = CCOL_DL_NEXT(ite2);
    }

    td = EXPRS_TYPE(expr);

  end:

    CCOL_DL_CLEAR(&aryTds);
    CCOL_DL_CLEAR(&aryRefs);
    return td;
}


/**
 * \brief
 * resolve type for EC_ARRAY_REF
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_arrayRef(CExprOfBinaryNode *expr)
{
    if(isSubArrayRef((CExpr*)expr))
        return resolveType_subArrayRef(expr);

    if(checkExprsTypeDescOfChildren((CExpr*)expr) == 0)
        return NULL;

    CExpr *aryExpr = expr->e_nodes[0];
    assertExpr((CExpr*)expr, aryExpr);
    CExprOfTypeDesc *td0 = EXPRS_TYPE(aryExpr);
    if(td0 == NULL)
        return NULL;

    CExprOfTypeDesc *aryOrPtrTd = getRefType(td0);
    int err = 0;

    if(aryOrPtrTd == NULL) {
        err = 1;
    } else if(ETYP_IS_PTR_OR_ARRAY(aryOrPtrTd) == 0) {
        err = 1;
    }

    if(err) {
        addError((CExpr*)expr, CERR_076);
        return NULL;
    }

    if(ETYP_IS_ARRAY(aryOrPtrTd)) {
        //create pointer type of array elem for outx_ARRAY_REF()
        setPointerTypeOfArrayElem(aryOrPtrTd);
    }
    
    return EXPR_T(aryOrPtrTd->e_typeExpr);
}


/**
 * \brief
 * resolve type for  EC_XMP_COARRAY_REF
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_coArrayRef(CExprOfBinaryNode *expr)
{
    if(checkExprsTypeDescOfChildren((CExpr*)expr) == 0)
        return NULL;
    CExpr *coExpr = expr->e_nodes[0];
    assertExpr((CExpr*)expr, coExpr);
    assertExprCode(coExpr, EC_IDENT);

    CExprOfTypeDesc *coTd = EXPRS_TYPE(coExpr);
    assertExpr(coExpr, coTd);
    CExprOfTypeDesc *coTd0 = getRefType(coTd);

    if(coTd0 == NULL) {
        addError((CExpr*)expr, CERR_077);
        return NULL;
    }

    CExprOfTypeDesc *btd = coTd0;

    while(ETYP_IS_COARRAY(btd))
        btd = getRefType(EXPR_T(btd->e_typeExpr));

    if(btd == NULL ||
        (ETYP_IS_ARRAY(btd) == 0 && ETYP_IS_COMPOSITE(btd) == 0 &&
        isBasicTypeOrEnum(btd) == 0)) {
        addError((CExpr*)expr, CERR_077);
        return NULL;
    }

    CExprOfTypeDesc *td = ETYP_IS_ARRAY(btd) ? btd : allocPointerTypeDesc(btd);

    return td;
}


/**
 * \brief
 * resolve type for EC_POINTER_REF
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_pointerRef(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;
    CExprOfTypeDesc *refTd = EXPRS_TYPE(EXPR_U(expr)->e_node);

    if(refTd == NULL)
        return NULL;

    CExprOfTypeDesc *td = getRefType(refTd);

    if(ETYP_IS_PTR_OR_ARRAY(td) == 0) {
        addError(expr, CERR_078);
        return NULL;
    }

    CExprOfTypeDesc *rtd = getRefType(EXPR_T(td->e_typeExpr));
    if(ETYP_IS_FUNC(rtd)) {
        if(isExprCodeChildOf(expr, EC_FUNCTION_CALL, EXPR_PARENT(expr), NULL)) {
            // function pointer ref -> function pointer
            // ex. typedef void f(); (*f)();
            return td;
        } else {
            //pointer ref of function pointer is void pointer
            return &s_voidPtrTypeDesc;
        }
    }

    return EXPR_T(td->e_typeExpr);
}


/**
 * \brief
 * resolve type for EC_CONDEXPR
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_condExpr(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;
    CExprOfTypeDesc *td1, *td2;
    td2 = EXPRS_TYPE(exprListTailData(expr));
    if(td2 == NULL)
        return NULL;

    CExpr *cond = exprListHeadData(expr);
    CExpr *e1 = exprListNextNData(expr, 1);

    if(EXPR_ISNULL(e1))
        td1 = EXPRS_TYPE(cond);
    else
        td1 = EXPRS_TYPE(e1);

    if(td1 == NULL)
        return NULL;

    if(isLogicalExpr(cond) == 0) {
        addError(expr, CERR_127, s_CExprCodeInfos[EXPR_CODE(expr)].ec_opeName);
        return NULL;
    }

    CExprOfTypeDesc *td = getPriorityType(td1, td2);

    if(td == NULL) {
        addError(expr, CERR_079);
        DBGDUMPEXPR(td1);
        DBGDUMPEXPR(td2);
    }

    return td;
}


/**
 * \brief
 * resolve type for EC_ADDR_OF
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
resolveType_addrOf(CExpr *expr)
{
    if(checkExprsTypeDescOfChildren(expr) == 0)
        return NULL;
    CExpr *node = EXPR_U(expr)->e_node;

    CExprOfTypeDesc *baseTd = EXPRS_TYPE(node);
    if(baseTd == NULL) {
        addError(expr, CERR_080);
        return NULL;
    }

    if(EXPR_CODE(node) == EC_IDENT &&
        EXPR_SYMBOL(node)->e_symType == ST_FUNC &&
        isExprCodeChildOf(expr, EC_TYPE_DESC,
            EXPR_PARENT(expr), NULL) == 0) {
        // &func -> func
        return baseTd;
    }

    if(isLValue(expr, baseTd, 0, expr) == 0) {
        addError(expr, CERR_081);
        return NULL;
    }

    CExprOfTypeDesc *td = allocPointerTypeDesc(baseTd);
    exprCopyLineNum((CExpr*)td, expr);

    return td;
}


/**
 * \brief
 * get array index in specified by designator
 *
 *
 * @param designator
 *      designator
 * @param[out] idx1
 *      array designator
 * @param[out] idx2
 *      array designator's to index (gcc syntax)
 * @return
 *      0:error, 1:ok
 */
PRIVATE_STATIC int
getArrayDesignator(CExpr *designator, int *idx1, int *idx2)
{
    CExpr *e1 = EXPR_B(designator)->e_nodes[0];
    CExpr *e2 = EXPR_B(designator)->e_nodes[1];

    long n;
    if(getCastedLongValueOfExpr(e1, &n)) {
        *idx1 = (int)n;
    } else {
        addError(designator, CERR_082);
        return 0;
    }

    if(EXPR_ISNULL(e2)) {
        *idx2 = *idx1;
    } else if(getCastedLongValueOfExpr(e2, &n)) {
        *idx2 = (int)n;
    } else {
        addError(designator, CERR_083);
        return 0;
    }

    if(*idx1 < 0 || *idx2 < 0) {
        addError(designator, CERR_118);
        return 0;
    }

    return 1;
}


/**
 * \brief
 * get declarators of struct/union members
 *
 * @param td
 *      type descriptor
 * @param[out] declrs
 *      declarators list
 */
PRIVATE_STATIC void
getDeclaratorOfMembers(CExprOfTypeDesc *td, CCOL_DList *declrs)
{
    assertExpr((CExpr*)td, ETYP_IS_COMPOSITE(td));

    CExpr *memDecls = getMemberDeclsExpr(td);
    if(EXPR_ISNULL(memDecls))
        return;

    CCOL_DListNode *ite1, *ite2;
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
            CCOL_DL_ADD(declrs, declr);
            // union's first member
            if(ETYP_IS_UNION(td))
                return;
        }
    }
}


/**
 * \brief
 * get type descriptor of struct/union's first member
 *
 * @param td
 *      type descriptor
 * @return
 *      NULL or member's type descriptor
 */
PRIVATE_STATIC CExprOfTypeDesc*
getFirstMemberRefType(CExprOfTypeDesc *td)
{
    assertExpr((CExpr*)td, ETYP_IS_COMPOSITE(td));

    CExpr *memDecls = getMemberDeclsExpr(td);
    if(EXPR_ISNULL(memDecls))
        return NULL;

    CCOL_DListNode *ite1, *ite2;
    EXPR_FOREACH(ite1, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExpr *mems = memDecl->e_nodes[1];
        assertExpr((CExpr*)memDecl, mems);
        EXPR_FOREACH(ite2, mems) {
            CExprOfBinaryNode *memDeclr = EXPR_B(EXPR_L_DATA(ite2));
            CExprOfBinaryNode *declr = EXPR_B(memDeclr->e_nodes[0]);
            assertExpr((CExpr*)memDeclr, declr);
            CExprOfTypeDesc *mtd = EXPR_T(EXPR_B(declr)->e_nodes[0]);
            return getRefType(mtd);
        }
    }

    return NULL;
}


/**
 * \brief
 * create null initial value
 *
 * @param lvalTd
 *      initial value type descriptor
 * @param linenoExpr
 *      node which hash line number info
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExpr*
getNullInitValue(CExprOfTypeDesc *lvalTd, CExpr *linenoExpr)
{
    lvalTd = getRefType(lvalTd);
    CExpr *e;
    if(ETYP_IS_COMPOSITE(lvalTd)) {
        e = exprList(EC_INITIALIZERS);
        exprCopyLineNum(e, linenoExpr);
        EXPR_C(e)->e_isCompleted = 1;
        CExprOfTypeDesc *mtd = getFirstMemberRefType(lvalTd);
        if(mtd) {
            CExpr *ee = getNullInitValue(mtd, linenoExpr);
            exprListAdd(e, ee);
            EXPR_C(ee)->e_isCompleted = 1;
        }
    } else if(ETYP_IS_ARRAY(lvalTd)) {
        e = exprList(EC_INITIALIZERS);
        exprCopyLineNum(e, linenoExpr);
        EXPR_C(e)->e_isCompleted = 1;

        if(lvalTd->e_isSizeZero == 0) {
            CExprOfTypeDesc *etd = getRefType(EXPR_T(lvalTd->e_typeExpr));
            CExpr *ee = getNullInitValue(etd, linenoExpr);
            exprListAdd(e, ee);
            EXPR_C(ee)->e_isCompleted = 1;
        }
    } else {
        e = (CExpr*)allocExprOfNumberConst2(0, BT_INT);
        exprCopyLineNum(e, linenoExpr);
        EXPR_C(e)->e_isCompleted = 1;
    }

    return e;
}


/**
 * \brief
 * complete initial value as null array value
 *
 * @param elemTd
 *      array element type descriptor
 * @param[out] val
 *      initial value
 * @param toIndex
 *      array index to complete
 * @return
 *      0:not necessary to complete, 1:completed
 */
PRIVATE_STATIC int
completeArrayByNullValue(CExprOfTypeDesc *elemTd, CExpr *val, int toIndex)
{
    int sz = EXPR_L_SIZE(val);
    if(sz >= toIndex + 1)
        return 0;

    for(int i = sz; i <= toIndex; ++i)
        exprListAdd(val, getNullInitValue(elemTd, val));

    return 1;
}


/**
 * \brief
 * complete initial value as null composite value
 *
 * @param td
 *      lvalue type descriptor
 * @param[out] val
 *      initial value
 * @param stopMem
 *      member symbol to stop completion
 * @return
 *      0:not necessary to complete, 1:completed
 */
PRIVATE_STATIC int
completeCompositeByNullValue(CExprOfTypeDesc *td, CExpr *val,
    CExprOfSymbol *stopMem)
{
    assertExpr((CExpr*)td, ETYP_IS_COMPOSITE(td));

    CExpr *memDecls = getMemberDeclsExpr(td);
    assertExpr((CExpr*)td, memDecls);

    int isUnion = ETYP_IS_UNION(td);
    int idx = 0, isCompleted = 0;
    CCOL_DListNode *ite1, *ite2;
    EXPR_FOREACH(ite1, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExpr *mems = memDecl->e_nodes[1];
        assertExpr((CExpr*)memDecl, mems);
        EXPR_FOREACH(ite2, mems) {
            CExprOfBinaryNode *memDeclr = EXPR_B(EXPR_L_DATA(ite2));
            CExprOfBinaryNode *declr = EXPR_B(memDeclr->e_nodes[0]);
            assertExpr((CExpr*)memDeclr, declr);
            CExprOfSymbol *sym = EXPR_SYMBOL(declr->e_nodes[1]);
            if(stopMem && strcmp(sym->e_symName, stopMem->e_symName) == 0)
                goto end;
            CExprOfTypeDesc *mtd = EXPR_T(declr->e_nodes[0]);
            if(idx++ < EXPR_L_SIZE(val)) {
                if(isUnion)
                    return 0;
                continue;
            }
            exprListAdd(val, getNullInitValue(mtd, val));
            isCompleted = 1;

            if(isUnion) {
                isCompleted = 1;
                goto end;
            }
        }
    }

  end:

    return isCompleted;
}


/**
 * \brief
 * add warning for duplicated initialization
 *
 * @param val
 *      initial value
 * @param lineNumExpr
 *      node which has line number info
 */
PRIVATE_STATIC void
warnDuplicatedInit(CExpr *val, CExpr *lineNumExpr)
{
    assert(val);
    if(EXPR_C(val)->e_isCompleted == 0 &&
        EXPR_CODE(val) != EC_INITIALIZERS) {
        addWarn(lineNumExpr, CWRN_004);
    }
}


/**
 * \brief
 * add warning for duplicated initialization in union initializer
 *
 * @param val
 *      initial value
 * @param lineNumExpr
 *      node which has line number info
 * @return
 *      0:no warning, 1:add warning
 */
PRIVATE_STATIC int
warnDuplicatedInitUnion(CExpr *val, CExpr *lineNumExpr)
{
    if(EXPR_C(val)->e_isCompleted == 0 &&
        EXPR_CODE(val) != EC_INITIALIZERS) {
        addWarn(lineNumExpr, CWRN_004);
        return 1;
    } else if(EXPR_CODE(val) == EC_INITIALIZERS) {
        CExprIterator ite;
        EXPR_FOREACH_MULTI(ite, val) {
            if(warnDuplicatedInitUnion(ite.node, lineNumExpr))
                return 1;
        }
    }

    return 0;
}


/**
 * \brief
 * normalize initial value for array value
 *
 * @param lvalTdo
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @param[out] fixVal
 *      normalized initial value 
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC int
fixInitValByDesignator_array(CCOL_DListNode *desigIte, CExpr *designators,
    CExprOfTypeDesc *lvalTd, CExpr *initVal, CExpr **fixVal, int *index)
{
    lvalTd = getRefType(lvalTd);
    CExpr *designator = EXPR_L_DATA(desigIte);

    if(ETYP_IS_PTR_OR_ARRAY(lvalTd) == 0) {
        addError(designator, CERR_084);
        return 0;
    }

    int i1, i2;
    if(getArrayDesignator(designator, &i1, &i2) == 0)
        return 0;

    if(i1 > i2) {
        addError(designator, CERR_085);
        return 0;
    }

    int sz = EXPR_L_SIZE(*fixVal);
    assert(lvalTd->e_typeExpr);
    CExprOfTypeDesc *elemTd = getRefType(EXPR_T(lvalTd->e_typeExpr));
    completeArrayByNullValue(elemTd, *fixVal, i1 - 1);
    CExpr *fixVal1 = NULL;
    desigIte = CCOL_DL_NEXT(desigIte);
    for(int i = i1; i <= i2; ++i) {
        CExpr *cur = NULL;
        if(i < sz) {
            cur = exprListNextNData(*fixVal, i);
            assert(cur);
        }

        if(desigIte) {
            if(cur) {
                fixVal1 = cur;
            } else {
                exprListAdd(*fixVal, (fixVal1 = exprList(EC_INITIALIZERS)));
                exprCopyLineNum(fixVal1, *fixVal);
            }

            int index1 = 0;
            if(fixInitValByDesignator(desigIte, designators,
                elemTd, initVal, &fixVal1, &index1) == 0)
                return 0;
        } else {
            if(fixInitVal(elemTd, initVal, &fixVal1) == 0)
                return 0;

            if(cur) {
                warnDuplicatedInit(cur, fixVal1);
                EXPR_REF(fixVal1);
                exprReplace(*fixVal, cur, fixVal1);
            } else {
                exprListAdd(*fixVal, fixVal1);
            }
        }
    }

    if(i1 != i2)
        EXPR_ISGCCSYNTAX(*fixVal) = 1;

    *index = i2;

    return 1;
}


/**
 * \brief
 * get member declarator's index of specified symbol
 *
 * @param composTdo
 *      composite type descriptor
 * @return
 *      member declarator index or -1
 */
int
getDeclaratorIndexOfMember(CExprOfTypeDesc *composTdo, CExprOfSymbol *sym)
{
    assertExpr((CExpr*)composTdo, ETYP_IS_COMPOSITE(composTdo));

    CExpr *memDecls = getMemberDeclsExpr(composTdo);
    assert(memDecls);
    int idx = 0;
    CCOL_DListNode *ite1, *ite2;
    EXPR_FOREACH(ite1, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExpr *mems = memDecl->e_nodes[1];
        assertExpr((CExpr*)memDecl, mems);
        EXPR_FOREACH(ite2, mems) {
            CExprOfBinaryNode *memDeclr = EXPR_B(EXPR_L_DATA(ite2));
            CExprOfBinaryNode *declr = EXPR_B(memDeclr->e_nodes[0]);
            assertExpr((CExpr*)memDeclr, declr);
            CExprOfSymbol *msym = EXPR_SYMBOL(declr->e_nodes[1]);
            assertExpr((CExpr*)declr, msym);
            if(strcmp(msym->e_symName, sym->e_symName) == 0)
                return idx;
            ++idx;
        }
    }

    ABORT();
    return -1;
}


/**
 * \brief
 * normalize initial value for struct/union value
 *
 * @param lvalTdo
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @param[out] fixVal
 *      normalized initial value 
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC int
fixInitValByDesignator_composite(CCOL_DListNode *desigIte, CExpr *designators,
    CExprOfTypeDesc *lvalTd, CExpr *initVal, CExpr **fixVal, int *index)
{
    if(EXPR_ISERROR(lvalTd)) {
        addError(EXPR_L_DATA(desigIte), CERR_086);
        return 0;
    }

    lvalTd = getRefType(lvalTd);

    if(ETYP_IS_COMPOSITE(lvalTd) == 0 || EXPR_ISERROR(lvalTd)) {
        addError(EXPR_L_DATA(desigIte), CERR_086);
        return 0;
    }

    CExprOfTypeDesc *parentTd;
    CExprOfSymbol *parentSym;
    CExprOfSymbol *designator;

  again:

    parentTd = NULL;
    parentSym = NULL;
    designator = EXPR_SYMBOL(EXPR_L_DATA(desigIte));
    assertExprCode((CExpr*)designator, EC_IDENT);

    CExprOfTypeDesc *memTd = getMemberType(lvalTd, designator->e_symName,
        &parentTd, &parentSym);

    if(memTd == NULL || EXPR_ISERROR(memTd)) {
        addError((CExpr*)designator, CERR_087, designator->e_symName);
        return 0;
    }

    if(parentTd && parentTd != lvalTd) {
        // complete anonymous member designator
        CExprOfSymbol *parentSym1 = duplicateExprOfSymbol(parentSym);
        exprCopyLineNum((CExpr*)parentSym1, (CExpr*)designator);
        exprListCons((CExpr*)parentSym1, designators);
        desigIte = exprListHead(designators);
        ++s_numConvsAnonMemberAccess;
        goto again;
    }

    int mindex = getDeclaratorIndexOfMember(parentTd, designator);
    int isUnion = ETYP_IS_UNION(getRefType(parentTd));
    int isSetMemDesignator = 0;

    if(EXPR_STRUCT(*fixVal) != STRUCT_CExprOfList) {
        warnDuplicatedInitUnion(*fixVal, (CExpr*)initVal);
        CExpr *fixValParent = EXPR_PARENT(*fixVal);
        CExpr *fixVal1 = exprList(EC_INITIALIZERS);
        exprCopyLineNum(fixVal1, *fixVal);
        EXPR_REF(fixVal1);
        if(fixValParent)
            exprReplace(fixValParent, *fixVal, fixVal1);
        else
            free(*fixVal);
        *fixVal = fixVal1;
    }
    
    if(isUnion && mindex > 0) {
        isSetMemDesignator = 1;
        mindex = 0;
    } else {
        completeCompositeByNullValue(lvalTd, *fixVal, designator);
    }

    desigIte = CCOL_DL_NEXT(desigIte);

    if(isUnion) {
        if(desigIte == NULL)
            warnDuplicatedInitUnion(*fixVal, initVal);
    }

    assertExprStruct(*fixVal, STRUCT_CExprOfList);
    int sz = EXPR_L_SIZE(*fixVal);
    CExpr *cur = NULL, *fixVal1 = NULL;
    if(mindex < sz) {
        if(EXPR_STRUCT(*fixVal) == STRUCT_CExprOfList) {
            cur = exprListNextNData(*fixVal, mindex);
        } else {
            DBGDUMPEXPR((CExpr*)designator);
            DBGDUMPEXPR((CExpr*)memTd);
            DBGDUMPEXPR((CExpr*)*fixVal);
            DBGDUMPEXPR((CExpr*)initVal);
            ABORT();
        }
    }

    if(desigIte) {
        if(cur) {
            fixVal1 = cur;
        } else {
            exprListAdd(*fixVal, (fixVal1 = exprList(EC_INITIALIZERS)));
            exprCopyLineNum(fixVal1, *fixVal);
        }

        int index1 = 0;
        if(fixInitValByDesignator(desigIte, designators,
            memTd, initVal, &fixVal1, &index1) == 0)
            return 0;
    } else {
        if(fixInitVal(memTd, initVal, &fixVal1) == 0)
            return 0;

        if(cur) {
            if(isUnion == 0)
                warnDuplicatedInit(cur, fixVal1);
            EXPR_REF(fixVal1);
            exprReplace(*fixVal, cur, fixVal1);
        } else {
            exprListAdd(*fixVal, fixVal1);
        }
    }

    if(isSetMemDesignator) {
        //set member designator
        freeExpr((CExpr*)EXPR_L(*fixVal)->e_symbol);
        EXPR_L(*fixVal)->e_symbol = designator;
        EXPR_REF(designator);
    }

    *index = mindex;

    return 1;
}


/**
 * \brief
 * normalize initial value for scalar value
 *
 * @param lvalTdo
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @param[out] fixVal
 *      normalized initial value 
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC int
fixInitValByDesignator(CCOL_DListNode *desigIte, CExpr *designators,
    CExprOfTypeDesc *lvalTd, CExpr *initVal, CExpr **fixVal, int *index)
{
    CExpr *designator = EXPR_L_DATA(desigIte);

    if(EXPR_CODE(designator) == EC_ARRAY_DESIGNATOR) {
        return fixInitValByDesignator_array(desigIte, designators,
            lvalTd, initVal, fixVal, index);
    } else {
        return fixInitValByDesignator_composite(desigIte, designators,
            lvalTd, initVal, fixVal, index);
    }
}


/**
 * \brief
 * judge initVal is in external definition and
 * constant value or add error
 *
 * @param initVal
 *      initial value
 */
PRIVATE_STATIC int
checkExtDefConst(CExpr *initVal)
{
    if(EXPR_ISERROR(initVal))
        return 0;

    if(getCurrentSymbolTable()->stb_isGlobal == 0)
        return 1;

    if(isConstExpr(initVal, 1))
        return 1;

    addError(initVal, CERR_088);
    return 0;
}


/**
 * \brief
 * normalize initial value for scalar value
 *
 * @param lvalTdo
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @param[out] fixVal
 *      normalized initial value 
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
fixInitVal_scalar(CExprOfTypeDesc *lvalTdo, CExpr *initVal, CExpr **fixVal)
{
    if(checkExtDefConst(initVal) == 0)
        return NULL;

    if(EXPR_CODE(initVal) == EC_INITIALIZERS) {
        int sz = EXPR_L_SIZE(initVal);
        if(sz == 1) {
            // allow scalar initialization by single value in brace
            // ex. int a={1};
            CExpr *initr = exprListHeadData(initVal);
            assertExprCode(initr, EC_INITIALIZER);
            initVal = EXPR_B(initr)->e_nodes[1];
        } else if(sz == 0) {
            addError(initVal, CERR_089);
            return NULL;
        } else {
            addError(initVal, CERR_090);
            return NULL;
        }
    }

    CExprOfTypeDesc *td = resolveType(initVal);
    if(td == NULL)
        return NULL;

    switch(getImplicitCastResult(lvalTdo, td)) {
    case ICR_OK:
        break;
    case ICR_WRN_INCOMPAT_PTR:
        addWarn(initVal, CWRN_006);
        break;
    case ICR_WRN_INT_FROM_PTR:
        if(isConstZero(initVal) == 0)
            addWarn(initVal, CWRN_005);
        break;
    case ICR_WRN_PTR_FROM_INT:
        if(isConstZero(initVal) == 0)
            addWarn(initVal, CWRN_024);
        break;
    case ICR_ERR_INCOMPAT_TYPE:
        addError(initVal, CERR_094);
        return NULL;
    }
    
    td = resolveType(initVal);
    if(td) {
        *fixVal = initVal;
    }

    return td;
}


/**
 * \brief
 * get reference basic type of TD_POINTER, TD_ARRAY
 *
 * @param td
 *      type descriptor
 * @return
 *      basic type
 */
CBasicTypeEnum
getPointerOrArrayElemRefType(CExprOfTypeDesc *td)
{
    td = getRefType(td);
    if(ETYP_IS_PTR_OR_ARRAY(td) == 0)
        return BT_UNDEF;

    CExprOfTypeDesc *etd = getRefType(EXPR_T(td->e_typeExpr));
    if(ETYP_IS_BASICTYPE(etd) == 0)
        return BT_UNDEF;

    return etd->e_basicType;
}


/**
 * \brief
 * normalize initial value for string constant
 *
 * @param lvalTd
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @param[out] fixVal
 *      normalized initial value 
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
fixInitVal_charArray(CExprOfTypeDesc *lvalTdo, CExprOfTypeDesc *valTd,
    CExpr *initVal, CExpr **fixVal)
{
    CBasicTypeEnum elemBt = getPointerOrArrayElemRefType(lvalTdo);

    if(elemBt != BT_CHAR && elemBt != BT_UNSIGNED_CHAR &&
        elemBt != BT_WCHAR && elemBt != s_wcharType)
        return NULL;

    if(elemBt == BT_UNSIGNED_CHAR)
        elemBt = BT_CHAR;

    CExprOfTypeDesc *valTd1 = valTd;
    CExpr *initVal1 = initVal;

    if(EXPR_CODE(initVal1) == EC_INITIALIZERS) {
        if(EXPR_L_SIZE(initVal1) != 1)
            return NULL;

        //treat a[]={STRING_LITERAL} as a[]=STRING_LITERAL
        CExpr *initr1 = exprListHeadData(initVal1);
        initVal1 = EXPR_B(initr1)->e_nodes[1];
        valTd1 = resolveType(initVal1);
    }

    if(ETYP_IS_POINTER(getRefType(valTd1)) == 0)
        return NULL;

    CBasicTypeEnum valElemBt = getPointerOrArrayElemRefType(valTd1);

    if(valElemBt != elemBt &&
        (elemBt != s_wcharType || valElemBt != BT_WCHAR))
        return NULL;

    *fixVal = initVal1;

    if(lvalTdo->e_len.eln_isVariable &&
        EXPR_ISNULL(lvalTdo->e_len.eln_lenExpr)) {
        CExprOfNumberConst *numExpr = allocExprOfNumberConst2(
            EXPR_STRINGCONST(initVal1)->e_numChars + 1, BT_INT);
        EXPR_REF(numExpr);
        freeExpr(lvalTdo->e_len.eln_lenExpr);
        lvalTdo->e_len.eln_isVariable = 0;
        lvalTdo->e_len.eln_lenExpr = (CExpr*)numExpr;
    }

    return valTd;
}


/**
 * \brief
 * normalize initial value for compound value
 *
 * @param lvalTd
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @param[out] fixVal
 *      normalized initial value 
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
fixInitVal_compound(CExprOfTypeDesc *lvalTd, CExpr *initVal, CExpr **fixVal)
{
    CExprOfTypeDesc *lvalTdo = getRefType(lvalTd);
    int isCompos = ETYP_IS_COMPOSITE(lvalTdo);

    if(isCompos && lvalTdo->e_isNoMemDecl) {
        addError(initVal, CERR_116);
        return NULL;
    }

    int arySz = -1, index = 0;
    CCOL_DList *declrs = NULL;
    CCOL_DListNode *ite, *dite = NULL;
    CExprOfTypeDesc *td = NULL, *aryElemTd = NULL, *valTd = NULL;

    if(EXPRS_TYPE(initVal) == NULL) {
        if(EXPR_CODE(initVal) == EC_INITIALIZERS)
            exprSetExprsType(initVal, lvalTd);
        else if(resolveType(initVal) == NULL)
            return NULL;
    }

    valTd = resolveType(initVal);

    if(isCompos == 0) {
        //check if the initialization is for char array by char pointer
        td = fixInitVal_charArray(lvalTdo, valTd, initVal, fixVal);
        if(td)
            return td;
    }

    if(EXPR_CODE(initVal) != EC_INITIALIZERS) {
        if(checkExtDefConst(initVal) == 0)
            return NULL;

        if(isCompos == 0) {
            addError(initVal, CERR_095);
            return NULL;
        }

        if(getPriorityTypeForAssign(lvalTd, valTd) == NULL) {
            addError(initVal, CERR_096);
            return NULL;
        }

        *fixVal = initVal;
        return valTd;
    }

    if(isCompos) {
        declrs = XALLOC(CCOL_DList);
        getDeclaratorOfMembers(lvalTdo, declrs);
        arySz = CCOL_DL_SIZE(declrs);
        dite = CCOL_DL_HEAD(declrs);
    } else {
        if(lvalTdo->e_len.eln_isVariable == 0) {
            CNumValueWithType nvt;
            if(getConstNumValue(lvalTdo->e_len.eln_lenExpr, &nvt) == 0) {
                addError(initVal, CERR_097);
                return NULL;
            }
            arySz = (int)getCastedLongValue(&nvt);
        }
        aryElemTd = getRefType(EXPR_T(lvalTdo->e_typeExpr));
    }

    *fixVal = exprList(EC_INITIALIZERS); 
    exprCopyLineNum(*fixVal, initVal);
    exprSetExprsType(*fixVal, lvalTd);

    EXPR_FOREACH(ite, initVal) {
        CExpr *initr = EXPR_L_DATA(ite);
        assertExprCode(initr, EC_INITIALIZER);
        CExpr *designators = EXPR_B(initr)->e_nodes[0];
        CExpr *minitVal = EXPR_B(initr)->e_nodes[1];
        CExpr *fixVal1 = NULL;
        CExprOfBinaryNode *declr = NULL;

        if(EXPR_ISNULL(designators)) {
            if(arySz > 0 && index >= arySz) {
                addError(initVal, CERR_091);
                goto end;
            }

            CExprOfTypeDesc *mvalTd = NULL;

            if(isCompos) {
                assert(dite);
                declr = EXPR_B(EXPR_L_DATA(dite));
                mvalTd = fixInitVal(EXPR_T(declr->e_nodes[0]), minitVal,
                        &fixVal1);
            } else {
                mvalTd = fixInitVal(aryElemTd, minitVal, &fixVal1);
            }

            if(EXPR_L_SIZE(*fixVal) == index) {
                exprListAdd(*fixVal, fixVal1);
            } else {
                int isCompleted = 0;
                if(isCompos) {
                    isCompleted = completeCompositeByNullValue(
                        lvalTdo, *fixVal, EXPR_SYMBOL(declr->e_nodes[1]));
                } else {
                    isCompleted = completeArrayByNullValue(
                        aryElemTd, *fixVal, index - 1);
                }

                if(isCompleted) {
                    exprListAdd(*fixVal, fixVal1);
                } else {
                    CExpr *cur = exprListNextNData(*fixVal, index);
                    warnDuplicatedInit(cur, fixVal1);
                    EXPR_REF(fixVal1);
                    exprReplace(*fixVal, cur, fixVal1);
                }
            }
            //return after fixVal1 added because of avoiding memory leak
            if(mvalTd == NULL)
                goto end;
        } else {
            CCOL_DListNode *desigIte = exprListHead(designators);
            if(fixInitValByDesignator(desigIte, designators, lvalTd,
                minitVal, fixVal, &index) == 0) {
                goto end;
            }
            if(dite) {
                //update current position
                dite = CCOL_DL_HEAD(declrs);
                dite = CCOL_DL_NEXTN(dite, index);
            }
        }
        if(dite)
            dite = CCOL_DL_NEXT(dite);
        ++index;
    }

    td = lvalTd;

    if(isCompos == 0 && arySz < 0 && td->e_len.eln_isFlexible == 0 &&
        EXPR_ISNULL(td->e_len.eln_lenExpr)) {
        //complete array size
        int sz = EXPR_L_SIZE(*fixVal);
        CExprOfNumberConst *numExpr = allocExprOfNumberConst2(sz, BT_INT);
        EXPR_REF(numExpr);
        freeExpr(td->e_len.eln_lenExpr);
        td->e_len.eln_isVariable = 0;
        td->e_len.eln_lenExpr = (CExpr*)numExpr;
        CExprOfTypeDesc *sizeTd = getSizedType(aryElemTd);
        CExprOfTypeDesc *preDeclTd = td->e_preDeclType;
        td->e_size = sizeTd->e_size * sz;
        td->e_align = sizeTd->e_align;

        if(preDeclTd && preDeclTd->e_size == 0) {
            preDeclTd->e_size = td->e_size;
            preDeclTd->e_align = td->e_align;
        }
    }

  end:

    if(declrs) {
        CCOL_DL_CLEAR(declrs);
        XFREE(declrs);
    }

    return td;
}


/**
 * \brief
 * normalize initial value
 *
 * @param lvalTd
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @param[out] fixVal
 *      normalized initial value 
 * @return
 *      NULL or expression's type
 */
PRIVATE_STATIC CExprOfTypeDesc*
fixInitVal(CExprOfTypeDesc *lvalTd, CExpr *initVal, CExpr **fixVal)
{
    if(initVal == NULL)
        return NULL;

    CExprOfTypeDesc *lvalTdo = getRefType(lvalTd);
    CExprOfTypeDesc *td = NULL;

    if(isBasicTypeOrEnum(lvalTdo) || ETYP_IS_POINTER(lvalTdo)) {
        td = fixInitVal_scalar(lvalTdo, initVal, fixVal);
    } else if(ETYP_IS_ARRAY(lvalTdo) || ETYP_IS_COMPOSITE(lvalTdo)) {
        td = fixInitVal_compound(lvalTdo, initVal, fixVal);
    } else {
        addError(initVal, CERR_092);
        DBGDUMPEXPR(lvalTdo);
    }

    if(td) {
        exprSetExprsType(initVal, td);
        td->e_isUsed = 1;
    } else {
        EXPR_ISERROR(initVal) = 1;
    }

    return td;
}


/**
 * \brief
 * resolve type for initializers
 *
 * @param lvalTd
 *      lvalue type descriptor
 * @param initVal
 *      initial value
 * @return
 *      NULL or expression's type
 */
CExprOfTypeDesc*
resolveType_initVal(CExprOfTypeDesc *lvalTd, CExpr *initVal)
{
    CExpr *parent = EXPR_PARENT(initVal);

    if(EXPR_ISCOMPLETED(parent))
        return EXPRS_TYPE(initVal);
    EXPR_ISCOMPLETED(parent) = 1;

    CExpr *fixVal = NULL;
    CExprOfTypeDesc *td = fixInitVal(lvalTd, initVal, &fixVal);

    if(td) {
        if(initVal != fixVal) {
            EXPR_REF(fixVal);
            exprReplace(parent, initVal, fixVal);
        }
    } else {
        if(fixVal && EXPR_CODE(fixVal) == EC_INITIALIZERS)
            freeExpr(fixVal);
    }

    return td;
}


/**
 * \brief
 * call resolveType against children nodes until error occurring
 *
 * @param expr
 *      taret node
 * @return
 *      0:failed, 1:ok
 */
PRIVATE_STATIC int
checkExprsTypeDescOfChildren(CExpr *expr)
{
    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, expr) {
        CExpr *e = ite.node; 
        if(e && resolveType(e) == 0 && EXPR_ISERROR(e))
            return 0;
    }

    return 1;
}


/**
 * \brief
 * get and set CExprOfTypeDesc to CExprCommon.e_exprsType as
 * type descriptor of the expression.
 *
 * @param expr
 *      target node
 * @return
 *      NULL or expression's type descriptor
 */
CExprOfTypeDesc*
resolveType(CExpr *expr)
{
    int ignore = 0;
    int errSz = CCOL_SL_SIZE(&s_errorList);

    if(EXPR_ISNULL(expr) || EXPR_ISERROR(expr))
        return NULL;

    CExprOfTypeDesc *td = EXPRS_TYPE(expr);
    if(td) {
        if(td->e_isFixed == 0)
            fixTypeDesc(td);
        return td;
    }

    switch(EXPR_CODE(expr)) {
    case EC_TYPE_DESC: {
            td = EXPR_T(expr);
            if(ETYP_IS_TYPEREF(td) || ETYP_IS_GCC_TYPEOF(td))
                convertTypeRefToDerived(td);
            expr = NULL;
        }
        break;
    case EC_CHAR_CONST:
        td = (EXPR_CHARCONST(expr)->e_charType == CT_MB) ?
            &s_charTypeDesc : &s_wideCharTypeDesc;
        break;
    case EC_STRING_CONST:
        td = (EXPR_STRINGCONST(expr)->e_charType == CT_MB) ?
            &s_stringTypeDesc : &s_wideStringTypeDesc;
        break;
    case EC_NUMBER_CONST:
        td = &s_numTypeDescs[EXPR_NUMBERCONST(expr)->e_basicType];
        break;
    case EC_IDENT:
        td = resolveType_ident(expr, 1, &ignore);
        break;
    case EC_CAST:
        td = resolveType_cast(expr);
        break;
    case EC_POINTER_REF:
        td = resolveType_pointerRef(expr);
        break;
    case EC_ADDR_OF:
        td = resolveType_addrOf(expr);
        break;
    case EC_FUNCTION_CALL:
        td = resolveType_functionCall(expr);
        break;
    case EC_ARITH_EQ: case EC_ARITH_NE: case EC_ARITH_GE:
    case EC_ARITH_GT: case EC_ARITH_LE: case EC_ARITH_LT:
        td = resolveType_comparatorOp(expr);
        break;
    case EC_LOG_AND: case EC_LOG_OR: case EC_LOG_NOT:
        td = resolveType_logOp(expr);
        break;
    case EC_UNARY_MINUS:
    case EC_PRE_INCR: case EC_PRE_DECR:
    case EC_POST_INCR: case EC_POST_DECR:
        td = resolveType_unaryArithOp(EXPR_U(expr));
        break;
    case EC_PLUS: case EC_MINUS: case EC_MUL: case EC_DIV: case EC_MOD:
        td = resolveType_binaryArithOp(EXPR_B(expr));
        break;
    case EC_BIT_NOT:
        td = resolveType_unaryBitOp(EXPR_U(expr));
        break;
    case EC_LSHIFT: case EC_RSHIFT:
    case EC_BIT_AND: case EC_BIT_OR: case EC_BIT_XOR:
        td = resolveType_binaryBitOp(EXPR_B(expr));
        break;
    case EC_ASSIGN:
    case EC_ASSIGN_PLUS: case EC_ASSIGN_MINUS:
    case EC_ASSIGN_MUL: case EC_ASSIGN_DIV: case EC_ASSIGN_MOD:
    case EC_ASSIGN_LSHIFT: case EC_ASSIGN_RSHIFT:
    case EC_ASSIGN_BIT_AND: case EC_ASSIGN_BIT_OR: case EC_ASSIGN_BIT_XOR:
        td = resolveType_assignOp(EXPR_B(expr));
        break;
    case EC_POINTS_AT:
    case EC_MEMBER_REF:
        td = resolveType_memberRef(EXPR_B(expr));
        break;
    case EC_ARRAY_REF:
        td = resolveType_arrayRef(EXPR_B(expr));
        break;
    case EC_XMP_COARRAY_REF:
        td = resolveType_coArrayRef(EXPR_B(expr));
        break;
    case EC_ARRAY_DIMENSION:
    case EC_XMP_COARRAY_DIMENSIONS:
        if(checkExprsTypeDescOfChildren(expr))
            ignore = 1;
        break;
    case EC_CONDEXPR:
        td = resolveType_condExpr(expr);
        break;
    case EC_BRACED_EXPR: case EC_GCC_COMP_STMT_EXPR:
        if(checkExprsTypeDescOfChildren(expr)) {
            CExpr *node = EXPR_U(expr)->e_node;
            if(EXPR_ISNULL(node)) {
                if(EXPR_CODE(expr) == EC_BRACED_EXPR)
                    ignore = 1;
                else
                    td = &s_voidTypeDesc;
            } else
                td = resolveType(node);
        }
        break;
    case EC_EXPRS:
        if(EXPR_L_SIZE(expr) == 0)
            td = &s_voidTypeDesc;
        else if(checkExprsTypeDescOfChildren(expr)) {
            CExpr *node = exprListTailData(expr);
            td = EXPRS_TYPE(node);
            if(td == NULL && EXPR_ISERROR(node) == 0)
                ignore = 1;
        }
        break;
    case EC_COMP_STMT:
        if(EXPR_PARENT(expr) &&
            EXPR_CODE(EXPR_PARENT(expr)) == EC_GCC_COMP_STMT_EXPR) {
            CExpr *exprStmt = getLastExprStmt(expr);
            if(exprStmt == NULL) {
                td = &s_voidTypeDesc;
            } else if(checkExprsTypeDescOfChildren(expr)) {
                td = EXPRS_TYPE(EXPR_U(exprStmt)->e_node);
            }
        } else {
            ignore = 1;
        }
        break;
    case EC_EXPR_STMT:
        if(checkExprsTypeDescOfChildren(expr)) {
            CExpr *node = EXPR_U(expr)->e_node;
            td = EXPRS_TYPE(node);
            if(td == NULL && EXPR_ISERROR(node) == 0)
                ignore = 1;
        }
        break;
    case EC_INIT: {
            CExpr *declr = exprListHeadData(EXPR_PARENT(expr));
            ignore = 1;
            if(EXPR_ISERROR(declr) == 0) {
                CExprOfTypeDesc *initTd = resolveType(
                    EXPR_B(declr)->e_nodes[0]);
                if(initTd) {
                    CExpr *initVal = EXPR_U(expr)->e_node;
                    td = resolveType_initVal(initTd, initVal);
                }
            }
        }
        break;
    case EC_COMPOUND_LITERAL: {
            CExprOfTypeDesc *initTd = resolveType(
                EXPR_B(expr)->e_nodes[0]);
            if(initTd) {
                CExpr *initVal = EXPR_B(expr)->e_nodes[1];
                td = resolveType_initVal(initTd, initVal);
            }
        }
        break;
    case EC_GCC_BLTIN_OFFSET_OF:
        td = resolveType_gccBuiltinOffsetOf(expr);
        break;
    case EC_SIZE_OF:
        if(checkExprsTypeDescOfChildren(expr)) {
            CExpr *node = EXPR_U(expr)->e_node;
            CExprOfTypeDesc *td0 = (EXPR_CODE(node) == EC_TYPE_DESC ?
                EXPR_T(node) : EXPRS_TYPE(node));
            if(td0) {
                td0 = getSizedType(resolveType((CExpr*)td0));
                if(td0 && ETYP_IS_UNKNOWN_SIZE(td0))
                    addError(expr, CERR_129);
                else
                    td = &s_numTypeDescs[s_basicTypeSizeOf];
            } else
                td = &s_numTypeDescs[s_basicTypeSizeOf];
        }
        break;
    case EC_GCC_BLTIN_VA_ARG:
        if(checkExprsTypeDescOfChildren(expr))
            td = resolveType(EXPR_B(expr)->e_nodes[1]);
        break;
    case EC_GCC_ALIGN_OF:
    case EC_GCC_BLTIN_TYPES_COMPATIBLE_P:
        if(checkExprsTypeDescOfChildren(expr))
            td = &s_numTypeDescs[BT_INT];
        break;
    case EC_GCC_LABEL_ADDR:
        if(checkExprsTypeDescOfChildren(expr))
            td = EXPRS_TYPE(EXPR_U(expr)->e_node);
        break;
    case EC_UNDEF:
    case EC_END:
        ABORT();
        break;
    default:
        ignore = 1;
        break;
    }

    if(td) {
        if(EXPR_ISERROR(td))
            return NULL;

        exprSetExprsType(expr, td);

        if(td->e_isFixed == 0) {
            fixTypeDesc(td);
            if(EXPR_ISERROR(td))
                return NULL;
        }

        CExprOfTypeDesc *td1 = expr ? resolveType((CExpr*)td) : NULL;

        if(td1)
            return td1;
    } else if(ignore == 0) {
        EXPR_ISERROR(expr) = 1;
        if(errSz == 0 && CCOL_SL_SIZE(&s_errorList) == 0) {
#ifdef CEXPR_DEBUG
            DBGDUMPEXPR(expr);
            DBGDUMPERROR();
            ABORT();
#endif
            addFatal(expr, CFTL_002);
        }
    }

    return td;
}


/**
 * \brief
 * get function type in speicified type descriptor
 *
 * @param td
 *      type descriptor
 * @param[out] parentTd
 *      parent type descriptor of function type descriptor
 * @return
 *      NULL or function type descriptor
 */
CExprOfTypeDesc*
getFuncType(CExprOfTypeDesc *td, CExprOfTypeDesc **parentTd)
{
    assert(td);
    CExprOfTypeDesc *tdo = getRefTypeWithoutFix(td);

    switch(tdo->e_tdKind) {
    case TD_FUNC:
        return td;
    case TD_POINTER:
    case TD_ARRAY:
        if(tdo->e_typeExpr == NULL)
            return NULL;
        if(parentTd)
            *parentTd = td;
        return getFuncType(EXPR_T(tdo->e_typeExpr), parentTd);
    default:
        return NULL;
    }
}


/**
 * \brief
 * sub function of getOrCreateCmpType()
 */
PRIVATE_STATIC CExprOfTypeDesc*
getOrCreateCmpType0(CExprOfTypeDesc *td, CTypeQual *tq)
{
    if(tq) {
        if(td->e_tq.etq_isConst)
            tq->etq_isConst = 1;
        if(td->e_tq.etq_isVolatile)
            tq->etq_isVolatile = 1;
        if(td->e_tq.etq_isRestrict)
            tq->etq_isRestrict = 1;
    }

    CExprOfTypeDesc *bottomType = NULL;

    switch(td->e_tdKind) {
    case TD_DERIVED:
        return getOrCreateCmpType0(td->e_refType, tq);
    case TD_STRUCT:
    case TD_UNION:
    case TD_ENUM:
        if(td->e_refType) {
            //tagged type's refType does not derive type qual
            CExprOfTypeDesc *rtd = getOrCreateCmpType0(td->e_refType, NULL);
            if(tq)
                rtd->e_tq = *tq;
            return rtd;
        }
        bottomType = td;
        break;
    case TD_TYPEREF:
    case TD_GCC_TYPEOF:
        ABORT();
        break;
    default:
        break;
    }

    CExprOfTypeDesc *rtd = duplicateExprOfTypeDesc(td);
    rtd->e_isTemporary = 1;
    if(tq)
        rtd->e_tq = *tq;
    rtd->e_refType = bottomType;

    return rtd;
}


/**
 * \brief
 * get or create type descriptor for comparison
 *
 * @param td
 *      original type descriptor
 * @return
 *      NULL or created type descriptor
 */
PRIVATE_STATIC CExprOfTypeDesc*
getOrCreateCmpType(CExprOfTypeDesc *td)
{
    switch(td->e_tdKind) {
    case TD_TYPEREF:
    case TD_GCC_TYPEOF:
    case TD_DERIVED:
    case TD_STRUCT:
    case TD_UNION:
    case TD_ENUM:
        break;
    default:
        return NULL;
    }

    CTypeQual tq;
    memset(&tq, 0, sizeof(tq));
    CExprOfTypeDesc *refTd = NULL;
    tq = td->e_tq;

    switch(td->e_tdKind) {
    case TD_TYPEREF:
    case TD_GCC_TYPEOF:
        if(EXPR_CODE(td->e_typeExpr) == EC_TYPE_DESC) {
            refTd = EXPR_T(td->e_typeExpr);
        } else {
            assert(EXPRS_TYPE(td->e_typeExpr));
            refTd = EXPRS_TYPE(td->e_typeExpr);
        }
        break;
    case TD_STRUCT:
    case TD_UNION:
    case TD_ENUM:
        refTd = td->e_refType;
        if(refTd == NULL)
            refTd = td;
        break;
    case TD_DERIVED:
        refTd = td->e_refType;
        if(refTd == NULL)
            refTd = td;
    default:
        break;
    }

    return getOrCreateCmpType0(refTd, &tq);
}


/**
 * \brief
 * judge array types are compatible
 *
 * @param td1
 *      type descriptor 1
 * @param e1
 *      expression 1
 * @param td2
 *      type descriptor 2
 * @param e2
 *      expression 2
 * @param rlevel
 *      comparison restrict level 
 * @param checkGccAttr
 *      set to 1 to treat different types when gcc attributes set 
 * @return
 *      comparison result type 
 */
PRIVATE_STATIC CCompareTypeEnum
compareArrayType(CExprOfTypeDesc *td1, CExpr *e1,
    CExprOfTypeDesc *td2, CExpr *e2,
    CTypeRestrictLevel rlevel, int checkGccAttr)
{
    CCompareTypeEnum r = compareType0(
        EXPR_T(td1->e_typeExpr), e1,
        EXPR_T(td2->e_typeExpr), e2,
        rlevel, checkGccAttr);

    if(r != CMT_EQUAL)
        return r;

    if(rlevel == TRL_EXTERN || rlevel == TRL_PARAM)
        return CMT_EQUAL;

    // no need to compare array len qualifier

    int isVariable1 = td1->e_len.eln_isVariable;
    int isVariable2 = td2->e_len.eln_isVariable;
    int isSubArray1 = e1 ? isSubArrayRef(e1) : 0;
    int isSubArray2 = e2 ? isSubArrayRef(e2) : 0;

    if(isVariable1 == 0 && isVariable2 == 0 &&
        isSubArray1 == 0 && isSubArray2 == 0 &&
        (td1->e_len.eln_lenExpr == NULL ||
        td2->e_len.eln_lenExpr == NULL ||
        td1->e_len.eln_len != td2->e_len.eln_len))
        return CMT_DIFF_ARRAYLEN;

    return CMT_EQUAL;
}


/**
 * \brief
 * judge coarray types are compatible
 *
 * @param td1
 *      type descriptor 1
 * @param e1
 *      expression 1 (allow NULL)
 * @param td2
 *      type descriptor 2
 * @param e2
 *      expression 2 (allow NULL)
 * @param rlevel
 *      comparison restrict level 
 * @param checkGccAttr
 *      set to 1 to treat different types when gcc attributes set 
 * @return
 *      comparison result type 
 */
PRIVATE_STATIC CCompareTypeEnum
compareCoarrayType(CExprOfTypeDesc *td1, CExpr *e1,
    CExprOfTypeDesc *td2, CExpr *e2,
    CTypeRestrictLevel rlevel, int checkGccAttr)
{
    int isVariable = td1->e_len.eln_isVariable;
    if(isVariable != td2->e_len.eln_isVariable)
        return CMT_DIFF_ARRAYLEN;

    if(isVariable == 0 &&
        td1->e_len.eln_len != td2->e_len.eln_len) {
        return CMT_DIFF_ARRAYLEN;
    }

    return compareType0(
        EXPR_T(td1->e_typeExpr), e1,
        EXPR_T(td2->e_typeExpr), e2, rlevel, checkGccAttr);
}


/**
 * \brief
 * judge function types are compatible
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @param rlevel
 *      comparison restrict level 
 * @param checkGccAttr
 *      set to 1 to treat different types when gcc attributes set 
 * @return
 *      comparison result type 
 */
PRIVATE_STATIC CCompareTypeEnum
compareFuncType(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2,
    CTypeRestrictLevel rlevel, int checkGccAttr)
{
    assertExpr((CExpr*)td1, ETYP_IS_FUNC(td1));
    CExpr *paramExpr1 = td1->e_paramExpr;
    CExpr *paramExpr2 = td2->e_paramExpr;

    int sz1 = EXPR_L_SIZE(paramExpr1);
    int sz2 = EXPR_L_SIZE(paramExpr2);

    if(sz1 == 0 || sz2 == 0)
        return CMT_EQUAL;

    if(sz1 != sz2)
        return CMT_DIFF_FUNCPARAM;

    CCOL_DListNode *ite1 = CCOL_DL_HEAD(EXPR_DLIST(paramExpr1));
    CCOL_DListNode *ite2 = CCOL_DL_HEAD(EXPR_DLIST(paramExpr2));
    for(; ite1; ite1 = CCOL_DL_NEXT(ite1), ite2 = CCOL_DL_NEXT(ite2)) {

        CExpr *pdeclr1 = EXPR_L_DATA(ite1);
        CExpr *pdeclr2 = EXPR_L_DATA(ite2);
        int ec1 = EXPR_CODE(pdeclr1);
        int ec2 = EXPR_CODE(pdeclr2);

        if(ec1 == EC_ELLIPSIS && ec2 == EC_ELLIPSIS)
            continue;
        if(ec1 == EC_ELLIPSIS && ec2 != EC_ELLIPSIS)
            return CMT_DIFF_FUNCPARAM;
        if(ec1 != EC_ELLIPSIS && ec2 == EC_ELLIPSIS)
            return CMT_DIFF_FUNCPARAM;

        CExprOfTypeDesc *ptd1 = EXPR_T(EXPR_B(pdeclr1)->e_nodes[0]);
        CExprOfTypeDesc *ptd2 = EXPR_T(EXPR_B(pdeclr2)->e_nodes[0]);

        if(ptd1 == NULL && ptd2 == NULL)
            continue;
        if((ptd1 == NULL && ptd2) || (ptd1 && ptd2 == NULL))
            return CMT_DIFF_FUNCPARAM;

        CCompareTypeEnum ct = compareType0(
            ptd1, NULL, ptd2, NULL, TRL_PARAM, checkGccAttr);
        if(ct != CMT_EQUAL)
            return CMT_DIFF_FUNCPARAM;
    }

    CCompareTypeEnum ct = compareType0(
        EXPR_T(td1->e_typeExpr), NULL,
        EXPR_T(td2->e_typeExpr), NULL, rlevel, checkGccAttr);
    if(ct != CMT_EQUAL)
        return CMT_DIFF_FUNCRETURN;
    return CMT_EQUAL;
}


/**
 * \brief
 * judge basic types are same type exclude signed/unsigned flag
 *
 * @param b1
 *      basic type 1
 * @param b2
 *      basic type 2
 * @return
 *      0:differ, 1:equals
 */
PRIVATE_STATIC int
isNoSignBasicTypeEquals(CBasicTypeEnum b1, CBasicTypeEnum b2)
{
    switch(b1) {
    case BT_VOID:
        return 1;
    case BT_FLOAT:
    case BT_DOUBLE:
    case BT_LONGDOUBLE:
    case BT_FLOAT_IMAGINARY:
    case BT_DOUBLE_IMAGINARY:
    case BT_LONGDOUBLE_IMAGINARY:
    case BT_FLOAT_COMPLEX:
    case BT_DOUBLE_COMPLEX:
    case BT_LONGDOUBLE_COMPLEX:
    case BT_WCHAR:
    case BT_BOOL:
        break;
    case BT_CHAR:
    case BT_UNSIGNED_CHAR:
        return (b2 == BT_CHAR) || (b2 == BT_UNSIGNED_CHAR);
    case BT_SHORT:
    case BT_UNSIGNED_SHORT:
        return (b2 == BT_SHORT) || (b2 == BT_UNSIGNED_SHORT);
    case BT_INT:
    case BT_UNSIGNED_INT:
        return (b2 == BT_INT) || (b2 == BT_UNSIGNED_INT);
    case BT_LONG:
    case BT_UNSIGNED_LONG:
        return (b2 == BT_LONG) || (b2 == BT_UNSIGNED_LONG);
    case BT_LONGLONG:
    case BT_UNSIGNED_LONGLONG:
        return (b2 == BT_LONGLONG) || (b2 == BT_UNSIGNED_LONGLONG);
    case BT_UNDEF:
    case BT_END:
        ABORT();
    }

    return (b1 == b2);
}


/**
 * \brief
 * judge types are compatible
 *
 * @param td1
 *      type descriptor 1
 * @param e1
 *      expression 1 (allow NULL)
 * @param td2
 *      type descriptor 2
 * @param e2
 *      expression 2 (allow NULL)
 * @param rlevel
 *      comparison restrict level 
 * @param checkGccAttr
 *      set to 1 to treat different types when gcc attributes set 
 * @return
 *      comparison result type 
 */
PRIVATE_STATIC CCompareTypeEnum
compareType0(CExprOfTypeDesc *td1, CExpr *e1,
    CExprOfTypeDesc *td2, CExpr *e2,
    CTypeRestrictLevel rlevel, int checkGccAttr)
{
    assert(td1);
    assert(td2);

    if(td1->e_isFixed == 0)
        td1 = resolveType((CExpr*)td1);
    if(td2->e_isFixed == 0)
        td2 = resolveType((CExpr*)td2);

    if(td1 == NULL || td2 == NULL || EXPR_ISERROR(td1) || EXPR_ISERROR(td2))
        return CMT_DIFF_TYPE;

    if(td1 == td2)
        return CMT_EQUAL;

    if(checkGccAttr && (hasGccAttr(td1, GAK_ALL) || hasGccAttr(td2, GAK_ALL)))
        return CMT_DIFF_TYPE;

    CExprOfTypeDesc *tmpTd1 = getOrCreateCmpType(td1);
    CExprOfTypeDesc *tmpTd2 = getOrCreateCmpType(td2);

    if(tmpTd1)
        td1 = tmpTd1;
    if(tmpTd2)
        td2 = tmpTd2;

    CCompareTypeEnum ct = CMT_EQUAL;

    if(rlevel >= TRL_EXTERN &&
        isTypeQualEqualsExcludeInline(td1, td2) == 0) {
        ct = CMT_DIFF_TYPEQUAL;
        goto end;
    }

    switch(rlevel) {
    case TRL_PARAM:
        if((ETYP_IS_ARRAY(td1) || ETYP_IS_ARRAY(td2)) &&
            (td1->e_tdKind != td2->e_tdKind)) {
            CExprOfTypeDesc *aryTd = ETYP_IS_ARRAY(td1) ? td1 : td2;
            CExprOfTypeDesc *notAryTd = (aryTd == td1) ? td2 : td1;

            if(ETYP_IS_POINTER(notAryTd)) {
                ct = compareType0(
                    EXPR_T(aryTd->e_typeExpr), e1,
                    EXPR_T(notAryTd->e_typeExpr), e2, TRL_MAX, checkGccAttr);
            } else {
                ct = CMT_DIFF_TYPE;
            }
            goto end;
        }
        // no break
    default:
        if((ETYP_IS_ENUM(td1) && ETYP_IS_INT(td2)) ||
            (ETYP_IS_INT(td1) && ETYP_IS_ENUM(td2)))
            goto end;
        if(td1->e_tdKind != td2->e_tdKind) {
            ct = CMT_DIFF_TYPE;
            goto end;
        }
        break;
    }

    switch(td1->e_tdKind) {
    case TD_BASICTYPE:
        if(isNoSignBasicTypeEquals(td1->e_basicType, td2->e_basicType) == 0)
            ct = CMT_DIFF_TYPE;
        break;
    case TD_STRUCT:
    case TD_UNION:
    case TD_ENUM:
        if(td1->e_refType != td2->e_refType)
            ct = CMT_DIFF_TYPE;
        break;
    case TD_POINTER:
        ct = compareType0(
            EXPR_T(td1->e_typeExpr), e1,
            EXPR_T(td2->e_typeExpr), e2, rlevel, checkGccAttr);
        break;
    case TD_ARRAY:
        ct = compareArrayType(td1, e1, td2, e2, rlevel, checkGccAttr);
        break;
    case TD_FUNC:
        ct = compareFuncType(td1, td2, rlevel, checkGccAttr);
        break;
    case TD_GCC_BUILTIN:
        if(strcmp(EXPR_SYMBOL(td1->e_typeExpr)->e_symName,
            EXPR_SYMBOL(td2->e_typeExpr)->e_symName) != 0)
            ct = CMT_DIFF_TYPE;
        break;
    case TD_COARRAY:
        ct = compareCoarrayType(td1, e1, td2, e2, rlevel, checkGccAttr);
        break;
    case TD_FUNC_OLDSTYLE:
    case TD_UNDEF:
    case TD_END:
    case TD_TYPEREF:
    case TD_DERIVED:
    case TD_GCC_TYPEOF:
    case TD_GCC_BUILTIN_ANY:
        ABORT();
    }

  end:

    if(tmpTd1 && tmpTd1->e_isTemporary)
        freeExpr((CExpr*)tmpTd1);
    if(tmpTd2 && tmpTd2->e_isTemporary)
        freeExpr((CExpr*)tmpTd2);

    return ct;
}


/**
 * \brief
 * judge types are compatible
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @return
 *      comparison result type 
 */
CCompareTypeEnum
compareType(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2)
{
    return compareType0(td1, NULL, td2, NULL,
        (td1->e_sc.esc_isExtern || td2->e_sc.esc_isExtern) ? TRL_EXTERN : TRL_MAX, 1);
}


/**
 * \brief
 * judge types are compatible exclude gcc attributes
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @return
 *      comparison result type 
 */
CCompareTypeEnum
compareTypeExcludeGccAttr(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2)
{
    return compareType0(td1, NULL, td2, NULL,
        (td1->e_sc.esc_isExtern || td2->e_sc.esc_isExtern) ? TRL_EXTERN : TRL_DATADEF, 0);
}


/**
 * \brief
 * judge types are compatible for assignment expression
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @return
 *      comparison result type 
 */
CCompareTypeEnum
compareTypeForAssign(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2)
{
    return compareType0(td1, NULL, td2, NULL, TRL_ASSIGN, 0);
}


/**
 * \brief
 * judge td is type descriptof of void pointer
 *
 * @param td
 *      target type descriptor
 * @return
 *      0:no, 1:yes
 */
int
isVoidPtrType(CExprOfTypeDesc *td)
{
    td = getRefType(td);
    if(ETYP_IS_PTR_OR_ARRAY(td) == 0)
        return 0;
    CExprOfTypeDesc *ptd = getRefType(EXPR_T(td->e_typeExpr));
    return (ptd && ETYP_IS_VOID(ptd));
}


/**
 * \brief
 * judge type descriptors are compatible pointer type or
 * can be operands for binary operators
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @param[out] rtd
 *      prior type descriptor
 * @param inhibitPtrAndPtr
 *      inhibit comparison between pointer
 * @return
 *      0:incompatible, 1:compatible
 */
int
isCompatiblePointerType(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2,
    CExprOfTypeDesc **rtd, int inhibitPtrAndPtr)
{
    assert(td1);
    assert(td2);

    if(rtd)
        *rtd = NULL;

    CExprOfTypeDesc *tdo1 = getRefType(td1);
    CExprOfTypeDesc *tdo2 = getRefType(td2);

    int b1 = isBasicTypeOrEnum(tdo1);
    int b2 = isBasicTypeOrEnum(tdo2);

    if(b1 && b2) {
        if(rtd)
            *rtd = (getBasicTypeWithEnum(tdo1) >= getBasicTypeWithEnum(tdo2)) ? tdo1 : tdo2;
        return 1;
    }

    int p1 = ETYP_IS_POINTER(tdo1);
    int p2 = ETYP_IS_POINTER(tdo2);
    int a1 = ETYP_IS_ARRAY(tdo1);
    int a2 = ETYP_IS_ARRAY(tdo2);

    if(p1 == 0 && p2 == 0 && a1 == 0 && a2 == 0) 
        return 0;

    if(inhibitPtrAndPtr && (p1 || a1) && (p2 || a2))
        return 0;

    int i1 = isIntegerType(tdo1);
    int i2 = isIntegerType(tdo2);

    if(i1) {
        if(p2 || a2) {
            // integer is compatible with pointer
            if(rtd)
                *rtd = td2;
            return 1;
        } else
            return 0;
    }

    if(i2) {
        if(p1 || a1) {
            // integer is compatible with pointer
            if(rtd)
                *rtd = td1;
            return 1;
        } else
            return 0;
    }

    // here, td1 and td2 are not integer type

    if((p1 && p2 == 0) || (p1 == 0 && p2)) {

        CExprOfTypeDesc *ptdo1 = (p1 || a1) ? getRefType(EXPR_T(tdo1->e_typeExpr)) : NULL;
        CExprOfTypeDesc *ptdo2 = (p2 || a2) ? getRefType(EXPR_T(tdo2->e_typeExpr)) : NULL;

        int v1 = ptdo1 ? ETYP_IS_VOID(ptdo1) : 0;
        int v2 = ptdo2 ? ETYP_IS_VOID(ptdo2) : 0;

        if(p1 && a2 && (ptdo1->e_tdKind == ptdo2->e_tdKind || v1)) {
            if(rtd) {
                if(v1) {
                    *rtd = allocPointerTypeDesc(ptdo2);
                    EXPR_REF(*rtd);
                    CCOL_SL_CONS(&s_exprsTypeDescList, *rtd);
                } else
                    *rtd = td1;
            }
            return 1;
        } else if(p2 && a1 && (ptdo1->e_tdKind == ptdo2->e_tdKind || v2)) {
            if(rtd) {
                if(v2) {
                    *rtd = allocPointerTypeDesc(ptdo1);
                    EXPR_REF(*rtd);
                    CCOL_SL_CONS(&s_exprsTypeDescList, *rtd);
                } else
                    *rtd = td2;
            }
            return 1;
        }

        return 0;
    }

    // here, (p1 == p2)

    if(p1 == 0 && p2 == 0 && (a1 == 0 || a2 == 0))
        return 0;

    // here, (p1 && p2) || (a1 && a2)
    int v1 = isVoidPtrType(tdo1);
    int v2 = isVoidPtrType(tdo2);

    if(v1 || v2) {
        if(rtd) {
            if(v1)
                *rtd = td2;
            else
                *rtd = td1;
        }
        return 1;
    }

    CExprOfTypeDesc *ptdo1 = getRefType(EXPR_T(tdo1->e_typeExpr));
    CExprOfTypeDesc *ptdo2 = getRefType(EXPR_T(tdo2->e_typeExpr));
    if(compareTypeForAssign(ptdo1, ptdo2) == CMT_EQUAL) {
        if(rtd)
            *rtd = td1;
        return 1;
    }

    return 0;
}


/**
 * \brief
 * get prior type descriptor
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @return
 *      prior type descriptor
 */
CExprOfTypeDesc*
getPriorityType(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2)
{
    assert(td1);
    assert(td2);

    CExprOfTypeDesc *rtd;

    if(isCompatiblePointerType(td1, td2, &rtd, 0))
        return rtd;

    CExprOfTypeDesc *tdo1 = getRefType(td1);
    CExprOfTypeDesc *tdo2 = getRefType(td2);

    if(ETYP_IS_POINTER(tdo1) || ETYP_IS_POINTER(tdo2))
        return NULL;

    if(compareTypeExcludeGccAttr(td1, td2) == CMT_EQUAL)
        return td1;
    
    return NULL;
}


/**
 * \brief
 * get prior type descriptor for assignment expression
 * such as subarray assignment.
 *
 * @param td1
 *      type descriptor 1
 * @param e1
 *      expression 1
 * @param td2
 *      type descriptor 2
 * @param e2
 *      expression 2
 * @return
 *      prior type descriptor
 */
CExprOfTypeDesc*
getPriorityTypeForAssignExpr(CExprOfTypeDesc *td1, CExpr *e1,
    CExprOfTypeDesc *td2, CExpr *e2)
{
    assert(td1);
    assert(td2);

    CExprOfTypeDesc *tdo1 = getRefType(td1);
    CExprOfTypeDesc *tdo2 = getRefType(td2);
    int a1 = ETYP_IS_ARRAY(tdo1);
    int a2 = ETYP_IS_ARRAY(tdo2);

    if(a1 && a2 == 0)
        return NULL;

    if(a1 && a2 && (isSubArrayRef(e1) == 0 || isSubArrayRef(e2) == 0))
        return NULL;

    CExprOfTypeDesc *rtd;

    if(isCompatiblePointerType(td1, td2, &rtd, 0))
        return rtd;

    if(ETYP_IS_POINTER(tdo1) || ETYP_IS_POINTER(tdo2))
        return NULL;

    int rlevel = (td1->e_sc.esc_isExtern || td2->e_sc.esc_isExtern) ?
        TRL_EXTERN : TRL_DATADEF;

    CCompareTypeEnum cmt = compareType0(td1, e1, td2, e2, rlevel, 0);

    switch(cmt) {
    case CMT_DIFF_TYPEQUAL:
    case CMT_EQUAL:
        return td1;
    default:
        break;
    }

    return NULL;
}


/**
 * \brief
 * get prior type descriptor for assignment expression
 *
 * @param td1
 *      type descriptor 1
 * @param td2
 *      type descriptor 2
 * @return
 *      prior type descriptor
 */
CExprOfTypeDesc*
getPriorityTypeForAssign(CExprOfTypeDesc *td1, CExprOfTypeDesc *td2)
{
    return getPriorityTypeForAssignExpr(td1, NULL, td2, NULL);
}


/**
 * \brief
 * alloc pointer type descriptor
 *
 * @param refType
 *      type descriptor to reference
 * @return
 *      allocated type descriptor
 */
CExprOfTypeDesc*
allocPointerTypeDesc(CExprOfTypeDesc *refType)
{
    CExprOfTypeDesc *td = allocExprOfTypeDesc();
    exprCopyLineNum((CExpr*)td, (CExpr*)refType);
    td->e_tdKind = TD_POINTER;
    td->e_typeExpr = (CExpr*)refType;
    EXPR_REF(refType);

    return td;
}


/**
 * \brief
 * alloc type descriptor of TD_DERIVED
 *
 * @param td
 *      type descriptor to reference
 * @return
 *      allocated type descriptor
 */
CExprOfTypeDesc*
allocDerivedTypeDesc(CExprOfTypeDesc *td)
{
    CExprOfTypeDesc *dtd = allocExprOfTypeDesc();
    exprCopyLineNum((CExpr*)dtd, (CExpr*)td);
    dtd->e_tdKind = TD_DERIVED;
    dtd->e_refType = td;
    dtd->e_paramExpr = (CExpr*)td;
    EXPR_REF(td);
    dtd->e_tq = td->e_tq;
    dtd->e_sc = td->e_sc;
    dtd->e_isTypeDef = td->e_isTypeDef;
    memset(&td->e_tq, 0, sizeof(td->e_tq));
    memset(&td->e_sc, 0, sizeof(td->e_sc));

    return dtd;
}


/**
 * \brief
 * create pointer type of array elem and save it to array type descriptor
 *
 * @param aryTd
 *      type descriptor of TD_ARRAY
 * @return
 *      allocated type descriptor
 */
CExprOfTypeDesc*
setPointerTypeOfArrayElem(CExprOfTypeDesc *aryTd)
{
    assertExpr((CExpr*)aryTd, ETYP_IS_ARRAY(aryTd));
    if(aryTd->e_paramExpr)
        return EXPR_T(aryTd->e_paramExpr);
    CExprOfTypeDesc *ptd = allocPointerTypeDesc(EXPR_T(aryTd->e_typeExpr));
    aryTd->e_paramExpr = (CExpr*)ptd;
    EXPR_REF(ptd);
    return ptd;
}


/**
 * \brief
 * complete compling type descriptor of TD_ARRAY
 *
 * @param td
 *      target type descriptor
 */
PRIVATE_STATIC void
fixTypeDescOfArrayType(CExprOfTypeDesc *td)
{
    long long len = 0;
    // do constant folding for array size
    CNumValueWithType n;
    if(EXPR_ISNULL(td->e_len.eln_lenExpr) == 0) {
        compile1(td->e_len.eln_lenExpr, NULL);

        if(getConstNumValue(td->e_len.eln_lenExpr, &n)) {
            if(n.nvt_isConstButMutable == 0) {
                td->e_len.eln_orgLenExpr = td->e_len.eln_lenExpr;
                CExprOfNumberConst *numExpr = allocExprOfNumberConst1(&n);
                EXPR_REF(numExpr);
                td->e_len.eln_lenExpr = (CExpr*)numExpr;

                if(len < 0) {
                    addError((CExpr*)td, CERR_119);
                    return;
                } else if(len > INT_MAX) {
                    addError((CExpr*)td, CERR_120);
                    return;
                }

                len = n.nvt_numValue.ll;
                td->e_isSizeZero = (len == 0);
            } else {
                len = 1;
                td->e_len.eln_isVariable = 1;
                td->e_isSizeUnreducable = 1;
            }
        } else {
            len = 1;
            td->e_len.eln_isVariable = 1;
            td->e_isSizeUnreducable = 1;
        }
    } else {
        len = 0;
        td->e_len.eln_isVariable = 1;
    }
    int elemSize = getTypeSize(EXPR_T(td->e_typeExpr));
    td->e_size = elemSize * (unsigned int)len;
    td->e_len.eln_len = (unsigned int)len;
    td->e_align = getTypeAlign(EXPR_T(td->e_typeExpr));
}


/**
 * \brief
 * complete compling type descriptor of TD_UNION
 *
 * @param td
 *      target type descriptor
 */
PRIVATE_STATIC void
fixTypeDescOfUnionType(CExprOfTypeDesc *td)
{
    CExpr *memDecls = getMemberDeclsExpr(td);

    if(memDecls == NULL)
        return;

    CCOL_DListNode *ite, *ite2;
    int align = s_alignInt;
    int size = 0, unreducable = 0, zero = 0;

    EXPR_FOREACH(ite, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExpr *mems = memDecl->e_nodes[1];
        assertExpr((CExpr*)memDecl, mems);
        EXPR_FOREACH(ite2, mems) {
            CExpr *declr = EXPR_B(EXPR_L_DATA(ite2))->e_nodes[0];
            CExprOfTypeDesc *memTd0 = EXPR_T(EXPR_B(declr)->e_nodes[0]), *memTd;
            memTd = resolveType((CExpr*)memTd0);
            size = MAX(size, getTypeSize(memTd));
            align = MAX(align, getTypeAlign(memTd));
            if(memTd->e_isSizeUnreducable)
                unreducable = 1;
            if(memTd->e_isSizeZero)
                zero = 1;
        }
    }

    td->e_size = size;
    td->e_align = align;
    td->e_isSizeUnreducable = unreducable;
    td->e_isSizeZero = (size == 0 && zero);
}


/**
 * \brief
 * get aligned offset 
 *
 * @param align
 *      type alignment
 * @param[out] offs
 *      offset
 */
PRIVATE_STATIC void
getAlignedOffset(int align, int *offs)
{
    if(align)
        *offs = ROUND(*offs, align);
}


/**
 * \brief
 * get bit width and byte offset
 *
 * @param td
 *      type descriptor
 * @param[out] bits
 *      bit width
 * @param[out] offs
 *      offset
 */
PRIVATE_STATIC void
getBitsAndOffset(CExprOfTypeDesc *td, int *bits, int *offs)
{
    int fieldBits = td->e_bitLen;
    int sz = getTypeSize(td);

    if(fieldBits > 0) {
        if(*bits + fieldBits > sz * CHAR_BIT) {
            if(*bits) {
                getAlignedOffset(s_alignInt, offs);
                *offs += s_sizeInt;
            }
            *bits = fieldBits;
        } else {
            *bits += fieldBits;
        }
    } else {
        if(*bits) {
            getAlignedOffset(s_alignInt, offs);
            *offs += s_sizeInt;
            *bits = 0;
        }

        getAlignedOffset(getTypeAlign(td), offs);
        *offs += sz;
    }
}


/**
 * \brief
 * complete compling type descriptor of TD_STRUCT
 *
 * @param td
 *      target type descriptor
 */
PRIVATE_STATIC void
fixTypeDescOfStructType(CExprOfTypeDesc *td)
{
    CExpr *memDecls = getMemberDeclsExpr(td);
    if(EXPR_ISNULL(memDecls))
        return;

    CCOL_DListNode *ite1, *ite2;
    int align = s_alignInt;
    int error = 0, numMem = 0, unreducable = 0, zero = 0;

    EXPR_FOREACH(ite1, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExpr *mems = memDecl->e_nodes[1];
        assertExpr((CExpr*)memDecl, mems);
        EXPR_FOREACH(ite2, mems) {
            CExpr *declr = EXPR_B(EXPR_L_DATA(ite2))->e_nodes[0];
            CExprOfTypeDesc *memTd0 = EXPR_T(EXPR_B(declr)->e_nodes[0]);
            CExprOfTypeDesc *memTd = resolveType((CExpr*)memTd0);
            if(memTd == NULL) {
                error = 1;
                break;
            }
            align = MAX(align, getTypeAlign(memTd));
            ++numMem;
            if(memTd->e_isSizeUnreducable)
                unreducable = 1;
            if(memTd->e_isSizeZero)
                zero = 1;
        }
    }

    if(error)
        return;

    int bits = 0, offs = 0, memIdx = 0;

    EXPR_FOREACH(ite1, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExpr *mems = memDecl->e_nodes[1];
        assertExpr(memDecls, mems);
        EXPR_FOREACH(ite2, mems) {
            CExpr *declr = EXPR_B(EXPR_L_DATA(ite2))->e_nodes[0];
            CExprOfTypeDesc *memTd = EXPR_T(EXPR_B(declr)->e_nodes[0]);

            ++memIdx;
            if(memIdx == numMem && EXPR_ISNULL(memTd->e_len.eln_lenExpr) &&
                memTd->e_len.eln_isVariable) {
                //flexible array
                memTd->e_len.eln_isFlexible = 1;
                break;
            }

            getBitsAndOffset(memTd, &bits, &offs);

            if(bits && memTd->e_len.eln_isFlexible == 0) {
                getAlignedOffset(s_alignInt, &offs);
                offs += s_sizeInt;
            }
        }
    }

    if(align)
        offs = ROUND(offs, align);

    td->e_size = offs;
    td->e_align = align;
    td->e_isSizeUnreducable = unreducable;
    td->e_isSizeZero = (offs == 0 && zero && memIdx == 1);
}


/**
 * \brief
 * alloc EC_GCC_ATTR_ARG
 *
 * @param gak
 *      attribute kind
 * @param attrName
 *      attribute symbol
 * @param exprs
 *      attribute arguments
 * @return
 *      allocated node
 */
PRIVATE_STATIC CExprOfBinaryNode*
allocGccAttrArg(CGccAttrKindEnum gak, const char *attrName, CExpr *exprs)
{
    CExprOfBinaryNode *arg = allocExprOfBinaryNode2(EC_GCC_ATTR_ARG,
        (CExpr*)allocExprOfSymbol2(attrName), exprs);
    arg->e_gccAttrInfo = getGccAttrInfo(attrName);
    arg->e_gccAttrKind = gak;

    return arg;
}


/**
 * \brief
 * apply pragma pack to type descriptor 
 *
 * @param td
 *      target type descriptor
 */
PRIVATE_STATIC void
applyPragmaPack(CExprOfTypeDesc *td)
{
    assert(ETYP_IS_STRUCT(td) || ETYP_IS_UNION(td));

    if(s_pragmaPackEnabled == 0 || td->e_isNoMemDecl)
        return;

    CExpr *memDecls = getMemberDeclsExpr(td);
    if(EXPR_ISNULL(memDecls))
        return;

    CExprOfBinaryNode *packArg = allocGccAttrArg(GAK_TYPE, "packed", NULL);
    exprAddAttrToPre(td, packArg);

    CExprOfBinaryNode *alignArg = allocGccAttrArg(GAK_TYPE, "aligned",
        exprList1(EC_EXPRS,
            (CExpr*)allocExprOfNumberConst2(s_pragmaPackAlign, BT_INT)));

    CCOL_DListNode *ite1, *ite2;

    EXPR_FOREACH(ite1, memDecls) {
        CExprOfBinaryNode *memDecl = EXPR_B(EXPR_L_DATA(ite1));
        if(EXPR_IS_MEMDECL(memDecl) == 0)
            continue;
        CExpr *mems = memDecl->e_nodes[1];
        assertExpr(memDecls, mems);
        EXPR_FOREACH(ite2, mems) {
            CExpr *declr = EXPR_B(EXPR_L_DATA(ite2))->e_nodes[0];
            CExprOfTypeDesc *memTd = EXPR_T(EXPR_B(declr)->e_nodes[0]);
            exprAddAttrToPre(memTd, alignArg);
        }
    }

    ++s_numConvsPragmaPack;
}


/**
 * \brief
 * complete compling type descriptor
 *
 * @param td
 *      type descriptor
 */
PRIVATE_STATIC void
fixTypeDesc(CExprOfTypeDesc *td)
{
    CTypeDescKindEnum tk = td->e_tdKind;

    switch(tk) {
    case TD_TYPEREF:
    case TD_GCC_TYPEOF:
        resolveType((CExpr*)td);
        return;
    default:
        break;
    }

    if(td->e_isFixed || td->e_isCompiling)
        return;
    td->e_isFixed = 1;

    CExprOfTypeDesc *typeTd = EXPR_T(td->e_typeExpr);

    if(typeTd && EXPR_CODE(typeTd) == EC_TYPE_DESC)
        resolveType((CExpr*)typeTd);

    CExprOfTypeDesc *refTd = td->e_refType;
    if(refTd)
        resolveType((CExpr*)refTd);

    switch(tk) {
    case TD_BASICTYPE:
        td->e_size = getBasicTypeSize(td->e_basicType);
        td->e_align = getBasicTypeAlign(td->e_basicType);
        break;
    case TD_ENUM:
        td->e_size = getBasicTypeSize(BT_INT);
        td->e_align = getBasicTypeAlign(BT_INT);
        break;
    case TD_POINTER:
        td->e_size = s_sizeAddr;
        td->e_align = s_alignAddr;
        break;
    case TD_ARRAY:
        fixTypeDescOfArrayType(td);
        break;
    case TD_UNION:
    case TD_STRUCT:
    case TD_DERIVED:
        if(refTd) {
            CExprOfTypeDesc *sizeTd = getSizedType(refTd);
            if(sizeTd)
                ETYP_COPY_SIZE(td, sizeTd);
        } else if(ETYP_IS_UNION(td)) {
            applyPragmaPack(td);
            fixTypeDescOfUnionType(td);
        } else {
            applyPragmaPack(td);
            fixTypeDescOfStructType(td);
        }
        break;
    case TD_FUNC:
        td->e_size = s_charTypeDesc.e_size;
        td->e_align = s_charTypeDesc.e_align;
        break;
    case TD_GCC_BUILTIN:
        //TODO must be set proper size/align
        td->e_size = s_addrIntTypeDesc.e_size;
        td->e_align = s_addrIntTypeDesc.e_align;
        td->e_isSizeUnreducable = 1;
        break;
    case TD_COARRAY:
        break;
    default:
        ABORT();
    }

    if(hasGccAttrDerived(td, GAK_ALL))
        td->e_isSizeUnreducable = 1;
}


/**
 * \brief
 * get type which has size recursively for reference type
 *
 * @param td
 *      type descriptor
 * @return
 *      type which has size
 */
CExprOfTypeDesc*
getSizedType(CExprOfTypeDesc *td)
{
    if(td->e_isSizeUnreducable || td->e_size > 0)
        return td;
    if(td->e_refType)
        return getSizedType(td->e_refType);
    return NULL;
}


/**
 * \brief
 * get type size
 *
 * @param td
 *      type descriptor
 * @return
 *      size
 */
int
getTypeSize(CExprOfTypeDesc *td)
{
    if(td->e_isSizeUnreducable || td->e_size > 0)
        return td->e_size;
    if(td->e_refType)
        return (td->e_size = getTypeSize(td->e_refType));

    return 0;
}


/**
 * \brief
 * get type alignment
 *
 * @param td
 *      type descriptor
 * @return
 *      alignment
 */
int
getTypeAlign(CExprOfTypeDesc *td)
{
    if(td->e_isSizeUnreducable || td->e_align > 0)
        return td->e_align;
    if(td->e_refType)
        return (td->e_align = getTypeAlign(td->e_refType));

    return 0;
}


/**
 * \brief
 * alloc type descriptor of int
 *
 * @return
 *      allocated node
 */
CExprOfTypeDesc*
allocIntTypeDesc()
{
    CExprOfTypeDesc *expr = allocExprOfTypeDesc();
    expr->e_tdKind = TD_BASICTYPE;
    expr->e_basicType = BT_INT;

    return expr;
}
