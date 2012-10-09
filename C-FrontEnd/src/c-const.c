/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-const.c
 */

#include "c-expr.h"
#include "c-comp.h"

PRIVATE_STATIC int
isUnreducableConst(CExpr *e, int allowSymbolAddr);

/**
 * \brief
 * get number valud kind of basic type
 *
 * @param bt
 *      basic type
 * @return
 *      number value kind
 */
CNumValueKind
getNumValueKind(CBasicTypeEnum bt)
{
    switch(bt) {
    case BT_BOOL:
    case BT_CHAR:
    case BT_WCHAR:
    case BT_SHORT:
    case BT_INT:
    case BT_LONG:
    case BT_LONGLONG:
        return NK_LL;
    case BT_UNSIGNED_CHAR:
    case BT_UNSIGNED_SHORT:
    case BT_UNSIGNED_INT:
    case BT_UNSIGNED_LONG:
    case BT_UNSIGNED_LONGLONG:
        return NK_ULL;
    case BT_FLOAT:
    case BT_DOUBLE:
    case BT_LONGDOUBLE:
        return NK_LD;
    case BT_VOID:
    case BT_FLOAT_COMPLEX:
    case BT_DOUBLE_COMPLEX:
    case BT_LONGDOUBLE_COMPLEX:
    case BT_FLOAT_IMAGINARY:
    case BT_DOUBLE_IMAGINARY:
    case BT_LONGDOUBLE_IMAGINARY:
    case BT_UNDEF:
    case BT_END:
        ABORT();
    }

    return 0;
}


/**
 * \brief
 * get value as long of CNumValueWithType
 *
 * @param n
 *      number value
 * @return
 *      long value
 */
long
getCastedLongValue(CNumValueWithType *n)
{
    switch(n->nvt_numKind) {
    case NK_LL:  return (long)n->nvt_numValue.ll;
    case NK_ULL: return (long)n->nvt_numValue.ull;
    case NK_LD:  return (long)n->nvt_numValue.ld;
    default:  ABORT();
    }

    return 0;
}


/**
 * \brief
 * get value as long long of CNumValueWithType
 *
 * @param n
 *      number value
 * @return
 *      long long value
 */
long long
getCastedLongLongValue(CNumValueWithType *n)
{
    switch(n->nvt_numKind) {
    case NK_LL:  return (long long)n->nvt_numValue.ll;
    case NK_ULL: return (long long)n->nvt_numValue.ull;
    case NK_LD:  return (long long)n->nvt_numValue.ld;
    default:  ABORT();
    }

    return 0;
}


/**
 * \brief
 * cast number value to specified type
 *
 * @param n
 *      number value
 * @param bt
 *      basic type
 * @return
 *      0:failed, 1:ok
 */
int
castNumValue(CNumValueWithType *n, CBasicTypeEnum bt)
{
    int nk = n->nvt_numKind;
    if(bt == BT_UNDEF)
        bt = BT_INT;

    switch(bt) {
    case BT_BOOL:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (_Bool)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (_Bool)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (_Bool)n->nvt_numValue.ld; break;
        }
        break;
    case BT_CHAR:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (signed char)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (signed char)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (signed char)n->nvt_numValue.ld; break;
        }
        break;
    case BT_WCHAR:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (wchar_t)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (wchar_t)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (wchar_t)n->nvt_numValue.ld; break;
        }
        break;
    case BT_SHORT:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (signed short)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (signed short)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (signed short)n->nvt_numValue.ld; break;
        }
        break;
    case BT_INT:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (signed int)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (signed int)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (signed int)n->nvt_numValue.ld; break;
        }
        break;
    case BT_LONG:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (signed long)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (signed long)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (signed long)n->nvt_numValue.ld; break;
        }
        break;
    case BT_LONGLONG:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (signed long long)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (signed long long)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (signed long long)n->nvt_numValue.ld; break;
        }
        break;
    case BT_UNSIGNED_CHAR:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (unsigned char)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (unsigned char)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (unsigned char)n->nvt_numValue.ld; break;
        }
        break;
    case BT_UNSIGNED_SHORT:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (unsigned short)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (unsigned short)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (unsigned short)n->nvt_numValue.ld; break;
        }
        break;
    case BT_UNSIGNED_INT:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (unsigned int)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (unsigned int)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (unsigned int)n->nvt_numValue.ld; break;
        }
        break;
    case BT_UNSIGNED_LONG:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (unsigned long)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (unsigned long)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (unsigned long)n->nvt_numValue.ld; break;
        }
        break;
    case BT_UNSIGNED_LONGLONG:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ll = (unsigned long long)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ll = (unsigned long long)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ll = (unsigned long long)n->nvt_numValue.ld; break;
        }
        break;
    case BT_FLOAT:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ld = (float)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ld = (float)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ld = (float)n->nvt_numValue.ld; break;
        }
        break;
    case BT_DOUBLE:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ld = (double)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ld = (double)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ld = (double)n->nvt_numValue.ld; break;
        }
        break;
    case BT_LONGDOUBLE:
        switch(nk) {
        case NK_LL:  n->nvt_numValue.ld = (long double)n->nvt_numValue.ll; break;
        case NK_ULL: n->nvt_numValue.ld = (long double)n->nvt_numValue.ull; break;
        case NK_LD:  n->nvt_numValue.ld = (long double)n->nvt_numValue.ld; break;
        }
        break;
    case BT_FLOAT_COMPLEX:
    case BT_DOUBLE_COMPLEX:
    case BT_LONGDOUBLE_COMPLEX:
    case BT_FLOAT_IMAGINARY:
    case BT_DOUBLE_IMAGINARY:
    case BT_LONGDOUBLE_IMAGINARY:
    case BT_VOID:
    case BT_UNDEF:
    case BT_END:
        return 0;
    }

    n->nvt_basicType = bt;
    n->nvt_numKind = getNumValueKind(bt);

    return 1;
}


/**
 * \brief
 * convert CExprOfNumberConst to CNumValueWithType
 *
 * @param numConst
 *      source value
 * @param nvt
 *      destination value
 */
void
constToNumValueWithType(CExprOfNumberConst *numConst, CNumValueWithType *nvt)
{
    nvt->nvt_numValue = numConst->e_numValue;
    nvt->nvt_basicType = numConst->e_basicType;
    nvt->nvt_numKind = getNumValueKind(numConst->e_basicType);
}


/**
 * \brief
 * cast number value to prior type
 *
 * @param n1
 *      number value 1
 * @param n2
 *      number value 2
 * @return
 *      0:failed, 1:ok
 */
PRIVATE_STATIC int
fixNumValueType(CNumValueWithType *n1, CNumValueWithType *n2)
{
    int cast2to1 = 0;

    switch(n1->nvt_basicType) {
    case BT_BOOL:
        break;
    case BT_CHAR:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
            cast2to1 = 1;
        default:
            break;
        }
        break;
    case BT_SHORT:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
            cast2to1 = 1;
        default:
            break;
        }
        break;
    case BT_WCHAR:
    case BT_INT:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_LONG:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_UNSIGNED_INT:   case BT_WCHAR:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_LONGLONG:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_UNSIGNED_INT:   case BT_WCHAR:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_UNSIGNED_CHAR:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_UNSIGNED_SHORT:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_UNSIGNED_INT:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_WCHAR:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_UNSIGNED_LONG:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_UNSIGNED_INT:
        case BT_WCHAR:
        case BT_LONG:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_UNSIGNED_LONGLONG:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_UNSIGNED_INT:
        case BT_WCHAR:
        case BT_LONG:       case BT_UNSIGNED_LONG:
        case BT_LONGLONG:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_FLOAT:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_UNSIGNED_INT:
        case BT_WCHAR:
        case BT_LONG:       case BT_UNSIGNED_LONG:
        case BT_LONGLONG:   case BT_UNSIGNED_LONGLONG:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_DOUBLE:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_UNSIGNED_INT:
        case BT_WCHAR:
        case BT_LONG:       case BT_UNSIGNED_LONG:
        case BT_LONGLONG:   case BT_UNSIGNED_LONGLONG:
        case BT_FLOAT:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_LONGDOUBLE:
        switch(n2->nvt_basicType) {
        case BT_BOOL:
        case BT_CHAR:       case BT_UNSIGNED_CHAR:
        case BT_SHORT:      case BT_UNSIGNED_SHORT:
        case BT_INT:        case BT_UNSIGNED_INT:
        case BT_WCHAR:
        case BT_LONG:       case BT_UNSIGNED_LONG:
        case BT_LONGLONG:   case BT_UNSIGNED_LONGLONG:
        case BT_FLOAT:      case BT_DOUBLE:
            cast2to1 = 1;
            break;
        default:
            break;
        }
        break;
    case BT_VOID:
    case BT_FLOAT_COMPLEX:
    case BT_DOUBLE_COMPLEX:
    case BT_LONGDOUBLE_COMPLEX:
    case BT_FLOAT_IMAGINARY:
    case BT_DOUBLE_IMAGINARY:
    case BT_LONGDOUBLE_IMAGINARY:
    case BT_UNDEF:
    case BT_END:
        return 0;
    }

    CNumValueWithType *nn1, *nn2;

    if(cast2to1 == 0) {
        nn1 = n2;
        nn2 = n1;
    } else {
        nn1 = n1;
        nn2 = n2;
    }

    return castNumValue(nn2, nn1->nvt_basicType);
}


/**
 * \brief
 * judge expr is divided by zero or add error
 *
 * @param expr
 *      target node
 * @param nk
 *      number value kind
 * @param nv
 *      number value
 * @return
 *      0:no error, 1:error
 */
PRIVATE_STATIC int
checkDivisionByZero(CExpr *expr, CNumValueKind nk, CNumValue *nv)
{
    int err = 0;
    switch(nk) {
    case NK_LL:  err = (nv->ll == 0); break;
    case NK_ULL: err = (nv->ull == 0); break;
    case NK_LD:  err = (nv->ld == 0.0); break;
    }

    if(err)
        addError(expr, CERR_015);

    return err;
}


/**
 * \brief
 * merge flags int CNumValueWithType
 *
 * @param dst
 *      destination
 * @param src
 *      source
 */
PRIVATE_STATIC void
mergeConstFlag(CNumValueWithType *dst, CNumValueWithType *src)
{
    dst->nvt_isConstButMutable |= src->nvt_isConstButMutable;
    dst->nvt_isConstButUnreducable |= src->nvt_isConstButUnreducable;
}


/**
 * \brief
 * do constant folding
 *
 * @param expr
 *      target node
 * @param[out] result
 *      result value
 * @return
 *      0:is not constant, 1:constant
 */
int
getConstNumValue(CExpr *expr, CNumValueWithType *result)
{
    memset(result, 0, sizeof(CNumValueWithType));

    if(EXPR_ISNULL(expr) || EXPR_ISERROR(expr)) {
        result->nvt_numValue.ll = 0;
        result->nvt_basicType = BT_INT;
        result->nvt_numKind = getNumValueKind(result->nvt_basicType);
        return 1;
    }

    if(EXPR_CODE(expr) == EC_XMP_DESC_OF) return 0; /* not constant */

    CNumValueWithType n1, n2, n3;
    int use2 = 0, use3 = 0, isConst2 = 0, isConst3 = 0;

    switch(EXPR_STRUCT(expr)) {
    case STRUCT_CExprOfUnaryNode: {
            CExpr *node = EXPR_U(expr)->e_node;

            if(EXPR_CODE(expr) == EC_SIZE_OF || 
	       EXPR_CODE(expr) == EC_GCC_ALIGN_OF) {
                CExprOfTypeDesc *td = resolveType(node);

                if(td == NULL)
                    return 0;

                result->nvt_isConstButMutable = 1;
                result->nvt_basicType = BT_INT;
                result->nvt_numKind = getNumValueKind(result->nvt_basicType);
                if(EXPR_CODE(expr) == EC_SIZE_OF)
                    result->nvt_numValue.ll = getTypeSize(td);
                else {
                    assertExpr((CExpr*)td, getTypeAlign(td));
                    result->nvt_numValue.ll = getTypeAlign(td);
                }
                return 1;
            }

            if(getConstNumValue(node, &n1) == 0) {
                mergeConstFlag(result, &n1);
                return 0;
            }
        }
        break;

    case STRUCT_CExprOfBinaryNode: {
            if(isUnreducableConst(expr, 1)) {
                result->nvt_isConstButUnreducable = 1;
                return 0;
            }

            CExpr *node1 = EXPR_B(expr)->e_nodes[0];
            CExpr *node2 = EXPR_B(expr)->e_nodes[1];

            if(EXPR_CODE(node1) == EC_TYPE_DESC) {
                if(getConstNumValue(node2, &n2) == 0)
                    return 0;
                else
                    n1 = n2;
            } else {
                if(getConstNumValue(node1, &n1) == 0) {
                    mergeConstFlag(result, &n1);
                    return 0;
                }

                if(getConstNumValue(node2, &n2) == 0) {
                    mergeConstFlag(result, &n1);
                    mergeConstFlag(result, &n2);
                    return 0;
                }
                use2 = 1;
            }
        }
        break;

    case STRUCT_CExprOfList:
        if(EXPR_CODE(expr) == EC_CONDEXPR) {
            if(getConstNumValue(exprListNextNData(expr, 0), &n1) == 0) {
                mergeConstFlag(result, &n1);
                return 0;
            }
            isConst2 = getConstNumValue(exprListNextNData(expr, 1), &n2);
            isConst3 = getConstNumValue(exprListNextNData(expr, 2), &n3);
            use2 = use3 = 1;
        } else {
            //maybe comma expr
            if(getConstNumValue(exprListTailData(expr), &n1) == 0) {
                mergeConstFlag(result, &n1);
                return 0;
            }
        }
        break;

    case STRUCT_CExprOfCharConst:
        //wide char is not supported
        result->nvt_numValue.ll = EXPR_CHARCONST(expr)->e_token[0];
        result->nvt_basicType = BT_CHAR;
        result->nvt_numKind = getNumValueKind(result->nvt_basicType);
        return 1;

    case STRUCT_CExprOfNumberConst:
        constToNumValueWithType(EXPR_NUMBERCONST(expr), result);
        return 1;

    case STRUCT_CExprOfSymbol: {
            CExprOfSymbol *tsym = findSymbolByGroup(EXPR_SYMBOL(expr)->e_symName, STB_IDENT);
            if(tsym == NULL || (tsym->e_symType != ST_ENUM) ||
                tsym->e_isConstButUnreducable ||
                (tsym && getConstNumValue(tsym->e_valueExpr, &n1) == 0)) {
                if(tsym && tsym->e_isConstButUnreducable)
                    result->nvt_isConstButUnreducable = 1;
                mergeConstFlag(result, &n1);
                return 0;
            }
        }
        break;

    case STRUCT_CExprOfArrayDecl:
    case STRUCT_CExprOfTypeDesc:
    case STRUCT_CExprOfErrorNode:
    case STRUCT_CExprOfGeneralCode:
    case STRUCT_CExprOfNull:
        return 0;

    default:
        assertExpr(expr, 0);
        ABORT();
    }

    int r = 1;
    int isCondExpr = (EXPR_CODE(expr) == EC_CONDEXPR);
    //for lshift/rshift
    int ni2 = 0;
    mergeConstFlag(result, &n1);
    
    if(use2) {
        ni2 = (int)getCastedLongValue(&n2);
        if(fixNumValueType(&n1, &n2) == 0 && isCondExpr == 0)
            return 0;
        mergeConstFlag(result, &n2);
    }

    if(use3) {
        if(fixNumValueType(&n1, &n3) == 0 && isCondExpr == 0)
            return 0;
        mergeConstFlag(result, &n3);
    }

    //now n1/n2/n3 have same basicType and numKind
    CNumValueKind nk = n1.nvt_numKind;
    CNumValue *nvr = &result->nvt_numValue;
    CNumValue *nv1 = &n1.nvt_numValue;
    CNumValue *nv2 = &n2.nvt_numValue;
    CNumValue *nv3 = &n3.nvt_numValue;

    switch(EXPR_CODE(expr)) {
    case EC_EXPRS:
        //last expression
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll; break;
        case NK_ULL: nvr->ull = nv1->ull; break;
        case NK_LD:  nvr->ld  = nv1->ld; break;
        }
        break;
    case EC_BRACED_EXPR:
    case EC_IDENT: // enumerator
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll; break;
        case NK_ULL: nvr->ull = nv1->ull; break;
        case NK_LD:  nvr->ld  = nv1->ld; break;
        }
        break;
    case EC_UNARY_MINUS:
        switch(nk) {
        case NK_LL:  nvr->ll  = -nv1->ll; break;
        case NK_ULL: nvr->ull = -nv1->ull; break;
        case NK_LD:  nvr->ld  = -nv1->ld; break;
        }
        break;
    case EC_BIT_NOT:
        switch(nk) {
        case NK_LL:  nvr->ll  = ~nv1->ll; break;
        case NK_ULL: nvr->ull = ~nv1->ull; break;
        case NK_LD:  return 0;
        }
        break;
    case EC_LOG_NOT:
        switch(nk) {
        case NK_LL:  nvr->ll  = !nv1->ll; break;
        case NK_ULL: nvr->ull = !nv1->ull; break;
        case NK_LD:  nvr->ld  = !nv1->ld; break;
        }
        break;
    case EC_CAST: {
            CExprOfTypeDesc *td = resolveType(EXPR_B(expr)->e_nodes[0]);
            if(td == NULL)
                return 0;
            CExprOfTypeDesc *tdo = getRefType(td);
            if(castNumValue(&n1, tdo->e_basicType) == 0) {
                addError(expr, CERR_016);
                EXPR_ISERROR(expr) = 1;
                return 0;
            }
            switch(n1.nvt_numKind) {
            case NK_LL:  nvr->ll  = nv1->ll;  break;
            case NK_ULL: nvr->ull = nv1->ull; break;
            case NK_LD:  nvr->ld  = nv1->ld;  break;
            }
        }
        break;
    case EC_LSHIFT:
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  << ni2; break;
        case NK_ULL: nvr->ull = nv1->ull << ni2; break;
        case NK_LD:  return 0;
        }
        break;
    case EC_RSHIFT:
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  >> ni2; break;
        case NK_ULL: nvr->ull = nv1->ull >> ni2; break;
        case NK_LD:  return 0;
        }
        break;
    case EC_PLUS:
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  + nv2->ll; break;
        case NK_ULL: nvr->ull = nv1->ull + nv2->ull; break;
        case NK_LD:  nvr->ld  = nv1->ld  + nv2->ld; break;
        }
        break;
    case EC_MINUS:
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  - nv2->ll; break;
        case NK_ULL: nvr->ull = nv1->ull - nv2->ull; break;
        case NK_LD:  nvr->ld  = nv1->ld  - nv2->ld; break;
        }
        break;
    case EC_MUL:
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  * nv2->ll; break;
        case NK_ULL: nvr->ull = nv1->ull * nv2->ull; break;
        case NK_LD:  nvr->ld  = nv1->ld  * nv2->ld; break;
        }
        break;
    case EC_DIV:
        if(checkDivisionByZero(expr, nk, nv2))
            return 0;
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  / nv2->ll; break;
        case NK_ULL: nvr->ull = nv1->ull / nv2->ull; break;
        case NK_LD:  nvr->ld  = nv1->ld  / nv2->ld; break;
        }
        break;
    case EC_MOD:
        if(checkDivisionByZero(expr, nk, nv2))
            return 0;
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  % nv2->ll; break;
        case NK_ULL: nvr->ull = nv1->ull % nv2->ull; break;
        case NK_LD:  return 0;
        }
        break;
    case EC_ARITH_EQ:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  == nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull == nv2->ull); break;
        case NK_LD:  nvr->ld  = (nv1->ld  == nv2->ld); break;
        }
        break;
    case EC_ARITH_NE:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  != nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull != nv2->ull); break;
        case NK_LD:  nvr->ld  = (nv1->ld  != nv2->ld); break;
        }
        break;
    case EC_ARITH_GE:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  >= nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull >= nv2->ull); break;
        case NK_LD:  nvr->ld  = (nv1->ld  >= nv2->ld); break;
        }
        break;
    case EC_ARITH_GT:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  > nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull > nv2->ull); break;
        case NK_LD:  nvr->ld  = (nv1->ld  > nv2->ld); break;
        }
        break;
    case EC_ARITH_LE:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  <= nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull <= nv2->ull); break;
        case NK_LD:  nvr->ld  = (nv1->ld  <= nv2->ld); break;
        }
        break;
    case EC_ARITH_LT:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  < nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull < nv2->ull); break;
        case NK_LD:  nvr->ld  = (nv1->ld  < nv2->ld); break;
        }
        break;
    case EC_LOG_AND:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  && nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull && nv2->ull); break;
        case NK_LD:  return 0;
        }
        break;
    case EC_LOG_OR:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  || nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull || nv2->ull); break;
        case NK_LD:  return 0;
        }
        break;
    case EC_BIT_AND:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  & nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull & nv2->ull); break;
        case NK_LD:  return 0;
        }
        break;
    case EC_BIT_OR:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  | nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull | nv2->ull); break;
        case NK_LD:  return 0;
        }
        break;
    case EC_BIT_XOR:
        switch(nk) {
        case NK_LL:  nvr->ll  = (nv1->ll  ^ nv2->ll); break;
        case NK_ULL: nvr->ull = (nv1->ull ^ nv2->ull); break;
        case NK_LD:  return 0;
        }
        break;
    case EC_CONDEXPR:
        switch(nk) {
        case NK_LL:  nvr->ll  = nv1->ll  ? nv2->ll  : nv3->ll;
            r = (nv1->ll) ? isConst2 : isConst3; break;
        case NK_ULL: nvr->ull = nv1->ull ? nv2->ull : nv3->ull;
            r = (nv1->ull) ? isConst2 : isConst3; break;
        case NK_LD:  nvr->ld  = nv1->ld  ? nv2->ld  : nv3->ld;
            r = (nv1->ld) ? isConst2 : isConst3; break;
        }
        break;
    default:
        return 0;
    }

    result->nvt_basicType = n1.nvt_basicType;
    result->nvt_numKind = n1.nvt_numKind;

    return r;
}


/**
 * \brief
 * judge e is unreducable value
 *
 * @param e
 *      target node
 * @param allowSymbolAddr
 *      set to 1 and symbol address will be treated as constant
 * @return
 *      0:no, 1:yes
 */
PRIVATE_STATIC int
isUnreducableConst(CExpr *e, int allowSymbolAddr)
{
    if(EXPR_CODE(e) == EC_GCC_BLTIN_OFFSET_OF)
        return 1;

    if(EXPR_CODE(e) == EC_FUNCTION_CALL) {
        CExprOfTypeDesc *td = resolveType(EXPR_B(e)->e_nodes[0]);
        if(td == NULL)
            return 0;
        td = getRefType(td);
        if(ETYP_IS_POINTER(td))
            td = EXPR_T(td->e_typeExpr);
        if(ETYP_IS_FUNC(td) == 0)
            return 0;
        //treat static+inline+const func with
        //constant argument as constant.
        //this code needs to compile the linux kernel.
        //ex) case __fswab16(0x0800):
        int isConst = (td->e_sc.esc_isStatic &&
            td->e_tq.etq_isInline &&
            (td->e_tq.etq_isConst || td->e_isGccConst));
        if(isConst &&
            isConstExpr(EXPR_B(e)->e_nodes[1], allowSymbolAddr))
            return 1;
    }

    return 0;
}


/**
 * \brief
 * judge e is constant value
 *
 * @param e
 *      target node
 * @param allowSymbolAddr
 *      set to 1 and symbol address will be treated as constant
 * @return
 *      0:no, 1:yes
 */
int
isConstExpr(CExpr *e, int allowSymbolAddr)
{
    if(e == NULL)
        return 1;
    if(EXPR_ISCONSTVALUECHECKED(e) || EXPR_ISERROR(e))
        return EXPR_ISCONSTVALUE(e);
    EXPR_ISCONSTVALUECHECKED(e) = 1;

    switch(EXPR_CODE(e)) {
    case EC_SIZE_OF:
    case EC_GCC_ALIGN_OF:
    case EC_XMP_DESC_OF:
        goto end;
    case EC_FUNCTION_CALL: {
            if(isUnreducableConst(e, allowSymbolAddr) == 0)
                return 0;
            goto end;
        }
    case EC_GCC_BLTIN_VA_ARG:
    case EC_POINTER_REF:
    case EC_POINTS_AT:
    case EC_MEMBER_REF: {
            assert(EXPRS_TYPE(e));
            CExprOfTypeDesc *tdo = getRefType(EXPRS_TYPE(e));
            if(ETYP_IS_ARRAY(tdo))
                goto end;
            CExpr *parent = EXPR_PARENT(e);
            if(isExprCodeChildOf(e, EC_ADDR_OF, parent, NULL) == 0)
                return 0;
            goto end;
        }
    case EC_IDENT:
        switch(EXPR_SYMBOL(e)->e_symType) {
        case ST_ENUM:
        case ST_MEMBER:
            goto end;
        default:
            if(allowSymbolAddr) {
                CExprOfTypeDesc *td = EXPRS_TYPE(e);
                CExprOfTypeDesc *tdo = getRefType(EXPRS_TYPE(e));
                assert(td);
                CExpr *parent = EXPR_PARENT(e);

                if(ETYP_IS_FUNC(tdo)) {
                    if(isExprCodeChildOf(e, EC_FUNCTION_CALL, parent, NULL))
                        return 0;
                    CExprOfTypeDesc *ptd = allocPointerTypeDesc(td);
                    exprSetExprsType(e, ptd);
                    addTypeDesc(ptd);
                } else if(ETYP_IS_ARRAY(tdo)) {
                    CExpr *pp;
                    if(isExprCodeChildOf(e, EC_ARRAY_REF, parent, &pp)) {
                        if(isExprCodeChildOf(e, EC_ARRAY_REF, pp, NULL) == 0 &&
                            isExprCodeChildOf(e, EC_ADDR_OF, parent, NULL) == 0)
                            return 0;
                    }
                } else if(ETYP_IS_POINTER(tdo) == 0) {
                    if(isExprCodeChildOf(e, EC_ADDR_OF, parent, NULL) == 0)
                        return 0;
                }
            } else {
                return 0;
            }
        }
        break;
    case EC_CONDEXPR: {
            CExpr *cond = exprListHeadData(e);
            if(isConstExpr(cond, allowSymbolAddr) == 0)
                return 0;
            goto end;
        }
    default:
        break;
    }

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, e) {
        if(isConstExpr(ite.node, allowSymbolAddr) == 0)
            return 0;
    }

  end:

    EXPR_ISCONSTVALUE(e) = 1;
    return 1;
}


/**
 * \brief
 * judge e1 and e2 equals when they are reduced.
 * if e1 or e2 value is mutable at compile time, returns 0.
 *
 * @param e1
 *      target node 1
 * @param e2
 *      target node 2
 * @return
 *      0:no, 1:yes
 */
int
isConstNumEquals(CExpr *e1, CExpr *e2)
{
    CNumValueWithType nvt1;
    CNumValueWithType nvt2;

    if(getConstNumValue(e1, &nvt1) == 0 || nvt1.nvt_isConstButMutable)
        return 0;
    if(getConstNumValue(e2, &nvt2) == 0 || nvt2.nvt_isConstButMutable)
        return 0;

    return (nvt1.nvt_numValue.ll == nvt2.nvt_numValue.ll);
}


/**
 * \brief
 * judge e1 and e2 equals when they are reduced
 *
 * @param e1
 *      target node 1
 * @param e2
 *      target node 2
 * @return
 *      0:no, 1:yes
 */
int
isConstNumEqualsWithMutable(CExpr *e1, CExpr *e2)
{
    CNumValueWithType nvt1;
    CNumValueWithType nvt2;

    if(getConstNumValue(e1, &nvt1) == 0)
        return 0;
    if(getConstNumValue(e2, &nvt2) == 0)
        return 0;

    return (nvt1.nvt_numValue.ll == nvt2.nvt_numValue.ll);
}


#define MODE_NORM        0
#define MODE_IN_ESC      1
#define MODE_ESC_OCT     2
#define MODE_ESC_HEX     3

/**
 * \brief
 * decode string/char literal
 *
 * @param[out] s
 *      decoded string
 * @param cstr
 *      original token
 * @param[out] numChar
 *      number of characters
 * @param isWChar
 *      if cstr is wide string, set to 1
 * @return
 *      0:error, 1:ok
 *
 */
int
unescChar(char *s, char *cstr, int *numChar, int isWChar)
{
    #define ADD_TO_S(x) { if(s) { if(isWChar) ++s; *s++ = x; } }
    #define SET_TO_S(x) { if(s) { *s = x; } }
    #define INCR_S()    { if(s) { ++s; } }
    #define SWAP_WC()   { if(s) { char tmp = *(s - 1); *(s - 1) = *(s - 2); *(s - 1) = tmp; } }

    int mode = 0, stock = 0, stockCnt = 0;
    int result = 1, n = 0, uni = 0;
    char c;

    while((c = *cstr++)) {
  again:
        switch(mode) {
        case MODE_NORM:
            if (c == '\\')
                mode = MODE_IN_ESC;
            else {
                ADD_TO_S(c);
                ++n;
            }
            break;
        case MODE_IN_ESC:
            switch (c) {
            case 'a':  ADD_TO_S('\a'); ++n; mode = MODE_NORM; break;
            case 'b':  ADD_TO_S('\b'); ++n; mode = MODE_NORM; break;
            case 'f':  ADD_TO_S('\f'); ++n; mode = MODE_NORM; break;
            case 'n':  ADD_TO_S('\n'); ++n; mode = MODE_NORM; break;
            case 'r':  ADD_TO_S('\r'); ++n; mode = MODE_NORM; break;
            case 't':  ADD_TO_S('\t'); ++n; mode = MODE_NORM; break;
            case 'v':  ADD_TO_S('\v'); ++n; mode = MODE_NORM; break;
            case '\\': ADD_TO_S('\\'); ++n; mode = MODE_NORM; break;
            case '\'': ADD_TO_S('\''); ++n; mode = MODE_NORM; break;
            case '"':  ADD_TO_S('"');  ++n; mode = MODE_NORM; break;
            case '?':  ADD_TO_S('?');  ++n; mode = MODE_NORM; break;
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7':
                mode = MODE_ESC_OCT; stock = stockCnt = 0; goto again;
            case 'x':
                mode = MODE_ESC_HEX; stock = stockCnt = 0; break;
            case 'u':
                mode = MODE_ESC_HEX; uni = 1; stock = stockCnt = 0; break;
            default:
                mode = MODE_NORM; result = 0; break; // invalid escape character
            }
            break;
       case MODE_ESC_OCT:
            switch(c) {
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7':
                stock <<= 3;
                stock |= c - '0';
                ++stockCnt;
                SET_TO_S((char)(stock & 0xFF));
                if(stock > 0xFF || stockCnt >= 3) {
                    mode = MODE_NORM;
                    INCR_S();
                    ++n;
                    if(stock > 0xFF)
                        result = 0; // octet sequence out of range
                }
                break;
            default:
                if(stockCnt == 0)
                    result = 0; // imcomplete octet value
                else {
                    ADD_TO_S((char)(stock & 0xFF));
                    ++n;
                }
                mode = MODE_NORM;
                goto again;
            }
            break;
        case MODE_ESC_HEX:
            switch(c) {
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
            case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
            case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
                stock <<= 4;
                stock |= c - ((c <= '9') ? '0' : ((c <= 'F') ? 'A' : 'f'));
                SET_TO_S((char)(stock & 0xFF));
                ++stockCnt;
                if(stockCnt % 2 == 0) {
                    INCR_S();
                    ++n;
                    stock = 0;
                }
                if(uni == 0 && stockCnt >= 3) {
                    mode = MODE_NORM;
                    uni = 0;
                    result = 0; // hex sequence out of range
                }
                else if(uni && stockCnt >= 4) {
                    mode = MODE_NORM;
                    uni = 0;
                    SWAP_WC();
                    if(isWChar)
                        --n;
                }
                break;
            default:
                if(stockCnt == 0)
                    result = 0; // incomplete universal character / hex sequence
                else {
                    ++n;
                }
                mode = MODE_NORM;
                uni = 0;
                goto again;
            }
            break;
        default:
            break;
        }
    }

    if(uni) {
        if(stockCnt < 4)
            result = 0; // incomplete universal character
        else
            SWAP_WC();
    }

    SET_TO_S(0);
    *numChar = n;

    return result;
} 


/**
 * \brief
 * get values as long of EC_NUMBER_CONST node
 *
 * @param expr
 *      EC_NUMBER_CONST node
 * @return
 *      long value
 */
long
getNumberConstAsLong(CExpr *expr)
{
    assertExprCode(expr, EC_NUMBER_CONST);
    CNumValueWithType n;
    constToNumValueWithType(EXPR_NUMBERCONST(expr), &n);
    castNumValue(&n, BT_LONG);
    return (long)n.nvt_numValue.ll;
}


/**
 * \brief
 * get value as long of expr
 *
 * @param expr
 *      target node
 * @param[out] n
 *      long value
 * @return
 *      0:is not constant, 1:is constant
 */
int
getCastedLongValueOfExpr(CExpr *expr, long *n)
{
    CNumValueWithType nvt;
    if(getConstNumValue(expr, &nvt) == 0)
        return 0;
    castNumValue(&nvt, BT_LONG);
    *n = (long)nvt.nvt_numValue.ll;
    return 1;
}


/**
 * \brief
 * judge e is constant zero value
 *
 * @param
 *      taret node
 * @return
 *      0:no, 1:yes
 */
int
isConstZero(CExpr *e)
{
    long n;
    if(getCastedLongValueOfExpr(e, &n) == 0)
        return 0;
    return (n == 0);
}

