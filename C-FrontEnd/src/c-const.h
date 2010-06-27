/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-const.h
 */

#ifndef _C_CONST_H_
#define _C_CONST_H_

#include "c-expr.h"

extern int              getConstNumValue(CExpr *expr, CNumValueWithType *result);
extern CNumValueKind    getNumValueKind(CBasicTypeEnum bt);
extern long             getCastedLongValue(CNumValueWithType *n);
extern long long        getCastedLongLongValue(CNumValueWithType *n);
extern int              getCastedLongValueOfExpr(CExpr *expr, long *n);
extern int              castNumValue(CNumValueWithType *n, CBasicTypeEnum bt);
extern void             constToNumValueWithType(CExprOfNumberConst *numConst, CNumValueWithType *nvt);
extern int              isConstNumEquals(CExpr *e1, CExpr *e2);
extern int              isConstNumEqualsWithMutable(CExpr *e1, CExpr *e2);
extern long             getNumberConstAsLong(CExpr *expr);
extern int              isConstExpr(CExpr *e, int allowSymbolAddr);
extern int              isConstZero(CExpr *e);

#endif // _C_CONST_H_

