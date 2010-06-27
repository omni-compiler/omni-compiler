/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-comp.h
 */

#ifndef _C_COMP_H
#define _C_COMP_H

#include "c-expr.h"
#include "c-const.h"

extern void reduceExpr(CExpr *expr);
extern void compile(CExpr *expr);
extern void compile1(CExpr *expr, CExpr *parent);
extern void convertSyntax(CExpr *expr);
extern void collectTypeDesc(CExpr *expr);
extern void outputXcodeML(FILE *fp, CExpr *expr);
extern void addTypeDesc(CExprOfTypeDesc *td);

#endif /* _C_COMP_H_ */

