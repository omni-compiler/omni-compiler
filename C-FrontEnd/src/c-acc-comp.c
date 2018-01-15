/**
 * \file c-acc-comp.c
 */

#include "c-comp.h"
#include "c-acc.h"

PRIVATE_STATIC void
out_acc_subscript(CExpr *subscript, CExpr *parent)
{
  if(EXPR_CODE(subscript) != EC_UNDEF){
    compile1(subscript, parent);
  }else{
    CExpr *lower = exprListHeadData(subscript);
    if(! EXPR_ISNULL(lower)){
        compile1(lower, parent);
    }

    if(EXPR_L_SIZE(subscript) > 1){
      CExpr *length = exprListNextNData(subscript, 1);
      if(! EXPR_ISNULL(length)){
          compile1(length, parent);
      }
    }
  }
}


PRIVATE_STATIC void
compile_acc_arrayRef(CExprOfBinaryNode *arrayRef, CExpr *parent)
{
    CExpr *arrayExpr = arrayRef->e_nodes[0];
    CExpr *subscripts = arrayRef->e_nodes[1];

    compile1(arrayExpr, parent);

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, subscripts){
      CExpr *node = EXPR_L_DATA(ite);
      out_acc_subscript(node, parent);
    }
}

PRIVATE_STATIC void
compile_acc_name_list(CExprOfList *name_list, CExpr *parent)
{
    CCOL_DListNode *ite;

    EXPR_FOREACH(ite, name_list) {
	CExpr *node = EXPR_L_DATA(ite);

	if(EXPR_CODE(node) == EC_ARRAY_REF){
            compile_acc_arrayRef((CExprOfBinaryNode*)node, parent);
	}else{
            compile1(node, parent);
	}
    }
}

PRIVATE_STATIC void
compile_acc_clause(CExpr *clause, CExpr *parent)
{
    enum ACC_pragma_clause clause_code = ((CExprOfList*)clause)->e_aux;
    CExpr *arg = exprListHeadData(clause);

    switch(clause_code){
    case ACC_SEQ:
    case ACC_INDEPENDENT:
    case ACC_READ:
    case ACC_WRITE:
    case ACC_UPDATE_CLAUSE:
    case ACC_CAPTURE:
    case ACC_NOHOST:
        return;

    case ACC_IF:
    case ACC_ASYNC:
    case ACC_GANG:
    case ACC_NUM_GANGS:
    case ACC_WORKER:
    case ACC_NUM_WORKERS:
    case ACC_VECTOR:
    case ACC_VECT_LEN:
    case ACC_COLLAPSE:
        compile1(arg, parent);
        return;

    default:
        {
            CExprOfList *name_list = (CExprOfList *)arg;
            if(EXPR_L_ISNULL(name_list)) return;

            compile_acc_name_list(name_list, parent);
        }
    }
}

/**
 * \brief
 * compile #pragma acc
 *
 * @param parent
 *      parent node
 */
void
compile_acc_pragma(CExpr *expr, CExpr *parent)
{
    CExprOfList *body = (CExprOfList *)expr;
    CExprOfList *clauseList = (CExprOfList *)body->e_aux_info;

    int pragma_code = clauseList->e_aux;

    switch(pragma_code){
    case ACC_WAIT:
	if(! EXPR_L_ISNULL(clauseList)){
	  compile1(exprListHeadData((CExpr*)clauseList), expr);
	}
        break;
    case ACC_CACHE:
        compile_acc_name_list(clauseList, expr);
        break;
    default:
        {
            CCOL_DListNode *ite;
            EXPR_FOREACH(ite, clauseList) {
                CExpr *clause = EXPR_L_DATA(ite);
                compile_acc_clause(clause, parent);
            }
        }
    }
}
