/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include <stdlib.h>
#include <stdarg.h> 
#include <wchar.h>
#include "c-comp.h"
#include "c-option.h"
#include "c-acc.h"

#include "c-xcodeml.h"

void outx_ACC_Clause(FILE *fp, int indent, CExprOfList* clause);
void out_ACC_name_list(FILE *fp,int indent, CExprOfList *list);
void out_ACC_arrayRef(FILE *fp, int indent, CExprOfBinaryNode *arrayRef);
void out_ACC_subscript(FILE *fp, int indent, CExpr *subscript);

char *accDirectiveName(int c);
char *accClauseName(int c);

void
out_ACC_PRAGMA(FILE *fp, int indent, int pragma_code, CExpr* expr)
{
    CExprOfList *body = (CExprOfList *)expr;
    CExprOfList *clauseList = (CExprOfList *)body->e_aux_info;

    const char *ompPragmaTag = "ACCPragma";
    int indent1 = indent + 1;

    CCOL_DListNode *ite;
    outxTagForStmt(fp, indent,(CExpr*)clauseList, ompPragmaTag,0, NULL);
    outxPrint(fp,indent1,"<string>%s</string>\n",
	      accDirectiveName(pragma_code));

    switch(pragma_code){
    case ACC_WAIT:
    outxChildren(fp,indent1,(CExpr *)clauseList);
    goto end;
    case ACC_CACHE:
      if(EXPR_L_SIZE(clauseList) != 0)
	  out_ACC_name_list(fp, indent1, clauseList);
	goto end;
    }

    outxPrint(fp,indent1,"<list>\n");
    EXPR_FOREACH(ite, clauseList){
	CExpr *node = EXPR_L_DATA(ite);
	outx_ACC_Clause(fp,indent1+1,(CExprOfList *)node);
    }
    outxPrint(fp,indent1,"</list>\n");

  end:
    if(EXPR_L_SIZE(expr) != 0) outxChildren(fp,indent1,expr);
    outxTagClose(fp, indent,ompPragmaTag);
}

void
outx_ACC_Clause(FILE *fp, int indent, CExprOfList* clause)
{
  int indent1 = indent+1;
  CExpr *arg;
  CExprOfList *namelist;

  outxPrint(fp,indent,"<list>\n");
  outxPrint(fp,indent1,"<string>%s</string>\n",
	    accClauseName(clause->e_aux));
  arg = exprListHeadData((CExpr *)clause);

  switch(clause->e_aux){
  case ACC_SEQ:
  case ACC_INDEPENDENT:
      break;

  case ACC_IF:
  case ACC_ASYNC:
  case ACC_GANG:
  case ACC_NUM_GANGS:
  case ACC_WORKER:
  case ACC_NUM_WORKERS:
  case ACC_VECTOR:
  case ACC_VECT_LEN:
  case ACC_COLLAPSE:
      outxContext(fp,indent1+1,arg);
      break;

  default:
      namelist = (CExprOfList *)arg;
      if(EXPR_L_SIZE(namelist) != 0)
	  out_ACC_name_list(fp, indent1, namelist);
  }
  outxPrint(fp,indent,"</list>\n");
}

void out_ACC_name_list(FILE *fp,int indent, CExprOfList *list)
{
    int indent1 = indent+1;
    CCOL_DListNode *ite;
    outxPrint(fp,indent,"<list>\n");
    EXPR_FOREACH(ite, list) {
	CExpr *node = EXPR_L_DATA(ite);
	// outx_IDENT(fp,indent1+1,(CExprOfSymbol *)node);
	if(EXPR_CODE(node) == EC_ARRAY_REF){
	  out_ACC_arrayRef(fp,indent1, (CExprOfBinaryNode*)node);
	}else{
	outxPrint(fp,indent1,"<Var>%s</Var>\n",
		  ((CExprOfSymbol *)node)->e_symName);
	}
    }
    outxPrint(fp,indent,"</list>\n");
}

void out_ACC_arrayRef(FILE *fp,int indent, CExprOfBinaryNode *arrayRef)
{
    int indent1 = indent+1;
    CCOL_DListNode *ite;
    CExpr *arrayExpr = arrayRef->e_nodes[0];
    CExpr *subscripts = arrayRef->e_nodes[1];

    outxPrint(fp,indent,"<list>\n");

    outxPrint(fp,indent1,"<Var>%s</Var>\n",((CExprOfSymbol *)arrayExpr)->e_symName);
    EXPR_FOREACH(ite, subscripts){
      CExpr *node = EXPR_L_DATA(ite);
      out_ACC_subscript(fp, indent1, node);
    }

    outxPrint(fp,indent,"</list>\n");
}

void out_ACC_subscript(FILE *fp,int indent, CExpr *subscript)
{
  int indent1 = indent + 1;
  if(EXPR_CODE(subscript) != EC_UNDEF){
    outxContext(fp, indent, subscript); //single subscript
  }else{
    outxPrint(fp, indent, "<list>\n");

    CExpr *lower = exprListHeadData(subscript);
    CExpr *tmpLower = NULL;
    if(EXPR_ISNULL(lower)){
      lower = tmpLower = (CExpr*)allocExprOfNumberConst2(0, BT_INT);
    }

    outxContext(fp, indent1, lower);
    if(tmpLower){
      freeExpr(tmpLower);
    }

    if(EXPR_L_SIZE(subscript) > 1){
      CExpr *length = exprListNextNData(subscript, 1);
      if(! EXPR_ISNULL(length)){
	outxContext(fp,indent1, length);
      }
    }

    outxPrint(fp, indent, "</list>\n");
  }
}

char *accDirectiveName(int c)
{
  switch(c){
  case ACC_PARALLEL:  return "PARALLEL";
  case ACC_LOOP: return "LOOP";
  case ACC_KERNELS: return "KERNELS";
  case ACC_DATA: return "DATA";
  case ACC_HOST_DATA: return "HOST_DATA";
  case ACC_CACHE: return "CACHE";
  case ACC_DECLARE: return "DECLARE";
  case ACC_UPDATE: return "UPDATE";
  case ACC_WAIT: return "WAIT";
  case ACC_PARALLEL_LOOP: return "PARALLEL_LOOP";
  case ACC_KERNELS_LOOP:return "KERNELS_LOOP";
  default: return "??ACC??";
  }
}

char *accClauseName(int c)
{
  switch(c){
  case ACC_IF: return "IF";
  case ACC_ASYNC: return "ASYNC";
  case ACC_NUM_GANGS: return "NUM_GANGS";
  case ACC_NUM_WORKERS: return "NUM_WORKERS";
  case ACC_VECT_LEN: return "VECT_LEN";
  case ACC_COLLAPSE: return "COLLAPSE";
  case ACC_GANG: return "GANG";
  case ACC_WORKER: return "WORKER";
  case ACC_VECTOR: return "VECTOR";
  case ACC_SEQ: return "SEQ";
  case ACC_INDEPENDENT: return "INDEPENDENT";

  case ACC_HOST: return "HOST";
  case ACC_DEVICE: return "DEVICE";
  case ACC_DEVICEPTR: return "DEVICEPTR";
  case ACC_USE_DEVICE: return "USE_DEVICE";
  case ACC_DEV_RESIDENT: return "DEV_RESIDENT";

  case ACC_PRIVATE: return "PRIVATE";
  case ACC_FIRSTPRIVATE: return "FIRSTPRIVATE";
  case ACC_LASTPRIVATE: return "LASTPRIVATE";

  case ACC_CREATE: return "CREATE";
  case ACC_COPY: return "COPY";
  case ACC_COPYIN: return "COPYIN";
  case ACC_COPYOUT: return "COPYOUT";

  case ACC_PRESENT: return "PRESENT";
  case ACC_PRESENT_OR_CREATE: return "PRESENT_OR_CREATE";
  case ACC_PRESENT_OR_COPY: return "PRESENT_OR_COPY";
  case ACC_PRESENT_OR_COPYIN: return "PRESENT_OR_COPYIN";
  case ACC_PRESENT_OR_COPYOUT: return "PRESENT_OR_COPYOUT";

  case ACC_REDUCTION_PLUS: return "REDUCTION_PLUS";
  case ACC_REDUCTION_MINUS: return "REDUCTION_MINUS";
  case ACC_REDUCTION_MUL: return "REDUCTION_MUL";
  case ACC_REDUCTION_BITAND: return "REDUCTION_BITAND";
  case ACC_REDUCTION_BITOR: return "REDUCTION_BITOR";
  case ACC_REDUCTION_BITXOR: return "REDUCTION_BITXOR";
  case ACC_REDUCTION_LOGAND: return "REDUCTION_LOGAND";
  case ACC_REDUCTION_LOGOR: return "REDUCTION_LOGOR";
  case ACC_REDUCTION_MIN: return "REDUCTION_MIN";
  case ACC_REDUCTION_MAX: return "REDUCTION_MAX";

  default:  return "???ACC clause???";
  }
}

