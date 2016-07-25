#include <stdlib.h>
#include <stdarg.h> 
#include <wchar.h>
#include "c-comp.h"
#include "c-option.h"
#include "c-omp.h"

#include "c-xcodeml.h"

void outx_OMP_Clause(FILE *fp, int indent, CExprOfList* clause);
void out_OMP_name_list(FILE *fp,int indent, CExprOfList *list);

void
out_OMP_PRAGMA(FILE *fp, int indent, int pragma_code, CExpr* expr)
{
    CExprOfList *body = (CExprOfList *)expr;
    CExprOfList *clauseList = (CExprOfList *)body->e_aux_info;

    const char *ompPragmaTag = "OMPPragma";
    int indent1 = indent + 1;

    CCOL_DListNode *ite;
    outxTagForStmt(fp, indent,(CExpr*)clauseList, ompPragmaTag,0, NULL);
    outxPrint(fp,indent1,"<string>%s</string>\n",
	      ompDirectiveName(pragma_code));
    switch(pragma_code){
    case OMP_CRITICAL:
    case OMP_FLUSH:
    case OMP_THREADPRIVATE:
	out_OMP_name_list(fp, indent1, clauseList);
	goto end;
    }

    outxPrint(fp,indent1,"<list>\n");
    EXPR_FOREACH(ite, clauseList){
	CExpr *node = EXPR_L_DATA(ite);
	outx_OMP_Clause(fp,indent1+1,(CExprOfList *)node);
    }
    outxPrint(fp,indent1,"</list>\n");

  end:
    if(EXPR_L_SIZE(expr) != 0) outxChildren(fp,indent1,expr);
    outxTagClose(fp, indent,ompPragmaTag);
}

void
outx_OMP_Clause(FILE *fp, int indent, CExprOfList* clause)
{
  int indent1 = indent+1;
  CExpr *arg;
  CExprOfList *namelist;

  outxPrint(fp,indent,"<list>\n");
  outxPrint(fp,indent1,"<string>%s</string>\n",
	    ompClauseName(clause->e_aux));
  arg = exprListHeadData((CExpr *)clause);

  switch(clause->e_aux){
  case OMP_DIR_ORDERED:
  case OMP_DIR_NOWAIT:
      break;

  case OMP_DIR_IF:
  case OMP_DIR_NUM_THREADS:
  case OMP_COLLAPSE:
      outxContext(fp,indent1+1,arg);
      break;

  case OMP_DIR_SCHEDULE:
      outxPrint(fp,indent1,"<list>\n");
      outxPrint(fp,indent1+1,"<string>%s</string>\n",
		ompScheduleName(((CExprOfList *)arg)->e_aux));
      outxContext(fp,indent1+1,exprListHeadData(arg));
      outxPrint(fp,indent1,"</list>\n");
      break;

  case OMP_DATA_DEFAULT:
      outxPrint(fp,indent1+1,"<string>%s</string>\n",
		ompDataDefaultName(((CExprOfList *)arg)->e_aux));
      break;
      
  default:
      namelist = (CExprOfList *)arg;
      if(EXPR_L_SIZE(namelist) != 0)
	  out_OMP_name_list(fp, indent, namelist);
  }
  outxPrint(fp,indent,"</list>\n");
}

void out_OMP_name_list(FILE *fp,int indent, CExprOfList *list)
{
    int indent1 = indent+1;
    CCOL_DListNode *ite;
    outxPrint(fp,indent,"<list>\n");
    EXPR_FOREACH(ite, list) {
	CExpr *node = EXPR_L_DATA(ite);
	// outx_IDENT(fp,indent1+1,(CExprOfSymbol *)node);
	outxPrint(fp,indent1,"<Var>%s</Var>\n",
		  ((CExprOfSymbol *)node)->e_symName);
    }
    outxPrint(fp,indent,"</list>\n");
}

char *ompDirectiveName(int c)
{
  switch(c){
  case OMP_PARALLEL:  return "PARALLEL";
  case OMP_FOR: return "FOR";
  case OMP_SECTIONS: return "SECTIONS";
  case OMP_SECTION: return "SECTION";
  case OMP_SINGLE: return "SINGLE";
  case OMP_MASTER: return "MASTER";
  case OMP_CRITICAL: return "CRITICAL";
  case OMP_BARRIER: return "BARRIER";
  case OMP_ATOMIC: return "ATOMIC";
  case OMP_FLUSH:return "FLUSH";
  case OMP_ORDERED:return "ORDERED";
  case OMP_THREADPRIVATE:return "THREADPRIVATE";
  case OMP_PARALLEL_FOR:return "PARALLEL_FOR";
  case OMP_PARALLEL_SECTIONS:return "PARALLEL_SECTIONS";
  default: return "OMP???";
  }
}

char *ompClauseName(int c)
{
  switch(c){
  case OMP_DATA_DEFAULT: return "DATA_DEFAULT";
  case OMP_DATA_PRIVATE: return "DATA_PRIVATE";
  case OMP_DATA_SHARED: return "DATA_SHARED";
  case OMP_DATA_FIRSTPRIVATE: return "DATA_FIRSTPRIVATE";
  case OMP_DATA_LASTPRIVATE: return "DATA_LASTPRIVATE";
  case OMP_DATA_COPYIN: return "DATA_COPYIN";

  case OMP_DATA_REDUCTION_PLUS: return "DATA_REDUCTION_PLUS";
  case OMP_DATA_REDUCTION_MINUS: return "DATA_REDUCTION_MINUS";
  case OMP_DATA_REDUCTION_MUL: return "DATA_REDUCTION_MUL";
  case OMP_DATA_REDUCTION_BITAND: return "DATA_REDUCTION_BITAND";
  case OMP_DATA_REDUCTION_BITOR: return "DATA_REDUCTION_BITOR";
  case OMP_DATA_REDUCTION_BITXOR: return "DATA_REDUCTION_BITXOR";
  case OMP_DATA_REDUCTION_LOGAND: return "DATA_REDUCTION_LOGAND";
  case OMP_DATA_REDUCTION_LOGOR: return "DATA_REDUCTION_LOGOR";
  case OMP_DATA_REDUCTION_MIN: return "DATA_REDUCTION_MIN";
  case OMP_DATA_REDUCTION_MAX: return "DATA_REDUCTION_MAX";

  case OMP_DIR_ORDERED: return "DIR_ORDERED";
  case OMP_DIR_IF: return "DIR_IF";
  case OMP_DIR_NOWAIT: return "DIR_NOWAIT";
  case OMP_DIR_SCHEDULE: return "DIR_SCHEDULE";
  case OMP_DIR_NUM_THREADS: return "DIR_NUM_THREADS";
  case OMP_COLLAPSE: return "COLLAPSE";
  default:  return "???OMP???";
  }
}

char *ompScheduleName(int c)
{
    switch(c){
    case  OMP_SCHED_NONE: return "SCHED_NONE";
    case OMP_SCHED_STATIC: return "SCHED_STATIC";
    case OMP_SCHED_DYNAMIC: return "SCHED_DYNAMIC";
    case OMP_SCHED_GUIDED: return "SCHED_GUIDED";
    case OMP_SCHED_RUNTIME: return "SCHED_RUNTIME";
    case OMP_SCHED_AFFINITY: return "SCHED_AFFINITY";
    default: 
	return "SCHED_???";
    }
}

char *ompDataDefaultName(int c)
{
    switch(c){
    case OMP_DEFAULT_NONE:  return "DEFAULT_NONE";
    case OMP_DEFAULT_SHARED:  return "DEFAULT_SHARED";
    case OMP_DEFAULT_PRIVATE:  return "DEFAULT_PRIVATE";
    default:
	return "DEFAULT_???";
    }
}
