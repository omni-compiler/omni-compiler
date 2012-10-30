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

#include "c-xmp.h"
#include "c-xcodeml.h"

void outx_XMP_Clause(FILE *fp, int indent, CExprOfList* clause);
char *xmpDirectiveName(int c);

void
out_XMP_PRAGMA(FILE *fp, int indent, int pragma_code, CExpr* expr)
{
    CExprOfList *body = (CExprOfList *)expr;
    CExprOfList *clauseList = (CExprOfList *)body->e_aux_info;

    const char *ompPragmaTag = "XMPPragma";
    int indent1 = indent + 1;

    outxTagForStmt(fp, indent,(CExpr*)clauseList, ompPragmaTag,0, NULL);
    outxPrint(fp,indent1,"<string>%s</string>\n",
	      xmpDirectiveName(pragma_code));
    outx_XMP_Clause(fp,indent1,clauseList);
    if(EXPR_L_SIZE(expr) != 0) outxChildren(fp,indent1,expr); // body
    outxTagClose(fp, indent,ompPragmaTag);
}

void outx_XMP_Clause(FILE *fp, int indent, CExprOfList* clauseList)
{
    int indent1 = indent + 1;
    CCOL_DListNode *ite;

    outxPrint(fp,indent1,"<list>\n");
    EXPR_FOREACH(ite, clauseList){
	CExpr *node = EXPR_L_DATA(ite);
	if(node == NULL) 
	    outxPrint(fp,indent1,"<list/>\n");
	else if(EXPR_CODE(node) == EC_UNDEF)
	    outx_XMP_Clause(fp,indent1,(CExprOfList *)node);
	else
	    outxContext(fp,indent1,node);
    }
    outxPrint(fp,indent1,"</list>\n");
}

char *xmpDirectiveName(int c)
{
  switch(c){
  case XMP_NODES: return "NODES";
  case XMP_TEMPLATE: return "TEMPLATE";
  case XMP_DISTRIBUTE: return "DISTRIBUTE";
  case XMP_ALIGN: return "ALIGN";
  case XMP_SHADOW: return "SHADOW";
  case XMP_TASK: return "TASK";
  case XMP_TASKS: return "TASKS";
  case XMP_LOOP: return "LOOP";
  case XMP_REFLECT: return "REFLECT";
  case XMP_GMOVE: return "GMOVE";
  case XMP_BARRIER: return "BARRIER";
  case XMP_REDUCTION: return "REDUCTION";
  case XMP_BCAST: return "BCAST";
  case XMP_COARRAY: return "COARRAY";
  case XMP_TEMPLATE_FIX: return "TEMPLATE_FIX";
  default: return "OMP???";
  }
}
