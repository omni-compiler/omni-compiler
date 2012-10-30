/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */

#include <sys/param.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>

#include "c-expr.h"
#include "c-pragma.h"
#include "c-parser.h"
#include "c-const.h"
#include "c-option.h"
#include "c-xmp.h"

/*
 * <XMPPragma> <string> directive_name </string> args_list body </XMPPragma>
 *
 * nodes directive:
 *  args = <list> name_list node_size_list inherit_attr </list>
 *    name_list = ident | <list> ident1 ident2 ... </list>
 *    node_size_list = <list> size1 size2 ... </list> 
 *
 * template directive:
 * distribute directive:
 * align
 * shadow
 * task
 * tasks
 * loop
 * reflect
 * barrier
 * reduction
 * bcast
 *.
 */

static int parse_XMP_pragma(void);

static CExpr* parse_NODES_clause();
static CExpr* parse_TEMPLATE_clause();
static CExpr* parse_DISTRIBUTE_clause();
static CExpr* parse_ALIGN_clause();
static CExpr* parse_SHADOW_clause();
static CExpr* parse_TASK_clause();
static CExpr* parse_TASKS_clause();
static CExpr* parse_LOOP_clause();
static CExpr* parse_REFLECT_clause();
static CExpr* parse_REDUCTION_clause();
static CExpr* parse_BARRIER_clause();
static CExpr* parse_BCAST_clause();
static CExpr* parse_GMOVE_clause();

static CExpr* parse_COARRAY_clause();
static CExpr* parse_LOCAL_ALIAS_clause();

static CExpr* parse_COL2_name_list();
static CExpr* parse_XMP_subscript_list();
static CExpr* parse_XMP_C_subscript_list();
static CExpr* parse_ON_ref();
static CExpr* parse_XMP_dist_fmt_list();
static CExpr* parse_Reduction_opt();
static CExpr* parse_XMP_opt();

static CExpr* _xmp_pg_list(int omp_code,CExpr* args);

#define XMP_PG_LIST(pg,args) _xmp_pg_list(pg,args)
#define XMP_LIST1(arg1) (CExpr*)allocExprOfList1(EC_UNDEF,arg1)
#define XMP_LIST2(arg1,arg2) (CExpr*)allocExprOfList2(EC_UNDEF,arg1,arg2)
#define XMP_LIST3(arg1,arg2,arg3) (CExpr*)allocExprOfList3(EC_UNDEF,arg1,arg2,arg3)
#define XMP_LIST4(arg1,arg2,arg3,arg4) (CExpr*)allocExprOfList4(EC_UNDEF,arg1,arg2,arg3,arg4)

#define XMP_Error0(msg) addError(NULL,msg)
#define XMP_error1(msg,arg1) addError(NULL,msg,arg1)

static CExpr* _xmp_pg_list(int xmp_code,CExpr* args)
{
  CExprOfList *lp;
  lp = allocExprOfList1(EC_UNDEF,args);
  lp->e_aux = xmp_code;
  return (CExpr *)lp;
}

#define EMPTY_LIST (CExpr *)allocExprOfList(EC_UNDEF)

static int pg_XMP_pragma;
CExpr* pg_XMP_list;

int XMP_has_err = 0;

/*
 * for XcalableMP directives
 */
CExpr*
lexParsePragmaXMP(char *p, int *token) // p is buffer
{
  //skip pragma[space]xmp[space]*
  p = lexSkipSpace(lexSkipWordP(lexSkipSpace(lexSkipWord(lexSkipSpace(p)))));

  pg_cp = p; // set the pointer

  *token = parse_XMP_pragma();

  if(pg_XMP_list == NULL) pg_XMP_list = EMPTY_LIST;
  ((CExprOfList *)pg_XMP_list)->e_aux = pg_XMP_pragma; // attached aux
  
  return pg_XMP_list;
}

int parse_XMP_pragma()
{
    int ret = PRAGMA_EXEC; /* default */
    pg_XMP_list = NULL;

    pg_get_token();
    if(pg_tok != PG_IDENT) goto syntax_err;

    if (PG_IS_IDENT("nodes")) {
	pg_XMP_pragma = XMP_NODES;
	pg_get_token();
	pg_XMP_list = parse_NODES_clause();
    }
    else if (PG_IS_IDENT("template")) {
	pg_XMP_pragma = XMP_TEMPLATE;
	pg_get_token();
	pg_XMP_list = parse_TEMPLATE_clause();
    }
    else if (PG_IS_IDENT("distribute")) {
	pg_XMP_pragma = XMP_DISTRIBUTE;
	pg_get_token();
	pg_XMP_list = parse_DISTRIBUTE_clause();
    }
    else if (PG_IS_IDENT("align")) {
	pg_XMP_pragma = XMP_ALIGN;
	pg_get_token();
	pg_XMP_list = parse_ALIGN_clause();
    }
    else if (PG_IS_IDENT("shadow")) {
	pg_XMP_pragma = XMP_SHADOW;
	pg_get_token();
	pg_XMP_list = parse_SHADOW_clause();
    }
    else if (PG_IS_IDENT("task")) {
	pg_XMP_pragma = XMP_TASK;
	ret = PRAGMA_PREFIX;
	pg_get_token();
	pg_XMP_list = parse_TASK_clause();
    }
    else if (PG_IS_IDENT("tasks")) {
	pg_XMP_pragma = XMP_TASKS;
	ret = PRAGMA_PREFIX;
	pg_get_token();
	pg_XMP_list = parse_TASKS_clause();
    }
    else if (PG_IS_IDENT("loop")) {
	pg_XMP_pragma = XMP_LOOP;
	ret = PRAGMA_PREFIX;
	pg_get_token();
	pg_XMP_list = parse_LOOP_clause();
    }
    else if (PG_IS_IDENT("reflect")) {
	pg_XMP_pragma = XMP_REFLECT;
	pg_get_token();
	pg_XMP_list = parse_REFLECT_clause();
    }
    else if (PG_IS_IDENT("barrier")) {
	pg_XMP_pragma = XMP_BARRIER;
	pg_get_token();
	pg_XMP_list = parse_BARRIER_clause();
    }
    else if (PG_IS_IDENT("reduction")) {
	pg_XMP_pragma = XMP_REDUCTION;
	pg_get_token();
	pg_XMP_list = parse_REDUCTION_clause();
    }
    else if (PG_IS_IDENT("bcast")) {
	pg_XMP_pragma = XMP_BCAST;
	pg_get_token();
	pg_XMP_list = parse_BCAST_clause();
    }
    else if (PG_IS_IDENT("gmove")) {
	pg_XMP_pragma = XMP_GMOVE;
	ret = PRAGMA_PREFIX;
	pg_get_token();
	pg_XMP_list = parse_GMOVE_clause();
    }
    else if (PG_IS_IDENT("coarray")) {
	pg_XMP_pragma = XMP_COARRAY;
	pg_get_token();
	pg_XMP_list = parse_COARRAY_clause();
#ifdef not
    } else if (PG_IS_IDENT("sync_memory")) {
	pg_XMP_pragma = XMP_SYNC_MEMORY;
	pg_get_token();
	pg_XMP_list = null;
    } else if (PG_IS_IDENT("sync_all")) {
	pg_XMP_pragma = XMP_SYNC_ALL;
	pg_get_token();
	pg_XMP_list = null;
    } else if (PG_IS_IDENT("local_alias")) {
	pg_XMP_pragma = XMP_LOCAL_ALIAS;
	pg_get_token();
	pg_XMP_list = parse_LOCAL_ALIAS_clause();
#endif
    } else {
	addError(NULL,"unknown XcalableMP directive, '%s'",pg_tok_buf);
      syntax_err:
	return 0;
    }

    if(pg_tok != 0) addError(NULL,"extra arguments for XMP directive");
    return ret;
}

CExpr* parse_NODES_clause() {
    CExpr* nodesNameList = NULL;
    CExpr* nodesSizeList, *inheritedNodes;

    // parse <nodes-name>
    if (pg_tok == PG_IDENT) {
	nodesNameList = XMP_LIST1(pg_tok_val);
	pg_get_token();
    } 

    // parse (<nodes-size>, ...)
    if (pg_tok != '('){
	addError(NULL, "'(' is expected after <nodes-name>");
	goto err;
    } 
    nodesSizeList = parse_XMP_subscript_list();

    // parse { <empty> | =* | =<nodes-ref> }
    if (pg_tok == '=') {
      pg_get_token();
      if (pg_tok == '*') {
	  pg_get_token();
	  inheritedNodes = XMP_PG_LIST(XMP_NODES_INHERIT_EXEC,NULL);
      } else {
	  inheritedNodes = XMP_PG_LIST(XMP_NODES_INHERIT_NODES, 
				       parse_ON_ref());
      } 
    } else 
	inheritedNodes = NULL;
    
    if (nodesNameList == NULL) 
	nodesNameList = parse_COL2_name_list();
    
    return XMP_LIST3(nodesNameList, nodesSizeList, inheritedNodes);

  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_TEMPLATE_clause()
{
    CExpr* templateNameList = NULL;
    CExpr* templateSpecList;

    // parse <template-name>
    if (pg_tok == PG_IDENT) {
	templateNameList = XMP_LIST1(pg_tok_val);
	pg_get_token();
    } 

    // parse (<template-spec>, ...)
    if (pg_tok != '(') {
	XMP_Error0("'(' is expected after <template-name>");
	goto err;
    } else 
	templateSpecList = parse_XMP_subscript_list();

    if (templateNameList == NULL) 
	templateNameList = parse_COL2_name_list();
	
    return XMP_LIST2(templateNameList, templateSpecList);

  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr* parse_DISTRIBUTE_clause() 
{
    CExpr* templateNameList = NULL;
    CExpr* distFormatList, *nodesName;
    
    // parse <template-name>
    if (pg_tok == PG_IDENT) {
	templateNameList = XMP_LIST1(pg_tok_val);
	pg_get_token();
    } 

    // parse (<dist-format>, ...)
    if (pg_tok != '('){
	XMP_Error0("'(' is expected after <template-name>");
	goto err;
    } else
	distFormatList = parse_XMP_dist_fmt_list();
    
    if(PG_IS_IDENT("onto")){
        pg_get_token();
    } else {
	XMP_Error0("onto is missing");
	goto err;
    }

    if (pg_tok == PG_IDENT){
	nodesName = pg_tok_val;
	pg_get_token();
    } else {
	XMP_Error0("<nodes-name> is expected after 'onto'");
	goto err;
    }

    if (templateNameList == NULL) 
	templateNameList = parse_COL2_name_list();
    
    return XMP_LIST3(templateNameList, distFormatList, nodesName);

  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr* parse_ALIGN_clause()
{
    CExpr* arrayNameList = NULL;
    CExpr* alignSourceList, *alignSubscriptList, *templateName;

    // parse <array-name>
    if (pg_tok == PG_IDENT) {
	arrayNameList = XMP_LIST1(pg_tok_val);
	pg_get_token();
    } 

    // parse [align-source] ...
    if (pg_tok != '['){
      XMP_Error0("'[' is expected");
      goto err;
    } else 
	alignSourceList = parse_XMP_C_subscript_list();

    if(PG_IS_IDENT("with"))
	pg_get_token();
    else {
	XMP_Error0("'with' is missing");
	goto err;
    }

    if (pg_tok == PG_IDENT){
	templateName = pg_tok_val;
	pg_get_token();
    } else {
	XMP_Error0("<template-name> is expected after 'with'");
	goto err;
    }

    alignSubscriptList = parse_XMP_subscript_list();

    if (arrayNameList == NULL) 
	arrayNameList = parse_COL2_name_list();

    return XMP_LIST4(arrayNameList, alignSourceList, 
		     templateName, alignSubscriptList);
  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr* parse_SHADOW_clause() {
    CExpr* arrayNameList = NULL;
    CExpr* shadowWidthList;

    // parse <array-name>
    if (pg_tok == PG_IDENT) {
	arrayNameList = XMP_LIST1(pg_tok_val);
	pg_get_token();
    } 

    // parse [shadow-width] ...
    if (pg_tok != '['){
      XMP_Error0("'[' is expected");
      goto err;
    } else 
	shadowWidthList = parse_XMP_C_subscript_list();

    if (arrayNameList == NULL) 
	arrayNameList = parse_COL2_name_list();

    return XMP_LIST2(arrayNameList, shadowWidthList);

  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr* parse_TASK_clause() {
    CExpr* onRef = NULL;
    CExpr* opt;

    if(PG_IS_IDENT("on"))
	pg_get_token();
    else {
	XMP_Error0("'on' is missing");
	goto err;
    }
	
    onRef = parse_ON_ref();
    opt = parse_XMP_opt();
    
    return XMP_LIST2(onRef,opt);

  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr* parse_LOOP_clause()
{
    CExpr *subscriptList = NULL;
    CExpr *onRef, *reductionOpt, *opt;

    if(pg_tok == '('){
	subscriptList = parse_XMP_subscript_list();
    }

    if(PG_IS_IDENT("on"))
	pg_get_token();
    else {
	XMP_Error0("'on' is missing");
	goto err;
    }
    
    onRef = parse_ON_ref();
    reductionOpt = parse_Reduction_opt();
    opt = parse_XMP_opt();
    
    return XMP_LIST4(subscriptList,onRef,reductionOpt,opt);

  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_XMP_subscript_list()
{
    CExpr* list;
    CExpr *v1,*v2,*v3;

    list = EMPTY_LIST;
    if(pg_tok != '(') {
	addFatal(NULL,"parse_XMP_subscript_list: first token= '('");
    }

    pg_get_token();
    while(1){
	v1 = v2 = v3 = NULL;
	switch(pg_tok){
	case ')':  goto err;
	case ',':  goto err;
	case ':':
	    break;
	default:
	    v1 = pg_parse_expr();
	}
	
	if(pg_tok != ':') goto next;
	pg_get_token();
	switch(pg_tok){
	case ')':  goto next;
	case ',':  goto next;
	case ':':
	    break;
	default:
	    v2 = pg_parse_expr();
	}

	if(pg_tok != ':') goto next;
	pg_get_token();
	v3 = pg_parse_expr(); 
	
      next:
	list = exprListAdd(list, XMP_LIST3(v1,v2,v3));
	if(pg_tok == ')'){
	    pg_get_token();
	    break;
	}
	if(pg_tok == ',')  pg_get_token();
	else goto err;
    }

    return list;

  err:
    XMP_Error0("Syntax error in scripts of XMP directive");
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_XMP_C_subscript_list()
{
    CExpr* list;
    CExpr *v1,*v2,*v3;

    list = EMPTY_LIST;
    if(pg_tok != '[') {
	addFatal(NULL,"parse_XMP_C_subscript_list: first token != '['");
    }

    while(1){
	v1 = v2 = v3 = NULL;
	if(pg_tok != '[') break;
	pg_get_token();

	switch(pg_tok){
	case ']':  goto err;
	case ':':
	    break;
	default:
	    v1 = pg_parse_expr();
	}
	
	if(pg_tok != ':') goto next;
	pg_get_token();
	switch(pg_tok){
	case ']':  goto next;
	case ':':
	    break;
	default:
	    v2 = pg_parse_expr();
	}

	if(pg_tok != ':') goto next;
	pg_get_token();
	v3 = pg_parse_expr(); 
	
      next:
	list = exprListAdd(list, XMP_LIST3(v1,v2,v3));
	if(pg_tok == ']')  pg_get_token();
	else goto err;
    }
    return list;

  err:
    XMP_Error0("Syntax error in scripts of XMP directive");
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_XMP_dist_fmt_list()
{
    CExpr* list;
    CExpr *v, *width;

    list = EMPTY_LIST;
    if(pg_tok != '(') {
	addFatal(NULL,"parse_XMP_dist_fmt_list: first token= '('");
    }
    
    pg_get_token();
    while(1){
	// FIXME support gblock
	// parse <dist-format> := { * | block(n) | cyclic(n) }
	width = NULL;
	if (pg_tok == '*') {
	    pg_get_token();
	    v = XMP_PG_LIST(XMP_DIST_DUPLICATION,NULL);
	} else if (PG_IS_IDENT("block")) {
	    pg_get_token();
	    if (pg_tok == '(') {
		pg_get_token();
		width = pg_parse_expr();
		if (pg_tok != ')') {
		    XMP_Error0("')' is needed after <block-width>");
		    goto err;
		} 
		pg_get_token();
	    }
	    v = XMP_PG_LIST(XMP_DIST_BLOCK,width);
	} else if (PG_IS_IDENT("cyclic")) {
	    pg_get_token();
	    if (pg_tok == '(') {
		pg_get_token();
		width = pg_parse_expr();
		if (pg_tok != ')') {
		    XMP_Error0("')' is needed after <cyclic-width>");
		    goto err;
		} 
		pg_get_token();
	    }
	    v = XMP_PG_LIST(XMP_DIST_CYCLIC,width);
	} else goto syntax_err;

	list = exprListAdd(list, v);

	if(pg_tok == ')'){
	    pg_get_token();
	    break;
	} else if(pg_tok == ','){
	    pg_get_token();
	    continue;
	} else goto syntax_err;
	
    }
    return list;

  syntax_err:
    XMP_Error0("syntax error in distribution description");
  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_COL2_name_list()
{
    CExpr* list;
    
    list = EMPTY_LIST;
    if (pg_tok == PG_COL2) {
        pg_get_token();
	while(pg_tok == PG_IDENT){
	    list = exprListAdd(list, pg_tok_val);
	    pg_get_token();
	    if(pg_tok != ',') break;
	    pg_get_token();
	}
	return list;
    }
    XMP_Error0("name list is expected after :: ");
    return NULL;
}

/* (xmp_sbuscript) or id(xmp_subscript) */
CExpr *parse_ON_ref()
{
    CExpr *ident, *subscript;
    ident = NULL;

    if(pg_tok == PG_IDENT) {
	ident = pg_tok_val;
	pg_get_token();
    }

    if(pg_tok != '('){
	XMP_Error0("syntax error in reference object by 'on'");
	XMP_has_err = 1;
	return NULL;
    }
    subscript = parse_XMP_subscript_list();
    return XMP_LIST2(ident,subscript);
}

CExpr *parse_Reduction_opt()
{
    int op;
    CExpr *list;
    
    if(!PG_IS_IDENT("reduction")) return NULL;

    pg_get_token();
    if(pg_tok != '(') goto err;
    pg_get_token();
    switch(pg_tok){
    case '+':
	op = XMP_DATA_REDUCE_SUM; break;
    case '-':
	op = XMP_DATA_REDUCE_MINUS; break;
    case '*':
	op = XMP_DATA_REDUCE_PROD; break;
    case '|':
	op = XMP_DATA_REDUCE_BOR; break;
    case '&':
	op = XMP_DATA_REDUCE_BAND; break;
    case PG_ANDAND:
	op = XMP_DATA_REDUCE_LAND; break;
    case PG_OROR:
	op = XMP_DATA_REDUCE_LOR; break;
    case PG_IDENT:
	if(PG_IS_IDENT("max")) op = XMP_DATA_REDUCE_MAX;
	else if(PG_IS_IDENT("min")) op = XMP_DATA_REDUCE_MIN;
	else if(PG_IS_IDENT("firstmax")) op = XMP_DATA_REDUCE_FIRSTMAX;
	else if(PG_IS_IDENT("firstmin")) op = XMP_DATA_REDUCE_FIRSTMIN;
	else if(PG_IS_IDENT("lastmax")) op = XMP_DATA_REDUCE_LASTMAX;
	else if(PG_IS_IDENT("lastmin")) op = XMP_DATA_REDUCE_LASTMIN;
	else goto unknown_err;
	break;
    default:
      unknown_err:
	XMP_Error0("unknown operation in reduction clause");
	goto ret;
    }
    pg_get_token();
    if(pg_tok != ':') goto err;
    pg_get_token();
    list = EMPTY_LIST;
    while(pg_tok == PG_IDENT){
	list = exprListAdd(list, pg_tok_val);
	pg_get_token();
	if(pg_tok != ',') break;
	pg_get_token();
    }
    if(pg_tok != ')') goto err;
    pg_get_token();
    return XMP_PG_LIST(op,list);
  err:
    XMP_Error0("syntax error in reduction clause");
  ret:
    return NULL;
}

CExpr *parse_XMP_opt()
{
    return NULL;
}

CExpr* parse_TASKS_clause()
{
    return NULL;
}


static CExpr* parse_REFLECT_clause()
{
    return NULL;
}


static CExpr* parse_REDUCTION_clause()
{
    return NULL;
}

static CExpr* parse_BARRIER_clause()
{
    return NULL;
}

static CExpr* parse_BCAST_clause()
{
    return NULL;
}

static CExpr* parse_GMOVE_clause()
{
    return NULL;
}

static CExpr* parse_COARRAY_clause()
{
    return NULL;
}

static CExpr* parse_LOCAL_ALIAS_clause()
{
    return NULL;
}

