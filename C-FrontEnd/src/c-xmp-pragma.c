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
static CExpr* parse_ARRAY_clause();
static CExpr* parse_POST_clause();
static CExpr* parse_WAIT_clause();
static CExpr* parse_LOCAL_ALIAS_clause();
static CExpr* parse_WIDTH_list();
static CExpr* parse_WAIT_ASYNC_clause();
static CExpr* parse_TEMPLATE_FIX_clause();

static CExpr* parse_COL2_name_list();
static CExpr* parse_XMP_subscript_list();
static CExpr* parse_XMP_size_list();
static CExpr* parse_XMP_range_list();
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
    }
    else if (PG_IS_IDENT("array")) {
	pg_XMP_pragma = XMP_ARRAY;
	ret = PRAGMA_PREFIX;
	pg_get_token();
	pg_XMP_list = parse_ARRAY_clause();
    }
    else if (PG_IS_IDENT("post")) {
	pg_XMP_pragma = XMP_POST;
	pg_get_token();
	pg_XMP_list = parse_POST_clause();
    }
    else if (PG_IS_IDENT("wait")) {
	pg_XMP_pragma = XMP_WAIT;
	pg_get_token();
	pg_XMP_list = parse_WAIT_clause();
    }
    else if (PG_IS_IDENT("wait_async")) {
      pg_XMP_pragma = XMP_WAIT_ASYNC;
      pg_get_token();
      pg_XMP_list = parse_WAIT_ASYNC_clause();
    }
    else if (PG_IS_IDENT("template_fix")) {
      pg_XMP_pragma = XMP_TEMPLATE_FIX;
      pg_get_token();
      pg_XMP_list = parse_TEMPLATE_FIX_clause();
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

CExpr* parse_task_ON_ref();

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
    //nodesSizeList = parse_XMP_subscript_list();
    nodesSizeList = parse_XMP_size_list();

    // parse { <empty> | =* | =<nodes-ref> }
    if (pg_tok == '=') {
      pg_get_token();
      if (pg_tok == '*') {
	  pg_get_token();
	  inheritedNodes = XMP_PG_LIST(XMP_NODES_INHERIT_EXEC,NULL);
      } else {
	  inheritedNodes = XMP_PG_LIST(XMP_NODES_INHERIT_NODES, 
				       parse_task_ON_ref());
      } 
    } else 
      inheritedNodes = XMP_PG_LIST(XMP_NODES_INHERIT_GLOBAL, NULL);
    
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
      //templateSpecList = parse_XMP_subscript_list();
      templateSpecList = parse_XMP_range_list();

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

CExpr *parse_XMP_align_source_list(void);
CExpr *parse_XMP_align_subscript_list(void);

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
      //alignSourceList = parse_XMP_C_subscript_list();
      alignSourceList = parse_XMP_align_source_list();

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

    //    alignSubscriptList = parse_XMP_subscript_list();
    alignSubscriptList = parse_XMP_align_subscript_list();

    if (arrayNameList == NULL) 
	arrayNameList = parse_COL2_name_list();

    return XMP_LIST4(arrayNameList, alignSourceList, 
		     templateName, alignSubscriptList);
  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_XMP_shadow_width_list();

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
      //shadowWidthList = parse_XMP_C_subscript_list();
      shadowWidthList = parse_XMP_shadow_width_list();

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
	
    //onRef = parse_ON_ref();
    onRef = parse_task_ON_ref();
    opt = parse_XMP_opt();
    
    return XMP_LIST2(onRef,opt);

  err:
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_XMP_loop_subscript_list();

CExpr* parse_LOOP_clause()
{
    CExpr *subscriptList = NULL;
    CExpr *onRef, *reductionOpt, *opt;

    if(pg_tok == '('){
      //subscriptList = parse_XMP_subscript_list();
      subscriptList = parse_XMP_loop_subscript_list();
    }

    if(PG_IS_IDENT("on"))
	pg_get_token();
    else {
	XMP_Error0("'on' is missing");
	goto err;
    }
    
    onRef = parse_ON_ref();
    CExpr *reduction_opt = parse_Reduction_opt();
    reductionOpt = reduction_opt ? XMP_LIST1(reduction_opt) : EMPTY_LIST;
    opt = parse_XMP_opt();
    
    return XMP_LIST4(subscriptList,onRef,reductionOpt,opt);

  err:
    XMP_has_err = 1;
    return NULL;
}

#define SHADOW_NONE   400
#define SHADOW_NORMAL 401
#define SHADOW_FULL   402

CExpr *parse_XMP_shadow_width_list()
{
    CExpr* list;
    CExpr *v1,*v2;
    CExpr *type;

    list = EMPTY_LIST;
    if(pg_tok != '[') {
	addFatal(NULL,"parse_XMP_shadow_width_list: first token= '('");
    }

    while(1){

      v1 = v2 = NULL;
      type = (CExpr*)allocExprOfNumberConst2(SHADOW_NORMAL, BT_INT);

      if (pg_tok != '[') break;
      pg_get_token();

      switch(pg_tok){
      case ']':
      case ',':
      case ':':
	goto err;
      case '*':
	type = (CExpr*)allocExprOfNumberConst2(SHADOW_FULL, BT_INT);
	pg_get_token();
	goto next;
      default:
	v1 = pg_parse_expr();
      }
	
      if (pg_tok != ':'){
	v2 = v1;
	goto next;
      }

      pg_get_token();
      switch(pg_tok){
      case ']':
      case ',':
      case ':':
	goto err;
      default:
	v2 = pg_parse_expr();
      }

    next:
      if (v1 && v2 && isConstZero(v1) && isConstZero(v2)){
	type = (CExpr*)allocExprOfNumberConst2(SHADOW_NONE, BT_INT);
      }
      list = exprListAdd(list, XMP_LIST2(type, XMP_LIST2(v1,v2)));
      if(pg_tok == ']') pg_get_token();
      else goto err;
    }

    return list;

  err:
    XMP_Error0("Syntax error in scripts of XMP directive");
    XMP_has_err = 1;

    return NULL;
}

CExpr *parse_XMP_align_subscript_list()
{
    CExpr *list_var, *list_expr;
    CExpr *v, *var, *expr;

    list_var = EMPTY_LIST;
    list_expr = EMPTY_LIST;

    if(pg_tok != '(') {
	addFatal(NULL,"parse_XMP_align_subscript_list: first token= '('");
    }

    pg_get_token();

    while(1){

	v = NULL;

	switch(pg_tok){
	case ')':  goto err;
	case ',':  goto err;
	case ':':
	    break;
	case '*':
	  var = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "* @{ASTERISK}@", CT_UNDEF);
	  expr = (CExpr*)allocExprOfNumberConst2(0, BT_INT);
	  pg_get_token();
	  goto next;
	default:
	    v = pg_parse_expr();
	}

	switch (EXPR_CODE(v)){
	case EC_PLUS:
	  var = EXPR_B(v)->e_nodes[0];
	  expr = EXPR_B(v)->e_nodes[1];
	  break;

	case EC_MINUS:
	  var = EXPR_B(v)->e_nodes[0];
	  expr = exprUnary(EC_UNARY_MINUS, EXPR_B(v)->e_nodes[1]);
	  break;

	case EC_IDENT:
	  var = v;
	  expr = (CExpr*)allocExprOfNumberConst2(0, BT_INT);
	  break;

	default:
	  goto err;
	}

    next:

	list_var = exprListAdd(list_var, var);
	list_expr = exprListAdd(list_expr, expr);

	if (pg_tok == ')'){
	    pg_get_token();
	    break;
	}

	if (pg_tok == ',') pg_get_token();
	else goto err;
    }

    return XMP_LIST2(list_var, list_expr);

  err:
    XMP_Error0("Syntax error in scripts of XMP directive");
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
	case '*':
	  list = exprListAdd(list, NULL);
	  pg_get_token();
	  goto next2;
	default:
	    v1 = pg_parse_expr();
	}
	
	if(pg_tok != ':'){ // scalar
	  v2 = v1;
	  v3 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
	  goto next;
	}

	pg_get_token();
	switch(pg_tok){
	case ')':
	case ',':
	  goto next;
	case ':':
	    break;
	default:
	    v2 = pg_parse_expr();
	}

	if(pg_tok != ':') goto next;
	pg_get_token();
	v3 = pg_parse_expr(); 
	
      next:
	if (v3 == NULL) v3 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
	list = exprListAdd(list, XMP_LIST3(v1,v2,v3));

      next2:
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


CExpr *parse_XMP_size_list()
{
    CExpr* list;
    CExpr *v;

    list = EMPTY_LIST;
    if(pg_tok != '(') {
	addFatal(NULL,"parse_XMP_size_list: first token= '('");
    }

    pg_get_token();
    while(1){
	v = NULL;
	switch(pg_tok){
	case ')':
	case ',':
	case ':':
	  goto err;
	case '*':
	  //v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "* @{ASTERISK}@", CT_UNDEF);
	  v = NULL;
	  pg_get_token();
	  break;
	default:
	    v = pg_parse_expr();
	}
	
	list = exprListAdd(list, v);
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


CExpr *parse_XMP_range_list()
{
    CExpr* list;
    CExpr *v1,*v2;

    list = EMPTY_LIST;
    if(pg_tok != '(') {
	addFatal(NULL,"parse_XMP_range_list: first token= '('");
    }

    pg_get_token();
    while(1){
	v1 = v2 = NULL;
	switch(pg_tok){
	case ')':
	case ',':
	  goto err;
	case ':':
	  pg_get_token();
	  goto next;
	default:
	    v1 = pg_parse_expr();
	}
	
	if(pg_tok != ':'){
	  v2 = v1;
	  v1 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
	  goto next;
	}

	pg_get_token();
	switch(pg_tok){
	case ')':
	case ',':
	  v2 = v1;
	  v1 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
	  goto next;
	case ':':
	  goto err;
	default:
	    v2 = pg_parse_expr();
	}

      next:
	if (v1 == NULL && v2 == NULL)
	  list = exprListAdd(list, NULL);
	else
	  list = exprListAdd(list, XMP_LIST2(v1,v2));

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


CExpr *parse_XMP_loop_subscript_list()
{
    CExpr* list;
    CExpr *v;

    list = EMPTY_LIST;

    if (pg_tok != '('){
	addFatal(NULL,"parse_XMP_loop_subscript_list: first token != '['");
    }

    pg_get_token();

    while (1){

	switch (pg_tok){
	case ')':  goto err;
	case ',':  goto err;
	case ':':
	  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, ": @{COLON}@", CT_UNDEF);
	  pg_get_token();
	  break;
	case '*':
	  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "* @{ASTERISK}@", CT_UNDEF);
	  pg_get_token();
	  break;

	default:
	  v = pg_parse_expr(); break;
	}
	
	list = exprListAdd(list, v);

	if (pg_tok == ')'){
	    pg_get_token();
	    break;
	}

	if (pg_tok == ',') pg_get_token();
	else goto err;
    }

    return list;

  err:

    XMP_Error0("Syntax error in scripts of XMP directive");
    XMP_has_err = 1;

    return NULL;
}


CExpr *parse_XMP_align_source_list()
{
    CExpr* list;
    CExpr *v;

    list = EMPTY_LIST;

    if (pg_tok != '['){
	addFatal(NULL,"parse_XMP_align_source_list: first token != '['");
    }

    while (1){

	if (pg_tok != '[') break;
	pg_get_token();

	switch (pg_tok){
	case ']':  goto err;
	case ':':
	  v = NULL; break;
	case '*':
	  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "* @{ASTERISK}@", CT_UNDEF);
	  pg_get_token();
	  break;

	default:
	  v = pg_parse_expr(); break;
	}
	
	list = exprListAdd(list, v);
	if (pg_tok == ']') pg_get_token();
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
	    if (!width) v = XMP_PG_LIST(XMP_DIST_CYCLIC,width);
	    else v = XMP_PG_LIST(XMP_DIST_BLOCK_CYCLIC,width);
	} else if (PG_IS_IDENT("gblock")) {
	    pg_get_token();
	    if (pg_tok == '(') {
		pg_get_token();
		if (pg_tok == '*'){
		  width = NULL;
		  pg_get_token();
		}
		else
		  width = pg_parse_expr();
		if (pg_tok != ')') {
		    XMP_Error0("')' is needed after <mapping-array>");
		    goto err;
		} 
		pg_get_token();
	    }
	    v = XMP_PG_LIST(XMP_DIST_GBLOCK,width);
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

CExpr *parse_name_list()
{
    CExpr* list;
    
    list = EMPTY_LIST;
    if (pg_tok == '(') {
        pg_get_token();
	while(pg_tok == PG_IDENT){
	    list = exprListAdd(list, pg_tok_val);
	    pg_get_token();
	    if(pg_tok != ',') break;
	    pg_get_token();
	}
	if (pg_tok == ')'){
	  pg_get_token();
	  return list;
	}
    }

    XMP_Error0("name list is expected after :: ");
    return NULL;
}

CExpr *parse_name_list2()
{
    CExpr* list = EMPTY_LIST;

    while (pg_tok == PG_IDENT){
      list = exprListAdd(list, pg_tok_val);
      pg_get_token();
      if (pg_tok != ',') break;
      pg_get_token();
    }

    return list;
}

CExpr *parse_expr_list()
{
  CExpr *list = EMPTY_LIST;

  if (pg_tok == '(') {

    pg_get_token();

    while (1){
      list = exprListAdd(list, pg_parse_expr());
      //pg_get_token();
      if (pg_tok != ',') break;
      pg_get_token();
    }

    if (pg_tok == ')'){
      pg_get_token();
      return list;
    }
  }

  XMP_Error0("syntax error in expr list");
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
    subscript = parse_XMP_loop_subscript_list();
    return XMP_LIST2(ident,subscript);
}

CExpr *parse_task_ON_ref()
{
    CExpr *ident, *subscript;
    ident = NULL;

    if(pg_tok == PG_IDENT) {
	ident = pg_tok_val;
	pg_get_token();
    }

    if (pg_tok == '('){
      subscript = parse_XMP_subscript_list();
      return XMP_LIST2(ident,subscript);
    }
    else if (pg_tok == 0){
      return XMP_LIST2(ident, NULL);
    }
    else {
      XMP_Error0("syntax error in reference object by 'on'");
      XMP_has_err = 1;
      return NULL;
    }

/*     if(pg_tok != '('){ */
/* 	XMP_Error0("syntax error in reference object by 'on'"); */
/* 	XMP_has_err = 1; */
/* 	return NULL; */
/*     } */
/*     subscript = parse_XMP_subscript_list(); */
/*     return XMP_LIST2(ident,subscript); */
}

CExpr *parse_Reduction_opt()
{
    int op;
    CExpr *list;
    int loc_var_flag = 0;
    
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
	else if(PG_IS_IDENT("firstmax")){ op = XMP_DATA_REDUCE_FIRSTMAX; loc_var_flag = 1;}
	else if(PG_IS_IDENT("firstmin")){ op = XMP_DATA_REDUCE_FIRSTMIN; loc_var_flag = 1;}
	else if(PG_IS_IDENT("lastmax")){ op = XMP_DATA_REDUCE_LASTMAX; loc_var_flag = 1;}
	else if(PG_IS_IDENT("lastmin")){ op = XMP_DATA_REDUCE_LASTMIN; loc_var_flag = 1;}
	else goto unknown_err;
	break;
    default:
      unknown_err:
	XMP_Error0("unknown operation in reduction clause");
	goto ret;
    }
    pg_get_token();
    if(pg_tok != ':') goto err;
    list = EMPTY_LIST;

    do {

        pg_get_token();

	CExpr *specList = XMP_LIST1(pg_tok_val);
        CExpr *locVarList = EMPTY_LIST;

	pg_get_token();

	if (loc_var_flag){

	  if (pg_tok != '/'){
	    XMP_Error0("'/' is expected after <reduction-variable>");
	    return NULL;
	  }

          do {

            pg_get_token();

            if (pg_tok == PG_IDENT)
              locVarList = exprListAdd(locVarList, pg_tok_val);
	    /* else if (pg_tok() == '*'){  // Pointer Reference */
	    /*   pg_get_token(); */
	    /*   locationVariables.add(Xcons.String(pg_tok_buf())); */
	    /* } */
            else {
              XMP_Error0("syntax error on <location-variable>");
	      return NULL;
	    }

            pg_get_token();
            if (pg_tok == '/') break;
            else if (pg_tok == ',') continue;
            else {
              XMP_Error0("'/' or ',' is expected after <reduction-spec>");
	      return NULL;
	    }

          } while (1);

	  list = exprListAdd(list, exprListAdd(specList, locVarList));

          pg_get_token();
	  if (pg_tok != ',') break;

	  pg_get_token();
	}
	else
	  list = exprListAdd(list, exprListAdd(specList, locVarList));

	if (pg_tok == ')') break;
	else if (pg_tok == ',') continue;
	else
	  XMP_Error0("')' or ',' is expected after <reduction-spec>");
    } while (1);

    pg_get_token();
    return XMP_PG_LIST(op,list);

  err:
    XMP_Error0("syntax error in reduction clause");
  ret:
    return NULL;
}

CExpr *parse_Reduction_ref()
{
    int op;
    CExpr *list;
    
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
	/* else if(PG_IS_IDENT("firstmax")) op = XMP_DATA_REDUCE_FIRSTMAX; */
	/* else if(PG_IS_IDENT("firstmin")) op = XMP_DATA_REDUCE_FIRSTMIN; */
	/* else if(PG_IS_IDENT("lastmax")) op = XMP_DATA_REDUCE_LASTMAX; */
	/* else if(PG_IS_IDENT("lastmin")) op = XMP_DATA_REDUCE_LASTMIN; */
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
        list = exprListAdd(list, XMP_LIST2(pg_tok_val, EMPTY_LIST));
	pg_get_token();
	if(pg_tok != ',') break;
	pg_get_token();
    }
    if(pg_tok != ')') goto err;
    pg_get_token();
    return XMP_PG_LIST(op,list);
  err:
    XMP_Error0("syntax error in reduction");
  ret:
    return NULL;
}

CExpr *parse_XMP_opt()
{
  //return NULL;
  return (CExpr *)allocExprOfNull();
}

CExpr* parse_TASKS_clause()
{
    return (CExpr *)allocExprOfNull();
}

static CExpr* parse_REFLECT_clause()
{
    CExpr *arrayNameList = parse_name_list();
    CExpr *widthList = parse_WIDTH_list();
    CExpr *async = (CExpr *)allocExprOfNull();

    if (PG_IS_IDENT("async")){
      pg_get_token();
      if (pg_tok != '(') goto err;
      pg_get_token();
      async = pg_parse_expr();
      if (pg_tok != ')') goto err;
      pg_get_token();
    }

    CExpr *profileClause = (CExpr *)allocExprOfNull();
    /* if (pg_is_ident("profile")) { */
    /*     profileClause = Xcons.StringConstant("profile"); */
    /*     pg_get_token(); */
    /* } */

    return XMP_LIST4(arrayNameList, widthList, async, profileClause);

 err:
    XMP_Error0("syntax error in the REFLECT directive");
    XMP_has_err = 1;
    return NULL;
}


static CExpr* parse_REDUCTION_clause()
{
    CExpr* reductionRef = parse_Reduction_ref();
    CExpr* onRef = (CExpr *)allocExprOfNull();

    if (PG_IS_IDENT("on")) {
      pg_get_token();
      onRef = parse_task_ON_ref();
    }

    CExpr* profileClause = (CExpr *)allocExprOfNull();
    /* if (pg_is_ident("profile")) { */
    /* 	    profileClause = Xcons.StringConstant("profile"); */
    /* 	    pg_get_token(); */
    /* 	} */

    return XMP_LIST3(reductionRef, onRef, profileClause);
}

static CExpr* parse_BARRIER_clause()
{
    CExpr* onRef = (CExpr *)allocExprOfNull();
    if (PG_IS_IDENT("on")) {
      pg_get_token();
      onRef = parse_task_ON_ref();
    }

    CExpr* profileClause = (CExpr *)allocExprOfNull();
    /* if (pg_is_ident("profile")) { */
    /* 	profileClause = Xcons.StringConstant("profile"); */
    /* 	pg_get_token(); */
    /* } */

    return XMP_LIST2(onRef, profileClause);
}

static CExpr* parse_BCAST_clause()
{
    CExpr* varList = parse_name_list();

    CExpr* fromRef = (CExpr *)allocExprOfNull();;
    if (PG_IS_IDENT("from")) {
      pg_get_token();
      fromRef = parse_task_ON_ref();
    }
    else {
      fromRef = (CExpr *)allocExprOfNull();;
    }

    CExpr* onRef = (CExpr *)allocExprOfNull();;
    if (PG_IS_IDENT("on")) {
      pg_get_token();
      onRef = parse_task_ON_ref();
    }
    else onRef = (CExpr *)allocExprOfNull();;

    CExpr* profileClause = (CExpr *)allocExprOfNull();;
    /* if (PG_IS_IDENT("profile")) { */
    /*     profileClause = Xcons.StringConstant("profile"); */
    /*     pg_get_token(); */
    /* } */

    return XMP_LIST4(varList, fromRef, onRef, profileClause);
}

static CExpr* parse_GMOVE_clause()
{
    CExpr* gmoveClause = (CExpr *)allocExprOfNull();
    if (PG_IS_IDENT("in")) {
      gmoveClause = (CExpr*)allocExprOfNumberConst2(XMP_GMOVE_IN, BT_INT);
      pg_get_token();
    }
    else if (PG_IS_IDENT("out")) {
      gmoveClause = (CExpr*)allocExprOfNumberConst2(XMP_GMOVE_OUT, BT_INT);
      pg_get_token();
    }
    else gmoveClause = (CExpr*)allocExprOfNumberConst2(XMP_GMOVE_NORMAL, BT_INT);

    CExpr* profileClause = (CExpr *)allocExprOfNull();
    /* if (PG_IS_IDENT("profile")) { */
    /*   profileClause = Xcons.StringConstant("profile"); */
    /*   pg_get_token(); */
    /* } */

    return XMP_LIST2(gmoveClause, profileClause);
}

static CExpr* parse_COARRAY_clause()
{
    CExpr* coarrayNameList = parse_name_list2();

    if (pg_tok != ':') {
      XMP_Error0("':' is expected before <coarray-dimensions>");
    }

    int parsedLastDim = 0;
    CExpr* coarrayDims = EMPTY_LIST;

    pg_get_token();
    if (pg_tok != '[') {
      XMP_Error0("'[' is expected before <coarray-dim>");
    }

    pg_get_token();
    if (pg_tok == '*') {
      parsedLastDim = 1;
      coarrayDims = exprListAdd(coarrayDims, (CExpr *)allocExprOfNull());
      pg_get_token();
    } else {
      coarrayDims = exprListAdd(coarrayDims, pg_parse_expr());
    }

    if (pg_tok != ']') {
      XMP_Error0("']' is expected after <coarray-dim>");
    }
    
    while (1) {
      pg_get_token();
      if (pg_tok == '[') {
        if (parsedLastDim) {
          XMP_Error0("'*' in <coarray-dimension> is used in a wrong place");
        }

        pg_get_token();
        if (pg_tok == '*') {
          parsedLastDim = 1;
	  coarrayDims = exprListAdd(coarrayDims, (CExpr *)allocExprOfNull());
          pg_get_token();
        } else {
	  coarrayDims = exprListAdd(coarrayDims, pg_parse_expr());
        }

        if (pg_tok != ']') {
          XMP_Error0("']' is expected after <coarray-dim>");
        }
      } else {
        break;
      }
    }

    return XMP_LIST2(coarrayNameList, coarrayDims);
}

CExpr* parse_ARRAY_clause() {
    CExpr* onRef = NULL;
    CExpr* opt;

    if (PG_IS_IDENT("on")){
      pg_get_token();
	
      //onRef = parse_ON_ref();
      onRef = parse_task_ON_ref();
      opt = parse_XMP_opt();
    
      return XMP_LIST2(onRef,opt);
    }
    else {
      XMP_Error0("'on' is missing");
      goto err;
    }

  err:
    XMP_has_err = 1;
    return NULL;
}

static CExpr* parse_POST_clause()
{
    if (pg_tok != '(')
      XMP_Error0("'(' is expected before <nodes-name, tag>");

    pg_get_token();

    CExpr* onRef = parse_task_ON_ref();
    pg_get_token();
    CExpr* tag = pg_parse_expr();

    if (pg_tok != ')') {
      XMP_Error0("')' is expected after <nodes-name, tag>");
    }
    pg_get_token();

    return XMP_LIST2(onRef, tag);
}

static CExpr* parse_WAIT_clause()
{
    if (pg_tok != '('){
      return XMP_LIST1((CExpr*)allocExprOfNumberConst2(0, BT_INT));  // 0 is a number of args
    }
    else{
      pg_get_token(); 

      //CExpr* nodeName = parse_task_ON_ref();

      if (pg_tok != PG_IDENT){
	XMP_Error0("Syntax Error in WAIT");
      }

      CExpr* objName = pg_tok_val;

      pg_get_token();

      if (pg_tok != '(')
	XMP_Error0("Syntax Error in WAIT");

      CExpr* nodeNum = pg_parse_expr();
      
      CExpr* nodeName = XMP_LIST2(objName, nodeNum);

      if (pg_tok == ','){
	pg_get_token();
        CExpr* tag = pg_parse_expr();
	resolveType(tag);
        pg_get_token();
        return XMP_LIST3((CExpr*)allocExprOfNumberConst2(2, BT_INT), nodeName, tag);
      }
      else{  // if(pg_tok() == ')')
	pg_get_token();
	return XMP_LIST2((CExpr*)allocExprOfNumberConst2(1, BT_INT), nodeName);
      }
    }
}

static CExpr* parse_LOCAL_ALIAS_clause()
{
    return NULL;
}

static CExpr* parse_WIDTH_list()
{
  CExpr *list = EMPTY_LIST;
  CExpr *v1,*v2;
  int periodic_flag;

  if (!PG_IS_IDENT("width")) return NULL;

  pg_get_token();

  if (pg_tok == '(') {

    pg_get_token();

    while(1){

      v1 = v2 = NULL;
      periodic_flag = 0;

      if (pg_tok == '/'){
	pg_get_token();
	if (!PG_IS_IDENT("periodic")) goto err;

	pg_get_token();
	if (pg_tok != '/') goto err;

	periodic_flag = 1;
	pg_get_token();
      }

      switch (pg_tok){
      case ')':
      case ',':
      case ':':
	goto err;
      default:
	v1 = pg_parse_expr();
      }
	
      if (pg_tok != ':'){
	v2 = v1;
	goto next;
      }

      pg_get_token();

      switch (pg_tok){
      case ')':
      case ',':
      case ':':
	goto err;
      default:
	v2 = pg_parse_expr();
      }

    next:

      list = exprListAdd(list, XMP_LIST3(v1, v2, (CExpr*)allocExprOfNumberConst2(periodic_flag, BT_INT)));

      if (pg_tok == ')'){
	pg_get_token();
	break;
      }

      if (pg_tok == ',')  pg_get_token();
      else goto err;
    }

  }

  return list;

 err:

  XMP_Error0("syntax error in the WIDTH clause");
  XMP_has_err = 1;
  return NULL;

}


static CExpr* parse_WAIT_ASYNC_clause()
{
    CExpr *asyncIdList = parse_expr_list();
    return XMP_LIST1(asyncIdList);
}


static CExpr* parse_TEMPLATE_FIX_clause()
{

  CExpr *distFormatList = NULL;
  CExpr *templateNameList;
  CExpr *templateSpecList = NULL;

  // parse (<dist-format>, ...)
  if (pg_tok == '('){
    distFormatList = parse_XMP_dist_fmt_list();
  }

  // parse <template-name>
  if (pg_tok == PG_IDENT){
    templateNameList = XMP_LIST1(pg_tok_val);
    pg_get_token();
  }
  else {
    XMP_Error0("<template-name> is not optional.");
    goto err;
  }

  // parse (<template-spec>, ...)
  if (pg_tok == '('){
    templateSpecList = parse_XMP_range_list();
  }

  return XMP_LIST3(distFormatList, templateNameList, templateSpecList);

  err:
    XMP_has_err = 1;
    return NULL;
}

