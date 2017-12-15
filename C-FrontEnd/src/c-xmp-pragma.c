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

extern char* lexSkipSharp(char *);

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
static CExpr* parse_STATIC_DESC_clause();
static CExpr* parse_TASK_clause();
static CExpr* parse_TASKS_clause();
static CExpr* parse_LOOP_clause();
static CExpr* parse_REFLECT_clause();
static CExpr* parse_REDUCTION_clause();
static CExpr* parse_BARRIER_clause();
static CExpr* parse_BCAST_clause();
static CExpr* parse_GMOVE_clause();
static CExpr* parse_COARRAY_clause();
static CExpr* parse_COARRAY_clause_p1();
static CExpr* parse_COARRAY_clause_p2();
static CExpr* parse_COARRAY_clause_p3();
static CExpr* parse_ARRAY_clause();
static CExpr* parse_POST_clause();
static CExpr* parse_WAIT_clause();
static CExpr* parse_LOCK_clause();
static CExpr* parse_UNLOCK_clause();
static CExpr* parse_WIDTH_list();
static CExpr* parse_ASYNC_clause();
static CExpr* parse_WAIT_ASYNC_clause();
static CExpr* parse_TEMPLATE_FIX_clause();
static CExpr* parse_REFLECT_INIT_clause();
static CExpr* parse_REFLECT_DO_clause();
static CExpr* parse_task_ON_ref();

static CExpr* parse_COL2_name_list();
static CExpr* parse_XMP_subscript_list_round();
static CExpr* parse_XMP_subscript_list_square();
static CExpr* parse_XMP_size_list_round();
static CExpr* parse_XMP_size_list_square();
static CExpr* parse_XMP_range_list_round();
static CExpr* parse_XMP_range_list_square();
static CExpr* parse_ON_ref();
static CExpr* parse_XMP_dist_fmt_list_round();
static CExpr* parse_XMP_dist_fmt_list_square();
static CExpr *parse_XMP_align_source_list(void);
static CExpr *parse_XMP_align_subscript_list_round(void);
static CExpr *parse_XMP_align_subscript_list_square(void);
static CExpr* parse_Reduction_opt();
static CExpr* parse_XMP_opt();
static CExpr* parse_ACC_or_HOST_clause();
static CExpr* parse_PROFILE_clause();
static void parse_ASYNC_ACC_or_HOST_PROFILE(CExpr**, CExpr**, CExpr**);
static CExpr *parse_XMP_loop_subscript_list_round();
static CExpr *parse_XMP_loop_subscript_list_square();
static CExpr *parse_XMP_shadow_width_list();
static CExpr* _xmp_pg_list(int omp_code,CExpr* args);

static int pg_XMP_pragma;
CExpr* pg_XMP_list;
int XMP_has_err = 0;

#define XMP_PG_LIST(pg,args)                     _xmp_pg_list(pg,args)
#define XMP_LIST1(arg1)                          (CExpr*)allocExprOfList1(EC_UNDEF,arg1)
#define XMP_LIST2(arg1,arg2)                     (CExpr*)allocExprOfList2(EC_UNDEF,arg1,arg2)
#define XMP_LIST3(arg1,arg2,arg3)                (CExpr*)allocExprOfList3(EC_UNDEF,arg1,arg2,arg3)
#define XMP_LIST4(arg1,arg2,arg3,arg4)           (CExpr*)allocExprOfList4(EC_UNDEF,arg1,arg2,arg3,arg4)
#define XMP_LIST5(arg1,arg2,arg3,arg4,arg5)      (CExpr*)allocExprOfList5(EC_UNDEF,arg1,arg2,arg3,arg4,arg5)
#define XMP_LIST6(arg1,arg2,arg3,arg4,arg5,arg6) (CExpr*)allocExprOfList6(EC_UNDEF,arg1,arg2,arg3,arg4,arg5,arg6)

#define XMP_Error0(msg)      addError(NULL,msg)
#define XMP_error1(msg,arg1) addError(NULL,msg,arg1)
#define EMPTY_LIST (CExpr *)allocExprOfList(EC_UNDEF)

static CExpr* _xmp_pg_list(int xmp_code,CExpr* args)
{
  CExprOfList *lp;
  lp = allocExprOfList1(EC_UNDEF,args);
  lp->e_aux = xmp_code;
  
  return (CExpr *)lp;
}

/*
 * for XcalableMP directives
 */
CExpr* lexParsePragmaXMP(char *p, int *token) // p is buffer
{
  //skip pragma[space]xmp[space]*
  p = lexSkipSpace(lexSkipWordP(lexSkipSpace(lexSkipWord(lexSkipSpace(lexSkipSharp(lexSkipSpace(p)))))));
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
  else if (PG_IS_IDENT("static_desc")) {
    pg_XMP_pragma = XMP_STATIC_DESC;
    pg_get_token();
    pg_XMP_list = parse_STATIC_DESC_clause();
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
  else if (PG_IS_IDENT("reduce_shadow")) {
    pg_XMP_pragma = XMP_REDUCE_SHADOW;
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
  else if (PG_IS_IDENT("lock")) {
    pg_XMP_pragma = XMP_LOCK;
    pg_get_token();
    pg_XMP_list = parse_LOCK_clause();
  }
  else if (PG_IS_IDENT("unlock")) {
    pg_XMP_pragma = XMP_UNLOCK;
    pg_get_token();
    pg_XMP_list = parse_UNLOCK_clause();
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
  }
  else if (PG_IS_IDENT("reflect_init")) {
    pg_XMP_pragma = XMP_REFLECT_INIT;
    pg_get_token();
    pg_XMP_list = parse_REFLECT_INIT_clause();
  }
  else if (PG_IS_IDENT("reflect_do")) {
    pg_XMP_pragma = XMP_REFLECT_DO;
    pg_get_token();
    pg_XMP_list = parse_REFLECT_DO_clause();
  }
  else {
    addError(NULL,"unknown XcalableMP directive, '%s'",pg_tok_buf);
  syntax_err:
    return 0;
  }

  if(pg_tok != 0) addError(NULL,"extra arguments for XMP directive");
  return ret;
}

CExpr* parse_NODES_clause()
{
  CExpr* nodesNameList = NULL;
  CExpr* nodesSizeList, *inheritedNodes;

  // parse <nodes-name>
  if (pg_tok == PG_IDENT) {
    nodesNameList = XMP_LIST1(pg_tok_val);
    pg_get_token();
  } 

  // parse (<nodes-size>, ...)
  if (pg_tok != '(' && pg_tok != '['){
    addError(NULL, "'(' or '[' is expected after <nodes-name>");
    goto err;
  }
  else if(pg_tok == '('){
    nodesSizeList = parse_XMP_size_list_round();
  }
  else{  // '['
    nodesSizeList = parse_XMP_size_list_square();
  }
  
  // parse { <empty> | =* | =<nodes-ref> }
  if (pg_tok == '=') {
    pg_get_token();
    if (pg_tok == '*') {
      pg_get_token();
      inheritedNodes = XMP_PG_LIST(XMP_NODES_INHERIT_EXEC,NULL);
    }
    else {
      inheritedNodes = XMP_PG_LIST(XMP_NODES_INHERIT_NODES, parse_task_ON_ref());
    } 
  }
  else 
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
  if (pg_tok != '(' && pg_tok != '[') {
    XMP_Error0("'(' or '[' is expected after <template-name>");
    goto err;
  }
  else if(pg_tok == '('){
    templateSpecList = parse_XMP_range_list_round();
  }
  else{ // '['
    templateSpecList = parse_XMP_range_list_square();
  }
  
  if(templateNameList == NULL) 
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
  if(pg_tok != '(' && pg_tok != '['){
    XMP_Error0("'(' or '[' is expected after <template-name>");
    goto err;
  }

  if(pg_tok == '(')
    distFormatList = parse_XMP_dist_fmt_list_round();
  else
    distFormatList = parse_XMP_dist_fmt_list_square();
    
  if(PG_IS_IDENT("onto")){
    pg_get_token();
  }
  else {
    XMP_Error0("onto is missing");
    goto err;
  }

  if (pg_tok == PG_IDENT){
    nodesName = pg_tok_val;
    pg_get_token();
  }
  else {
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
    }
  else 
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
  }
  else {
    XMP_Error0("<template-name> is expected after 'with'");
    goto err;
  }

  if(pg_tok != '(' && pg_tok != '[') {
    addFatal(NULL,"parse_XMP_align_subscript_list: first token= '(' or '['");
  }
  
  if(pg_tok == '(')
    alignSubscriptList = parse_XMP_align_subscript_list_round();
  else
    alignSubscriptList = parse_XMP_align_subscript_list_square();

  if (arrayNameList == NULL) 
    arrayNameList = parse_COL2_name_list();

  return XMP_LIST4(arrayNameList, alignSourceList, 
		   templateName, alignSubscriptList);
 err:
  XMP_has_err = 1;
  return NULL;
}

CExpr* parse_SHADOW_clause()
{
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
  }
  else 
      shadowWidthList = parse_XMP_shadow_width_list();
  
  if (arrayNameList == NULL) 
    arrayNameList = parse_COL2_name_list();
  
  return XMP_LIST2(arrayNameList, shadowWidthList);
  
 err:
  XMP_has_err = 1;
  return NULL;
}

CExpr* parse_STATIC_DESC_clause()
{
  CExpr* objectNameList = NULL;

  // parse <array-name>
  if (pg_tok == PG_IDENT) {
    objectNameList = XMP_LIST1(pg_tok_val);
    pg_get_token();
  } 
  
  if (objectNameList == NULL) 
    objectNameList = parse_COL2_name_list();
  
  if (objectNameList != NULL)
    return XMP_LIST1(objectNameList);
  else {
    XMP_has_err = 1;
    return NULL;
  }
}

CExpr* parse_TASK_clause()
{
  CExpr* onRef = NULL;
  CExpr* opt;
  
  if(PG_IS_IDENT("on"))
    pg_get_token();
  else {
    XMP_Error0("'on' is missing");
    goto err;
  }
  
  onRef = parse_task_ON_ref();

  int nocomm_flag = 0;
  if (PG_IS_IDENT("nocomm")){
    pg_get_token();
    nocomm_flag = 1;
  }
  
  opt = parse_XMP_opt();

  CExpr *async, *profile, *acc_or_host; // async and acc_or_host is not used
  parse_ASYNC_ACC_or_HOST_PROFILE(&async, &acc_or_host, &profile);
  
  return XMP_LIST4(onRef, (CExpr*)allocExprOfNumberConst2(nocomm_flag, BT_INT), opt, profile);
  
 err:
  XMP_has_err = 1;
  return NULL;
}

CExpr* parse_LOOP_clause()
{
  CExpr *subscriptList = NULL;
  CExpr *onRef, *reductionOpt, *opt;

  if(pg_tok == '(')
    subscriptList = parse_XMP_loop_subscript_list_round();
  else if(pg_tok == '[')
    subscriptList = parse_XMP_loop_subscript_list_square();

  if(PG_IS_IDENT("on"))
    pg_get_token();
  else{
    XMP_Error0("'on' is missing");
    goto err;
  }
    
  onRef = parse_ON_ref();
  CExpr *reduction_opt = parse_Reduction_opt();
  reductionOpt = reduction_opt ? XMP_LIST1(reduction_opt) : EMPTY_LIST;
  opt = parse_XMP_opt();

  CExpr *async, *profile, *acc_or_host; // async and acc_or_host is not used
  parse_ASYNC_ACC_or_HOST_PROFILE(&async, &acc_or_host, &profile);

  return XMP_LIST5(subscriptList,onRef,reductionOpt,opt,profile);

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

static CExpr *parse_XMP_align_subscript_list_round()
{
  CExpr *list_var  = EMPTY_LIST;
  CExpr *list_expr = EMPTY_LIST;
  CExpr *v, *var, *expr;
  
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

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "ROUND", CT_UNDEF);
  return exprListAdd(XMP_LIST2(list_var, list_expr), v);
  
 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

static CExpr *parse_XMP_align_subscript_list_square()
{
  CExpr *list_var  = EMPTY_LIST;
  CExpr *list_expr = EMPTY_LIST;
  CExpr *v, *var, *expr;

  pg_get_token();
  while(1){
    v = NULL;
    switch(pg_tok){
    case ']': goto err;
    case ':': break;
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
      var  = EXPR_B(v)->e_nodes[0];
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
    list_var  = exprListAdd(list_var, var);
    list_expr = exprListAdd(list_expr, expr);

    if (pg_tok == ']'){
      pg_get_token();
      if (pg_tok == '['){
	pg_get_token();
	continue;
      }
      else
	break;
    }
    else goto err;
  }

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "SQUARE", CT_UNDEF);
  return exprListAdd(XMP_LIST2(list_var, list_expr), v);

 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_subscript_list_round()
{
    CExpr* list = EMPTY_LIST;
    CExpr *v1,*v2,*v3;

    pg_get_token();
    while(1){
	v1 = v2 = v3 = NULL;
	switch(pg_tok){
	case ')':  goto err;
	case ',':  goto err;
	case ':':  break;
	case '*':
	  list = exprListAdd(list, NULL);
	  pg_get_token();
	  goto last;
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

      last:
	if(pg_tok == ')'){
	    pg_get_token();
	    break;
	}
	if(pg_tok == ',')  pg_get_token();
	else goto err;
    }


    CExpr *v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "ROUND", CT_UNDEF);
    return exprListAdd(list, v);

  err:
    XMP_Error0("Syntax error in scripts of XMP directive");
    XMP_has_err = 1;
    return NULL;
}

CExpr *parse_XMP_subscript_list_square()
{
  CExpr* list = EMPTY_LIST;
  CExpr *v1,*v2,*v3;

  pg_get_token();
  while(1){
    v1 = v2 = v3 = NULL;
    switch(pg_tok){
    case ']':  goto err;
    case '[':  goto err;
    case ':':  break;
    case '*':
      list = exprListAdd(list, NULL);
      pg_get_token();
      goto last;
    default:
      v1 = pg_parse_expr();
    }

    if(pg_tok != ':'){ // scalar
      v2 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
      v3 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
      goto next;
    }

    pg_get_token();
    switch(pg_tok){
    case ']': goto next;
    case ':': break;
    default:
      v2 = pg_parse_expr();
    }

    if(pg_tok != ':') goto next;
    pg_get_token();
    v3 = (pg_tok != ':')? pg_parse_expr() : (CExpr*)allocExprOfNumberConst2(1, BT_INT);

  next:
    if(v3 == NULL) v3 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
    list = exprListAdd(list, XMP_LIST3(v1,v2,v3));

  last:
    if(pg_tok != ']') goto err;
    
    pg_get_token();
    if(pg_tok == '[')  pg_get_token();
    else               break;
  }

  CExpr *v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "SQUARE", CT_UNDEF);
  return exprListAdd(list, v);

 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_size_list_round()
{
  CExpr *list = EMPTY_LIST;
  CExpr *v;

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

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "ROUND", CT_UNDEF);
  return exprListAdd(list, v);

 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_size_list_square()
{
  CExpr *list = EMPTY_LIST;
  CExpr *v;

  pg_get_token();
  while(1){
    v = NULL;
    switch(pg_tok){
    case ']':
    case ',':
    case ':':
      goto err;
    case '*':
      v = NULL;
      pg_get_token();
      break;
    default:
      v = pg_parse_expr();
    }

    list = exprListAdd(list, v);
    if(pg_tok == ']'){
      pg_get_token();
    }
    if(pg_tok == '[')  pg_get_token();
    else break;
  }

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "SQUARE", CT_UNDEF);
  return exprListAdd(list, v);

 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_range_list_round()
{
  CExpr* list = EMPTY_LIST;
  CExpr *v1,*v2;
  
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
    if(pg_tok == ':')
      goto err;
    else
      v2 = pg_parse_expr();
    
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

  CExpr *v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "ROUND", CT_UNDEF);
  return exprListAdd(list, v);

 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_range_list_square()
{
  CExpr* list = EMPTY_LIST;
  CExpr *v1,*v2;

  pg_get_token();
  while(1){
    v1 = v2 = NULL;
    switch(pg_tok){
    case ']': goto err;
    case '[': goto err;
    case ':':
      pg_get_token();
      goto next;
    default:
      v1 = pg_parse_expr();
    }

    if(pg_tok != ':'){
      v2 = v1;
      v1 = (CExpr*)allocExprOfNumberConst2(0, BT_INT);
      goto next;
    }

    // if pg_tok == ':'
    pg_get_token();
    if(pg_tok == ':')
      goto err;
    else
      v2 = pg_parse_expr();

  next:
    if (v1 == NULL && v2 == NULL)
      list = exprListAdd(list, NULL);
    else
      list = exprListAdd(list, XMP_LIST2(v1,v2));

    if(pg_tok == ']'){
      pg_get_token();
      if(pg_tok == '[')
	pg_get_token();
      else
	break;
    }
    else
      goto err;
  }

  CExpr *v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "SQUARE", CT_UNDEF);
  return exprListAdd(list, v);

 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_loop_subscript_list_round()
{
  CExpr* list = EMPTY_LIST;
  CExpr *v;

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

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "ROUND", CT_UNDEF);
  return exprListAdd(list, v);
   
 err:
  XMP_Error0("Syntax error in scripts of XMP directive");
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_loop_subscript_list_square()
{
  CExpr* list = EMPTY_LIST;
  CExpr *v;

  pg_get_token();
  while(1){
    switch (pg_tok){
    case ']':  goto err;
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
    
    if(pg_tok == ']'){
      pg_get_token();
      if(pg_tok == '['){
	pg_get_token();
	continue;
      }
      else
	break;
    }
    else goto err;
  }

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "SQUARE", CT_UNDEF);
  return exprListAdd(list, v);

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

#if isnot
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
#endif

CExpr *parse_XMP_dist_fmt_list_round()
{
  CExpr* list = EMPTY_LIST;
  CExpr *v, *width;

  pg_get_token();
  while(1){
    // parse <dist-format> := { * | block(n) | cyclic(n) }
    width = NULL;
    if (pg_tok == '*') {
      pg_get_token();
      v = XMP_PG_LIST(XMP_DIST_DUPLICATION,NULL);
    }
    else if (PG_IS_IDENT("block")) {
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
    }
    else if (PG_IS_IDENT("cyclic")) {
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
    }
    else if (PG_IS_IDENT("gblock")) {
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
    }
    else goto syntax_err;

    list = exprListAdd(list, v);
    
    if(pg_tok == ')'){
      pg_get_token();
      break;
    }
    else if(pg_tok == ','){
      pg_get_token();
      continue;
    }
    else goto syntax_err;
	
  }

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "ROUND", CT_UNDEF);
  return exprListAdd(list, v);

 syntax_err:
  XMP_Error0("syntax error in distribution description");
 err:
  XMP_has_err = 1;
  return NULL;
}

CExpr *parse_XMP_dist_fmt_list_square()
{
  CExpr* list = EMPTY_LIST;
  CExpr *v, *width;

  pg_get_token();
  while(1){
    // parse <dist-format> := { * | block(n) | cyclic(n) }
    width = NULL;
    if(pg_tok == '*'){
      pg_get_token();
      v = XMP_PG_LIST(XMP_DIST_DUPLICATION,NULL);
    }
    else if (PG_IS_IDENT("block")) {
      pg_get_token();
      if(pg_tok == '('){
	pg_get_token();
	width = pg_parse_expr();
	if(pg_tok != ')'){
	  XMP_Error0("')' is needed after <block-width>");
	  goto err;
	}
	pg_get_token();
      }
      v = XMP_PG_LIST(XMP_DIST_BLOCK,width);
    }
    else if (PG_IS_IDENT("cyclic")) {
      pg_get_token();
      if(pg_tok == '('){
	pg_get_token();
	width = pg_parse_expr();
	if(pg_tok != ')'){
	  XMP_Error0("')' is needed after <cyclic-width>");
	  goto err;
	}
	pg_get_token();
      }
      if (!width) v = XMP_PG_LIST(XMP_DIST_CYCLIC,width);
      else v = XMP_PG_LIST(XMP_DIST_BLOCK_CYCLIC,width);
    }
    else if (PG_IS_IDENT("gblock")) {
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
    }
    else
      goto syntax_err;

    list = exprListAdd(list, v);

    if(pg_tok == ']'){
      pg_get_token();
      if(pg_tok == '['){
	pg_get_token();
	continue;
      }
      else
	break;
    }
    else
      goto syntax_err;
  }

  v = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "SQUARE", CT_UNDEF);
  return exprListAdd(list, v);

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

static CExpr* parse_XMP_C_subscript_list()
{
  CExpr* list;
  CExpr *v1,*v2;

  list = EMPTY_LIST;

  if(pg_tok != '[') {
    addError(NULL,"parse_XMP_C_subscript_list: first token= '['");
  }
  pg_get_token();

  while(1){
    v1 = v2 = NULL;
    switch(pg_tok){
    case ']':  goto err;
    case ',':  goto err;
      break;
    case ':':
      v1 = (CExpr*)allocExprOfNumberConst2(0, BT_INT);
      break;
    default:
      v1 = pg_parse_expr();
    }

    if(pg_tok == ':') goto subarray;
    list = exprListAdd(list, v1);
    goto next;

  subarray:
    pg_get_token();
    if(pg_tok != ']'){
      v2 = pg_parse_expr();
    }
    list = exprListAdd(list, XMP_LIST2(v1,v2));

  next:
    if(pg_tok == ']'){
      pg_get_token();
    }else goto err;

    if(pg_tok != '['){
      break;
    }else{
      pg_get_token();
    }
  }

  return list;

 err:
  addError(NULL, "Syntax error in scripts of XMP directive");
  return NULL;
}

CExpr *parse_name_list()
{
    CExpr* list;
    
    list = EMPTY_LIST;
    if (pg_tok == '(') {
        pg_get_token();
	while(pg_tok == PG_IDENT){
	  CExpr *v = pg_tok_val;
	  pg_get_token();
	  if(pg_tok != '[')
	    list = exprListAdd(list, v);
	  else
	    list = exprListAdd(list, XMP_LIST2(v, parse_XMP_C_subscript_list()));

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

    if(pg_tok == '('){
      subscript = parse_XMP_loop_subscript_list_round();
    }
    else if(pg_tok == '['){
      subscript = parse_XMP_loop_subscript_list_square();
    }
    else{
      XMP_Error0("syntax error in reference object by 'on'");
      XMP_has_err = 1;
      return NULL;
    }

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
      subscript = parse_XMP_subscript_list_round();
      return XMP_LIST2(ident,subscript);
    }
    else if (pg_tok == '['){
      subscript = parse_XMP_subscript_list_square();
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
    case '^':
        op = XMP_DATA_REDUCE_BXOR; break;
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
    case '^':
        op = XMP_DATA_REDUCE_BXOR; break;
    case PG_ANDAND:
	op = XMP_DATA_REDUCE_LAND; break;
    case PG_OROR:
	op = XMP_DATA_REDUCE_LOR; break;
    case PG_IDENT:
	if(PG_IS_IDENT("max"))      op = XMP_DATA_REDUCE_MAX;
	else if(PG_IS_IDENT("min")) op = XMP_DATA_REDUCE_MIN;
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
      //      list = exprListAdd(list, pg_tok_val);
	pg_get_token();
	if(pg_tok == '/'){
	  if(op == XMP_DATA_REDUCE_MAX)      op = XMP_DATA_REDUCE_MAXLOC;
	  else if(op == XMP_DATA_REDUCE_MIN) op = XMP_DATA_REDUCE_MINLOC;
	  else goto err;
	  do{
	    pg_get_token();
	    CExpr *v = pg_tok_val;
	    pg_get_token();
	    if(pg_tok != '[')
	      list = exprListAdd(list, v);
	    else
	      list = exprListAdd(list, XMP_LIST2(v, parse_XMP_C_subscript_list()));
	  } while(pg_tok == ',');
	  if(pg_tok != '/') goto err;
	  pg_get_token();
	}
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

static CExpr *parse_XMP_opt()
{
  return (CExpr *)allocExprOfNull();
}

static CExpr *parse_TASKS_clause()
{
  return NULL;
}

static CExpr* parse_REFLECT_clause()
{
  CExpr *arrayNameList = parse_name_list();
  CExpr *widthList = parse_WIDTH_list();
  CExpr *async, *acc_or_host, *profile;
  
  parse_ASYNC_ACC_or_HOST_PROFILE(&async, &acc_or_host, &profile);
  return XMP_LIST5(arrayNameList, widthList, async, acc_or_host, profile);
}

static CExpr* parse_REDUCTION_clause()
{
  CExpr* reductionRef = parse_Reduction_ref();
  CExpr* onRef        = (CExpr *)allocExprOfNull();
  CExpr *async, *profile, *acc_or_host;
  
  if (PG_IS_IDENT("on")) {
    pg_get_token();
    onRef = parse_task_ON_ref();
  }
  
  parse_ASYNC_ACC_or_HOST_PROFILE(&async, &acc_or_host, &profile);
  
  return XMP_LIST5(reductionRef, onRef, async, acc_or_host, profile);
}

static CExpr* parse_BARRIER_clause()
{
  CExpr* onRef = (CExpr *)allocExprOfNull();
  if (PG_IS_IDENT("on")) {
    pg_get_token();
    onRef = parse_task_ON_ref();
  }

  CExpr *async, *profile, *acc_or_host; // async and acc_or_host is not used
  parse_ASYNC_ACC_or_HOST_PROFILE(&async, &acc_or_host, &profile);
  
  return XMP_LIST2(onRef, profile);
}

static CExpr* parse_BCAST_clause()
{
  CExpr* varList = parse_name_list();
  CExpr *async, *acc_or_host, *profile;
  
  CExpr* fromRef = (CExpr *)allocExprOfNull();
  if (PG_IS_IDENT("from")) {
    pg_get_token();
    fromRef = parse_task_ON_ref();
  }
  else {
    fromRef = (CExpr *)allocExprOfNull();
  }
  
  CExpr* onRef = (CExpr *)allocExprOfNull();
  if (PG_IS_IDENT("on")) {
    pg_get_token();
    onRef = parse_task_ON_ref();
  }
  else{
    onRef = (CExpr *)allocExprOfNull();
  }
  
  parse_ASYNC_ACC_or_HOST_PROFILE(&async, &acc_or_host, &profile);
  
  return XMP_LIST6(varList, fromRef, onRef, async, acc_or_host, profile);
}

static CExpr* parse_GMOVE_clause()
{
  CExpr* gmoveClause = (CExpr *)allocExprOfNull();
  CExpr *async, *acc_or_host, *profile;
  
  if (PG_IS_IDENT("in")) {
    gmoveClause = (CExpr*)allocExprOfNumberConst2(XMP_GMOVE_IN, BT_INT);
    pg_get_token();
  }
  else if (PG_IS_IDENT("out")) {
    gmoveClause = (CExpr*)allocExprOfNumberConst2(XMP_GMOVE_OUT, BT_INT);
    pg_get_token();
  }
  else gmoveClause = (CExpr*)allocExprOfNumberConst2(XMP_GMOVE_NORMAL, BT_INT);
  
  parse_ASYNC_ACC_or_HOST_PROFILE(&async, &acc_or_host, &profile);  

  return XMP_LIST4(gmoveClause, async, acc_or_host, profile);
}

static CExpr* parse_COARRAY_clause()
{
    /*-- Version 1.2 specification + alpha --*/
    if (PG_IS_IDENT("on")) {  
        CExpr* nodesNameList = parse_COARRAY_clause_p1();
        CExpr* coarrayNameList = parse_COARRAY_clause_p2();
        CExpr* coarrayDims;
        if (pg_tok == ':')
            coarrayDims = parse_COARRAY_clause_p3();
        else
            coarrayDims = EMPTY_LIST;
        return XMP_LIST3(coarrayNameList, coarrayDims, nodesNameList);
    }

    /*-- Else, Version 1.0 specification --*/
    CExpr* coarrayNameList = parse_COARRAY_clause_p2();
    CExpr* coarrayDims = parse_COARRAY_clause_p3();
    return XMP_LIST2(coarrayNameList, coarrayDims);
}

static CExpr* parse_COARRAY_clause_p1()
{
    CExpr* nodesNameList = EMPTY_LIST;

    if (!PG_IS_IDENT("on"))
        return nodesNameList;

    pg_get_token();        // skip "on"

    nodesNameList = exprListAdd(nodesNameList, pg_tok_val);   // set nodes name
    pg_get_token();        // pass nodes name

    if (pg_tok == PG_COL2)
        pg_get_token();    // skip "::"
    else
        XMP_Error0("'::' is expected after <nodes-name>");

    return nodesNameList;
}

static CExpr* parse_COARRAY_clause_p2()
{
    CExpr* coarrayNameList = parse_name_list2();
    return coarrayNameList;
}

static CExpr* parse_COARRAY_clause_p3()
{
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

    return coarrayDims;
}

CExpr* parse_ARRAY_clause() {
    CExpr* onRef = NULL;
    CExpr* opt;

    if (PG_IS_IDENT("on")){
      pg_get_token();
	
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
  if(pg_tok != '(')
    return NULL;  // no argument

  pg_get_token(); 
  CExpr* nodeNum = parse_ON_ref();
  
  // only node
  if(pg_tok == ')'){
    pg_get_token();
    return XMP_LIST1(nodeNum);
  }

  // node and tag
  pg_get_token();
  CExpr* tag = pg_parse_expr();

  if(pg_tok != ')')
    XMP_Error0("')' is expected after <nodes-name, tag>");

  pg_get_token();
  return XMP_LIST2(nodeNum, tag);
}

static CExpr* parse_LOCK_clause()
{
  if(pg_tok != '(')
    XMP_Error0("'(' is expected before <coarray>");
  else
    pg_get_token();

  CExpr* coarrayRef = EMPTY_LIST;
  coarrayRef = exprListAdd(coarrayRef, pg_tok_val);
  pg_get_token();

  if(pg_tok == ')'){ // Without coarray image (e.g. #pragma xmp lock (lockobj))
    pg_get_token();
    return XMP_LIST1(coarrayRef);
  }

  if(pg_tok != ':'){
    if(pg_tok != '['){
      XMP_Error0("'[' is expected before <coarray-dim>");
    }
    else{
      pg_get_token();
      coarrayRef = exprListAdd(coarrayRef, pg_parse_expr());
      if(pg_tok != ']')
	XMP_Error0("']' is expected after <coarray-dim>");
      
      while(1){
	pg_get_token();
	if(pg_tok == '['){
	  pg_get_token();
	  coarrayRef = exprListAdd(coarrayRef, pg_parse_expr());
	  if(pg_tok != ']')
	    XMP_Error0("']' is expected after <coarray-dim>");
	}
	else
	  break;
      }
    }
  }
  
  if(pg_tok == ')'){ // Without coarray image (e.g. #pragma xmp lock (lockobj[c]))
    pg_get_token();
    return XMP_LIST1(coarrayRef);
  }
  
  CExpr* coarrayDim = EMPTY_LIST;
  pg_get_token(); // skip ':'
  if(pg_tok != '[')
    XMP_Error0("'[' is expected before <coarray-dim>");
  else
    pg_get_token();
  coarrayDim = exprListAdd(coarrayDim, pg_parse_expr());
  if(pg_tok != ']')
    XMP_Error0("']' is expected after <coarray-dim>");

  while(1){
    pg_get_token();
    if(pg_tok == '['){
      pg_get_token();
      coarrayDim = exprListAdd(coarrayDim, pg_parse_expr());
      if(pg_tok != ']')
	XMP_Error0("']' is expected after <coarray-dim>");
    }
    else
      break;
  }
  
  if(pg_tok != ')')
    XMP_Error0("')' is expected after <coarray>");
  else
    pg_get_token();

  return XMP_LIST2(coarrayRef, coarrayDim);
}

static CExpr* parse_UNLOCK_clause()
{
  return parse_LOCK_clause();
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

static CExpr* parse_ASYNC_clause()
{
  if (PG_IS_IDENT("async")){
    pg_get_token();
    if (pg_tok != '(') goto err;
    pg_get_token();
    CExpr* async = pg_parse_expr();
    if (pg_tok != ')') goto err;
    pg_get_token();
    
    return async;
  }
  else{
    return NULL;
  }

 err:
    XMP_Error0("syntax error in an ASYNC clause");
    XMP_has_err = 1;
    return NULL;
}

static CExpr* parse_WAIT_ASYNC_clause()
{
    CExpr *asyncIdList = parse_expr_list();
    CExpr* onRef = NULL;
    if (PG_IS_IDENT("on")){
      pg_get_token();
      onRef = parse_task_ON_ref();
    }
    return XMP_LIST2(asyncIdList, onRef);
}

static CExpr* parse_TEMPLATE_FIX_clause()
{
  CExpr *distFormatList = NULL;
  CExpr *templateNameList;
  CExpr *templateSpecList = NULL;

  // parse (<dist-format>, ...)
  if(pg_tok == '('){
    distFormatList = parse_XMP_dist_fmt_list_round();
  }
  else if(pg_tok == '['){
    distFormatList = parse_XMP_dist_fmt_list_square();
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
  if(pg_tok == '('){
    templateSpecList = parse_XMP_range_list_round();
  }
  else if(pg_tok == '['){
    templateSpecList = parse_XMP_range_list_square();
  }

  return XMP_LIST3(distFormatList, templateNameList, templateSpecList);

  err:
    XMP_has_err = 1;
    return NULL;
}

static CExpr* parse_REFLECT_INIT_clause()
{
  if (pg_tok != '('){
    XMP_Error0(" #pragma xmp reflect_init (array-name) [width (reflect-width)] [host|acc].");
    XMP_has_err = 1;
    return NULL;
  }

  CExpr *arrayNameList = parse_name_list();
  CExpr *widthList = parse_WIDTH_list();
  CExpr *acc_or_host1 = (CExpr *)allocExprOfNull();
  CExpr *acc_or_host2 = (CExpr *)allocExprOfNull();

  if(PG_IS_IDENT("acc") || PG_IS_IDENT("host")){
    acc_or_host1 = XMP_LIST1(pg_tok_val);
    pg_get_token();
  }

  if(PG_IS_IDENT("acc") || PG_IS_IDENT("host")){
    acc_or_host2 = XMP_LIST1(pg_tok_val);
    pg_get_token();
  }
  
  return XMP_LIST4(arrayNameList, widthList, acc_or_host1, acc_or_host2);
}

static CExpr* parse_REFLECT_DO_clause()
{
  if (pg_tok != '('){
    XMP_Error0(" #pragma xmp reflect_do (array-name) [host|acc].");
    XMP_has_err = 1;
    return NULL;
  }

  CExpr *arrayNameList = parse_name_list();
  CExpr *acc_or_host1 = (CExpr *)allocExprOfNull();
  CExpr *acc_or_host2 = (CExpr *)allocExprOfNull();

  if(PG_IS_IDENT("acc") || PG_IS_IDENT("host")){
    acc_or_host1 = XMP_LIST1(pg_tok_val);
    pg_get_token();
  }

  if(PG_IS_IDENT("acc") || PG_IS_IDENT("host")){
    acc_or_host2 = XMP_LIST1(pg_tok_val);
    pg_get_token();
  }

  return XMP_LIST3(arrayNameList, acc_or_host1, acc_or_host2);
}

 static CExpr* parse_ACC_or_HOST_clause()
 {
   CExpr* acc_or_host = EMPTY_LIST;

   if(PG_IS_IDENT("acc") || PG_IS_IDENT("host")){
     acc_or_host = exprListAdd(acc_or_host, pg_tok_val);
     pg_get_token();
   }
   if(PG_IS_IDENT("acc") || PG_IS_IDENT("host")){
     acc_or_host = exprListAdd(acc_or_host, pg_tok_val);
     pg_get_token();
   }

   if(acc_or_host != EMPTY_LIST)
     return acc_or_host;
   else
     return NULL;
 }

static CExpr* parse_PROFILE_clause()
{
  if(PG_IS_IDENT("profile")){
    CExpr* profile = pg_tok_val;
    pg_get_token();
    return profile;
  }
  else{
    return NULL;
  }
}

static void parse_ASYNC_ACC_or_HOST_PROFILE(CExpr** async, CExpr** acc_or_host, CExpr** profile)
{
  *async = *acc_or_host = *profile = NULL;

  while(pg_tok != 0){ // Check until the end of clause
    if(*async       == NULL){
      if((*async       = parse_ASYNC_clause())       != NULL) continue;
    }
    if(*acc_or_host == NULL){
      if((*acc_or_host = parse_ACC_or_HOST_clause()) != NULL) continue;
    }
    if(*profile     == NULL){
      if((*profile     = parse_PROFILE_clause())     != NULL) continue;
    }

    break;
  }

  if(*async == NULL)       *async       = (CExpr *)allocExprOfNull();
  if(*acc_or_host == NULL) *acc_or_host = (CExpr *)allocExprOfNull();
  if(*profile == NULL)     *profile     = (CExpr *)allocExprOfNull();
}
