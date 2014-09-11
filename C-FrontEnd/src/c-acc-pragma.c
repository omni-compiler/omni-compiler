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
#include "c-acc.h"

/*
 * <ACCPragma> <string> directive_name </string> 
 *             [clauses] [body] </OMPPragma>
 * [clauses] = <list> [clause] </list>
 *   C_Front: (direcive clause1 clause2 ... )
 *
 * [data_clause] = 
 *     <list> <string> [data_clause_name] </string> [name_list] </list>
 *     [data_clause_name] = ACC_DATA_PRIVATE|
 *              ACC_DATA_FIRSTPRIVATE|ACC_DATA_LASTPRIVATE|
 *		ACC_DATA_COPYIN|
 *               ACC_DATA_REDUCTION_***
 *     [name_list] = <list> variable ... </list>
 *  C_Front: (data_clause_name (LIST ident ... ))
 * 
 * [default_clause] = 
 *      <list> <string> ACC_DATA_DEFAULT </string> 
 *           <string> ACC_DEFAULT_*** </string> </list>
 *  C_Front: (ACC_DATA_DEFAULT (ACC_DEFAULT_*** null))
 * 
 * [if_clause] = <list> <string> ACC_DIR_IF </string> cond_expr </list>
 *  C_Front: (ACC_DIR_IF cond_expr)
 *
 * [schedule_clause] = 
 *       <list> <string> ACC_DIR_SCHEDULE </string>
 *           <list> <string> ACC_SCHED_*** </string> expr </list> </list>
 *  C_Front: (ACC_DIR_SCHEDULE (ACC_SCHED_*** expr))
 *
 * [ordered_clause] = <list> <string> ACC_DIR_ORDERED </strign> null </list>
 *  C_Front: (ACC_DIR_ORDERED null) 
 *
 * [nowait_clause] = <list> <string> ACC_DIR_NOWAIT </strign> null </list>
 *  C_Front: (ACC_DIR_NOWAIT null) 
 *
 * [num_threads_clause] = 
 *    <list> <string> ACC_DIR_NUM_THREADS </strign> expr </list>
 *  C_Front: (ACC_DIR_NUM_THREADS expr) 
 *
 */

static int parse_ACC_pragma(void);
static CExpr* parse_ACC_clauses(void);
static CExpr* parse_ACC_namelist(void);
static CExpr* parse_ACC_reduction_namelist(int *r);
static CExpr* parse_ACC_clause_arg(void);
static CExpr* parse_ACC_C_subscript_list(void);
static CExpr* parse_XACC_layout_clause_arg(void);
static CExpr* parse_XACC_shadow_clause_arg(void);
static CExpr* parse_XACC_device_arg(void);

#define ACC_PG_LIST(pg,args) _omp_pg_list(pg,args)
#define ACC_LIST2(arg1,arg2) (CExpr*)allocExprOfList2(EC_UNDEF,arg1,arg2)
#define ACC_LIST3(arg1,arg2,arg3) (CExpr*)allocExprOfList3(EC_UNDEF,arg1,arg2,arg3)

static CExpr* _omp_pg_list(int omp_code,CExpr* args)
{
  CExprOfList *lp;
  lp = allocExprOfList1(EC_UNDEF,args);
  lp->e_aux = omp_code;
  return (CExpr *)lp;
}

#define EMPTY_LIST (CExpr *)allocExprOfList(EC_UNDEF)

#ifdef not
static expv compile_ACC_SECTIONS_statement(expr x);
static void compile_ACC_pragma_clause(expr x, int pragma, int is_parallel, expv *pc, expv *dc);
static void compile_ACC_name_list _ANSI_ARGS_((expr x));
#endif

static int pg_ACC_pragma;
CExpr* pg_ACC_list;

extern int s_useXACC;

/*
 * for OpenACC directives
 */
CExpr*
lexParsePragmaACC(char *p, int *token) // p is buffer
{
  //skip pragma[space]omp[space]*
  p = lexSkipSpace(lexSkipWordP(lexSkipSpace(lexSkipWord(lexSkipSpace(lexSkipSharp(lexSkipSpace(p)))))));

  pg_cp = p; // set the pointer

  *token = parse_ACC_pragma();

  if(pg_ACC_list == NULL) pg_ACC_list = EMPTY_LIST;
  ((CExprOfList *)pg_ACC_list)->e_aux = pg_ACC_pragma;
  
  return pg_ACC_list;
}

int parse_ACC_pragma()
{
    int ret = PRAGMA_PREFIX; /* default */
    pg_ACC_list = NULL;

    pg_get_token();
    if(pg_tok != PG_IDENT) goto syntax_err;

    if(PG_IS_IDENT("parallel")){
	pg_get_token();
	if(pg_tok == PG_IDENT){
	    if(PG_IS_IDENT("loop")){	/* parallel for */
		pg_ACC_pragma = ACC_PARALLEL_LOOP;
		pg_get_token();
		if((pg_ACC_list = parse_ACC_clauses()) == NULL) 
		    goto syntax_err;
		goto chk_end;
	    }
	}
	pg_ACC_pragma = ACC_PARALLEL;
	if((pg_ACC_list = parse_ACC_clauses()) == NULL) goto syntax_err;
	goto chk_end;
    }
  
    if(PG_IS_IDENT("kernels")){
	pg_get_token();
	if(pg_tok == PG_IDENT){
	    if(PG_IS_IDENT("loop")){	/* parallel for */
		pg_ACC_pragma = ACC_KERNELS_LOOP;
		pg_get_token();
		if((pg_ACC_list = parse_ACC_clauses()) == NULL) 
		    goto syntax_err;
		goto chk_end;
	    }
	}
	pg_ACC_pragma = ACC_KERNELS;
	if((pg_ACC_list = parse_ACC_clauses()) == NULL) goto syntax_err;
	goto chk_end;
    }
  
    if(PG_IS_IDENT("loop")){
	pg_ACC_pragma = ACC_LOOP;
	pg_get_token();
	if((pg_ACC_list = parse_ACC_clauses()) == NULL) goto syntax_err;
	goto chk_end;
    }

    if(PG_IS_IDENT("data")){
	pg_ACC_pragma = ACC_DATA;
	pg_get_token();
	if((pg_ACC_list = parse_ACC_clauses()) == NULL) goto syntax_err;
	goto chk_end;
    }

    if(PG_IS_IDENT("host_data")){
	pg_ACC_pragma = ACC_HOST_DATA;
	pg_get_token();
	if((pg_ACC_list = parse_ACC_clauses()) == NULL)  goto syntax_err;
	goto chk_end;
    }

    if(PG_IS_IDENT("declare")){
	pg_ACC_pragma = ACC_DECLARE;
	pg_get_token();
	if((pg_ACC_list = parse_ACC_clauses()) == NULL)  goto syntax_err;
	ret= PRAGMA_EXEC;
	goto chk_end;
    }

    if(PG_IS_IDENT("update")){
	pg_ACC_pragma = ACC_UPDATE;
	pg_get_token();
	if((pg_ACC_list = parse_ACC_clauses()) == NULL)  goto syntax_err;
	ret= PRAGMA_EXEC;
	goto chk_end;
    }

    if(PG_IS_IDENT("cache")){
	pg_ACC_pragma = ACC_CACHE;
	pg_get_token();
	if(pg_tok == '('){
	    if((pg_ACC_list = parse_ACC_namelist()) == NULL) goto syntax_err;
	} else pg_ACC_list = NULL;
	ret= PRAGMA_EXEC;
	goto chk_end;
    }

    if(PG_IS_IDENT("wait")){
	pg_ACC_pragma = ACC_WAIT;
	pg_get_token();
	CExpr *x = NULL;

	if(pg_tok == '('){
	  if((x = parse_ACC_clause_arg()) == NULL)
	    goto syntax_err;
	  x = ACC_PG_LIST(ACC_WAIT_ARG, x);
	}
	//CExpr* pg_ACC_list_2;
	if((pg_ACC_list = parse_ACC_clauses()) == NULL){
	  pg_ACC_list = EMPTY_LIST;
	}
	if(x != NULL){
	  pg_ACC_list = exprListCons(x, pg_ACC_list);
	}

	ret= PRAGMA_EXEC;
	goto chk_end;
    }

    if(PG_IS_IDENT("enter")){
	pg_get_token();
	if(pg_tok == PG_IDENT){
	    if(PG_IS_IDENT("data")){	/* enter data */
		pg_ACC_pragma = ACC_ENTER_DATA;
		pg_get_token();
		if((pg_ACC_list = parse_ACC_clauses()) == NULL) goto syntax_err;
		ret= PRAGMA_EXEC;
		goto chk_end;
	    }
	}
    }

    if(PG_IS_IDENT("exit")){
	pg_get_token();
	if(pg_tok == PG_IDENT){
	    if(PG_IS_IDENT("data")){	/* enter data */
		pg_ACC_pragma = ACC_EXIT_DATA;
		pg_get_token();
		if((pg_ACC_list = parse_ACC_clauses()) == NULL) goto syntax_err;
		ret= PRAGMA_EXEC;
		goto chk_end;
	    }
	}
    }

    if (s_useXACC && PG_IS_IDENT("device")){
	pg_ACC_pragma = XACC_DEVICE;
	pg_get_token();
	if((pg_ACC_list = parse_XACC_device_arg()) == NULL)  goto syntax_err;
	ret= PRAGMA_EXEC;
	goto chk_end;
    }

    addError(NULL,"ACC: unknown ACC directive, '%s'",pg_tok_buf);
  syntax_err:
    return 0;

  chk_end:
    if(pg_tok != 0) addError(NULL,"ACC: extra arguments for ACC directive");
    return ret;
}

static CExpr* parse_ACC_clauses()
{
  CExpr *args,*v,*c;
  int r = 0;

  args = EMPTY_LIST;

  while(pg_tok == PG_IDENT){
      if(PG_IS_IDENT("private")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_PRIVATE,v);
      } else if(PG_IS_IDENT("firstprivate")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_FIRSTPRIVATE,v);
      } else if(PG_IS_IDENT("lastprivate")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_LASTPRIVATE,v);
      } else if(PG_IS_IDENT("copyin")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_COPYIN,v);
      } else if(PG_IS_IDENT("copyout")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_COPYOUT,v);
      } else if(PG_IS_IDENT("copy")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_COPY,v);
      } else if(PG_IS_IDENT("create")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_CREATE,v);
      } else if(PG_IS_IDENT("delete")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_DELETE,v);
      } else if(PG_IS_IDENT("present")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_PRESENT,v);
      } else if(PG_IS_IDENT("present_or_copy") || PG_IS_IDENT("pcopy")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_PRESENT_OR_COPY,v);
      } else if(PG_IS_IDENT("present_or_copyin") || PG_IS_IDENT("pcopyin")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_PRESENT_OR_COPYIN,v);
      } else if(PG_IS_IDENT("present_or_copyout") || PG_IS_IDENT("pcopyout")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_PRESENT_OR_COPYOUT,v);
      } else if(PG_IS_IDENT("present_or_create") || PG_IS_IDENT("pcreate")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_PRESENT_OR_CREATE,v);
      } else if(PG_IS_IDENT("deviceptr")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_DEVICEPTR,v);
      } else if(PG_IS_IDENT("host")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_HOST,v);
      } else if(PG_IS_IDENT("device")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_DEVICE,v);
      } else if(PG_IS_IDENT("use_device")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_USE_DEVICE,v);
      } else if(PG_IS_IDENT("device_resident")){
	  pg_get_token();
	  if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_DEV_RESIDENT,v);
      } else if(PG_IS_IDENT("reduction")){
	  pg_get_token();
	  if((v = parse_ACC_reduction_namelist(&r)) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(r,v);
      } else if(PG_IS_IDENT("if")){  // arg
	  pg_get_token();
	  if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_IF,v);
      } else if(PG_IS_IDENT("async")){  // arg
	  pg_get_token();
	  if(pg_tok != '(') v = NULL;
	    else if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(ACC_ASYNC,v);
      } else if(PG_IS_IDENT("wait")){  // arg
	  pg_get_token();
	  if(pg_tok != '(') v = NULL;
	  else if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_WAIT_ARG,v);
      } else if(PG_IS_IDENT("gang")){
	  pg_get_token();
	  if(pg_tok != '(') v = NULL;
	  else if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	  c = ACC_PG_LIST(ACC_GANG,v);
      } else if(PG_IS_IDENT("num_gangs")){
	    pg_get_token();
	    if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(ACC_NUM_GANGS,v);
	} else if(PG_IS_IDENT("worker")){
	    pg_get_token();
	    if(pg_tok != '(') v = NULL;
	    else if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(ACC_WORKER,v);
	} else if(PG_IS_IDENT("num_workers")){
	    pg_get_token();
	    if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(ACC_NUM_WORKERS,v);
	} else if(PG_IS_IDENT("vector")){
	    pg_get_token();
	    if(pg_tok != '(') v = NULL;
	    else if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(ACC_VECTOR,v);
	} else if(PG_IS_IDENT("vector_length")){
	    pg_get_token();
	    if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(ACC_VECT_LEN,v);
	} else if(PG_IS_IDENT("collapse")){
	    pg_get_token();
	    if((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(ACC_COLLAPSE,v);
	} else if(PG_IS_IDENT("seq")){
	    pg_get_token();
	    c = ACC_PG_LIST(ACC_SEQ,NULL);
	} else if(PG_IS_IDENT("independent")){
	    pg_get_token();
	    c = ACC_PG_LIST(ACC_INDEPENDENT,NULL);
	} else if (s_useXACC && PG_IS_IDENT("on_device")){
	    pg_get_token();
	    if ((v = parse_ACC_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(XACC_ON_DEVICE, v);
	} else if (s_useXACC && PG_IS_IDENT("layout")){
	    pg_get_token();
	    if ((v = parse_XACC_layout_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(XACC_LAYOUT, v);
	} else if (s_useXACC && PG_IS_IDENT("on")){
	    pg_get_token();
	    if((v = parse_ACC_namelist()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(XACC_ON, v);
	} else if (s_useXACC && PG_IS_IDENT("shadow")){
	    pg_get_token();
	    if((v = parse_XACC_shadow_clause_arg()) == NULL) goto syntax_err;
	    c = ACC_PG_LIST(XACC_SHADOW, v);
	} else {
	  addError(NULL,"unknown ACC directive clause '%s'",pg_tok_buf);
	    goto syntax_err;
	}
	args = exprListAdd(args,c);
    }
    return args;
 syntax_err:
    addError(NULL,"ACC: syntax error in ACC pragma clause");
    return NULL;
}

static CExpr* parse_ACC_namelist()
{
    CExpr* args;
    CExpr* v = NULL;
    CExpr* list = NULL;

    args = EMPTY_LIST;
    if(pg_tok != '(') {
      addError(NULL,"ACC: ACC directive clause requires name list");
	return NULL;
    }
    pg_get_token();
 next:
    if(pg_tok != PG_IDENT){
      addError(NULL,"ACC: empty name list in ACC directive clause");
	return NULL;
    }

    v = pg_tok_val;
    pg_get_token();

    if(pg_tok != '['){
      args = exprListAdd(args, v);
    }else{
      list = parse_ACC_C_subscript_list();
      CExpr* arrayRef = exprBinary(EC_ARRAY_REF, v, list);
      args = exprListAdd(args, (CExpr*)arrayRef);
    }

    if(pg_tok == ','){
	pg_get_token();
	goto next;
    } else if(pg_tok == ')'){
	pg_get_token();
	return args;
    } 

    addError(NULL,"ACC: syntax error in ACC pragma clause");
    return NULL;
}

static CExpr* parse_ACC_reduction_namelist(int *r)
{
  CExpr* args;

  args = EMPTY_LIST;
    if(pg_tok != '('){
      addError(NULL,"ACC reduction clause requires name list");
	return NULL;
    }
    pg_get_token();
    switch(pg_tok){
    case '+': *r = ACC_REDUCTION_PLUS; break;
    case '-': *r = ACC_REDUCTION_MINUS; break;
    case '*': *r = ACC_REDUCTION_MUL; break;
    case '&': *r = ACC_REDUCTION_BITAND; break;
    case '|': *r = ACC_REDUCTION_BITOR; break;
    case '^': *r = ACC_REDUCTION_BITXOR; break;
    case PG_ANDAND: *r = ACC_REDUCTION_LOGAND; break;
    case PG_OROR: *r = ACC_REDUCTION_LOGOR; break;
    case PG_IDENT:
	if(PG_IS_IDENT("max")) { *r = ACC_REDUCTION_MAX; break; }
	if(PG_IS_IDENT("min")) { *r = ACC_REDUCTION_MIN; break; }
    default:
	return NULL;	/* syntax error */
    }
    pg_get_token();
    if(pg_tok != ':') return NULL;
    pg_get_token();

 next:
    if(pg_tok != PG_IDENT){
      addError(NULL,"empty name list in ACC reduction clause");
	return NULL;
    }
    args = exprListAdd(args,pg_tok_val);
    pg_get_token();
    if(pg_tok == ','){
	pg_get_token();
	goto next;
    } else if(pg_tok == ')'){
	pg_get_token();
	return args;
    } 

    addError(NULL,"syntax error in ACC directive clause");
    return NULL;
}

static CExpr* parse_ACC_clause_arg()
{
    CExpr *v;

    if(pg_tok != '('){
      addError(NULL,"ACC reduction clause requires argument");
      return NULL;
    }
    pg_get_token();
    if((v = pg_parse_expr()) == NULL) goto syntax_err;
    if(pg_tok != ')') goto syntax_err;
    pg_get_token();
    return v;

  syntax_err:
    addError(NULL,"ACC: syntax error in ACC pragma clause");
    return NULL;
}

static CExpr* parse_ACC_C_subscript_list()
{
  CExpr* list;
  CExpr *v1,*v2;

  list = EMPTY_LIST;

  if(pg_tok != '[') {
    addError(NULL,"parse_ACC_C_subscript_list: first token= '['");
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
    case '*':
      if(! s_useXACC) goto err;
      v1 = (CExpr *)allocExprOfStringConst(EC_STRING_CONST, "* @{ASTERISK}@", CT_UNDEF);
      pg_get_token();
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
    list = exprListAdd(list, ACC_LIST2(v1,v2));

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
  addError(NULL, "Syntax error in scripts of ACC directive");
  return NULL;
}


CExpr *parse_XACC_layout_fmt_list()
{
    CExpr* list;
    CExpr *v;

    list = EMPTY_LIST;
    if(pg_tok != '(') {
	addFatal(NULL,"parse_ACC_dist_fmt_list: first token= '('");
    }
    
    pg_get_token();
    while(1){
	// parse <dist-format> := { * | block }
	if(pg_tok != '[') goto syntax_err;
	pg_get_token();

	if (pg_tok == '*') {
	    pg_get_token();
	    v = ACC_PG_LIST(XACC_LAYOUT_DUPLICATION,NULL);
	} else if (PG_IS_IDENT("block")) {
	    pg_get_token();
	    v = ACC_PG_LIST(XACC_LAYOUT_BLOCK,NULL);
	} else goto syntax_err;

	list = exprListAdd(list, v);

	if(pg_tok != ']') goto syntax_err;
	pg_get_token();

	if(pg_tok == ')'){
	    pg_get_token();
	    break;
	} else if(pg_tok == '['){
	    continue;
	} else goto syntax_err;
	
    }
    return list;

  syntax_err:
    addError(NULL, "syntax error in layout description");
    return NULL;
}

CExpr* parse_XACC_layout_clause_arg() 
{
    CExpr* layoutFormatList;

    // parse (<dist-format>, ...)
    if (pg_tok != '('){
	addError(NULL,"'(' is expected after 'layout' clause");
	goto err;
    } else {
	layoutFormatList = parse_XACC_layout_fmt_list();
    }

    return layoutFormatList;

  err:
    return NULL;
}

CExpr* parse_XACC_shadow_clause_arg() {

  CExpr* list;
  CExpr *v1,*v2;
  //  CExpr *type;
  int type;

  list = EMPTY_LIST;
  if(pg_tok != '(') {
    addFatal(NULL,"parse_XACC_shadow_width_list: first token= '('");
  }
  pg_get_token();

  while(1){
    v1 = v2 = NULL;
    //type = (CExpr*)allocExprOfNumberConst2(XACC_SHADOW_NORMAL, BT_INT);
    //type = (CExpr*)ACC_PG_LIST(XACC_SHADOW_NORMAL, NULL);
    type = XACC_SHADOW_NORMAL;

    if (pg_tok != '[') break;
    pg_get_token();

    switch(pg_tok){
    case ']':
    case ',':
    case ':':
      goto err;
    case '*':
      //type = (CExpr*)ACC_PG_LIST(XACC_SHADOW_FULL, NULL);
      type = XACC_SHADOW_FULL;
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
      //type = (CExpr*)ACC_PG_LIST(XACC_SHADOW_NONE, NULL);
      type = XACC_SHADOW_NONE;
    }
    //list = exprListAdd(list, ACC_LIST2(type, ACC_LIST2(v1,v2)));
    list = exprListAdd(list, ACC_PG_LIST(type, ACC_LIST2(v1,v2)));

    if(pg_tok == ']') pg_get_token();
    else goto err;
  }

  if(pg_tok != ')'){
    goto err;
  }
  pg_get_token();

  return list;

 err:
  addError(NULL,"Syntax error in scripts of XACC directive");
  return NULL;
}

static CExpr* parse_XACC_device_arg()
{
  CExpr* deviceName = NULL;
  CExpr* deviceRef;
  CExpr *ref_device = allocExprOfSymbol2("acc_device_default");

  //
  // parse <device-name>
  //

  if (pg_tok != PG_IDENT){
    addError(NULL, "'<device-name>' is expected");
    goto err;
  }

  deviceName = pg_tok_val;
  pg_get_token();

  //
  // parse =<device-ref>[subscript] }
  //

  if (pg_tok == '='){
    //specified ref_device
    pg_get_token();    

    // parse <device-ref>

    if (pg_tok != PG_IDENT) {
      addError(NULL, "'<device-ref>' is expected");
      goto err;
    }

    ref_device = pg_tok_val;
    pg_get_token();
  }

  // parse [subscript]

  if (pg_tok != '('){
    addError(NULL, "'(' is expected");
    goto err;
  }

  pg_get_token();

  CExpr *v1 = NULL, *v2 = NULL, *v3 = NULL;

  switch (pg_tok){
  case ')':
  case ',':
    goto err;
  case ':':
    break;
  case '*':
    pg_get_token();
    goto end;
  default:
    v1 = pg_parse_expr();
  }
	
  if (pg_tok != ':'){ // scalar
    v2 = v1;
    v3 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
    goto end;
  }

  pg_get_token();
  switch (pg_tok){
  case ')':
    goto err;
  case ':':
  default:
    v2 = pg_parse_expr();
  }

  if (pg_tok != ':'){
    v3 = (CExpr*)allocExprOfNumberConst2(1, BT_INT);
    goto end;
  }
  pg_get_token();
  v3 = pg_parse_expr(); 
	
 end:

  deviceRef = ACC_LIST2(ref_device, ACC_LIST3(v1,v2,v3));

  if (pg_tok != ')'){
    addError(NULL, "')' is expected");
    goto err;
  }

  pg_get_token();
  
  return ACC_LIST2(deviceName, deviceRef);

 err:
  return NULL;
}
