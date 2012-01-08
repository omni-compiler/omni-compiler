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
#include "c-omp.h"

static int parse_OMP_pragma(void);
static CExpr* parse_OMP_clause(void);
static CExpr* parse_OMP_namelist(void);
static CExpr* parse_OMP_reduction_namelist(int *r);

#define OMP_PG_LIST(pg,args) _omp_pg_list(pg,args)

static CExpr* _omp_pg_list(int omp_code,CExpr* args)
{
  CExprOfList *lp;
  lp = allocExprOfList1(EC_UNDEF,args);
  lp->e_aux = omp_code;
  return (CExpr *)lp;
}

#define EMPTY_LIST (CExpr *)allocExprOfList(EC_UNDEF)

#ifdef not
static expv compile_OMP_SECTIONS_statement(expr x);
static void compile_OMP_pragma_clause(expr x, int pragma, int is_parallel, expv *pc, expv *dc);
static void compile_OMP_name_list _ANSI_ARGS_((expr x));
#endif

static int pg_OMP_pragma;
CExpr* pg_OMP_list;

/*
 * for OpenMP directives
 */
CExpr*
lexParsePragmaOMP(char *p, int *token) // p is buffer
{
  //skip pragma[space]omp[space]*
  p = lexSkipSpace(lexSkipWordP(lexSkipSpace(lexSkipWord(lexSkipSpace(p)))));

  pg_cp = p; // set the pointer

  *token = parse_OMP_pragma();

  if(pg_OMP_list == NULL) pg_OMP_list = EMPTY_LIST;
  ((CExprOfList *)pg_OMP_list)->e_aux = pg_OMP_pragma;
  
  return pg_OMP_list;
}

int parse_OMP_pragma()
{
  int ret = PRAGMA_PREFIX; /* default */
  pg_OMP_list = NULL;

  pg_get_token();
  if(pg_tok != PG_IDENT) goto syntax_err;

  /* parallel block directive */
  if(PG_IS_IDENT("parallel")){
	pg_get_token();
	if(pg_tok == PG_IDENT){
	  if(PG_IS_IDENT("for")){	/* parallel for */
	    pg_OMP_pragma = OMP_PARALLEL_FOR;
	    pg_get_token();
	    if((pg_OMP_list = parse_OMP_clause()) == NULL) goto syntax_err;
	    goto chk_end;
	  }
	  if(PG_IS_IDENT("sections")){	/* parallel for */
	    pg_OMP_pragma = OMP_PARALLEL_SECTIONS;
	    pg_get_token();
	    if((pg_OMP_list = parse_OMP_clause()) == NULL) goto syntax_err;
	    goto chk_end;
	  }
	}
	pg_OMP_pragma = OMP_PARALLEL;
	if((pg_OMP_list = parse_OMP_clause()) == NULL) goto syntax_err;
	goto chk_end;
  }
  
  if(PG_IS_IDENT("for")){
    pg_OMP_pragma = OMP_FOR;
    pg_get_token();
    if((pg_OMP_list = parse_OMP_clause()) == NULL) goto syntax_err;
    goto chk_end;
  }

  if(PG_IS_IDENT("sections")){
    pg_OMP_pragma = OMP_SECTIONS;
    pg_get_token();
    if((pg_OMP_list = parse_OMP_clause()) == NULL) goto syntax_err;
    goto chk_end;
  }

  if(PG_IS_IDENT("single")){
    pg_OMP_pragma = OMP_SINGLE;
    pg_get_token();
    if((pg_OMP_list = parse_OMP_clause()) == NULL)  goto syntax_err;
    goto chk_end;
  }

  if(PG_IS_IDENT("master")){
    pg_OMP_pragma = OMP_MASTER;
    pg_get_token();
    goto chk_end;
  }

  if(PG_IS_IDENT("critical")){
    pg_OMP_pragma = OMP_CRITICAL;
    pg_get_token();
    if(pg_tok == '('){
      if((pg_OMP_list = parse_OMP_namelist()) == NULL) goto syntax_err;
    } else pg_OMP_list = NULL;
    goto chk_end;
  }

  if(PG_IS_IDENT("ordered")){
    pg_OMP_pragma = OMP_ORDERED;
    pg_get_token();
    goto chk_end;
  }

  if(PG_IS_IDENT("section")){
    pg_OMP_pragma = OMP_SECTION;
    pg_get_token();
    ret = PRAGMA_EXEC;
    goto chk_end;
  }

  if(PG_IS_IDENT("barrier")){
    pg_OMP_pragma = OMP_BARRIER;
    ret = PRAGMA_EXEC;
    pg_get_token();
    goto chk_end;
  }
  
    if(PG_IS_IDENT("atomic")){
	pg_OMP_pragma = OMP_ATOMIC;
	ret = PRAGMA_PREFIX;
	pg_get_token();
	goto chk_end;
    }

    if(PG_IS_IDENT("flush")){
	pg_OMP_pragma = OMP_FLUSH;
	pg_get_token();
	if(pg_tok == '('){
	    if((pg_OMP_list = parse_OMP_namelist()) == NULL) goto syntax_err;
	} else pg_OMP_list = NULL;
	ret= PRAGMA_EXEC;
	goto chk_end;
    }

    if(PG_IS_IDENT("threadprivate")){
	pg_OMP_pragma = OMP_THREADPRIVATE;
	pg_get_token();
	if((pg_OMP_list = parse_OMP_namelist()) == NULL) goto syntax_err;
	ret = PRAGMA_EXEC;
	goto chk_end;
    }
    addError(NULL,"OMP:unknown OMP directive, '%s'",pg_tok_buf);
 syntax_err:
    return 0;

 chk_end:
    if(pg_tok != 0) addError(NULL,"OMP:extra arguments for OMP directive");
    return ret;
}

static CExpr* parse_OMP_clause()
{
  CExpr *args,*v,*c;
  int r = 0;

    args = EMPTY_LIST;

    while(pg_tok == PG_IDENT){
	if(PG_IS_IDENT("private")){
	    pg_get_token();
	    if((v = parse_OMP_namelist()) == NULL) goto syntax_err;
	    c = OMP_PG_LIST(OMP_DATA_PRIVATE,v);
	} else if(PG_IS_IDENT("shared")){
	    pg_get_token();
	    if((v = parse_OMP_namelist()) == NULL) goto syntax_err;
	    c = OMP_PG_LIST(OMP_DATA_SHARED,v);
	} else if(PG_IS_IDENT("firstprivate")){
	    pg_get_token();
	    if((v = parse_OMP_namelist()) == NULL) goto syntax_err;
	    c = OMP_PG_LIST(OMP_DATA_FIRSTPRIVATE,v);
	} else if(PG_IS_IDENT("lastprivate")){
	    pg_get_token();
	    if((v = parse_OMP_namelist()) == NULL) goto syntax_err;
	    c = OMP_PG_LIST(OMP_DATA_LASTPRIVATE,v);
	} else if(PG_IS_IDENT("copyin")){
	    pg_get_token();
	    if((v = parse_OMP_namelist()) == NULL) goto syntax_err;
	    c = OMP_PG_LIST(OMP_DATA_COPYIN,v);
	} else if(PG_IS_IDENT("reduction")){
	    pg_get_token();
	    if((v = parse_OMP_reduction_namelist(&r)) == NULL) goto syntax_err;
	    c = OMP_PG_LIST(r,v);
	} else if(PG_IS_IDENT("default")){
	    pg_get_token();
	    if(pg_tok != '(') goto syntax_err;
	    pg_get_token();
	    if(pg_tok != PG_IDENT) goto syntax_err;
	    if(PG_IS_IDENT("shared")) 
	      r = OMP_DEFAULT_SHARED;
	    else if(PG_IS_IDENT("none"))
	      r = OMP_DEFAULT_NONE;
	    else goto syntax_err;
	    pg_get_token();
	    if(pg_tok != ')') goto syntax_err;
	    pg_get_token();
	    v = OMP_PG_LIST(r,EMPTY_LIST);
	    c = OMP_PG_LIST(OMP_DATA_DEFAULT,v);
	} else if(PG_IS_IDENT("if")){
	    pg_get_token();
	    if(pg_tok != '(') goto syntax_err;
	    pg_get_token();
	    if((v = pg_parse_expr()) == NULL) goto syntax_err;
	    if(pg_tok != ')') goto syntax_err;
	    pg_get_token();
	    c = OMP_PG_LIST(OMP_DIR_IF,v);
	} else if(PG_IS_IDENT("schedule")){
	    pg_get_token();
	    if(pg_tok != '(') goto syntax_err;
	    pg_get_token();
	    if(pg_tok != PG_IDENT) goto syntax_err;
	    if(PG_IS_IDENT("static")) r = (int)OMP_SCHED_STATIC;
	    else if(PG_IS_IDENT("dynamic")) r = (int)OMP_SCHED_DYNAMIC;
	    else if(PG_IS_IDENT("guided")) r = (int)OMP_SCHED_GUIDED;
	    else if(PG_IS_IDENT("runtime")) r = (int)OMP_SCHED_RUNTIME;
	    else if(PG_IS_IDENT("affinity")) r = (int)OMP_SCHED_AFFINITY;
	    else goto syntax_err;
	    pg_get_token();

	    if(pg_tok == ','){
	      pg_get_token();
	      if((v = pg_parse_expr()) == NULL) goto syntax_err;
	      v = OMP_PG_LIST(r,v);
	    } else v = OMP_PG_LIST(r,NULL);

	    if(pg_tok != ')') goto syntax_err;
	    pg_get_token();
	    c = OMP_PG_LIST(OMP_DIR_SCHEDULE,v);
	} else if(PG_IS_IDENT("ordered")){
	    pg_get_token();
	    c = OMP_PG_LIST(OMP_DIR_ORDERED,NULL);
	} else if(PG_IS_IDENT("nowait")){
	    pg_get_token();
	    c = OMP_PG_LIST(OMP_DIR_NOWAIT,NULL);
	} else {
	  addError(NULL,"unknown OMP directive clause '%s'",pg_tok_buf);
	    goto syntax_err;
	}
	args = exprListAdd(args,c);
    }
    return args;
 syntax_err:
    addError(NULL,"OMP: syntax error in OMP pragma clause");
    return NULL;
}

static CExpr* parse_OMP_namelist()
{
    CExpr* args;

    args = EMPTY_LIST;
    if(pg_tok != '(') {
      addError(NULL,"OMP: OMP directive clause requires name list");
	return NULL;
    }
    pg_get_token();
 next:
    if(pg_tok != PG_IDENT){
      addError(NULL,"OMP: empty name list in OMP directive clause");
	return NULL;
    }
    args = exprListAdd(args, pg_tok_val);
    pg_get_token();
    if(pg_tok == ','){
	pg_get_token();
	goto next;
    } else if(pg_tok == ')'){
	pg_get_token();
	return args;
    } 

    addError(NULL,"OMP: syntax error in OMP pragma clause");
    return NULL;
}

static CExpr* parse_OMP_reduction_namelist(int *r)
{
  CExpr* args;

  args = EMPTY_LIST;
    if(pg_tok != '('){
      addError(NULL,"OMP reduction clause requires name list");
	return NULL;
    }
    pg_get_token();
    switch(pg_tok){
    case '+': *r = OMP_DATA_REDUCTION_PLUS; break;
    case '-': *r = OMP_DATA_REDUCTION_MINUS; break;
    case '*': *r = OMP_DATA_REDUCTION_MUL; break;
    case '&': *r = OMP_DATA_REDUCTION_BITAND; break;
    case '|': *r = OMP_DATA_REDUCTION_BITOR; break;
    case '^': *r = OMP_DATA_REDUCTION_BITXOR; break;
    case PG_ANDAND: *r = OMP_DATA_REDUCTION_LOGAND; break;
    case PG_OROR: *r = OMP_DATA_REDUCTION_LOGOR; break;
    case PG_IDENT:
	if(PG_IS_IDENT("max")) { *r = OMP_DATA_REDUCTION_MAX; break; }
	if(PG_IS_IDENT("min")) { *r = OMP_DATA_REDUCTION_MIN; break; }
    default:
	return NULL;	/* syntax error */
    }
    pg_get_token();
    if(pg_tok != ':') return NULL;
    pg_get_token();

 next:
    if(pg_tok != PG_IDENT){
      addError(NULL,"empty name list in OMP reduction clause");
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

    addError(NULL,"syntax error in OMP directive clause");
    return NULL;
}

#ifdef not
/*
 * compile pragma, called from compile_statement 
 */

expv compile_OMP_pragma(enum OMP_pragma pragma,expr x)
{
    expv v,c;
    expv pclause,dclause;

    switch(pragma){
    case OMP_PARALLEL: 		/* parallel <clause_list> */
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_PARALLEL,TRUE,
				  &pclause,&dclause);
	v = compile_statement(EXPR_ARG3(x));
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),pclause,v);

    case OMP_PARALLEL_FOR:	/* parallel for <clause_list> */
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_FOR,TRUE,
				  &pclause,&dclause);
	v = compile_statement(EXPR_ARG3(x));
	return elist3(EXPR_LINE(x),OMP_PRAGMA,
		      make_enode(INT_CONSTANT, (void *)OMP_PARALLEL), pclause,
		      elist3(EXPR_LINE(x),OMP_PRAGMA,
			     make_enode(INT_CONSTANT, (void *)OMP_FOR),
			     dclause,v));

    case OMP_PARALLEL_SECTIONS: /* parallel sections <clause_list> */
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SECTIONS,TRUE,
				  &pclause,&dclause);
	v = compile_OMP_SECTIONS_statement(EXPR_ARG3(x));
	return elist3(EXPR_LINE(x),OMP_PRAGMA,
		      make_enode(INT_CONSTANT, (void *)OMP_PARALLEL), pclause,
		      elist3(EXPR_LINE(x),OMP_PRAGMA,
			     make_enode(INT_CONSTANT, (void *)OMP_SECTIONS),
			     dclause, v));

    case OMP_FOR:		/* for <clause_list> */
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_FOR,FALSE,
				  &pclause,&dclause);
	v = compile_statement(EXPR_ARG3(x));
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),dclause,v);
		     
    case OMP_SECTIONS:		/* sections <clause_list> */
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SECTIONS,FALSE,
				  &pclause,&dclause);
	if((v = compile_OMP_SECTIONS_statement(EXPR_ARG3(x))) == NULL)
	  break;
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),dclause,v);

    case OMP_SINGLE:		/* single <clause list> */
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SINGLE,FALSE,
				  &pclause,&dclause);
	v = compile_statement(EXPR_ARG3(x));
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),dclause,v);

    case OMP_MASTER:		/* master */
    case OMP_ORDERED:		/* ordered */
	v = compile_statement(EXPR_ARG3(x));
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),NULL,v);

    case OMP_CRITICAL:		/* critical <name> */
	v = compile_statement(EXPR_ARG3(x));
	c = EXPR_ARG2(x);
	if(c != NULL && LIST_NEXT(EXPR_LIST(c)) != NULL){
	    error_at_node(x,"bad critical section name");
	    break;
	}
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),c,v);

    case OMP_ATOMIC:		/* atomic */
	/* should check next statment */
	if((v = compile_statement(EXPR_ARG3(x))) == NULL) 
	  break;
	if(EXPV_CODE(v) != EXPR_STATEMENT){
	    error_at_node(x,"bad statement for OMP atomic directive");
	    break;
	}
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),NULL,v);

    case OMP_FLUSH:		/* flush <namelist> */
	c = EXPR_ARG2(x);
	compile_OMP_name_list(c);
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),c,NULL);

    case OMP_SECTION:		/* section */
	/* section directive must appear in section block */
	error_at_node(x,"'section' directive in SECTIONS");
	break;

    case OMP_BARRIER:		/* barrier */
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),NULL,NULL);

    case OMP_THREADPRIVATE:
	c = EXPR_ARG2(x);
	compile_OMP_name_list(c);
	return elist3(EXPR_LINE(x),OMP_PRAGMA,EXPR_ARG1(x),c,NULL);

    default:
	fatal("compile_pragma_line: unknown pragma %d",pragma);
    }
    return NULL;
}

static expv compile_OMP_SECTIONS_statement(expr x)
{
    expr xx;
    expv section_list,current_section;
    list lp;

    if(EXPR_CODE(x) != COMPOUND_STATEMENT){
	error_at_node(x,"sections directive must be followed by compound statement block");
	return NULL;
    }
    xx = EXPR_ARG1(x);
    if(xx != NULL){
	error_at_node(xx,"declarations in sections block");
	return NULL;
    }
    section_list = EMPTY_LIST;
    current_section = NULL;
    FOR_ITEMS_IN_LIST(lp,EXPR_ARG2(x)){
	xx = LIST_ITEM(lp);
	if(EXPR_CODE(xx) == PRAGMA_LINE &&
	   EXPR_INT(EXPR_ARG1(xx)) == OMP_SECTION){
	    if(current_section != NULL){
		current_section = list3(COMPOUND_STATEMENT,
					list0(ID_LIST),list0(LIST),
					current_section);
		section_list = exprListAdd(section_list,current_section);
	    }
	    current_section = EMPTY_LIST;
	    continue;
	}
	if(current_section == NULL){
	    /* error_at_node(xx,"statement is not in any section");
	    return NULL; */
	    current_section = EMPTY_LIST;
	}
	current_section = exprListAdd(current_section,
					compile_statement(xx));
    }
    current_section = list3(COMPOUND_STATEMENT,
			    list0(ID_LIST),list0(LIST),
			    current_section);
    section_list = exprListAdd(section_list,current_section);
    return section_list;
}

/* PARALLEL - private,firstprivate,reduction,default,shared,copyin,if
 * FOR      - private,firstprivate,lastprivate,reduction,ordered,shed,nowait
 * SECTIONS - private,firstprivate,lastprivate,reduction,nowait
 * SINGLE   - private,firstprivate,nowait
 */
static void compile_OMP_pragma_clause(expr x, int pragma, int is_parallel,
			  expv *pc,expv *dc)
{
    list lp;
    expr c,v;
    expv pclause = NULL;
    expv dclause;

    if(is_parallel) pclause = EMPTY_LIST;
    dclause = EMPTY_LIST;
    FOR_ITEMS_IN_LIST(lp,x){
	c = LIST_ITEM(lp);
	switch(EXPR_INT(EXPR_ARG1(c))){
	case OMP_DATA_DEFAULT:	/* default(shared|none) */
	    if(!is_parallel){
		error_at_node(x,"'default' clause must be in PARALLEL");
		break;
	    }
	    pclause = exprListAdd(pclause,c);
	    break;
	case OMP_DATA_SHARED:
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(!is_parallel){
		error_at_node(x,"'shared' clause must be in PARALLEL");
		break;
	    }
	    pclause = exprListAdd(pclause,c);
	    break;
	case OMP_DATA_COPYIN:
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(!is_parallel){
		error_at_node(x,"'copyin' clause must be in PARALLEL");
		break;
	    }
	    pclause = exprListAdd(pclause,c);
	    break;
	case OMP_DIR_IF:
	    if(!is_parallel){
		error_at_node(x,"'if' clause must be in PARALLEL");
		break;
	    }
	    v = compile_expression(EXPR_ARG2(c));
	    pclause = exprListAdd(pclause,
					list2(LIST,EXPR_ARG1(c),v));
	    break;

	case OMP_DATA_PRIVATE:
	case OMP_DATA_FIRSTPRIVATE:
	    /* all pragma can have these */
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(pragma == OMP_PARALLEL)
	      pclause = exprListAdd(pclause,c);
	    else     
	      dclause = exprListAdd(dclause,c);
	    break;

	case OMP_DATA_LASTPRIVATE:
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(pragma != OMP_FOR && pragma != OMP_SECTIONS){
		error_at_node(x,"'lastprivate' clause must be in FOR or SECTIONS");
		break;
	    }
	    dclause = exprListAdd(dclause,c);
	    break;

	case OMP_DATA_REDUCTION_PLUS:
	case OMP_DATA_REDUCTION_MINUS:
	case OMP_DATA_REDUCTION_MUL:
	case OMP_DATA_REDUCTION_BITAND:
	case OMP_DATA_REDUCTION_BITOR:
	case OMP_DATA_REDUCTION_BITXOR:
	case OMP_DATA_REDUCTION_LOGAND:
	case OMP_DATA_REDUCTION_LOGOR:
	case OMP_DATA_REDUCTION_MIN:
	case OMP_DATA_REDUCTION_MAX:
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(pragma == OMP_PARALLEL)
	      pclause = exprListAdd(pclause,c);
	    else if(pragma == OMP_FOR || pragma == OMP_SECTIONS)
	      dclause = exprListAdd(dclause,c);
	    else 
	      error_at_node(x,"'reduction' clause must not be in SINGLE");
	    break;

	case OMP_DIR_ORDERED:
	    if(pragma != OMP_FOR){
		error_at_node(x,"'ordered' clause must be in FOR");
		break;
	    }
	    dclause = exprListAdd(dclause,c);
	    break;

	case OMP_DIR_SCHEDULE:
	    if(pragma != OMP_FOR){
		error_at_node(x,"'schedule' clause must be in FOR");
		break;
	    }
	    v = EXPR_ARG2(EXPR_ARG2(c));
	    if(v != NULL && 
	       EXPR_INT(EXPR_ARG1(EXPR_ARG2(c))) != (int)OMP_SCHED_AFFINITY){
		v = compile_expression(v);
		c = list2(LIST,EXPR_ARG1(c),
			  list2(LIST,EXPR_ARG1(EXPR_ARG2(c)),v));
	    }
	    dclause = exprListAdd(dclause,c);
	    break;

	case OMP_DIR_NOWAIT:
	    if(is_parallel){
		error_at_node(x,"'nowait' clause must not be in PARALLEL");
		break;
	    }
	    dclause = exprListAdd(dclause,c);
	    break;

	default:
	    fatal("compile_OMP_paragma_clause");
	}
    }

    /* combination with PARALLEL, don't have to wait */
    if(is_parallel && (pragma != OMP_PARALLEL))
	dclause = exprListAdd(dclause, OMP_PG_LIST(OMP_DIR_NOWAIT, NULL));

    *pc = pclause;
    *dc = dclause;
}

static CExpr* compile_OMP_name_list(expr x)
{
    list lp;
    expr v;
    ID id;
    TYPE_DESC tp;

    FOR_ITEMS_IN_LIST(lp,x){
	v = LIST_ITEM(lp);
	id = lookup_ident(v);
	if(id == NULL){
	    error_at_node(x, "undefined variable, %s in pragma", 
			  SYM_NAME(EXPR_SYM(v)));
	    continue;
	}
	switch(ID_CLASS(id)){
	case AUTO:	/* auto variable */
	case PARAM:	/* paramter */
	case EXTERN:	/* extern variable */
	case EXTDEF:	/* external defition */
	case STATIC:	/* static variable */
	case REGISTER:	/* register variable */
	    tp = ID_TYPE(id);
	    if ( IS_FUNCTION(tp) ){
		error_at_node(x, "function name, %s in pragma", 
			      SYM_NAME(EXPR_SYM(v)));
	    }
	    break;
	default:
	  error_at_node(x, "identifer, %s is not variable in pragma",
			SYM_NAME(EXPR_SYM(v)));
	}
    }
}

#endif

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
  default:  return "???OMP???";
  }
}
