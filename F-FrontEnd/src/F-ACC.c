/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "F-front.h"

extern CTL *ctl_top_saved;
extern expv CURRENT_STATEMENTS_saved;

/* expv OMP_check_SECTION(expr x); */
/* expv OMP_pragma_list(enum OMP_pragma pragma,expv arg1,expv arg2); */
/* expv OMP_FOR_pragma_list(expv clause,expv statements); */
/* expv OMP_atomic_statement(expv v); */

/* void compile_OMP_name_list(expr x); */
/* void compile_OMP_pragma_clause(expr x, int pragma, int is_parallel, */
/* 			  expv *pc,expv *dc); */
expv ACC_pragma_list(enum ACC_pragma pragma,expv arg1,expv arg2);
/* void check_for_OMP_pragma(expr x); */

void init_for_ACC_pragma();
void check_for_ACC_pragma(expr x);
static void check_for_ACC_pragma_2(expr x, enum ACC_pragma dir);
static expv compile_clause_list(expr x);
static int is_ACC_pragma(expr e, enum ACC_pragma p);

static int _ACC_do_required = FALSE;
static int _ACC_st_required = 0; //FALSE;

/* enum OMP_st_pragma { */
/*     OMP_ST_NONE, */
/*     OMP_ST_ATOMIC, */
/*     OMP_ST_END */
/* }; */

/* static enum OMP_st_pragma OMP_st_required, OMP_st_flag; */


int ACC_reduction_op(expr v)
{
    char *s;

    if(EXPR_CODE(v) != IDENT) fatal("ACC_reduction_op: no IDENT");
    s = SYM_NAME(EXPR_SYM(v));
    if(strcmp("max",s) == 0) return  (int)ACC_CLAUSE_REDUCTION_MAX;
    if(strcmp("min",s) == 0) return  (int)ACC_CLAUSE_REDUCTION_MIN;
    if(strcmp("iand",s) == 0) return (int)ACC_CLAUSE_REDUCTION_BITAND;
    if(strcmp("ior",s) == 0) return  (int)ACC_CLAUSE_REDUCTION_BITOR;
    if(strcmp("ieor",s) == 0) return (int)ACC_CLAUSE_REDUCTION_BITXOR;

    error("bad intrinsic function in REDUCTION clause of ACC");
    return ACC_CLAUSE_REDUCTION_PLUS;	/* dummy */
}

int ACC_num_attr(expr v)
{
    char *s;

    if(EXPR_CODE(v) != IDENT) fatal("ACC_num_attr: no IDENT");
    s = SYM_NAME(EXPR_SYM(v));
    if(strcmp("num",s) == 0) return (int)ACC_CLAUSE_NUM_GANGS;
    //    if(strcmp("length",s) == 0) return (int)ACC_VECTOR_LENGTH;
    if(strcmp("static",s) == 0) return (int)ACC_CLAUSE_STATIC;

    error("bad int-expr attribute for ACC pragma");
    return ACC_CLAUSE_VECTOR_LENGTH;	/* dummy */
}

void ACC_check_num_attr(expr v, enum ACC_pragma attr)
{
    char *s;
    enum ACC_pragma a;

    if(EXPR_CODE(v) != IDENT) fatal("ACC_num_attr: no IDENT");
    s = SYM_NAME(EXPR_SYM(v));
    a = ACC_CLAUSE_END;
    if(strcmp("num",s) == 0 )        a = ACC_CLAUSE_NUM_WORKERS;
    else if(strcmp("length",s) == 0) a = ACC_CLAUSE_VECTOR_LENGTH;
    else if(strcmp("static",s) == 0) a = ACC_CLAUSE_STATIC;

    if(a != attr){
      error("bad int-expr attribute for ACC pragma");
    }
}

void init_for_ACC_pragma()
{
  _ACC_do_required = FALSE;
  _ACC_st_required = 0; //FALSE;
}


static void push_ACC_construct(enum ACC_pragma dir, expv clauses)
{
  push_ctl(CTL_ACC);
  CTL_ACC_ARG(ctl_top) = ACC_pragma_list(dir,clauses,NULL);
  EXPR_LINE(CTL_ACC_ARG(ctl_top)) = current_line;
}

static void pop_ACC_construct(enum ACC_pragma dir)
{
  if(CTL_TYPE(ctl_top) == CTL_ACC &&
     CTL_ACC_ARG_DIR(ctl_top) == dir){
    CTL_BLOCK(ctl_top) = 
      ACC_pragma_list(dir,CTL_ACC_ARG_CLAUSE(ctl_top),
		      CURRENT_STATEMENTS);
    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_ACC_ARG(ctl_top));
    pop_ctl();
  } else {
    error("OpenACC block is not closed");
  }
}

static void pop_ACC_loop_construct(enum ACC_pragma dir)
{
  if(CTL_TYPE(ctl_top) != CTL_ACC || CTL_ACC_ARG_DIR(ctl_top) != dir){
    error("OpenACC LOOP | PARALLEL LOOP | KERNELS LOOP is not closed");
  }

  expv xx, statements = CURRENT_STATEMENTS;

  /* extract a single do statement from a list which contains only the statement*/
  if(EXPR_CODE(statements) == LIST) {
    list lp;
    FOR_ITEMS_IN_LIST(lp,statements){
      xx = LIST_ITEM(lp);
      if(EXPR_CODE(xx) == F_PRAGMA_STATEMENT) 
	CTL_SAVE(ctl_top) = list_put_last(CTL_SAVE(ctl_top),xx);
      else {
	statements = xx;
	break;
      }
    }
  }

  if(EXPR_CODE(statements) != F_DO_STATEMENT){
    fatal("ACC LOOP | PARALLEL LOOP | KERNELS LOOP directive must be followed by do statements");
  }

  CTL_BLOCK(ctl_top) = ACC_pragma_list(dir,CTL_ACC_ARG_CLAUSE(ctl_top),statements);
  EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_ACC_ARG(ctl_top));
  pop_ctl();
}

static void pop_ACC_atomic_construct(enum ACC_pragma dir)
{
  if(CTL_TYPE(ctl_top) != CTL_ACC || CTL_ACC_ARG_DIR(ctl_top) != dir){
    error("OpenACC ATOMIC is not closed");
  }

  expv statements = CURRENT_STATEMENTS;
  /* expv xx; */
  /* /\* extract a single do statement from a list which contains only the statement*\/ */
  /* if(EXPR_CODE(statements) == LIST) { */
  /*   list lp; */
  /*   FOR_ITEMS_IN_LIST(lp,statements){ */
  /*     xx = LIST_ITEM(lp); */
  /*     if(EXPR_CODE(xx) == F_PRAGMA_STATEMENT)  */
  /* 	CTL_SAVE(ctl_top) = list_put_last(CTL_SAVE(ctl_top),xx); */
  /*     else { */
  /* 	statements = xx; */
  /* 	break; */
  /*     } */
  /*   } */
  /* } */

  /* if(EXPR_CODE(statements) != F_LET_STATEMENT){ */
  /*   fatal("ACC ATOMIC directive must be followed by assignment"); */
  /* } */

  CTL_BLOCK(ctl_top) = ACC_pragma_list(dir,CTL_ACC_ARG_CLAUSE(ctl_top),statements);
  EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_ACC_ARG(ctl_top));
  pop_ctl();
}

static enum ACC_pragma get_begin_directive(enum ACC_pragma dir)
{
  switch(dir){
  case ACC_END_PARALLEL:
    return ACC_PARALLEL;
  case ACC_END_KERNELS:
    return ACC_KERNELS;
  case ACC_END_DATA:
    return ACC_DATA;
  case ACC_END_PARALLEL_LOOP:
    return ACC_PARALLEL_LOOP;
  case ACC_END_KERNELS_LOOP:
    return ACC_KERNELS_LOOP;
  case ACC_END_ATOMIC:
    return ACC_ATOMIC;
  case ACC_END_HOST_DATA:
    return ACC_HOST_DATA;

  default:
    fatal("get_begin_directive: not end directive");
  }
  return ACC_DIR_END;
}

void compile_ACC_directive(expr x)
{
    if(x == NULL) return;	/* error */

    if (debug_flag) {
	fprintf(stderr, "ACC_directive:\n");
	expv_output(x, stderr);
	fprintf(stderr, "\n");
    }

    //check_for_OMP_pragma(x);
    //check_for_XMP_pragma(-1, x);

    //loop, parallel loop, or kernels loop
    /* if(_ACC_do_required){ */
    /* 	error("OpenACC LOOP directive must be followed by do statement"); */
    /* 	_ACC_do_required = FALSE; */
    /* 	return; */
    /* } */

    /* if(ACC_st_required != ACC_ST_NONE){ */
    /* 	error("OpenACC ATOMIC directive must be followed by assignment"); */
    /* 	return; */
    /* } */

    expr directive = EXPR_ARG1(x);  /* direcive name */
    expv clauses = compile_clause_list(EXPR_ARG2(x));
    enum ACC_pragma dir_enum = EXPR_INT(directive);

    check_for_ACC_pragma_2(x, dir_enum);

    switch(dir_enum){
      //Constructs with end pragma
    case ACC_PARALLEL:
    case ACC_KERNELS:
    case ACC_DATA:
    case ACC_HOST_DATA:
      check_INEXEC();
      push_ACC_construct(dir_enum, clauses);
      return;
    case ACC_END_PARALLEL:
    case ACC_END_KERNELS:
    case ACC_END_DATA:
    case ACC_END_HOST_DATA:
      check_INEXEC();
      pop_ACC_construct(get_begin_directive(dir_enum));
      return;

      //Constructs for do with/without end pragma
    case ACC_LOOP:
    case ACC_PARALLEL_LOOP:
    case ACC_KERNELS_LOOP:
      check_INEXEC();
      push_ACC_construct(dir_enum, clauses);
      _ACC_do_required = TRUE;
      return;
    case ACC_END_PARALLEL_LOOP:
    case ACC_END_KERNELS_LOOP:
      check_INEXEC();
      pop_ACC_loop_construct(get_begin_directive(dir_enum));
      return;

      //Construct for statement with/without end pragma
    case ACC_ATOMIC:
      check_INEXEC();
      push_ACC_construct(dir_enum, clauses);
      if(is_ACC_pragma(EXPR_ARG1(clauses), ACC_CLAUSE_CAPTURE)){
	_ACC_st_required = 2;
      }else{
	_ACC_st_required = 1;
      }
      return;
    case ACC_END_ATOMIC:
      check_INEXEC();
      pop_ACC_construct(get_begin_directive(dir_enum));
      return;

      //Executable directives
    case ACC_WAIT:
    case ACC_CACHE:
    case ACC_ENTER_DATA:
    case ACC_EXIT_DATA:
    case ACC_UPDATE_D:
    case ACC_INIT:
    case ACC_SHUTDOWN:
    case ACC_SET:
      check_INEXEC();
      output_statement(ACC_pragma_list(dir_enum, clauses, NULL));
      return;

      //Declaration directives
    case ACC_ROUTINE:
    case ACC_DECLARE:
      check_INDCL();
      output_statement(ACC_pragma_list(dir_enum, clauses, NULL));
      return;

    default:
	fatal("unknown ACC pragma");
    }
}

expv ACC_pragma_list(enum ACC_pragma pragma,expv arg1,expv arg2)
{
    return list3(ACC_PRAGMA,expv_int_term(INT_CONSTANT,NULL,(int)pragma),
		 arg1,arg2);
}

static void check_for_ACC_pragma_2(expr x, enum ACC_pragma dir)
{
  int ret = 1;

  if(_ACC_do_required){
    // don't care the order of pragma around ACC LOOP
    if(EXPR_CODE(x) == F_PRAGMA_STATEMENT) return;
    if(EXPR_CODE(x) != F_DO_STATEMENT){
      error("ACC LOOP directives must be followed by do statement");
    }
    _ACC_do_required = FALSE; //1. ここでまずloopの次にdoがくることを確認
    return;
  }
  
  if(_ACC_st_required > 0){
    if(EXPR_CODE(x) != F_LET_STATEMENT){
      error("OpenACC ATOMIC directives must be followed by assignment");
    }
    _ACC_st_required -= 1; //FALSE;
    return;
  }


  //2. ctlスタックのtopにloop, parallel loop, kernels loopがあればcloseする。endが省略されたとき用
  if(CTL_TYPE(ctl_top) != CTL_ACC) return;

  enum ACC_pragma dir_enum = CTL_ACC_ARG_DIR(ctl_top);

  if(dir != ACC_END_PARALLEL_LOOP && dir != ACC_END_KERNELS_LOOP){
  /* close LOOP | PARALLEL_LOOP | KERNELS_LOOP directive if possible */
    if(dir_enum == ACC_LOOP
       || dir_enum == ACC_PARALLEL_LOOP
       || dir_enum == ACC_KERNELS_LOOP
       ){
      pop_ACC_loop_construct(dir_enum);
    }
  }

  if(dir != ACC_END_ATOMIC){
    /* close ATOMIC directive if possible */
    if(dir_enum == ACC_ATOMIC){
      expv clauses = CTL_ACC_ARG_CLAUSE(ctl_top);
      int is_clause = is_ACC_pragma(EXPR_ARG1(clauses), ACC_CLAUSE_CAPTURE);
      if(! is_clause){
	pop_ACC_atomic_construct(dir_enum);
      }
    }
  }
}

void check_for_ACC_pragma(expr x)
{
  check_for_ACC_pragma_2(x, ACC_DIR_END);
}

static int is_ACC_pragma(expr e, enum ACC_pragma p)
{
  if(EXPR_CODE(e) != ACC_PRAGMA){
    return FALSE;
  }
  
  if(EXPR_INT(EXPR_ARG1(e)) != p){
    return FALSE;
  }
  return TRUE;
}

static expv compile_clause(expr x);

static expv compile_subscript(expr x)
{
  expr xx;
  list lp;
  expv vv, ret_list;

  if(x == NULL) return NULL;

  ret_list = EMPTY_LIST;

  if( EXPR_CODE(x) != LIST){
    //index
    vv = compile_expression(x);
  }else{
    //index-range
    FOR_ITEMS_IN_LIST(lp,x){
      xx = LIST_ITEM(lp);
      vv = compile_expression(xx);
      ret_list = list_put_last(ret_list, vv);
    }
  }
  return ret_list;  
}

static expv compile_subscript_list(expr x)
{
  expr xx;
  list lp;
  expv vv, ret_list;

  if(x == NULL) return NULL;

  if( EXPR_CODE(x) != LIST){
    error("compile_subscript_list: not list");
  }

  ret_list = EMPTY_LIST;
  FOR_ITEMS_IN_LIST(lp,x){
    xx = LIST_ITEM(lp);
    vv = compile_subscript(xx);
    ret_list = list_put_last(ret_list, vv);
  }
  return ret_list;  
}

static expv compile_subarray(expr x)
{
  if(x == NULL) return NULL;

  if(EXPR_CODE(x) != LIST){
    error("compile_subarray: not list");
  }

  expv var = compile_expression(EXPR_ARG1(x));
  expv subscripts = compile_subscript_list(EXPR_ARG2(x));

  if(EXPV_CODE(var) != F_VAR){
    error("variable or arrayname is expected in OpenACC directive");
  }
		   
  return list2(LIST, var, subscripts);

}

static expv compile_clause_arg(expr x)
{
  if(x == NULL) return NULL;
  
  if(EXPR_CODE(x) == ACC_PRAGMA){
    //none, present, etc.
    return compile_clause(x);
  }else if(EXPR_CODE(x) == LIST){
    //F_ARRAY_REF
    return compile_expression(x);
  }else{
    //int-expr, identifier
    return compile_expression(x);
  }
}

static expv compile_clause_arg_list(expr x)
{
  expr xx;
  list lp;
  expv vv, ret_list;

  if(x == NULL) return NULL;

  ret_list = EMPTY_LIST;

  if( EXPR_CODE(x) != LIST){
    error("compile_clause_arg_list: not list");
  }
  FOR_ITEMS_IN_LIST(lp,x){
    xx = LIST_ITEM(lp);
    vv = compile_clause_arg(xx);
    ret_list = list_put_last(ret_list, vv);
  }
  return ret_list;  
}

static expv compile_clause(expr x)
{
  if(x == NULL) return NULL;

  if( EXPR_CODE(x) != ACC_PRAGMA){
    error("compile_clause: not clause");
  }
  
  expr arg = EXPR_ARG2(x);
  if(arg){
    expv argv;
    if(EXPR_CODE(arg) != LIST){
      argv = compile_clause_arg(arg);
    }else{
      argv = compile_clause_arg_list(arg);
    }
    return list2(ACC_PRAGMA, EXPR_ARG1(x), argv);
  }else{
    return list1(ACC_PRAGMA, EXPR_ARG1(x));
  }
}

static expv compile_clause_list(expr x)
{
  expr xx;
  list lp;
  expv vv, ret_list;

  ret_list = EMPTY_LIST;
  
  if(x == NULL) return NULL;

  if( EXPR_CODE(x) != LIST){
    error("compile_clause: not list");
  }
  FOR_ITEMS_IN_LIST(lp,x){
    xx = LIST_ITEM(lp);
    vv = compile_clause(xx);
    ret_list = list_put_last(ret_list, vv);
  }
  return ret_list;
}
