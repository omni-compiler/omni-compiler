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
expv OMN_pragma_list(int pragma, char *dir_name, expv arg1, expv arg2);
//void check_for_OMP_pragma(expr x);
//int check_for_XMP_pragma(int st_no, expr x);

//void init_for_ACC_pragma();
//void check_for_ACC_pragma(expr x);
//static void check_for_ACC_pragma_2(expr x, enum ACC_pragma dir);
static expv compile_clause_list(expr x);
//static int is_ACC_pragma(expr e, enum ACC_pragma p);

static int _OMN_do_required = FALSE;
static int _OMN_st_required = 0; //FALSE;

/* enum OMP_st_pragma { */
/*     OMP_ST_NONE, */
/*     OMP_ST_ATOMIC, */
/*     OMP_ST_END */
/* }; */

/* static enum OMP_st_pragma OMP_st_required, OMP_st_flag; */


/* int ACC_reduction_op(expr v) */
/* { */
/*     char *s; */

/*     if(EXPR_CODE(v) != IDENT) fatal("ACC_reduction_op: no IDENT"); */
/*     s = SYM_NAME(EXPR_SYM(v)); */
/*     if(strcmp("max",s) == 0) return  (int)ACC_CLAUSE_REDUCTION_MAX; */
/*     if(strcmp("min",s) == 0) return  (int)ACC_CLAUSE_REDUCTION_MIN; */
/*     if(strcmp("iand",s) == 0) return (int)ACC_CLAUSE_REDUCTION_BITAND; */
/*     if(strcmp("ior",s) == 0) return  (int)ACC_CLAUSE_REDUCTION_BITOR; */
/*     if(strcmp("ieor",s) == 0) return (int)ACC_CLAUSE_REDUCTION_BITXOR; */

/*     error("bad intrinsic function in REDUCTION clause of ACC"); */
/*     return ACC_CLAUSE_REDUCTION_PLUS;	/\* dummy *\/ */
/* } */

/* int ACC_num_attr(expr v) */
/* { */
/*     char *s; */

/*     if(EXPR_CODE(v) != IDENT) fatal("ACC_num_attr: no IDENT"); */
/*     s = SYM_NAME(EXPR_SYM(v)); */
/*     if(strcmp("num",s) == 0) return (int)ACC_CLAUSE_NUM_GANGS; */
/*     //    if(strcmp("length",s) == 0) return (int)ACC_VECTOR_LENGTH; */
/*     if(strcmp("static",s) == 0) return (int)ACC_CLAUSE_STATIC; */

/*     error("bad int-expr attribute for ACC pragma"); */
/*     return ACC_CLAUSE_VECTOR_LENGTH;	/\* dummy *\/ */
/* } */

/* void ACC_check_num_attr(expr v, enum ACC_pragma attr) */
/* { */
/*     char *s; */
/*     enum ACC_pragma a; */

/*     if(EXPR_CODE(v) != IDENT) fatal("ACC_num_attr: no IDENT"); */
/*     s = SYM_NAME(EXPR_SYM(v)); */
/*     a = ACC_CLAUSE_END; */
/*     if(strcmp("num",s) == 0 )        a = ACC_CLAUSE_NUM_WORKERS; */
/*     else if(strcmp("length",s) == 0) a = ACC_CLAUSE_VECTOR_LENGTH; */
/*     else if(strcmp("static",s) == 0) a = ACC_CLAUSE_STATIC; */

/*     if(a != attr){ */
/*       error("bad int-expr attribute for ACC pragma"); */
/*     } */
/* } */

void init_for_OMN_pragma()
{
  _OMN_do_required = FALSE;
  _OMN_st_required = 0; //FALSE;
}


static void push_OMN_construct(char *dir_name, expv clauses)
{
  push_ctl(CTL_OMN);
  CTL_OMN_ARG(ctl_top) = OMN_pragma_list(OMN_PRAGMA, dir_name, clauses, NULL);
  EXPR_LINE(CTL_OMN_ARG(ctl_top)) = current_line;
}

/* static void pop_OMN_construct(char *dir_name) */
/* { */
/*   //  if (CTL_TYPE(ctl_top) == CTL_OMN && CTL_OMN_ARG_DIR(ctl_top) == dir_name){ */
/*     CTL_BLOCK(ctl_top) = OMN_pragma_list(dir_name, CTL_OMN_ARG_CLAUSE(ctl_top), */
/* 					 CURRENT_STATEMENTS); */
/*     EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMN_ARG(ctl_top)); */
/*     pop_ctl(); */
/*     //  } */
/*     //  else { */
/*     //    error("OMN block is not closed"); */
/*     //} */
/* } */

/* static void pop_OMN_loop_construct(char *dir_name) */
/* { */
/*   expv xx, statements = CURRENT_STATEMENTS; */

/*   /\* extract a single do statement from a list which contains only the statement*\/ */
/*   if (EXPR_CODE(statements) == LIST) { */
/*     list lp; */
/*     FOR_ITEMS_IN_LIST(lp,statements){ */
/*       xx = LIST_ITEM(lp); */
/*       if(EXPR_CODE(xx) == F_PRAGMA_STATEMENT) */
/* 	CTL_SAVE(ctl_top) = list_put_last(CTL_SAVE(ctl_top),xx); */
/*       else { */
/* 	statements = xx; */
/* 	break; */
/*       } */
/*     } */
/*   } */

/*   if (EXPR_CODE(statements) != F_DO_STATEMENT){ */
/*     fatal("OMN  directive must be followed by do statements"); */
/*   } */

/*   CTL_BLOCK(ctl_top) = OMN_pragma_list(dir_name, CTL_OMN_ARG_CLAUSE(ctl_top), statements); */
/*   EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMN_ARG(ctl_top)); */
/*   pop_ctl(); */
/* } */

/* static void pop_ACC_atomic_construct(enum ACC_pragma dir) */
/* { */
/*   if(CTL_TYPE(ctl_top) != CTL_ACC || CTL_ACC_ARG_DIR(ctl_top) != dir){ */
/*     error("OpenACC ATOMIC is not closed"); */
/*   } */

/*   expv statements = CURRENT_STATEMENTS; */
/*   /\* expv xx; *\/ */
/*   /\* /\\* extract a single do statement from a list which contains only the statement*\\/ *\/ */
/*   /\* if(EXPR_CODE(statements) == LIST) { *\/ */
/*   /\*   list lp; *\/ */
/*   /\*   FOR_ITEMS_IN_LIST(lp,statements){ *\/ */
/*   /\*     xx = LIST_ITEM(lp); *\/ */
/*   /\*     if(EXPR_CODE(xx) == F_PRAGMA_STATEMENT)  *\/ */
/*   /\* 	CTL_SAVE(ctl_top) = list_put_last(CTL_SAVE(ctl_top),xx); *\/ */
/*   /\*     else { *\/ */
/*   /\* 	statements = xx; *\/ */
/*   /\* 	break; *\/ */
/*   /\*     } *\/ */
/*   /\*   } *\/ */
/*   /\* } *\/ */

/*   /\* if(EXPR_CODE(statements) != F_LET_STATEMENT){ *\/ */
/*   /\*   fatal("ACC ATOMIC directive must be followed by assignment"); *\/ */
/*   /\* } *\/ */

/*   CTL_BLOCK(ctl_top) = ACC_pragma_list(dir,CTL_ACC_ARG_CLAUSE(ctl_top),statements); */
/*   EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_ACC_ARG(ctl_top)); */
/*   pop_ctl(); */
/* } */

/* static enum ACC_pragma get_begin_directive(enum ACC_pragma dir) */
/* { */
/*   switch(dir){ */
/*   case ACC_END_PARALLEL: */
/*     return ACC_PARALLEL; */
/*   case ACC_END_KERNELS: */
/*     return ACC_KERNELS; */
/*   case ACC_END_DATA: */
/*     return ACC_DATA; */
/*   case ACC_END_PARALLEL_LOOP: */
/*     return ACC_PARALLEL_LOOP; */
/*   case ACC_END_KERNELS_LOOP: */
/*     return ACC_KERNELS_LOOP; */
/*   case ACC_END_ATOMIC: */
/*     return ACC_ATOMIC; */
/*   case ACC_END_HOST_DATA: */
/*     return ACC_HOST_DATA; */

/*   default: */
/*     fatal("get_begin_directive: not end directive"); */
/*   } */
/*   return ACC_DIR_END; */
/* } */

void compile_OMN_directive(expr x)
{
    if (x == NULL) return;	/* error */

    if (debug_flag) {
	fprintf(stderr, "OMN_directive:\n");
	expv_output(x, stderr);
	fprintf(stderr, "\n");
    }

    char *dir_name = EXPR_STR(EXPR_ARG1(x));  /* direcive name */
    expv clauses = compile_clause_list(EXPR_ARG2(x));

    check_INEXEC();
    push_OMN_construct(dir_name, clauses);
    _OMN_do_required = TRUE;

    return;
}

void compile_OMN_decl_directive(expr x)
{
    if (x == NULL) return;	/* error */

    if (debug_flag) {
	fprintf(stderr, "OMN_directive:\n");
	expv_output(x, stderr);
	fprintf(stderr, "\n");
    }

    char *dir_name = EXPR_STR(EXPR_ARG1(x));  /* direcive name */
    expv clauses = compile_clause_list(EXPR_ARG2(x));

    output_statement(OMN_pragma_list(OMNDECL_PRAGMA, dir_name, clauses, NULL));

    return;
}

expv OMN_pragma_list(int pragma, char *dir_name, expv arg1, expv arg2)
{
  expv xx = arg2;
  
  if (arg2 != NULL && EXPR_CODE(arg2) == LIST &&
      EXPR_CODE(EXPR_ARG1(arg2)) == F_DO_STATEMENT){
    xx = EXPR_ARG1(arg2);
  }
  
  return list3(pragma, expv_str_term(STRING_CONSTANT, NULL, dir_name),
	       arg1, xx);
}

void check_for_OMN_pragma(expr x)
{
  if (CTL_TYPE(ctl_top) != CTL_OMN) return;

  //char *ctl_top_dir = CTL_OMN_ARG_DIR(ctl_top);
  //  pop_OMN_loop_construct(ctl_top_dir);

}

/* static int is_ACC_pragma(expr e, enum ACC_pragma p) */
/* { */
/*   if(EXPR_CODE(e) != ACC_PRAGMA){ */
/*     return FALSE; */
/*   } */
  
/*   if(EXPR_INT(EXPR_ARG1(e)) != p){ */
/*     return FALSE; */
/*   } */
/*   return TRUE; */
/* } */

static expv compile_clause(expr x);


static expv compile_clause_arg(expr x)
{
  if(x == NULL) return NULL;
  
  if(EXPR_CODE(x) == OMN_PRAGMA){
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

/* static expv compile_clause(expr x) */
/* { */
/*   if(x == NULL) return NULL; */

/*   if( EXPR_CODE(x) != OMN_PRAGMA){ */
/*     error("compile_clause: not clause"); */
/*   } */
  
/*   expr arg = EXPR_ARG2(x); */
/*   if(arg){ */
/*     expv argv; */
/*     if(EXPR_CODE(arg) != LIST){ */
/*       argv = compile_clause_arg(arg); */
/*     }else{ */
/*       argv = compile_clause_arg_list(arg); */
/*     } */
/*     return list2(OMN_PRAGMA, EXPR_ARG1(x), argv); */
/*   }else{ */
/*     return list1(OMN_PRAGMA, EXPR_ARG1(x)); */
/*   } */
/* } */

static expv compile_clause(expr x)
{
  if (x == NULL) return NULL;

  expv argv;
  if (EXPR_CODE(x) != LIST){
    argv = compile_clause_arg(x);
  }
  else{
    argv = compile_clause_arg_list(x);
  }

  return argv;
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

/* int is_ACC_loop_pragma(expv x) */
/* { */
/*   if(EXPR_CODE(x) != ACC_PRAGMA) return FALSE; */

/*   enum ACC_pragma code = EXPR_INT(EXPR_ARG1(x)); */
/*   if(code == ACC_LOOP || */
/*      code == ACC_PARALLEL_LOOP ||  */
/*      code == ACC_KERNELS_LOOP ){ */
/*     return TRUE; */
/*   } */

/*   return FALSE; */
/* } */
