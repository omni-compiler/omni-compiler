/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "F-front.h"

expv XMP_check_TASK(expr x);
expv XMP_pragma_list(enum XMP_pragma pragma,expv arg1,expv arg2);
static int	close_XMP_IO_closure(int st_no, expr x);

// void compile_XMP_name_list(expr x);

int XMP_do_required;
int XMP_gmove_required;
int XMP_io_desired_statements = 0;
expv XMP_gmove_directive;

int XMP_reduction_op(expr v)
{
    char *s;

    if(EXPR_CODE(v) != IDENT) fatal("XMP_reduction_op: no IDENT");
    s = SYM_NAME(EXPR_SYM(v));
    if(strcmp("max",s) == 0) return (int)XMP_DATA_REDUCE_MAX;
    if(strcmp("min",s) == 0) return (int)XMP_DATA_REDUCE_MIN;
    if(strcmp("iand",s) == 0) return (int)XMP_DATA_REDUCE_BAND;
    if(strcmp("ior",s) == 0) return (int)XMP_DATA_REDUCE_BOR;
    if(strcmp("ieor",s) == 0) return (int)XMP_DATA_REDUCE_BXOR;

    error("bad intrinsic function in REDUCTION clause of XMP");
    return XMP_DATA_REDUCE_SUM;	/* dummy */
}

expv XMP_pragma_list(enum XMP_pragma pragma,expv arg1,expv arg2)
{
    return list3(XMP_PRAGMA,expv_int_term(INT_CONSTANT,NULL,(int)pragma),
		 arg1,arg2);
}

/*
 * called at the begining of program unit (subroutine)
 */
void init_for_XMP_pragma()
{
    XMP_do_required = FALSE;
    XMP_gmove_required = FALSE;
    XMP_io_desired_statements = 0;
}

/*
 * called from Parser
 */
void compile_XMP_directive(expr x)
{
    expr dir;

    if(x == NULL) return;	/* error */

    if(debug_flag){
	printf("XMP_directive:\n");
	expv_output(x,stdout);
	printf("\n");
    }

    if(XMP_do_required){
	error("OpenMP DO directives must be followed by do statement");
	XMP_do_required = FALSE;
	return;
    }

    if(XMP_gmove_required){
	error("OpenMP ATOMIC directives must be followed by assignment");
	XMP_do_required = FALSE;
	return;
    }

    dir = EXPR_ARG1(x);  /* direcive name */

#ifdef not
    if(EXPR_INT(dir) != XMP_TASKS && EXPR_INT(dir) != XMP_TASK)
	check_for_XMP_pragma(NULL);  /* close DO directives if any */
#endif	

    switch(EXPR_INT(dir)){
      
    case XMP_NODES:
      check_INDCL();
      /* check arg: (maptype, nameNames, nodeSizeList, inherit) */
      output_statement(x);
      break;

    case XMP_TEMPLATE:
      check_INDCL();
      /* check arg: (templateNameList, templateSpecList) */
      output_statement(x);
      break;

    case XMP_DISTRIBUTE:
      check_INDCL();
      /* check arg: (templateNameList, templateSpecList) */
      output_statement(x);
      break;

    case XMP_ALIGN:
      check_INDCL();
      /* check arg: (arrayNameList, alignSrcList,templateName, alignSubsript) */
      output_statement(x);
      break;

    case XMP_SHADOW:
      check_INDCL();
      /* check arg: (arrayName, shadowWidthList) */
      output_statement(x);
      break;

    case XMP_TASK:
      check_INEXEC();
      /* check: arg */
      push_ctl(CTL_XMP);
      CTL_XMP_ARG(ctl_top) = x;
      EXPR_LINE(CTL_XMP_ARG(ctl_top)) = current_line;
      return;

    case XMP_END_TASK:
      check_INEXEC();
      if(CTL_TYPE(ctl_top) == CTL_XMP &&
	 CTL_XMP_ARG_DIR(ctl_top) == XMP_TASK){
	CTL_BLOCK(ctl_top) = 
	  XMP_pragma_list(XMP_TASK,CTL_XMP_ARG_CLAUSE(ctl_top),
			  CURRENT_STATEMENTS);
	EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_XMP_ARG(ctl_top));
	pop_ctl();
      } else  error("XMP TASK block is not closed");
      break;

    case XMP_TASKS:
      check_INEXEC();
      /* check: arg */
      push_ctl(CTL_XMP);
      CTL_XMP_ARG(ctl_top) = x;
      EXPR_LINE(CTL_XMP_ARG(ctl_top)) = current_line;
      break;

    case XMP_END_TASKS:
      check_INEXEC();
      if(CTL_TYPE(ctl_top) == CTL_XMP &&
	 CTL_XMP_ARG_DIR(ctl_top) == XMP_TASKS){
	CURRENT_STATEMENTS = XMP_check_TASK(CURRENT_STATEMENTS);
	CTL_BLOCK(ctl_top) = 
	  XMP_pragma_list(XMP_TASK,CTL_XMP_ARG_CLAUSE(ctl_top),
			  CURRENT_STATEMENTS);
	EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_XMP_ARG(ctl_top));
	pop_ctl();
      } else  error("XMP TASKS block is not closed");
      break;

    case XMP_LOOP:
      check_INEXEC();
      push_ctl(CTL_XMP);
      CTL_XMP_ARG(ctl_top) = x;
      EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
      XMP_do_required = TRUE;
      break;

    case XMP_GMOVE:
      check_INEXEC();
      XMP_gmove_required = TRUE;
      XMP_gmove_directive = x;
      break;

    case XMP_REFLECT:
      check_INEXEC();
      /* check arg: (arrayNameList???, opt) */
      output_statement(x);
      break;

    case XMP_BARRIER:
      check_INEXEC();
      output_statement(x);
      break;

    case XMP_REDUCTION:
      check_INEXEC();
      output_statement(x);
      break;

    case XMP_BCAST:
      check_INEXEC();
      output_statement(x);
      break;

    case XMP_TEMPLATE_FIX:
      check_INEXEC();
      output_statement(x);
      break;

    case XMP_COARRAY:
      check_INDCL();
      /* not yet */
      output_statement(x);
      break;

    case XMP_GLOBAL_IO_BEGIN:
    case XMP_MASTER_IO_BEGIN:
      check_INEXEC();
      XMP_io_desired_statements = EXPR_INT(EXPR_ARG1(EXPR_ARG2(x)));
      push_ctl(CTL_XMP);
      CTL_XMP_ARG(ctl_top) = x;
      EXPR_LINE(CTL_XMP_ARG(ctl_top)) = current_line;
      break;

    case XMP_END_GLOBAL_IO:
      if (CTL_TYPE(ctl_top) == CTL_XMP) {
	if (CTL_XMP_ARG_DIR(ctl_top) == XMP_GLOBAL_IO_BEGIN) {
	  (void)close_XMP_IO_closure(0, CURRENT_STATEMENTS);
	} else {
	  error("about to close \"global_io begin\" with "
		"\"end master_io\"");
	}
      } else {
	error("a closure for a global_io is not yet opened.");
      }
      XMP_io_desired_statements = 0;
      break;

    case XMP_END_MASTER_IO:
      if (CTL_TYPE(ctl_top) == CTL_XMP) {
	if (CTL_XMP_ARG_DIR(ctl_top) == XMP_MASTER_IO_BEGIN) {
	  (void)close_XMP_IO_closure(0, CURRENT_STATEMENTS);
	} else {
	  error("about to close \"master_io begin\" with "
		"\"end global_io\"");
	}
      } else {
	error("a closure for a master_io is not yet opened.");
      }
      XMP_io_desired_statements = 0;
      break;

    default:
	fatal("unknown XMP pragma");
    }
}

static int
isIOStatement(expr x)
{
  return (EXPR_CODE(x) == F_PRINT_STATEMENT ||
	  EXPR_CODE(x) == F_WRITE_STATEMENT ||
	  EXPR_CODE(x) == F_READ_STATEMENT ||
	  EXPR_CODE(x) == F_READ1_STATEMENT ||
	  EXPR_CODE(x) == F_OPEN_STATEMENT ||
	  EXPR_CODE(x) == F_CLOSE_STATEMENT ||
	  EXPR_CODE(x) == F_BACKSPACE_STATEMENT ||
	  EXPR_CODE(x) == F_ENDFILE_STATEMENT ||
	  EXPR_CODE(x) == F_REWIND_STATEMENT ||
	  EXPR_CODE(x) == F_INQUIRE_STATEMENT) ? 1 : 0;
}

/* 
 * called before every statement 
 */
/**
 * Check an expression for previously opened XMP clausure.
 *
 *	@param [in] st_no	A statement number.
 *	@param [in] x		An expression.
 *
 *	@retval	0
 *		The caller doesn't need to perform futher compilation of he x.
 *	@retval 1
 *		The caller need to perform futher compilation of he x.
 */
int check_for_XMP_pragma(int st_no, expr x)
{
  expv statements;
  int ret = 1;

  if(XMP_do_required){
    if(EXPR_CODE(x) != F_DO_STATEMENT)
      error("XMP LOOP directives must be followed by do statement");
    XMP_do_required = FALSE;
    ret = 0;
    goto done;
  }
  
  if(XMP_gmove_required){
    if(EXPR_CODE(x) != F_LET_STATEMENT)
      error("XMP GMOVE directives must be followed by assignment");
    XMP_gmove_required = FALSE;
    ret = 0;
    goto done;
  }

  if (XMP_io_desired_statements > 0) {
      if (isIOStatement(x) != 1) {
	  error("XMP IO directives must be followed by I/O statements.");
	  XMP_io_desired_statements = 0;
	  ret = 0;
	  goto done;
      }
  }

  /* check DO directive, close it */
  if(CTL_TYPE(ctl_top) == CTL_XMP &&
     CTL_XMP_ARG_DIR(ctl_top) == XMP_LOOP){
    statements = CURRENT_STATEMENTS;
    if(EXPR_CODE(statements) == LIST) {
      if(LIST_NEXT(EXPR_LIST(statements)) != NULL)
	fatal("XMP_LOOP: bad statements\n");
      statements = EXPR_ARG1(statements);
    }
    if(EXPR_CODE(statements) != F_DO_STATEMENT)
      error("DO LOOP dirctives must be followed by do statements");
    CTL_BLOCK(ctl_top) = 
      XMP_pragma_list(XMP_LOOP,CTL_XMP_ARG_CLAUSE(ctl_top),
		      statements);
    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_XMP_ARG(ctl_top));
    pop_ctl();
    ret = 1;
    goto done;
  }

  if (XMP_io_desired_statements == 1) {
      ret = close_XMP_IO_closure(st_no, x);
  }

done:
  return ret;
}

/*
 * Close XMP_{MASTER|GLOBAL}_IO_BEGIN closure.
 */
static int
close_XMP_IO_closure(int st_no, expr x) {
    int ret = 1;
    extern ID this_label;

    if (CTL_TYPE(ctl_top) == CTL_XMP &&
	(CTL_XMP_ARG_DIR(ctl_top) == XMP_MASTER_IO_BEGIN ||
	 CTL_XMP_ARG_DIR(ctl_top) == XMP_GLOBAL_IO_BEGIN)) {

	enum XMP_pragma p = XMP_DIR_END;
	expv arg = NULL;

	switch (CTL_XMP_ARG_DIR(ctl_top)) {
	    case XMP_MASTER_IO_BEGIN:
		p = XMP_MASTER_IO;
		arg = list0(LIST);	/* dummy */
		break;
	    case XMP_GLOBAL_IO_BEGIN:
		p = XMP_GLOBAL_IO;
		arg = list1(
		    LIST,
		    expv_int_term(
			INT_CONSTANT, NULL,
			EXPR_INT(EXPR_ARG2(CTL_XMP_ARG_CLAUSE(ctl_top)))));
		break;
	    default:
		fatal("must not happen.");
		break;
	}

	if (st_no > 0) {
	    if (LAB_TYPE(this_label) != LAB_FORMAT) {
		output_statement(list1(STATEMENT_LABEL,
				       ID_ADDR(this_label)));
	    }
	} else {
	    this_label = NULL;
	}
	CTL_BLOCK(ctl_top) =
	    XMP_pragma_list(p, arg,
			    (XMP_io_desired_statements > 1) ?
			    x : list1(LIST, x));
	EXPR_LINE(CTL_BLOCK(ctl_top)) = (XMP_io_desired_statements > 1) ?
	    current_line : EXPR_LINE(CTL_XMP_ARG(ctl_top));
	pop_ctl();
	XMP_io_desired_statements = 0;

	ret = 0;
    }

    return ret;
}

expv XMP_check_TASK(expr x)
{
    expr xx;
    expv task_list,current_task;
    list lp;
    
    if(x == NULL) return NULL;
    if(EXPR_CODE(x) != LIST) fatal("XMP_check_SECION: not LIST");

    task_list = EMPTY_LIST;
    current_task = NULL;
    FOR_ITEMS_IN_LIST(lp,x){
	xx = LIST_ITEM(lp);
	if(EXPR_CODE(xx) == XMP_PRAGMA &&
	   EXPR_INT(EXPR_ARG1(xx)) == XMP_TASK){
	    if(current_task != NULL)
		task_list = list_put_last(task_list,current_task);
	    current_task = EMPTY_LIST;
	    continue;
	}
	if(current_task == NULL){
	    error_at_node(xx,"statement is not in any TASK");
	    return NULL;
	}
	current_task = list_put_last(current_task,xx);
    }
    task_list = list_put_last(task_list,current_task);
    return task_list;
}


#ifdef not

void check_XMP_loop_var(SYMBOL do_var_sym)
{
    CTL *cp;
    expr x,c;
    list lp,lq;
    enum XMP_pragma_clause cdir;

    /* clause to be inserted. */
    c = list2(LIST,expv_int_term(INT_CONSTANT,NULL,XMP_DATA_PRIVATE),
	      list1(LIST,expv_sym_term(IDENT,NULL,do_var_sym)));

    /* find any data attribute clauses on do_var_sym */
    for(cp = ctl_top; cp >= ctls; cp--){
	if(CTL_TYPE(cp) != CTL_OMP) continue;
	if(CTL_XMP_ARG_DCLAUSE(cp) != NULL){
	    FOR_ITEMS_IN_LIST(lp,CTL_XMP_ARG_DCLAUSE(cp)){
		x = LIST_ITEM(lp); 
		cdir = (enum XMP_pragma_clause) EXPR_INT(EXPR_ARG1(x));
		if(IS_XMP_DATA_CLAUSE(cdir)){
		    if(EXPR_ARG2(x) == NULL) continue;
		    FOR_ITEMS_IN_LIST(lq,EXPR_ARG2(x)){
			x = LIST_ITEM(lq);
			if(EXPR_CODE(x) == IDENT && EXPR_SYM(x) == do_var_sym)
			    goto found;
		    }
		}
	    }
	}
	/* check on PCLAUSE */
	if(CTL_XMP_ARG_PCLAUSE(cp) != NULL){
	    FOR_ITEMS_IN_LIST(lp,CTL_XMP_ARG_PCLAUSE(cp)){
		x = LIST_ITEM(lp); 
		cdir = (enum XMP_pragma_clause) EXPR_INT(EXPR_ARG1(x));
		if(IS_XMP_DATA_CLAUSE(cdir)){
		    if(EXPR_ARG2(x) == NULL) continue;
		    FOR_ITEMS_IN_LIST(lq,EXPR_ARG2(x)){
			x = LIST_ITEM(lq);
			if(EXPR_CODE(x) == IDENT && EXPR_SYM(x) == do_var_sym)
			    goto found;
		    }
		}
	    }
	}
	
	/* not found, then make loop variable private in parallel region */
	switch(CTL_XMP_ARG_DIR(cp)){
	case XMP_F_PARALLEL:
	case XMP_F_PARALLEL_DO:
	case XMP_F_PARALLEL_SECTIONS:
	    if(CTL_XMP_ARG_PCLAUSE(cp) == NULL)
		CTL_XMP_ARG_PCLAUSE(cp) = list1(LIST,c);
	    else 
		list_put_last(CTL_XMP_ARG_PCLAUSE(cp),c);
	    return;
	}
    }
    return; /* nothing to do, not in parallel region. */

found:
    if(cdir == XMP_DATA_PRIVATE || 
       cdir == XMP_DATA_FIRSTPRIVATE || 
       cdir == XMP_DATA_LASTPRIVATE)
	return; /* already private */
    
    if(IS_XMP_REDUCTION_DATA_CLAUSE(cdir)){
	error("loop control variable must not be OpenMP induction variable");
	return;
    }
    
    if(ctl_top == cp){
	error("parallel loop control variable is declared as SHARED");
	return;
    }

    /* check where parallel loop or not */
    /* if loop var of parallel loop, it is forced to be private */
    if(CTL_TYPE(ctl_top) == CTL_OMP && 
       (CTL_XMP_ARG_DIR(ctl_top) == XMP_F_DO ||
	CTL_XMP_ARG_DIR(ctl_top) == XMP_F_PARALLEL_DO)){
	    if(CTL_XMP_ARG_DCLAUSE(cp) == NULL)
		CTL_XMP_ARG_DCLAUSE(cp) = list1(LIST,c);
	    else 
		list_put_last(CTL_XMP_ARG_DCLAUSE(cp),c);
    }
}

#endif
