/* 
 * $TSUKUBA_Release: Omni XcalableMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "F-front.h"

expv XMP_check_TASK(expr x);
expv XMP_pragma_list(enum XMP_pragma pragma,expv arg1,expv arg2);
static int close_XMP_IO_closure(int st_no, expr x);
int check_for_XMP_pragma(int st_no, expr x);

// void compile_XMP_name_list(expr x);

int XMP_do_required;
int XMP_gmove_required;
int XMP_io_desired_statements = 0;
expv XMP_gmove_directive;

typedef enum _xmp_list_context {
    XMP_LIST_NODES,
    XMP_LIST_ON_REF,
    XMP_LIST_TEMPLATE,
    XMP_LIST_DISTRIBUTE,
    XMP_LIST_ALIGN,
    XMP_LIST_SHADOW,
    XMP_LIST_ID_LIST,
    XMP_LIST_END
} xmp_list_context;

expv XMP_compile_subscript_list(expr list,xmp_list_context context);
expv XMP_compile_ON_ref(expr x);
expv XMP_compile_clause_opt(expr x);

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

expv XMP_gmove_statement(expv v){
    return XMP_pragma_list(XMP_GMOVE,EXPR_ARG2(XMP_gmove_directive),v);
}

/*
 * called from Parser
 */
void compile_XMP_directive(expr x)
{
    expr dir;
    expr c, x1,x2,x3,x4;

    if(x == NULL) return;	/* error */

    if(debug_flag){
	printf("XMP_directive:\n");
	expv_output(x,stdout);
	printf("\n");
    }

    if(XMP_do_required){
	error("XcalableMP LOOP directives must be followed by do statement");
	XMP_do_required = FALSE;
	return;
    }

    if(XMP_gmove_required){
	error("XcalableMP GMOVE directives must be followed by assignment");
	XMP_gmove_required = FALSE;
	return;
    }

    dir = EXPR_ARG1(x);  /* direcive name */
    c = EXPR_ARG2(x);

    if(EXPR_INT(dir) != XMP_END_TASKS && 
       EXPR_INT(dir) != XMP_END_TASK)
	check_for_XMP_pragma(-1,NULL);  /* close DO directives if any */

    switch(EXPR_INT(dir)){
    case XMP_NODES:
      check_INDCL();
      /* check arg: (nameNames, nodeSizeList, inherit) */
      x1 = EXPR_ARG1(c); /* indent */
      x2 = XMP_compile_subscript_list(EXPR_ARG2(c),XMP_LIST_NODES);
      x3 = XMP_compile_ON_ref(EXPR_ARG3(c));
      c = list3(LIST,x1,x2,x3);
      output_statement(XMP_pragma_list(XMP_NODES,c,NULL));
      break;

    case XMP_TEMPLATE:
      check_INDCL();
      /* check arg: (templateNameList, templateSpecList) */
      x1 = EXPR_ARG1(c); /* name list */
      x2 = XMP_compile_subscript_list(EXPR_ARG2(c),XMP_LIST_TEMPLATE);
      c = list2(LIST,x1,x2);
      output_statement(XMP_pragma_list(XMP_TEMPLATE,c,NULL));
      break;

    case XMP_DISTRIBUTE:
      check_INDCL();
      /* check arg: (templateNameList, dist_fmt_list, nodes_ident) */
      x1 = EXPR_ARG1(c); /* name list */
      x2 = XMP_compile_subscript_list(EXPR_ARG2(c),XMP_LIST_DISTRIBUTE);
      x3 = EXPR_ARG3(c);
      c = list3(LIST,x1,x2,x3);
      output_statement(XMP_pragma_list(XMP_DISTRIBUTE,c,NULL));
      break;

    case XMP_ALIGN:
      check_INDCL();
      /* check arg: (arrayNameList, alignSrcList,templateName, alignSubsript) */
      x1 = EXPR_ARG1(c); /* arrayNameList */
      x2 = XMP_compile_subscript_list(EXPR_ARG2(c),XMP_LIST_ID_LIST);
      x3 = EXPR_ARG3(c); /* templateName */
      x4 = XMP_compile_subscript_list(EXPR_ARG4(c),XMP_LIST_ALIGN);
      c = list4(LIST,x1,x2,x3,x4);
      output_statement(XMP_pragma_list(XMP_ALIGN,c,NULL));
      break;

    case XMP_SHADOW:
      check_INDCL();
      /* check arg: (arrayName, shadowWidthList) */
      x1 = EXPR_ARG1(c);
      x2 = XMP_compile_subscript_list(EXPR_ARG2(c),XMP_LIST_SHADOW);
      c = list2(LIST,x1,x2);
      output_statement(XMP_pragma_list(XMP_SHADOW,c,NULL));
      break;

    case XMP_TASK:
      check_INEXEC();
      /* check arg: node_ref opt */
      x1 = XMP_compile_ON_ref(EXPR_ARG1(c));
      x2 = XMP_compile_clause_opt(EXPR_ARG2(c));
      c = list2(LIST,x1,x2);
      push_ctl(CTL_XMP);
      CTL_XMP_ARG(ctl_top) = XMP_pragma_list(XMP_TASK,c,NULL);
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
      /* check arg: no arg */
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
      /* check arg: (index_list on_ref reduction_opt opt)  */
      x1 = XMP_compile_subscript_list(EXPR_ARG1(c),XMP_LIST_ID_LIST);
      x2 = XMP_compile_ON_ref(EXPR_ARG2(c));
      x3 = EXPR_ARG3(c);
      x4 = EXPR_ARG4(c);
      c = list4(LIST,x1,x2,x3,x4);
      push_ctl(CTL_XMP);
      CTL_XMP_ARG(ctl_top) = XMP_pragma_list(XMP_LOOP,c,NULL);;
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
    goto done;
  }
  
  if(XMP_gmove_required){
    if(EXPR_CODE(x) != F_LET_STATEMENT)
      error("XcalableMP GMOVE directives must be followed by assignment");
    goto done;
  }

  if (XMP_io_desired_statements > 0 && x != NULL) {
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
    goto done;
  }

  if (XMP_io_desired_statements == 1 && x != NULL) {
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


expv XMP_compile_subscript_list(expr l,xmp_list_context context)
{
    expr x;
    list lp;
    expv v,v1,v2,ret_list;
    
    ret_list = EMPTY_LIST;
    FOR_ITEMS_IN_LIST(lp,l){
	x = LIST_ITEM(lp);
	v = v1 = v2 = NULL;
	switch(context){
	case XMP_LIST_NODES: /* element must be integer scalar or * */
	    if(x != NULL) { /* must a list */
		if(EXPR_ARG1(x) == NULL){
		    error("bad subscript in nodes directive");
		    break;
		}
		if(EXPR_ARG2(x) == NULL &&  EXPR_ARG3(x) == NULL){
		    v = compile_expression(EXPR_ARG1(x));
		    if(!IS_INT_CONST_V(v))
			error("subscript in nodes must be an integer constant");
		} else 
		    error("bad subscript in nodes directive");
	    }
	    break;
	case XMP_LIST_ON_REF: /* expr, triplet, * */
	    if(x != NULL){
		v = NULL;
		if(EXPR_ARG1(x) != NULL)
		    v = compile_expression(EXPR_ARG1(x));
		if(EXPR_ARG2(x) != NULL)
		    v1 = compile_expression(EXPR_ARG2(x));
		if(EXPR_ARG3(x) != NULL)
		    v2 = compile_expression(EXPR_ARG3(x));
		if(v1 != NULL || v2 != NULL)
		    v = list3(LIST,v,v1,v2);
	    }
	    break;
	case XMP_LIST_TEMPLATE: /* int-expr [: int-expr], or : */
	    if(x == NULL)
		error("subscript in template must not be *");
	    else {
		if(EXPR_ARG3(x) != NULL)
		    error("subscript in template cannot have step");
		if(EXPR_ARG1(x) != NULL){
		    v = compile_expression(EXPR_ARG1(x));
		} else {
		    if(EXPR_ARG2(x) != NULL){
			error("bad subscript in template");
		    }  else {
			/* both arg1 and arg2 are null, ':' */
			v = list1(LIST,NULL);
		    }
		    break;
		}
		if(EXPR_ARG2(x) != NULL){
		    v1 = compile_expression(EXPR_ARG2(x));
		    v = list3(LIST,v,v1,NULL);
		}
	    }
	    break;
	case XMP_LIST_DISTRIBUTE: /* * or (id expr) */
	    if(x != NULL){
		if(EXPR_ARG2(x) != NULL){
		    v = list2(LIST,EXPR_ARG1(x),
			      compile_expression(EXPR_ARG2(x)));
		} else v = x;
	    }
	    break;
	case XMP_LIST_ID_LIST: /* id, or *,: */
	    if(x != NULL){
		if(EXPR_ARG2(x) == NULL &&  EXPR_ARG3(x) == NULL){
		    x = EXPR_ARG1(x);
		    if(x == NULL){
			v = list1(LIST,NULL);
			break;
		    } else if(EXPR_CODE(x) == IDENT){
			v = x;
			break;
		    }
		}
		error("susbscript in align source or index must be identifier or ':'");
	    }
	    break;
	case XMP_LIST_ALIGN:  /* expr=v+off, or *, : */
	    if(x != NULL){
		if(EXPR_ARG2(x) == NULL &&  EXPR_ARG3(x) == NULL){
		    x = EXPR_ARG1(x);
		    if(x == NULL){
			v = list1(LIST,NULL);
			break;
		    } else {
			switch(EXPR_CODE(x)){
			case IDENT:
			    v = x;
			    break;
			case F_PLUS_EXPR:
			case F_MINUS_EXPR:
			    if(EXPR_CODE(EXPR_ARG1(x)) != IDENT)
				error("left expression must be identifier in align target");
			    v = expv_cons(EXPR_CODE(x) == F_PLUS_EXPR ?
					  PLUS_EXPR:MINUS_EXPR,
					  type_INT,
					  EXPR_ARG1(x),
					  compile_expression(EXPR_ARG2(x)));
			    break;
			default:
			    error("bad expression in align target");
			}
			break;
		    }
		    error("subscript in align target must be expression or ':'");
		}
		error("susbscript in align target must be an expression or ':'");
	    }
	    break;

	case XMP_LIST_SHADOW: /* expr expr:expr or * */
	    if(x != NULL){
		if(EXPR_ARG3(x) != NULL){
		    error("bad subscript in shadow");
		    break;
		}
		if(EXPR_ARG1(x) != NULL){
		    v = compile_expression(EXPR_ARG1(x));
		    if(EXPR_ARG2(x) != NULL){
			v2 = compile_expression(EXPR_ARG2(x));
			v = list3(LIST,v,v2,NULL);
		    } 
		    break;
		}
		error("bad subscript in shadow");
	    }
	    break;

	default:
	    fatal("XMP_compile_subscript_list: unknown context");
	}
	ret_list = list_put_last(ret_list,v);
    }
    return ret_list;
}

/*
 * check node ref
 */
expv XMP_compile_ON_ref(expr x)
{
    expv v;
    if(x == NULL) return NULL;
    v = XMP_compile_subscript_list(EXPR_ARG2(x),XMP_LIST_ON_REF);
    return list2(LIST,EXPR_ARG1(x),v);
}

expv XMP_compile_clause_opt(expr x)
{
    return x; /* nothing at this moment */
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
