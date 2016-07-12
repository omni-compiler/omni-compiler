#include "F-front.h"

extern CTL *ctl_top_saved;
extern expv CURRENT_STATEMENTS_saved;

expv OMP_check_SECTION(expr x);
expv OMP_pragma_list(enum OMP_pragma pragma,expv arg1,expv arg2);
expv OMP_FOR_pragma_list(expv clause,expv statements);
expv OMP_atomic_statement(expv v);

void compile_OMP_name_list(expr x);
void compile_OMP_pragma_clause(expr x, int pragma, int is_parallel,
			  expv *pc,expv *dc);

void check_for_OMP_pragma(expr x);
int check_for_XMP_pragma(int st_no, expr x);
void check_for_ACC_pragma(expr x);
int is_ACC_loop_pragma(expv x);

static int OMP_do_required;

enum OMP_st_pragma {
    OMP_ST_NONE,
    OMP_ST_ATOMIC,
    OMP_ST_END
};

static enum OMP_st_pragma OMP_st_required, OMP_st_flag;

int OMP_reduction_op(expr v)
{
    char *s;
    if(EXPR_CODE(v) != IDENT) fatal("OMP_reduction_op: no IDENT");
    s = SYM_NAME(EXPR_SYM(v));
    if(strcmp("max",s) == 0) return (int)OMP_DATA_REDUCTION_MAX;
    if(strcmp("min",s) == 0) return (int)OMP_DATA_REDUCTION_MIN;
    if(strcmp("iand",s) == 0) return (int)OMP_DATA_REDUCTION_BITAND;
    if(strcmp("ior",s) == 0) return (int)OMP_DATA_REDUCTION_BITOR;
    if(strcmp("ieor",s) == 0) return (int)OMP_DATA_REDUCTION_BITXOR;

    error("bad intrinsic function in REDUCTION clause of OpenMP");
    return OMP_DATA_REDUCTION_PLUS;	/* dummy */
}

int OMP_depend_op(expv v)
{
    char *s;
    if(EXPR_CODE(v) != IDENT) fatal("OMP_depend_op: no IDENT");
    s = SYM_NAME(EXPR_SYM(v));
    if(strcmp("in",s) == 0) return (int)OMP_DATA_DEPEND_IN;
    if(strcmp("out",s) == 0) return (int)OMP_DATA_DEPEND_OUT;
    if(strcmp("inout",s) == 0) return (int)OMP_DATA_DEPEND_INOUT;
    error("bad intrinsic function in REPEND clause of OpenMP");
    return OMP_DATA_DEPEND_IN;     /* dummy */
}
void init_for_OMP_pragma()
{
    OMP_do_required = FALSE;
    OMP_st_required = OMP_ST_NONE;
}

void check_for_OMP_pragma(expr x)
{
    if(OMP_do_required){
        if (EXPR_CODE(x) == XMP_PRAGMA && EXPR_INT(EXPR_ARG1(x)) == XMP_LOOP) return;
        if (is_ACC_loop_pragma(x)) return;
	if(EXPR_CODE(x) != F_DO_STATEMENT)
	    error("OpenMP DO directives must be followed by do statement");
	OMP_do_required = FALSE;
	return;
    }
    if(OMP_st_required != OMP_ST_NONE){
	if(EXPR_CODE(x) != F_LET_STATEMENT)
	    error("OpenMP ATOMIC directives must be followed by assignment");
	return;
    }

    /* check DO directive, close it */
    if(CTL_TYPE(ctl_top) == CTL_OMP &&
       (CTL_OMP_ARG_DIR(ctl_top) == OMP_F_DO ||
	CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_DO)){
	if(CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_DO){
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_PARALLEL,CTL_OMP_ARG_PCLAUSE(ctl_top),
				OMP_FOR_pragma_list(
				    CTL_OMP_ARG_DCLAUSE(ctl_top),
				    CURRENT_STATEMENTS));
	} else {
	    CTL_BLOCK(ctl_top) = 
		OMP_FOR_pragma_list(CTL_OMP_ARG_DCLAUSE(ctl_top),
				    CURRENT_STATEMENTS);
	}
	EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	pop_ctl();
    }
}

void OMP_check_LET_statement()
{
    OMP_st_flag = OMP_st_required;
    OMP_st_required = OMP_ST_NONE;
}

int OMP_output_st_pragma(expv v)
{
    switch(OMP_st_flag){
    case OMP_ST_ATOMIC:
	output_statement(OMP_pragma_list(OMP_ATOMIC, list0(LIST), 
					 list1(EXPR_STATEMENT, v)));
	return TRUE;
    default:
	return FALSE;
    }
}
    
expv OMP_pragma_list(enum OMP_pragma pragma,expv arg1,expv arg2)
{
    return list3(OMP_PRAGMA,expv_int_term(INT_CONSTANT,NULL,(int)pragma),
		 arg1,arg2);
}

expv OMP_FOR_pragma_list(expv clause,expv statements)
{
    list lp;

#ifdef not
    if(EXPR_CODE(statements) == F_DO_STATEMENT) 
	return OMP_pragma_list(OMP_FOR,clause,statements);
    else {
	error_at_node(clause,"OpenMP DO directive must be followed by DO");
	return NULL;
    }
#endif

    if(EXPR_CODE(statements) != LIST) 
	fatal("OMP_FOR_pragma_list: unknown list");
    FOR_ITEMS_IN_LIST(lp,statements){
        expv xx = LIST_ITEM(lp);
	if(EXPR_CODE(xx) == F_DO_STATEMENT || 
	   is_ACC_loop_pragma(xx)){
	    LIST_ITEM(lp) = OMP_pragma_list(OMP_FOR,clause,xx);
	    break;
	}
    }
    return statements;
}


void compile_OMP_directive(expr x)
{
    expr dir;
    expr c = NULL;
    expv pclause,dclause;

    if(x == NULL) return;	/* error */

    if (debug_flag) {
	fprintf(stderr, "OMP_directive:\n");
	expv_output(x, stderr);
	fprintf(stderr, "\n");
    }

    check_for_ACC_pragma(x);
    check_for_OMP_pragma(x);
    check_for_XMP_pragma(-1, x);

    if(OMP_do_required){
	error("OpenMP DO directived must be followed by do statement");
	OMP_do_required = FALSE;
	return;
    }

    if(OMP_st_required != OMP_ST_NONE){
	error("OpenMP ATOMIC directives must be followed by assignment");
	return;
    }

    dir = EXPR_ARG1(x);

    if (EXPR_INT(dir) == OMP_F_THREADPRIVATE) {
        check_INDCL();
    } else {
        check_INEXEC();
    }

    if (EXPR_INT(dir) != OMP_F_END_DO &&
        EXPR_INT(dir) != OMP_F_END_PARALLEL_DO &&
        EXPR_INT(dir) != OMP_F_ATOMIC) {
	check_for_OMP_pragma(NULL);  /* close DO directives if any */
    }

    switch(EXPR_INT(dir)){
    case OMP_F_PARALLEL:
	push_ctl(CTL_OMP);
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_PARALLEL,TRUE,
				  &pclause,&dclause);
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	return;
    case OMP_F_END_PARALLEL:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL){
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_PARALLEL,CTL_OMP_ARG_PCLAUSE(ctl_top),
				CURRENT_STATEMENTS);
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	} else  error("OpenMP PARALLEL block is not closed");
	return;

    case OMP_F_PARALLEL_DO:
	push_ctl(CTL_OMP);
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_FOR,TRUE,
				  &pclause,&dclause);
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	OMP_do_required = TRUE;
	return;
    case OMP_F_END_PARALLEL_DO:
/* 	if(CTL_TYPE(ctl_top) == CTL_OMP && */
/* 	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_DO){ */
/* 	    CTL_BLOCK(ctl_top) =  */
/* 		OMP_pragma_list(OMP_PARALLEL,CTL_OMP_ARG_PCLAUSE(ctl_top), */
/* 				OMP_FOR_pragma_list( */
/* 				    CTL_OMP_ARG_DCLAUSE(ctl_top), */
/* 				    CURRENT_STATEMENTS)); */
/* 	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top)); */
/* 	    pop_ctl(); */
/* 	} else  error("OpenMP PARALLEL DO block is not closed"); */
	return;
	
    case OMP_F_DO:
	push_ctl(CTL_OMP);
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_FOR,FALSE,
				  &pclause,&dclause);
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	OMP_do_required = TRUE;
	return;
    case OMP_F_END_DO:
        /* OMP_F_DO has been already closed at F_ENDDO */
        /* Here, only the nowait clause is handled. */
      if (ctl_top_saved){
	dclause = CTL_OMP_ARG_DCLAUSE(ctl_top_saved);
	if (EXPR_ARG2(x) != NULL) list_put_last(dclause, EXPR_ARG2(x));
	CTL_BLOCK(ctl_top_saved) = OMP_FOR_pragma_list(dclause, CURRENT_STATEMENTS_saved);
	EXPR_LINE(CTL_BLOCK(ctl_top_saved)) = EXPR_LINE(CTL_OMP_ARG(ctl_top_saved));
	ctl_top_saved = NULL;
      }

	/* if(CTL_TYPE(ctl_top) == CTL_OMP && */
	/*    CTL_OMP_ARG_DIR(ctl_top) == OMP_F_DO){ */
	/*     dclause = CTL_OMP_ARG_DCLAUSE(ctl_top); */
	/*     if(EXPR_ARG2(x) != NULL) list_put_last(dclause,EXPR_ARG2(x)); */
	/*     CTL_BLOCK(ctl_top) =  */
	/* 	OMP_FOR_pragma_list(dclause,CURRENT_STATEMENTS); */
	/*     EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top)); */
	/*     pop_ctl(); */
	/* } else error("OpenMP DO block is not closed"); */

      return;
	
    case OMP_F_PARALLEL_SECTIONS:
	push_ctl(CTL_OMP);
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SECTIONS,TRUE,
				  &pclause,&dclause);
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	return;
    case OMP_F_END_PARALLEL_SECTIONS:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_SECTIONS){
	    CURRENT_STATEMENTS = OMP_check_SECTION(CURRENT_STATEMENTS);
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_PARALLEL,CTL_OMP_ARG_PCLAUSE(ctl_top),
				OMP_pragma_list(OMP_SECTIONS,
						CTL_OMP_ARG_DCLAUSE(ctl_top),
						CURRENT_STATEMENTS));
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	} else  error("OpenMP PARALLEL SECTIONS block is not closed");
	return;

    case OMP_F_SECTIONS:
	push_ctl(CTL_OMP);
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SECTIONS,FALSE,
				  &pclause,&dclause);
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	return;
    case OMP_F_END_SECTIONS:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_SECTIONS){
	    CURRENT_STATEMENTS = OMP_check_SECTION(CURRENT_STATEMENTS);
	    dclause = CTL_OMP_ARG_DCLAUSE(ctl_top);
	    if(EXPR_ARG2(x) != NULL) list_put_last(dclause,EXPR_ARG2(x));
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_SECTIONS,dclause,CURRENT_STATEMENTS);
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	} else  error("OpenMP SECTIONS block is not closed");
	return;
	
    case OMP_F_SECTION:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   (CTL_OMP_ARG_DIR(ctl_top) == OMP_F_SECTIONS ||
	    CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_SECTIONS)){
	    output_statement(OMP_pragma_list(OMP_SECTION,NULL,NULL));
	} else error("OpenMP SECTION appears outside SECTOINS"); 
	return;
	
    case OMP_F_SINGLE:
	push_ctl(CTL_OMP);
	compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SINGLE,FALSE,
				  &pclause,&dclause);
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	return;
	
    case OMP_F_END_SINGLE:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_SINGLE){
	    dclause = CTL_OMP_ARG_DCLAUSE(ctl_top);
	    //if(EXPR_ARG2(x) != NULL) list_put_last(dclause,EXPR_ARG2(x));
	    if (EXPR_ARG2(x) != NULL){
	      list lp;
	      FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(x)){
		list_put_last(dclause, LIST_ITEM(lp));
	      }
	    }
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_SINGLE,dclause,CURRENT_STATEMENTS);
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	} else  error("OpenMP SINGLE block is not closed");
	return;

    case OMP_F_MASTER:
    case OMP_F_ORDERED:
	push_ctl(CTL_OMP);
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,NULL,NULL);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	return;
	
    case OMP_F_END_MASTER:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_MASTER){
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_MASTER, list0(LIST), CURRENT_STATEMENTS);
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	} else  error("OpenMP MASTER block is not closed");
	return;
	
    case OMP_F_END_ORDERED:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_ORDERED){
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_ORDERED, list0(LIST), CURRENT_STATEMENTS);
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	} else  error("OpenMP ORDERED block is not closed");
	return;
	
    case OMP_F_CRITICAL:
	push_ctl(CTL_OMP);
	c = EXPR_ARG2(x);
	if(c != NULL && LIST_NEXT(EXPR_LIST(c)) != NULL){
	    fatal("bad critical section name");
	    break;
	}
	CTL_OMP_ARG(ctl_top) = list3(LIST,dir,c,NULL);
	EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
	return;
	
    case OMP_F_END_CRITICAL:
	if(CTL_TYPE(ctl_top) == CTL_OMP &&
	   CTL_OMP_ARG_DIR(ctl_top) == OMP_F_CRITICAL &&
	   ((EXPR_ARG2(x) == NULL &&
	     EXPR_ARG2(CTL_OMP_ARG(ctl_top)) == NULL) ||
	    (EXPR_ARG2(x) != NULL &&
	     EXPR_ARG2(CTL_OMP_ARG(ctl_top)) != NULL &&
	     EXPR_SYM(EXPR_ARG1(EXPR_ARG2(x))) == 
	     EXPR_SYM(EXPR_ARG1(EXPR_ARG2(CTL_OMP_ARG(ctl_top))))))){
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_CRITICAL,EXPR_ARG2(CTL_OMP_ARG(ctl_top)),
				CURRENT_STATEMENTS);
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	} else  error("OpenMP CRITICAL block is not closed");
	return;
	
    case OMP_F_BARRIER:
	output_statement(OMP_pragma_list(OMP_BARRIER, list0(LIST), NULL));
	return;

#ifdef notyet
    case OMP_F_TASKWAIT:
	output_statement(OMP_pragma_list(OMP_TASKWAIT, list0(LIST), NULL));
	return;
#endif /* notyet */

    case OMP_F_FLUSH:
	c = EXPR_ARG2(x);
	compile_OMP_name_list(c);
	output_statement(OMP_pragma_list(OMP_FLUSH,c,NULL));
	return;

    case OMP_F_THREADPRIVATE:
	c = EXPR_ARG2(x);
	compile_OMP_name_list(c);
	output_statement(OMP_pragma_list(OMP_THREADPRIVATE,c,NULL));
      //warning("OMP THREADPRIVATE not supported yet");
	return;

    case OMP_F_WORKSHARE:
      warning("OMP WORKSHARE not supported yet");
      return;

    case OMP_F_END_WORKSHARE:
      //warning("OMP END WORKSHARE not supported yet");
      return;

    case OMP_F_PARALLEL_WORKSHARE:
      warning("OMP PARALLEL WORKSHARE not supported yet");
      return;

    case OMP_F_END_PARALLEL_WORKSHARE:
      //warning("OMP END PARALLEL WORKSHARE not supported yet");
      return;

    case OMP_F_ATOMIC:
	OMP_st_required = OMP_ST_ATOMIC;
	break;

    case OMP_F_TASK:
      push_ctl(CTL_OMP);
      compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_TASK,FALSE,
				&pclause,&dclause);
      CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
      EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
      return;
    case OMP_F_END_TASK:
      if(CTL_TYPE(ctl_top) == CTL_OMP &&
         CTL_OMP_ARG_DIR(ctl_top) == OMP_F_TASK){
        CTL_BLOCK(ctl_top) =
          OMP_pragma_list(OMP_TASK,CTL_OMP_ARG_DCLAUSE(ctl_top),
                          CURRENT_STATEMENTS);
        EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
        pop_ctl();
      } else  error("OpenMP TASK block is not closed");
      return;

    case OMP_F_SIMD:
      push_ctl(CTL_OMP);
      compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SIMD,TRUE,
				&pclause,&dclause);
      CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
      EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
      return;

    case OMP_F_END_SIMD:
      if(CTL_TYPE(ctl_top) == CTL_OMP &&
         CTL_OMP_ARG_DIR(ctl_top) == OMP_F_SIMD){
        CTL_BLOCK(ctl_top) =
          OMP_pragma_list(OMP_SIMD,CTL_OMP_ARG_DCLAUSE(ctl_top),
                          CURRENT_STATEMENTS);
        EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
        pop_ctl();
      } else  error("OpenMP SIMD block is not closed");
      return;
    
    case OMP_F_DO_SIMD:
      push_ctl(CTL_OMP);
      compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SIMD,TRUE,
				&pclause,&dclause);
      CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
      EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
      return;
    case OMP_F_END_DO_SIMD:
      if(CTL_TYPE(ctl_top) == CTL_OMP &&
         CTL_OMP_ARG_DIR(ctl_top) == OMP_F_DO_SIMD){
	CTL_BLOCK(ctl_top) =
	  OMP_pragma_list(OMP_FOR,CTL_OMP_ARG_DCLAUSE(ctl_top),
			  OMP_pragma_list(OMP_SIMD,
					  CTL_OMP_ARG_DCLAUSE(ctl_top),
					  CURRENT_STATEMENTS));
	/*        CTL_BLOCK(ctl_top) =
          OMP_pragma_list(OMP_DO_SIMD,CTL_OMP_ARG_DCLAUSE(ctl_top),
	  CURRENT_STATEMENTS);*/
        EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
        pop_ctl();
      } else  error("OpenMP DO SIMD block is not closed");
      return;

    case OMP_F_DECLARE_SIMD:
      push_ctl(CTL_OMP);
      compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SIMD,TRUE,
				&pclause,&dclause);
      CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
      EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
      return;
    case OMP_F_END_DECLARE_SIMD:
      if(CTL_TYPE(ctl_top) == CTL_OMP &&
         CTL_OMP_ARG_DIR(ctl_top) == OMP_F_DECLARE_SIMD){
        CTL_BLOCK(ctl_top) =
          OMP_pragma_list(OMP_DECLARE,CTL_OMP_ARG_DCLAUSE(ctl_top),
                          OMP_pragma_list(OMP_SIMD,
                                          CTL_OMP_ARG_DCLAUSE(ctl_top),
                                          CURRENT_STATEMENTS));
	/*          OMP_pragma_list(OMP_DECLARE_SIMD,CTL_OMP_ARG_DCLAUSE(ctl_top),
		    CURRENT_STATEMENTS);*/
        EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
        pop_ctl();
      } else  error("OpenMP DECLARE SIMD block is not closed");
      return;

    case OMP_F_PARALLEL_DO_SIMD:
      push_ctl(CTL_OMP);
      compile_OMP_pragma_clause(EXPR_ARG2(x),OMP_SIMD,TRUE,
				&pclause,&dclause);
      CTL_OMP_ARG(ctl_top) = list3(LIST,dir,pclause,dclause);
      EXPR_LINE(CTL_OMP_ARG(ctl_top)) = current_line;
      return;

    case OMP_F_END_PARALLEL_DO_SIMD:
      if(CTL_TYPE(ctl_top) == CTL_OMP &&
         CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_DO_SIMD){
        CTL_BLOCK(ctl_top) =
          OMP_pragma_list(OMP_PARALLEL,CTL_OMP_ARG_DCLAUSE(ctl_top),
                          OMP_pragma_list(OMP_FOR,
                                          CTL_OMP_ARG_DCLAUSE(ctl_top),
					  OMP_pragma_list(OMP_SIMD,
							  CTL_OMP_ARG_DCLAUSE(ctl_top),
							  CURRENT_STATEMENTS)));
	/*
          OMP_pragma_list(OMP_PARALLEL_DO_SIMD,CTL_OMP_ARG_PCLAUSE(ctl_top),
	  CURRENT_STATEMENTS);*/
        EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
        pop_ctl();
      } else  error("OpenMP PARALLEL DO SIMD block is not closed");
      return;

    default:
	fatal("unknown OMP pragma");
    }
}

expv OMP_check_SECTION(expr x)
{
    expr xx;
    expv section_list,current_section;
    list lp;
    
    if(x == NULL) return NULL;
    if(EXPR_CODE(x) != LIST) fatal("OMP_check_SECION: not LIST");

    section_list = EMPTY_LIST;
    current_section = NULL;
    FOR_ITEMS_IN_LIST(lp,x){
	xx = LIST_ITEM(lp);
	if(EXPR_CODE(xx) == OMP_PRAGMA &&
	   EXPR_INT(EXPR_ARG1(xx)) == OMP_SECTION){
	    if(current_section != NULL)
		section_list = list_put_last(section_list,current_section);
	    current_section = EMPTY_LIST;
	    continue;
	}
	if(current_section == NULL){
	    error_at_node(xx,"statement is not in any section");
	    return NULL;
	}
	current_section = list_put_last(current_section,xx);
    }
    section_list = list_put_last(section_list,current_section);
    return section_list;
}


/* PARALLEL - private,firstprivate,reduction,default,shared,copyin,if
 * FOR      - private,firstprivate,lastprivate,reduction,ordered,shed,nowait
 * SECTIONS - private,firstprivate,lastprivate,reduction,nowait
 * SINGLE   - private,firstprivate,nowait
 */
void compile_OMP_pragma_clause(expr x, int pragma, int is_parallel,
			  expv *pc,expv *dc)
{
    list lp;
    expr c,v;
    expv pclause,dclause;
    pclause = EMPTY_LIST;
    dclause = EMPTY_LIST;
    if(x == NULL) goto ret; /* empy */

    FOR_ITEMS_IN_LIST(lp,x){
	c = LIST_ITEM(lp);
	switch(EXPR_INT(EXPR_ARG1(c))){
	case OMP_DATA_DEFAULT:	/* default(shared|none|private) */
	  if(!is_parallel){
		error_at_node(x,"'default' clause must be in PARALLEL");
		break;
	    }
	    pclause = list_put_last(pclause,
				    list2(LIST,EXPR_ARG1(c),
					  EXPR_ARG1(EXPR_ARG2(c))));
	    break;
	case OMP_DATA_SHARED:
	    /* all pragma can have these */
	  /* compile_OMP_name_list(EXPR_ARG2a(c));
	    if(pragma == OMP_PARALLEL)
	      pclause = list_put_last(pclause,c);
	    else     
	      dclause = list_put_last(dclause,c);
	    break;
	  */
	  compile_OMP_name_list(EXPR_ARG2(c));
	    if(!is_parallel && pragma!=OMP_TASK){
		error_at_node(x,"'shared' clause must be in PARALLEL");
		break;
	    }
            if(is_parallel){
	    pclause = list_put_last(pclause,c);
	    }
	    else if(pragma == OMP_TASK){
	    dclause = list_put_last(dclause,c);
	    }
	    break;
	  
	case OMP_DATA_COPYIN:
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(!is_parallel){
		error_at_node(x,"'copyin' clause must be in PARALLEL");
		break;
	    }
	    pclause = list_put_last(pclause,c);
	    break;
	case OMP_DIR_IF:
	    if(!is_parallel && pragma != OMP_TASK){
		error_at_node(x,"'if' clause must be in PARALLEL");
		break;
	    }
            if(pragma == OMP_PARALLEL){
	    v = compile_expression(EXPR_ARG2(c));
	    pclause = list_put_last(pclause,
				    list2(LIST,EXPR_ARG1(c),v));
	    }
	    else if(pragma == OMP_TASK){
	      v = compile_expression(EXPR_ARG2(c));
	      dclause = list_put_last(dclause,
				      list2(LIST,EXPR_ARG1(c),v));
	    }
	    break;

	case OMP_DIR_NUM_THREADS:
            if(!is_parallel){
                error_at_node(x,"'num_threads' clause must be in PARALLEL");
                break;
            }
            v = compile_expression(EXPR_ARG2(c));
	    pclause = list_put_last(pclause,
					list2(LIST,EXPR_ARG1(c),v));
            break;
	case OMP_DATA_PRIVATE:
	case OMP_DATA_FIRSTPRIVATE:
	    /* all pragma can have these */
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(pragma == OMP_PARALLEL)
	      pclause = list_put_last(pclause,c);
	    else     
	      dclause = list_put_last(dclause,c);
	    break;


	case OMP_DATA_LASTPRIVATE:
	    compile_OMP_name_list(EXPR_ARG2(c));
	    if(pragma != OMP_FOR && pragma != OMP_SECTIONS){
		error_at_node(x,"'lastprivate' clause must be in FOR or SECTIONS");
		break;
	    }
	    dclause = list_put_last(dclause,c);
	    break;
	case OMP_DATA_DEPEND_IN:
	case OMP_DATA_DEPEND_OUT:
	case OMP_DATA_DEPEND_INOUT:
	  compile_OMP_name_list(EXPR_ARG2(c));
	  if(pragma != OMP_TASK){
	    error_at_node(x,"'depend' clause must be in TASK");
	    break;
	  }
	  dclause = list_put_last(dclause,c);
	  break;
	case OMP_DIR_UNTIED:
	  if(pragma != OMP_TASK){
	    error_at_node(x,"'untied' clause must be in TASK");
	    break;
	  }
	  dclause = list_put_last(dclause,c);
	  break;
	case OMP_DIR_MERGEABLE:
	  if(pragma != OMP_TASK){
	    error_at_node(x,"'mergeable' clause must be in TASK");
		break;
	  }
	  dclause = list_put_last(dclause,c);
	  break;
	case OMP_DATA_FINAL:
	    if(!is_parallel && pragma != OMP_TASK){
		error_at_node(x,"'if' clause must be in PARALLEL");
		break;
	    }
            if(pragma == OMP_PARALLEL){
	    v = compile_expression(EXPR_ARG2(c));
	    pclause = list_put_last(pclause,
				    list2(LIST,EXPR_ARG1(c),v));
	    }
	    else if(pragma == OMP_TASK){
	      v = compile_expression(EXPR_ARG2(c));
	      dclause = list_put_last(dclause,
				      list2(LIST,EXPR_ARG1(c),v));
	    }
	    break;
	    /*
	  if(pragma != OMP_TASK){
		  error_at_node(x,"'final' clause must be in TASK");
		  break;
	  }
	  v = compile_expression(EXPR_ARG2(c));
	  dclause = list_put_last(dclause,
				  list2(LIST,EXPR_ARG1(c),v));
	  break;
	    */
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
	      pclause = list_put_last(pclause,c);
	    else if(pragma == OMP_FOR || pragma == OMP_SECTIONS)
	      dclause = list_put_last(dclause,c);
	    else 
	      error_at_node(x,"'reduction' clause must not be in SINGLE");
	    break;

	case OMP_DIR_ORDERED:
	    if(pragma != OMP_FOR){
		error_at_node(x,"'ordered' clause must be in FOR");
		break;
	    }
	    dclause = list_put_last(dclause,c);
	    break;

	case OMP_DIR_SCHEDULE:
	    if(pragma != OMP_FOR){
		error_at_node(x,"'schedule' clause must be in FOR");
		break;
	    }
	    v = EXPR_ARG2(EXPR_ARG2(c));
	    if(v != NULL){
#ifdef not
		if(EXPR_INT(EXPR_ARG1(EXPR_ARG2(c))) == OMP_SCHED_AFFINITY){
		    list lp; expv vv;
		    extern void TEA_map_expr();
		    FOR_ITEMS_IN_LIST(lp,EXPR_ARG2(v)){
			vv = LIST_ITEM(lp);
			if(vv != NULL && 
			   EXPR_INT(EXPR_ARG1(vv)) == TEA_MAP_EXPR){
			    TEA_map_expr(EXPR_ARG2(vv));
			}
		    }
		} else {
#endif
		    v = compile_expression(v);
		    c = list2(LIST,EXPR_ARG1(c),
			      list2(LIST,EXPR_ARG1(EXPR_ARG2(c)),v));
#ifdef not
		} 
#endif
	    }
	    dclause = list_put_last(dclause,c);
	    break;

	case OMP_DIR_NOWAIT:
	    if(is_parallel){
		error_at_node(x,"'nowait' clause must not be in PARALLEL");
		break;
	    }
	    dclause = list_put_last(dclause,c);
	    break;

	default:
	    fatal("compile_OMP_paragma_clause");
	}
    }
ret:

    /* combination with PARALLEL, don't have to wait */
    if(is_parallel && (pragma != OMP_PARALLEL))
	dclause = 
	    list_put_last(dclause, 
			  list2(LIST,make_enode(INT_CONSTANT,
						((void *)OMP_DIR_NOWAIT)),
				NULL));
    *pc = pclause;
    *dc = dclause;
}

void compile_OMP_name_list(expr x)
{
    list lp;
    expr v;
    ID id;
    //EXT_ID ext_id;

    FOR_ITEMS_IN_LIST(lp,x){
	v = LIST_ITEM(lp);
	if(EXPR_CODE(v) == LIST){
	    /* for reference to common name */
	    v = EXPR_ARG1(v);
	    if(EXPR_CODE(v) != IDENT){
		error("common name is expected in list of OpenMP directive");
		continue;
	    }

	    //if((ext_id = findCommon(v)) == NULL){
	    if ((id = find_common_ident(EXPR_SYM(v))) == NULL){
		error("common name '%s' in OpenMP directive is undefined",
		      SYM_NAME(EXPR_SYM(v)));
		continue;
	    }

	    LIST_ITEM(lp) = expv_sym_term(IDENT, NULL, ID_SYM(id));
	    continue;
	} else if(EXPR_CODE(v) != IDENT){
	    error("variable is expected in list of OpenMP directive");
	    continue;
	}
	id = declare_ident(EXPR_SYM(v),CL_UNKNOWN);
	if(ID_CLASS(id) != CL_VAR){
	    if(ID_CLASS(id) == CL_UNKNOWN) declare_variable(id);
	    else {
		error("identifier '%s' is not a variable in OpenMP directive",
		      ID_NAME(id));
		continue;
	    }
	}
    }
}

struct {
    char *runtime_name;
    BASIC_DATA_TYPE type;
} omp_runtime_table[] = {
    {"omp_set_num_threads",TYPE_SUBR },
    {"omp_get_num_threads",TYPE_INT },
    {"omp_get_max_threads",TYPE_INT },
    {"omp_get_thread_num",TYPE_INT },
    {"omp_get_num_procs",TYPE_INT },
    {"omp_in_parallel",TYPE_LOGICAL },
    {"omp_set_dynamic",TYPE_SUBR },
    {"omp_get_dynmaic",TYPE_LOGICAL },
    {"omp_set_nested",TYPE_SUBR },
    {"omp_get_nested",TYPE_LOGICAL },
    {"omp_init_lock",TYPE_SUBR },
    {"omp_destroy_lock",TYPE_SUBR },
    {"omp_set_lock",TYPE_SUBR },
    {"omp_unset_lock",TYPE_SUBR },
    {"omp_test_lock",TYPE_LOGICAL },
    {NULL}};

void check_OMP_runtime_function(ID id)
{
    char *s,*ss;
    int i;
    TYPE_DESC tp;
    BASIC_DATA_TYPE t;

    if(!OMP_flag) return;

    s = ID_NAME(id);
    if(strncmp(s,"omp_",4) != 0) return;
    for(i = 0; (ss = omp_runtime_table[i].runtime_name) != NULL; i++){
	if(strcmp(s,ss) == 0){
	    t = omp_runtime_table[i].type;
	    if((tp = ID_TYPE(id)) != NULL){
		if(TYPE_BASIC_TYPE(tp) != t)
		    warning("data type of OpenMP runtime function '%s' may be wrong",s);
	    } else ID_TYPE(id) = BASIC_TYPE_DESC(t);
	    break;
	}
    }
}

/* called from F-compile.c, force loop variable to be private. */
void check_OMP_loop_var(SYMBOL do_var_sym)
{
    CTL *cp;
    expr x,c;
    list lp,lq;
    enum OMP_pragma_clause cdir;

    /* clause to be inserted. */
    c = list2(LIST,expv_int_term(INT_CONSTANT,NULL,OMP_DATA_PRIVATE),
	      list1(LIST,expv_sym_term(IDENT,NULL,do_var_sym)));

    /* find any data attribute clauses on do_var_sym */
    for(cp = ctl_top; cp >= ctls; cp--){
	if(CTL_TYPE(cp) != CTL_OMP) continue;
	if(CTL_OMP_ARG_DCLAUSE(cp) != NULL){
	    FOR_ITEMS_IN_LIST(lp,CTL_OMP_ARG_DCLAUSE(cp)){
		x = LIST_ITEM(lp); 
		cdir = (enum OMP_pragma_clause) EXPR_INT(EXPR_ARG1(x));
		if(IS_OMP_DATA_CLAUSE(cdir)){
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
	if(CTL_OMP_ARG_PCLAUSE(cp) != NULL){
	    FOR_ITEMS_IN_LIST(lp,CTL_OMP_ARG_PCLAUSE(cp)){
		x = LIST_ITEM(lp); 
		cdir = (enum OMP_pragma_clause) EXPR_INT(EXPR_ARG1(x));
		if(IS_OMP_DATA_CLAUSE(cdir)){
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
	switch(CTL_OMP_ARG_DIR(cp)){
	case OMP_F_PARALLEL:
	case OMP_F_PARALLEL_DO:
	case OMP_F_PARALLEL_SECTIONS:
	    if(CTL_OMP_ARG_PCLAUSE(cp) == NULL)
		CTL_OMP_ARG_PCLAUSE(cp) = list1(LIST,c);
	    else 
		list_put_last(CTL_OMP_ARG_PCLAUSE(cp),c);
	    return;
	}
    }
    return; /* nothing to do, not in parallel region. */

found:
    if(cdir == OMP_DATA_PRIVATE || 
       cdir == OMP_DATA_FIRSTPRIVATE || 
       cdir == OMP_DATA_LASTPRIVATE)
	return; /* already private */
    
    if(IS_OMP_REDUCTION_DATA_CLAUSE(cdir)){
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
       (CTL_OMP_ARG_DIR(ctl_top) == OMP_F_DO ||
	CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_DO)){
	    if(CTL_OMP_ARG_DCLAUSE(cp) == NULL)
		CTL_OMP_ARG_DCLAUSE(cp) = list1(LIST,c);
	    else 
		list_put_last(CTL_OMP_ARG_DCLAUSE(cp),c);
    }
}

