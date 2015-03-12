/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-compile.c
 */

#include "F-front.h"
#include "F-module-procedure.h"
#include "module-manager.h"

#include <sys/wait.h>

/* program unit control stack */
UNIT_CTL unit_ctls[MAX_UNIT_CTL];
int unit_ctl_level;
int unit_ctl_contains_level;

/* flags and defaults */
int save_all = FALSE;
int sub_stars = FALSE;
enum storage_class default_stg = STG_SAVE;

/* procedure context */
enum procedure_state current_proc_state;

/* module context */
enum module_state current_module_state = M_DEFAULT;

char *current_module_name = (char *) NULL;
#define INMODULE()    (current_module_name != NULL)

/* for partial module compile with fork.  */
static long module_start_offset = 0;
extern long last_initial_line_pos;
extern long prelast_initial_line_pos;

extern char xmodule_path[MAX_PATH_LEN];
extern char *myName;
extern int  flag_module_compile;
extern char *original_source_file_name;
extern int fixed_line_len_kind;
extern int auto_save_attr_kb;


/* control stack */
CTL ctls[MAX_CTL];
CTL *ctl_top;
CTL *ctl_top_saved = NULL;

expv CURRENT_STATEMENTS_saved = NULL;

/* current statement label */
ID this_label;

TYPE_DESC type_REAL, type_INT, type_SUBR, type_CHAR, type_LOGICAL;
TYPE_DESC type_DREAL, type_COMPLEX, type_DCOMPLEX;
TYPE_DESC type_MODULE;
TYPE_DESC type_GNUMERIC_ALL;
TYPE_DESC type_NAMELIST;
TYPE_DESC basic_type_desc[N_BASIC_TYPES];
expv expv_constant_1,expv_constant_0,expv_constant_m1;
expv expv_float_0;

static int isInFinalizer = FALSE;

static void cleanup_unit_ctl(UNIT_CTL uc);
static void initialize_unit_ctl(void);
static void begin_procedure(void);
static void end_procedure(void);
static void compile_exec_statement(expr x);
static void compile_DO_statement(int range_st_no,
                            expr construct_name,
                            expr var, expr init,
                            expr limit, expr incr);
static void compile_DOWHILE_statement(int range_st_no, expr cond,
                            expr construct_name);
static void check_DO_end(ID label);
static void end_declaration(void);
static void end_interface(void);
static void compile_CALL_statement(expr x);
static void compile_RETURN_statement(expr x);
static void compile_STOP_PAUSE_statement(expr x);
static void compile_NULLIFY_statement(expr x);
static void compile_ALLOCATE_DEALLOCATE_statement(expr x);
static void compile_ARITHIF_statement(expr x);
static void compile_GOTO_statement(expr x);
static void compile_COMPGOTO_statement(expr x);
static void compile_ASSIGN_LABEL_statement(expr x);
static void compile_ASGOTO_statement(expr x);
static void compile_PUBLIC_PRIVATE_statement(expr x, int (*markAs)(ID));
static void compile_TARGET_POINTER_ALLOCATABLE_statement(expr x);
static void compile_OPTIONAL_statement(expr x);
static void compile_INTENT_statement(expr x);
static void compile_INTERFACE_statement(expr x);
static void compile_MODULEPROCEDURE_statement(expr x);
static int  markAsPublic(ID id);
static int  markAsPrivate(ID id);
static void compile_POINTER_SET_statement(expr x);
static void compile_USE_decl(expr x, expr x_args);
static void compile_USE_ONLY_decl(expr x, expr x_args);
static expv compile_scene_range_expression_list(
                            expr scene_range_expression_list);
static void fix_array_dimensions_recursive(ID ip);
static void fix_pointer_pointee_recursive(TYPE_DESC tp);
static void compile_data_style_decl(expr x);


void init_for_OMP_pragma();
void check_for_OMP_pragma(expr x);

expv OMP_pragma_list(enum OMP_pragma pragma,expv arg1,expv arg2);
expv OMP_FOR_pragma_list(expv clause,expv statements);

void init_for_XMP_pragma();
int check_for_XMP_pragma(int st_no, expr x);

void set_parent_implicit_decls(void);

void
initialize_compile()
{
    int t;
    TYPE_DESC tp;

    for(t = 0; t < N_BASIC_TYPES; t++){
        if((BASIC_DATA_TYPE)t == TYPE_UNKNOWN ||
           (BASIC_DATA_TYPE)t == TYPE_ARRAY){
            basic_type_desc[t] = NULL;
            continue;
        }
        tp = new_type_desc();
        TYPE_BASIC_TYPE(tp) = (BASIC_DATA_TYPE)t;

        basic_type_desc[t] = tp;
    }
    type_REAL = BASIC_TYPE_DESC(TYPE_REAL);
    type_DREAL= BASIC_TYPE_DESC(TYPE_DREAL);
    type_COMPLEX = BASIC_TYPE_DESC(TYPE_COMPLEX);
    type_DCOMPLEX = BASIC_TYPE_DESC(TYPE_DCOMPLEX);
    type_INT = BASIC_TYPE_DESC(TYPE_INT);
    type_SUBR = BASIC_TYPE_DESC(TYPE_SUBR);
    type_LOGICAL = BASIC_TYPE_DESC(TYPE_LOGICAL);
    type_CHAR = BASIC_TYPE_DESC(TYPE_CHAR);
    TYPE_CHAR_LEN(type_CHAR) = 1;
    type_GNUMERIC_ALL = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
    type_NAMELIST = BASIC_TYPE_DESC(TYPE_NAMELIST);

    expv_constant_1 = expv_int_term(INT_CONSTANT,type_INT,1);
    expv_constant_0 = expv_int_term(INT_CONSTANT,type_INT,0);
    expv_constant_m1 = expv_int_term(INT_CONSTANT,type_INT,-1);
    expv_float_0 = expv_float_term(FLOAT_CONSTANT,type_REAL,0.0, "0.0");

    type_MODULE = BASIC_TYPE_DESC(TYPE_MODULE);

    initialize_intrinsic();

    initialize_compile_procedure();
    initialize_unit_ctl();

    isInFinalizer = FALSE;
}

void finalize_compile()
{
    isInFinalizer = TRUE;
    begin_procedure();
}

/* initialize for each procedure */
void
initialize_compile_procedure()
{
    save_all = FALSE;
    sub_stars = FALSE;

    this_label = NULL;
    need_keyword = 0;
    
    /* control stack */
    ctl_top = ctls;
    CTL_TYPE(ctl_top) = CTL_NONE;

    init_for_OMP_pragma();
    init_for_XMP_pragma();
}

void
output_statement(v)
     expv v;
{
    if (v == NULL)
        return;
    /* check line number */
    if(EXPR_LINE(v) == NULL) EXPR_LINE(v) = current_line;

    if (CURRENT_STATEMENTS == NULL) {
        CURRENT_STATEMENTS = list1(LIST, v);
    } else if(EXPV_CODE(CURRENT_STATEMENTS) == LIST) {
        CURRENT_STATEMENTS = list_put_last(CURRENT_STATEMENTS, v);
    } else {
        CURRENT_STATEMENTS = list2(LIST, CURRENT_STATEMENTS, v);
    }
}


/* enter control block */
void
push_ctl(ctl)
     enum control_type ctl;
{
    if(++ctl_top >= &ctls[MAX_CTL])
      fatal("too many nested loop or if-then-else");
    CTL_TYPE(ctl_top) = ctl;
    CTL_SAVE(ctl_top) = CURRENT_STATEMENTS;
    CURRENT_STATEMENTS = NULL;
    CURRENT_BLK_LEVEL++;
}

/* pop control block and output statement block */
void
pop_ctl()
{
    /* restore previous statements */
    CURRENT_STATEMENTS = CTL_SAVE(ctl_top); 
    output_statement(CTL_BLOCK(ctl_top));

    /* pop */
    if(ctl_top-- <= ctls) fatal("control stack empty");
    CURRENT_BLK_LEVEL--;
}

void
compile_statement(st_no,x)
     int st_no;
     expr x;
{
    int doCont = 0;
    if(x == NULL) return; /* error recovery */

    if(debug_flag){
        fprintf(debug_fp,"##line(%d):\n",st_no);
        expr_print(x,debug_fp);
    }

    check_for_OMP_pragma(x);
    doCont = check_for_XMP_pragma(st_no, x);

    if (st_no != 0 && doCont == 1) {
        this_label = declare_label(st_no, LAB_UNKNOWN, TRUE);
        if (LAB_TYPE(this_label) != LAB_FORMAT) {
            output_statement(list1(STATEMENT_LABEL, ID_ADDR(this_label)));
        }
    } else this_label = NULL;

    if (doCont == 1)
	compile_statement1(st_no,x);

    /* check do range */
    if(this_label) check_DO_end(this_label);
}

void compile_statement1(int st_no, expr x)
{
    expv v,st;
    list lp;

    // TODO inside where statement, only assign statement available.
    // TODO inside select statement, only case label available.

    /* If top level in contains statement, */
    if (unit_ctl_level > 0
        && CURRENT_STATE == OUTSIDE
        /* FUNCTION, SUBROUTINE statement is allowed */
        && EXPR_CODE(x) != F_FUNCTION_STATEMENT
        && EXPR_CODE(x) != F_SUBROUTINE_STATEMENT
        /* END of parent's statement is allowed */
        && EXPR_CODE(x) != F95_ENDFUNCTION_STATEMENT
        && EXPR_CODE(x) != F95_ENDSUBROUTINE_STATEMENT
        && EXPR_CODE(x) != F95_ENDPROGRAM_STATEMENT
        && EXPR_CODE(x) != F95_ENDMODULE_STATEMENT
        && EXPR_CODE(x) != F95_ENDINTERFACE_STATEMENT
        && EXPR_CODE(x) != F_END_STATEMENT
        /* differ CONTAIN from INTERFASE */
        && PARENT_STATE != ININTR
        /* MODULE PROCEDURE statement is allower under INTERFACE */
        && (EXPR_CODE(x) != F95_MODULEPROCEDURE_STATEMENT || PARENT_STATE != ININTR)
        && EXPR_CODE(x) != F_INCLUDE_STATEMENT)
    {
        /* otherwise error */
        error("only function/subroutine statement are allowed "
            "in contains top level");
        return;
    }
    else if(unit_ctl_level > 0
            && PARENT_STATE == ININTR)
    {
        if(
        CURRENT_STATE == OUTSIDE
        &&EXPR_CODE(x) != F95_ENDINTERFACE_STATEMENT
        /* FUNCTION, SUBROUTINE statement is allowed */
        && EXPR_CODE(x) != F_FUNCTION_STATEMENT
        && EXPR_CODE(x) != F_SUBROUTINE_STATEMENT
        /* MODULE PROCEDURE statement is allower under INTERFACE */
        && EXPR_CODE(x) != F95_MODULEPROCEDURE_STATEMENT
        && EXPR_CODE(x) != F_INCLUDE_STATEMENT)
        {
        error("only function/subroutine/module procedure statement are allowed "
            "in contains top level");
        return;
        }
    }

    switch(EXPR_CODE(x)){

    case F95_MODULE_STATEMENT: /* (F95_MODULE_STATEMENT) */
        begin_procedure();
        declare_procedure(CL_MODULE, EXPR_ARG1(x), type_MODULE, NULL, NULL, NULL);
        begin_module(EXPR_ARG1 (x));
        break;

    case F95_ENDMODULE_STATEMENT: /* (F95_ENDMODULE_STATEMENT) */      
    do_end_module:
        check_INDCL();
	// move into end_procedure()
	//if (endlineno_flag)
	//ID_END_LINE_NO(CURRENT_PROCEDURE) = current_line->ln_no;
        end_procedure();
        end_module();
        break;

    /* (F_PROGRAM_STATEMENT name) need: option or lias */
    case F95_USE_STATEMENT:
        check_INDCL();
        compile_USE_decl(EXPR_ARG1(x), EXPR_ARG2(x));
        break;

    case F95_USE_ONLY_STATEMENT:
        check_INDCL();
        compile_USE_ONLY_decl(EXPR_ARG1(x), EXPR_ARG2(x));
        break;

    case F95_INTERFACE_STATEMENT:
        check_INDCL();
        compile_INTERFACE_statement(x);
        break;

    case F95_ENDINTERFACE_STATEMENT:
        end_interface();
        break;

    case F95_MODULEPROCEDURE_STATEMENT:
        compile_MODULEPROCEDURE_statement(x);
        break;

    case F_PROGRAM_STATEMENT:   /* (F_PROGRAM_STATEMENT name) */
        begin_procedure();
        declare_procedure(CL_MAIN, EXPR_ARG1(x), NULL, NULL, NULL, NULL);
        break;
    case F_BLOCK_STATEMENT:     /* (F_BLOCK_STATEMENT name) */
        begin_procedure();
        declare_procedure(CL_BLOCK, EXPR_ARG1(x), NULL, NULL, NULL, NULL);
        break;
    case F_SUBROUTINE_STATEMENT:
        /* (F_SUBROUTINE_STATEMENT name dummy_arg_list) */
        begin_procedure();
        declare_procedure(CL_PROC,
                          EXPR_ARG1(x), new_type_subr(), EXPR_ARG2(x), EXPR_ARG3(x), NULL);
        break;
        /* entry statements */
    case F_FUNCTION_STATEMENT:
        /* (F_FUNCTION_STATEMENT name dummy_arg_list type) */
        begin_procedure();
        if (EXPR_ARG3(x) && EXPR_CODE(EXPR_ARG3(x)) == IDENT) {
            /* in case of struct */
            TYPE_DESC tp = declare_struct_type_wo_component(EXPR_ARG3(x));
            declare_procedure(CL_PROC, EXPR_ARG1(x),
                              tp,
                              EXPR_ARG2(x), EXPR_ARG4(x), EXPR_ARG5(x));
        }else{
            declare_procedure(CL_PROC, EXPR_ARG1(x),
                              compile_type(EXPR_ARG3(x)),
                              EXPR_ARG2(x), EXPR_ARG4(x), EXPR_ARG5(x));
        }
        break;
    case F_ENTRY_STATEMENT:
        /* (F_ENTRY_STATEMENT name dummy_arg_list) */
        if(CURRENT_STATE == OUTSIDE ||
           CURRENT_PROC_CLASS == CL_MAIN ||
           CURRENT_PROC_CLASS == CL_BLOCK){
            error("misplaced entry statement");
            break;
        }
        declare_procedure(CL_ENTRY,
                          EXPR_ARG1(x), NULL, EXPR_ARG2(x),
                          NULL, EXPR_ARG3(x));
        break;
    case F_INCLUDE_STATEMENT:
        /* (F_INCLUDE_STATEMENT filename) */
        v = EXPR_ARG1(x);
        if(v == NULL) break; /* error recovery */
        if(EXPR_CODE(v) == STRING_CONSTANT) {
            include_file(EXPR_STR(v), FALSE);
        }
        else error("bad file name in include statement");
        break;


    case F95_ENDFUNCTION_STATEMENT:  /* (F95_END_FUNCTION_STATEMENT) */
    case F95_ENDSUBROUTINE_STATEMENT:  /* (F95_END_SUBROUTINE_STATEMENT) */
        check_INEXEC();
	// move into end_procedure()
	//if (endlineno_flag)
	//ID_END_LINE_NO(CURRENT_PROCEDURE) = current_line->ln_no;
        end_procedure();
        break;

    case F95_ENDPROGRAM_STATEMENT:  /* (F95_END_PROGRAM_STATEMENT) */
        check_INEXEC();
	// move into end_procedure()
	//if (endlineno_flag)
	//if (CURRENT_EXT_ID && EXT_LINE(CURRENT_EXT_ID))
	//EXT_END_LINE_NO(CURRENT_EXT_ID) = current_line->ln_no;
        end_procedure();
        break;
    case F_END_STATEMENT:       /* (F_END_STATEMENT) */
        if((CURRENT_PROC_NAME == NULL ||
            (CURRENT_PROC_CLASS == CL_MODULE)) &&
            current_module_name != NULL) {
            goto do_end_module;
        } else {
            check_INEXEC();
	    // move into end_procedure()
	    //if (endlineno_flag){
	    //	      if (CURRENT_PROCEDURE)
	    //ID_END_LINE_NO(CURRENT_PROCEDURE) = current_line->ln_no;
	    //else if (CURRENT_EXT_ID && EXT_LINE(CURRENT_EXT_ID))
	    //EXT_END_LINE_NO(CURRENT_EXT_ID) = current_line->ln_no;
	    //}
	    check_for_OMP_pragma(x); /* close DO directives if any */
	    check_for_XMP_pragma(st_no, x); /* close LOOP directives if any */
            end_procedure();
        }
        break;
    case F95_CONTAINS_STATEMENT:
        check_INEXEC();
        push_unit_ctl(INCONT);
        break;

        /* 
         * declaration statement
         */
    case F_TYPE_DECL: /* (F_TYPE_DECL type (LIST data ....) (LIST attr ...)) */
        check_INDCL();
        compile_type_decl(EXPR_ARG1(x), NULL, EXPR_ARG2(x),EXPR_ARG3(x));
        /* in case of data-style initializer like "INTEGER A / 10 /",
         * F_TYPE_DECL has data structure like, (LIST, IDENTIFIER,
         * dims, length, (F_DATA_DECL, LIST(..) ), data_val_list) so
         * separate and compile data declarations after type
         * declarations. */
        compile_data_style_decl(EXPR_ARG2(x));
        break;

    case F95_DIMENSION_DECL: /* (F95_DIMENSION_DECL (LIST data data)) */
        check_INDCL();
        compile_type_decl(NULL, NULL, EXPR_ARG1(x), NULL);
        break;

    case F_COMMON_DECL: /* (F_COMMON_DECL common_decl) */
        check_INDCL();
        /* common_decl = (LIST common_name (LIST var dims) ...) */
        compile_COMMON_decl(EXPR_ARG1(x));
        break;

    case F_EQUIV_DECL: /* (F_EQUIVE_DECL (LIST lhs ...) ...) */
        check_INDCL();
        if (UNIT_CTL_EQUIV_DECLS(CURRENT_UNIT_CTL) == NULL) {
            UNIT_CTL_EQUIV_DECLS(CURRENT_UNIT_CTL) = list0(LIST);
        }
        list_put_last(UNIT_CTL_EQUIV_DECLS(CURRENT_UNIT_CTL), EXPR_ARG1(x));
        break;

    case F_IMPLICIT_DECL:
        check_INDCL();
	if (EXPR_ARG1(x)){
	  FOR_ITEMS_IN_LIST(lp,EXPR_ARG1(x)){
            v = LIST_ITEM(lp);
            /* implicit none?  result in peek the data structture.  */
            if (EXPR_CODE (EXPR_ARG1 (EXPR_ARG1(v)))== F_TYPE_NODE)
                compile_IMPLICIT_decl(EXPR_ARG1(v),EXPR_ARG2(v));
            else {
                v = EXPR_ARG1(v);
                compile_IMPLICIT_decl(EXPR_ARG1(v),EXPR_ARG2(v));
            }
	  }
	} else { /* implicit none */
            if (UNIT_CTL_IMPLICIT_TYPE_DECLARED(CURRENT_UNIT_CTL))
                error("IMPLICIT NONE and IMPLICIT type declaration "
                      "cannot co-exist");
            UNIT_CTL_IMPLICIT_NONE(CURRENT_UNIT_CTL) = TRUE;
            set_implicit_type_uc(CURRENT_UNIT_CTL, NULL, 'a', 'z', TRUE);
            list_put_last(UNIT_CTL_IMPLICIT_DECLS(CURRENT_UNIT_CTL),
                          create_implicit_decl_expv(NULL, "a", "z"));
	}

        break;

    case F_FORMAT_DECL: {
        if (this_label == NULL) {
            error("format without statement label.");
            break;
        }
        this_label = declare_label(st_no, LAB_FORMAT, FALSE);
        if (LAB_TYPE(this_label) != LAB_FORMAT) {
            fatal("can't generate label for format.");
        }
        compile_FORMAT_decl(st_no, x);
        break;
    }

    case F_PARAM_DECL:
        check_INDCL();
        compile_PARAM_decl(EXPR_ARG1(x));
        break;

    case F_CRAY_POINTER_DECL:
        NOT_YET();
        break;

    case F_EXTERNAL_DECL:
        check_INDCL();
        compile_EXTERNAL_decl(EXPR_ARG1(x));
        break;

    case F_DATA_DECL:
        check_INDCL();
        /* compilataion is executed later in end_declaration */
        list_put_last(CURRENT_INITIALIZE_DECLS, x);
        break;

    case F_INTRINSIC_DECL:
        check_INDCL();
        compile_INTRINSIC_decl(EXPR_ARG1(x));
        break;

    case F_SAVE_DECL:
        check_INDCL();
        compile_SAVE_decl(EXPR_ARG1(x));
        break;

    case F95_TARGET_STATEMENT:
    case F95_POINTER_STATEMENT:
    case F95_ALLOCATABLE_STATEMENT:
        check_INDCL();
        compile_TARGET_POINTER_ALLOCATABLE_statement(x);
        break;

    case F95_OPTIONAL_STATEMENT:
        check_INDCL();
        compile_OPTIONAL_statement(x);
        break;

    case F95_INTENT_STATEMENT:
        check_INDCL();
        compile_INTENT_statement(x);
        break;

    case F_NAMELIST_DECL:
        check_INDCL();
        compile_NAMELIST_decl(EXPR_ARG1(x));
        break;

    case F_IF_STATEMENT: /* (F_IF_STATEMENT condition statement) */
        check_INEXEC();

        push_ctl(CTL_IF);
        /* evaluate condition and make IF_STATEMENT clause */
        v = compile_logical_expression(EXPR_ARG1(x));
        st = list5(IF_STATEMENT,v,NULL,NULL,NULL,NULL);
        output_statement(st);
        CTL_BLOCK(ctl_top) = CURRENT_STATEMENTS;
        CURRENT_STATEMENTS = NULL;

        /* construct name */
        if (EXPR_HAS_ARG3(x)) {
	  //list_put_last(st, EXPR_ARG3(x));
	  EXPR_ARG4(st) = EXPR_ARG3(x);
        }
        /* set current IF_STATEMENT */
        CTL_IF_STATEMENT(ctl_top) = st;
        if(EXPR_ARG2(x)){
            compile_exec_statement(EXPR_ARG2(x));
            CTL_IF_THEN(ctl_top) = CURRENT_STATEMENTS;
	    if (endlineno_flag){
	      if (current_line->end_ln_no){
		EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->end_ln_no;
	      }
	      else {
		EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
	      }
	    }
            pop_ctl();  /* pop and output */
            break;
        }
        break;
    case F_ELSE_STATEMENT: /* (F_ELSE_STATEMENT) */
        check_INEXEC();
        if(CTL_TYPE(ctl_top) == CTL_IF){
            /* store current statements to 'then' part, and clear */
            CTL_IF_THEN(ctl_top) = CURRENT_STATEMENTS;
            CURRENT_STATEMENTS = NULL;

            /* change to CTL_ELSE */
            CTL_TYPE(ctl_top) = CTL_ELSE;

	    if (endlineno_flag){
	      st = list0(F_ELSE_STATEMENT);
	      output_statement(st);
	      CURRENT_STATEMENTS = NULL;
	      EXPR_ARG5(CTL_IF_STATEMENT(ctl_top)) = st;
	    }

        } else error("'else', out of place");
        break;
    case F_ELSEIF_STATEMENT: /* (F_IF_STATEMENT condition) */
        check_INEXEC();
        if(CTL_TYPE(ctl_top) == CTL_IF){
            /* store current statements to 'then' part, and clear */
            CTL_IF_THEN(ctl_top) = CURRENT_STATEMENTS;
            CURRENT_STATEMENTS = NULL;

            /* evaluate condition and make IF_STATEMENT clause */
            v = compile_logical_expression(EXPR_ARG1(x));
            st = list5(IF_STATEMENT,v,NULL,NULL,NULL,NULL);
            output_statement(st);
            CTL_IF_ELSE(ctl_top) = CURRENT_STATEMENTS;
            CURRENT_STATEMENTS = NULL;

            /* set current IF_STATEMENT clause */
            CTL_IF_STATEMENT(ctl_top) = st;

        } else {
            v = compile_logical_expression(EXPR_ARG1(x)); /* error check */
            error("'elseif', out of place");
        }
        break;
    case F_ENDIF_STATEMENT: /* (F_ENDIF_STATEMENT) */
        check_INEXEC();
        if(CTL_TYPE(ctl_top) == CTL_IF){
            /* use current_statements */
            CTL_IF_THEN(ctl_top) = CURRENT_STATEMENTS;

	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();
        }  else if(CTL_TYPE(ctl_top) == CTL_ELSE) {
            CTL_IF_ELSE(ctl_top) = CURRENT_STATEMENTS;

	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();
        } else error("'endif', out of place");

        break;

    case F_DO_STATEMENT: {
        int doStNo = -1;
        check_INEXEC();
        /* (F_DO_STATEMENT label do_spec) */
        /* do_spec := (LIST id  e1 e2 e3) */

        if (EXPR_ARG1(x) != NULL) {
            expv stLabel = expr_label_value(EXPR_ARG1(x));
            if (stLabel == NULL) {
                error("illegal label in DO");
                break;
            }
            doStNo = EXPV_INT_VALUE(stLabel);
        }
        if (EXPR_ARG2(x) == NULL) {
            /* f95 type do */
            compile_DO_statement(doStNo,
                                 EXPR_ARG3(x), /* construct name */
                                 NULL,
                                 NULL,
                                 NULL,
                                 NULL);
        } else {
            compile_DO_statement(doStNo,
                                 EXPR_ARG3(x), /* construct name */
                                 EXPR_ARG1(EXPR_ARG2(x)),
                                 EXPR_ARG2(EXPR_ARG2(x)),
                                 EXPR_ARG3(EXPR_ARG2(x)),
                                 EXPR_ARG4(EXPR_ARG2(x)));
        }
        break;
    }

    case F_ENDDO_STATEMENT:
        check_INEXEC();
        check_DO_end(NULL);

	if (CTL_TYPE(ctl_top) == CTL_OMP){
	  if (CTL_OMP_ARG_DIR(ctl_top) == OMP_F_PARALLEL_DO){
	    CTL_BLOCK(ctl_top) = 
		OMP_pragma_list(OMP_PARALLEL, CTL_OMP_ARG_PCLAUSE(ctl_top),
				OMP_FOR_pragma_list(
				    CTL_OMP_ARG_DCLAUSE(ctl_top),
				    CURRENT_STATEMENTS));
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    pop_ctl();
	  }
	  else if (CTL_OMP_ARG_DIR(ctl_top) == OMP_F_DO){
	    expv dclause = CTL_OMP_ARG_DCLAUSE(ctl_top);
	    //if (EXPR_ARG2(x) != NULL) list_put_last(dclause, EXPR_ARG2(x));
	    CTL_BLOCK(ctl_top) = 
		OMP_FOR_pragma_list(dclause, CURRENT_STATEMENTS);
	    EXPR_LINE(CTL_BLOCK(ctl_top)) = EXPR_LINE(CTL_OMP_ARG(ctl_top));
	    ctl_top_saved = ctl_top;
	    CURRENT_STATEMENTS_saved = CURRENT_STATEMENTS;
	    pop_ctl();
	  }
	}

        break;

    case F_DOWHILE_STATEMENT: {
        int doStNo = -1;
        check_INEXEC();
        /* (F_DOWHILE_STATEMENT label cond_expr) */

        if (EXPR_ARG1(x) != NULL) {
            expv stLabel = expr_label_value(EXPR_ARG1(x));
            if (stLabel == NULL) {
                error("illegal label in DO WHILE");
                break;
            }
            doStNo = EXPV_INT_VALUE(stLabel);
        }

	compile_DOWHILE_statement(doStNo, EXPR_ARG2(x), EXPR_ARG3(x));

        break;

	//    case F_DOWHILE_STATEMENT:
	//        check_INEXEC();
	//        /* (F_DOWHILE_STATEMENT cond_expr) */
	//        compile_DOWHILE_statement(EXPR_ARG2(x), EXPR_ARG3(x));
	//        break;
    }

    /* case where statement*/
    case F_WHERE_STATEMENT:
        check_INEXEC();
        push_ctl(CTL_WHERE);

        /* evaluate condition and make WHERE_STATEMENT clause */
        v = compile_logical_expression_with_array(EXPR_ARG1(x));

        st = list5(F_WHERE_STATEMENT,v,NULL,NULL,NULL,NULL);
        output_statement(st);

        CTL_BLOCK(ctl_top) = CURRENT_STATEMENTS;
        CURRENT_STATEMENTS = NULL;

        /* set current WHERE_STATEMENT */
        CTL_WHERE_STATEMENT(ctl_top) = st;
        if(EXPR_ARG2(x) != NULL) {
            compile_statement1(st_no, EXPR_ARG2(x));
            /* TODO x must be array assignment expression,
             * and shape of array is equal to v
             */

            CTL_WHERE_THEN(ctl_top) = CURRENT_STATEMENTS;
            pop_ctl();  /* pop and output */
            break;
        }
        break;
    case F_ELSEWHERE_STATEMENT:
        check_INEXEC();
        if(CTL_TYPE(ctl_top) == CTL_WHERE){ /* check WHERE-BLOCK  */
            if( EXPR_LIST(x)==NULL ){ /*  no condition  */
                /* store current statements to 'then' part, and clear */
                CTL_WHERE_THEN(ctl_top) = CURRENT_STATEMENTS;
                CURRENT_STATEMENTS = NULL;

                /* change to CTL_ELSE_WHERE */
                CTL_TYPE(ctl_top) = CTL_ELSE_WHERE;
                
                if (endlineno_flag){
                    st = list0(F_ELSEWHERE_STATEMENT);
                    output_statement(st);
                    CURRENT_STATEMENTS = NULL;
                    EXPR_ARG5(CTL_WHERE_STATEMENT(ctl_top)) = st;
                }
            }else{ /*  has condition  */
                CTL_WHERE_THEN(ctl_top) = CURRENT_STATEMENTS;
                CURRENT_STATEMENTS = NULL;
                
                /* evaluate condition and make WHERE_STATEMENT clause */
                v = compile_logical_expression_with_array(EXPR_ARG1(x));
                
                st = list5(F_WHERE_STATEMENT,v,NULL,NULL,NULL,NULL);
                output_statement(st);
                
                CTL_WHERE_ELSE(ctl_top) = CURRENT_STATEMENTS;
                CURRENT_STATEMENTS = NULL;
                
                /* set current WHERE_STATEMENT */
                CTL_WHERE_STATEMENT(ctl_top) = st;

                if(EXPR_ARG2(x) != NULL) {
                    compile_statement1(st_no, EXPR_ARG2(x));
                    /* TODO x must be array assignment expression,
                     * and shape of array is equal to v
                     */
                
                    CTL_WHERE_THEN(ctl_top) = CURRENT_STATEMENTS;
                    pop_ctl();  /* pop and output */
                    break;
                }
            }
        } else error("'elsewhere', out of place");
        break;
    case F_ENDWHERE_STATEMENT:
        if(CTL_TYPE(ctl_top) == CTL_WHERE) {
            /* store current statements to 'then' part, and clear */
            CTL_WHERE_THEN(ctl_top) = CURRENT_STATEMENTS;

	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();

        } else if(CTL_TYPE(ctl_top) == CTL_ELSE_WHERE){
            /* store current statements to 'else' part, and clear */
            CTL_WHERE_ELSE(ctl_top) = CURRENT_STATEMENTS;

	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();

        } else error("'end where', out of place");
        break;
    /* end case where statement */

    case F_SELECTCASE_STATEMENT:
        check_INEXEC();

        push_ctl(CTL_SELECT);

        v = compile_expression(EXPR_ARG1(x));
        st = list3(F_SELECTCASE_STATEMENT, v, NULL, EXPR_ARG2(x));

        CTL_BLOCK(ctl_top) = st;

        break;
    case F_CASELABEL_STATEMENT:
        if(CTL_TYPE(ctl_top) == CTL_SELECT  ||
           CTL_TYPE(ctl_top) == CTL_CASE) {

            if (CTL_TYPE(ctl_top) == CTL_CASE) {
                CTL_CASE_BLOCK(ctl_top) = CURRENT_STATEMENTS;
                CURRENT_STATEMENTS = NULL;

		if (endlineno_flag)
		  EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

                pop_ctl();
            }

            v = compile_scene_range_expression_list(EXPR_ARG1(x));
            push_ctl(CTL_CASE);

            /*
             *  (F_CASELABEL_STATEMENT
             *    (LIST (scene range expression) ...)
             *    (LIST (exec statement) ...)
             *    (IDENTIFIER))
             */
            st = list3(F_CASELABEL_STATEMENT,v,NULL,EXPR_ARG2(x));

            CTL_BLOCK(ctl_top) = st;

        } else error("'case label', out of place");
        break;
    case F_ENDSELECT_STATEMENT:
        if(CTL_TYPE(ctl_top) == CTL_SELECT) {
            CTL_SELECT_STATEMENT_BODY(ctl_top) = CURRENT_STATEMENTS;

	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();
        } else if (CTL_TYPE(ctl_top) == CTL_CASE) {
            CTL_CASE_BLOCK(ctl_top) = CURRENT_STATEMENTS;

	    // For previous CASE.
	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();

            if(CTL_TYPE(ctl_top) != CTL_SELECT)
                error("'end select', out of place");

            CTL_SELECT_STATEMENT_BODY(ctl_top) = CURRENT_STATEMENTS;

	    // For SELECT
	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();
        } else error("'end select', out of place");
        break;

    case F_PRAGMA_STATEMENT:
        compile_pragma_statement(x);
        break;

    case F95_TYPEDECL_STATEMENT:
        check_INDCL();
        /* (F95_TYPEDECL_STATEMENT (LIST <I> <NULL>) <NULL>) */
        compile_struct_decl(EXPR_ARG1(x), EXPR_ARG2(x));
        break;

    case F95_ENDTYPEDECL_STATEMENT:
        check_INDCL();
        /* (F95_ENDTYPEDECL_STATEMENT <NULL>) */
        compile_struct_decl_end();
        break;

    case F95_SEQUENCE_STATEMENT:
        compile_SEQUENCE_statement();
        break;

    case F95_NULLIFY_STATEMENT:
        check_INEXEC();
        compile_NULLIFY_statement(x);
        break;

    case F95_PUBLIC_STATEMENT:
        check_INDCL();
        compile_PUBLIC_PRIVATE_statement(EXPR_ARG1(x), markAsPublic);
        break;

    case F95_PRIVATE_STATEMENT:
        check_INDCL();
        compile_PUBLIC_PRIVATE_statement(EXPR_ARG1(x), markAsPrivate);
        break;

    default:
        compile_exec_statement(x);
        break;
    }
}


int temp_gen = 0;

SYMBOL
gen_temp_symbol(const char *leader)
{
    char name[128];
    sprintf(name,"%s%03d", leader, temp_gen++);
    return find_symbol(name);
}


static expv
allocate_temp(TYPE_DESC tp)
{
    ID id;
    SYMBOL sym;

    sym = gen_temp_symbol("omnitmp");
    id = declare_ident(sym,CL_VAR);
    ID_TYPE(id) = tp;
    ID_STORAGE(id) = STG_AUTO;
    ID_LINE(id) = new_line_info(get_file_id(source_file_name),0);
    declare_variable(id);
    return ID_ADDR(id);
}

/* 
 * executable statement 
 */
static void
compile_exec_statement(expr x)
{
    expr x1;
    expv w,v1,v2;
    SYMBOL s;
    ID id;

    extern void OMP_check_LET_statement();
    extern int OMP_output_st_pragma(expv w);
    extern void XMP_check_LET_statement();
    extern int XMP_output_st_pragma(expv w);

    if(EXPR_CODE(x) != F_LET_STATEMENT) check_INEXEC();

    switch(EXPR_CODE(x)){

    case F_LET_STATEMENT: /* (F_LET_STATEMENT lhs rhs) */
	OMP_check_LET_statement();
	XMP_check_LET_statement();

      if (CURRENT_STATE == OUTSIDE) {
	begin_procedure();
	//declare_procedure(CL_MAIN, NULL, NULL, NULL, NULL, NULL);
	declare_procedure(CL_MAIN, make_enode(IDENT, find_symbol("main")),
			  NULL, NULL, NULL, NULL);
      }

      x1 = EXPR_ARG1(x);
      switch (EXPR_CODE(x1)){

      case F_ARRAY_REF: /* for a statement function because it looks like an array reference. */

	if (EXPR_CODE(EXPR_ARG1(x1)) == IDENT){
	  s = EXPR_SYM(EXPR_ARG1(x1));
	  v1 = EXPR_ARG2(x1);
	  v2 = EXPR_ARG2(x);

	  /* If the first argument is a triplet,
	   * it is not a function statement .*/
	  if (EXPR_LIST(v1) == NULL ||
	      EXPR_ARG1(v1) == NULL ||
	      EXPR_CODE(EXPR_ARG1(v1)) != F95_TRIPLET_EXPR){
	    id = find_ident(s);
	    if (id == NULL)
	      id = declare_ident(s, CL_UNKNOWN);
            if (ID_IS_AMBIGUOUS(id)) {
                error_at_node(x, "an ambiguous reference to symbol '%s'", ID_NAME(id));
                return;
            }
	    if (ID_CLASS(id) == CL_UNKNOWN){
	      if (CURRENT_STATE != INEXEC) {
		declare_statement_function(id,v1,v2);
		break;
	      }
	    }
	  }
	}
	/* fall through */
	
      case IDENT:
      case F_SUBSTR_REF:
      case F95_MEMBER_REF:
      case XMP_COARRAY_REF:

	if (NOT_INDATA_YET) end_declaration();
	if ((v1 = compile_lhs_expression(x1)) == NULL ||
            (v2 = compile_expression(EXPR_ARG2(x))) == NULL) {
            break;
        }
	if (!expv_is_lvalue(v1) && !expv_is_str_lvalue(v1)) {
            error_at_node(x, "bad lhs expression in assignment");
            break;
	}
	if ((w = expv_assignment(v1,v2)) == NULL) {
            break;
	}

	if(OMP_output_st_pragma(w)) break;
	if(XMP_output_st_pragma(w)) break;

	output_statement(w);
	break;

      default:
	error("assignment to a non-variable");
      }

      break;

    case F_CONTINUE_STATEMENT:
        output_statement(list0(F_CONTINUE_STATEMENT));
        break; 

    case F_GOTO_STATEMENT:
        compile_GOTO_statement(x);        
        break;

    case F_CALL_STATEMENT:
        compile_CALL_statement(x);
        break;
        
    case F_RETURN_STATEMENT:
        compile_RETURN_statement(x);
        break;

        /* 
         * action statement 95
         */
    case F95_CYCLE_STATEMENT:
    case F95_EXIT_STATEMENT:
        output_statement(list1(EXPR_CODE(x), EXPR_ARG1(x)));
        break;

    case F_STOP_STATEMENT:
    case F_PAUSE_STATEMENT:
        compile_STOP_PAUSE_statement(x);
        break;

    case F_ARITHIF_STATEMENT:
        compile_ARITHIF_statement(x);
        break;

    case F_COMPGOTO_STATEMENT:
        compile_COMPGOTO_statement(x);
        break;


    case F95_NULLIFY_STATEMENT:
        compile_NULLIFY_statement(x);
        break;

        /* 
         * I/O statements
         */
    case F_WRITE_STATEMENT:
    case F_PRINT_STATEMENT:
    case F_READ_STATEMENT:
    case F_READ1_STATEMENT:
        compile_IO_statement(x);
        break;

    case F_OPEN_STATEMENT:
        compile_OPEN_statement(x);
        break;

    case F_CLOSE_STATEMENT:
        compile_CLOSE_statement(x);
        break;

    case F_BACKSPACE_STATEMENT:
    case F_ENDFILE_STATEMENT:
    case F_REWIND_STATEMENT:
        compile_FPOS_statement(x);
        break;

    case F_INQUIRE_STATEMENT:
        compile_INQUIRE_statement(x);
        break;

    case F_ASSIGN_LABEL_STATEMENT:
        compile_ASSIGN_LABEL_statement(x);
        break;
        
    case F_ASGOTO_STATEMENT:
        compile_ASGOTO_statement(x);
        break;

    case F95_ALLOCATE_STATEMENT:
        compile_ALLOCATE_DEALLOCATE_statement(x);
        break;

    case F95_DEALLOCATE_STATEMENT:
        compile_ALLOCATE_DEALLOCATE_statement(x);
        break;

    case F95_POINTER_SET_STATEMENT:
        compile_POINTER_SET_statement(x);
        break;

    default:
        fatal("unknown statement");
    }
}

/**
 * Checks if the current context is inside interface block
 * (between `INTERFACE` and `END INTERFACE`)
 */
static int
check_inside_INTERFACE_body() {
    int i;
    for (i = 0; i <= unit_ctl_level; i++) {
        if (UNIT_CTL_CURRENT_STATE(unit_ctls[i]) == ININTR) {
            return TRUE;
        }
    }
    return FALSE;
}

/* 
 * context control. keep track of context
 */
/* add the in module state virtually.  */
static void
begin_procedure()
{
    if (isInFinalizer == FALSE &&
        CURRENT_STATE >= INSIDE) {
        error("unexpected procedure start.");
        return;
    }

    if(CURRENT_STATE != OUTSIDE) {
        end_procedure();
    }
    CURRENT_STATE = INSIDE;
    CURRENT_PROC_CLASS = CL_MAIN;       /* default */
    current_proc_state = P_DEFAULT;

    /*
     * NOTE:
     * The function/subroutine in the interface body
     * don't host-assciate with the outer scope,
     * and the implicit type conversion rule can be propagated
     * only from the host-associated scopes
     */
    if (unit_ctl_level > 0 && check_inside_INTERFACE_body() == FALSE) {
        set_parent_implicit_decls();
    }

    if (isInFinalizer == FALSE) {
        module_procedure_manager_init();
    }
}

/* now this is not called.  */
void
check_INDATA()
{
    if (CURRENT_STATE == OUTSIDE) {
        begin_procedure();
        declare_procedure(CL_MAIN, NULL, NULL, NULL, NULL, NULL);
    }
    if(NOT_INDATA_YET){
        end_declaration();
        CURRENT_STATE = INDATA;
    }
}

void
check_INDCL()
{
    switch (CURRENT_STATE) {
    case OUTSIDE:       
        begin_procedure();
        if (unit_ctl_level == 0)
	  //declare_procedure(CL_MAIN, NULL, NULL, NULL, NULL, NULL);
	  declare_procedure(CL_MAIN, make_enode(IDENT, find_symbol("main")),
			    NULL, NULL, NULL, NULL);
    case INSIDE:        
        CURRENT_STATE = INDCL;
    case INDCL: 
        break;
    default:
        error("declaration among executables");
    }
}

void
check_INEXEC()
{
    if (CURRENT_STATE == OUTSIDE) {
        begin_procedure();
        if (unit_ctl_level == 0)
            declare_procedure(CL_MAIN, make_enode(IDENT, find_symbol("main")),
                              NULL, NULL, NULL, NULL);
        else
            declare_procedure(CL_MAIN, NULL, NULL, NULL, NULL, NULL);
    }
    if(NOT_INDATA_YET) end_declaration();
}


void
checkTypeRef(ID id) {
    TYPE_DESC tp = ID_TYPE(id);

    while (tp != NULL) {
        if (TYPE_REF(tp) == tp) {
            fatal("%s: TYPE_REF(tp) == tp 0x%p, %s.",
                  __func__,
                  tp,
                  SYM_NAME(ID_SYM(id)));
        }
        tp = TYPE_REF(tp);
    }
}


#define classNeedFix(ip) \
(ID_CLASS(ip) == CL_UNKNOWN ||                  \
 ID_CLASS(ip) == CL_VAR ||                      \
 ID_CLASS(ip) == CL_PARAM ||                                    \
 (ID_CLASS(ip) == CL_PROC && (PROC_CLASS(ip) != P_EXTERNAL && PROC_CLASS(ip) != P_DEFINEDPROC)) || \
 ID_CLASS(ip) == CL_ENTRY)


void
fix_type(ID id) {
    if (classNeedFix(id)) {
        implicit_declaration(id);
    }
    if (VAR_IS_UNCOMPILED(id) == TRUE) {
        TYPE_DESC tp = ID_TYPE(id);
        if (tp != NULL) {
            expr x = VAR_UNCOMPILED_DECL(id);
            ID_TYPE(id) = NULL;
            compile_type_decl(NULL, tp,
                              list1(LIST, EXPR_ARG1(x)), EXPR_ARG2(x));
        }
        VAR_IS_UNCOMPILED(id) = FALSE;
        VAR_IS_UNCOMPILED_ARRAY(id) = FALSE;
        VAR_UNCOMPILED_DECL(id) = NULL;
    }
}


void
unset_save_attr_in_dummy_args(EXT_ID ep)
{
    expv v;
    list lp;
    TYPE_DESC tp;
    ID id;

    FOR_ITEMS_IN_LIST(lp, EXT_PROC_ARGS(ep)) {
        v = LIST_ITEM(lp);
        id = find_ident(EXPV_NAME(EXPR_ARG1(v)));
        if (id != NULL) {
            TYPE_UNSET_SAVE(id);
            tp = ID_TYPE(id);
            if (tp != NULL) {
                TYPE_UNSET_SAVE(tp);
            }
        }
    }
}

static void
union_parent_type(ID id)
{
    ID parent_id;
    TYPE_DESC my_tp, parent_tp;

    if (TYPE_IS_OVERRIDDEN(id)) { /* second time */
        return;
    }
    TYPE_SET_OVERRIDDEN(id);
    if (ID_TYPE(id) == NULL) {
        return;
    }
    parent_id = find_ident_parent(ID_SYM(id));

    if (parent_id == NULL || ID_DEFINED_BY(parent_id) != id)
        return;

    my_tp = ID_TYPE(id);
    parent_tp = ID_TYPE(parent_id);
    assert(my_tp != NULL);

    if(parent_tp == NULL) {
        ID_TYPE(parent_id) = my_tp;
        return;
    }

    if(ID_CLASS(parent_id) == CL_PROC &&
       PROC_CLASS(parent_id) == P_UNDEFINEDPROC) {
        ID_TYPE(parent_id) = my_tp;
        PROC_CLASS(parent_id) = P_DEFINEDPROC;
    }else if (TYPE_IS_EXPLICIT(parent_tp)) {
        if (TYPE_IS_EXPLICIT(my_tp) && ID_CLASS(parent_id) != CL_PROC) {
            error("%s is declared both parent and contains", ID_NAME(id));
        } else {
            /* copy basic type and ref */
            TYPE_BASIC_TYPE(my_tp) = TYPE_BASIC_TYPE(parent_tp);
            TYPE_REF(my_tp) = TYPE_REF(parent_tp);

            assert(TYPE_REF(my_tp) == NULL ||
                TYPE_BASIC_TYPE(my_tp) == TYPE_BASIC_TYPE(TYPE_REF(my_tp)));
        }
    } else {
        if(IS_ARRAY_TYPE(my_tp)) {
            parent_tp = copy_array_type(my_tp);
            ID_TYPE(parent_id) = parent_tp;
        } else {
            TYPE_BASIC_TYPE(parent_tp) = TYPE_BASIC_TYPE(my_tp);
            TYPE_REF(parent_tp) = TYPE_REF(my_tp);

            assert(TYPE_REF(parent_tp) == NULL ||
                TYPE_BASIC_TYPE(parent_tp) == TYPE_BASIC_TYPE(TYPE_REF(parent_tp)));
        }
    }
}

static int isAlreadyMarked(ID id)
{
    TYPE_DESC tp = ID_TYPE(id);

    if (tp == NULL)
        return (TYPE_IS_PUBLIC(id) || TYPE_IS_PRIVATE(id));
    else
        return (TYPE_IS_PUBLIC(id) || TYPE_IS_PRIVATE(id) ||
                TYPE_IS_PUBLIC(tp) || TYPE_IS_PRIVATE(tp));
}


/* called at the end of declaration part */
static void
end_declaration()
{
    ID ip = NULL;
    ID myId = NULL;
    EXT_ID myEId = NULL;
    EXT_ID ep;
    ID vId;
    list lp;
    expv v;
    TYPE_DESC tp;
    UNIT_CTL uc = CURRENT_UNIT_CTL;

    CURRENT_STATE = INEXEC; /* the next status is EXEC */

    if (debug_flag) {
        fprintf(debug_fp,"--- end_declaration ---\n");
        print_IDs(LOCAL_SYMBOLS, debug_fp, TRUE);
        print_IDs(LOCAL_COMMON_SYMBOLS, debug_fp, TRUE);
        print_interface_IDs(LOCAL_SYMBOLS, debug_fp);
        print_types(LOCAL_STRUCT_DECLS, debug_fp);
    }

    if (CURRENT_PROCEDURE != NULL) {

        myId = CURRENT_PROCEDURE;
        
        myEId = declare_current_procedure_ext_id();
        assert(myEId != NULL && EXT_PROC_TYPE(myEId) != NULL);

        if (ID_CLASS(myId) == CL_PROC) {
            PROC_EXT_ID(myId) = myEId;
            EXT_PROC_CLASS(myEId) = EP_PROC;
        }

        if (ID_CLASS(myId) == CL_PROC &&
            PROC_RESULTVAR(myId) != NULL) {
            /*
             * If this is function and declared with result variable,
             * fix type of the result variable.
             */
            TYPE_DESC tp = NULL;
            expv resultV = NULL;
            expr resX = PROC_RESULTVAR(myId);
            SYMBOL resS = EXPR_SYM(resX);
            ID resId = find_ident(resS);

            if (resId == NULL) {
                resId = declare_ident(resS, CL_VAR);
            }
            if (ID_TYPE(resId) == NULL) {
                /*
                 * The result var is not fixed. Use function's type.
                 */
                tp = ID_TYPE(myId);
                if (tp == NULL) {
                    fatal("%s: return type of function '%s' "
                          "is not determined.",
                          __func__, SYM_NAME(ID_SYM(myId)));
                    /* not reached. */
                    return;
                }
            } else {
                /*
                 * Otherwise the result var is fixed.
                 */
                tp = ID_TYPE(resId);
            }
            ID_TYPE(myId) = NULL;
            ID_TYPE(resId) = NULL;
            declare_id_type(myId, tp);
            declare_id_type(resId, tp);

            resId = declare_function_result_id(resS, tp);
            if (resId == NULL) {
                fatal("%s: can't declare function result ident '%s'.",
                      __func__, SYM_NAME(resS));
                /* not reached. */
                return;
            }
            resultV = expv_sym_term(F_VAR, tp, resS);

            /*
             * Set result varriable info.
             */
            EXT_PROC_RESULTVAR(myEId) = resultV;
        }

        /*
         * Fix return type of function when it is struct
         */
        if (CURRENT_PROC_CLASS == CL_PROC &&
            CURRENT_PROCEDURE != NULL &&
            IS_STRUCT_TYPE(ID_TYPE(CURRENT_PROCEDURE))) {
            TYPE_DESC tp = ID_TYPE(CURRENT_PROCEDURE);
            TYPE_DESC tq = NULL;
            TYPE_DESC ts = NULL;
            /*
             * to preserve effect of wrap_type(), we substitute TYPE_DESC
             * in TYPE_REF */
            tq = tp;
            while (TYPE_REF(tp) != NULL) {
                tq = tp;
                tp = TYPE_REF(tp);
            }
            if (!TYPE_IS_DECLARED(tp)) {
                /*
                 * since struct defined in contains cannot be reffered
                 * from parent scope, we use find_struct_decl_parent()
                 */
                ts = find_struct_decl_parent(ID_SYM(TYPE_TAGNAME(tp)));
                if (ts != NULL) {
                    TYPE_REF(tq) = ts;
                } else {
                    error_at_id(CURRENT_PROCEDURE, "function returns undeclared "
                                "struct type \"%s\".", ID_NAME(CURRENT_PROCEDURE));
                }
            }
        }

        /* for recursive */
        assert(ID_TYPE(myId) != NULL);
        if (TYPE_IS_RECURSIVE(myId) ||
            PROC_IS_RECURSIVE(myId)) {
            TYPE_SET_RECURSIVE(ID_TYPE(myId));
            TYPE_SET_RECURSIVE(EXT_PROC_TYPE(myEId));
        }
        /* for pure */
        if (TYPE_IS_PURE(myId) ||
            PROC_IS_PURE(myId)) {
            TYPE_SET_PURE(ID_TYPE(myId));
            TYPE_SET_PURE(EXT_PROC_TYPE(myEId));
        }
        /* for elemental */
        if (TYPE_IS_ELEMENTAL(myId) ||
            PROC_IS_ELEMENTAL(myId)) {
            TYPE_SET_ELEMENTAL(ID_TYPE(myId));
            TYPE_SET_ELEMENTAL(EXT_PROC_TYPE(myEId));
        }
    }

    /*
     * Then fix variable and proc definition so far
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        if (classNeedFix(ip)) {
            /*
             * not declare_variable() but implicit_declaration(), cuz
             * gotta decide type at least.
             */
            implicit_declaration(ip);
        }
#ifdef not      /* cannot decide type of argument until reference */
        if (ID_STORAGE(ip) == STG_ARG && ID_CLASS(ip) == CL_UNKNOWN) {
            declare_variable(ip);
        }
#endif
        checkTypeRef(ip);
        union_parent_type(ip);
    }

    /*
     * Then, before fix arrays dimension, compile type for any IDs if
     * it is not compiled yet.
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        fix_type(ip);
    }

    /*
     * Fix arrays dimension recursively.
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        fix_array_dimensions_recursive(ip);
    }

    /*
     * Fix pointee (is_target) recursively.
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        fix_pointer_pointee_recursive(ID_TYPE(ip));
    }

    /*
     * Fix type attributes.
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {

        struct type_attr_check *check;
        EXT_ID ep;

        tp = ID_TYPE(ip);

        if (tp == NULL) {
            implicit_declaration(ip);
            tp = ID_TYPE(ip);
        }

        /* fix external identifier whose type is not fixed */
        if (tp == NULL && 
            ID_CLASS(ip) == CL_PROC &&
            PROC_CLASS(ip) == P_EXTERNAL) {
            ep = find_ext_id(ID_SYM(ip));
            if (ep != NULL && EXT_PROC_TYPE(ep) != NULL) {
                tp = EXT_PROC_TYPE(ep);
            } else {
                tp = new_type_subr();
                PROC_IS_FUNC_SUBR_AMBIGUOUS(ip) = TRUE;
            }
            declare_id_type(ip, tp);
        }

        if(tp == NULL)
            continue;

        /* public or private attribute is handled only in module. */
        if (CURRENT_PROC_CLASS == CL_MODULE) {
            if (ID_MAY_HAVE_ACCECIBILITY(ip) && !isAlreadyMarked(ip)) {
                if (current_module_state == M_PUBLIC) {
                    TYPE_SET_PUBLIC(ip);
                }
                if (current_module_state == M_PRIVATE) {
                    TYPE_SET_PRIVATE(ip);
                }
            }
        }

        /* merge type attribute flags except SAVE attr*/
        TYPE_ATTR_FLAGS(tp) |= (TYPE_ATTR_FLAGS(ip) & ~TYPE_ATTR_SAVE);

        /* copy type attribute flags to EXT_PROC_TYPE */
        ep = PROC_EXT_ID(ip);
        if (ep != NULL && EXT_PROC_TYPE(ep) != NULL)
            TYPE_ATTR_FLAGS(EXT_PROC_TYPE(ep)) = TYPE_ATTR_FLAGS(tp);

        if (TYPE_IS_EXTERNAL(tp)) {
            if(ID_STORAGE(ip) != STG_ARG)
                ID_STORAGE(ip) = STG_EXT;
            if(PROC_CLASS(ip) == P_UNKNOWN)
                PROC_CLASS(ip) = P_EXTERNAL;
            else if(PROC_CLASS(ip) != P_EXTERNAL) {
                error_at_id(ip, "invalid external declaration");
                continue;
            }
            if(ID_CLASS(ip) == CL_UNKNOWN)
                ID_CLASS(ip) = CL_PROC;
            else if(ID_CLASS(ip) != CL_PROC && ID_STORAGE(ip) != STG_ARG) {
                error_at_id(ip, "invalid external declaration");
                continue;
            }
        }

        if (ID_CLASS(ip) == CL_MAIN ||
            ID_CLASS(ip) == CL_PROC ||
            ID_CLASS(ip) == CL_ENTRY) {
            continue;
        }

        /* for save */
        if (TYPE_IS_PARAMETER(tp) ||
            TYPE_IS_PARAMETER(ip) ||
            ID_STORAGE(ip) == STG_ARG ||
            (IS_ARRAY_TYPE(tp) &&
             is_array_size_const(tp) == FALSE &&
             !TYPE_IS_ALLOCATABLE(tp) &&
             !TYPE_IS_POINTER(tp))) {
            /*
             * parameter, dummy args, variable size array
             * must not saved.
             */
            TYPE_UNSET_SAVE(ip);
            TYPE_UNSET_SAVE(tp);
        } else if ((TYPE_IS_SAVE(ip) || current_proc_state == P_SAVE) &&
                   (ID_CLASS(ip) != CL_TAGNAME)) {
            /*
             * others can be saved.
             */
            TYPE_SET_SAVE(tp);
        }

        if (ID_CLASS(ip) != CL_MODULE) {
            /* multiple type attribute check */
            for (check = type_attr_checker; check->flag; check++) {
                if (TYPE_ATTR_FLAGS(tp) & check->flag) {
                    uint32_t a = TYPE_ATTR_FLAGS(tp) & 
                        ~check->acceptable_flags;
                    if (debug_flag) {
                        fprintf(stderr,
                                "ID '%s' attr 0x%08x : "
                                "matches 0x%08x ('%s'), "
                                "flags allowed 0x%08x (negation: 0x%08x), "
                                "logical AND: 0x%08x\n",
                                ID_NAME(ip), TYPE_ATTR_FLAGS(tp),
                                check->flag,
                                check->flag_name,
                                check->acceptable_flags,
                                ~check->acceptable_flags,
                                a);
                    }
                    if (TYPE_ATTR_FLAGS(tp) & ~check->acceptable_flags) {
                        struct type_attr_check *e;
                        for (e = type_attr_checker; e->flag; e++) {
                            if (TYPE_ATTR_FLAGS(tp) & e->flag) {
                                fprintf(stderr, "%s has %s\n", ID_NAME(ip),
                                        e->flag_name);
                            }
                        }
                        fatal("type attr error: "
                              "symbol=%s attribute=%s flags=0x%08x",
                              ID_NAME(ip),
                              check->flag_name,
                              TYPE_ATTR_FLAGS(tp));
                    }
                }
            }
        }
    }

    /*
     * Generate PROC_EXT_ID and function_type() for externals.
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        if (PROC_CLASS(ip) == P_EXTERNAL && PROC_EXT_ID(ip) == NULL) {
            assert(ID_TYPE(ip));

            /* don't call declare_externa_id() */
            EXT_ID ep = new_external_id_for_external_decl(
                ID_SYM(ip), ID_TYPE(ip));
            PROC_EXT_ID(ip) = ep;
        }
    }

    /*
     * If a variable is in a common, it should't have save attr
     * otherwise gfortran can't live like that. So make it sure that
     * those variable don't have save attr.
     */
    FOREACH_ID (ip, LOCAL_COMMON_SYMBOLS) {
        if (COM_IS_SAVE(ip)) {
            FOR_ITEMS_IN_LIST(lp, COM_VARS(ip)) {
                v = LIST_ITEM(lp);
                vId = find_ident(EXPV_NAME(v));
                if (vId != NULL) {
                    TYPE_UNSET_SAVE(vId);
                    tp = ID_TYPE(vId);
                    if (tp != NULL) {
                        TYPE_UNSET_SAVE(tp);
                    }
                }
            }
        }
    }

    /*
     * Eliminate save attr from dummy args.
     */
    if (CURRENT_EXT_ID != NULL) {
        unset_save_attr_in_dummy_args(CURRENT_EXT_ID);
    }
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        if (ID_CLASS(ip) == CL_ENTRY ||
            ID_CLASS(ip) == CL_PROC) {
            ep = find_ext_id(ID_SYM(ip));
            if (ep == NULL) {
                ep = PROC_EXT_ID(ip);
            }
            if (ep != NULL) {
                unset_save_attr_in_dummy_args(ep);
            }                
        }
    }

    /*
     * Fix dummy argument types
     */

    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        if(ID_CLASS(ip) != CL_PROC)
            continue;
        ep = find_ext_id(ID_SYM(ip));
        if(ep == NULL || EXT_PROC_ARGS(ep) == NULL)
            continue;
        FOR_ITEMS_IN_LIST (lp, EXT_PROC_ARGS(ep)) {
            expv varg, vid;
            ID idarg;
            
            varg = LIST_ITEM(lp);
            vid = EXPR_ARG1(varg);
            idarg = find_ident(EXPR_SYM(vid));
            if(idarg == NULL)
                continue;
            if(ID_CLASS(idarg) == CL_PROC) {
                // for high order function
                EXPV_PROC_EXT_ID(vid) = PROC_EXT_ID(idarg);
            }
            EXPV_TYPE(vid) = ID_TYPE(idarg);
        }
    }

    /*
     * Check errors
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        
        tp = ID_TYPE(ip);

        if (tp) {
            if (TYPE_IS_ALLOCATABLE(tp) && IS_ARRAY_TYPE(tp) == FALSE &&
		!tp->codims) {
                error_at_id(ip, "ALLOCATABLE is applied only to array");
            } else if (TYPE_IS_OPTIONAL(tp) && !(ID_IS_DUMMY_ARG(ip))) {
                warning_at_id(ip, "OPTIONAL is applied only "
                              "to dummy argument");
            } else if ((TYPE_IS_INTENT_IN(tp) ||
                        TYPE_IS_INTENT_OUT(tp) ||
                        TYPE_IS_INTENT_INOUT(tp)) &&
                       !(ID_IS_DUMMY_ARG(ip))) {
                warning_at_id(ip, "INTENT is applied only "
                              "to dummy argument");
            }
        }
    }


#if 0
    if (myId != NULL &&
        ID_CLASS(myId) == CL_PROC) {
        /*
         * One more, fix 
         */
        if (myEId != NULL) {
            expv idAddrV;
            expv identV;
            list fixedArgs = list

            FOR_ITEMS_IN_LIST(lp, EXT_PROC_ARGS(myEId)) {
                v = LIST_ITEM(lp);
                if (v != NULL) {
                    identV = EXPR_ARG1(v);
                    idAddrV = EXPR_ARG2(v);

                    if (identV == NULL) {
                        fatal("%s: no named argument??",
                              __func__);
                        /* not reached. */
                        return;
                    }
                    if (EXPV_CODE(identV) != IDENT) {
                        fatal("%s: not an ident.", __func__);
                        /* not reached. */
                        return;
                    }

                    if (idAddrV != NULL) {
                        /*
                         * already declared.
                         */
                        continue;
                    }

                    vId = find_ident(EXPV_NAME(identV));
                    if (vId == NULL) {
                        fatal("%s: '%s' is not declared.",
                              __func__, EXPV_NAME(identV));
                    }
                    declare_variable(vId);
                }
            }
        }
    }
#endif

    FOR_ITEMS_IN_LIST (lp, UNIT_CTL_EQUIV_DECLS(uc)) {
        compile_EQUIVALENCE_decl(LIST_ITEM(lp));
    }

    /* execute postponed compilation of initial values */
    FOR_ITEMS_IN_LIST (lp, CURRENT_INITIALIZE_DECLS) {
        v = LIST_ITEM(lp);
        switch (EXPR_CODE(v)) {
        case F_PARAM_DECL:
            postproc_PARAM_decl(EXPR_ARG1(v), EXPR_ARG2(v));
            break;
        case F_DATA_DECL:
            compile_DATA_decl(EXPR_ARG1(v));
            break;
        default:
            continue;
        }
    }
    delete_list(CURRENT_INITIALIZE_DECLS);
    CURRENT_INITIALIZE_DECLS = EMPTY_LIST;

    /*
     * Finally, ready for exec-statements.
     */
    if (CURRENT_PROCEDURE != NULL) {
        /*
         * Mark here.
         */
        output_statement(make_enode(FIRST_EXECUTION_POINT, (void *)NULL));
    }
}


EXT_ID
define_external_function_id(ID id) {
    expr args;
    TYPE_DESC tp = NULL;
    TYPE_DESC tq = NULL;
    list lp;
    SYMBOL sp;
    ID ip;
    expr x;
    EXT_ID ext_id = NULL;

    if (ID_TYPE(id) == NULL || TYPE_IS_IMPLICIT(ID_TYPE(id))) {
        /*
         * The type is not yet fixed.
         * or type is implicit.
         */
        if (PROC_RESULTVAR(id) != NULL) {
            /*
             * The procedure has a result variable, check the type of
             * it.
             */
            ID resId = find_ident(EXPR_SYM(PROC_RESULTVAR(id)));
            if (resId == NULL) {
                resId = declare_ident(EXPR_SYM(PROC_RESULTVAR(id)), CL_VAR);
            }
            if (ID_TYPE(resId) != NULL) {
                tp = ID_TYPE(resId);
            }
        }
    } else {
        tp = ID_TYPE(id);
    }
    if (tp == NULL) {
        /*
         * Both the id and resId has no TYPE_DESC. Try implicit.
         */
        implicit_declaration(id);
        tp = ID_TYPE(id);
    }

    /* inherits the public or private attribute from the parent */
    if (!tp){
      tp = new_type_desc();
      TYPE_BASIC_TYPE(tp) = TYPE_GNUMERIC_ALL;
      ID_TYPE(id) = tp;
    }
    ID pid;
    if (tp && (pid = find_ident_parent(ID_SYM(id)))){
      if (TYPE_IS_PUBLIC(pid)) TYPE_SET_PUBLIC(tp);
      else if (TYPE_IS_PRIVATE(pid)) TYPE_SET_PRIVATE(tp);
    }

    args = EMPTY_LIST;
    /* make external entry */
    ext_id = declare_external_proc_id(ID_SYM(id), tp, TRUE);

    /* copy arg list */
    FOR_ITEMS_IN_LIST(lp, PROC_ARGS(id)){
        x = LIST_ITEM(lp);
        if (EXPR_CODE(x) != IDENT) {
            error("%s: not ident", __func__);
            return NULL;
        }
        sp = EXPR_SYM(x);
        if ((ip = find_ident(sp)) == NULL) {
            fatal("%s: ident is not found", __func__);
        }

        if (ID_CLASS(ip) == CL_PROC) {
            /* dummy procedure must be declared by 'external' */
            implicit_declaration(ip);
            /* make ID_ADDR */
            tq = function_type(ID_TYPE(ip));
            ID_ADDR(ip) = expv_sym_term(F_FUNC, tq, ID_SYM(ip));
        } else {
#if 0
            declare_variable(ip);
            if (ID_ADDR(ip) == NULL) {
                error("'%s' is not declared", ID_NAME(ip));
                return NULL;
            }
            tq = ID_TYPE(ip);
#else
            tq = NULL;
            /*
             * Don't fix type/class of the ip here, let it be fixed by
             * end_declaration().
             */
#endif
        }
        x = list2(LIST, expv_sym_term(IDENT, tq, sp), ID_ADDR(ip));
        list_put_last(args, x);
    }

    EXT_PROC_ARGS(ext_id) = args;
    EXT_PROC_TYPE(ext_id) = tp;

    return ext_id;
}


static void
setLocalInfoToCurrentExtId(int asModule)
{
    EXT_PROC_BODY(CURRENT_EXT_ID) = CURRENT_STATEMENTS;
    EXT_PROC_ID_LIST(CURRENT_EXT_ID) = LOCAL_SYMBOLS;
    EXT_PROC_STRUCT_DECLS(CURRENT_EXT_ID) = LOCAL_STRUCT_DECLS;

    if(asModule) {
        EXT_PROC_COMMON_ID_LIST(CURRENT_EXT_ID) = NULL;
        EXT_PROC_LABEL_LIST(CURRENT_EXT_ID) = NULL;
    } else {
        EXT_PROC_COMMON_ID_LIST(CURRENT_EXT_ID) = LOCAL_COMMON_SYMBOLS;
        EXT_PROC_LABEL_LIST(CURRENT_EXT_ID) = LOCAL_LABELS;
    }
}


static void define_internal_subprog(EXT_ID child_ext_ids);

static void
end_contains()
{
    EXT_ID localExtSyms;

    if (PARENT_STATE != INCONT) {
        fatal("unexpected end of CONTAINS");
    }

    if(PARENT_CONTAINS == NULL) {
        PARENT_CONTAINS = LOCAL_EXTERNAL_SYMBOLS;
    } else {
        error("multiple CONTAINS");
        goto error;
    }

    localExtSyms = LOCAL_EXTERNAL_SYMBOLS;
    define_internal_subprog(localExtSyms);
    pop_unit_ctl();

    return;

  error:

    pop_unit_ctl();
    return;
}

/**
 * search for the defined procedure from the unit ctl procedure stack.
 */
static EXT_ID
procedure_defined(ID f_id, EXT_ID unit_ctl_procs[], int redefine_unit_ctl_level)
{
    EXT_ID ep, defined_proc;
    int i;

    if(f_id == NULL || ID_CLASS(f_id) != CL_PROC) {
        if(debug_flag)
            warning("unexpected id '%s' in '%s', id is not procedure",ID_NAME(f_id), __func__);
        return NULL;
    }

    if(PROC_CLASS(f_id) != P_UNDEFINEDPROC) {
        if(debug_flag)
            warning("unexpected id '%s' in '%s', id is already defined",ID_NAME(f_id), __func__);
        return NULL;
    }

    for(i = redefine_unit_ctl_level; i >= 0; i--) {
        defined_proc = unit_ctl_procs[i];
        FOREACH_EXT_ID(ep, defined_proc) {
            if(EXT_SYM(ep) == ID_SYM(f_id))
                return ep;
        }
    }

    return NULL;
}

/*
 * fix undefined procedure with already defined procedure.
 */
static void
redefine_procedures(EXT_ID proc, EXT_ID unit_ctl_procs[], int redefine_unit_ctl_level)
{
    EXT_ID ep;
    ID id, local_ids;
    int i;

    if (proc == NULL)
        return;
    unit_ctl_procs[redefine_unit_ctl_level] = EXT_PROC_CONT_EXT_SYMS(proc);

    if(debug_flag) {
        for(i = redefine_unit_ctl_level; i >= 0; i--) fprintf(debug_fp,"  ");
        fprintf(debug_fp,"redefine '%s'\n", SYM_NAME(EXT_SYM(proc)));
        for(i = redefine_unit_ctl_level; i >= 0; i--) fprintf(debug_fp,"  ");
        fprintf(debug_fp,"contain procedure : {\n");
        FOREACH_EXT_ID(ep, unit_ctl_procs[redefine_unit_ctl_level]){
            for(i = redefine_unit_ctl_level; i >= 0; i--) fprintf(debug_fp,"  ");
            fprintf(debug_fp,"  %s\n", SYM_NAME(EXT_SYM(ep)));
        }
        for(i = redefine_unit_ctl_level; i >= 0; i--) fprintf(debug_fp,"  ");
        fprintf(debug_fp,"}\n");
    }

    FOREACH_EXT_ID(ep, EXT_PROC_CONT_EXT_SYMS(proc)) {
        /* redefine recursive. */
        redefine_procedures(ep, unit_ctl_procs, redefine_unit_ctl_level + 1);
    }

    local_ids = EXT_PROC_ID_LIST(proc);

    FOREACH_ID(id, local_ids) {
        EXT_ID contained_proc;

        if(ID_CLASS(id) != CL_PROC ||
           PROC_CLASS(id) != P_UNDEFINEDPROC)
            continue;

        contained_proc = procedure_defined(id, unit_ctl_procs, redefine_unit_ctl_level);
        if (contained_proc == NULL) {
            EXT_ID external_proc = NULL;
            PROC_CLASS(id)  = P_EXTERNAL;

            EXT_ID ep;
            FOREACH_EXT_ID(ep, EXTERNAL_SYMBOLS){
                if (EXT_SYM(ep) == ID_SYM(id)){
                    external_proc = ep;
                    break;
                }
            }

            if (!external_proc)
                external_proc = declare_external_proc_id(ID_SYM(id), ID_TYPE(id), TRUE);

            EXT_TAG(external_proc) = STG_EXT;
            PROC_EXT_ID(id) = external_proc;

        } else {
            /* undefine procedure is defined in contains statement. */
            PROC_CLASS(id)  = P_DEFINEDPROC;
            PROC_EXT_ID(id) = contained_proc;
        }
    }
}

/* get rough type size */
static int
get_rough_type_size(TYPE_DESC t)
{
    if(t == NULL)
        return 0;

    ID id;
    expv v;
    int rsz;
    int bt = TYPE_BASIC_TYPE(t);

    switch(bt) {
    case TYPE_INT:
    case TYPE_REAL:
    case TYPE_COMPLEX:
        v = expv_reduce(TYPE_KIND(t), TRUE);
        if (v == NULL || EXPV_CODE(v) != INT_CONSTANT)
            return 4;
        return EXPV_INT_VALUE(v) * (bt == TYPE_COMPLEX ? 2 : 1);
    case TYPE_DREAL:
        return KIND_PARAM_DOUBLE;
    case TYPE_DCOMPLEX:
        return KIND_PARAM_DOUBLE * 2;
    case TYPE_ARRAY:
        v = expv_reduce(TYPE_DIM_SIZE(t), TRUE);
        rsz = get_rough_type_size(TYPE_REF(t));
        if (v == NULL || EXPV_CODE(v) != INT_CONSTANT)
            return rsz;
        return EXPV_INT_VALUE(v) * rsz;
    case TYPE_STRUCT:
        rsz = 0;
        FOREACH_ID(id, TYPE_MEMBER_LIST(t)) {
            rsz += get_rough_type_size(ID_TYPE(id));
        }
        return rsz;
    }

    return 0;
}

/* end of procedure. generate variables, epilogs, and prologs */
static void
end_procedure()
{
    ID id;
    EXT_ID ext;

    if (unit_ctl_level > 0 && CURRENT_PROC_NAME == NULL) {
        /* if CURRENT_PROC_NAME == NULL, then this is the end of CONTAINS */
        end_contains();
    }

    /* Since module procedures may be defined not only in contains block but */
    /* also in used modules, the following code is moved from end_contains. */

    if (CURRENT_PROC_CLASS == CL_MAIN ||
        CURRENT_PROC_CLASS == CL_PROC ||
        CURRENT_PROC_CLASS == CL_MODULE ||
        CURRENT_PROC_CLASS == CL_BLOCK) {
        if (CURRENT_EXT_ID == NULL) {
            /* Any other errors already occured, let compilation carry on. */
            return;
        }
        /* check if module procedures are defined in contains block */
        EXT_ID intr, intrDef, ep;
        FOREACH_EXT_ID(intr, EXT_PROC_INTERFACES(CURRENT_EXT_ID)) {
            int hasSub = FALSE, hasFunc = FALSE;

            if(EXT_IS_BLANK_NAME(intr))
                continue;

            FOREACH_EXT_ID(intrDef, EXT_PROC_INTR_DEF_EXT_IDS(intr)) {
                if(EXT_PROC_IS_MODULE_PROCEDURE(intrDef)) {
                    /*
                     * According to JIS X 3001-1, When module procedure is
                     * declared with "module" keyword, procedure should be
                     * declared in that module. But, gfortran seems not to
                     * implement this check. So, we won't implement this
                     * check too.
                     */
                    ep = NULL;
                    ID id = find_ident(EXT_SYM(intrDef));
                    if(id != NULL
                       && ID_CLASS(id) == CL_PROC
                       && ID_IS_OFMODULE(id)) {
                        // intrDef is use associated module procedure.
                        ep = PROC_EXT_ID(id);
                    } else if (EXT_IS_OFMODULE(intrDef)) {
                        continue;
                    } else if (id != NULL) {
                        ep = PROC_EXT_ID(id);
                    }
                    if(ep == NULL || EXT_TAG(ep) != STG_EXT ||
                        EXT_PROC_TYPE(ep) == NULL) {
                        error("%s is not defined.", SYM_NAME(EXT_SYM(intrDef)));
                        break;
                    }
                    EXT_PROC_TYPE(intrDef) = EXT_PROC_TYPE(ep);
                    EXT_PROC_ARGS(intrDef) = EXT_PROC_ARGS(ep);
                    EXT_PROC_ID_LIST(intrDef) = EXT_PROC_ID_LIST(ep);
                } else {
                    ep = intrDef;
                }
                if(IS_GENERIC_TYPE(EXT_PROC_TYPE(ep))) {
                    continue;
                } else if(IS_SUBR(EXT_PROC_TYPE(ep))) {
                    hasSub = TRUE;
                } else {
                    hasFunc = TRUE;
                }
            }

            if(hasSub && hasFunc) {
                error("function does not belong in a generic subroutine interface");
            }
        }
    }

/*  next: */

    if (endlineno_flag){
      if (CURRENT_PROCEDURE)
	ID_END_LINE_NO(CURRENT_PROCEDURE) = current_line->ln_no;
      else if (CURRENT_EXT_ID && EXT_LINE(CURRENT_EXT_ID))
	EXT_END_LINE_NO(CURRENT_EXT_ID) = current_line->ln_no;
    }

    if(NOT_INDATA_YET) end_declaration();

    /*
     * Automatically add save attributes to varriables whose
     * rough size are larger than auto_save_attr_kb kbytes.
     */
    if(auto_save_attr_kb >= 0 &&
        (CURRENT_PROC_CLASS == CL_PROC || CURRENT_PROC_CLASS == CL_MAIN) &&
        TYPE_IS_RECURSIVE(EXT_PROC_TYPE(CURRENT_EXT_ID)) == FALSE) {

        FOREACH_ID (id, LOCAL_SYMBOLS) {
            int sz;
            TYPE_DESC t = ID_TYPE(id);
            if(ID_STORAGE(id) != STG_AUTO || ID_CLASS(id) == CL_PARAM
                || t == NULL || TYPE_IS_SAVE(t))
                continue;
            sz = get_rough_type_size(ID_TYPE(id));
            if (sz >= (auto_save_attr_kb << 10))
                TYPE_SET_SAVE(ID_TYPE(id));
        }
    }

    FinalizeFormat();

    if (EXT_PROC_TYPE(CURRENT_EXT_ID))
      TYPE_SET_FOR_FUNC_SELF(EXT_PROC_TYPE(CURRENT_EXT_ID));

    /* check undefined variable */
    FOREACH_ID(id, LOCAL_SYMBOLS) {
        if(ID_CLASS(id) == CL_UNKNOWN){
#ifdef not
            warning("variable '%s' is defined, but never used",ID_NAME(id));
#endif
            declare_variable(id);
        }
        if (ID_CLASS(id) == CL_VAR) {
            declare_variable(id);
        }

        if ((ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_THISPROC) ||
            ID_CLASS(id) == CL_ENTRY) {
            PROC_CLASS(id) = P_DEFINEDPROC;
            if(unit_ctl_level != 0) {
                TYPE_DESC tp;
                ID id_in_parent = NULL;
                ID parent_id_list;

                id_in_parent = find_ident_parent(ID_SYM(id));
                parent_id_list = UNIT_CTL_LOCAL_SYMBOLS(PARENT_UNIT_CTL);

                if(id_in_parent == NULL) {
                    ID ip, last_ip;
                    id_in_parent = new_ident_desc(ID_SYM(id));

                    last_ip = NULL;
                    FOREACH_ID(ip, parent_id_list) {
                        last_ip = ip;
                    }
                    ID_LINK_ADD(id_in_parent, parent_id_list, last_ip);
                }

                PROC_ARGS(id_in_parent) = PROC_ARGS(id);
                ID_CLASS(id_in_parent) = ID_CLASS(id);
                ID_STORAGE(id_in_parent) = STG_EXT;
                PROC_EXT_ID(id_in_parent) = PROC_EXT_ID(id);
                PROC_CLASS(id_in_parent) = P_DEFINEDPROC;
                PROC_IS_RECURSIVE(id_in_parent) = PROC_IS_RECURSIVE(id);
                PROC_IS_PURE(id_in_parent) = PROC_IS_PURE(id);
                PROC_IS_ELEMENTAL(id_in_parent) = PROC_IS_ELEMENTAL(id);

                tp = ID_TYPE(id_in_parent);
                ID_TYPE(id_in_parent) = ID_TYPE(id);
                if (tp != NULL) {
                    while(tp != NULL) {
                        if(IS_TYPE_PUBLICORPRIVATE(tp)) {
                            if (TYPE_IS_PUBLIC(tp)) {
                                TYPE_SET_PUBLIC(ID_TYPE(id));
                            }
                            if (TYPE_IS_PRIVATE(tp)) {
                                TYPE_SET_PRIVATE(ID_TYPE(id));
                            }
                            break;
                        }
                        tp = TYPE_REF(tp);
                    }
                } else {
                    if (current_module_state == M_PUBLIC) {
                        TYPE_SET_PUBLIC(ID_TYPE(id));
                    }
                    if (current_module_state == M_PRIVATE) {
                        TYPE_SET_PRIVATE(ID_TYPE(id));
                    }
                }
                ID_DEFINED_BY(id_in_parent) = id;
            }
        }

        if(ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_UNDEFINEDPROC) {
            if(PROC_EXT_ID(id) != NULL) {
                /* undefined procedure is defined in contain statement.  */
                EXT_IS_DEFINED(PROC_EXT_ID(id)) = TRUE;
            } else {
                implicit_declaration(id);
            }
        }
    }

    /* check undefined label */
    FOREACH_ID(id, LOCAL_LABELS) {
        if (LAB_TYPE(id) != LAB_UNKNOWN && 
            LAB_IS_USED(id) && !LAB_IS_DEFINED(id)) {
            error("missing statement number %d", LAB_ST_NO(id));
        }
        checkTypeRef(id);
    }
    
    /*
     * Special case.
     */
    if (CURRENT_STATEMENTS != NULL &&
        EXPV_CODE(CURRENT_STATEMENTS) == FIRST_EXECUTION_POINT) {
        /*
         * Means no body.
         */
        CURRENT_STATEMENTS = NULL;
    }

    /*
     * set self in parent to procedure.
     */
    if(CURRENT_PROC_CLASS == CL_PROC
       && (id = find_ident_parent(CURRENT_PROC_NAME)) != NULL) {
        ID_CLASS(id) = CL_PROC;
    }

    /* output */
    switch (CURRENT_PROC_CLASS) {
    case CL_MAIN:
        setLocalInfoToCurrentExtId(FALSE);
        if(debug_flag){
            fprintf(debug_fp,"\n*** CL_MAIN:\n");
            print_IDs(LOCAL_SYMBOLS, debug_fp, TRUE);
            print_types(LOCAL_STRUCT_DECLS, debug_fp);
            expv_output(CURRENT_STATEMENTS, debug_fp);
        }
        break;
    case CL_BLOCK:
        setLocalInfoToCurrentExtId(FALSE);
        if(debug_flag){
            fprintf(debug_fp,"\n*** CL_BLOCK:\n");
            print_IDs(LOCAL_SYMBOLS, debug_fp,TRUE);
        }
        break;
    case CL_PROC:
        if (CURRENT_EXT_ID != NULL) {
            setLocalInfoToCurrentExtId(FALSE);
        }
        if(debug_flag){
            fprintf(debug_fp,"\n*** CL_PROC('%s'):\n",
                    SYM_NAME(CURRENT_PROC_NAME));
            print_IDs(LOCAL_SYMBOLS, debug_fp,TRUE);
            print_types(LOCAL_STRUCT_DECLS, debug_fp);
            expv_output(CURRENT_STATEMENTS, debug_fp);
        }
        break;
    case CL_MODULE:
        setLocalInfoToCurrentExtId(TRUE);
        if(debug_flag){
            fprintf(debug_fp,"\n*** CL_MODULE:\n");
            print_IDs(LOCAL_SYMBOLS, debug_fp,TRUE);
            print_types(LOCAL_STRUCT_DECLS, debug_fp);
            expv_output(CURRENT_STATEMENTS, debug_fp);
        }
        break;
    default:
        fatal("end_procedure: unknown current_proc_class");
    }

    /* resolve undefined procedure recursively. */
    switch (CURRENT_PROC_CLASS) {
    case CL_MAIN:
    case CL_PROC:
    case CL_MODULE: {
        /* EXT_ID list, used as a stack.*/
        EXT_ID unit_ctl_procs[MAX_UNIT_CTL];
        if(unit_ctl_level != 0)
            break;
        ext = UNIT_CTL_CURRENT_EXT_ID(CURRENT_UNIT_CTL);
        if(ext == NULL)
            break;
        redefine_procedures(ext, unit_ctl_procs, unit_ctl_level);
    } break;
    default:
        break;
    }

    fixup_all_module_procedures();
    if (debug_flag) {
        dump_all_module_procedures(stderr);
    }

    if(CURRENT_PROC_CLASS == CL_MODULE) {
        SYMBOL sym = find_symbol(current_module_name);
        if(!export_module(sym, LOCAL_SYMBOLS,
                          UNIT_CTL_USE_DECLS(CURRENT_UNIT_CTL))) {
            error("internal error, fail to export module.");
            exit(1);
        }
    }

    /* check control nesting */
    if(ctl_top > ctls) error("DO loop or BLOCK IF not closed");

#ifdef not
    donmlist();
    dobss();
#endif

    /* clean up for next procedure */
    initialize_compile_procedure();
    cleanup_unit_ctl(CURRENT_UNIT_CTL);
}

/* 
 * DO loop
 */
static void
compile_DO_statement(range_st_no, construct_name, var, init, limit, incr)
     int range_st_no;
     expr construct_name, var, init, limit, incr;
{
    expv do_var = NULL, do_init = NULL, do_limit = NULL, do_incr = NULL;
    ID do_label = NULL;
    TYPE_DESC var_tp = NULL;
    SYMBOL do_var_sym = NULL;
    int incsign = 0;
    CTL *cp;

    if (range_st_no > 0) {
        do_label = declare_label(range_st_no, LAB_EXEC, FALSE);
        if (do_label == NULL) return;
        if (LAB_IS_DEFINED(do_label)) {
            error("no backward DO loops");
            return;
        }
        /* turn off, becuase this is not branch */
        LAB_IS_USED(do_label) = FALSE;
    }

    if(var || init || limit || incr) {
        if (EXPR_CODE(var) != IDENT) {
            fatal("compile_DO_statement: DO var is not IDENT");
        }
        do_var_sym = EXPR_SYM(var);
        
        /* check nested loop with the same variable */
        for (cp = ctls; cp < ctl_top; cp++) {
            if(CTL_TYPE(cp) == CTL_DO && CTL_DO_VAR(cp) == do_var_sym) {
                error("nested loops with variable '%s'", SYM_NAME(do_var_sym));
                break;
            }
        }

        do_var = compile_lhs_expression(var);
        if (!expv_is_lvalue(do_var)) error("bad DO variable");

        do_init = expv_reduce(compile_expression(init), FALSE);
        do_limit = expv_reduce(compile_expression(limit), FALSE);
        if (incr != NULL) do_incr = expv_reduce(compile_expression(incr),
                                                FALSE);
        else do_incr = expv_constant_1;

        if (do_var == NULL || do_init == NULL || 
            do_limit == NULL || do_incr == NULL) return;
        
        var_tp = EXPV_TYPE(do_var);
        if (!IS_INT(var_tp) && !IS_REAL(var_tp)) {
            error("bad type on do variable");
            return;
        }

        if (!IS_INT_OR_REAL(EXPV_TYPE(do_init)) &&
            !IS_GNUMERIC(EXPV_TYPE(do_init)) &&
            !IS_GNUMERIC_ALL(EXPV_TYPE(do_init))) {
            error("bad type on DO initialize parameter");
            return;
        }

        if (!IS_INT_OR_REAL(EXPV_TYPE(do_limit)) &&
            !IS_GNUMERIC(EXPV_TYPE(do_limit)) &&
            !IS_GNUMERIC_ALL(EXPV_TYPE(do_limit))) {
            error("bad type on DO limitation parameter");
            return;
        }

        if (!IS_INT_OR_REAL(EXPV_TYPE(do_incr)) &&
            !IS_GNUMERIC(EXPV_TYPE(do_incr)) &&
            !IS_GNUMERIC_ALL(EXPV_TYPE(do_incr))) {
            error("bad type on DO increment parameter");
            return;
        }

        if (expr_is_constant(do_incr)) {
            do_incr = expv_reduce_conv_const(var_tp, do_incr);
            if (EXPV_CODE(do_incr) == INT_CONSTANT) {
                if(EXPV_INT_VALUE(do_incr) == 0)
                    error("zero DO increment");
                else if(EXPV_INT_VALUE(do_incr) > 0)
                    incsign = 1;
                else
                    incsign = -1;
            }
            /* cannot check if do_incr is FLOAT_CONSTANT
             * because FLOAT_CONSTANT cannot be reduced */
        }
        
        if (expr_is_constant(do_limit)) {
            do_limit = expv_reduce_conv_const(var_tp, do_limit);
        }

        if (expr_is_constant(do_init)) {
            do_init = expv_reduce_conv_const(var_tp, do_init);
        }

        if (expr_is_constant(do_limit) && expr_is_constant(do_init)) {
            if (incsign > 0) {              /* increment */
                if ((IS_INT(var_tp) && 
                     EXPV_INT_VALUE(do_limit) < EXPV_INT_VALUE(do_init))) {
                    warning("DO range never executed");
                }
            } else if (incsign < 0) {       /* decrement */
                if ((IS_INT(var_tp) && 
                     EXPV_INT_VALUE(do_limit) > EXPV_INT_VALUE(do_init))) {
                    warning("DO range never executed");
                }
            }
        }
    }

    push_ctl(CTL_DO);
    CTL_DO_VAR(ctl_top) = do_var_sym;
    CTL_DO_LABEL(ctl_top) = do_label;

    /* 
     * output DO loop in Fortran90
     */
    CTL_BLOCK(ctl_top) = list2(F_DO_STATEMENT,
                               construct_name,
                               list5(LIST,
                                     do_var, do_init, do_limit, do_incr,
                                     NULL));
}

static void  compile_DOWHILE_statement(range_st_no, cond, construct_name)
     int range_st_no;
     expr cond, construct_name;
{
    expv v;
    ID do_label = NULL;

    if(cond == NULL) return; /* error recovery */

    if (range_st_no > 0) {
        do_label = declare_label(range_st_no, LAB_EXEC, FALSE);
        if (do_label == NULL) return;
        if (LAB_IS_DEFINED(do_label)) {
            error("no backward DO loops");
            return;
        }
        /* turn off, becuase this is not branch */
        LAB_IS_USED(do_label) = FALSE;
    }

    v = compile_expression(cond);
    push_ctl(CTL_DO);
    CTL_DO_VAR(ctl_top) = NULL;
    CTL_DO_LABEL(ctl_top) = do_label;
    CTL_BLOCK(ctl_top) = list3(F_DOWHILE_STATEMENT,v,NULL,construct_name);
}

static void
check_DO_end(ID label)
{
    CTL *cp;

    if (label == NULL) {
        /*
         * do ... enddo case.
         */
        if (CTL_TYPE(ctl_top) == CTL_DO) {
            if (EXPR_CODE(CTL_BLOCK(ctl_top)) == F_DOWHILE_STATEMENT) {
                /*
                 * DOWHILE
                 */
                if (CTL_DO_LABEL(ctl_top) != NULL) {
                    /*
                     * An obsolete/unexpected syntax like:
                     *	      do 300 while (.true.)
                     *          ...
                     *  300   end do
                     * warn just for our mental health.
                     */
                    warning("Unexpected (maybe obsolete) syntax of "
                            "DO WHILE - ENDDO statements, "
                            "DO WHILE having a statement label '%s' "
                            "and ended ENDDO.",
                            SYM_NAME(ID_SYM(CTL_DO_LABEL(ctl_top))));
                }
                EXPR_ARG2(CTL_BLOCK(ctl_top)) = CURRENT_STATEMENTS;
            } else {
                /*
                 * else DO_STATEMENT
                 */  
                if (CTL_DO_LABEL(ctl_top) != NULL) {
                    /*
                     * An obsolete/unexpected syntax like:
                     *	      do 300 i = 1, 10
                     *          ...
                     *  300   end do
                     * warn just for our mental health.
                     */
                    warning("Unexpected (maybe obsolete) syntax of "
                            "DO - ENDDO statements, "
                            "DO having a statement label '%s' "
                            "and ended ENDDO.",
                            SYM_NAME(ID_SYM(CTL_DO_LABEL(ctl_top))));
                }
                CTL_DO_BODY(ctl_top) = CURRENT_STATEMENTS;
            }

	    if (endlineno_flag)
	      EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();
        } else {
            error("'do' is not found for 'enddo'");
        }
        return;
    }

    // do - continue case

    while (CTL_TYPE(ctl_top) == CTL_DO && 
           CTL_DO_LABEL(ctl_top) == label) {

      /* close DO block */
      if (EXPR_CODE(CTL_BLOCK(ctl_top)) == F_DOWHILE_STATEMENT) {
	/*
	 * DOWHILE
	 */
	EXPR_ARG2(CTL_BLOCK(ctl_top)) = CURRENT_STATEMENTS;
      }
      else {
	/*
	 * else DO
	 */  
        CTL_DO_BODY(ctl_top) = CURRENT_STATEMENTS;
      }

      if (endlineno_flag)
	EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

      pop_ctl();
    }

    /* check DO loop which is not propery closed. */
    for (cp = ctl_top; cp >= ctls; cp--) {
        if (CTL_TYPE(cp) == CTL_DO && CTL_DO_LABEL(cp) == label) {
            error("DO loop or IF-block not closed");
            ctl_top = cp;
            pop_ctl();
        }
    }
}

/* line number for module begin for MC.  */
static int module_start_ln_no;
extern int last_ln_no;

/* set the module from NAME.  */
void
begin_module(expr name)
{
    SYMBOL s;
    if (name) {
        if (EXPR_CODE(name) == IDENT &&
            (s = EXPR_SYM(name)) != NULL &&
            SYM_NAME(s) != NULL) {
            /*
             * call the module_procedure_manager_init() very here, not
             * after the current_module_name != NULL.
             */
            module_procedure_manager_init();
            current_module_name = SYM_NAME(s);
            module_start_ln_no = last_ln_no;
            module_start_offset = prelast_initial_line_pos;
        } else {
            fatal("internal error, module name is not "
                  "IDENT in %s().", __func__);
            /* not reached. */
        }
    } else {
        fatal("internal error, module name is NULL in %s().", __func__);
        /* not reached. */
    }
}

/*
 * compile END MODULE statement and
 * output module's XcodeML file.
 */
void
end_module() {
    current_module_state = M_DEFAULT;
    current_module_name = NULL;
    CURRENT_STATE = OUTSIDE; /* goto outer, outside state.  */
}

int
is_in_module(void) {
    return (INMODULE()) ? TRUE : FALSE;
}

const char *
get_current_module_name(void) {
    return current_module_name;
}

struct use_argument {
    struct use_argument * next;
    SYMBOL use;   /* use name or NULL*/
    SYMBOL local; /* local name, not NULL */
    int used;
};

#define FOREACH_USE_ARG(arg, arg_list)\
    for((arg) = (args); (arg) != NULL; (arg) = (arg)->next)

extern ID find_ident_head(SYMBOL s, ID head);

static void
import_module_procedure(const char * genName, EXT_ID mep) {
    TYPE_DESC tp = EXT_PROC_TYPE(mep);
    expr modArgs = EXT_PROC_ARGS(mep);
    // TODO(shingo-s): fix name if the module procedure is private.
    const char * modName = SYM_NAME(EXT_SYM(mep));
    mod_proc_t mp = add_module_procedure(genName,
                                         modName,
                                         tp,
                                         modArgs,
                                         NULL);
    MOD_PROC_EXT_ID(mp) = mep;
}

/**
 * import id as generic procedure.
 */
static void
import_generic_procedure(ID id) {
    EXT_ID ep;
    EXT_ID modProcs = NULL;
    EXT_ID aProc;

    const char *genName = SYM_NAME(ID_SYM(id));
    add_generic_procedure(genName, NULL);

    ep = PROC_EXT_ID(id);
    modProcs = EXT_PROC_INTR_DEF_EXT_IDS(ep);

    FOREACH_EXT_ID(aProc, modProcs) {
        import_module_procedure(genName, aProc);
    }
}

static EXT_ID
shallow_copy_ext_id(EXT_ID original) {
    EXT_ID ret = NULL, ep, new_ep;
    FOREACH_EXT_ID(ep, original) {
        if (ep == original) {
            new_ep = new_external_id(EXT_SYM(ep));
            ret = new_ep;
        } else {
            EXT_NEXT(new_ep) = new_external_id(EXT_SYM(ep));
            new_ep = EXT_NEXT(new_ep);
        }
        *new_ep = *ep;
        EXT_NEXT(new_ep) = NULL;
    }
    return ret;
}

#define ID_SEEM_GENERIC_PROCEDURE(id)                                   \
    (ID_TYPE((id)) != NULL &&                                           \
     ((ID_CLASS((id)) == CL_PROC &&                                     \
       TYPE_BASIC_TYPE(ID_TYPE((id))) == TYPE_GENERIC) ||               \
      (TYPE_BASIC_TYPE(ID_TYPE((id))) == TYPE_FUNCTION &&               \
       TYPE_REF(ID_TYPE((id))) != NULL &&                               \
       TYPE_BASIC_TYPE(TYPE_REF(ID_TYPE((id)))) == TYPE_GNUMERIC_ALL)))

struct replicated_type {
  TYPE_DESC original;
  TYPE_DESC replica;
  struct replicated_type * next;
};

struct replicated_type * replicated_type_list = NULL;

static void
initialize_replicated_type_list() {
    replicated_type_list = NULL;
}

static void
finalize_replicated_type_list() {
    struct replicated_type * lp;
    while(replicated_type_list != NULL) {
        lp = replicated_type_list->next;
        free(replicated_type_list);
        replicated_type_list = lp;
    }
}

static void
append_replicated_type_list(const TYPE_DESC original,
                            const TYPE_DESC replica) {
    struct replicated_type * lp;
    if(original != NULL && replica != NULL) {
        lp = XMALLOC(struct replicated_type *,sizeof(struct replicated_type));
        lp->original = original;
        lp->replica = replica;
        lp->next = replicated_type_list;
        replicated_type_list = lp;
    }
}

/**
 * Checks if a type has the replica of itself.
 *
 * @param replica if tp has the replica, then set replica to it.
 */
static int
type_has_replica(const TYPE_DESC tp, TYPE_DESC * replica) {
    struct replicated_type * lp;
    if (tp != NULL) {
        for(lp = replicated_type_list; lp != NULL; lp = lp->next) {
            if(tp == lp->original) {
                if(replica != NULL) {
                    *replica = lp->replica;
                }
                return TRUE;
            }
        }
    }
    return FALSE;
}

/**
 * Checks if a type is the replicated one.
 */
static int
type_is_replica(const TYPE_DESC tp) {
    struct replicated_type * lp;
    if(tp != NULL) {
        for(lp = replicated_type_list; lp != NULL; lp = lp->next) {
            if(tp == lp->replica) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

/**
 * Creates the type thas is shallow copied for the module id.
 */
static TYPE_DESC
shallow_copy_type_for_module_id(TYPE_DESC original) {
    TYPE_DESC new_tp;

    new_tp = new_type_desc();
    *new_tp = *original;

    /* PUBLIC/PRIVATE attribute may be given by the module user */
    TYPE_UNSET_PUBLIC(new_tp);
    TYPE_UNSET_PRIVATE(new_tp);

    append_replicated_type_list(original, new_tp);

    return new_tp;
}

static void
deep_copy_and_overwrite_for_module_id_type(TYPE_DESC * ptp);

/**
 * Copy the reference types recursively
 *  until there is no reference type or
 *  the reference type is already replicated.
 */
static void
deep_ref_copy_for_module_id_type(TYPE_DESC tp) {
    ID id;
    TYPE_DESC itp, cur, old;
    cur = tp;
    while(TYPE_REF(cur) != NULL) {
        old = TYPE_REF(cur);
        deep_copy_and_overwrite_for_module_id_type(&(TYPE_REF(cur)));
        if (old == TYPE_REF(cur))
            break;
        cur = TYPE_REF(cur);
    }

    if(IS_STRUCT_TYPE(cur)) {
        FOREACH_MEMBER(id, cur) {
            itp = ID_TYPE(id);
            deep_copy_and_overwrite_for_module_id_type(&(ID_TYPE(id)));
        }
    }
}

/**
 * Deep-copy the type and overwrite it
 */
static void
deep_copy_and_overwrite_for_module_id_type(TYPE_DESC * ptp) {
    TYPE_DESC tp;

    if(ptp == NULL || (*ptp == NULL)) {
      return;
    }

    if (type_is_replica(*ptp)) {
      ;  /* do nothing */
    } else if (type_has_replica(*ptp, &tp)) {
      /* overwrite the type with replicated one */
      *ptp = tp;
    } else {
      /* shallow-copy type and deep-copy the  type referenced by this type */
      *ptp
          = shallow_copy_type_for_module_id(*ptp);
      deep_ref_copy_for_module_id_type(*ptp);
    }
}



/**
 * solve conflict between local identifier and use associated identifier.
 *
 * @id local identifier (only LOCAL, neither parent identifier nor sibling one)
 * @mid use associated identifier
 */
static void
solve_use_assoc_conflict(ID id, ID mid)
{
    if(ID_SEEM_GENERIC_PROCEDURE(id) && ID_SEEM_GENERIC_PROCEDURE(mid)) {
        // ignore a conflict between generic functions.
        /* NOTE:
         * Generic functions with functions with different type of arguments is not conflict.
         * Generic function occurres a conflict if it cotains functions with same type of arguments,
         * but the current type system couldn't detect it.
         */
        EXT_ID current_ep, module_ep, head, ep;
        if(IS_GENERIC_TYPE(ID_TYPE(mid))) {
            import_generic_procedure(mid);
        }
        current_ep = PROC_EXT_ID(id);
        module_ep = PROC_EXT_ID(mid);

        if (!EXT_PROC_INTR_DEF_EXT_IDS(module_ep))
            return;
        head = shallow_copy_ext_id(EXT_PROC_INTR_DEF_EXT_IDS(module_ep));
        FOREACH_EXT_ID(ep, head) {
            EXT_IS_OFMODULE(ep) = TRUE;
        }

        if (EXT_PROC_INTR_DEF_EXT_IDS(current_ep)) {
            extid_put_last(EXT_PROC_INTR_DEF_EXT_IDS(current_ep), head);
        } else if (EXT_PROC_INTR_DEF_EXT_IDS(current_ep) == NULL) {
            EXT_PROC_INTR_DEF_EXT_IDS(current_ep) = head;
        }
        return;
    }
    if(!id->use_assoc) {
        // conflict between (sub)program, argument, or module
        /* NOTE:
         * If id is not use associated,
         * id is (sub)program name, argument name, or module name.
         * It is because that USE statement appear before any declaration.
         */
        if(debug_flag) {
            fprintf(debug_fp,
                    "conflict symbol '%s' between current scope and module '%s'\n",
                    SYM_NAME(ID_SYM(mid)),
                    SYM_NAME(mid->use_assoc->module_name));
        }
        id->use_assoc_conflicted = TRUE;
    } else {
        // conflict between use associated ids
        /* NOTE:
         * If two ids are defined with same name, and in same module,
         * two ids are same one. So there are no conflict.
         */
        if((id->use_assoc->module_name == mid->use_assoc->module_name)
           && (id->use_assoc->original_name == mid->use_assoc->original_name)) {
            // DO NOTHING
            if(debug_flag) {
                fprintf(debug_fp,
                        "duplicate use assoc symbol '%s' (original '%s') from module '%s'\n",
                        SYM_NAME(ID_SYM(mid)),
                        SYM_NAME(mid->use_assoc->original_name),
                        SYM_NAME(mid->use_assoc->module_name));
            }
        } else {
            if(debug_flag) {
                fprintf(debug_fp,
                        "conflict symbol '%s' between the followings\n"
                        " - original '%s' from module '%s'\n"
                        " - original '%s' from module '%s'\n",
                        SYM_NAME(ID_SYM(id)),
                        SYM_NAME(id->use_assoc->original_name),
                        SYM_NAME(mid->use_assoc->original_name),
                        SYM_NAME(id->use_assoc->module_name),
                        SYM_NAME(mid->use_assoc->module_name));
            }
            id->use_assoc_conflicted = TRUE;
        }
    }
}

/**
 * import id from module to id list.
 */
static int
import_module_id(ID mid,
                 ID *head, ID *tail,
                 TYPE_DESC *sthead, TYPE_DESC *sttail,
                 EXT_ID *ehead, EXT_ID *etail,
                 SYMBOL use_name, int need_wrap_type)
{
    ID existed_id, id;
    EXT_ID ep, mep;

    if ((existed_id = find_ident_head(use_name?:ID_SYM(mid), *head)) != NULL) {
        solve_use_assoc_conflict(existed_id, mid);
        return TRUE;
    }

    id = new_ident_desc(ID_SYM(mid));
    *id = *mid;

    PROC_EXT_ID(id) = NULL;
    mep = PROC_EXT_ID(mid);
    if (mep != NULL) {
        PROC_EXT_ID(id) = new_external_id(EXT_SYM(mep));
        ep = PROC_EXT_ID(id);
        *ep = *mep;
        EXT_IS_OFMODULE(ep) = TRUE;
        EXT_NEXT(ep) = NULL;
        EXT_PROC_INTR_DEF_EXT_IDS(ep) = NULL;

        /* hmm, this code is really required? */
        if(!type_is_replica(EXT_PROC_TYPE(mep))) {
            EXT_PROC_TYPE(mep)
                    = shallow_copy_type_for_module_id(EXT_PROC_TYPE(mep));
        }

        if (EXT_PROC_INTR_DEF_EXT_IDS(mep) != NULL) {
            EXT_ID head, p;
            head = shallow_copy_ext_id(EXT_PROC_INTR_DEF_EXT_IDS(mep));
            FOREACH_EXT_ID(p, head) {
                EXT_IS_OFMODULE(p) = TRUE;
            }
            EXT_PROC_INTR_DEF_EXT_IDS(ep) = head;
        }
    }

    if(use_name)
        ID_SYM(id) = use_name;

    /*
     * In module, use associated id may be given PUBLIC or PRIVATE
     * attribute. OR, If id is tagname and rename required, then type
     * will be given different tagname.
     */
    if(need_wrap_type || (ID_STORAGE(id) == STG_TAGNAME && use_name)) {
        // shallow copy type from module
        ID_TYPE(id) = shallow_copy_type_for_module_id(ID_TYPE(id));
        TYPE_UNSET_PUBLIC(id);
        TYPE_UNSET_PRIVATE(id);
    }

    if(ID_STORAGE(id) == STG_TAGNAME) {
        TYPE_TAGNAME(ID_TYPE(id)) = id;
        TYPE_SLINK_ADD(ID_TYPE(id), *sthead, *sttail);
    }

    ID_LINK_ADD(id, *head, *tail);

    if(IS_GENERIC_TYPE(ID_TYPE(id)))
        import_generic_procedure(id);

    if(debug_flag) {
        fprintf(debug_fp,
                "import '%s' from module '%s'\n",
                SYM_NAME(ID_SYM(mid)),
                SYM_NAME(mid->use_assoc->module_name));
        if(use_name)
        fprintf(debug_fp,
                "as '%s'",
                SYM_NAME(use_name));
    }

    return TRUE;
}

/**
 * Copy the expv as function argments.
 */
static expv
copy_function_args(const expv args) {
    expv v, new_args, varg, new_varg;
    list lp;
    //TYPE_DESC tp;

    new_args = XMALLOC(expv, sizeof(*new_args));
    *new_args = *args;
    EXPR_LIST(new_args) = NULL;

    FOR_ITEMS_IN_LIST (lp, args) {
        varg = EXPR_ARG1(LIST_ITEM(lp));
        new_varg = XMALLOC(expv, sizeof(*new_varg));
        *new_varg = *varg;

        v = list1(LIST, new_varg);
        list_put_last(new_args, v);
    }

    return new_args;
}

/**
 * common use assoc
 */
int
use_assoc_common(SYMBOL name, struct use_argument * args, int isRename)
{
    struct module *mod;
    ID mid, id, last_id = NULL, prev_mid, first_mid;
    TYPE_DESC tp, sttail = NULL;
    EXT_ID ep, last_ep = NULL, mep;
    struct use_argument * arg;
    int ret = TRUE;
    int wrap_type = TRUE;

    initialize_replicated_type_list();

    if(!import_module(name, &mod)) {
        return FALSE;
    }

    if(debug_flag) {
        fprintf(debug_fp, "######## BEGIN USE ASSOC #######\n");
        print_IDs(mod->head, debug_fp, TRUE);
    }

    FOREACH_ID(id, LOCAL_SYMBOLS) {
        last_id = id;
    }
    prev_mid = last_id;

    for (tp = LOCAL_STRUCT_DECLS; tp != NULL; tp = TYPE_SLINK(tp)) {
        sttail = tp;
    }

    for (ep = LOCAL_EXTERNAL_SYMBOLS; ep != NULL; ep = EXT_NEXT(ep)){
        last_ep = ep;
    }
    ep = LOCAL_EXTERNAL_SYMBOLS;

    FOREACH_ID(mid, mod->head) {
        int rename_count = 0;
        FOREACH_USE_ARG(arg, args) {
            wrap_type = TRUE;
            if(arg->local != ID_SYM(mid))
                continue;
            import_module_id(mid,
                             &LOCAL_SYMBOLS, &last_id,
                             &LOCAL_STRUCT_DECLS, &sttail,
                             &LOCAL_EXTERNAL_SYMBOLS, &last_ep,
                             arg->use, wrap_type);
            arg->used = TRUE;
            rename_count++;
        }
        if(isRename && rename_count == 0) {
            wrap_type = TRUE;
            import_module_id(mid,
                             &LOCAL_SYMBOLS, &last_id,
                             &LOCAL_STRUCT_DECLS, &sttail,
                             &LOCAL_EXTERNAL_SYMBOLS, &last_ep,
                             NULL, wrap_type);
        }
    }

    // deep-copy types now!
    first_mid = ID_NEXT(prev_mid);
    FOREACH_ID(mid, first_mid) {
        // deep copy of types!
        deep_ref_copy_for_module_id_type(ID_TYPE(mid));

        // deep copy for function types!
        if((mep = PROC_EXT_ID(mid)) != NULL) {
          //ID id;
          expv v;
          list lp;

          deep_ref_copy_for_module_id_type(EXT_PROC_TYPE(mep));

          /*
           * copy the arguments of the function type
           *
           * NOTE:
           *  It may be required to deep-copy whole of the EXT_ID.
           */
          if (EXT_PROC_ARGS(mep) != NULL) {
              EXT_PROC_ARGS(mep) = copy_function_args(EXT_PROC_ARGS(mep));

              FOR_ITEMS_IN_LIST(lp, EXT_PROC_ARGS(mep)) {
                  v = EXPR_ARG1(LIST_ITEM(lp));
                  deep_copy_and_overwrite_for_module_id_type(&(EXPV_TYPE(v)));
              }
          }
        }
    }

    FOREACH_USE_ARG(arg, args) {
        if(!arg->used) {
            error("'%s' is not found in module '%s'", SYM_NAME(arg->local), SYM_NAME(name));
            ret = FALSE;
        }
    }

    finalize_replicated_type_list();

    if(debug_flag)
        fprintf(debug_fp, "########   END USE ASSOC #######\n");

    return ret;
}


/**
 * use association with rename arguments.
 * import public identifiers from module to LOCAL_SYMBOLS.
 */
int
use_assoc_rename(SYMBOL name, struct use_argument * args)
{
    int isRename = TRUE;
    return use_assoc_common(name, args, isRename);
}

/**
 * use association with only arguments.
 * import public identifiers from module to LOCAL_SYMBOLS.
 */
int
use_assoc_only(SYMBOL name, struct use_argument * args)
{
    int isRename = FALSE;
    return use_assoc_common(name, args, isRename);
}

/*
 * compiles use statement.
 */
static void
compile_USE_decl (expr x, expr x_args)
{
    expv args, v;
    struct list_node *lp;
    struct use_argument * use_args = NULL;

    if(x_args != NULL && EXPR_ARG1(x_args) == NULL)
        return;

    args = list0(LIST);

    FOR_ITEMS_IN_LIST(lp, x_args) {
        expr useExpr, localExpr, x = LIST_ITEM(lp);
        struct use_argument * use_arg = XMALLOC(struct use_argument *, sizeof(struct use_argument));
        *use_arg = (struct use_argument){0};

        useExpr = EXPR_ARG1(x);
        localExpr = EXPR_ARG2(x);

        assert(EXPV_CODE(localExpr) == IDENT);
        assert(EXPV_CODE(useExpr) == IDENT);

        args = list_put_last(args, list2(LIST, useExpr, localExpr));

        use_arg->local = EXPV_NAME(localExpr);
        use_arg->use = EXPV_NAME(useExpr);
        if(use_args != NULL) {
            use_arg->next = use_args;
        }
        use_args = use_arg;
    }

    v = expv_cons(F95_USE_STATEMENT, NULL, x, args);
    EXPV_LINE(v) = EXPR_LINE(x);
    output_statement(v);

    use_assoc_rename(EXPR_SYM(x), use_args);

    list_put_last(UNIT_CTL_USE_DECLS(CURRENT_UNIT_CTL), x);
}

/*
 * compiles use only statement.
 */
static void
compile_USE_ONLY_decl (expr x, expr x_args)
{
    expv args, v;
    struct list_node *lp;
    expr useExpr, localExpr, a;
    struct use_argument * use_args = NULL;

    if(x_args == NULL || EXPR_ARG1(x_args) == NULL)
        return;

    args = list0(LIST);

    FOR_ITEMS_IN_LIST(lp, x_args) {
        struct use_argument * use_arg = XMALLOC(struct use_argument *, sizeof(struct use_argument));
        *use_arg = (struct use_argument){0};

        a = LIST_ITEM(lp);

        if (EXPV_CODE(a) == LIST) {
            useExpr = EXPR_ARG1(a);
            localExpr = EXPR_ARG2(a);

            assert(EXPV_CODE(useExpr) == IDENT);
            assert(EXPV_CODE(localExpr) == IDENT);

            args = list_put_last(args, list2(LIST, useExpr, localExpr));

            use_arg->use = EXPV_NAME(useExpr);
            use_arg->local = EXPV_NAME(localExpr);
        } else {
            assert(EXPV_CODE(a) == IDENT);
            args = list_put_last(args, list2(LIST, NULL, a));
            use_arg->local = EXPV_NAME(a);
            use_arg->use = NULL;
        }
        if(use_args != NULL) {
            use_arg->next = use_args;
        }
        use_args = use_arg;
    }

    v = expv_cons(F95_USE_ONLY_STATEMENT, NULL, x, args);
    EXPV_LINE(v) = EXPR_LINE(x);
    output_statement(v);

    use_assoc_only(EXPR_SYM(x), use_args);

    list_put_last(UNIT_CTL_USE_DECLS(CURRENT_UNIT_CTL), x);
}

static char*
genBlankInterfaceName()
{
    static int seq = 0;
    char buf[256];
    sprintf(buf, "$blank_interface_name%d", seq++);
    return strdup(buf);
}


/*
 * complies INTERFACE statement
 */
static void
compile_INTERFACE_statement(expr x)
{
    EXT_ID ep = NULL, use_associated_ep = NULL;
    ID iid;
    expr identOrOp;
    SYMBOL s = NULL;
    int hasName;
    struct interface_info * info =
        XMALLOC(struct interface_info *, sizeof(struct interface_info));
    info->class = INTF_DECL;

    identOrOp = EXPR_ARG1(x);
    hasName = identOrOp ? TRUE : FALSE;

    if(hasName) {
        switch(EXPR_CODE(identOrOp)) {
        case IDENT:
            /* generic function/subroutine */
            s = EXPR_SYM(identOrOp);
            iid = find_ident_local(s);
            if(iid == NULL) {
                iid = declare_ident(s, CL_PROC);
                if(iid == NULL)
                    return;
            } else if(ID_STORAGE(iid) == STG_UNKNOWN) {
                ID_STORAGE(iid) = STG_EXT;
                ID_CLASS(iid) = CL_PROC;
            } else if(ID_IS_OFMODULE(iid)){
                if(TYPE_BASIC_TYPE(ID_TYPE((iid))) != TYPE_GENERIC)
                    error_at_node(x,
                                  "'%s' is already defined"
                                  " as a generic procedure in module '%s'",
                                  SYM_NAME(s), iid->use_assoc->module_name);
                else
                    use_associated_ep = PROC_EXT_ID(iid);
            }
            break;
        case F95_ASSIGNOP: {
            /* user define assingment operator */
            s = find_symbol(EXPR_CODE_SYMBOL(EXPR_CODE(identOrOp)));
            info->class = INTF_ASSINGMENT;
        } break;
        case F95_USER_DEFINED: {
#define END_LENGTH 2
#define MAXLEN_USEROP 31
            expr id = EXPR_ARG1(identOrOp);
            char * name;
            char operator_name[MAXLEN_USEROP];
            assert(EXPR_CODE(id) == IDENT);

            name = SYM_NAME(EXPR_SYM(id));

            if (strlen(name) - END_LENGTH > MAXLEN_USEROP) {
                error("a name of operator is too long");
                return;
            }

            sprintf(operator_name, ".%s.", name);

            s = find_symbol(operator_name);

            info->class = INTF_USEROP;
        } break;
        case F95_POWEOP:
        case F95_MULOP:
        case F95_DIVOP:
        case F95_PLUSOP:
        case F95_MINUSOP:
        case F95_EQOP:
        case F95_NEOP:
        case F95_LTOP:
        case F95_LEOP:
        case F95_GEOP:
        case F95_GTOP:
        case F95_NOTOP:
        case F95_ANDOP:
        case F95_OROP:
        case F95_EQVOP:
        case F95_NEQVOP:
        case F95_CONCATOP:
        {
            s = find_symbol(EXPR_CODE_SYMBOL(EXPR_CODE(identOrOp)));
            info->class = INTF_OPERATOR;
        } break;
        default:
            NOT_YET();
        break;
        }
    } else {
        s = find_symbol(genBlankInterfaceName());
    }

    ep = new_external_id(s);
    EXT_LINE(ep) = EXPR_LINE(x);
    EXT_TAG(ep) = STG_EXT;
    EXT_IS_BLANK_NAME(ep) = !hasName;
    EXT_IS_DEFINED(ep) = TRUE;
    EXT_IS_OFMODULE(ep) = FALSE;
    EXT_PROC_CLASS(ep) = EP_INTERFACE;

    EXT_PROC_INTERFACE_INFO(ep) = info;

    EXT_NEXT(ep) = NULL;
    if(use_associated_ep)
        EXT_PROC_INTR_DEF_EXT_IDS(ep) = EXT_PROC_INTR_DEF_EXT_IDS(use_associated_ep);

    push_unit_ctl(ININTR);
    CURRENT_INTERFACE = ep;
}

/*
 * complies END INTERFACE statement
 */
static void
end_interface()
{
    EXT_ID ep, localExtSyms, intr;
    ID fid, iid;
    int hasSub = FALSE, hasFunc = FALSE;

    if(unit_ctl_level == 0 ||
        PARENT_STATE != ININTR) {
        error("unexpected END INTERFACE statement");
        pop_unit_ctl();
        CURRENT_STATE = INDCL;
        return;
    }

    localExtSyms = LOCAL_EXTERNAL_SYMBOLS;
    intr = CURRENT_INTERFACE;


    /* add symbols in INTERFACE to INTERFACE symbol */
    if (EXT_PROC_INTR_DEF_EXT_IDS(intr) == NULL) {
        EXT_PROC_INTR_DEF_EXT_IDS(intr) = localExtSyms;
    } else {
        extid_put_last(
            EXT_PROC_INTR_DEF_EXT_IDS(intr), localExtSyms);
    }

    /* add INTERFACE symbol to parent */
    if (EXT_PROC_INTERFACES(PARENT_EXT_ID) == NULL) {
        EXT_PROC_INTERFACES(PARENT_EXT_ID) = intr;
    } else {
        extid_put_last(EXT_PROC_INTERFACES(PARENT_EXT_ID), intr);
    }

    if (endlineno_flag) {
        if (CURRENT_INTERFACE && EXT_LINE(CURRENT_INTERFACE))
            EXT_END_LINE_NO(CURRENT_INTERFACE) = current_line->ln_no;
    }

    pop_unit_ctl();
    CURRENT_STATE = INDCL;

    /* add function symbol to parent local symbols */
    FOREACH_EXT_ID(ep, localExtSyms) {
        if(IS_GENERIC_TYPE(EXT_PROC_TYPE(ep))) {
            fid = declare_ident(EXT_SYM(ep), CL_PROC);
            if(fid == NULL)
                return;
            ID_TYPE(fid) = EXT_PROC_TYPE(ep);
            ID_STORAGE(fid) = STG_EXT;
            PROC_CLASS(fid) = P_EXTERNAL;
            PROC_EXT_ID(fid) = ep;
            EXT_PROC_CLASS(ep) = EP_INTERFACE_DEF;
        } else if(IS_SUBR(EXT_PROC_TYPE(ep))) {
            hasSub = TRUE;
        } else if(EXT_PROC_IS_MODULE_PROCEDURE(ep) == FALSE) {
            hasFunc = TRUE;
            fid = declare_ident(EXT_SYM(ep), CL_PROC);
            if(fid == NULL)
                return;
            ID_TYPE(fid) = EXT_PROC_TYPE(ep);
            ID_STORAGE(fid) = STG_EXT;
            PROC_CLASS(fid) = P_EXTERNAL;
            PROC_EXT_ID(fid) = ep;
            EXT_PROC_CLASS(ep) = EP_INTERFACE_DEF;
        }
    }

    if(EXT_IS_BLANK_NAME(intr) == FALSE) {
        if(hasSub && hasFunc) {
            error("function does not belong in a generic subroutine interface");
            return;
        }

        /* add interface symbol to parent local symbols */
        iid = find_ident(EXT_SYM(intr));
        if(iid == NULL) {
            iid = declare_ident(EXT_SYM(intr), CL_PROC);
            if(iid == NULL)
                return;
        }

        /* type should be calculated from
         * declared functions, not always TYPE_GNUMERIC */
        ID_CLASS(iid) = CL_PROC;
        ID_TYPE(iid) = type_basic(hasSub ? TYPE_SUBR : TYPE_GENERIC);
        TYPE_ATTR_FLAGS(ID_TYPE(iid)) = TYPE_ATTR_FLAGS(iid);
        ID_STORAGE(iid) = STG_EXT;
        PROC_CLASS(iid) = P_EXTERNAL;
        PROC_EXT_ID(iid) = intr;
        EXT_PROC_CLASS(intr) = EP_INTERFACE;
        EXT_PROC_TYPE(intr) = ID_TYPE(iid);

        /* define interface external symbol in parent */
        define_internal_subprog(intr);
    }

    /* TODO: we should check errors such as "ambiguous interfaces" */
}


static void
switch_id_to_proc(ID id)
{
    if(ID_CLASS(id) == CL_PROC)
        return;
    memset(&id->info.proc_info, 0, sizeof(id->info.proc_info));
    ID_CLASS(id) = CL_PROC;
}

/*
 * while reading module, read module procedure.
 */
static void
accept_MODULEPROCEDURE_statement_in_module(expr x)
{
    list lp;
    expr ident;
    ID id;

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        ident = LIST_ITEM(lp);
        assert(EXPR_CODE(ident) == IDENT);

        /*
         * FIXME:
         *	It is not good idea to set ID_TYPE() to
         *	BASIC_TYPE_DESC(TYPE_GENERIC). Need to replace the
         *	ID_TYPE() anyway.
         */

        id = find_ident(EXPR_SYM(ident));
        if (id == NULL) {
            id = declare_ident(EXPR_SYM(ident), CL_PROC);
        } else {
            switch_id_to_proc(id);
        }
        ID_TYPE(id) = BASIC_TYPE_DESC(TYPE_GENERIC);
        ID_CLASS(id) = CL_PROC;
        PROC_CLASS(id) = P_DEFINEDPROC;
        declare_function(id);
    }
}

/*
 * compile MODULE PROCEDURE statement
 */
static void
compile_MODULEPROCEDURE_statement(expr x)
{
    list lp;
    expr ident;
    ID id;
    EXT_ID ep;
    const char *genProcName = NULL;

    if (PARENT_STATE != ININTR) {
        error("unexpected MODULE PROCEDURE statement");
        return;
    }

    if (checkInsideUse()) {
        accept_MODULEPROCEDURE_statement_in_module(x);
        return;
    }

    if (EXT_IS_BLANK_NAME(CURRENT_INTERFACE)) {
        error("MODULE PROCEDURE must be in a generic module interface");
        return;
    }

    genProcName = SYM_NAME(EXT_SYM(CURRENT_INTERFACE));

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        ident = LIST_ITEM(lp);
        assert(EXPR_CODE(ident) == IDENT);
        id = find_ident(EXPR_SYM(ident));
        if (id == NULL) {
            id = declare_ident(EXPR_SYM(ident), CL_PROC);
        } else {
            switch_id_to_proc(id);
        }

        ep = declare_external_proc_id(EXPR_SYM(ident), NULL, TRUE);
        if (ep == NULL) {
            fatal("can't allocate an EXT_ID for a module procedure.");
            /* not reached. */
            continue;
        }
        EXT_LINE(ep) = EXPR_LINE(x);
        EXT_PROC_CLASS(ep) = EP_MODULE_PROCEDURE;
        EXT_PROC_IS_MODULE_SPECIFIED(ep) = (EXPR_INT(EXPR_ARG2(x)) == 1);

        if (add_module_procedure(genProcName, SYM_NAME(EXPR_SYM(ident)),
                                 NULL, NULL, NULL) == NULL) {
            fatal("can't add a module procedure '%s' for '%s'.",
                  SYM_NAME(EXPR_SYM(ident)), genProcName);
            /* not reached. */
        }
    }

    if (debug_flag) {
        dump_all_module_procedures(stderr);
    }
}

/*
 * compiles the scene range expression of case label.
 *
 * expr : scene_range_expression*
 * expv : list((value | indexRange )*)
 */
static expv
compile_scene_range_expression_list(expr scene_range_expression_list)
{
    expr r = scene_range_expression_list;
    expv v, value, lower, upper, prev = NULL, next;

    struct list_node *lp;


    if (r == NULL) {
        /* not error, but case DEFAULT.*/
        return NULL;
    }

    if (EXPR_CODE(r) != LIST) {
        error("internal error, unexpected code.");
        abort();
    }

    FOR_ITEMS_IN_LIST(lp,r) {
        v = LIST_ITEM(lp);

        if(EXPR_ARG1(v) != NULL) {
            value = compile_expression(EXPR_ARG1(v));
            next = list3(F_SCENE_RANGE_EXPR, value, NULL, NULL);

        } else {
            lower = compile_expression(EXPR_ARG2(v));
            upper = compile_expression(EXPR_ARG3(v));

            next = list3(F_SCENE_RANGE_EXPR, NULL, lower, upper);
        }

        if(prev == NULL) {
            prev = list1(LIST, next);
        } else {
            prev = list_put_last(prev, next);
        }
    }

    return prev;
}

expv
compile_set_expr(expr x) {
    expv ret = NULL;

    if (EXPR_CODE(x) == F_SET_EXPR) {
        ret = compile_expression(EXPR_ARG2(x));
        if (ret != NULL) {
            char *keyword = SYM_NAME(EXPR_SYM(EXPR_ARG1(x)));
            if (keyword != NULL && *keyword != '\0') {
                EXPV_KWOPT_NAME(ret) = (const char *)strdup(keyword);
            }
        }
    } else {
        fatal("%s: not F_SET_EXPR.", __func__);
    }

    return ret;
}


expv
compile_member_ref(expr x)
{
    ID member_id;
    expr mX;
    expv struct_v, new_v;
    expv shape = list0(LIST);
    TYPE_DESC tp;
    TYPE_DESC stVTyp = NULL;

    if (EXPR_CODE(x) != F95_MEMBER_REF) {
        fatal("%s: not F95_MEMBER_REF", __func__);
        return NULL;
    }

    struct_v = compile_expression(EXPR_ARG1(x));
    if (struct_v == NULL) {
        return NULL;
    }

    if (EXPV_CODE(struct_v) != F95_MEMBER_REF
        && EXPV_CODE(struct_v) != F_VAR
        && EXPV_CODE(struct_v) != ARRAY_REF
	&& EXPV_CODE(struct_v) != XMP_COARRAY_REF) {
        error("invalid left operand of '\%%'", EXPV_CODE(struct_v));
        return NULL;
    }

    stVTyp = EXPV_TYPE(struct_v);

    if(IS_ARRAY_TYPE(stVTyp)) {
        shape = list0(LIST);
        generate_shape_expr(EXPV_TYPE(struct_v), shape);
        stVTyp = bottom_type(stVTyp);
    }

    mX = EXPR_ARG2(x);
    assert(EXPR_CODE(mX) == IDENT);

    member_id = find_struct_member(stVTyp, EXPR_SYM(mX));

    if (member_id == NULL) {
        error("'%s' is not a member", SYM_NAME(EXPR_SYM(mX)));
        return NULL;
    }

    // TODO:
    //	merge type override all cases (array/substr/plain scalar).
    if (TYPE_IS_POINTER(stVTyp) ||
        TYPE_IS_TARGET(stVTyp)) {
        /*
         * If type of struct_v has pointer/pointee flags on, members
         * should have those flags on too.
         */
        TYPE_DESC mVTyp = ID_TYPE(member_id);
        TYPE_DESC retTyp = NULL;
        if (IS_ARRAY_TYPE(mVTyp)) {
            generate_shape_expr(mVTyp, shape);
            mVTyp = bottom_type(mVTyp);
        }
        retTyp = wrap_type(mVTyp);

        TYPE_ATTR_FLAGS(retTyp) |= TYPE_IS_POINTER(mVTyp);
        TYPE_ATTR_FLAGS(retTyp) |= TYPE_IS_TARGET(mVTyp);
        TYPE_ATTR_FLAGS(retTyp) |= TYPE_IS_ALLOCATABLE(mVTyp);

        /*
         * To avoid overwrite, check original flags before copy.
         */
        if (!TYPE_IS_POINTER(retTyp)) {
            TYPE_ATTR_FLAGS(retTyp) |= TYPE_IS_POINTER(stVTyp);
        }
        if (!TYPE_IS_TARGET(retTyp)) {
            TYPE_ATTR_FLAGS(retTyp) |= TYPE_IS_TARGET(stVTyp);
        }

        tp = retTyp;
    } else {
        tp = ID_TYPE(member_id);

    }

    tp = compile_dimensions(tp, shape);
    fix_array_dimensions(tp);

    new_v = expv_cons(F95_MEMBER_REF, tp, struct_v, mX);
    EXPV_LINE(new_v) = EXPR_LINE(x);

    return new_v;
}


static void
compile_STOP_PAUSE_statement(expr x)
{
    expr x1;
    expv v1;

    x1 = EXPR_ARG1(x);
    if(x1 != NULL) {
        v1 = expv_reduce(compile_expression(x1), FALSE);
        if(v1 == NULL)
            return;
        if(EXPR_CODE(v1) != INT_CONSTANT &&
            EXPR_CODE(v1) != STRING_CONSTANT) {
            error("bad expression in %s statement",
                  EXPR_CODE(x) == F_STOP_STATEMENT ? "STOP":"PAUSE");
            return;
        }
    }
    output_statement(list1(EXPR_CODE(x), x1));
}


static void
compile_NULLIFY_statement (expr x)
{
    expv args, v;
    list lp;

    args = list0(LIST);
    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        expv ev = compile_lhs_expression(LIST_ITEM(lp));
        if (ev == NULL)
            continue;
        if (EXPV_CODE(ev) != F95_MEMBER_REF && EXPV_CODE(ev) != F_VAR) {
            error("argument is not a variable nor structure element");
            continue;
        }
        if (!TYPE_IS_POINTER(EXPV_TYPE(ev))) {
            error("argument is not a pointer type");
            continue;
        }
        args = list_put_last(args, ev);
    }
    v = expv_cons(F95_NULLIFY_STATEMENT, NULL, args, NULL);
    EXPV_LINE(v) = EXPR_LINE(x);
    output_statement(v);
}


static int
isVarSetTypeAttr(expv v, uint32_t typeAttrFlags)
{
    TYPE_DESC tp;

    switch(EXPV_CODE(v)) {
    case F_VAR:
    case F95_MEMBER_REF:
        tp = EXPV_TYPE(v);
        return tp && ((TYPE_ATTR_FLAGS(tp) & typeAttrFlags) > 0);
    case ARRAY_REF:
    case XMP_COARRAY_REF:
        return isVarSetTypeAttr(EXPR_ARG1(v), typeAttrFlags);
    default:
        break;
    }
    abort();
}

extern int is_in_alloc;

static void
compile_ALLOCATE_DEALLOCATE_statement (expr x)
{
    /* (F95_ALLOCATE_STATEMENT args) */
    expr r, kwd;
    expv args, v, vstat = NULL;
    list lp;
    enum expr_code code = EXPR_CODE(x);
    expv tmpAssignV = NULL;

    args = list0(LIST);
    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        r = LIST_ITEM(lp);

        if(EXPR_CODE(r) == F_SET_EXPR) {
            kwd = EXPR_ARG1(r);
            if(vstat || EXPR_CODE(kwd) != IDENT ||
                strcmp(SYM_NAME(EXPR_SYM(kwd)), "stat") != 0) {
                error("invalid keyword list");
                break;
            }
            vstat = compile_expression(EXPR_ARG2(r));
            if(vstat == NULL || (EXPR_CODE(vstat) != F_VAR &&
				 EXPR_CODE(vstat) != ARRAY_REF &&
				 EXPR_CODE(vstat) != F95_MEMBER_REF)){
                error("invalid status variable");
            } else if(IS_INT(EXPV_TYPE(vstat)) == FALSE) {
                error("status variable is not a integer type");
            }
            if (EXPR_CODE(vstat) != F_VAR) {
                expv orgStat = vstat;
                vstat = allocate_temp(type_INT);
                tmpAssignV = expv_assignment(orgStat, vstat);
            }
        } else {
            if(vstat) {
                error("syntax error after status variable");
                continue;
            }

	    is_in_alloc = TRUE;
            expv ev = compile_lhs_expression(r);
	    is_in_alloc = FALSE;

            if (ev == NULL)
                continue;

            switch(EXPV_CODE(ev)) {
            case F95_MEMBER_REF:
            case F_VAR:
            case ARRAY_REF:
	    case XMP_COARRAY_REF:
                if(isVarSetTypeAttr(ev,
                    TYPE_ATTR_POINTER | TYPE_ATTR_ALLOCATABLE) == FALSE) {
                    error("argument is not a pointer nor allocatable type");
                    continue;
                }
                args = list_put_last(args, ev);
                break;
            case F_SET_EXPR:
                break;
            default:
                error("argument is not a variable nor array nor structure element");
                break;
            }
        }
    }

    v = expv_cons(code, NULL, args, vstat);
    EXPV_LINE(v) = EXPR_LINE(x);
    output_statement(v);
    if (tmpAssignV != NULL) {
        output_statement(tmpAssignV);
    }
}


static void
compile_ASSIGN_LABEL_statement(expr x)
{
    /* (F_ASSIGN_LABEL_STATEMENT label id) */
    ID idLabel;
    expr x1;
    expv v1, v2, w;

    x1 = EXPR_ARG1(x);
    v1 = expr_label_value(x1);
    if (v1 == NULL) {
        error("illegal label");
        return;
    } 

    if(EXPV_CODE(v1) != INT_CONSTANT)
        fatal("label is not integer constant");

    idLabel = declare_label(EXPV_INT_VALUE(v1),LAB_EXEC,FALSE);

    if(idLabel == NULL)
        return;

    if(EXPR_CODE(EXPR_ARG2(x)) != IDENT)
        fatal("F_ASSIGN_LABEL_STATEMENT: not ident");

    v2 = compile_lhs_expression(EXPR_ARG2(x));

    if(IS_INT(EXPV_TYPE(v2)) == FALSE) {
        error("variable must be integer type in ASSIGN statement");
        return;
    }

    w = expv_assignment(v2, v1);
    output_statement(w);
}


static void
compile_CALL_statement(expr x)
{
    expr x1;
    ID id;
    expv v;

    /* (F_CALL_STATEMENT identifier args)*/
    x1 = EXPR_ARG1(x);
    if (EXPR_CODE(x1) != IDENT) {
        fatal("compile_exec_statement: bad id in call");
    }
    id = find_external_ident_head(EXPR_SYM(x1));
    if(id == NULL) {
        id = declare_ident(EXPR_SYM(x1), CL_UNKNOWN);
        if (ID_CLASS(id) == CL_UNKNOWN) {
            ID_CLASS(id) = CL_PROC;
        }
        if (PROC_CLASS(id) == P_UNKNOWN) {
            PROC_CLASS(id) = P_EXTERNAL;
            TYPE_SET_EXTERNAL(id);
        }
    }
    if (ID_IS_AMBIGUOUS(id)) {
        error("an ambiguous reference to symbol '%s'", ID_NAME(id));
        return;
    }
    if (PROC_CLASS(id) == P_EXTERNAL &&
        (ID_TYPE(id) == NULL || IS_SUBR(ID_TYPE(id)) == FALSE)) {
        TYPE_DESC tp = type_basic(TYPE_SUBR);
        TYPE_UNSET_IMPLICIT(tp);
        TYPE_SET_USED_EXPLICIT(tp);
        if(ID_TYPE(id)) {
            if (!TYPE_IS_IMPLICIT(ID_TYPE(id)) && !IS_GENERIC_TYPE(ID_TYPE(id))) {
                error("called '%s' which has a type like a subroutine", ID_NAME(id));
                return;
            }
            TYPE_ATTR_FLAGS(tp) = TYPE_ATTR_FLAGS(ID_TYPE(id));
        }
        ID_TYPE(id) = tp;

        if(PROC_EXT_ID(id))
            EXT_PROC_TYPE(PROC_EXT_ID(id)) = tp;
    }

#if 0
    /*
     * FIXME:
     *	Even if the ident is really a subroutine, we can't
     *	determine until it is "CALLED". Furthermore, even if
     *	the ident is declared as an external, we only can know
     *	that the ident is a function or a subroutine. So
     *	technically said, we can't check if a function is
     *	invoked as a subtroutine or not. To do that, we need a
     *	multiple pass parser.
     */

    if (ID_CLASS(id) == CL_PROC && !IS_SUBR(ID_TYPE(id))) {
        if (ID_STORAGE(id) != STG_ARG) {
            error_at_node(x, "function is invoked as subroutine");
        }
    }
#endif

    if (ID_IS_DUMMY_ARG(id)) {
        v = compile_highorder_function_call(id, EXPR_ARG2(x), TRUE);
    } else {
        v = compile_function_call(id, EXPR_ARG2(x));
    }
    EXPV_TYPE(v) = type_basic(TYPE_SUBR);
    output_statement(v);
}


static void
compile_RETURN_statement(expr x)
{
    /* (F_RETURN_STATMENT arg) */
    if(EXPR_ARG1(x) != NULL){
        error("alternative return is not supported");
        return;
    }
    if(CURRENT_PROC_CLASS != CL_PROC)
        warning("RETURN statement in main or block data");
    output_statement(list0(F_RETURN_STATEMENT));
}


static void
compile_GOTO_statement(expr x)
{
    /* (F_GOTO_STATEMENT label) */
    expr x1;
    expv stLabel;
    ID id;

    x1 = EXPR_ARG1(x);
    stLabel = expr_label_value(x1);
    if (stLabel == NULL) {
        error("illegal label");
        return;
    }
    id = declare_label(EXPV_INT_VALUE(stLabel), LAB_EXEC,FALSE);
    if (id == NULL) return;
    output_statement(list1(GOTO_STATEMENT,
                           expv_sym_term(IDENT,NULL,ID_SYM(id))));
}

static void
compile_COMPGOTO_statement(expr x)
{
    /* (F_COMPGOTO_STATEMENT (LIST ) expr) */
    expv stLabel;
    expr x1;
    expv v1;
    ID id;
    list lp;

    v1 = compile_expression(EXPR_ARG2(x));
    if(EXPR_ARG1(x) == NULL) return; /* error recovery */
    FOR_ITEMS_IN_LIST(lp,EXPR_ARG1(x)){
        x1 = LIST_ITEM(lp);
        if (EXPR_CODE(x1) != INT_CONSTANT) {
            error("illegal label in computed GOTO");
            v1 = NULL;
            return;
        }
    }
    if(v1 == NULL) return;
    if(!IS_INT(EXPV_TYPE(v1))){
        error("expression must be integer in computed GOTO");
        return;
    }
    FOR_ITEMS_IN_LIST(lp,EXPR_ARG1(x)){
        x1 = LIST_ITEM(lp);
        stLabel = expr_label_value(x1);
        if (stLabel == NULL) {
            error("illegal label in computed GOTO");
            return;
        }
        if((id = declare_label(EXPV_INT_VALUE(stLabel),LAB_EXEC,FALSE)) == NULL){
            return;
        }
    }
    output_statement(list2(F_COMPGOTO_STATEMENT, EXPR_ARG1(x), v1));
}


static void
compile_ASGOTO_statement(expr x)
{
    /* (F_ASGOTO_STATEMENT IDENT list) */
    expr x1;
    expv v1, v2, stLabel, cases, w;
    list lp;
    ID idLabel;

    if(EXPR_ARG2(x) == NULL){
        error("line number list must be specified in assigned GOTO");
        return;
    }
    
    if(EXPR_CODE(EXPR_ARG1(x)) != IDENT)
        fatal("F_ASGOTO_STATEMENT: not ident");
    v1 = compile_lhs_expression(EXPR_ARG1(x));
    if(v1 == NULL) return;
    if(!IS_INT(EXPV_TYPE(v1)))
        error("variable must be integer type in assigned GOTO");

    cases = EMPTY_LIST;
    EXPV_LINE(cases) = EXPR_LINE(x);

    FOR_ITEMS_IN_LIST(lp,EXPR_ARG2(x)){
        x1 = LIST_ITEM(lp);
        if (EXPR_CODE(x1) != INT_CONSTANT) {
            error("illegal label in assigned GOTO");
            cases = NULL;
            break;
        }
        stLabel = expr_label_value(x1);
        if (stLabel == NULL) {
            error("illegal label in assigned GOTO");
            cases = NULL;
            break;
        }
        idLabel = declare_label(EXPV_INT_VALUE(stLabel),LAB_EXEC,FALSE);
        if(idLabel == NULL){
            cases = NULL;
            break;
        }
        v2 = list3(F_CASELABEL_STATEMENT,
                list3(F_SCENE_RANGE_EXPR,
                    expv_int_term(INT_CONSTANT,
                        type_INT,EXPV_INT_VALUE(stLabel)),
                    NULL, NULL),
                list1(GOTO_STATEMENT,
                    expv_sym_term(IDENT, NULL, ID_SYM(idLabel))),
                NULL);
        EXPV_LINE(v2) = EXPR_LINE(x);
        list_put_last(cases, v2);
    }

    if(cases == NULL) return;

    w = list3(F_SELECTCASE_STATEMENT, v1, cases, NULL);
    EXPV_LINE(w) = EXPR_LINE(x);
    output_statement(w);
}


static void
compile_ARITHIF_statement(expr x)
{
    /* (F_ARITHIF_STATEMENT expr l1 l2 l3) */
    expv w, cond, vTmp, stIf, stElse;
    expv label[3];
    ID idLabel;
    int i;
    static enum expr_code compops[] =
        { LOG_LT_EXPR, LOG_EQ_EXPR, LOG_GT_EXPR };

    cond = compile_expression(EXPR_ARG1(x));
    if(cond == NULL) return;

    if (EXPR_CODE(EXPR_ARG2(x)) != INT_CONSTANT ||
        EXPR_CODE(EXPR_ARG3(x)) != INT_CONSTANT ||   
        EXPR_CODE(EXPR_ARG4(x)) != INT_CONSTANT) {
        error("illegal label in arithmetic IF");
        return;
    }

    label[0] = expr_label_value(EXPR_ARG2(x));
    label[1] = expr_label_value(EXPR_ARG3(x));
    label[2] = expr_label_value(EXPR_ARG4(x));

    if(!IS_INT(EXPV_TYPE(cond)) && !IS_REAL(EXPV_TYPE(cond))){
        error("expression must be integer or real in arithmetic IF");
        return;
    }

    /*
     * To avoid side effect by evaluating v1 more than once,
     * have to generate temporary variable.
     */
    vTmp = allocate_temp(EXPV_TYPE(cond));
    EXPV_LINE(vTmp) = EXPR_LINE(x);
    output_statement(expv_assignment(vTmp, cond));

    stIf = NULL;
    stElse = NULL;

    for(i = 0; i < 3; ++i) {
        idLabel = declare_label(
            EXPV_INT_VALUE(label[i]),LAB_EXEC,FALSE);
        if(idLabel == NULL) return;

        stIf = list5(IF_STATEMENT,
            expv_cons(
                compops[i], type_LOGICAL, vTmp, expv_constant_0),
            list1(GOTO_STATEMENT,
                expv_sym_term(IDENT, NULL, ID_SYM(idLabel))),
	    stElse, NULL, NULL);
        stElse = stIf;
    }

    w = stIf;
    EXPV_LINE(w) = EXPR_LINE(x);
    output_statement(w);
}


static int markAsPublic(ID id)
{
    TYPE_DESC tp = ID_TYPE(id);
    if (TYPE_IS_PRIVATE(id) || (tp != NULL && TYPE_IS_PRIVATE(tp))) {
        error("'%s' is already specified as private.", ID_NAME(id));
        return FALSE;
    }
    TYPE_SET_PUBLIC(id);
    TYPE_UNSET_PRIVATE(id);

    return TRUE;
}

static int markAsPrivate(ID id)
{
    TYPE_DESC tp = ID_TYPE(id);
    if (TYPE_IS_PUBLIC(id) || (tp != NULL && TYPE_IS_PUBLIC(tp))) {
        error("'%s' is already specified as public.", ID_NAME(id));
        return FALSE;
    }
    TYPE_UNSET_PUBLIC(id);
    TYPE_SET_PRIVATE(id);

    return TRUE;
}

static void
compile_PUBLIC_PRIVATE_statement(expr id_list, int (*markAs)(ID))
{
    list lp;
    expr ident;
    ID id;
    
    if (!INMODULE()) {
        error("not in module.");
        return;
    }

    if (id_list == NULL) {
        /*
         * for single private/public statement
         */

        if ((CTL_TYPE(ctl_top) == CTL_STRUCT)
                     && (markAs == markAsPrivate)) {
            TYPE_DESC struct_tp = CTL_STRUCT_TYPEDESC(ctl_top);
            TYPE_SET_INTERNAL_PRIVATE(struct_tp);
            return;
        } else if (markAs == markAsPublic) {
            current_module_state = M_PUBLIC;
        } else if (markAs == markAsPrivate)  {
            current_module_state = M_PRIVATE;
        }

        /* private/public is set to ids, later in end_declaration */

        return;
    }
    
    FOR_ITEMS_IN_LIST(lp, id_list) {
        ident = LIST_ITEM(lp);
        switch (EXPR_CODE(ident)) {
            case IDENT: {
                if ((id = find_ident_local(EXPR_SYM(ident))) == NULL) {
                    id = declare_ident(EXPR_SYM(ident), CL_UNKNOWN);
                    if (id == NULL) {
                        /* must not happen. */
                        continue;
                    }
                    ID_COULD_BE_IMPLICITLY_TYPED(id) = TRUE;
                }
                (void)markAs(id);
                break;
            }
            case F95_GENERIC_SPEC: {
                expr arg;
                arg = EXPR_ARG1(ident);
                SYMBOL sym = find_symbol(EXPR_CODE_SYMBOL(EXPR_CODE(arg)));
                if ((id = find_ident_local(sym)) == NULL) {
                    id = declare_ident(sym, CL_UNKNOWN);
                    if (id == NULL) {
                        /* must not happen. */
                        continue;
                    }
                }
                (void)markAs(id);
                break;
            }
            case F95_USER_DEFINED: {
                expr arg;
                arg = EXPR_ARG1(ident);
                if ((id = find_ident_local(EXPR_SYM(arg))) == NULL) {
                    id = declare_ident(EXPR_SYM(arg), CL_UNKNOWN);
                    if (id == NULL) {
                        /* must not happen. */
                        continue;
                    }
                }
                (void)markAs(id);
                break;
            }
            default: {
                fatal("illegal item(s) in public/private statement. %d", EXPR_CODE(ident));
                break;
            }
        }
    }
}


static void
compile_POINTER_SET_statement(expr x) {
    list lp;
    int nArgs = 0;
    expv vPointer = NULL;
    expv vPointee = NULL;
    TYPE_DESC vPtrTyp = NULL;
    TYPE_DESC vPteTyp = NULL;
    expv v = NULL;

    FOR_ITEMS_IN_LIST(lp, x) {
        nArgs++;
    }

    if (nArgs != 2) {
        fatal("%s: Invalid arguments number, expect 2 but %d.",
              __func__, nArgs);
        return;
    }

    vPointer = compile_lhs_expression(EXPR_ARG1(x));
    vPointee = compile_expression(EXPR_ARG2(x));
    if (vPointer == NULL || vPointee == NULL) {
        return;
    }

    if(EXPV_CODE(vPointee) == FUNCTION_CALL)
        goto accept;

    vPtrTyp = EXPV_TYPE(vPointer);
    if (vPtrTyp == NULL || TYPE_BASIC_TYPE(vPtrTyp) == TYPE_UNKNOWN) {
        fatal("%s: Undetermined type for a pointer.", __func__);
        return;
    }
    vPteTyp = EXPV_TYPE(vPointee);
    if (vPteTyp == NULL || TYPE_BASIC_TYPE(vPteTyp) == TYPE_UNKNOWN) {
        fatal("%s: Undetermined type for a pointee.", __func__);
        return;
    }

    if (!TYPE_IS_POINTER(vPtrTyp)) {
        error_at_node(x, "'%s' is not a pointer.",
                      SYM_NAME(EXPR_SYM(EXPR_ARG1(x))));
        return;
    }
    if (!TYPE_IS_TARGET(vPteTyp) && !TYPE_IS_POINTER(vPteTyp)) {
        if(EXPR_CODE(EXPR_ARG2(x)) == IDENT)
            error_at_node(x, "'%s' is not a pointee.",
                          SYM_NAME(EXPR_SYM(EXPR_ARG2(x))));
        else
            error_at_node(x, "right hand side expression is not a pointee.");
        return;
    }

    if (TYPE_N_DIM(IS_REFFERENCE(vPtrTyp)?TYPE_REF(vPtrTyp):vPtrTyp) !=
        TYPE_N_DIM(IS_REFFERENCE(vPteTyp)?TYPE_REF(vPteTyp):vPteTyp)) {
#if 0
    if (TYPE_N_DIM(vPtrTyp) != TYPE_N_DIM(vPteTyp)) {
#endif
        error_at_node(x, "Rank mismatch.");
        return;
    }

    if (get_basic_type(vPtrTyp) != get_basic_type(vPteTyp)) {
        error_at_node(x, "Type mismatch.");
        return;
    }

accept:

    EXPV_LINE(vPointer) = EXPR_LINE(x);
    EXPV_LINE(vPointee) = EXPR_LINE(x);
    v = list2(F95_POINTER_SET_STATEMENT,
              (expr)vPointer,
              (expr)vPointee);
    EXPV_LINE(v) = EXPR_LINE(x);
    if (TYPE_BASIC_TYPE(EXPV_TYPE(vPointee)) == TYPE_LHS) {
        EXPV_TYPE(vPointee) = EXPV_TYPE(vPointer);
    }

    output_statement(v);
}


static void
compile_TARGET_POINTER_ALLOCATABLE_statement(expr x)
{
    list lp;
    expr aloc, ident, dims;
    ID id;

    assert(EXPR_CODE(x) == F95_TARGET_STATEMENT ||
        EXPR_CODE(x) == F95_POINTER_STATEMENT ||
        EXPR_CODE(x) == F95_ALLOCATABLE_STATEMENT);

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        aloc = LIST_ITEM(lp);
        ident = EXPR_ARG1(aloc);
        dims = EXPR_ARG2(aloc);

        assert(EXPR_CODE(aloc) == F95_ARRAY_ALLOCATION);
        assert(EXPR_CODE(ident) == IDENT);
        assert(dims == NULL || EXPR_CODE(dims) == LIST);

        id = declare_ident(EXPR_SYM(ident), CL_VAR);
        if(id == NULL)
            return;
        if(ID_IS_OFMODULE(id)) {
            error("can't change attributes of USE-assoicated symbol '%s'", ID_NAME(id));
            return;
        } else if (ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            return;
        }

        ID_COULD_BE_IMPLICITLY_TYPED(id) = TRUE;

        switch(EXPR_CODE(x)) {
        case F95_TARGET_STATEMENT:
            TYPE_SET_TARGET(id);
            break;
        case F95_POINTER_STATEMENT:
            TYPE_SET_POINTER(id);
            break;
        case F95_ALLOCATABLE_STATEMENT:
            TYPE_SET_ALLOCATABLE(id);
            break;
        default:
            abort();
        }

        if(dims) {
            compile_type_decl(NULL, NULL, list1(LIST, aloc), NULL);
        }
    }
}


static void
compile_OPTIONAL_statement(expr x)
{
    list lp;
    expr ident;
    ID id;

    assert(EXPR_CODE(x) == F95_OPTIONAL_STATEMENT);

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        ident = LIST_ITEM(lp);

        assert(EXPR_CODE(ident) == IDENT);

        /*
         * Dummy args must be declared as local symbol.
         */
        id = find_ident_local(EXPR_SYM(ident));
        if (id == NULL) {
            error_at_node(x, "\"%s\" is not declared yet.",
                          SYM_NAME(EXPR_SYM(ident)));
            continue;
        }
        if (!ID_IS_DUMMY_ARG(id)) {
            error_at_node(x, "\"%s\" is not a dummy argument.",
                          SYM_NAME(ID_SYM(id)));
            continue;
        }

        if(ID_IS_OFMODULE(id)) {
            error("can't change attributes of USE-assoicated symbol '%s'",
                  ID_NAME(id));
            return;
        } else if (ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            return;
        }

        /*
         * Like any variable, any function/subroutine also could
         * have optional attribute.
         */
        if (ID_CLASS(id) == CL_UNKNOWN ||
            ID_CLASS(id) == CL_VAR ||
            ID_CLASS(id) == CL_PROC) {
            /*
             * NOTE:
             *	Don't fix the type/class, not even calling
             *	declare_ident(), here.
             */
            TYPE_SET_OPTIONAL(id);

#if 0
            /*
             * XXX FIXME:
             *	Not sure I could overwrite the storage class here, but
             *	it seems here could be the place.
             */
            ID_STORAGE(id) = STG_ARG;
#else
            /*
             * A fix:
             *  Since we checked this id IS dummy arg now, no need to
             *	overwrite it. Furthermore, the id could be STG_EXT if
             *	it is function/subroutine.
             */
#endif
        }
    }
}


static void
compile_INTENT_statement(expr x)
{
    list lp;
    expr spec, ident;
    ID id;

    assert(EXPR_CODE(x) == F95_INTENT_STATEMENT);

    spec = EXPR_ARG1(x);

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(x)) {
        ident = LIST_ITEM(lp);

        assert(EXPR_CODE(ident) == IDENT);

        /*
         * Dummy args must be declared as local symbol.
         */
        id = find_ident_local(EXPR_SYM(ident));
        if (id == NULL) {
            error_at_node(x, "\"%s\" is not declared yet.",
                          SYM_NAME(EXPR_SYM(ident)));
            continue;
        }
        if (!ID_IS_DUMMY_ARG(id)) {
            error_at_node(x, "\"%s\" is not a dummy argument.",
                          SYM_NAME(ID_SYM(id)));
            continue;
        }

        if(ID_IS_OFMODULE(id)) {
            error("can't change attributes of USE-assoicated symbol '%s'",
                  ID_NAME(id));
            return;
        } else if (ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            return;
        }

        switch(EXPR_CODE(spec)) {
        case F95_IN_EXTENT:
            TYPE_SET_INTENT_IN(id);
            break;
        case F95_OUT_EXTENT:
            TYPE_SET_INTENT_OUT(id);
            break;
        case F95_INOUT_EXTENT:
            TYPE_SET_INTENT_INOUT(id);
            break;
        default:
            abort();
        }
    }
}


static void
fix_array_dimensions_recursive(ID ip)
{
    ID memp;
    TYPE_DESC tp = ID_TYPE(ip);

    if (IS_ARRAY_TYPE(tp)) {
        fix_array_dimensions(tp);
        implicit_declaration(ip);
        if (ID_TYPE(ip) == NULL) {
            error("can't determine type of '%s'", ID_NAME(ip));
        }
    } else if (IS_STRUCT_TYPE(tp)) {
        FOREACH_MEMBER(memp, tp) {
            fix_array_dimensions_recursive(memp);
        }
    }
}

static void
fix_pointer_pointee_recursive(TYPE_DESC tp)
{
    if (tp == NULL) {
        return;
    }
    if (TYPE_IS_TARGET(tp) ||
        TYPE_IS_POINTER(tp) ||
        TYPE_IS_ALLOCATABLE(tp)) {

        TYPE_DESC refT = TYPE_REF(tp);

        if (refT != NULL) {
            if (TYPE_IS_TARGET(tp)) {
                TYPE_SET_TARGET(refT);
            }
            if (TYPE_IS_POINTER(tp)) {
                TYPE_SET_POINTER(refT);
            }
            if (TYPE_IS_ALLOCATABLE(tp)) {
                TYPE_SET_ALLOCATABLE(refT);
            }
            fix_pointer_pointee_recursive(refT);
        } else {
            if (IS_STRUCT_TYPE(tp)) {
                /*
                 * TYPE_STRUCT base. Don't mark this node as
                 * pointer/pointee.
                 */
                TYPE_UNSET_POINTER(tp);
                TYPE_UNSET_TARGET(tp);
                TYPE_UNSET_ALLOCATABLE(tp);
            }
        }
    }
}

expv
create_implicit_decl_expv(TYPE_DESC tp, char * first, char * second)
{
    expr impl_expv, first_symbol, second_symbol;

    first_symbol = list0(IDENT);
    EXPR_SYM(first_symbol) = XMALLOC(SYMBOL, sizeof(struct symbol));
    SYM_NAME(EXPR_SYM(first_symbol)) = first;

    second_symbol = list0(IDENT);
    EXPR_SYM(second_symbol) = XMALLOC(SYMBOL, sizeof(struct symbol));
    SYM_NAME(EXPR_SYM(second_symbol)) = second;

    impl_expv = list2(LIST,first_symbol,second_symbol);
    EXPV_TYPE(impl_expv) = tp;

    return impl_expv;
}

void
set_parent_implicit_decls()
{
    int i;
    expv v;
    list lp;

    for (i = 0; i < unit_ctl_level; i++) {
        FOR_ITEMS_IN_LIST(lp, UNIT_CTL_IMPLICIT_DECLS(unit_ctls[i])) {
            v = LIST_ITEM(lp);
            if(EXPR_CODE(v) == IDENT)
                set_implicit_type_uc(CURRENT_UNIT_CTL,
                                     EXPV_TYPE(v),
                                     *(SYM_NAME(EXPR_SYM(v))),
                                     * (SYM_NAME(EXPR_SYM(v))),
                                     TRUE);
            else
                set_implicit_type_uc(CURRENT_UNIT_CTL,
                                     EXPV_TYPE(v),
                                     *SYM_NAME(EXPR_SYM(EXPR_ARG1(v))),
                                     *SYM_NAME(EXPR_SYM(EXPR_ARG2(v))),
                                     TRUE);
        }
    }
}

/**
 * cleanup UNIT_CTL for each procedure
 * Notes: local_external_symbols is not null cleared.
 */
void
cleanup_unit_ctl(UNIT_CTL uc)
{
    UNIT_CTL_CURRENT_PROC_NAME(uc) = NULL;
    UNIT_CTL_CURRENT_PROC_CLASS(uc) = CL_UNKNOWN;
    UNIT_CTL_CURRENT_PROCEDURE(uc) = NULL;
    UNIT_CTL_CURRENT_STATEMENTS(uc) = NULL;
    UNIT_CTL_CURRENT_BLK_LEVEL(uc) = 1;
    UNIT_CTL_CURRENT_EXT_ID(uc) = NULL;
    UNIT_CTL_CURRENT_STATE(uc) = OUTSIDE;
    UNIT_CTL_LOCAL_SYMBOLS(uc) = NULL;
    UNIT_CTL_LOCAL_STRUCT_DECLS(uc) = NULL;
    UNIT_CTL_LOCAL_COMMON_SYMBOLS(uc) = NULL;
    UNIT_CTL_LOCAL_LABELS(uc) = NULL;
    UNIT_CTL_IMPLICIT_DECLS(uc) = list0(LIST);
    UNIT_CTL_EQUIV_DECLS(uc) = list0(LIST);
    UNIT_CTL_USE_DECLS(uc) = list0(LIST);

    /* UNIT_CTL_LOCAL_EXTERNAL_SYMBOLS(uc) is not cleared */
    //if (unit_ctl_level == 0) { /* for main */
      if (doImplicitUndef == TRUE) {
        UNIT_CTL_IMPLICIT_NONE(uc) = TRUE;
	set_implicit_type_uc(uc, NULL, 'a', 'z', TRUE);
        UNIT_CTL_IMPLICIT_TYPE_DECLARED(uc) = 0;
	list_put_last(UNIT_CTL_IMPLICIT_DECLS(uc), create_implicit_decl_expv(NULL, "a", "z"));
      } else {
	/* default implicit type */
        /* implicit none is not set */
        UNIT_CTL_IMPLICIT_NONE(uc) = FALSE;
        /* implicit type is not declared yet */
        UNIT_CTL_IMPLICIT_TYPE_DECLARED(uc) = 0;
	/* a - z : initialize all to real. */
	set_implicit_type_uc(uc, BASIC_TYPE_DESC(defaultSingleRealType), 'a', 'z', TRUE);
	list_put_last(UNIT_CTL_IMPLICIT_DECLS(uc),
		      create_implicit_decl_expv(BASIC_TYPE_DESC(defaultSingleRealType), "a", "z"));
	/* i - n : initialize to int. */
	set_implicit_type_uc(uc, BASIC_TYPE_DESC(TYPE_INT), 'i', 'n', TRUE);
	list_put_last(UNIT_CTL_IMPLICIT_DECLS(uc),
		      create_implicit_decl_expv(BASIC_TYPE_DESC(TYPE_INT), "i", "n"));
      }
      //    }
    set_implicit_storage_uc(uc, default_stg, 'a', 'z');        /* set class */
}

static UNIT_CTL
new_unit_ctl()
{
    UNIT_CTL uc;

    uc = XMALLOC(UNIT_CTL, sizeof(*uc));
    if (uc == NULL)
        fatal("memory allocation failed");
    cleanup_unit_ctl(uc);
    UNIT_CTL_LOCAL_EXTERNAL_SYMBOLS(uc) = NULL;
    UNIT_CTL_INITIALIZE_DECLS(uc) = EMPTY_LIST;
    return uc;
}

static void
initialize_unit_ctl()
{
    int i;

    for (i = 0; i < MAX_UNIT_CTL; ++i) {
        unit_ctls[i] = NULL;
    }
    unit_ctls[0] = new_unit_ctl();
    unit_ctl_level = 0;
    unit_ctl_contains_level = 0;
}

/**
 * save current context before contains/interface statement to UNIT_CTL,
 * then push program unit control stack.
 */
void
push_unit_ctl(enum prog_state state)
{
    ID top_proc;
    int max_unit_ctl_contains = MAX_UNIT_CTL_CONTAINS;

    if (unit_ctl_level < 0) {
        fatal("push_unit_ctl() bug");
        return;
    }
    top_proc = UNIT_CTL_CURRENT_PROCEDURE(unit_ctls[0]);
    if (top_proc != NULL && ID_CLASS(top_proc) != CL_MODULE) {
        /* if top procedure is not module, stack len restriction become -1 */
        max_unit_ctl_contains --;
    }
    if (state == INCONT && unit_ctl_contains_level + 1 >= max_unit_ctl_contains) {
        error("Too many CONTAINS nest");
        return;
    }
    if (unit_ctl_level + 1 >= MAX_UNIT_CTL) {
        error("Too many nest");
        return;
    }
    if (CURRENT_EXT_ID && EXT_PROC_CONT_EXT_LINE(CURRENT_EXT_ID) == NULL)
       EXT_PROC_CONT_EXT_LINE(CURRENT_EXT_ID) = current_line;

    CURRENT_STATE = state;
    unit_ctl_level ++;
    if(state == INCONT)
        unit_ctl_contains_level ++;

    assert(unit_ctls[unit_ctl_level] == NULL);
    unit_ctls[unit_ctl_level] = new_unit_ctl();
    if (check_inside_INTERFACE_body() == FALSE) {
        set_parent_implicit_decls();
    }
}


/**
 * define EXT_ID for contains function/subroutine
 */
static void
define_internal_subprog(EXT_ID child_ext_ids)
{
    ID ip;
    EXT_ID ep, ext_id;
    TYPE_DESC tp;

    FOREACH_EXT_ID(ep, child_ext_ids) {
        if(EXT_PROC_CLASS(ep) == EP_PROC || EXT_PROC_CLASS(ep) == EP_INTERFACE) {
	    EXT_PROC_IS_INTERNAL(ep) = TRUE;
            tp = EXT_PROC_TYPE(ep);
            ip = find_ident(EXT_SYM(ep));
            if (PROC_EXT_ID(ip) == ep)
                continue;
            if (PROC_CLASS(ip) == P_UNDEFINEDPROC) {
                continue;
            }
            if (ip != NULL && ID_TYPE(ip) != NULL)
                tp = ID_TYPE(ip);
            ext_id = declare_external_proc_id(EXT_SYM(ep), tp, TRUE);
            if (ip != NULL) {
                PROC_EXT_ID(ip) = ext_id;
                ID_STORAGE(ip) = STG_EXT;
                PROC_CLASS(ip) = P_DEFINEDPROC;
            }
        }
    }
}


/**
 * pop program unit control stack,
 * then restore the context.
 */
void
pop_unit_ctl()
{
    if (unit_ctl_level >= MAX_UNIT_CTL) {
        fatal("pop_unit_ctl() bug");
        return;
    }
    if (unit_ctl_level - 1 < 0) {
        error("Too many end procedure");
        return;
    }
    unit_ctls[unit_ctl_level] = NULL;
    unit_ctl_level --;
    if(CURRENT_STATE == INCONT)
        unit_ctl_contains_level --;
}

/**
 * for type declaration with data style initializer
 * compile 'data .../... /' after compiling type declarations
 **/
static
void
compile_data_style_decl(expr decl_list)
{
    expr x, value;
    list lp;
    if( decl_list == NULL )return;
    FOR_ITEMS_IN_LIST(lp, decl_list) {
        x = LIST_ITEM(lp);
        if( x == NULL )continue;
        value  = EXPR_ARG4(x);
        if( value != NULL && EXPR_CODE(value) == F_DATA_DECL ){
            /* compilataion is executed later in end_declaration */
            list_put_last(CURRENT_INITIALIZE_DECLS,
                list1(F_DATA_DECL, EXPR_ARG1(value)));
        }
    }
}

