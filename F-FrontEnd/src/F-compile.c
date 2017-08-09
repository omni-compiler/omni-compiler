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

ENV current_local_env;

/* flags and defaults */
int save_all = FALSE;
int sub_stars = FALSE;
enum storage_class default_stg = STG_SAVE;

/* procedure context */
enum procedure_state current_proc_state;

/* module context */
enum module_state current_module_state = M_DEFAULT;

SYMBOL current_module_name = NULL;
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


/* Translate image control statements to xmp subroutine call statements */
int XMP_coarray_flag = TRUE;


/* control stack */
static struct control _ctl_base = {0};
CTL ctl_base = &_ctl_base;
CTL ctl_top;
CTL ctl_top_saved = NULL;

expv CURRENT_STATEMENTS_saved = NULL;

/* current statement label */
ID this_label;

TYPE_DESC type_REAL, type_INT, type_SUBR, type_CHAR, type_LOGICAL;
TYPE_DESC type_DREAL, type_COMPLEX, type_DCOMPLEX;
TYPE_DESC type_VOID;
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
static void compile_DOCONCURRENT_statement(expr range_st_no, expr cond,
                            expr construct_name);
static void check_DO_end(ID label);
static void end_declaration(void);
static void end_interface(void);
static void begin_type_bound_procedure_decls(void);
static void end_type_bound_procedure_decls(void);
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
static int  markAsProtected(ID id);
static void compile_POINTER_SET_statement(expr x);
static void compile_USE_decl(expr x, expr x_args, int is_intrinsic);
static void compile_USE_ONLY_decl(expr x, expr x_args, int is_intrinsic);
static expv compile_scene_range_expression_list(
                            expr scene_range_expression_list);
static void fix_array_dimensions_recursive(ID ip);
static void check_array_length(ID ip);
static void fix_pointer_pointee_recursive(TYPE_DESC tp);
static void compile_data_style_decl(expr x);

static int check_image_control_statement_available();
static int check_inside_CRITICAL_construct();

static void compile_SYNCALL_statement(expr x);
static void compile_SYNCIMAGES_statement(expr x);
static void compile_SYNCMEMORY_statement(expr x);
static void compile_LOCK_statement(expr x);
static void compile_UNLOCK_statement(expr x);
static void compile_CRITICAL_statement(expr x);
static void compile_ENDCRITICAL_statement(expr x);

static void compile_IMPORT_statement(expr x); // IMPORT statement
static void compile_BLOCK_statement(expr x);
static void compile_ENDBLOCK_statement(expr x);
static void compile_FORALL_statement(int st_no, expr x);
static void compile_ENDFORALL_statement(expr x);

static int check_valid_construction_name(expr x, expr y);
static void move_vars_to_parent_from_type_guard(void);
static void check_select_types(expr x, TYPE_DESC tp);

static void   compile_end_forall_header(expv init);
static ID     unify_id_list(ID parents, ID childs, int overshadow);
static void   unify_submodule_symbol_table(void);
static EXT_ID unify_ext_id_list(EXT_ID parents, EXT_ID childs, int overshadow);

void init_for_OMP_pragma();
void check_for_OMP_pragma(expr x);

expv OMP_pragma_list(enum OMP_pragma pragma,expv arg1,expv arg2);
expv OMP_FOR_pragma_list(expv clause,expv statements);

void init_for_XMP_pragma();
int check_for_XMP_pragma(int st_no, expr x);

void init_for_ACC_pragma();
void check_for_ACC_pragma(expr x);

void set_parent_implicit_decls(void);

void
push_env(ENV env)
{
    ENV parent_local_env;
    parent_local_env = current_local_env;
    current_local_env = env;
    current_local_env->parent = parent_local_env;
}


void
clean_env(ENV env)
{
    env->symbols = NULL;
    env->struct_decls = NULL;
    env->common_symbols = NULL;
    env->labels = NULL;
    env->external_symbols = NULL;
    env->interfaces = NULL;
    env->blocks = NULL;
    env->use_decls = list0(LIST);
}

void
pop_env()
{
    ENV old = current_local_env;
    if (current_local_env->parent == NULL) {
        error("no more parent environments");
        return;
    }
    current_local_env = current_local_env->parent;
    clean_env(old);
}

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

    type_VOID = BASIC_TYPE_DESC(TYPE_VOID);

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
    ctl_top = ctl_base;
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
    if (CTL_NEXT(ctl_top) == NULL) {
        CTL_NEXT(ctl_top) = new_ctl();
        CTL_PREV(CTL_NEXT(ctl_top)) = ctl_top;
    } else {
        cleanup_ctl(CTL_NEXT(ctl_top));
        CTL_TYPE(CTL_NEXT(ctl_top)) = CTL_NONE;
    }
    ctl_top = CTL_NEXT(ctl_top);

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
    if(CTL_PREV(ctl_top) == NULL) fatal("control stack empty");
    ctl_top = CTL_PREV(ctl_top);
    CTL_NEXT(ctl_top) = NULL;
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

    check_for_ACC_pragma(x);
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

void
compile_statement1(int st_no, expr x)
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
        && EXPR_CODE(x) != F08_ENDSUBMODULE_STATEMENT
        && EXPR_CODE(x) != F08_ENDPROCEDURE_STATEMENT
        && EXPR_CODE(x) != F_END_STATEMENT
        /* differ CONTAIN from INTERFASE */
        && PARENT_STATE != ININTR
        && EXPR_CODE(x) != F95_MODULEPROCEDURE_STATEMENT
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
        && EXPR_CODE(x) != F08_PROCEDURE_STATEMENT
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
        declare_procedure(CL_MODULE, EXPR_ARG1(x), type_MODULE, NULL, NULL, NULL, NULL);
        begin_module(EXPR_ARG1 (x));
        break;

    case F95_ENDMODULE_STATEMENT: /* (F95_ENDMODULE_STATEMENT) */
    do_end_module:
        check_INDCL();
        // move into end_procedure()
        //if (endlineno_flag)
        //ID_END_LINE_NO(CURRENT_PROCEDURE) = current_line->ln_no;
        end_procedure();
        end_module(EXPR_HAS_ARG1(x)?EXPR_ARG1(x):NULL);
        break;


    case F08_SUBMODULE_STATEMENT: /* (F08_SUBMODULE_STATEMENT submodule_name ancester_name parent_name ) */
        begin_procedure();
        declare_procedure(CL_SUBMODULE, EXPR_ARG1(x), type_MODULE, NULL, NULL, NULL, NULL);
        begin_submodule(EXPR_ARG1(x), EXPR_ARG2(x), EXPR_ARG3(x));
        break;

    case F08_ENDSUBMODULE_STATEMENT: /* (F08_ENDSUBMODULE_STATEMENT submodule_name) */
    do_end_submodule:
        check_INDCL();
        unify_submodule_symbol_table();
        end_procedure();
        end_submodule(EXPR_HAS_ARG1(x)?EXPR_ARG1(x):NULL);
        break;


    /* (F_PROGRAM_STATEMENT name) need: option or lias */
    case F95_USE_STATEMENT:
        check_INDCL();
        compile_USE_decl(EXPR_ARG1(x), EXPR_ARG2(x), FALSE);
        break;
    case F03_USE_INTRINSIC_STATEMENT:
        check_INDCL();
        compile_USE_decl(EXPR_ARG1(x), EXPR_ARG2(x), TRUE);
        break;

    case F95_USE_ONLY_STATEMENT:
        check_INDCL();
        compile_USE_ONLY_decl(EXPR_ARG1(x), EXPR_ARG2(x), FALSE);
        break;
    case F03_USE_ONLY_INTRINSIC_STATEMENT:
        check_INDCL();
        compile_USE_ONLY_decl(EXPR_ARG1(x), EXPR_ARG2(x), TRUE);
        break;

    case F95_INTERFACE_STATEMENT:
        check_INDCL();
        compile_INTERFACE_statement(x);
        break;

    case F95_ENDINTERFACE_STATEMENT:
        end_interface();
        break;

    case F08_PROCEDURE_STATEMENT: /* fall through */
    case F95_MODULEPROCEDURE_STATEMENT:
        compile_MODULEPROCEDURE_statement(x);
        break;

    case F_PROGRAM_STATEMENT:   /* (F_PROGRAM_STATEMENT name) */
        begin_procedure();
        declare_procedure(CL_MAIN, EXPR_ARG1(x), NULL, NULL, NULL, NULL, NULL);
        break;
    case F_BLOCK_STATEMENT:     /* (F_BLOCK_STATEMENT name) */
        begin_procedure();
        declare_procedure(CL_BLOCK, EXPR_ARG1(x), NULL, NULL, NULL, NULL, NULL);
        break;
    case F_SUBROUTINE_STATEMENT:
        /* (F_SUBROUTINE_STATEMENT name dummy_arg_list) */
        begin_procedure();
        declare_procedure(CL_PROC,
                          EXPR_ARG1(x), subroutine_type(), EXPR_ARG2(x), 
                          EXPR_ARG3(x), NULL, EXPR_ARG4(x));
        break;
        /* entry statements */
    case F_FUNCTION_STATEMENT: {
        /* (F_FUNCTION_STATEMENT name dummy_arg_list type) */
        TYPE_DESC tp;
        begin_procedure();
        tp = compile_type(EXPR_ARG3(x), TRUE);
        declare_procedure(CL_PROC, EXPR_ARG1(x),
                          function_type(tp),
                          EXPR_ARG2(x), EXPR_ARG4(x), EXPR_ARG5(x),
                          EXPR_ARG6(x));
        break;
    }
    case F_ENTRY_STATEMENT:
        /* (F_ENTRY_STATEMENT name dummy_arg_list) */
        if(CURRENT_STATE == OUTSIDE ||
           CURRENT_PROC_CLASS == CL_MAIN ||
           CURRENT_PROC_CLASS == CL_BLOCK ||
           CURRENT_PROC_CLASS == CL_MODULE ||
           CURRENT_PROC_CLASS == CL_SUBMODULE){
            error("misplaced entry statement");
            break;
        }
        declare_procedure(CL_ENTRY,
                          EXPR_ARG1(x), NULL, EXPR_ARG2(x),
                          NULL, EXPR_ARG3(x), NULL);
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


    case F08_ENDPROCEDURE_STATEMENT: /* (F08_END_PROCEDURE_STATEMENT) */
    case F95_ENDFUNCTION_STATEMENT:  /* (F95_END_FUNCTION_STATEMENT) */
    case F95_ENDSUBROUTINE_STATEMENT:  /* (F95_END_SUBROUTINE_STATEMENT) */
    case F95_ENDBLOCKDATA_STATEMENT:
        check_INEXEC();
	// move into end_procedure()
	//if (endlineno_flag)
	//ID_END_LINE_NO(CURRENT_PROCEDURE) = current_line->ln_no;
        end_procedure();
        break;

    case F95_ENDPROGRAM_STATEMENT:  /* (F95_END_PROGRAM_STATEMENT) */
        check_INEXEC();
        if (!check_image_control_statement_available()) return;
	// move into end_procedure()
	//if (endlineno_flag)
	//if (CURRENT_EXT_ID && EXT_LINE(CURRENT_EXT_ID))
	//EXT_END_LINE_NO(CURRENT_EXT_ID) = current_line->ln_no;
        end_procedure();
        break;
    case F_END_STATEMENT:       /* (F_END_STATEMENT) */
        if (!check_image_control_statement_available()) return;
        if (CURRENT_PROC_CLASS == CL_SUBMODULE ||
            (CURRENT_PROC_CLASS == CL_UNKNOWN &&
             unit_ctl_level > 1 &&
             PARENT_PROC_CLASS == CL_SUBMODULE)) {
            goto do_end_submodule;

        } else if (CURRENT_PROC_CLASS == CL_MODULE ||
            (CURRENT_PROC_CLASS == CL_UNKNOWN &&
             unit_ctl_level > 1 &&
             PARENT_PROC_CLASS == CL_MODULE)) {
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
            check_for_ACC_pragma(x); /* close LOOP directives if any */
            check_for_XMP_pragma(st_no, x); /* close LOOP directives if any */
            end_procedure();
        }
        break;
    case F95_CONTAINS_STATEMENT:
        if (CTL_TYPE(ctl_top) == CTL_STRUCT) {
            /* For type bound procedure */
            begin_type_bound_procedure_decls();
        } else {
            check_INEXEC();
            push_unit_ctl(INCONT);
        }
        break;

        /*
         * declaration statement
         */
    case F_TYPE_DECL: /* (F_TYPE_DECL type (LIST data ....) (LIST attr ...)) */
        if (CURRENT_STATE != IN_TYPE_PARAM_DECL)
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
        check_NOT_INBLOCK();
        /* common_decl = (LIST common_name (LIST var dims) ...) */
        compile_COMMON_decl(EXPR_ARG1(x));
        break;

    case F_EQUIV_DECL: /* (F_EQUIVE_DECL (LIST lhs ...) ...) */
        check_INDCL();
        check_NOT_INBLOCK();

        if (UNIT_CTL_EQUIV_DECLS(CURRENT_UNIT_CTL) == NULL) {
            UNIT_CTL_EQUIV_DECLS(CURRENT_UNIT_CTL) = list0(LIST);
        }
        list_put_last(UNIT_CTL_EQUIV_DECLS(CURRENT_UNIT_CTL), EXPR_ARG1(x));
        break;

    case F_IMPLICIT_DECL:
        check_INDCL();
        check_NOT_INBLOCK();
        if (EXPR_ARG1(x)){
            FOR_ITEMS_IN_LIST(lp,EXPR_ARG1(x)){
                v = LIST_ITEM(lp);
                /* implicit none?  result in peek the data structture.  */
                if (EXPR_CODE(EXPR_ARG1(EXPR_ARG1(v))) == F_TYPE_NODE) {
                    compile_IMPLICIT_decl(EXPR_ARG1(v), EXPR_ARG2(v));
                } else if (EXPR_CODE(EXPR_ARG1(EXPR_ARG1(v))) == F03_PARAMETERIZED_TYPE) {
                    compile_IMPLICIT_decl(EXPR_ARG1(EXPR_ARG1(v)), EXPR_ARG2(v));
                } else if (EXPR_CODE(EXPR_ARG1(EXPR_ARG1(v))) == F03_CLASS) {
                    compile_IMPLICIT_decl(EXPR_ARG1(EXPR_ARG1(v)), EXPR_ARG2(v));
                } else {
                    v = EXPR_ARG1(v);
                    compile_IMPLICIT_decl(EXPR_ARG1(v), EXPR_ARG2(v));
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
        check_NOT_INBLOCK();
        compile_OPTIONAL_statement(x);
        break;

    case F95_INTENT_STATEMENT:
        check_INDCL();
        check_NOT_INBLOCK();
        compile_INTENT_statement(x);
        break;

    case F_NAMELIST_DECL:
        check_INDCL();
        check_NOT_INBLOCK();
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

    case F03_SELECTTYPE_STATEMENT:
          check_INEXEC();
          push_ctl(CTL_SELECT_TYPE);
          v = compile_expression(EXPR_ARG1(x));
          ID selector = find_ident(EXPR_SYM(EXPR_ARG1(x)));
          if(EXPR_HAS_ARG3(x)){
            ID associate_name = find_ident(EXPR_SYM(EXPR_ARG3(x)));
            if(associate_name == NULL){
                // Define the associate variable
                associate_name = declare_ident(EXPR_SYM(EXPR_ARG3(x)), CL_VAR);
                ID_IS_ASSOCIATIVE(associate_name) = TRUE;
                ID_TYPE(associate_name) = ID_TYPE(selector);
            }
            expv tmp = expv_sym_term(IDENT, ID_TYPE(associate_name),
                ID_SYM(associate_name));
            st = list4(F03_SELECTTYPE_STATEMENT, v, NULL, EXPR_ARG2(x), tmp);
          } else {
            st = list4(F03_SELECTTYPE_STATEMENT, v, NULL, EXPR_ARG2(x), NULL);
          }

          CTL_BLOCK(ctl_top) = st;
          break;
    case F_CASELABEL_STATEMENT:
        if(CTL_TYPE(ctl_top) == CTL_SELECT  ||
           CTL_TYPE(ctl_top) == CTL_CASE) {
            expr const_name = EXPR_ARG2(x);
            expr parent_const_name = NULL;

            if (CTL_TYPE(ctl_top) == CTL_CASE) {
                CTL_CASE_BLOCK(ctl_top) = CURRENT_STATEMENTS;
                CURRENT_STATEMENTS = NULL;

                if (endlineno_flag)
                    EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

                parent_const_name = CTL_CASE_CONST_NAME(ctl_top);

                pop_ctl();
            } else {
                parent_const_name = CTL_SELECT_CONST_NAME(ctl_top);
            }

            v = compile_scene_range_expression_list(EXPR_ARG1(x));
            push_ctl(CTL_CASE);

            (void)check_valid_construction_name(parent_const_name, const_name);

            /*
             *  (F_CASELABEL_STATEMENT
             *    (LIST (scene range expression) ...)
             *    (LIST (exec statement) ...)
             *    (IDENTIFIER))
             */
            st = list3(F_CASELABEL_STATEMENT, v, NULL, const_name);

            CTL_BLOCK(ctl_top) = st;

        } else error("'case label', out of place");
        break;
        case F03_TYPEIS_STATEMENT:
        case F03_CLASSIS_STATEMENT:
            if(CTL_TYPE(ctl_top) == CTL_SELECT_TYPE ||
               CTL_TYPE(ctl_top) == CTL_TYPE_GUARD)
            {
                ID id = NULL;
                TYPE_DESC tp = NULL;
                expr const_name = EXPR_ARG2(x);
                expr parent_const_name = NULL;
                expv type = NULL;
                expv selector = NULL;

                if (CTL_TYPE(ctl_top) == CTL_TYPE_GUARD) {
                    CTL_CASE_BLOCK(ctl_top) = CURRENT_STATEMENTS;
                    CURRENT_STATEMENTS = NULL;

                    if (endlineno_flag)
                         EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
                    pop_ctl();
                    move_vars_to_parent_from_type_guard();
                    pop_env();

                    parent_const_name = CTL_TYPE_GUARD_CONST_NAME(ctl_top);
                } else {
                    parent_const_name = CTL_SELECT_CONST_NAME(ctl_top);
                }

                (void)check_valid_construction_name(parent_const_name, const_name);

                push_ctl(CTL_TYPE_GUARD);
                push_env(CTL_TYPE_GUARD_LOCAL_ENV(ctl_top));

                if (EXPR_ARG1(x) != NULL) { // NULL for CLASS DEFAULT
                    tp = compile_type(EXPR_ARG1(x), /* allow_predecl=*/ FALSE);
                    type = expv_sym_term(IDENT, tp, EXPR_SYM(EXPR_ARG1(x)));
                }
                if (EXPR_CODE(x) == F03_CLASSIS_STATEMENT) {
                    if (tp != NULL && !IS_STRUCT_TYPE(tp)) {
                        error("'class is' accepts only derived-type");
                        break;
                    }
                }

                check_select_types(x, tp);

                selector = CTL_SELECT_TYPE_ASSICIATE(CTL_PREV(ctl_top))?:CTL_SELECT_TYPE_SELECTOR(CTL_PREV(ctl_top));
                id = declare_ident(EXPR_SYM(selector), CL_VAR);
                declare_id_type(id, tp);

                st = list3(EXPR_CODE(x), type, NULL, const_name);
                CTL_BLOCK(ctl_top) = st;
            } else {
                error("'class is/type is label', out of place");
            }
            break;
    case F_ENDSELECT_STATEMENT:
        if (CTL_TYPE(ctl_top) == CTL_SELECT ||
            CTL_TYPE(ctl_top) == CTL_SELECT_TYPE) {
            expr const_name = EXPR_HAS_ARG1(x)?EXPR_ARG1(x):NULL;
            expr parent_const_name = NULL;
            CTL_SELECT_STATEMENT_BODY(ctl_top) = CURRENT_STATEMENTS;

            parent_const_name = CTL_SELECT_CONST_NAME(ctl_top);
            (void)check_valid_construction_name(parent_const_name, const_name);

            if (endlineno_flag)
                EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

        } else if (CTL_TYPE(ctl_top) == CTL_CASE ||
                   CTL_TYPE(ctl_top) == CTL_TYPE_GUARD) {
            expr const_name = EXPR_HAS_ARG1(x)?EXPR_ARG1(x):NULL;
            expr parent_const_name = NULL;

            CTL_CASE_BLOCK(ctl_top) = CURRENT_STATEMENTS;

            if (CTL_TYPE(ctl_top) == CTL_CASE) {
                parent_const_name = CTL_CASE_CONST_NAME(ctl_top);
            } else {
                parent_const_name = CTL_TYPE_GUARD_CONST_NAME(ctl_top);
            }

            (void)check_valid_construction_name(parent_const_name, const_name);

            // For previous CASE.
            if (endlineno_flag)
                EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

            pop_ctl();

            if (CTL_TYPE(ctl_top) == CTL_SELECT_TYPE) {
                move_vars_to_parent_from_type_guard();
                pop_env();
            }

            if (CTL_TYPE(ctl_top) != CTL_SELECT &&
                CTL_TYPE(ctl_top) != CTL_SELECT_TYPE) {
                error("'end select', out of place");
            }

            CTL_SELECT_STATEMENT_BODY(ctl_top) = CURRENT_STATEMENTS;

            // For SELECT
            if (endlineno_flag)
                EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;

        } else {
            error("'end select', out of place");
        }

        pop_ctl();

        break;

    case F_PRAGMA_STATEMENT:
        compile_pragma_statement(x);
        break;

    case F95_TYPEDECL_STATEMENT:
        check_INDCL();
        /* (F95_TYPEDECL_STATEMENT (LIST <I> <NULL> <NULL> <NULL>) */
        CURRENT_STATE = IN_TYPE_PARAM_DECL;
        compile_struct_decl(EXPR_ARG1(x), EXPR_ARG2(x), EXPR_ARG3(x));
        break;

    case F95_ENDTYPEDECL_STATEMENT:
        /* if the current state is in:
         * - the type-parameter declaration part
         * - the type-bound procedure dclaration part
         * turn the state into the declaration part
         */
        if (CURRENT_STATE == IN_TYPE_PARAM_DECL ||
            CURRENT_STATE == IN_TYPE_BOUND_PROCS) {
            end_type_bound_procedure_decls();
            CURRENT_STATE = INDCL;
        } else {
            check_INDCL();
        }

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
        if (CURRENT_STATE == IN_TYPE_PARAM_DECL) {
            /* Expects inside the derived-type declaration */
            /* NOTE: PRIVATE and PROTECTED can be written in the derived-type declaration */
            CURRENT_STATE = INDCL;
        } else if (CURRENT_STATE != IN_TYPE_BOUND_PROCS) {
            /* PRIVATE statement in type-bound procedure is allowed*/
            check_INDCL();
        }
        compile_PUBLIC_PRIVATE_statement(EXPR_ARG1(x), markAsPrivate);
        break;

    case F2008_CRITICAL_STATEMENT:
        check_INEXEC();
        compile_CRITICAL_statement(x);
        break;

    case F2008_ENDCRITICAL_STATEMENT:
        check_INEXEC();
        compile_ENDCRITICAL_statement(x);
        break;

    case F03_PROTECTED_STATEMENT:
        check_INDCL();
        compile_PUBLIC_PRIVATE_statement(EXPR_ARG1(x), markAsProtected);
        break;

    case F03_IMPORT_STATEMENT: // IMPORT statement
        check_INDCL();
        compile_IMPORT_statement(x);
        break;

    case F2008_BLOCK_STATEMENT:
        check_INEXEC();
        compile_BLOCK_statement(x);
        break;

    case F2008_ENDBLOCK_STATEMENT:
        check_INEXEC();
        compile_ENDBLOCK_statement(x);
        break;

    case F03_VOLATILE_STATEMENT:
        check_INDCL();
        compile_VOLATILE_statement(EXPR_ARG1(x));
        break;

    case F03_ASYNCHRONOUS_STATEMENT:
        check_INDCL();
        compile_ASYNCHRONOUS_statement(EXPR_ARG1(x));
        break;

    case F03_TYPE_BOUND_PROCEDURE_STATEMENT:
        if (CURRENT_STATE != IN_TYPE_BOUND_PROCS) {
            error("TYPE-BOUDNED PROCEDURE out of the derived-type declaration");
        }
        compile_type_bound_procedure(x);
        break;

    case F03_TYPE_BOUND_GENERIC_STATEMENT:
        if (CURRENT_STATE != IN_TYPE_BOUND_PROCS) {
            error("TYPE-BOUND GENERIC out of the derived-type declaration");
        }
        compile_type_generic_procedure(x);
        break;

    case F03_TYPE_BOUND_FINAL_STATEMENT:
        if (CURRENT_STATE != IN_TYPE_BOUND_PROCS) {
            error("FINAL statement is out of the derived-type declaration");
            return;
        }
        if (!INMODULE()) {
            error("FINAL statement should be inside a MODULE specification part");
            return;
        }

        compile_FINAL_statement(x);
        break;

    case F03_PROCEDURE_DECL_STATEMENT:
        if (CURRENT_STATE == IN_TYPE_PARAM_DECL) {
            CURRENT_STATE = INDCL;
        }
        check_INDCL();
        compile_procedure_declaration(x);
        break;

    case F03_VALUE_STATEMENT:
        check_INDCL();
        compile_VALUE_statement(EXPR_ARG1(x));
        break;

    case F_FORALL_STATEMENT:
        check_INEXEC();
        compile_FORALL_statement(st_no, x);
        break;

    case F_ENDFORALL_STATEMENT:
        compile_ENDFORALL_statement(x);
        check_INEXEC();
        break;

    case F08_DOCONCURRENT_STATEMENT: {
        check_INEXEC();
        compile_DOCONCURRENT_statement(EXPR_ARG1(x), EXPR_ARG2(x), EXPR_ARG3(x));
    } break;

    case F08_CONTIGUOUS_STATEMENT:
        compile_CONTIGUOUS_statement(x);
        check_INDCL();
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
            declare_procedure(CL_MAIN, make_enode(IDENT, find_symbol(NAME_FOR_NONAME_PROGRAM)),
                              NULL, NULL, NULL, NULL, NULL);
        }

        x1 = EXPR_ARG1(x);
        switch (EXPR_CODE(x1)) {

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

                if (TYPE_IS_PROTECTED(EXPV_TYPE(v1)) && TYPE_IS_READONLY(EXPV_TYPE(v1))) {
                    error_at_node(x, "assignment to a PROTECTED variable");
                }

                if (TYPE_BASIC_TYPE(EXPV_TYPE(v1)) == TYPE_FUNCTION) {
                    /*
                     * If a left expression is a function result,
                     * the type of compile_lhs_expression(x) is a non-function type.
                     */
                    error_at_node(x, "a lhs expression is function or subroutine");
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
        if (!check_image_control_statement_available()) return;
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

    case F2008_SYNCALL_STATEMENT:
        compile_SYNCALL_statement(x);
        break;

    case F2008_SYNCIMAGES_STATEMENT:
        compile_SYNCIMAGES_statement(x);
        break;

    case F2008_SYNCMEMORY_STATEMENT:
        compile_SYNCMEMORY_statement(x);
        break;

    case F2008_LOCK_STATEMENT:
        compile_LOCK_statement(x);
        break;

    case F2008_UNLOCK_STATEMENT:
        compile_UNLOCK_statement(x);
        break;

    case F03_WAIT_STATEMENT:
        compile_WAIT_statement(x);
        break;
        
    case F03_FLUSH_STATEMENT:
        compile_FLUSH_statement(x);
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
        declare_procedure(CL_MAIN, NULL, NULL, NULL, NULL, NULL, NULL);
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
            declare_procedure(CL_MAIN, make_enode(IDENT, find_symbol(NAME_FOR_NONAME_PROGRAM)),
                              NULL, NULL, NULL, NULL, NULL);
    case INSIDE:
        CURRENT_STATE = INDCL;
    case INDCL:
        break;
    case IN_TYPE_PARAM_DECL:
        error("declaration in TYPE PARAMETER DECLARATION part");
        break;
    case IN_TYPE_BOUND_PROCS:
        error("declaration in TYPE BOUND PROCEDURE DECLARATION part");
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
            declare_procedure(CL_MAIN, make_enode(IDENT, find_symbol(NAME_FOR_NONAME_PROGRAM)),
                              NULL, NULL, NULL, NULL, NULL);
        else {
            if (PARENT_STATE != INCONT) {
                /* Don't make MAIN program in the CONTAINS block*/
                declare_procedure(CL_MAIN, NULL, NULL, NULL, NULL, NULL, NULL);
            }
        }
    }
    if(NOT_INDATA_YET) end_declaration();
}


int
inblock()
{
    CTL cp;
    FOR_CTLS_BACKWARD(cp) {
        switch (CTL_TYPE(cp)) {
            case CTL_BLOCK:
                return TRUE;
                break;
            case CTL_INTERFACE:
                /* INTERFACE has its own scoping unit which differs from BLOCK's one */
                return FALSE;
            default:
                continue;
                break;
        }
    }
    return FALSE;
}


void
check_NOT_INBLOCK()
{
    if (inblock()) {
        error("unexpected statement in the block construct");
    }
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
fix_type(ID id) 
{
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
    parent_id = find_ident_outer_scope(ID_SYM(id));

    if (parent_id == NULL || ID_DEFINED_BY(parent_id) != id) {
        return;
    }

    my_tp = ID_TYPE(id);
    parent_tp = ID_TYPE(parent_id);
    assert(my_tp != NULL);

    if (parent_tp == NULL) {
        ID_TYPE(parent_id) = my_tp;
        return;
    }

    if (ID_CLASS(parent_id) == CL_PROC &&
        PROC_CLASS(parent_id) == P_UNDEFINEDPROC) {
        if (FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(ID_TYPE(parent_id))) {
            if (!function_type_is_compatible(ID_TYPE(id), ID_TYPE(parent_id))) {
                error_at_id(id,
                            "Type mismatch from the procedure is called");
            }
        }

        if (TYPE_IS_USED_EXPLICIT(ID_TYPE(parent_id))) {
            TYPE_DESC return_type = FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(parent_id));
            *return_type = *FUNCTION_TYPE_RETURN_TYPE(my_tp);
        }
        ID_TYPE(parent_id) = my_tp;
        EXPV_TYPE(ID_ADDR(parent_id)) = my_tp;
        PROC_CLASS(parent_id) = P_DEFINEDPROC;

    } else if (ID_CLASS(parent_id) == CL_PROC &&
               TYPE_IS_EXTERNAL(ID_TYPE(parent_id))) {
        error("external function/subroutine %s in the contain block", ID_NAME(id));

    } else if (TYPE_IS_EXPLICIT(parent_tp)) {
        if (TYPE_IS_EXPLICIT(my_tp) && ID_CLASS(parent_id) != CL_PROC) {
            error("%s is declared both parent and contains", ID_NAME(id));
        } else {
            TYPE_DESC tp;
            if (!IS_PROCEDURE_TYPE(parent_tp)) {
                tp = FUNCTION_TYPE_RETURN_TYPE(my_tp);
                if (tp == parent_tp)
                    return;

                TYPE_BASIC_TYPE(tp)
                        = TYPE_BASIC_TYPE(parent_tp);
                TYPE_REF(tp)
                        = TYPE_REF(parent_tp);

                assert(TYPE_REF(tp) == NULL ||
                       TYPE_BASIC_TYPE(tp) == TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE((my_tp))));

            } else if (TYPE_IS_MODULE(my_tp)){
                /* module funciton/subroutine types should be compatible */
                if (!function_type_is_compatible(my_tp, parent_tp)) {
                    error("The type of the predeclared "
                          "module funcion/subroutine is not compatible");
                }

            } else {
                tp = FUNCTION_TYPE_RETURN_TYPE(my_tp);
                if (tp == parent_tp)
                    return;

                /* copy basic type and ref */
                TYPE_BASIC_TYPE(my_tp) = TYPE_BASIC_TYPE(parent_tp);

                TYPE_BASIC_TYPE(tp)
                        = TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(parent_tp));
                TYPE_REF(tp)
                        = TYPE_REF(FUNCTION_TYPE_RETURN_TYPE(parent_tp));

                assert(TYPE_REF(tp) == NULL ||
                       TYPE_BASIC_TYPE(tp) == TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE((my_tp))));
            }
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
        return (TYPE_IS_PUBLIC(id) || TYPE_IS_PRIVATE(id)) || TYPE_IS_PROTECTED(id);
    else
        return (TYPE_IS_PUBLIC(id) || TYPE_IS_PRIVATE(id) || TYPE_IS_PROTECTED(id) ||
                TYPE_IS_PUBLIC(tp) || TYPE_IS_PRIVATE(tp) || TYPE_IS_PROTECTED(tp));
}


/*
 * Resolve a forward declaration of procedure variable's reference.
 *
 * ( 'f' in the following example)
 *
 * ex)
 *
 *  PROCEDURE( f ), POINTER :: p => g ! id is this f
 *
 *  CONTAINS
 *    FUNCTION f(a)
 *    END FUNCTION f           ! target is this f
 *
 */
static void
update_procedure_variable(ID id, const ID target, int is_final)
{
    if (target == NULL) {
        return;
    }

    if (ID_CLASS(target) == CL_VAR) {
        /* target is also a procedure variable, skip */
        return;
    }


    if (ID_TYPE(target) == NULL || !IS_PROCEDURE_TYPE(ID_TYPE(target))) {
        if (is_final) {
            error_at_id(VAR_REF_PROC(id), "not procedure");
        } else {
            return;
        }
    }

    if (!FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(get_bottom_ref_type(ID_TYPE(target)))) {
        if (is_final) {
            error_at_id(VAR_REF_PROC(id),
                        "%s should have an explicit interface",
                        SYM_NAME(ID_SYM(id)));
        } else {
            return;
        }
    }

    if (IS_FUNCTION_TYPE(ID_TYPE(target))) {
        TYPE_DESC ret;
        TYPE_DESC dummy_ret;
        ret = FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(target));

        dummy_ret = FUNCTION_TYPE_RETURN_TYPE(TYPE_REF(ID_TYPE(id)));
        if (ret != dummy_ret && ret != TYPE_REF(dummy_ret)) {
            TYPE_BASIC_TYPE(dummy_ret) = TYPE_BASIC_TYPE(ret);
            TYPE_REF(dummy_ret) = ret;
        }
    } else { /* IS_SUBR(ID_TYPE(target) == TRUE */
        TYPE_BASIC_TYPE(ID_TYPE(VAR_REF_PROC(id))) = TYPE_SUBR;
        TYPE_BASIC_TYPE(ID_TYPE(id)) = TYPE_SUBR;
    }
    TYPE_REF(ID_TYPE(id)) = ID_TYPE(target);
    ID_DEFINED_BY(VAR_REF_PROC(id)) = target;
    PROC_CLASS(VAR_REF_PROC(id)) = P_DEFINEDPROC;
}


/*
 * Update procedure typed variables
 *
 * Assign type bound procedure to explicit interface OR module procedure
 *
 * ids -- procedure variables are in them
 * struct_decls -- the derived-type to check
 * targets -- candidates to which the procedure variables refer
 * is_final -- if TRUE, raise error if the target doesn't have an appropriate type
 */
static void
update_procedure_variables_forall(ID ids, TYPE_DESC struct_decls, BLOCK_ENV block,
                                  const ID targets, int is_final)
{
    ID id;
    ID target;
    TYPE_DESC stp;
    BLOCK_ENV bp;

    FOREACH_ID(id, ids) {
        if (ID_USEASSOC_INFO(id) &&
            current_module_name != ID_MODULE_NAME(id)) {
            continue;
        }

        if (IS_PROCEDURE_TYPE(ID_TYPE(id)) && TYPE_REF(ID_TYPE(id)) != NULL) {
            if (VAR_REF_PROC(id) == NULL)
                continue;

            target = find_ident_head(ID_SYM(VAR_REF_PROC(id)), targets);
            update_procedure_variable(id, target, is_final);
        }
    }

    FOREACH_STRUCTDECLS(stp, struct_decls) {
        if (TYPE_TAGNAME(stp) &&
            ID_USEASSOC_INFO(TYPE_TAGNAME(stp)) &&
            current_module_name != ID_MODULE_NAME(TYPE_TAGNAME(stp))) {
            continue;
        }

        update_procedure_variables_forall(TYPE_MEMBER_LIST(stp), NULL, NULL,
                                          targets, is_final);
    }


    FOREACH_BLOCKS(bp, block) {
        update_procedure_variables_forall(BLOCK_LOCAL_SYMBOLS(bp),
                                          BLOCK_LOCAL_STRUCT_DECLS(bp),
                                          BLOCK_CHILDREN(bp),
                                          targets, is_final);
    }
}

void
begin_type_bound_procedure_decls(void)
{
    CURRENT_STATE = IN_TYPE_BOUND_PROCS;
    TYPE_UNSET_INTERNAL_PRIVATE(CTL_STRUCT_TYPEDESC(ctl_top));
    enable_need_type_keyword = FALSE;
}

void
end_type_bound_procedure_decls(void)
{
    enable_need_type_keyword = TRUE;
}


/*
 * Update type_bound_procedures in derived-types of struct declarations.
 *
 * Assign type bound procedure to explicit interface OR module procedure
 */
void
update_type_bound_procedures_forall(TYPE_DESC struct_decls, ID targets)
{
    TYPE_DESC tp;
    ID mem;
    ID target;

    if (struct_decls == NULL || targets == NULL) {
        return;
    }

    FOREACH_STRUCTDECLS(tp, struct_decls) {
        /*
         * First, update type-bound procedure
         */
        FOREACH_TYPE_BOUND_PROCEDURE(mem, tp) {
            if (TYPE_REF(ID_TYPE(mem)) != NULL) {
                continue;
            }

            target = find_ident_head(ID_SYM(TBP_BINDING(mem)), targets);

            if (target != NULL) {
                if (!IS_PROCEDURE_TYPE(ID_TYPE(target))) {
                    continue;
                }

                if (!check_tbp_pass_arg(tp, ID_TYPE(mem), ID_TYPE(target))) {
                    error("%s should have have a PASS argument",
                          SYM_NAME(ID_SYM(target)));
                    return;
                }
                /*
                 * update function type
                 */
                if (debug_flag) {
                    fprintf(debug_fp, "bind %s to %s%%%s\n",
                            SYM_NAME(ID_SYM(target)),
                            SYM_NAME(ID_SYM(TYPE_TAGNAME(tp))),
                            SYM_NAME(ID_SYM(mem)));
                }
                TYPE_REF(ID_TYPE(mem)) = ID_TYPE(target);
                if (IS_FUNCTION_TYPE(ID_TYPE(target))) {
                    /*
                     * update the dummy return type
                     */
                    TYPE_DESC ret;
                    TYPE_DESC dummy_ret;
                    ret = FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(target));
                    dummy_ret = FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(mem));
                    TYPE_BASIC_TYPE(dummy_ret) = TYPE_BASIC_TYPE(ret);
                    TYPE_REF(dummy_ret) = ret;
                } else { /* IS_SUBR(ID_TYPE(target) == TRUE */
                    TYPE_BASIC_TYPE(ID_TYPE(mem)) = TYPE_SUBR;
                }
            }
        }
    }
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

    if (CURRENT_PROCEDURE != NULL && CTL_TYPE(ctl_top) != CTL_BLOCK) {

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

            if (IS_FUNCTION_TYPE(tp)) {
                ID_TYPE(myId) = NULL;
                declare_id_type(myId, tp);
                replace_or_assign_type(&FUNCTION_TYPE_RETURN_TYPE(tp), ID_TYPE(resId));
                declare_id_type(resId, FUNCTION_TYPE_RETURN_TYPE(tp));
            } else {
                // declare_id_type(myId, function_type(tp));

                replace_or_assign_type(&FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(myId)), tp);
                declare_id_type(resId, tp);
            }

            resId = declare_function_result_id(resS, ID_TYPE(resId));
            if (resId == NULL) {
                fatal("%s: can't declare function result ident '%s'.",
                      __func__, SYM_NAME(resS));
                /* not reached. */
                return;
            }
            resultV = expv_sym_term(F_VAR, tp, resS);

            /*
             * Set result variable info.
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

        /* for bind feature */
        if(TYPE_HAS_BIND(myId) || PROC_HAS_BIND(myId)) {
            TYPE_SET_BIND(ID_TYPE(myId));
            TYPE_SET_BIND(EXT_PROC_TYPE(myEId));
            if(PROC_BIND(myId)) {
                TYPE_BIND_NAME(ID_TYPE(myId)) = PROC_BIND(myId);
                TYPE_BIND_NAME(EXT_PROC_TYPE(myEId)) = PROC_BIND(myId);
            }
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
        checkTypeRef(ip);
        if (ip != myId) {
            union_parent_type(ip);
        }
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
     * Check if array is too long.
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        check_array_length(ip);
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
                tp = subroutine_type();
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
                if (current_module_state == M_PROTECTED) {
                    TYPE_SET_PROTECTED(ip);
                }
            }
        }

        if (TYPE_IS_UNCHANGABLE(tp)) {
            if ((TYPE_ATTR_FLAGS(tp) | TYPE_ATTR_FLAGS(ip)) != TYPE_ATTR_FLAGS(tp)) {
                error_at_id(ip, "The type of '%s' can not be changed",
                            SYM_NAME(ID_SYM(ip)));
            }
        }

        /* merge type attribute flags except SAVE attr*/
        TYPE_ATTR_FLAGS(tp) |= (TYPE_ATTR_FLAGS(ip) & ~TYPE_ATTR_SAVE);
        if (IS_FUNCTION_TYPE(tp) && TYPE_REF(tp) == NULL) {
            /*
             * The type attributes for the function (PURE, ELEMENETAL, etc) are
             * never set to local symbol, so there is no need to filter out them.
             */
            TYPE_ATTR_FLAGS(FUNCTION_TYPE_RETURN_TYPE(tp))
                    |= (TYPE_ATTR_FLAGS(ip) & ~(TYPE_ATTR_SAVE|TYPE_ATTR_BIND|TYPE_ATTR_PUBLIC|TYPE_ATTR_PRIVATE));
        }
        if (TYPE_IS_EXTERNAL(tp) && !IS_PROCEDURE_TYPE(tp)) {
            tp = function_type(tp);
            TYPE_UNSET_SAVE(tp);
            ID_TYPE(ip) = tp;
        }

        if (FUNCTION_TYPE_IS_VISIBLE_INTRINSIC(tp)) {
            if (!IS_PROCEDURE_TYPE(tp)) {
                tp = function_type(tp);
                TYPE_SET_INTRINSIC(tp);
                ID_TYPE(ip) = tp;
            }
            if (FUNCTION_TYPE_IS_VISIBLE_INTRINSIC(tp)) {
                ID_STORAGE(ip) = STG_EXT;
            }
        }

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
                        fprintf(debug_fp,
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
                                warning_at_id(ip, "%s has %s\n",
                                              ID_NAME(ip), e->flag_name);

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
     * Check type parameter values like '*' or ':'
     */
    FOREACH_ID (ip, LOCAL_SYMBOLS) {
        if (ID_TYPE(ip) && IS_STRUCT_TYPE(ID_TYPE(ip))) {
            TYPE_DESC struct_tp = ID_TYPE(ip);
            if (!TYPE_TYPE_PARAM_VALUES(struct_tp))
                continue;

            FOR_ITEMS_IN_LIST(lp, TYPE_TYPE_PARAM_VALUES(struct_tp)) {
                if (EXPV_CODE(LIST_ITEM(lp)) == F08_LEN_SPEC_COLON) {
                    if (!TYPE_IS_POINTER(struct_tp) && !TYPE_IS_ALLOCATABLE(struct_tp)) {
                        error_at_id(ip,
                                    "type parameter value ':' should be used "
                                    "with a POINTER or ALLOCATABLE object");
                    }
                } else if (EXPV_CODE(LIST_ITEM(lp)) == LEN_SPEC_ASTERISC) {
                    if (!ID_IS_DUMMY_ARG(ip)) {
                        error_at_id(ip,
                                    "type parameter value '*' should be used "
                                    "with a dummy argument");
                    }
                }
            }

            if (TYPE_IS_CLASS(struct_tp)) {
                /*
                 * CLASS() shoule be a POINTER object, an ALLOCATABLE object, or a dummy argument
                 */
                if (!TYPE_IS_POINTER(struct_tp) &&
                    !TYPE_IS_ALLOCATABLE(struct_tp) &&
                    !ID_IS_DUMMY_ARG(ip)) {
                    error_at_id(ip,
                                "CLASS should be used "
                                "to a POINTER object, an ALLOCATABLE object, or a dummy argument");
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
            if (TYPE_IS_OPTIONAL(tp) && !(ID_IS_DUMMY_ARG(ip))) {
                warning_at_id(ip, "OPTIONAL is applied only "
                              "to dummy argument");
            } else if ((TYPE_IS_INTENT_IN(tp) ||
                        TYPE_IS_INTENT_OUT(tp) ||
                        TYPE_IS_INTENT_INOUT(tp)) &&
                       !(ID_IS_DUMMY_ARG(ip))) {
                warning_at_id(ip, "INTENT is applied only "
                              "to dummy argument");
            } else if (ID_STORAGE(ip) != STG_TAGNAME && type_is_nopolymorphic_abstract(tp)) {
                error_at_id(ip, "No derived type should not have the ABSTRACT attribute");
            } else if (TYPE_IS_CONTIGUOUS(tp) &&
                       !(IS_ARRAY_TYPE(tp) && (TYPE_IS_POINTER(tp) || TYPE_IS_ARRAY_ASSUMED_SHAPE(tp)))) {
                error_at_id(ip, "Only an array pointer or an assumed-shape array can have the CONTIGUOUS attribute");
            } else if (IS_PROCEDURE_TYPE(tp) && TYPE_IS_PROCEDURE(tp)) {
                if (ID_STORAGE(ip) != STG_ARG) {
                    if (!TYPE_IS_POINTER(tp)) {
                        error_at_id(ip, "PROCEDURE variable should have the POINTER attribute");
                    }
                    if (TYPE_IS_OPTIONAL(tp)) {
                        error_at_id(ip, "PROCEDURE variable should not have the OPTINAL attribute");
                    }
                    if (TYPE_IS_INTENT_IN(tp) || TYPE_IS_INTENT_OUT(tp) || TYPE_IS_INTENT_INOUT(tp)) {
                        error_at_id(ip, "PROCEDURE variable should not have the INTENT attribute");
                    }
                }
            }
        }
    }

    if (myId != NULL) {
        /*
         * Update function type
         */
        implicit_declaration(myId);
        function_type_udpate(ID_TYPE(myId), LOCAL_SYMBOLS);
        union_parent_type(myId);

        if (unit_ctl_level > 0) {
            update_procedure_variables_forall(PARENT_LOCAL_SYMBOLS,
                                              PARENT_LOCAL_STRUCT_DECLS,
                                              UNIT_CTL_LOCAL_BLOCKS(PARENT_UNIT_CTL),
                                              myId,
                                              /*is_final = */ FALSE);

            FOREACH_EXT_ID(ep, LOCAL_EXTERNAL_SYMBOLS) {
                update_procedure_variables_forall(EXT_PROC_ID_LIST(ep),
                                                  EXT_PROC_STRUCT_DECLS(ep),
                                                  EXT_PROC_BLOCKS(ep),
                                                  myId,
                                                  /*is_final = */ FALSE);
            }
        }


        /*
         * Update type bound procedure
         */
        if (unit_ctl_level > 0 && is_in_module()) {
            update_type_bound_procedures_forall(PARENT_LOCAL_STRUCT_DECLS, myId);
        }

        if (TYPE_IS_MODULE(ID_TYPE(myId)) && unit_ctl_level > 0) {
            ID parent = find_ident_outer_scope(ID_SYM(myId));

            if (parent && ID_TYPE(parent)) {
                if (ID_TYPE(myId) != ID_TYPE(parent) &&
                    FUNCTION_TYPE_IS_DEFINED(ID_TYPE(parent)) &&
                    PARENT_STATE != ININTR) {
                    error_at_id(myId,
                                "A module function/subroutine '%s' is already defined",
                                SYM_NAME(ID_SYM(myId)));
                }

                if (!function_type_is_compatible(ID_TYPE(myId), ID_TYPE(parent))) {
                    error_at_id(myId,
                                "A module function/subroutine type is not compatible");
                }

            }
        }
    }

    /*
     * Update type bound procedure against exteranl functions
     */
    update_type_bound_procedures_forall(LOCAL_STRUCT_DECLS, LOCAL_SYMBOLS);

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
    if (tp == NULL || (IS_FUNCTION_TYPE(tp) && FUNCTION_TYPE_RETURN_TYPE(tp) == NULL)) {
        /*
         * Both the id and resId has no TYPE_DESC. Try implicit.
         */
        implicit_declaration(id);
        tp = ID_TYPE(id);
    }

    ID pid;
    if (tp && (pid = find_ident_outer_scope(ID_SYM(id)))){
        if (TYPE_IS_PUBLIC(pid)) TYPE_SET_PUBLIC(tp);
        else if (TYPE_IS_PRIVATE(pid)) TYPE_SET_PRIVATE(tp);
        else if (TYPE_IS_PROTECTED(pid)) TYPE_SET_PROTECTED(tp);
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
	  tq = NULL;
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
    EXT_PROC_BLOCKS(CURRENT_EXT_ID) = LOCAL_BLOCKS;
    EXT_PROC_INTERFACES(CURRENT_EXT_ID) = LOCAL_INTERFACES;

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
        if (EXT_SYM(proc)) {
            fprintf(debug_fp,"redefine '%s'\n", SYM_NAME(EXT_SYM(proc)));
        } else {
            fprintf(debug_fp,"redefine (anonymous)\n");
        }

        for(i = redefine_unit_ctl_level; i >= 0; i--) fprintf(debug_fp,"  ");
        fprintf(debug_fp,"contain procedure : {\n");
        FOREACH_EXT_ID(ep, unit_ctl_procs[redefine_unit_ctl_level]){
            for(i = redefine_unit_ctl_level; i >= 0; i--) fprintf(debug_fp,"  ");
            if (EXT_SYM(ep)) {
                fprintf(debug_fp,"  %s\n", SYM_NAME(EXT_SYM(ep)));
            } else {
                fprintf(debug_fp,"  (anonymous)\n");
            }
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

            EXT_ID ep;
            FOREACH_EXT_ID(ep, EXTERNAL_SYMBOLS){
                if (EXT_SYM(ep) == ID_SYM(id)){
                    external_proc = ep;
                    break;
                }
            }

            if (external_proc == NULL) {
                if (ID_TYPE(id) != NULL &&
                    ID_STORAGE(id) != STG_EXT &&
                    PROC_CLASS(id) == P_UNDEFINEDPROC &&
                    IS_PROCEDURE_TYPE(ID_TYPE(id)) &&
                    FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(ID_TYPE(id))) {
                    error_at_id(id,
                                "%s is used as an explicit interface but not defined",
                                SYM_NAME(ID_SYM(id)));
                    continue;
                } else {
                    external_proc = declare_external_proc_id(ID_SYM(id), ID_TYPE(id), TRUE);
                }
            }

            PROC_CLASS(id)  = P_EXTERNAL;
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

static void
check_labels_in_block(BLOCK_ENV block) {
    ID id;
    BLOCK_ENV bp;

    FOREACH_ID(id, BLOCK_LOCAL_LABELS(block)) {
        if (LAB_TYPE(id) != LAB_UNKNOWN &&
            LAB_IS_USED(id) && !LAB_IS_DEFINED(id)) {
            error("missing statement number %d", LAB_ST_NO(id));
        }
        checkTypeRef(id);
    }

    FOREACH_BLOCKS(bp, BLOCK_CHILDREN(block)) {
        check_labels_in_block(bp);
    }
}


static int
is_operator_proc(TYPE_DESC ftp)
{
    ID arg1;
    ID arg2;
    ID args;

    if (ftp == NULL) {
        return FALSE;
    }

    args = FUNCTION_TYPE_ARGS(ftp);

    if (IS_SUBR(ftp)) {
        return FALSE;
    }

    if (args == NULL || ID_NEXT(args) == NULL || ID_NEXT(ID_NEXT(args)) != NULL) {
        return FALSE;
    }

    arg1 = args;
    arg2 = ID_NEXT(args);

    if (ID_TYPE(arg1) == NULL || !(TYPE_IS_INTENT_IN(ID_TYPE(arg1)))) {
        return FALSE;
    }

    if (ID_TYPE(arg2) == NULL || !(TYPE_IS_INTENT_IN(ID_TYPE(arg2)))) {
        return FALSE;
    }

    return TRUE;
}


static int
is_assignment_proc(TYPE_DESC ftp)
{
    ID arg1;
    ID arg2;
    ID args;

    if (ftp == NULL) {
        return FALSE;
    }

    args = FUNCTION_TYPE_ARGS(ftp);

    if (IS_FUNCTION_TYPE(ftp)) {
        return FALSE;
    }

    if (args == NULL || ID_NEXT(args) == NULL || ID_NEXT(ID_NEXT(args)) != NULL) {
        return FALSE;
    }

    arg1 = args;
    arg2 = ID_NEXT(args);

    if (ID_TYPE(arg1) == NULL || !(TYPE_IS_INTENT_OUT(ID_TYPE(arg1)) ||
                                   TYPE_IS_INTENT_INOUT(ID_TYPE(arg1)))) {
        return FALSE;
    }

    if (ID_TYPE(arg2) == NULL || !(TYPE_IS_INTENT_IN(ID_TYPE(arg2)))) {
        return FALSE;
    }

    return TRUE;
}


static void
check_procedure_variables_for_idlist(ID id_list, TYPE_DESC const stp, int is_final)
{
    ID id;
    ID target;
    TYPE_DESC ftp;
    expv init_expr;

    FOREACH_ID(id, id_list) {
        if (ID_USEASSOC_INFO(id) &&
            current_module_name != ID_MODULE_NAME(id)) {
            continue;
        }

        target = NULL;
        if (IS_PROCEDURE_TYPE(ID_TYPE(id)) && TYPE_IS_PROCEDURE(ID_TYPE(id))) {
            if (VAR_INIT_VALUE(id) == NULL ||
                EXPV_NEED_TYPE_FIXUP(VAR_INIT_VALUE(id)) == FALSE) {
                continue;
            }

            if ((target = find_ident(ID_SYM(VAR_REF_PROC(id)))) == NULL) {
                if (is_final)
                    error_at_id(id,
                                "Interface %s is not found",
                                SYM_NAME(ID_SYM(VAR_REF_PROC(id))));
                continue;
            }
            ftp = get_bottom_ref_type(ID_TYPE(target));

            if (ID_TYPE(target) == NULL ||
                !FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(ftp)) {
                if (is_final)
                    error_at_id(id,
                                "Interface %s does not have explict interface",
                                SYM_NAME(ID_SYM(VAR_REF_PROC(id))));
                continue;
            }

            if (stp != NULL && !check_tbp_pass_arg(stp, ID_TYPE(id), ftp)) {
                fprintf(stderr, "Interface %s does not have a PASS argument",
                        SYM_NAME(ID_SYM(target)));
                if (is_final)
                    error_at_id(id,
                                "Interface %s does not have a PASS argument",
                                SYM_NAME(ID_SYM(target)));
                continue;
            }

            init_expr = VAR_INIT_VALUE(id);
            if (init_expr != NULL && EXPR_CODE(init_expr) == F_VAR) {
                target = find_ident(EXPR_SYM(init_expr));
                if (target == NULL) {
                    if (is_final)
                        error_at_id(id, "invalid initialization");
                    continue;
                }

                /* they are not the same function/subroutine */
                if (!procedure_is_assignable(ID_TYPE(id), ID_TYPE(target))) {
                    if (is_final)
                        error_at_id(id, "type mismatch in the initialization");
                    continue;
                }

                EXPV_TYPE(init_expr) = ID_TYPE(target);
                EXPV_NEED_TYPE_FIXUP(init_expr) = FALSE;

            }
        }
    }
}


static void
check_procedure_variables_in_block(BLOCK_ENV block, int is_final)
{
    BLOCK_ENV bp;

    FOREACH_BLOCKS(bp, block) {
        check_procedure_variables_for_idlist(BLOCK_LOCAL_SYMBOLS(bp), NULL, is_final);
        check_procedure_variables_in_block(BLOCK_CHILDREN(bp), is_final);
    }
}


static void
check_procedure_variables_forall(int is_final)
{
    /*
     * Check a function refered exists
     *
     * PROCEDURE ( *f* ) :: p => h
     *  check f exists
     *  check h is available
     *
     */
    TYPE_DESC stp;
    EXT_ID ep;
    BLOCK_ENV bp;

    check_procedure_variables_for_idlist(LOCAL_SYMBOLS, NULL, is_final);

    FOREACH_STRUCTDECLS(stp, LOCAL_STRUCT_DECLS) {
        if (TYPE_TAGNAME(stp) &&
            ID_USEASSOC_INFO(TYPE_TAGNAME(stp)) &&
            current_module_name != ID_MODULE_NAME(TYPE_TAGNAME(stp))) {
            continue;
        }

        check_procedure_variables_for_idlist(TYPE_MEMBER_LIST(stp), stp, is_final);
    }

    FOREACH_EXT_ID(ep, LOCAL_EXTERNAL_SYMBOLS) {
        check_procedure_variables_for_idlist(EXT_PROC_ID_LIST(ep),
                                             EXT_PROC_STRUCT_DECLS(ep), is_final);
    }


    FOREACH_BLOCKS(bp, LOCAL_BLOCKS) {
        check_procedure_variables_in_block(bp, is_final);
    }
}

static int
is_defined_io_formatted(const TYPE_DESC ftp, const TYPE_DESC stp, int is_read)
{
    /*
     * SUBROUTINE my_read_routine_formatted
     *         (dtv,
     *          unit,
     *          iotype,
     *          v_list,
     *          iostat,
     *          iomsg)
     *         ! the derived-type variable
     *         `dtv-type-spec` , INTENT(INOUT)  :: dtv
     *         INTEGER, INTENT(IN)              :: unit ! unit number
     *         ! the edit descriptor string
     *         CHARACTER (LEN=*), INTENT(IN)    :: iotype
     *         INTEGER, INTENT(IN)              :: v_list(:)
     *         INTEGER, INTENT(OUT)             :: iostat
     *         CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
     * END
     *
     * OR
     *
     * SUBROUTINE my_write_routine_formatted
     *         (dtv,
     *          unit,
     *          iotype,
     *          v_list,
     *          iostat,
     *          iomsg)
     *         ! the derived-type value/variable
     *         dtv-type-spec , INTENT(IN) :: dtv
     *         INTEGER, INTENT(IN) :: unit
     *         ! the edit descriptor string
     *         CHARACTER (LEN=*), INTENT(IN) :: iotype
     *         INTEGER, INTENT(IN) :: v_list(:)
     *         INTEGER, INTENT(OUT) :: iostat
     *         CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
     * END
     */
    ID dtv, unit, iotype, v_list, iostat, iomsg;
    TYPE_DESC tp;
    int dtv_attr_flags = 0;

    if (is_read == TRUE) {
        dtv_attr_flags = TYPE_ATTR_INTENT_INOUT;
    } else {
        dtv_attr_flags = TYPE_ATTR_INTENT_IN;
    }

    if (ftp == NULL) {
        return FALSE;
    }

    if (!IS_SUBR(ftp)) {
        return FALSE;
    }

    dtv    = FUNCTION_TYPE_ARGS(ftp);
    unit   = dtv?ID_NEXT(dtv):NULL;
    iotype = unit?ID_NEXT(unit):NULL;
    v_list = iotype?ID_NEXT(iotype):NULL;
    iostat = v_list?ID_NEXT(v_list):NULL;
    iomsg  = iostat?ID_NEXT(iostat):NULL;

    if (dtv == NULL || strcmp("dtv", SYM_NAME(ID_SYM(dtv))) != 0) {
        debug("expect 'dtv' as a 1st argument, but got %s", unit?SYM_NAME(ID_SYM(dtv)):"null");
        return FALSE;
    }
    tp = ID_TYPE(dtv);
    if (!IS_STRUCT_TYPE(tp) || (stp != NULL && TYPE_REF(tp) != stp) ||
        (TYPE_ATTR_FLAGS(tp) != dtv_attr_flags &&
         TYPE_ATTR_FLAGS(tp) != (TYPE_ATTR_CLASS | dtv_attr_flags))) {
        debug("unexpected type of 'dtv'");
        return FALSE;
    }

    if (unit == NULL || strcmp("unit", SYM_NAME(ID_SYM(unit))) != 0) {
        debug("expect 'unit' as a 2nd arg, but got %s", unit?SYM_NAME(ID_SYM(unit)):"null");
        return FALSE;
    }
    tp = ID_TYPE(unit);
    if (!IS_INT(tp) || TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_IN) {
        debug("unexpected type of 'unit'");
        return FALSE;
    }

    if (iotype == NULL || strcmp("iotype", SYM_NAME(ID_SYM(iotype))) != 0) {
        debug("expect 'iotype' as a 3rd arg, but got %s", unit?SYM_NAME(ID_SYM(iostat)):"null");
        return FALSE;
    }
    tp = ID_TYPE(iotype);
    if (!IS_CHAR(tp) ||
        !IS_CHAR_LEN_UNFIXED(tp) ||
        TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_IN) {
        debug("unexpected type of 'iotype'");
        return FALSE;
    }

    if (v_list == NULL || strcmp("v_list", SYM_NAME(ID_SYM(v_list))) != 0) {
        debug("expect 'v_list' as a 4th arg, but got %s", unit?SYM_NAME(ID_SYM(v_list)):"null");
        return FALSE;
    }
    tp = ID_TYPE(v_list);
    if (!IS_ARRAY_TYPE(tp) ||
        !TYPE_IS_ARRAY_ASSUMED_SHAPE(tp) ||
        !IS_INT(TYPE_REF(tp)) ||
        TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_IN) {
        debug("unexpected type of 'v_list'");
        return FALSE;
    }

    if (iostat == NULL || strcmp("iostat", SYM_NAME(ID_SYM(iostat))) != 0) {
        debug("expect 'iostat' as a 5th arg, but got %s",
              unit?SYM_NAME(ID_SYM(iostat)):"null");
        return FALSE;
    }
    tp = ID_TYPE(iostat);
    if (!IS_INT(tp) || TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_OUT) {
        debug("unexpected type of 'iostat'");
        return FALSE;
    }

    if (iomsg == NULL || strcmp("iomsg", SYM_NAME(ID_SYM(iomsg))) != 0) {
        debug("expect 'iomsg' as a 6th arg, but got %s",
              unit?SYM_NAME(ID_SYM(iostat)):"null");
        return FALSE;
    }
    tp = ID_TYPE(iomsg);
    if (!IS_CHAR(tp) ||
        !IS_CHAR_LEN_UNFIXED(tp) ||
        TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_INOUT) {
        debug("unexpected type of 'iomsg'");
        return FALSE;
    }

    if (ID_NEXT(iomsg) != NULL) {
        debug("Unexpected 7th arg");
        return FALSE;
    }

    return TRUE;
}

static int
is_defined_io_unformatted(const TYPE_DESC ftp, const TYPE_DESC stp, int is_read)
{
    /*
     * SUBROUTINE my_read_routine_unformatted
     *         (dtv,
     *          unit,
     *          iostat,
     *          iomsg)
     *         dtv-type-spec , INTENT(INOUT) :: dtv
     *         INTEGER, INTENT(IN) :: unit
     *         INTEGER, INTENT(OUT) :: iostat
     *         CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
     *         END
     *
     * OR
     *
     * SUBROUTINE my_write_routine_unformatted
     *         (dtv,
     *          unit,
     *          iostat,
     *          iomsg)
     *         dtv-type-spec , INTENT(IN) :: dtv
     *         INTEGER, INTENT(IN) :: unit
     *         INTEGER, INTENT(OUT) :: iostat
     *         CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
     * END
     */
    ID dtv, unit, iostat, iomsg;
    TYPE_DESC tp;
    int dtv_attr_flags = 0;

    if (is_read == TRUE) {
        dtv_attr_flags = TYPE_ATTR_INTENT_INOUT;
    } else {
        dtv_attr_flags = TYPE_ATTR_INTENT_IN;
    }

    if (ftp == NULL) {
        return FALSE;
    }

    if (!IS_SUBR(ftp)) {
        return FALSE;
    }

    dtv    = FUNCTION_TYPE_ARGS(ftp);
    unit   = dtv?ID_NEXT(dtv):NULL;
    iostat = unit?ID_NEXT(unit):NULL;
    iomsg  = iostat?ID_NEXT(iostat):NULL;

    if (dtv == NULL || strcmp("dtv", SYM_NAME(ID_SYM(dtv))) != 0) {
        debug("expect 'dtv' as a 1st arg, but got %s",
              unit?SYM_NAME(ID_SYM(dtv)):"null");
        return FALSE;
    }
    tp = ID_TYPE(dtv);
    if (!IS_STRUCT_TYPE(tp) ||
        (stp != NULL && TYPE_REF(tp) != stp) ||
        (TYPE_ATTR_FLAGS(tp) != dtv_attr_flags &&
         TYPE_ATTR_FLAGS(tp) != (TYPE_ATTR_CLASS | dtv_attr_flags))) {
        debug("unexpected type of 'dtv'");
        return FALSE;
    }

    if (unit == NULL || strcmp("unit", SYM_NAME(ID_SYM(unit))) != 0) {
        debug("expect 'unit' as a 2nd arg, but got %s",
              unit?SYM_NAME(ID_SYM(unit)):"null");
        return FALSE;
    }
    tp = ID_TYPE(unit);
    if (!IS_INT(tp) || TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_IN) {
        debug("unexpected type of 'unit'");
        return FALSE;
    }

    if (iostat == NULL || strcmp("iostat", SYM_NAME(ID_SYM(iostat))) != 0) {
        debug("expect 'iostat' as a 3rd arg, but got %s",
              unit?SYM_NAME(ID_SYM(iostat)):"null");
        return FALSE;
    }
    tp = ID_TYPE(iostat);
    if (!IS_INT(tp) || TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_OUT) {
        debug("unexpected type of 'iostat'");
        return FALSE;
    }

    if (iomsg == NULL || strcmp("iomsg", SYM_NAME(ID_SYM(iomsg))) != 0) {
        debug("expect 'iomsg' as a 4th arg, but got %s",
              unit?SYM_NAME(ID_SYM(iomsg)):"null");
        return FALSE;
    }
    tp = ID_TYPE(iomsg);
    if (!IS_CHAR(tp) ||
        !IS_CHAR_LEN_UNFIXED(tp) ||
        TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_INTENT_INOUT) {
        debug("Unexpected type of 'iomsg'");
        return FALSE;
    }

    if (ID_NEXT(iomsg) != NULL) {
        debug("unexpected 5th arg");
        return FALSE;
    }

    return TRUE;
}

static int
is_defined_io_read_formatted(const TYPE_DESC ftp, const TYPE_DESC stp)
{
    return is_defined_io_formatted(ftp, stp, /*is_read=*/TRUE);
}

static int
is_defined_io_write_formatted(const TYPE_DESC ftp, const TYPE_DESC stp)
{
    return is_defined_io_formatted(ftp, stp, /*is_read=*/FALSE);
}

static int
is_defined_io_read_unformatted(const TYPE_DESC ftp, const TYPE_DESC stp)
{
    return is_defined_io_unformatted(ftp, stp, /*is_read=*/TRUE);
}

static int
is_defined_io_write_unformatted(const TYPE_DESC ftp, const TYPE_DESC stp)
{
    return is_defined_io_unformatted(ftp, stp, /*is_read=*/FALSE);
}

static int
is_defined_io_procedure(const ID id, const TYPE_DESC stp)
{
    TYPE_DESC ftp;

    if (id == NULL || ID_TYPE(id) == NULL || stp == NULL) {
        return FALSE;
    }

    ftp = ID_TYPE(id);
    while (TYPE_REF(ftp) != NULL) {
        ftp = TYPE_REF(ftp);
    }

    if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_WRITE &&
        TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_FORMATTED) {
        return is_defined_io_write_formatted(ftp, stp);
    } else if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_WRITE &&
               TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_UNFORMATTED) {
        return is_defined_io_write_unformatted(ftp, stp);
    } else if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_READ &&
               TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_FORMATTED) {
        return is_defined_io_read_formatted(ftp, stp);
    } else if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_READ &&
               TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_UNFORMATTED) {
        return is_defined_io_read_unformatted(ftp, stp);
    } else {
        return FALSE;
    }
}


static void
check_type_bound_procedures()
{
    ID mem;
    ID tbp;
    ID binding;
    ID bindto;
    TYPE_DESC tp;
    TYPE_DESC parent;
    TYPE_DESC ftp;

    FOREACH_STRUCTDECLS(tp, LOCAL_STRUCT_DECLS) {

        if (TYPE_TAGNAME(tp) &&
            ID_USEASSOC_INFO(TYPE_TAGNAME(tp)) &&
            current_module_name != ID_MODULE_NAME(TYPE_TAGNAME(tp))) {
            /*
             * This derived-type is defined in the other module,
             * skip check.
             */
            continue;
        }

        parent = TYPE_PARENT(tp)? TYPE_PARENT_TYPE(tp) : NULL;

        /*
         * Marks each type-bound procedure if it is specified by type-bound generics
         */
        FOREACH_TYPE_BOUND_GENERIC(mem, tp) {
            FOREACH_ID(binding, TBP_BINDING(mem)) {
                bindto = find_struct_member(tp, ID_SYM(binding));
                TBP_BINDING_ATTRS(bindto) |= TBP_BINDING_ATTRS(mem) & (
                        TYPE_BOUND_PROCEDURE_IS_OPERATOR |
                        TYPE_BOUND_PROCEDURE_IS_ASSIGNMENT |
                        TYPE_BOUND_PROCEDURE_WRITE |
                        TYPE_BOUND_PROCEDURE_READ |
                        TYPE_BOUND_PROCEDURE_FORMATTED |
                        TYPE_BOUND_PROCEDURE_UNFORMATTED);
            }
        }


        FOREACH_TYPE_BOUND_PROCEDURE(tbp, tp) {
            /*
             * Check a type-bound procedure is bound
             */
            if (TYPE_REF(ID_TYPE(tbp)) == NULL) {
                bindto = TBP_BINDING(tbp)?:tbp;
                error_at_id(tbp,
                            "\"%s\" must be a module procedure or "
                            "an external procedure with an explicit interface",
                            SYM_NAME(ID_SYM(bindto)));
            }

            /*
             * Check a type-bound procedure works as defined io procedure
             */
            if (TBP_IS_DEFINED_IO(tbp)) {
                if (!is_defined_io_procedure(tbp, tp)) {
                    error("type-bound procedure is used as defined i/o procedure, "
                          "but its procedure signature is wrong");
                }
            }

            if ((ftp = TYPE_REF(ID_TYPE(tbp))) != NULL) {
                /* already bounded, so check type */
                if (TBP_IS_OPERATOR(tbp)) {
                    if (!is_operator_proc(ftp)) {
                        error_at_id(tbp, "should be a function");
                        return;
                    }
                }
                if (TBP_IS_ASSIGNMENT(tbp)) {
                    if (!is_assignment_proc(ftp)) {
                        error_at_id(tbp, "not assiginment");
                        return;
                    }
                }
            }

            /*
             * If the parent type exists, check override.
             */
            if (parent) {
                ID parent_tbp = find_struct_member_allow_private(tp, ID_SYM(tbp), TRUE);
                if (ID_CLASS(tbp) != CL_TYPE_BOUND_PROC) {
                    /* never reached */
                    error_at_id(tbp, "should not override member");
                }

                if (!type_bound_procedure_types_are_compatible(
                        ID_TYPE(tbp), ID_TYPE(parent_tbp))) {
                    error_at_id(tbp,
                                "type mismatch to override %s",
                                SYM_NAME(ID_SYM(tbp)));
                }
            }
        }

    }
}

static int
check_final_subroutine_is_valid(ID id, TYPE_DESC stp)
{
    TYPE_DESC tp;
    TYPE_DESC ftp;
    ID arg;

    if (id == NULL || ID_TYPE(id) == NULL) {
        fatal("unexpected final subroutine");
        return FALSE;
    }

    ftp = ID_TYPE(id);

    if (!IS_SUBR(ftp)) {
        error("FINAL subroutine should be a subroutine");
        return FALSE;
    }

    arg = FUNCTION_TYPE_ARGS(ftp);

    if (arg == NULL) {
        error("FINAL subroutine should have one argument");
        return FALSE;
    }

    if (ID_NEXT(arg) != NULL) {
        error("FINAL subroutine has too many argument");
        return FALSE;
    }

    tp = ID_TYPE(arg);
    if (tp == NULL || get_bottom_ref_type(tp) != stp) {
        error("FINAL subroutine's argument should "
              "be the derived type");
        return FALSE;
    }

    if (TYPE_IS_POINTER(tp) ||
        TYPE_IS_ALLOCATABLE(tp) ||
        TYPE_IS_CLASS(tp) ||
        TYPE_IS_VALUE(tp) ||
        TYPE_IS_INTENT_OUT(tp)) {
        error("FINAL subroutine's argument should "
              "not be POINTER/ALLOCATABLE/CLASS/VALUE/INTENT(OUT)");
        return FALSE;
    }

    if (TYPE_HAS_TYPE_PARAMS(tp)) {
        ID ip;

        list lp = EXPR_LIST(TYPE_TYPE_PARAM_VALUES(tp));
        FOREACH_ID(ip, TYPE_TYPE_PARAMS(stp)) {
            if (ID_TYPE(ip) != NULL && TYPE_IS_LEN(ID_TYPE(ip))) {
                if (EXPV_CODE(LIST_ITEM(lp)) != LEN_SPEC_ASTERISC) {
                    error("FINAL subroutine's argument should "
                          "have an assumed length for the length parameter");
                    return FALSE;
                }
            }
            lp = LIST_NEXT(lp);
        }
    }

    return TRUE;
}


static void
check_final_subroutines()
{
    ID binding;
    TYPE_DESC tp;

    SYMBOL sym = find_symbol(FINALIZER_PROCEDURE);

    FOREACH_STRUCTDECLS(tp, LOCAL_STRUCT_DECLS) {
        ID final = NULL;

        if (TYPE_TAGNAME(tp) &&
            ID_USEASSOC_INFO(TYPE_TAGNAME(tp)) &&
            current_module_name != ID_MODULE_NAME(TYPE_TAGNAME(tp))) {
            /*
             * This derived-type is defined in the other module,
             * skip check.
             */
            continue;
        }

        if ((final = find_struct_member(tp, sym)) != NULL) {
            ID fin, fin1, fin2;

            FOREACH_ID(binding, TBP_BINDING(final)) {
                if ((fin = find_ident(ID_SYM(binding))) == NULL) {
                    error("FINAL subroutine %s does not exist",
                          SYM_NAME(ID_SYM(binding)));
                    return;
                }
                /* DIRTY CODE, use type attribute for type-bound procedure as a flag */
                if (TBP_BINDING_ATTRS(fin) & TYPE_BOUND_PROCEDURE_IS_FINAL) {
                    error("FINAL subroutine %s used duplicately",
                          SYM_NAME(ID_SYM(fin)));
                    return;
                }
                if (!check_final_subroutine_is_valid(fin, tp)) {
                    return;
                }
                TBP_BINDING_ATTRS(fin) |= TYPE_BOUND_PROCEDURE_IS_FINAL;
                ID_TYPE(binding) = ID_TYPE(fin);
            }

            FOREACH_ID(fin1, TBP_BINDING(final)) {
                FOREACH_ID(fin2, ID_NEXT(fin1)) {
                    if (function_type_is_compatible(ID_TYPE(fin1), ID_TYPE(fin2))) {
                        error("duplicate FINAL SUBROUTINE types");
                    }
                }
            }
        }
    }
}


/* end of procedure. generate variables, epilogs, and prologs */
static void
end_procedure()
{
    ID id;
    EXT_ID ext;
    BLOCK_ENV bp;
    EXT_ID ep;

    /* Check if a block construct is closed */
    if (CTL_TYPE(ctl_top) == CTL_BLOCK &&
        EXPR_BLOCK(CTL_BLOCK_STATEMENT(ctl_top)) == NULL) {
        error("expecting END BLOCK statement");
    }

    /* Check if a forall construct is closed */
    if (CTL_TYPE(ctl_top) == CTL_FORALL) {
        error("expecting END FORALL statement");
    }

    if (unit_ctl_level > 0 && CURRENT_PROC_NAME == NULL &&\
        CTL_TYPE(ctl_top) != CTL_BLOCK) {
        /* if CURRENT_PROC_NAME == NULL, then this is the end of CONTAINS */
        end_contains();
    }

    /* Since module procedures may be defined not only in contains block but */
    /* also in used modules, the following code is moved from end_contains. */

    if (CURRENT_PROC_CLASS == CL_MAIN ||
        CURRENT_PROC_CLASS == CL_PROC ||
        CURRENT_PROC_CLASS == CL_MODULE ||
        CURRENT_PROC_CLASS == CL_SUBMODULE ||
        CURRENT_PROC_CLASS == CL_BLOCK) {
        if (CURRENT_EXT_ID == NULL) {
            /* Any other errors already occured, let compilation carry on. */
            return;
        }
        /* check if module procedures are defined in contains block */
        EXT_ID intr, intrDef, ep;
        FOREACH_EXT_ID(intr, LOCAL_INTERFACES) {
            int hasSub = FALSE, hasFunc = FALSE;

            if (EXT_IS_BLANK_NAME(intr))
                continue;

            FOREACH_EXT_ID(intrDef, EXT_PROC_INTR_DEF_EXT_IDS(intr)) {
                if (EXT_PROC_IS_MODULE_PROCEDURE(intrDef)) {
                    /*
                     * According to JIS X 3001-1, When module procedure is
                     * declared with "module" keyword, procedure should be
                     * declared in that module. But, gfortran seems not to
                     * implement this check. So, we won't implement this
                     * check too.
                     */
                    ep = NULL;
                    ID id = find_ident(EXT_SYM(intrDef));
                    if (id != NULL
                       && ID_CLASS(id) == CL_PROC
                       && ID_IS_OFMODULE(id)) {
                        // intrDef is use associated module procedure.
                        ep = PROC_EXT_ID(id);
                    } else if (EXT_IS_OFMODULE(intrDef)) {
                        continue;
                    } else if (id != NULL) {
                        ep = PROC_EXT_ID(id);
                    }
                    if (ep == NULL || EXT_TAG(ep) != STG_EXT ||
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
                if (FUNCTION_TYPE_IS_GENERIC(EXT_PROC_TYPE(ep))) {
                    continue;
                } else if(IS_SUBR(EXT_PROC_TYPE(ep))) {
                    hasSub = TRUE;
                } else {
                    hasFunc = TRUE;
                }
            }

            if (hasSub && hasFunc) {
                error("function does not belong in a generic subroutine interface");
            }
            if (hasSub) {
                TYPE_BASIC_TYPE(EXT_PROC_TYPE(intr)) = TYPE_SUBR;
                TYPE_DESC tp = FUNCTION_TYPE_RETURN_TYPE(EXT_PROC_TYPE(intr));
                if (tp != NULL) {
                    TYPE_BASIC_TYPE(tp) = TYPE_VOID;
                } else {
                    FUNCTION_TYPE_RETURN_TYPE(EXT_PROC_TYPE(intr)) = type_VOID;
                }
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

    if (CURRENT_PROC_CLASS != CL_MAIN && CURRENT_PROC_CLASS != CL_BLOCK &&
        EXT_PROC_TYPE(CURRENT_EXT_ID) == NULL) {
        error("Function result %s has no IMPLICIT type.", ID_NAME(CURRENT_EXT_ID));
    }

    if (NOT_INDATA_YET) end_declaration();

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
               || t == NULL || TYPE_IS_SAVE(t) || IS_PROCEDURE_TYPE(t))
                continue;
            sz = get_rough_type_size(ID_TYPE(id));
            if (sz >= (auto_save_attr_kb << 10))
                TYPE_SET_SAVE(ID_TYPE(id));
        }
    }

    FinalizeFormat();

    if (EXT_PROC_TYPE(CURRENT_EXT_ID)) {
        TYPE_SET_FOR_FUNC_SELF(EXT_PROC_TYPE(CURRENT_EXT_ID));
    }

    /* expand CL_MULTI */
    FOREACH_ID(id, LOCAL_SYMBOLS) {
        if (ID_CLASS(id) == CL_MULTI && MULTI_ID_LIST(id) != NULL) {
            ID ip, iq;
            ID next;
            SAFE_FOREACH_ID(ip, iq, MULTI_ID_LIST(id)) {
                next = ID_NEXT(id);
                ID_NEXT(id) = ip;
                ID_NEXT(ip) = next;
            }
            MULTI_ID_LIST(id) = NULL;
        }
    }


    /* check undefined variable */
    FOREACH_ID(id, LOCAL_SYMBOLS) {
        if(ID_CLASS(id) == CL_UNKNOWN){
#if 0 // to be solved
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
                            if (TYPE_IS_PROTECTED(tp)) {
                                TYPE_SET_PROTECTED(ID_TYPE(id));
                            }
                            break;
                        }
                        tp = TYPE_REF(tp);
                    }
                } else if (ID_TYPE(id) != NULL){
                    if (current_module_state == M_PUBLIC) {
                        TYPE_SET_PUBLIC(ID_TYPE(id));
                    }
                    if (current_module_state == M_PRIVATE) {
                        TYPE_SET_PRIVATE(ID_TYPE(id));
                    }
                    if (current_module_state == M_PROTECTED) {
                        TYPE_SET_PROTECTED(ID_TYPE(id));
                    }
                }
                ID_DEFINED_BY(id_in_parent) = id;
            }
        }

        if (ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_UNDEFINEDPROC) {

            if(PROC_EXT_ID(id) != NULL) {
                /* undefined procedure is defined in contain statement.  */
                EXT_IS_DEFINED(PROC_EXT_ID(id)) = TRUE;
            } else {
                implicit_declaration(id);
            }
        }
    }


    FOREACH_EXT_ID(ep, LOCAL_EXTERNAL_SYMBOLS) {
        /*
         * Update procedure variables
         */
        update_procedure_variables_forall(EXT_PROC_ID_LIST(ep),
                                          EXT_PROC_STRUCT_DECLS(ep),
                                          EXT_PROC_BLOCKS(ep),
                                          LOCAL_SYMBOLS, /* is_final = */ TRUE);
    }



    if (CTL_TYPE(ctl_top) == CTL_BLOCK) {
        return;
    }

    /* check undefined label */
    FOREACH_ID(id, LOCAL_LABELS) {
        if (LAB_TYPE(id) != LAB_UNKNOWN &&
            LAB_IS_USED(id) && !LAB_IS_DEFINED(id)) {
            error("missing statement number %d", LAB_ST_NO(id));
        }
        checkTypeRef(id);
    }
    FOREACH_BLOCKS(bp, LOCAL_BLOCKS) {
        check_labels_in_block(bp);
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
    case CL_SUBMODULE: /* fall through */
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
    case CL_MODULE:
    case CL_SUBMODULE: {
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

    check_procedure_variables_forall(/*is_final*/ unit_ctl_level == 0);

    check_type_bound_procedures();

    check_final_subroutines();

    if (CURRENT_PROC_CLASS == CL_MODULE) {
        if(!export_module(current_module_name, LOCAL_SYMBOLS,
                          LOCAL_USE_DECLS)) {
#if 0
            error("internal error, fail to export module.");
            exit(1);
#else
            return;
#endif
        }

    }
    if (CURRENT_PROC_CLASS == CL_SUBMODULE) {
        if(!export_submodule(current_module_name,
                             EXT_MODULE_ANCESTOR(CURRENT_EXT_ID)?:EXT_MODULE_PARENT(CURRENT_EXT_ID),
                             LOCAL_SYMBOLS,
                             LOCAL_USE_DECLS)) {
#if 0
            error("internal error, fail to export module.");
            exit(1);
#else
            return;
#endif
        }

    }


    /* if (CURRENT_PROC_CLASS != CL_MODULE) { */
    /* } */

    /* check control nesting */
    if (ctl_top != ctl_base) error("DO loop or BLOCK IF not closed");

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
    CTL cp;

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
        FOR_CTLS(cp) {
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
        if (TYPE_IS_PROTECTED(var_tp) && TYPE_IS_READONLY(var_tp)) {
            error("do variable is PROTECTED");
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

        if (!expr_has_param(do_incr) && expr_is_constant(do_incr)) {
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

        if (!expr_has_param(do_limit) && expr_is_constant(do_limit)) {
            do_limit = expv_reduce_conv_const(var_tp, do_limit);
        }

        if (!expr_has_param(do_init) && expr_is_constant(do_init)) {
            do_init = expv_reduce_conv_const(var_tp, do_init);
        }

        if (!expr_has_param(do_limit) && !expr_has_param(do_init) &&
	    expr_is_constant(do_limit) && expr_is_constant(do_init)) {
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
compile_DO_concurrent_end()
{
    expv init;

    if (CTL_TYPE(ctl_top) != CTL_DO) {
        error("'END DO', out of place");
        return;
    }

    if (debug_flag) {
        fprintf(debug_fp,"\n*** IN END DO:\n");
        print_IDs(LOCAL_SYMBOLS, debug_fp, TRUE);
        print_types(LOCAL_STRUCT_DECLS, debug_fp);
        expv_output(CURRENT_STATEMENTS, debug_fp);
    }

    EXPR_ARG2(CTL_BLOCK(ctl_top)) = CURRENT_STATEMENTS;

    if (endlineno_flag) {
        EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
    }

    init = EXPR_ARG1(EXPR_ARG1(CTL_BLOCK(ctl_top)));

    compile_end_forall_header(init);

    pop_ctl();
    pop_env();
    CURRENT_STATE = INEXEC;

    /*
     * Close the block construct which is genereted in compile_FORALL_statement().
     */
    if (CTL_TYPE(ctl_top) == CTL_BLOCK) {
        compile_ENDBLOCK_statement(list0(F2008_ENDBLOCK_STATEMENT));
    }
}

static void
check_DO_end(ID label)
{
    CTL cp;

    if (label == NULL) {
        /*
         * do ... enddo case.
         */
        if (CTL_TYPE(ctl_top) == CTL_DO) {
            if (endlineno_flag) {
                EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
            }

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
                pop_ctl();
            } else if (EXPR_CODE(CTL_BLOCK(ctl_top)) == F08_DOCONCURRENT_STATEMENT) {
                /*
                 * DO CONCURRENT
                 */
                compile_DO_concurrent_end();

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
                pop_ctl();
            }
        } else {
            error("'do' is not found for 'enddo'");
        }

        return;

    } else {
        /*
         * do - continue case
         */
        while (CTL_TYPE(ctl_top) == CTL_DO &&
               CTL_DO_LABEL(ctl_top) == label) {

            /* close DO block */
            if (EXPR_CODE(CTL_BLOCK(ctl_top)) == F_DOWHILE_STATEMENT) {
                /*
                 * DOWHILE
                 */
                EXPR_ARG2(CTL_BLOCK(ctl_top)) = CURRENT_STATEMENTS;
            } else if (EXPR_CODE(CTL_BLOCK(ctl_top)) == F08_DOCONCURRENT_STATEMENT) {
                /*
                 * DO CONCURRENT
                 */
                compile_DO_concurrent_end();
                EXPR_ARG2(CTL_BLOCK(ctl_top)) = CURRENT_STATEMENTS;
            } else {
                /*
                 * else DO
                 */
                CTL_DO_BODY(ctl_top) = CURRENT_STATEMENTS;
            }

            if (endlineno_flag) {
                EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
            }
            pop_ctl();
        }

        /* check DO loop which is not propery closed. */
        FOR_CTLS(cp) {
            if (CTL_TYPE(cp) == CTL_DO && CTL_DO_LABEL(cp) == label) {
                error("DO loop or IF-block not closed");
                ctl_top = cp;
                pop_ctl();
            }
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
            current_module_name = s;
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
end_module(expr name)
{
    SYMBOL s;

    if (name) {
        if (EXPR_CODE(name) == IDENT &&
            (s = EXPR_SYM(name)) != NULL &&
            SYM_NAME(s) != NULL) {
            if (current_module_name != s) {
                error("expects module name '%s'",
                      SYM_NAME(s));
            }
        } else {
            fatal("internal error, module name is not "
                  "IDENT in %s().", __func__);
        }
    }

    current_module_state = M_DEFAULT;
    current_module_name = NULL;
    CURRENT_STATE = OUTSIDE; /* goto outer, outside state.  */
}

int associate_parent_module(const SYMBOL, const SYMBOL);


void
begin_submodule(expr name, expr module, expr submodule)
{
    /* NOTE:
     *
     * The submodule has host-association with its parent (sub)module.  To
     * represent this behaviour, begin_submoule makes two UNITs.  Identifiers
     * from the parent (sub)module are imported into the parent-side UNIT.  On
     * the otherhand, identifiers declared in the submodule are placed in the
     * child-side UNIT.  These two UNITs are unified into one UNIT in
     * end_submodule().
     */

    SYMBOL module_name = module?EXPR_SYM(module):NULL;
    SYMBOL submodule_name = submodule?EXPR_SYM(submodule):NULL;

    begin_module(name);
    EXT_MODULE_IS_SUBMODULE(CURRENT_EXT_ID) = TRUE;
    EXT_MODULE_ANCESTOR(CURRENT_EXT_ID) = submodule_name?module_name:NULL;
    EXT_MODULE_PARENT(CURRENT_EXT_ID) = submodule_name?:module_name;

    if (associate_parent_module(module_name, submodule_name) == FALSE) {
        error("failed to associate");
    }

    push_unit_ctl(INSIDE); /* just dummy */
    CURRENT_STATE = INSIDE;
    CURRENT_PROC_CLASS = CL_SUBMODULE;
    CURRENT_PROC_NAME = EXPR_SYM(name);
    CURRENT_EXT_ID = PARENT_EXT_ID;
}


static ID
unify_id_list(ID parents, ID childs, int overshadow)
{
    ID ip;
    ID iq;
    ID ret = NULL;
    ID last = NULL;

    SAFE_FOREACH_ID(ip, iq, parents) {
        if (find_ident_head(ID_SYM(ip), childs) != NULL) {
            if (overshadow) {
                /* the child id shadows the parent id */
                /* free(ip); */
                continue;
            } else {
                fatal("internal error, unexpected symbol confliction", __func__);
            }
        }
        ID_LINK_ADD(ip, ret, last);
    }

    SAFE_FOREACH_ID(ip, iq, childs) {
        ID_LINK_ADD(ip, ret, last);
    }
    return ret;
}


static ID
unify_submodule_id_list(ID parents, ID childs)
{
    return unify_id_list(parents, childs, /*overshadow=*/TRUE);
}


static EXT_ID
unify_ext_id_list(EXT_ID parents, EXT_ID childs, int overshadow)
{
    EXT_ID ep;
    EXT_ID eq;
    EXT_ID ret = NULL;
    EXT_ID last = NULL;

    SAFE_FOREACH_EXT_ID(ep, eq, parents) {
        if (find_ext_id_head(ID_SYM(ep), childs) != NULL) {
            if (overshadow) {
                /* the child ext id shadows the parent ext id */
                /* free(ep); */
                continue;
            } else {
                fatal("internal error, unexpected symbol confliction", __func__);
            }
        }
        EXT_LINK_ADD(ep, ret, last);
    }

    SAFE_FOREACH_EXT_ID(ep, eq, childs) {
        EXT_LINK_ADD(ep, ret, last);
    }
    return ret;
}



static EXT_ID
unify_submodule_ext_id_list(EXT_ID parents, EXT_ID childs)
{
    return unify_ext_id_list(parents, childs, /*overshadow=*/TRUE);
}


static TYPE_DESC
unify_struct_decls(TYPE_DESC parents, TYPE_DESC childs, int overshadow)
{
    TYPE_DESC tp;
    TYPE_DESC tq;
    TYPE_DESC ret = NULL;
    TYPE_DESC last = NULL;

    SAFE_FOREACH_STRUCTDECLS(tp, tq, parents) {
        if (overshadow) {
            /* the child struct shadows the parent struct */
            /* free(tp); */
            continue;
        } else {
            fatal("internal error, unexpected symbol confliction", __func__);
        }
        TYPE_SLINK_ADD(tp, ret, last);
    }

    SAFE_FOREACH_STRUCTDECLS(tp, tq, childs) {
        TYPE_SLINK_ADD(tp, ret, last);
    }
    return ret;
}

static TYPE_DESC
unify_submodule_struct_decls(TYPE_DESC parents, TYPE_DESC childs)
{
    return unify_struct_decls(parents, childs, /*overshadow=*/TRUE);
}


static void
unify_submodule_symbol_table()
{
    ENV submodule;
    ENV parent;

    if (CURRENT_PROC_NAME == NULL && CTL_TYPE(ctl_top) != CTL_BLOCK) {
        end_contains();
    }

    submodule = UNIT_CTL_LOCAL_ENV(CURRENT_UNIT_CTL);
    parent = UNIT_CTL_LOCAL_ENV(PARENT_UNIT_CTL);

    ENV_SYMBOLS(parent) =
            unify_submodule_id_list(ENV_SYMBOLS(parent),
                                      ENV_SYMBOLS(submodule));
    ENV_STRUCT_DECLS(parent) =
            unify_submodule_struct_decls(ENV_STRUCT_DECLS(parent),
                                           ENV_STRUCT_DECLS(submodule));
    ENV_COMMON_SYMBOLS(parent) =
            unify_submodule_id_list(ENV_COMMON_SYMBOLS(parent),
                                      ENV_COMMON_SYMBOLS(submodule));
    ENV_EXTERNAL_SYMBOLS(parent) =
            unify_submodule_ext_id_list(ENV_EXTERNAL_SYMBOLS(parent),
                                          ENV_EXTERNAL_SYMBOLS(submodule));
    ENV_INTERFACES(parent) =
            unify_submodule_ext_id_list(ENV_INTERFACES(parent),
                                          ENV_INTERFACES(submodule));

    ENV_USE_DECLS(parent) = ENV_USE_DECLS(submodule);

    pop_unit_ctl();
}

void
end_submodule(expr name) {
    SYMBOL s;
    if (name) {
        if (EXPR_CODE(name) == IDENT &&
            (s = EXPR_SYM(name)) != NULL &&
            SYM_NAME(s) != NULL) {
            if (current_module_name != s) {
                error("expects submodule name '%s'",
                      SYM_NAME(s));
            }
        } else {
            fatal("internal error, submodule name is not "
                  "IDENT in %s().", __func__);
        }
    }
    end_module(NULL);
}

int
is_in_module(void)
{
    return (INMODULE()) ? TRUE : FALSE;
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
    /*
     * TODO(shingo-s):
     *   If the module procedure is private and use-associated,
     *   its name should be invisible from the current scope.
     *   So it may be required to rename the name of module procedure and
     *   make invisible.
     */
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
    EXT_ID ret = NULL, ep, new_ep = NULL;
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

#define ID_SEEM_GENERIC_PROCEDURE(id)                                          \
    (ID_TYPE((id)) != NULL &&                                                  \
     ((ID_CLASS((id)) == CL_PROC &&                                            \
       TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE((id))))               \
         == TYPE_GENERIC) ||                                                   \
      (TYPE_BASIC_TYPE(ID_TYPE((id))) == TYPE_FUNCTION &&                      \
       TYPE_REF(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE((id)))) != NULL &&           \
       TYPE_BASIC_TYPE(TYPE_REF(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE((id)))))     \
         == TYPE_GNUMERIC_ALL)))

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
    assert(new_tp != NULL);

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
    ID last_ip = NULL;
    ID new_members = NULL;
    TYPE_DESC cur, old;

    cur = tp;
    while (TYPE_REF(cur) != NULL) {
        old = TYPE_REF(cur);
        deep_copy_and_overwrite_for_module_id_type(&(TYPE_REF(cur)));
        if (old == TYPE_REF(cur))
            break;
        cur = TYPE_REF(cur);
    }

    if (IS_STRUCT_TYPE(cur)) {
        if(TYPE_PARENT(cur) && TYPE_PARENT_TYPE(cur)) {
            id = new_ident_desc(ID_SYM(TYPE_PARENT(cur)));
            *id = *TYPE_PARENT(cur);
            TYPE_PARENT(cur) = id;
            deep_copy_and_overwrite_for_module_id_type(&(TYPE_PARENT_TYPE(cur)));
        }

        FOREACH_MEMBER(id, cur) {
            ID new_id = new_ident_desc(ID_SYM(id));
            *new_id = *id;
            deep_copy_and_overwrite_for_module_id_type(&(ID_TYPE(new_id)));

            /*
             * PUBLIC/PRIVATE should be inherited
             */
            if (TYPE_IS_PUBLIC(ID_TYPE(id))) {
                TYPE_SET_PUBLIC(ID_TYPE(new_id));
            }
            if (TYPE_IS_PRIVATE(ID_TYPE(id))) {
                TYPE_SET_PRIVATE(ID_TYPE(new_id));
            }

            ID_LINK_ADD(new_id, new_members, last_ip);
        }
        TYPE_MEMBER_LIST(cur) = new_members;

    } else if (IS_PROCEDURE_TYPE(tp)) {
        SYMBOL s;
        ID last_ip = NULL;
        ID ip;
        ID new_args = NULL;

        deep_copy_and_overwrite_for_module_id_type(&FUNCTION_TYPE_RETURN_TYPE(tp));

        FOREACH_ID(id, FUNCTION_TYPE_ARGS(tp)) {
            s = ID_SYM(id);
            ip = new_ident_desc(s);
            *ip = *id;
            deep_copy_and_overwrite_for_module_id_type(&ID_TYPE(ip));
            ID_LINK_ADD(ip, new_args, last_ip);
        }
        FUNCTION_TYPE_ARGS(tp) = new_args;
    }

}

/**
 * Deep-copy the type and overwrite it
 */
static void
deep_copy_and_overwrite_for_module_id_type(TYPE_DESC * ptp) {
    TYPE_DESC tp;

    if (ptp == NULL || (*ptp == NULL)) {
        return;
    }

    if (type_is_replica(*ptp)) {
        ;  /* do nothing */
    } else if (type_has_replica(*ptp, &tp)) {
        /* overwrite the type with replicated one */
        *ptp = tp;
    } else {
        /* shallow-copy type and deep-copy the  type referenced by this type */
        *ptp = shallow_copy_type_for_module_id(*ptp);
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
    if ((ID_CLASS(id) == CL_TAGNAME &&
         (ID_CLASS(mid) == CL_PROC && IS_GENERIC_PROCEDURE_TYPE(ID_TYPE(mid))))
        ||
        (ID_CLASS(mid) == CL_TAGNAME &&
         (ID_CLASS(id) == CL_PROC && IS_GENERIC_PROCEDURE_TYPE(ID_TYPE(id))))) {
        ID next = ID_NEXT(id);
        id_multilize(id);
        ID_NEXT(id) = MULTI_ID_LIST(id);
        ID_NEXT(ID_NEXT(id)) = next;
        MULTI_ID_LIST(id) = NULL;
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
        ID_IS_AMBIGUOUS(id) = TRUE;
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
            ID_IS_AMBIGUOUS(id) = TRUE;
        }
    }
}

/**
 * import id from module to id list.
 */
static void
import_module_id(ID mid,
                 ID *head, ID *tail,
                 TYPE_DESC *sthead, TYPE_DESC *sttail,
                 SYMBOL use_name, int need_wrap_type, int fromParentModule)
{
    ID existed_id, id;
    EXT_ID ep, mep;

    if ((existed_id = find_ident_head(use_name?:ID_SYM(mid), *head)) != NULL) {
        solve_use_assoc_conflict(existed_id, mid);
        if (ID_CLASS(existed_id) == CL_MULTI) {
            ID ip;
            /* recheck tail */
            FOREACH_ID(ip, *head) {
                *tail = ip;
            }
        } else {
            return;
        }
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
            EXT_PROC_TYPE(ep)
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
    if(need_wrap_type ||
       (ID_STORAGE(id) == STG_TAGNAME && use_name) ||
       TYPE_IS_PROTECTED(ID_TYPE(id))) {
        // shallow copy type from module
        ID_TYPE(id) = shallow_copy_type_for_module_id(ID_TYPE(id));
        TYPE_UNSET_PUBLIC(id);
        TYPE_UNSET_PRIVATE(id);

        /*
         * If type is PROTECTED and id is not imported to SUBMODULE,
         * id should be READ ONLY
         */
        if (TYPE_IS_PROTECTED(ID_TYPE(id)) && !fromParentModule) {
            TYPE_SET_READONLY(ID_TYPE(id));
        }

        ID_ADDR(id) = expv_sym_term(F_VAR, ID_TYPE(id), ID_SYM(id));
    }

    if(ID_TYPE(id) != NULL &&
       IS_PROCEDURE_TYPE(ID_TYPE(id)) &&
       TYPE_IS_PROCEDURE(ID_TYPE(id)) &&
       TYPE_REF(ID_TYPE(id)) == NULL) {
        /*
         * Import 'PROCEDURE(), POINTER :: p'
         * So setup id as unfixed procedure variable.
         */
        TYPE_DESC old = ID_TYPE(id);
        ID_TYPE(id) = function_type(NULL);
        implicit_declaration(id);
        TYPE_SET_IMPLICIT(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)));
        TYPE_SET_IMPLICIT(ID_TYPE(id));
        TYPE_REF(old) = ID_TYPE(id);
        ID_TYPE(id) = old;
        TYPE_SET_NOT_FIXED(ID_TYPE(id));
    }


    if(ID_STORAGE(id) == STG_TAGNAME) {
        TYPE_TAGNAME(ID_TYPE(id)) = id;
        TYPE_SLINK_ADD(ID_TYPE(id), *sthead, *sttail);
    }

    ID_LINK_ADD(id, *head, *tail);

    if(IS_GENERIC_TYPE(ID_TYPE(id)))
        import_generic_procedure(id);

    if(fromParentModule)
        ID_IS_FROM_PARENT_MOD(id) = TRUE;

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

    return;
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


void
deep_copy_id_types(ID mids)
{
    ID mid;
    EXT_ID mep;

    FOREACH_ID(mid, mids) {
        // deep copy of types!
        deep_ref_copy_for_module_id_type(ID_TYPE(mid));

        // deep copy for function types!
        if ((mep = PROC_EXT_ID(mid)) != NULL) {
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
}


static int
import_module_ids(struct module *mod, struct use_argument * args,
                  int isOnly, int fromParentModule)
{
    ID mid, id, last_id = NULL, prev_mid, first_mid;
    TYPE_DESC tp, sttail = NULL;
    struct use_argument * arg;
    int ret = TRUE;
    int wrap_type = TRUE;

    initialize_replicated_type_list();

    if (debug_flag) {
        if (!fromParentModule) {
            fprintf(debug_fp, "######## BEGIN USE ASSOC #######\n");
        } else {
            fprintf(debug_fp, "######## BEGIN HOST ASSOCIATION FROM SUBMODULE  #######\n");
        }
        print_IDs(MODULE_ID_LIST(mod), debug_fp, TRUE);
    }

    FOREACH_ID(id, LOCAL_SYMBOLS) {
        last_id = id;
    }
    prev_mid = last_id;

    FOREACH_STRUCTDECLS(tp, LOCAL_STRUCT_DECLS) {
        sttail = tp;
    }

    FOREACH_ID(mid, MODULE_ID_LIST(mod)) {
        if (args != NULL) {
            FOREACH_USE_ARG(arg, args) {
                wrap_type = TRUE;
                if (arg->local != ID_SYM(mid))
                    continue;
                import_module_id(mid,
                                 &LOCAL_SYMBOLS, &last_id,
                                 &LOCAL_STRUCT_DECLS, &sttail,
                                 arg->use, wrap_type, fromParentModule);
                arg->used = TRUE;
            }
        } else {
            if (!isOnly) {
                wrap_type = TRUE;
                import_module_id(mid,
                                 &LOCAL_SYMBOLS, &last_id,
                                 &LOCAL_STRUCT_DECLS, &sttail,
                                 NULL, wrap_type, fromParentModule);
            }
        }
    }

    FOREACH_USE_ARG(arg, args) {
        if (!arg->used) {
            error("'%s' is not found in module '%s'",
                  SYM_NAME(arg->local), SYM_NAME(MODULE_NAME(mod)));
            ret = FALSE;
        }
    }

    // deep-copy types now!
    first_mid = prev_mid ? ID_NEXT(prev_mid) : NULL;
    deep_copy_id_types(first_mid);

    finalize_replicated_type_list();

    if(debug_flag) {
        if (!fromParentModule) {
            fprintf(debug_fp, "########   END USE ASSOC #######\n");
        } else {
            fprintf(debug_fp, "########   END HOST ASSOCIATION FROM SUBMODULE  #######\n");
        }

    }
    return ret;
}

/**
 * common use assoc
 */
int
use_assoc_common(SYMBOL name, struct use_argument * args, int isOnly)
{
    struct module *mod;

    if (!import_module(name, &mod)) {
        return FALSE;
    }

    return import_module_ids(mod, args, isOnly, FALSE);
}

/**
 * use association with rename arguments.
 * import public identifiers from module to LOCAL_SYMBOLS.
 */
int
use_assoc(SYMBOL name, struct use_argument * args)
{
    int isOnly = FALSE;
    return use_assoc_common(name, args, isOnly);
}

/**
 * use association with only arguments.
 * import public identifiers from module to LOCAL_SYMBOLS.
 */
int
use_assoc_only(SYMBOL name, struct use_argument * args)
{
    int isOnly = TRUE;
    return use_assoc_common(name, args, isOnly);
}

/*
 * compiles use statement.
 */
static void
compile_USE_decl (expr x, expr x_args, int is_intrinsic)
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
    if(is_intrinsic){
        v = expv_cons(F03_USE_INTRINSIC_STATEMENT, NULL, x, args);
    } else {
        v = expv_cons(F95_USE_STATEMENT, NULL, x, args);
    }
    
    EXPV_LINE(v) = EXPR_LINE(x);
    output_statement(v);

    use_assoc(EXPR_SYM(x), use_args);

    list_put_last(LOCAL_USE_DECLS, x);
}

/*
 * compiles use only statement.
 */
static void
compile_USE_ONLY_decl (expr x, expr x_args, int is_intrinsic)
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

    if(is_intrinsic) {
        v = expv_cons(F03_USE_ONLY_INTRINSIC_STATEMENT, NULL, x, args);
    } else {
        v = expv_cons(F95_USE_ONLY_STATEMENT, NULL, x, args);
    }
    
    EXPV_LINE(v) = EXPR_LINE(x);
    output_statement(v);

    use_assoc_only(EXPR_SYM(x), use_args);

    list_put_last(LOCAL_USE_DECLS, x);
}

int
associate_parent_module(const SYMBOL module, const SYMBOL submodule)
{
    struct module *mod;

    if (!import_submodule(module, submodule, &mod)) {
        return FALSE;
    }

    return import_module_ids(mod, NULL, FALSE, TRUE);
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
            } else if(ID_CLASS(iid) == CL_TAGNAME) {
                /*
                 * There is the derived-type with the same name,
                 * so turn id into the multi class identifier.
                 */
                id_multilize(iid);
                iid = declare_ident(s, CL_PROC);

            } else if(ID_STORAGE(iid) == STG_UNKNOWN) {
                ID_STORAGE(iid) = STG_EXT;
                ID_CLASS(iid) = CL_PROC;
            } else if(ID_IS_OFMODULE(iid)) {
                if(!IS_GENERIC_PROCEDURE_TYPE(ID_TYPE((iid))))
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
            info->class = INTF_ASSIGNMENT;
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
        case F03_GENERIC_WRITE: {
            expr formatted = EXPR_ARG1(identOrOp);
            switch (EXPR_CODE(formatted)) {
                case F03_FORMATTED:
                    s = find_symbol("_write_formatted");
                    info->class = INTF_GENERIC_WRITE_FORMATTED;
                    break;
                case F03_UNFORMATTED:
                    s = find_symbol("_write_unformatted");
                    info->class = INTF_GENERIC_WRITE_UNFORMATTED;
                    break;
                default:
                    /* never reach */
                    break;
            }
        } break;
        case F03_GENERIC_READ: {
            expr formatted = EXPR_ARG1(identOrOp);
            switch (EXPR_CODE(formatted)) {
                case F03_FORMATTED:
                    s = find_symbol("_read_formatted");
                    info->class = INTF_GENERIC_READ_FORMATTED;
                    break;
                case F03_UNFORMATTED:
                    s = find_symbol("_read_unformatted");
                    info->class = INTF_GENERIC_READ_UNFORMATTED;
                    break;
                default:
                    /* never reach */
                    break;
            }
        } break;
        case F03_ABSTRACT_SPEC: {
            hasName = FALSE;
            info->class = INTF_ABSTRACT;
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

    push_ctl(CTL_INTERFACE);
    push_unit_ctl(ININTR);

    /* replace the current contol list */
    UNIT_CTL_INTERFACE_SAVE_CTL(CURRENT_UNIT_CTL) = ctl_top;
    UNIT_CTL_INTERFACE_SAVE_CTL_BASE(CURRENT_UNIT_CTL) = ctl_base;
    ctl_base = new_ctl();
    ctl_top = ctl_base;

    CURRENT_INTERFACE = ep;
}

static int
check_interface_type(EXT_ID ep, TYPE_DESC ftp)
{
    switch (EXT_PROC_INTERFACE_INFO(ep)->class) {
        case INTF_GENERIC_READ_FORMATTED:
            return is_defined_io_read_formatted(ftp, NULL);
        case INTF_GENERIC_READ_UNFORMATTED:
            return is_defined_io_read_unformatted(ftp, NULL);
        case INTF_GENERIC_WRITE_FORMATTED:
            return is_defined_io_write_formatted(ftp, NULL);
        case INTF_GENERIC_WRITE_UNFORMATTED:
            return is_defined_io_write_unformatted(ftp, NULL);
        default:
            return TRUE;
    }
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


    if (endlineno_flag) {
        if (CURRENT_INTERFACE && EXT_LINE(CURRENT_INTERFACE))
            EXT_END_LINE_NO(CURRENT_INTERFACE) = current_line->ln_no;
    }

    ctl_top = UNIT_CTL_INTERFACE_SAVE_CTL(CURRENT_UNIT_CTL);
    ctl_base = UNIT_CTL_INTERFACE_SAVE_CTL_BASE(CURRENT_UNIT_CTL);
    pop_unit_ctl();
    pop_ctl();

    /* add INTERFACE symbol to parent */
    if (LOCAL_INTERFACES == NULL) {
        LOCAL_INTERFACES = intr;
    } else {
        /* extid_put_last(EXT_PROC_INTERFACES(PARENT_EXT_ID), intr); */
        extid_put_last(LOCAL_INTERFACES, intr);
    }

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

        if (INTF_IS_ABSTRACT(EXT_PROC_INTERFACE_INFO(intr))) {
            /*
             * FUNCTION/SUBROUTINE inside ABSTRACT INTERFACE are
             * abstract procedures.
             */
            TYPE_SET_ABSTRACT(EXT_PROC_TYPE(ep));
        }

        if (!check_interface_type(intr, EXT_PROC_TYPE(ep))) {
            return;
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
        if (ID_CLASS(iid) == CL_MULTI) {
            iid = multi_find_class(iid, CL_PROC);
        }

        /* type should be calculated from
         * declared functions, not always TYPE_GNUMERIC */
        ID_CLASS(iid) = CL_PROC;
        ID_TYPE(iid) = hasSub ? generic_subroutine_type() : generic_function_type();
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
        ID_TYPE(id) = generic_procedure_type();
        ID_CLASS(id) = CL_PROC;
        PROC_CLASS(id) = P_DEFINEDPROC;
        declare_function(id);
    }
}


/*
 * compile MODULE PROCEDURE statement in the INTERFACE block
 */
static void
compile_interface_MODULEPROCEDURE_statement(expr x)
{
    list lp;
    expr ident;
    ID id;
    EXT_ID ep;
    const char *genProcName = NULL;

    assert(PARENT_STATE == ININTR);

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
        if (EXT_PROC_TYPE(ep) != NULL) {
            FUNCTION_TYPE_SET_MOUDLE_PROCEDURE(EXT_PROC_TYPE(ep));
        }

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
 * compile MODULE PROCEDURE statement in the CONTAINS block of the submodule
 */
static void
compile_separate_MODULEPROCEDURE_statement(expr x)
{
    SYMBOL s;
    expr name;
    ID ip = NULL;
    ID id;
    ID arg;

    assert(PARENT_STATE == INCONT);
    assert(EXPR_HAS_ARG1(EXPR_ARG1(x)));
    assert(!EXPR_HAS_ARG2(EXPR_ARG1(x)));

    name = EXPR_ARG1(EXPR_ARG1(x));
    s = EXPR_SYM(name);

    if ((ip = find_ident(s)) == NULL) {
        error("module procedure interface doesn't exsit");
        return;
    } else if(!IS_PROCEDURE_TYPE(ID_TYPE(ip))) {
        error("parent should be a procedure");
        return;
    } else if(!TYPE_IS_MODULE(ID_TYPE(ip))) {
        error("parent should have a modure prefix");
        return;
    } else if (FUNCTION_TYPE_IS_DEFINED(ID_TYPE(ip))) {
        error("%s is already defined", SYM_NAME(ID_SYM(ip)));
        return;
    }

    FUNCTION_TYPE_SET_DEFINED(ID_TYPE(ip));

    begin_procedure();
    declare_procedure(CL_PROC, name, ID_TYPE(ip), NULL, NULL, NULL, NULL);
    EXT_PROC_IS_PROCEDUREDECL(CURRENT_EXT_ID) = TRUE;

    /*
     * setup local symbols in the function
     */
    if (FUNCTION_TYPE_RESULT(ID_TYPE(ip))) {
        s = FUNCTION_TYPE_RESULT(ID_TYPE(ip));
        declare_function_result_id(s, FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(ip)));
    }

    FOREACH_ID(arg, FUNCTION_TYPE_ARGS(ID_TYPE(ip))) {
        id = declare_ident(ID_SYM(arg), CL_VAR);
        ID_STORAGE(id) = STG_ARG;
        declare_id_type(id, ID_TYPE(arg));
        TYPE_SET_UNCHANGABLE(ID_TYPE(id));
    }
}


/*
 * compile MODULE PROCEDURE statement
 */
static void
compile_MODULEPROCEDURE_statement(expr x)
{
    if (PARENT_STATE == ININTR) {
        compile_interface_MODULEPROCEDURE_statement(x);
    } else if (PARENT_STATE == INCONT) {
        compile_separate_MODULEPROCEDURE_statement(x);
    } else {
        error("unexpected MODULE PROCEDURE statement");
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
    //	 should work for all cases (array/substr/plain scalar).
    if (!IS_FUNCTION_TYPE(ID_TYPE(member_id)) && (
            TYPE_HAS_SUBOBJECT_PROPAGATE_ATTRS(stVTyp) ||
            TYPE_IS_COINDEXED(stVTyp))) {
        /*
         * If type of struct_v has pointer/pointee flags on, members
         * should have those flags on too.
         *
         * And if type of struct_v is coarray, members are coarray.
         */
        TYPE_DESC mVTyp = ID_TYPE(member_id);
        TYPE_DESC retTyp = NULL;
        if (IS_ARRAY_TYPE(mVTyp)) {
            generate_shape_expr(mVTyp, shape);
            mVTyp = bottom_type(mVTyp);
        }
        retTyp = wrap_type(mVTyp);

        TYPE_SET_SUBOBJECT_PROPAGATE_ATTRS(retTyp, mVTyp);
        TYPE_SET_SUBOBJECT_PROPAGATE_EXTATTRS(retTyp, mVTyp);
        TYPE_ATTR_FLAGS(retTyp) |= TYPE_IS_ALLOCATABLE(mVTyp);

        TYPE_SET_SUBOBJECT_PROPAGATE_ATTRS(retTyp, stVTyp);
        TYPE_SET_SUBOBJECT_PROPAGATE_EXTATTRS(retTyp, stVTyp);
        TYPE_CODIMENSION(retTyp) = TYPE_CODIMENSION(stVTyp);

        tp = retTyp;
        tp = compile_dimensions(tp, shape);
    } else {
        tp = ID_TYPE(member_id);

    }

    //tp = compile_dimensions(tp, shape);
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
        if (TYPE_IS_PROTECTED(EXPV_TYPE(ev)) && TYPE_IS_READONLY(EXPV_TYPE(ev))) {
            error("argument is a PROTECTED type");
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
compile_ALLOCATE_DEALLOCATE_statement(expr x)
{
    /* (F95_ALLOCATE_STATEMENT args) */
    expr r, kwd;
    expv args, v, vstat = NULL, vmold = NULL, vsource = NULL, verrmsg = NULL;
    list lp;
    enum expr_code code = EXPR_CODE(x);

    expr type = EXPR_HAS_ARG2(x)?EXPR_ARG2(x):NULL;
    TYPE_DESC tp = NULL;

    int isImageControlStatement = FALSE;

    args = list0(LIST);
    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        r = LIST_ITEM(lp);

        if(EXPR_CODE(r) == F_SET_EXPR) {
            kwd = EXPR_ARG1(r);
            if(EXPR_CODE(kwd) != IDENT ||
               (strcmp(SYM_NAME(EXPR_SYM(kwd)), "stat") != 0 &&
                strcmp(SYM_NAME(EXPR_SYM(kwd)), "mold") != 0 &&
                strcmp(SYM_NAME(EXPR_SYM(kwd)), "errmsg") != 0 &&
                strcmp(SYM_NAME(EXPR_SYM(kwd)), "source") != 0)) {
                error("invalid keyword list");
                break;
            }
            v = compile_expression(EXPR_ARG2(r));
            if (strcmp(SYM_NAME(EXPR_SYM(kwd)), "stat") == 0) {

                if (vstat != NULL) {
                    error("duplicate stat keyword");
                }

                vstat = compile_expression(v);

                if (vstat == NULL || (EXPR_CODE(vstat) != F_VAR &&
                                      EXPR_CODE(vstat) != ARRAY_REF &&
                                      EXPR_CODE(vstat) != F95_MEMBER_REF)){
                    error("invalid status variable");
                }

            } else if (strcmp(SYM_NAME(EXPR_SYM(kwd)), "mold") == 0) {
                if (code == F95_DEALLOCATE_STATEMENT) {
                    error("MOLD keyword argument in DEALLOCATE statement");
                }

                if (vmold != NULL) {
                    error("duplicate mold keyword");
                }

                vmold = compile_expression(v);

            } else if (strcmp(SYM_NAME(EXPR_SYM(kwd)), "source") == 0) {
                if (code == F95_DEALLOCATE_STATEMENT) {
                    error("SOURCE keyword argument in DEALLOCATE statement");
                }

                if (vsource != NULL) {
                    error("duplicate source keyword");
                }

                vsource = compile_expression(v);

            } else if (strcmp(SYM_NAME(EXPR_SYM(kwd)), "errmsg") == 0) {
                if (verrmsg != NULL) {
                    error("duplicate errmsg keyword");
                }

                verrmsg = compile_expression(v);

                if (verrmsg == NULL || (EXPR_CODE(verrmsg) != F_VAR &&
                                        EXPR_CODE(verrmsg) != ARRAY_REF &&
                                        EXPR_CODE(verrmsg) != F95_MEMBER_REF)){
                    error("invalid errmsg variable");

                }
                
                if(IS_CHAR(EXPV_TYPE(verrmsg)) == FALSE) {
                    error("errmsg variable is not a scala character type");
                }


            }
        } else {
            if (vstat || vmold || vsource || verrmsg) {
                error("non-keyword arguments after keyword arguments");
                continue;
            }

            is_in_alloc = TRUE;
            expv ev = compile_lhs_expression(r);
            is_in_alloc = FALSE;

            if (ev == NULL)
                continue;

            if (TYPE_IS_COINDEXED(EXPV_TYPE(ev))) {
                isImageControlStatement = TRUE;
            }

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

    if (isImageControlStatement && !check_image_control_statement_available())
        return;

    if (type) {
        tp = compile_type(type, /*allow_predecl=*/FALSE);
    }

    if (vstat) {
        if (TYPE_IS_PROTECTED(EXPV_TYPE(vstat)) && TYPE_IS_READONLY(EXPV_TYPE(vstat))) {
            error("an argument for STAT is PROTECTED");
        }
    }
    if (verrmsg) {
        if (TYPE_IS_PROTECTED(EXPV_TYPE(verrmsg)) && TYPE_IS_READONLY(EXPV_TYPE(verrmsg))) {
            error("an argument for ERRMSG is PROTECTED");
        }
    }

    /*
     * Now check type for allocation
     */

    FOR_ITEMS_IN_LIST(lp, args) {
        if (tp) {
            if (type_is_compatible_for_allocation(EXPV_TYPE(LIST_ITEM(lp)),
                                                  tp)) {
                error("type incompatible");
                return;
            }

            if (TYPE_IS_PROTECTED(tp) && TYPE_IS_READONLY(tp)) {
                error("an argument for STAT is PROTECTED");
            }
        }

        if (vsource) {
            if (type_is_compatible_for_allocation(EXPV_TYPE(LIST_ITEM(lp)),
                                                  EXPV_TYPE(vsource))) {
                error("type incompatible");
                return;
            }
        }
        if (vmold) {
            if (type_is_compatible_for_allocation(EXPV_TYPE(LIST_ITEM(lp)),
                                                  EXPV_TYPE(vmold))) {
                return;
                error("type incompatible");
            }
        }
    }

    v = expv_cons(code, NULL, args, list4(LIST, vstat, vmold, vsource, verrmsg));
    EXPV_TYPE(v) = tp;

    EXPV_LINE(v) = EXPR_LINE(x);
    output_statement(v);
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
compile_CALL_type_bound_procedure_statement(expr x)
{
    ID tpd;
    expv structRef;
    expr x1, x2, args;
    TYPE_DESC stp;
    TYPE_DESC tp;
    expv v = NULL;
    expv a;

    x1 = EXPR_ARG1(EXPR_ARG1(x));
    x2 = EXPR_ARG2(EXPR_ARG1(x));
    args = EXPR_ARG2(x);

    a = compile_args(args);

    structRef = compile_lhs_expression(x1);

    if (!IS_STRUCT_TYPE(EXPV_TYPE(structRef))) {
        error("invalid type bound procedure call to non derived-type");
        return;
    }

    stp = EXPV_TYPE(structRef);

    tpd = find_struct_member(stp, EXPR_SYM(x2));

    if (tpd == NULL || ID_CLASS(tpd) != CL_TYPE_BOUND_PROC) {
        error("'%s' is not type bound procedure", SYM_NAME(EXPR_SYM(x2)));
        return;
    }

    if ((tp = ID_TYPE(tpd)) == NULL) {
        /*
         * If type bound procedure is bound to module procedure, its type does
         * not yet exists.  So create it in this timing.
         */
        tp = new_type_desc();
        TYPE_SET_USED_EXPLICIT(tp);
        TYPE_BASIC_TYPE(tp) = TYPE_SUBR;
    }

    tp = ID_TYPE(tpd);
    if (TYPE_BOUND_GENERIC_TYPE_GENERICS(tp)) {
        /* for type-bound GENERIC */
        ID bind;
        ID bindto;
        tp = NULL;
        FOREACH_ID(bind, TBP_BINDING(tpd)) {
            bindto = find_struct_member_allow_private(stp, ID_SYM(bind), TRUE);
            if (bindto && function_type_is_appliable(ID_TYPE(bindto), a, TRUE)) 
            {
                tp = ID_TYPE(bindto);
            }
        }
        if (tp == NULL) {
            if (debug_flag)
                fprintf(debug_fp, "invalid argument for type-bound generic");
        }
        /* type-bound generic procedure type does not exist in XcodeML */
        tp = NULL;
    }

    v = list2(FUNCTION_CALL,
              expv_cons(F95_MEMBER_REF, tp, structRef, x2),
              a);

    EXPV_TYPE(v) = type_VOID;
    output_statement(v);

    return;
}


static void
compile_CALL_subroutine_statement(expr x)
{
    expr x1;
    ID id;
    expv v;

    x1 = EXPR_ARG1(x);

    id = find_ident(EXPR_SYM(x1));
    if (id == NULL) {
        id = find_external_ident_head(EXPR_SYM(x1));
    }
    if(id == NULL) {
        id = declare_ident(EXPR_SYM(x1), CL_UNKNOWN);
        if (ID_CLASS(id) == CL_UNKNOWN) {
            ID_CLASS(id) = CL_PROC;
        }
        if (is_intrinsic_function(id)) {
            PROC_CLASS(id) = P_INTRINSIC;
            TYPE_SET_INTRINSIC(id);
            ID_STORAGE(id) = STG_NONE;
            ID_IS_DECLARED(id) = TRUE;
        } else if (PROC_CLASS(id) == P_UNKNOWN) {
            PROC_CLASS(id) = P_EXTERNAL;
            /* DO NOT TYPE_SET_EXTERNAL(id), this is not an explicit exernal subroutine */
            TYPE_SET_IMPLICIT(id);
        }
    }
    if (ID_IS_AMBIGUOUS(id)) {
        error("an ambiguous reference to symbol '%s'", ID_NAME(id));
        return;
    }
    if (ID_TYPE(id) != NULL) {
        if(IS_FUNCTION_TYPE(ID_TYPE(id)) &&
           TYPE_IS_USED_EXPLICIT(ID_TYPE(id))) {
            error("'%s' is a function, not a subroutine", ID_NAME(id));

        } else if (TYPE_IS_ABSTRACT(ID_TYPE(id))) {
            error("'%s' is abstract", ID_NAME(id));

        }
    }

    if ((PROC_CLASS(id) == P_EXTERNAL || PROC_CLASS(id) == P_UNKNOWN) &&
        (ID_TYPE(id) == NULL || (
            IS_SUBR(ID_TYPE(id)) == FALSE &&
            TYPE_IS_PROCEDURE(ID_TYPE(id)) == FALSE))) {
        TYPE_DESC tp;
        if (ID_TYPE(id)) {
            if (!TYPE_IS_IMPLICIT(ID_TYPE(id)) &&
                !FUNCTION_TYPE_IS_GENERIC(ID_TYPE(id)) &&
                !(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)) != NULL &&
                  (IS_VOID(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id))) ||
                   TYPE_IS_IMPLICIT(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id))) ||
                   IS_GENERIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)))))) {
                error("called '%s' which doesn't have a type like a subroutine", ID_NAME(id));
                return;
            }

            tp = subroutine_type();
            TYPE_ATTR_FLAGS(tp) = TYPE_ATTR_FLAGS(ID_TYPE(id));
        } else {
            tp = subroutine_type();
        }
        TYPE_SET_IMPLICIT(tp);
        TYPE_SET_USED_EXPLICIT(tp);
        ID_TYPE(id) = tp;

        if(PROC_EXT_ID(id)) {
            EXT_PROC_TYPE(PROC_EXT_ID(id)) = tp;
        }
    }
    else if (ID_TYPE(id) != NULL && TYPE_IS_PROCEDURE(ID_TYPE(id))) {
        if (!IS_SUBR(ID_TYPE(id))) {
            TYPE_BASIC_TYPE(ID_TYPE(id)) = TYPE_SUBR;
        }
    }
    else if (PROC_CLASS(id) == P_INTRINSIC && ID_TYPE(id) != NULL){
        TYPE_DESC tp = ID_TYPE(id);
        TYPE_BASIC_TYPE(tp) = TYPE_SUBR;
        FUNCTION_TYPE_RETURN_TYPE(tp) = type_VOID;
        TYPE_UNSET_IMPLICIT(tp);
        TYPE_SET_USED_EXPLICIT(tp);
        ID_TYPE(id) = tp;
        if (PROC_EXT_ID(id)) EXT_PROC_TYPE(PROC_EXT_ID(id)) = tp;
    }
    else if (ID_TYPE(id) != NULL && !IS_SUBR(ID_TYPE(id))) {
        TYPE_DESC tp = subroutine_type();
        TYPE_EXTATTR_FLAGS(tp) = TYPE_EXTATTR_FLAGS(ID_TYPE(id));
        TYPE_UNSET_IMPLICIT(tp);
        TYPE_SET_USED_EXPLICIT(tp);
        ID_TYPE(id) = tp;
    }


#if 0
    // to be solved
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
       v = compile_function_call_check_intrinsic_arg_type(id, EXPR_ARG2(x), TRUE);
       if (v == NULL && PROC_CLASS(id) == P_INTRINSIC) {
           TYPE_DESC tp = type_basic(TYPE_SUBR);
           /* Retry to compile as 'CALL external_subroutine(..)' . */

           id = declare_ident(EXPR_SYM(x1), CL_PROC);
           ID_TYPE(id) = tp;

           /* NOTE: DO NOT 'TYPE_SET_EXTERNAL(id)', this is not an explicit exteranl function  */
           ID_IS_DECLARED(id) = FALSE;
           ID_STORAGE(id) = STG_EXT;
           PROC_CLASS(id) = P_EXTERNAL;

           TYPE_UNSET_IMPLICIT(tp);
           TYPE_SET_USED_EXPLICIT(tp);

           v = compile_function_call(id, EXPR_ARG2(x));

        }
    }

    EXPV_TYPE(v) = type_basic(TYPE_VOID);
    output_statement(v);
}

static void
compile_CALL_statement(expr x)
{
    expr x1;
    /* (F_CALL_STATEMENT identifier args)*/
    x1 = EXPR_ARG1(x);
    if (EXPR_CODE(x1) == IDENT) {
        compile_CALL_subroutine_statement(x);
    } else if (EXPR_CODE(x1) == F95_MEMBER_REF) {
        compile_CALL_type_bound_procedure_statement(x);
    } else {
        fatal("compile_exec_statement: bad id in call");
    }
}


static void
compile_RETURN_statement(expr x)
{
    /* (F_RETURN_STATMENT arg) */
    if (check_inside_CRITICAL_construct()) {
        error("RETURN statement in CRITICAL block");
        return;
    }

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

static int markAsProtected(ID id)
{
    TYPE_DESC tp = ID_TYPE(id);
    if (TYPE_IS_PRIVATE(id) || (tp != NULL && TYPE_IS_PRIVATE(tp))) {
        error("'%s' is already specified as private.", ID_NAME(id));
        return FALSE;
    }
    if (TYPE_IS_PUBLIC(id) || (tp != NULL && TYPE_IS_PUBLIC(tp))) {
        error("'%s' is already specified as public.", ID_NAME(id));
        return FALSE;
    }
    TYPE_UNSET_PUBLIC(id);
    TYPE_SET_PROTECTED(id);

    return TRUE;
}

static int
have_type_bound_procedure(ID ids)
{
    ID ip;
    FOREACH_ID(ip, ids) {
        if (ID_CLASS(ip) == CL_TYPE_BOUND_PROC) {
            return TRUE;
        }
    }
    return FALSE;
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
            if (have_type_bound_procedure(TYPE_MEMBER_LIST(struct_tp))) {
                error("PRIVATE after type-bound procedure");
            }
            TYPE_SET_INTERNAL_PRIVATE(struct_tp);
            return;
        } else if (markAs == markAsPublic) {
            current_module_state = M_PUBLIC;
        } else if (markAs == markAsPrivate)  {
            current_module_state = M_PRIVATE;
        } else if (markAs == markAsProtected) {
            current_module_state = M_PROTECTED;
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
        if (EXPR_CODE(EXPR_ARG1(x)) == IDENT) {
            error_at_node(x, "'%s' is not a pointer.",
                          SYM_NAME(EXPR_SYM(EXPR_ARG1(x))));
        } else {
            error_at_node(x, "lhs is not a pointer.",
                          SYM_NAME(EXPR_SYM(EXPR_ARG1(x))));
        }
        return;
    }

    if (TYPE_IS_PROTECTED(vPtrTyp) && TYPE_IS_READONLY(vPtrTyp)) {
        error_at_node(x, "'%s' is PROTECTED.",
                      SYM_NAME(EXPR_SYM(EXPR_ARG1(x))));
        return;
    }

    if (IS_PROCEDURE_TYPE(EXPV_TYPE(vPointer)) &&
        FUNCTION_TYPE_IS_TYPE_BOUND(EXPV_TYPE(vPointer))) {
            error("lhs expr is type bound procedure.");
            return;
    }
    if (IS_PROCEDURE_TYPE(EXPV_TYPE(vPointee)) &&
        FUNCTION_TYPE_IS_TYPE_BOUND(EXPV_TYPE(vPointee))) {
            error("rhs expr is type bound procedure.");
            return;
    }

    if (IS_PROCEDURE_TYPE(vPtrTyp)) {
        /* if left operand is a procedure type,
         * right operand is a function/subroutine,
         * and they may be declared in the CONTAINS block.
         */
        if (!IS_PROCEDURE_TYPE(vPteTyp) &&
            TYPE_BASIC_TYPE(vPteTyp) != TYPE_UNKNOWN &&
            !TYPE_IS_IMPLICIT(vPteTyp)) {
            error_at_node(x, "'%s' is not a function/subroutine",
                          SYM_NAME(EXPR_SYM(EXPR_ARG2(x))));
        }
        if (TYPE_IS_ABSTRACT(vPteTyp)) {
            error_at_node(x, "'%s' is an abstract interface",
                          SYM_NAME(EXPR_SYM(EXPR_ARG2(x))));
        }


        if (EXPR_CODE(vPointee) == F_VAR) {
            ID id = find_ident(EXPR_SYM(vPointee));
            if (!IS_PROCEDURE_TYPE(vPteTyp)) {
                assert(id != NULL); /* declared in compile_expression() */
                if (TYPE_IS_IMPLICIT(vPteTyp) && TYPE_REF(vPtrTyp)) {
                    /*
                     * ex)
                     *  ! f is a procedure pointer
                     *  f => g ! g is pointee
                     *
                     *  So assumption: g is a procedure
                     *
                     *  redefine_procedures() will check 'g' is defined or not
                     */
                    TYPE_DESC tp;
                    TYPE_DESC ftp = get_bottom_ref_type(vPtrTyp);
                    int attrs = TYPE_ATTR_FLAGS(vPteTyp);
                    int extattrs = TYPE_EXTATTR_FLAGS(vPteTyp);

                    *vPteTyp = *ftp;
                    tp = new_type_desc();
                    *tp = *FUNCTION_TYPE_RETURN_TYPE(vPteTyp);
                    FUNCTION_TYPE_RETURN_TYPE(vPteTyp) = tp;
                    TYPE_ATTR_FLAGS(vPteTyp) = attrs & !(TYPE_ATTR_PUBLIC | TYPE_ATTR_PRIVATE);
                    TYPE_EXTATTR_FLAGS(vPteTyp) = extattrs;
                    FUNCTION_TYPE_HAS_EXPLICIT_ARGS(vPteTyp) = TRUE;
                } else {
                    /*
                     * POINTEE is used as a function/subroutine,
                     * so fix its type
                     *
                     * ex)
                     *
                     *   REAL :: g ! may be a external function
                     *
                     *   f => g
                     */

                    TYPE_DESC old;

                    old = ID_TYPE(id);

                    if (IS_FUNCTION_TYPE(vPtrTyp)) {
                        ID_TYPE(id) = function_type(old);

                    } else {
                        ID_TYPE(id) = subroutine_type();
                        TYPE_ATTR_FLAGS(ID_TYPE(id)) = TYPE_ATTR_FLAGS(old);
                        TYPE_EXTATTR_FLAGS(ID_TYPE(id)) = TYPE_EXTATTR_FLAGS(old);
                    }
                }
            } else {
                if (get_bottom_ref_type(vPtrTyp) == get_bottom_ref_type(vPteTyp)) {
                    /* DO NOTHING */
                } else if ((IS_FUNCTION_TYPE(vPteTyp) &&
                     TYPE_IS_IMPLICIT(FUNCTION_TYPE_RETURN_TYPE(vPteTyp)))
                    && TYPE_REF(vPtrTyp)) {
                    /*
                     * ex)
                     *  i = g()
                     *  ! f is a procedure pointer
                     *  f => g ! g is pointee
                     *
                     *  So assumption: g is a procedure
                     *
                     *  redefine_procedures() will check 'g' is defined or not
                     */
                    TYPE_DESC ftp = get_bottom_ref_type(vPtrTyp);

                    TYPE_REF(vPteTyp) = ftp;
                    TYPE_REF(FUNCTION_TYPE_RETURN_TYPE(vPteTyp)) = FUNCTION_TYPE_RETURN_TYPE(ftp);
                    TYPE_ATTR_FLAGS(vPteTyp) = 0;
                    TYPE_EXTATTR_FLAGS(vPteTyp) = 0;
                    TYPE_ATTR_FLAGS(FUNCTION_TYPE_RETURN_TYPE(vPteTyp)) = 0;
                    TYPE_EXTATTR_FLAGS(FUNCTION_TYPE_RETURN_TYPE(vPteTyp)) = 0;
                    FUNCTION_TYPE_HAS_EXPLICIT_ARGS(vPteTyp) = TRUE;

                }
            }

            ID_CLASS(id) = CL_PROC;
            PROC_CLASS(id) = P_UNDEFINEDPROC;

            if (ID_LINE(id) == NULL) {
                ID_LINE(id) = EXPR_LINE(x);
            }
        }

    } else {
        if (!TYPE_IS_TARGET(vPteTyp) &&
            !TYPE_IS_POINTER(vPteTyp) &&
            !IS_PROCEDURE_TYPE(vPteTyp)) {
            if(EXPR_CODE(EXPR_ARG2(x)) == IDENT)
                error_at_node(x, "'%s' is not a pointee.",
                              SYM_NAME(EXPR_SYM(EXPR_ARG2(x))));
            else
                error_at_node(x, "right hand side expression is not a pointee.");
            return;
        }
    }

    if (TYPE_N_DIM(IS_REFFERENCE(vPtrTyp)?TYPE_REF(vPtrTyp):vPtrTyp) !=
        TYPE_N_DIM(IS_REFFERENCE(vPteTyp)?TYPE_REF(vPteTyp):vPteTyp)) {
        error_at_node(x, "Rank mismatch.");
        return;
    }

    if (IS_PROCEDURE_TYPE(vPtrTyp)) {
        if (!procedure_is_assignable(vPtrTyp, vPteTyp)) {
            error_at_node(x, "Type mismatch.");
            return;
        }

        if (TYPE_IS_NOT_FIXED(vPtrTyp)) {
            TYPE_DESC subr, ftp;

            subr = get_bottom_ref_type(vPteTyp);
            if (IS_SUBR(subr)) {
                ftp = get_bottom_ref_type(vPtrTyp);
                TYPE_BASIC_TYPE(ftp) = TYPE_SUBR;
                TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(ftp)) = TYPE_VOID;
            }
            TYPE_UNSET_NOT_FIXED(vPtrTyp);
        }

    } else {
        if (get_basic_type(vPtrTyp) != get_basic_type(vPteTyp)) {
            error_at_node(x, "Type mismatch.");
            return;
        }
    }

    if (TYPE_IS_VOLATILE(vPtrTyp) != TYPE_IS_VOLATILE(vPteTyp)) {
        error_at_node(x, "VOLATILE attribute mismatch.");
        return;
    }
    if (TYPE_IS_ASYNCHRONOUS(vPtrTyp) != TYPE_IS_ASYNCHRONOUS(vPteTyp)) {
        error_at_node(x, "ASYNCHRONOUS attribute mismatch.");
        return;
    }

    if (IS_STRUCT_TYPE(vPtrTyp) &&
        !struct_type_is_compatible_for_assignment(vPtrTyp, vPteTyp, TRUE)) {
        error_at_node(x, "Derived-type mismatch.");
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
            error("can't change attributes of USE-associated symbol '%s'", ID_NAME(id));
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
            error("can't change attributes of USE-associated symbol '%s'",
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
            error("can't change attributes of USE-associated symbol '%s'",
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

/*
 * Check if rank + corank <= MAX_DIM
 *
 */
static void
check_array_length(ID id)
{
    if (id == NULL || ID_TYPE(id) == NULL) {
        return;
    }

    if (!IS_ARRAY_TYPE(ID_TYPE(id))) {
        return;
    }

    if (!TYPE_CODIMENSION(ID_TYPE(id))) {
        return;
    }

    if (TYPE_N_DIM(ID_TYPE(id)) + TYPE_CODIMENSION(ID_TYPE(id))->corank > MAX_DIM) {
        error_at_id(id, "Too long array (rank + corank > %d)", MAX_DIM);
    }
}


static void
fix_pointer_pointee_recursive(TYPE_DESC tp)
{
    if (tp == NULL) {
        return;
    }
    if (IS_PROCEDURE_TYPE(tp)) {
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
            if (IS_STRUCT_TYPE(tp) && !TYPE_IS_CLASS(tp)) {
                /*
                 * TYPE_STRUCT base. Don't mark this node as
                 * pointer/pointee, EXCEPT 'CLASS(*)'.
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
    UNIT_CTL_LOCAL_INTERFACES(uc) = NULL;
    UNIT_CTL_IMPLICIT_DECLS(uc) = list0(LIST);
    UNIT_CTL_EQUIV_DECLS(uc) = list0(LIST);
    UNIT_CTL_LOCAL_USE_DECLS(uc) = list0(LIST);

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

    current_local_env = UNIT_CTL_LOCAL_ENV(CURRENT_UNIT_CTL);
    current_local_env->parent = NULL;
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
    if (top_proc != NULL &&
        ID_CLASS(top_proc) != CL_MODULE &&
        ID_CLASS(top_proc) != CL_SUBMODULE) {
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

    push_env(UNIT_CTL_LOCAL_ENV(CURRENT_UNIT_CTL));
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
            FUNCTION_TYPE_SET_INTERNAL(tp);
            ip = find_ident(EXT_SYM(ep));
            if (ID_CLASS(ip) == CL_MULTI) {
                ip = multi_find_class(ip, CL_PROC);
                if (ip == NULL) {
                    fatal("multi class id bug");
                    return;
                }
            }
            if (PROC_EXT_ID(ip) == ep)
                continue;
            if (PROC_CLASS(ip) == P_UNDEFINEDPROC) {
                continue;
            }
            if (ID_DEFINED_BY(ip)) {
                ip = ID_DEFINED_BY(ip);
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
    pop_env();

    if(CURRENT_STATE == INCONT)
        unit_ctl_contains_level --;
}

void
cleanup_ctl(CTL ctl) {
    CTL_TYPE(ctl) = CTL_NONE;
}


CTL
new_ctl() {
    CTL ctl;
    ctl = XMALLOC(CTL, sizeof(*ctl));
    if (ctl == NULL)
        fatal("memory allocation failed");
    cleanup_ctl(ctl);
    CTL_BLOCK_LOCAL_EXTERNAL_SYMBOLS(ctl) = NULL;
    return ctl;
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

/*
 * Common function for compile_sync_stat_args and compile_lock_stat_args
 */
static int
compile_stat_args(expv st, expr x, int expect_acquired_lock) {
    list lp;
    int has_keyword_acquired_lock = FALSE;
    int has_keyword_stat = FALSE;
    int has_keyword_errmsg = FALSE;

    if (x == NULL)
        return TRUE;

    FOR_ITEMS_IN_LIST(lp, x) {
        expr v, arg;

        v = LIST_ITEM(lp);

        if (EXPR_CODE(v) != F_SET_EXPR) {
            fatal("%s: not F_SET_EXPR.", __func__);
        }

        arg = compile_expression(EXPR_ARG2(v));

        char *keyword = SYM_NAME(EXPR_SYM(EXPR_ARG1(v)));
        if (keyword == NULL || *keyword == '\0') {
            fatal("%s: invalid F_SET_EXPR.", __func__);
        }

        if (strcmp("stat", keyword) == 0) {
            if (has_keyword_stat == TRUE) {
                error("no specifier shall appear more than once");
                return FALSE;
            }
            has_keyword_stat = TRUE;

            if (!IS_INT(EXPV_TYPE(arg))) {
                error("stat variable should be interger");
                return FALSE;
            }

        } else if (strcmp("errmsg", keyword) == 0) {
            if (has_keyword_errmsg == TRUE) {
                error("no specifier shall appear more than once");
                return FALSE;
            }
            has_keyword_errmsg = TRUE;

            if (!IS_CHAR(EXPV_TYPE(arg))) {
                error("errmsg variable should be character");
                return FALSE;
            }

        } else if (expect_acquired_lock &&
                   strcmp("acquired_lock", keyword) == 0) {
            if (has_keyword_acquired_lock == TRUE) {
                error("no specifier shall appear more than once");
                return FALSE;
            }
            has_keyword_acquired_lock = TRUE;

            if (!IS_LOGICAL(EXPV_TYPE(arg))) {
                error("acquired_lock variable should be logical");
                return FALSE;
            }

            if (TYPE_IS_PROTECTED(EXPV_TYPE(arg)) && TYPE_IS_READONLY(EXPV_TYPE(arg))) {
                error("acquired_lock variable is PROTECTED");
                return FALSE;
            }


        } else {
            error("unexpected specifier '%s'", keyword);
            return FALSE;
        }

        EXPV_KWOPT_NAME(arg) = (const char *)strdup(keyword);

        list_put_last(st, arg);
    }
    return TRUE;
}


static int
compile_sync_stat_args(expv st, expr x) {
    return compile_stat_args(st, x, FALSE);
}


static int
compile_lock_stat_args(expv st, expr x) {
    return compile_stat_args(st, x, TRUE);
}


static void
replace_CALL_statement(const char * subroutine_name, expv args)
{
    expr callStaement= list2(
        F_CALL_STATEMENT,
        make_enode(IDENT, (void *)find_symbol(subroutine_name)),
        args);
    compile_CALL_statement(callStaement);
}


static void
compile_SYNCALL_statement(expr x) {
    expv st;

    if (!check_image_control_statement_available()) return;

    st = list0(F2008_SYNCALL_STATEMENT);
    /* Check and compile sync stat args */
    if (!compile_sync_stat_args(st, EXPR_ARG1(x))) return;

    if (XMP_coarray_flag) {
        if (EXPR_ARG1(x) == NULL) {
            replace_CALL_statement("xmpf_sync_all", NULL);
        } else {
            replace_CALL_statement("xmpf_sync_all_stat", EXPR_ARG1(x));
        }
    } else {
        output_statement(st);
    }
}


/*
 *  (F2008_SYNCALL_STATEMENT
 *     expr
 *     (LIST expr*))
 */
static void
compile_SYNCIMAGES_statement(expr x) {
    expv sync_stat;
    expv image_set = NULL;

    if (EXPR_ARG1(x) != NULL) {
        TYPE_DESC tp;
        BASIC_DATA_TYPE bt;

        image_set = compile_expression(EXPR_ARG1(x));
        tp = EXPV_TYPE(image_set);

        if ((IS_ARRAY_TYPE(tp) && TYPE_N_DIM(tp) > 1) ||
            ((bt = get_basic_type(tp)) != TYPE_INT &&
             bt != TYPE_GNUMERIC &&
             bt != TYPE_GNUMERIC_ALL)) {
            error("The first argument of SYNC IMAGES statement must be "
                  "INTEGER (scalar or rank 1)");
            return;
        }
    }

    if (!check_image_control_statement_available()) return;

    sync_stat = list0(LIST);
    /* Check and compile sync stat args */
    if (!compile_sync_stat_args(sync_stat, EXPR_ARG2(x))) return;

    if (XMP_coarray_flag) {
        expr args;
        if (EXPR_ARG1(x) == NULL) {
            /* if NULL, change the argment to '*' for xmpf_sync_images */
            EXPR_ARG1(x) = make_enode(STRING_CONSTANT,  (void *)strdup("*"));
        }
        if (EXPR_HAS_ARG2(x) && EXPR_ARG2(x) != NULL) {
            args = list_cons(EXPR_ARG1(x), EXPR_ARG2(x));
        } else {
            args = list1(LIST, EXPR_ARG1(x));
        }

        replace_CALL_statement("xmpf_sync_images", args);

    } else {
        output_statement(list2(F2008_SYNCIMAGES_STATEMENT, image_set, sync_stat));
    }
}


/*
 *  (F2008_SYNCMEMORY_STATEMENT
 *     (LIST expr*))
 */
static void
compile_SYNCMEMORY_statement(expr x) {
    expv st;

    if (!check_image_control_statement_available()) return;

    st = list0(F2008_SYNCMEMORY_STATEMENT);
    /* Check and compile sync stat args */
    if (!compile_sync_stat_args(st, EXPR_ARG1(x))) return;

    if (XMP_coarray_flag) {
        replace_CALL_statement("xmpf_sync_memory", EXPR_ARG1(x));
    } else {
        output_statement(st);
    }
}

/*
 * Check a type is LOCK_TYPE of the intrinsic module ISO_FORTRAN_ENV
 */
static int
type_is_LOCK_TYPE(TYPE_DESC tp) {
    ID tagname;

    if (!IS_STRUCT_TYPE(tp))
        return FALSE;

    while (TYPE_REF(tp) && IS_STRUCT_TYPE(tp)) {
        tp = TYPE_REF(tp);
    }
    tagname = TYPE_TAGNAME(tp);

    if (tagname != NULL &&
        ID_USEASSOC_INFO(tagname) != NULL &&
        strcmp("lock_type",
               SYM_NAME(ID_USEASSOC_INFO(tagname)->original_name)) == 0 &&
        strcmp("iso_fortran_env",
               SYM_NAME(ID_USEASSOC_INFO(tagname)->module_name)) == 0 &&
        ID_USEASSOC_INFO(tagname)->module->is_intrinsic) {
        return TRUE;
    }

    return FALSE;
}


/*
 *  (F2008_LOCK_STATEMENT
 *     expr
 *     (LIST expr*))
 */
static void
compile_LOCK_statement(expr x) {
    expv lock_variable;
    expv sync_stat_list;

    if (!check_image_control_statement_available()) return;

    lock_variable = compile_expression(EXPR_ARG1(x));
    /* CHECK lock_variable */
    if (!type_is_LOCK_TYPE(EXPV_TYPE(lock_variable))) {
        error("The first argument of lock statement must be LOCK_TYPE");
        return;
    }
    if (TYPE_IS_PROTECTED(EXPV_TYPE(lock_variable)) &&
        TYPE_IS_READONLY(EXPV_TYPE(lock_variable))) {
        error("an argument is PROTECTED");
    }

    sync_stat_list = list0(LIST);
    /* Check and compile lock stat args */
    if (!compile_lock_stat_args(sync_stat_list, EXPR_ARG2(x))) return;

    if (XMP_coarray_flag) {
        expr args;
        if (EXPR_HAS_ARG2(x) && EXPR_ARG2(x) != NULL) {
            args = list_cons(EXPR_ARG1(x), EXPR_ARG2(x));
        } else {
            args = list1(LIST, EXPR_ARG1(x));
        }

        replace_CALL_statement("xmpf_lock", args);

    } else {
        output_statement(
            list2(F2008_LOCK_STATEMENT, lock_variable, sync_stat_list));
    }
}


/*
 *  (F2008_UNLOCK_STATEMENT
 *     expr
 *     (LIST expr*))
 */
static void
compile_UNLOCK_statement(expr x) {
    expv lock_variable;
    expv sync_stat_list;

    if (!check_image_control_statement_available()) return;

    lock_variable = compile_expression(EXPR_ARG1(x));
    /* CHECK lock_variable */
    if (!type_is_LOCK_TYPE(EXPV_TYPE(lock_variable))) {
        error("The first argument of unlock statement must be LOCK_TYPE");
        return;
    }
    if (TYPE_IS_PROTECTED(EXPV_TYPE(lock_variable)) &&
        TYPE_IS_READONLY(EXPV_TYPE(lock_variable))) {
        error("an argument is PROTECTED");
    }

    sync_stat_list = list0(LIST);
    /* Check and compile sync stat args */
    if (!compile_sync_stat_args(sync_stat_list, EXPR_ARG2(x))) return;

    if (XMP_coarray_flag) {
        expr args;
        if (EXPR_HAS_ARG2(x) && EXPR_ARG2(x) != NULL) {
            args = list_cons(EXPR_ARG1(x), EXPR_ARG2(x));
        } else {
            args = list1(LIST, EXPR_ARG1(x));
        }

        replace_CALL_statement("xmpf_unlock", args);

    } else {
        output_statement(
            list2(F2008_UNLOCK_STATEMENT, lock_variable, sync_stat_list));
    }
}


/*
 *  (F2008_CRITICAL_STATEMENT expr)
 */
static void
compile_CRITICAL_statement(expr x) {
    expv st;

    if (!check_image_control_statement_available()) return;

    push_ctl(CTL_CRITICAL);

    st = list2(F2008_CRITICAL_STATEMENT, NULL, NULL);
    output_statement(st);
    CTL_BLOCK(ctl_top) = CURRENT_STATEMENTS;
    CTL_CRIT_STATEMENT(ctl_top) = st;

    /* save construct name */
    if (EXPR_HAS_ARG1(x)) {
        CTL_CRIT_CONST_NAME(ctl_top) = EXPR_ARG1(x);
    }

    CURRENT_STATEMENTS = NULL;

    if (XMP_coarray_flag) {
        replace_CALL_statement("xmpf_critical", NULL);
        /* No need to return. */
    }

    if (endlineno_flag){
        if (current_line->end_ln_no) {
            EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->end_ln_no;
        } else {
            EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
        }
    }
}


static int
check_valid_construction_name(expr x, expr y)
{
    if (x != NULL && y == NULL) {
        error("expect construnct name");
        return FALSE;
    } else if (x == NULL && y != NULL) {
        error("unexpected construnct name");
        return FALSE;
    } else if (x != NULL && y != NULL) {
        if (EXPR_SYM(x) != EXPR_SYM(y)) {
            error("unmatched construct name");
            return FALSE;
        }
    }
    return TRUE;
}


/*
 *  (F2008_ENDCRITICAL_STATEMENT expr)
 */
static void
compile_ENDCRITICAL_statement(expr x) {
    if (CTL_TYPE(ctl_top) != CTL_CRITICAL) {
        error("'endcritical', out of place");
        return;
    }

    /* check construct name */
    if (!check_valid_construction_name(CTL_CRIT_CONST_NAME(ctl_top), EXPR_ARG1(x)))
        return;

    if (XMP_coarray_flag) {
        replace_CALL_statement("xmpf_end_critical", NULL);
    }

    CTL_CRIT_BODY(ctl_top) = CURRENT_STATEMENTS;

    if (endlineno_flag) {
        EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
    }

    pop_ctl();

}


/*
 * Check if the statemenet exists inside CRITICAL construct
 */
static int
check_inside_CRITICAL_construct() {
    CTL cp;
    FOR_CTLS_BACKWARD(cp) {
        if (CTL_TYPE(cp) == CTL_CRITICAL) {
            return TRUE;
        }
    }
    return FALSE;
}

/*
 * Check if image control statement can exist
 */
static int
check_image_control_statement_available() {
    if (check_inside_CRITICAL_construct()) {
        error("Image control statement in CRITICAL block");
        return FALSE;
    }

    return TRUE;
}

/*
 * IMPORT statement
 */
static void
compile_IMPORT_statement(expr x)
{
    if(check_inside_INTERFACE_body() == FALSE){
        error("IMPORT statement allowed only in interface body");
    }
    expv ident_list, arg;
    list lp;
    ident_list = EXPR_ARG1(x);
    if(EXPR_LIST(ident_list)) {
        FOR_ITEMS_IN_LIST(lp, ident_list) {
            arg = LIST_ITEM(lp);
            ID ident = find_ident(EXPR_SYM(arg));
            if(ident == NULL){
                error("%s part of the IMPORT statement has not been declared yet.", SYM_NAME(EXPR_SYM(arg)));
            }
        }
    }
    output_statement(list1(F03_IMPORT_STATEMENT, EXPR_ARG1(x)));
}

static void
compile_BLOCK_statement(expr x)
{
    expv st;

    push_ctl(CTL_BLOCK);
    push_env(CTL_BLOCK_LOCAL_ENV(ctl_top));

    st = list2(F2008_BLOCK_STATEMENT, NULL, NULL);
    output_statement(st);
    CTL_BLOCK(ctl_top) = CURRENT_STATEMENTS;
    CTL_BLOCK_STATEMENT(ctl_top) = st;
    EXPR_BLOCK(CTL_BLOCK_STATEMENT(ctl_top)) = NULL;

    /* save construct name */
    if (EXPR_HAS_ARG1(x)) {
        CTL_BLOCK_CONST_NAME(ctl_top) = EXPR_ARG1(x);
    }

    CURRENT_STATE = INDCL;
    CURRENT_STATEMENTS = NULL;
    current_proc_state = P_DEFAULT;

    if (endlineno_flag){
        if (current_line->end_ln_no) {
            EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->end_ln_no;
        } else {
            EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
        }
    }
}

static void
move_implicit_variables_to_parent()
{
    ID ip;
    ID iq;
    ID replaced = NULL;
    ID moved = NULL;
    ID replaced_last = NULL;
    ID moved_last;
    ID last;
    ID parent = PARENT_LOCAL_SYMBOLS;

    SAFE_FOREACH_ID(ip, iq, LOCAL_SYMBOLS) {
        if (ID_TYPE(ip) == NULL || TYPE_IS_IMPLICIT(ID_TYPE(ip))) {
            ID_LINK_ADD(ip, moved, moved_last);
        } else {
            ID_LINK_ADD(ip, replaced, replaced_last);
        }
    }

    LOCAL_SYMBOLS = replaced;
    FOREACH_ID(ip, parent) {
        last = ip;
    }
    SAFE_FOREACH_ID(ip, iq, moved) {
        ID_LINK_ADD(ip, parent, last);
    }
}


static void
compile_ENDBLOCK_statement(expr x)
{
    BLOCK_ENV current_block;
    BLOCK_ENV bp, tail;

    if (CTL_TYPE(ctl_top) != CTL_BLOCK) {
        error("'endblock', out of place");
        return;
    }

    /* check construct name */
    if (CTL_BLOCK_CONST_NAME(ctl_top) != NULL) {
        if (!EXPR_HAS_ARG1(x) || EXPR_ARG1(x) == NULL) {
            error("expects construnct name");
            return;
        } else if (EXPR_SYM(CTL_BLOCK_CONST_NAME(ctl_top)) !=
                   EXPR_SYM(EXPR_ARG1(x))) {
            error("unmatched construct name");
            return;
        }
    } else if (EXPR_HAS_ARG1(x) && EXPR_ARG1(x) != NULL) {
        error("unexpected construnct name");
        return;
    }

    if (debug_flag) {
        fprintf(debug_fp,"\n*** IN BLOCK:\n");
        print_IDs(LOCAL_SYMBOLS, debug_fp, TRUE);
        print_types(LOCAL_STRUCT_DECLS, debug_fp);
        expv_output(CURRENT_STATEMENTS, debug_fp);
    }

    CTL_BLOCK_BODY(ctl_top) = CURRENT_STATEMENTS;

    if (endlineno_flag) {
        EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
    }

    move_implicit_variables_to_parent();

    current_block = XMALLOC(BLOCK_ENV, sizeof(*current_block));
    BLOCK_LOCAL_SYMBOLS(current_block) = LOCAL_SYMBOLS;
    BLOCK_LOCAL_LABELS(current_block) = LOCAL_LABELS;
    BLOCK_LOCAL_INTERFACES(current_block) = LOCAL_INTERFACES;
    BLOCK_LOCAL_EXTERNAL_SYMBOLS(current_block) = LOCAL_EXTERNAL_SYMBOLS;
    BLOCK_CHILDREN(current_block) = LOCAL_BLOCKS;
    EXPR_BLOCK(CTL_BLOCK_STATEMENT(ctl_top)) = current_block;

    end_procedure();
    pop_ctl();
    pop_env();

    FOREACH_BLOCKS(bp, LOCAL_BLOCKS) {
        tail = bp;
    }
    BLOCK_LINK_ADD(current_block, LOCAL_BLOCKS, tail);

    CURRENT_STATE = INEXEC;
}


void
compile_VALUE_statement(expr x)
{
    list lp;
    expr ident;
    ID id;

    assert(EXPR_CODE(x) == F03_VALUE_STATEMENT);

    FOR_ITEMS_IN_LIST(lp, x) {
        ident = LIST_ITEM(lp);

        assert(EXPR_CODE(ident) == IDENT);

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
            error("can't change attributes of USE-associated symbol '%s'",
                  ID_NAME(id));
            return;
        } else if (ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            return;
        }

        TYPE_SET_VALUE(id);
        if (ID_TYPE(id)) {
            TYPE_SET_VALUE(ID_TYPE(id));
        }
    }
}


/*
 * x is (LIST
 *       (LIST triplet ...)
 *       mask
 *       type)
 */
static expv
compile_forall_header(expr x)
{
    expr type;
    expr triplets;
    expr mask;
    expv vmask = NULL;
    expv init;
    expv forall_header;
    list lp;
    TYPE_DESC tp;

    triplets       = EXPR_ARG1(x);
    mask           = EXPR_ARG2(x);
    type           = EXPR_ARG3(x);

    if (type) {
        tp = compile_type(type, /*allow_predecl=*/ FALSE);
    } else {
        tp = new_type_desc();
        TYPE_BASIC_TYPE(tp) = TYPE_INT;
    }
    CURRENT_STATE = INEXEC;

    init = list0(LIST);
    FOR_ITEMS_IN_LIST(lp, triplets) {
        ID id;
        SYMBOL sym;
        SYMBOL new_sym = NULL;
        expr x1, x2, x3;
        expv low_limit;
        expv top_limit;
        expv step;

        assert(EXPR_CODE(LIST_ITEM(lp)) == F95_TRIPLET_EXPR);

        sym = EXPR_SYM(EXPR_ARG1(LIST_ITEM(lp)));
        x1 = EXPR_ARG1(EXPR_ARG2(LIST_ITEM(lp)));
        x2 = EXPR_ARG2(EXPR_ARG2(LIST_ITEM(lp)));
        x3 = EXPR_ARG3(EXPR_ARG2(LIST_ITEM(lp)));

        if (find_ident_local(sym) != NULL) {
            error("duplicate index");
            return NULL;
        }

        if (type) {
            id = declare_ident(sym, CL_VAR);
            ID_STORAGE(id) = STG_INDEX;
        } else {
            TYPE_DESC tp;
            id = find_ident(sym);
            if (id) {
                tp = ID_TYPE(id);
                if (tp == NULL) {
                    error("%s is not declared", SYM_NAME(sym));
                } else if (!IS_INT(tp)) {
                    error("%s is not integer", SYM_NAME(sym));
                }
                id = declare_ident(sym, CL_VAR);

            } else {
                id = declare_ident(sym, CL_VAR);
                implicit_declaration(id);

            }

            ID_STORAGE(id) = STG_AUTO;
        }
        declare_id_type(id, tp);
        declare_variable(id);

        for (;;) {
            new_sym = gen_temp_symbol("omnitmp");
            if (find_ident(new_sym) == NULL) {
                break;
            }
        }
        /*
         * Renaming trick:
         *
         * Replace the name of the index here.
         *
         * When compile_expression() is applied to this identifier,
         * new_sym will be used.
         *
         * The index's own name will be replaced in
         * compile_ENDFORALL_statement().
         *
         */
        EXPV_NAME(ID_ADDR(id)) = new_sym;

        ID_LINE(id) = EXPR_LINE(x);

        low_limit = compile_expression(x1);
        top_limit = compile_expression(x2);
        step      = compile_expression(x3);

        if (low_limit == NULL || (
                !IS_INT(EXPV_TYPE(low_limit)) &&
                !IS_GNUMERIC(EXPV_TYPE(low_limit)) &&
                !IS_GNUMERIC_ALL(EXPV_TYPE(low_limit)))) {
            error("invalid expression");
        }
        if (top_limit == NULL || (
                !IS_INT(EXPV_TYPE(top_limit)) &&
                !IS_GNUMERIC(EXPV_TYPE(top_limit)) &&
                !IS_GNUMERIC_ALL(EXPV_TYPE(top_limit)))) {
            error("invalid expression");
        }
        if (step != NULL && (
                !IS_INT(EXPV_TYPE(step)) &&
                !IS_GNUMERIC(EXPV_TYPE(step)) &&
                !IS_GNUMERIC_ALL(EXPV_TYPE(step)))) {
            error("invalid expression");
        }

        init = list_put_last(init, list2(F_SET_EXPR,
                                         expv_sym_term(F_VAR, tp, sym),
                                         list3(F_INDEX_RANGE,
                                               low_limit, top_limit, step)));
    }

    if (mask) {
        vmask = compile_expression(mask);
        if (!IS_LOGICAL(EXPV_TYPE(vmask)) &&
            !IS_GNUMERIC(EXPV_TYPE(vmask)) &&
            !IS_GNUMERIC_ALL(EXPV_TYPE(vmask))) {
            error("invalid expression");
        }
    }

    forall_header = list2(LIST, init, vmask);

    return forall_header;
}


/*
 * (F_FORALL_STATEMENT
 *   (LIST
 *     (LIST triplet ...)
 *     mask
 *     type)
 *   assignment
 *   construct_name)
 */
static void
compile_FORALL_statement(int st_no, expr x)
{
    expr st;
    expv forall_header;

    /*
     * Insert a block construct.
     *
     * compile_FORALL_statement will rename the index variabls,
     * so it may be good to confine these index variables with the BLOCK construct.
     *
     * ex)
     *
     *    FORALL(I = 1:3); ...; ENDFORALL
     *
     *  will be transrated into
     *
     *    BLOCK
     *      INTEGER :: omnitmp001
     *      FORALL(omnitmp001 = 1:3); ...; ENDFORALL
     *    END BLOCK
     *
     */
    if (CTL_TYPE(ctl_top) != CTL_FORALL) {
        compile_BLOCK_statement(list0(F2008_BLOCK_STATEMENT));
    }

    push_ctl(CTL_FORALL);
    push_env(CTL_FORALL_LOCAL_ENV(ctl_top));

    assert(LOCAL_SYMBOLS == NULL);

    st = list3(F_FORALL_STATEMENT, NULL, NULL, NULL);

    if ((forall_header = compile_forall_header(EXPR_ARG1(x))) == NULL) {
        return;
    }

    CTL_BLOCK(ctl_top) = st;
    // CTL_FORALL_STATEMENT(ctl_top) = st;
    CTL_FORALL_HEADER(ctl_top) = forall_header;

    /* save construct name */
    if (EXPR_HAS_ARG3(x)) {
        CTL_FORALL_CONST_NAME(ctl_top) = EXPR_ARG3(x);
    }

    CURRENT_STATEMENTS = NULL;
    current_proc_state = P_DEFAULT;

    /*
     * If FORALL has forall-assign-statment,
     * compile this FORALL as FORALL-statement
     *
     * ex)
     *
     * FORALL(...) A(I) = B(I)
     *
     */
    if (EXPR_ARG2(x)) {
        expv forall_assign;
        compile_statement(st_no, EXPR_ARG2(x));
        forall_assign = LIST_ITEM(EXPV_LIST(CURRENT_STATEMENTS));
        if (EXPR_CODE(forall_assign) != F_LET_STATEMENT &&
            EXPR_CODE(forall_assign) != F95_POINTER_SET_STATEMENT) {
            error_at_node(forall_assign,
                          "not allowed statement in the FORALL statement");
        }
        compile_ENDFORALL_statement(NULL);
    }
}


static void
compile_end_forall_header(expv init)
{
    ID ip;
    list lp;
    ENV parent;

    FOR_ITEMS_IN_LIST(lp, init) {
        ip = find_ident_head(EXPR_SYM(EXPR_ARG1(LIST_ITEM(lp))), LOCAL_SYMBOLS);
        if (ip) {
            debug("#### rename %s to %s",
                  SYM_NAME(ID_SYM(ip)),
                  SYM_NAME(EXPV_NAME(ID_ADDR(ip))));
            /*
             * Rename symbol names those are generated in compile_forall_header()
             */
            ID_SYM(ip) = EXPV_NAME(ID_ADDR(ip));
            EXPR_SYM(EXPR_ARG1(LIST_ITEM(lp))) = EXPV_NAME(ID_ADDR(ip));
        }
    }

    parent = ENV_PARENT(current_local_env);

    ENV_SYMBOLS(parent) = unify_id_list(
        ENV_SYMBOLS(parent),
        ENV_SYMBOLS(current_local_env),
        /*overshadow=*/FALSE);

    ENV_EXTERNAL_SYMBOLS(parent) = unify_ext_id_list(
        ENV_EXTERNAL_SYMBOLS(parent),
        ENV_EXTERNAL_SYMBOLS(current_local_env),
        /*overshadow=*/FALSE);
}


static void
compile_ENDFORALL_statement(expr x)
{
    list lp;
    expv init;

    if (CTL_TYPE(ctl_top) != CTL_FORALL) {
        error("'endforall', out of place");
        return;
    }

    /* check construct name */
    if (CTL_FORALL_CONST_NAME(ctl_top) != NULL) {
        if (x == NULL || !EXPR_HAS_ARG1(x) || EXPR_ARG1(x) == NULL) {
            error("expects construnct name");
            return;
        } else if (EXPR_SYM(CTL_FORALL_CONST_NAME(ctl_top)) !=
                   EXPR_SYM(EXPR_ARG1(x))) {
            error("unmatched construct name");
            return;
        }
    } else if (x != NULL && EXPR_HAS_ARG1(x) && EXPR_ARG1(x) != NULL) {
        error("unexpected construnct name");
        return;
    }

    if (debug_flag) {
        fprintf(debug_fp,"\n*** IN FORALL:\n");
        print_IDs(LOCAL_SYMBOLS, debug_fp, TRUE);
        print_types(LOCAL_STRUCT_DECLS, debug_fp);
        expv_output(CURRENT_STATEMENTS, debug_fp);
    }

    CTL_FORALL_BODY(ctl_top) = CURRENT_STATEMENTS;

    FOR_ITEMS_IN_LIST(lp, CTL_FORALL_BODY(ctl_top)) {
        switch (EXPV_CODE(LIST_ITEM(lp))) {
            case F_FORALL_STATEMENT:
            case F_WHERE_STATEMENT:
            case F_LET_STATEMENT:
            case F95_POINTER_SET_STATEMENT:
                continue;
                break;
            default:
                error_at_node(LIST_ITEM(lp),
                              "not allowed statement in the FORALL construct");
                break;
        }
    }

    if (endlineno_flag) {
        EXPR_END_LINE_NO(CTL_BLOCK(ctl_top)) = current_line->ln_no;
    }

    init = CTL_FORALL_INIT(ctl_top);

    compile_end_forall_header(init);

    /* no declarations in the forall construct */
    assert(ENV_STRUCT_DECLS(current_local_env) == NULL);
    assert(ENV_COMMON_SYMBOLS(current_local_env) == NULL);
    assert(ENV_INTERFACES(current_local_env) == NULL);

    /* no use statements in the forall construct */
    assert(ENV_USE_DECLS(current_local_env) == NULL);

    /* BLOCK cannot exists in FORALL construct */
    assert(LOCAL_BLOCKS == NULL);

    pop_ctl();
    pop_env();
    CURRENT_STATE = INEXEC;

    /*
     * Close the block construct which is genereted in compile_FORALL_statement().
     */
    if (CTL_TYPE(ctl_top) == CTL_BLOCK) {
        compile_ENDBLOCK_statement(list0(F2008_ENDBLOCK_STATEMENT));
    }
}

/*
 * Move implict declared identifiers in a TYPE GUARD clause to parent's LOCAL_SYMBOLS
 *
 * ex)
 *  1  SELECT TYPE(p)
 *  2    TYPE IS (INTEGER)
 *  3      a = 1             ! a is declared implicitly
 *  4    TYPE IS (REAL
 *  5      a = 2             ! a is the same one with 'a' in line 3
 *  6  END SELECT TYPE
 *
 *  'a' in line 3 is declared inside the environment of CTL_TYPE_GUARD,
 *  but it should be moved to the parent environment.
 */
static void
move_vars_to_parent_from_type_guard()
{
    ID ip, iq, last = NULL;
    ENV parent = ENV_PARENT(current_local_env);

    if (parent == NULL) {
        return;
    }
    if (LOCAL_SYMBOLS == NULL) {
        return;
    }

    FOREACH_ID(ip, ENV_SYMBOLS(parent)) {
        last = ip;
    }

    SAFE_FOREACH_ID(ip, iq, LOCAL_SYMBOLS) {
        if (ip == LOCAL_SYMBOLS) {
            continue;
        }
        ID_NEXT(ip) = NULL;
        ID_LINK_ADD(ip, ENV_SYMBOLS(parent), last);
    }
    if (LOCAL_SYMBOLS)
        ID_NEXT(LOCAL_SYMBOLS) = NULL;
}


/*
 * Checks types of each type guard statements under SELECT TYPE construct
 */
static void
check_select_types(expr x, TYPE_DESC tp)
{
    list lp;

    if (CTL_TYPE(ctl_top) != CTL_TYPE_GUARD) {
        return;
    }

    FOR_ITEMS_IN_LIST(lp, CTL_SAVE(ctl_top)) {
        expv statement;
        TYPE_DESC tq;

        statement = LIST_ITEM(lp);

        if (EXPR_CODE(statement) != F03_TYPEIS_STATEMENT &&
            EXPR_CODE(statement) != F03_CLASSIS_STATEMENT) {
            continue;
        }

        tq = EXPR_ARG1(statement)?EXPV_TYPE(EXPR_ARG1(statement)):NULL;

        if (tp == NULL && tq == NULL) {
            error_at_node(x, "duplicate CLASS DEFAULT");
            return;
        }

        if (tp == NULL || tq == NULL) {
            continue;
        }

        if (EXPR_CODE(x) == EXPR_CODE(statement)) {
            if (IS_STRUCT_TYPE(tp) && IS_STRUCT_TYPE(tq)) {
                TYPE_DESC btp, btq;
                btp = get_bottom_ref_type(tp);
                btq = get_bottom_ref_type(tq);
                if (type_is_strict_compatible(tp, tq, TRUE) 
                    && TYPE_TAGNAME(btp) == TYPE_TAGNAME(btq)) 
                {
                    error_at_node(x, "duplicate derived-types in SELECT TYPE construct");
                }
            } else if (type_is_strict_compatible(tp, tq, TRUE)) {
                error_at_node(x, "duplicate types in SELECT TYPE construct");
                return;
            }
        }
    }
}

static void
compile_DOCONCURRENT_statement(expr range_st_no,
                               expr forall_header,
                               expr construct_name)
{
    expv vforall_header = NULL;
    int do_stmt_num = -1;
    ID do_label = NULL;

    if (CTL_TYPE(ctl_top) != CTL_DO) {
        compile_BLOCK_statement(list0(F2008_BLOCK_STATEMENT));
    }

    if (range_st_no != NULL) {
        expv stmt_label = expr_label_value(range_st_no);
        if (stmt_label == NULL) {
            error("illegal label in DO CONCURRENT");
            return;
        }
        do_stmt_num = EXPV_INT_VALUE(stmt_label);
    }

    if (do_stmt_num > 0) {
        do_label = declare_label(do_stmt_num, LAB_EXEC, FALSE);
        if (do_label == NULL) return;
        if (LAB_IS_DEFINED(do_label)) {
            error("no backward DO loops");
            return;
        }
        /* turn off, becuase this is not branch */
        LAB_IS_USED(do_label) = FALSE;
    }

    push_ctl(CTL_DO);
    push_env(CTL_DO_LOCAL_ENV(ctl_top));

    if ((vforall_header = compile_forall_header(forall_header)) == NULL) {
        return;
    }

    CTL_DO_VAR(ctl_top) = NULL;
    CTL_DO_LABEL(ctl_top) = do_label;

    CTL_BLOCK(ctl_top) = list3(F08_DOCONCURRENT_STATEMENT,
                               vforall_header, NULL, construct_name);
}

void
compile_CONTIGUOUS_statement(expr x)
{
    list lp;
    expr ident;
    ID id;

    assert(EXPR_CODE(x) == F08_CONTIGUOUS_STATEMENT);

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        ident = LIST_ITEM(lp);

        assert(EXPR_CODE(ident) == IDENT);

        id = declare_ident(EXPR_SYM(ident), CL_VAR);
        if(id == NULL)
            return;
        if(ID_IS_OFMODULE(id)) {
            error("can't change attributes of USE-associated symbol '%s'", ID_NAME(id));
            return;
        } else if (ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            return;
        }

        TYPE_SET_CONTIGUOUS(id);
    }
}
