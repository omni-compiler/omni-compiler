/**
 * \file F95-parser.y
 */
/* F95 parser */
%token EOS              /* end of statement */
%token CONSTANT         /* any constant */
%token IDENTIFIER       /* name */
%token GENERIC_SPEC     /* operator ( defined operator ) or assignment(=) */
%token UNKNOWN
%token STATEMENT_LABEL_NO
%token TRUE_CONSTANT
%token FALSE_CONSTANT
%token KW_INTEGER
%token KW_REAL
%token KW_COMPLEX
%token KW_DOUBLE
%token KW_DCOMPLEX
%token KW_LOGICAL
%token KW_CHARACTER
%token KW_UNDEFINED
%token KW_NONE
/* %token KW_STATIC */

/* keyword */
%token PARAMETER
/* %token PUNCH */
%token INCLUDE
%token LET              /* dummy */
%token ARITHIF
%token LOGIF
%token IFTHEN
%token ASSIGN
%token ASSIGNMENT
%token BLOCKDATA
%token CALL
%token CLOSE
%token COMMON
%token CONTINUE
%token DATA
%token DIMENSION
%token CODIMENSION
%token DO
%token ENDDO
%token DOWHILE
%token WHILE
%token ELSE
%token ELSEIF
%token ELSEIFTHEN
%token END
%token ENDFILE
%token ENDFILE_P
%token ENDIF
%token ENTRY
%token EQUIV
%token EXTERNAL
%token FORMAT
%token FUNCTION
%token GOTO
/* %token ASGOTO */
/* %token COMPGOTO */
%token IMPLICIT
%token IMPLICIT_NONE /* special for implicit none */

 /* handlign of implicit like integer(a-z).  */
%token SET_LEN /* ( len = */
%token SET_KIND /* ( kind = */

%token INTRINSIC
%token NAMELIST
%token PAUSE
%token PRINT
%token PROGRAM
%token READ
%token READ_P
%token RETURN
%token SAVE
%token STOP
%token SUBROUTINE
%token THEN
%token P_THEN /* ) then */
%token KW_TO
%token WRITE
%token WRITE_P
%token OPEN
%token INQUIRE
%token BACKSPACE
%token BACKSPACE_P
%token REWIND
%token REWIND_P
%token POINTER

/* F95 keywords */
%token ENDPROGRAM
%token MODULE
%token ENDMODULE
%token INTERFACE
%token INTERFACEASSIGNMENT
%token INTERFACEOPERATOR
%token ENDINTERFACE
%token PROCEDURE
%token MODULEPROCEDURE
%token PRIVATE
%token SEQUENCE
%token RESULT
%token RECURSIVE
%token PURE
%token ELEMENTAL
%token CONTAINS
%token KW_TYPE
%token ENDTYPE
%token ALLOCATABLE
%token INTENT
%token EXIT
%token CYCLE
%token PUBLIC
%token OPTIONAL
%token TARGET
%token WHERE
%token ELSEWHERE
%token ENDWHERE
%token FORALL
%token ENDFORALL
%token ENDFUNCTION
%token ENDSUBROUTINE
%token ENDBLOCKDATA
%token SELECT   /* select case */
%token CASEDEFAULT /* case defualt */
%token CASE     /* case */
%token ENDSELECT
%token KW_DEFAULT
%token KW_WHILE
%token KW_USE
%token KW_ONLY
%token ALLOCATE
%token DEALLOCATE
%token NULLIFY
%token KW_STAT

/* F03 keywords */
%token PROTECTED

/* Coarray keywords #060 */
%token SYNCALL
%token SYNCIMAGES
%token SYNCMEMORY
%token LOCK
%token UNLOCK
%token CRITICAL
%token ENDCRITICAL
%token ERRORSTOP
%token KW_SYNC
%token KW_ALL
%token KW_IMAGES
%token KW_MEMORY
%token KW_ERROR


%token REF_OP

%token L_ARRAY_CONSTRUCTOR /* /( */
%token R_ARRAY_CONSTRUCTOR /* (/ */

%token KW_IN
%token KW_OUT
%token KW_INOUT

%token KW_LEN
%token KW_KIND

%token KW_DBL
%token KW_SELECT
%token KW_BLOCK
%token KW_ENDBLOCK
%token KW_GO
%token KW_PRECISION
%token OPERATOR

%token COL2     /* :: */

%token POWER    /* ** */
%token CONCAT   /* // */
%token AND      /* .and. */
%token OR       /* .or. */
%token NEQV     /* .neqv. */
%token EQV      /* .eqv. */
%token NOT      /* .not. */
%token EQ       /* .eq. */
%token LT       /* .lt. */
%token GT       /* .gt. */
%token LE       /* .le. */
%token GE       /* .ge. */
%token NE       /* .ne. */
%token USER_DEFINED_OP /* .USER_DEFINED. */

/* Specify precedences and associativities. */
%left ','
%nonassoc ':'
%right '='
%left USER_DEFINED_OP
%left EQV NEQV
%left OR
%left AND
%left NOT
%nonassoc LT GT LE GE EQ NE
%left CONCAT REF_OP
%left '+' '-'
%left '*' '/'
%right POWER
%left '%'


%token PRAGMA_SLINE /* do not parse omp token.  */
%token PRAGMA_HEAD /*  pragma leading char like !$ etc.  */

/* OpenMP directives */
%token OMPKW_LINE
%token OMPKW_PARALLEL
%token OMPKW_TASK
%token OMPKW_END
%token OMPKW_PRIVATE
%token OMPKW_SHARED
%token OMPKW_DEFAULT
%token OMPKW_NONE
%token OMPKW_FIRSTPRIVATE
%token OMPKW_REDUCTION
%token OMPKW_IF
%token OMPKW_FINAL
%token OMPKW_UNTIED
%token OMPKW_MERGEABLE
%token OMPKW_DEPEND
%token OMPKW_DEPEND_IN
%token OMPKW_DEPEND_OUT
%token OMPKW_DEPEND_INOUT
%token OMPKW_SAFELEN
%token OMPKW_SIMDLEN
%token OMPKW_LINEAR
%token OMPKW_ALIGNED
%token OMPKW_NUM_THREADS
%token OMPKW_COPYIN
%token OMPKW_DO
%token OMPKW_SIMD
%token OMPKW_DECLARE
%token OMPKW_LASTPRIVATE
%token OMPKW_SCHEDULE
%token OMPKW_STATIC
%token OMPKW_DYNAMIC
%token OMPKW_GUIDED
%token OMPKW_ORDERED
%token OMPKW_RUNTIME
%token OMPKW_AFFINITY
%token OMPKW_SECTIONS
%token OMPKW_SECTION
%token OMPKW_NOWAIT
%token OMPKW_SINGLE
%token OMPKW_MASTER
%token OMPKW_CRITICAL
%token OMPKW_BARRIER
%token OMPKW_ATOMIC
%token OMPKW_FLUSH
%token OMPKW_THREADPRIVATE
%token OMPKW_WORKSHARE
%token OMPKW_COPYPRIVATE

%type <val> omp_directive omp_nowait_option omp_end_clause_option omp_end_clause_list omp_end_clause omp_clause_option omp_clause_list omp_clause omp_list /*omp_common_list*/ omp_default_attr omp_copyin_list omp_schedule_arg
%type <code> omp_schedule_attr omp_reduction_op omp_depend_op

/* XcalableMP directive */
%token XMPKW_LINE

%token XMPKW_END
%token XMPKW_NODES
%token XMPKW_TEMPLATE
%token XMPKW_TEMPLATE_FIX
%token XMPKW_DISTRIBUTE
%token XMPKW_ALIGN
%token XMPKW_SHADOW
%token XMPKW_TASK
%token XMPKW_TASKS
%token XMPKW_LOOP
%token XMPKW_REFLECT
%token XMPKW_GMOVE
%token XMPKW_BARRIER
%token XMPKW_REDUCTION
%token XMPKW_BCAST
%token XMPKW_WAIT_ASYNC
%token XMPKW_COARRAY
%token XMPKW_IMAGE
%token XMPKW_WAIT
%token XMPKW_POST
%token XMPKW_CRITICAL
%token XMPKW_ARRAY
%token XMPKW_LOCAL_ALIAS
%token XMPKW_SAVE_DESC

%token XMPKW_ON
%token XMPKW_ONTO
%token XMPKW_WITH
%token XMPKW_FROM

%token XMPKW_WIDTH
%token XMPKW_PERIODIC

%token XMPKW_ASYNC
%token XMPKW_NOWAIT
%token XMPKW_MASTER /* not used */
%token XMPKW_NOCOMM

%token XMPKW_IN
%token XMPKW_OUT

%token XMPKW_BEGIN
%token XMPKW_MASTER_IO
%token XMPKW_GLOBAL_IO

%token XMPKW_ATOMIC
%token XMPKW_DIRECT

%token XMPKW_ACC

%type <val> xmp_directive xmp_nodes_clause xmp_template_clause xmp_distribute_clause xmp_align_clause xmp_shadow_clause xmp_template_fix_clause xmp_task_clause xmp_loop_clause xmp_reflect_clause xmp_gmove_clause xmp_barrier_clause xmp_bcast_clause xmp_reduction_clause xmp_array_clause xmp_save_desc_clause xmp_wait_async_clause xmp_end_clause

 //%type <val> xmp_subscript_list xmp_subscript xmp_dist_fmt_list xmp_dist_fmt xmp_obj_ref xmp_reduction_opt xmp_reduction_opt1 xmp_reduction_spec xmp_reduction_var_list xmp_reduction_var xmp_pos_var_list xmp_gmove_opt xmp_expr_list xmp_name_list xmp_clause_opt xmp_clause_list xmp_clause_one xmp_master_io_options xmp_global_io_options xmp_width_opt xmp_width_opt1 xmp_async_opt xmp_async_opt1 xmp_width_list xmp_width
 //%type <val> xmp_subscript_list xmp_subscript xmp_dist_fmt_list xmp_dist_fmt xmp_obj_ref xmp_reduction_opt xmp_reduction_opt1 xmp_reduction_spec xmp_reduction_var_list xmp_reduction_var xmp_pos_var_list xmp_gmove_opt xmp_nocomm_opt xmp_expr_list xmp_name_list xmp_clause_opt xmp_clause_list xmp_clause_one xmp_master_io_options xmp_global_io_options xmp_async_opt xmp_width_list xmp_width
%type <val> xmp_subscript_list xmp_subscript xmp_dist_fmt_list xmp_dist_fmt xmp_obj_ref xmp_reduction_opt xmp_reduction_opt1 xmp_reduction_spec xmp_reduction_var_list xmp_reduction_var xmp_pos_var_list xmp_nocomm_opt xmp_expr_list xmp_name_list xmp_clause_opt xmp_clause_list xmp_clause_one xmp_master_io_options xmp_global_io_options xmp_async_opt xmp_width_list xmp_width xmp_coarray_clause xmp_image_clause xmp_acc_opt

%type <code> xmp_reduction_op

/* OpenACC directives */
%token ACCKW_LINE
%token ACCKW_END
%token ACCKW_PARALLEL
%token ACCKW_DATA
%token ACCKW_LOOP
%token ACCKW_KERNELS
%token ACCKW_ATOMIC
%token ACCKW_WAIT
%token ACCKW_CACHE
%token ACCKW_ROUTINE
%token ACCKW_ENTER
%token ACCKW_EXIT
%token ACCKW_HOST_DATA
%token ACCKW_DECLARE
%token ACCKW_INIT
%token ACCKW_SHUTDOWN
%token ACCKW_SET

/* OpenACC clauses */
%token ACCKW_IF
%token ACCKW_ASYNC
%token ACCKW_DEVICE_TYPE
%token ACCKW_COPY
%token ACCKW_COPYIN
%token ACCKW_COPYOUT
%token ACCKW_CREATE
%token ACCKW_PRESENT
%token ACCKW_PRESENT_OR_COPY
%token ACCKW_PRESENT_OR_COPYIN
%token ACCKW_PRESENT_OR_COPYOUT
%token ACCKW_PRESENT_OR_CREATE
%token ACCKW_DEVICEPTR
%token ACCKW_NUM_GANGS
%token ACCKW_NUM_WORKERS
%token ACCKW_VECTOR_LENGTH
%token ACCKW_REDUCTION
%token ACCKW_PRIVATE
%token ACCKW_FIRSTPRIVATE
%token ACCKW_DEFAULT
%token ACCKW_NONE
%token ACCKW_COLLAPSE
%token ACCKW_GANG
%token ACCKW_WORKER
%token ACCKW_VECTOR
%token ACCKW_SEQ
%token ACCKW_AUTO
%token ACCKW_TILE
%token ACCKW_INDEPENDENT
%token ACCKW_BIND
%token ACCKW_NOHOST
%token ACCKW_READ
%token ACCKW_WRITE
%token ACCKW_UPDATE
%token ACCKW_CAPTURE
%token ACCKW_DELETE
%token ACCKW_FINALIZE
%token ACCKW_USE_DEVICE
%token ACCKW_DEVICE_RESIDENT
%token ACCKW_LINK
%token ACCKW_HOST
%token ACCKW_DEVICE
%token ACCKW_IF_PRESENT
%token ACCKW_DEVICE_NUM
%token ACCKW_DEFAULT_ASYNC

%type <code> acc_reduction_op
%type <code> acc_end_clause

%type <val> acc_directive acc_if_clause acc_parallel_clause_list acc_data_clause_list acc_loop_clause_list acc_parallel_loop_clause_list acc_kernels_loop_clause_list acc_wait_clause_list acc_expr_list acc_data_clause acc_var acc_var_list acc_subscript acc_subscript_list acc_csep acc_parallel_clause acc_kernels_clause_list acc_kernels_clause acc_routine_clause_list acc_enter_data_clause_list acc_exit_data_clause_list acc_host_data_clause_list acc_declare_clause_list acc_update_clause_list acc_init_clause_list acc_shutdown_clause_list acc_set_clause_list

/* abstract clause */
%type <val> acc_loop_clause acc_atomic_clause acc_enter_data_clause acc_exit_data_clause acc_declare_clause acc_update_clause acc_set_clause acc_compute_clause acc_parallel_loop_clause acc_kernels_loop_clause acc_routine_clause acc_init_clause acc_shutdown_clause acc_host_data_clause

/* clause */
%type <val> acc_async_clause acc_wait_clause acc_device_type_clause acc_num_gangs_clause acc_num_workers_clause acc_vector_length_clause acc_reduction_clause acc_private_clause acc_firstprivate_clause acc_default_clause acc_default_clause_arg acc_collapse_clause acc_gang_clause acc_worker_clause acc_vector_clause acc_seq_clause acc_auto_clause acc_tile_clause acc_independent_clause acc_bind_clause acc_nohost_clause acc_delete_clause acc_finalize_clause acc_copy_clause acc_copyin_clause acc_copyout_clause acc_create_clause acc_present_clause acc_present_or_copy_clause acc_present_or_copyin_clause acc_present_or_copyout_clause acc_present_or_create_clause acc_use_device_clause acc_device_resident_clause acc_link_clause acc_host_clause acc_device_clause acc_if_present_clause acc_device_num_clause acc_default_async_clause acc_deviceptr_clause

/* others */
%type <val> acc_id_list acc_gang_arg_list acc_num_expr acc_length_expr acc_size_expr acc_size_expr_list acc_gang_arg

%{
#include "F-front.h"
static int st_no;

static char *formatString = NULL;

/* omp buffer for simple omp lex.  */
static char *pragmaString = NULL;

typedef union {
    expr val;
    int code;
} yyStackType;

#define YYSTYPE yyStackType

extern void     yyerror _ANSI_ARGS_((char *s));
extern int      yylex _ANSI_ARGS_((void));
static int      yylex0 _ANSI_ARGS_((void));
static void     flush_line _ANSI_ARGS_((void));

static void set_pragma_str _ANSI_ARGS_((char *p));
static void append_pragma_str _ANSI_ARGS_((char *p));

#define GEN_NODE(TYPE, VALUE) make_enode((TYPE), ((void *)((_omAddrInt_t)(VALUE))))
#define OMP_LIST(op, args) list2(LIST, GEN_NODE(INT_CONSTANT, op), args)
#define XMP_LIST(op, args) list2(XMP_PRAGMA, GEN_NODE(INT_CONSTANT, op), args)
#define ACC_LIST(op, args) list2(ACC_PRAGMA, GEN_NODE(INT_CONSTANT, op), args)

/* statement name */
expr st_name;

/************************* NOT USED
static expr
gen_default_real_kind(void) {
    return list2(F_ARRAY_REF,
                 GEN_NODE(IDENT, find_symbol("kind")),
                 list1(LIST,
                       make_float_enode(F_DOUBLE_CONSTANT,
                                        0.0,
                                        strdup("0.0D0"))));
}
**********************************/

%}

%type <val> statement label
%type <val> expr /*expr1*/ lhs member_ref lhs_alloc member_ref_alloc substring expr_or_null complex_const array_constructor_list
%type <val> program_name dummy_arg_list dummy_args dummy_arg file_name
%type <val> declaration_statement executable_statement action_statement action_statement_let action_statement_key assign_statement_or_null assign_statement
%type <val> declaration_list entity_decl type_spec type_spec0 length_spec common_decl
%type <val> common_block external_decl intrinsic_decl equivalence_decl
%type <val> cray_pointer_list cray_pointer_pair cray_pointer_var
%type <val> equiv_list data data_list data_val_list data_val value simple_value save_list save_item const_list const_item common_var data_var data_var_list image_dims image_dim_list image_dim image_dims_alloc image_dim_list_alloc image_dim_alloc dims dim_list dim ubound label_list implicit_decl imp_list letter_group letter_groups namelist_decl namelist_list ident_list access_ident_list access_ident
%type <val> do_spec arg arg_list parenthesis_arg_list image_selector cosubscript_list
%type <val> parenthesis_arg_list_or_null
%type <val> set_expr
%type <val> io_statement format_spec ctl_list io_clause io_list_or_null io_list io_item
%type <val> IDENTIFIER CONSTANT const kind_parm GENERIC_SPEC USER_DEFINED_OP
%type <val> string_const_substr

%type <val> name name_or_null generic_name defined_operator intrinsic_operator func_prefix prefix_spec
%type <val> declaration_statement95 attr_spec_list attr_spec access_spec
%type <val> intent_spec kind_selector kind_or_len_selector char_selector len_key_spec len_spec kind_key_spec array_allocation_list  array_allocation defered_shape_list defered_shape
%type <val> result_opt type_keyword
%type <val> action_statement95
%type <val> action_coarray_statement other_coarray_keyword
%type <val> sync_stat_arg_list sync_stat_arg image_set
%type <val> use_rename_list use_rename use_only_list use_only 
%type <val> allocation_list allocation
%type <val> scene_list scene_range


%start program
%%

program: /* empty */
        | program one_statement EOS
        ;

KW: { need_keyword = TRUE; };

NEED_CHECK: {	      need_check_user_defined = FALSE; };

one_statement:
          STATEMENT_LABEL_NO  /* null statement */
        | STATEMENT_LABEL_NO statement
        { compile_statement(st_no,$2);}
	| OMPKW_LINE omp_directive
	{ compile_OMP_directive($2); }
	| XMPKW_LINE { need_keyword = TRUE; } xmp_directive
	{ compile_XMP_directive($3); }
	| ACCKW_LINE { need_keyword = TRUE; } acc_directive
	{ compile_ACC_directive($3); }
        | PRAGMA_HEAD  PRAGMA_SLINE /* like !$ ... */
	{
	    if (pragmaString != NULL)
		compile_statement(
		    st_no,
		    list1(F_PRAGMA_STATEMENT,
			  GEN_NODE(STRING_CONSTANT,
				   pragmaString)));
	}
        | error
        { flush_line(); yyerrok; yyclearin; }
        ;

statement:      /* entry */
          PROGRAM IDENTIFIER
          { $$ = list1(F_PROGRAM_STATEMENT,$2); }
        | ENDPROGRAM name_or_null
          { $$ = list1(F95_ENDPROGRAM_STATEMENT,$2); }
        | MODULE name
          { $$ = list1(F95_MODULE_STATEMENT,$2); }
        | ENDMODULE name_or_null
          { $$ = list1(F95_ENDMODULE_STATEMENT,$2); }
        | INTERFACEOPERATOR NEED_CHECK '(' defined_operator ')'
          {
	      $$ = list1(F95_INTERFACE_STATEMENT, $4);
	      need_check_user_defined = TRUE;
          }
        | INTERFACEASSIGNMENT '(' '=' ')'
          { $$ = list1(F95_INTERFACE_STATEMENT, list0(F95_ASSIGNOP)); }
        | INTERFACE generic_name
          { $$ = list1(F95_INTERFACE_STATEMENT, $2); }
        | INTERFACE
          { $$ = list1(F95_INTERFACE_STATEMENT,NULL); }
        | ENDINTERFACE generic_name
          { $$ = list1(F95_ENDINTERFACE_STATEMENT,$2); }
        | ENDINTERFACE OPERATOR '(' '=' ')'
          { $$ = list1(F95_ENDINTERFACE_STATEMENT,
                       GEN_NODE(IDENT, find_symbol("="))); }
        | ENDINTERFACE ASSIGNMENT '(' '=' ')'
          { $$ = list1(F95_ENDINTERFACE_STATEMENT,
                       GEN_NODE(IDENT, find_symbol("="))); }
        | ENDINTERFACE OPERATOR '(' intrinsic_operator ')'
          { $$ = list1(F95_ENDINTERFACE_STATEMENT, $4); }
        | ENDINTERFACE OPERATOR '(' USER_DEFINED_OP ')'
          { $$ = list1(F95_ENDINTERFACE_STATEMENT, $4); }
        | ENDINTERFACE
          { $$ = list1(F95_ENDINTERFACE_STATEMENT,NULL); }
        | MODULEPROCEDURE ident_list
          { $$ = list2(F95_MODULEPROCEDURE_STATEMENT, $2, make_int_enode(1)); }
        | PROCEDURE ident_list
          { $$ = list2(F95_MODULEPROCEDURE_STATEMENT, $2, make_int_enode(0)); }
        | BLOCKDATA program_name
          { $$ = list1(F_BLOCK_STATEMENT,$2); }
        | ENDBLOCKDATA name_or_null
          { $$ = list1(F95_ENDBLOCKDATA_STATEMENT,$2); }
        | SUBROUTINE IDENTIFIER dummy_arg_list
          { $$ = list3(F_SUBROUTINE_STATEMENT,$2,$3,NULL); }
        | func_prefix SUBROUTINE IDENTIFIER dummy_arg_list
          { $$ = list3(F_SUBROUTINE_STATEMENT,$3,$4,$1); }
        | ENDSUBROUTINE name_or_null
          { $$ = list1(F95_ENDSUBROUTINE_STATEMENT,$2); }
        | FUNCTION IDENTIFIER dummy_arg_list KW result_opt
          { $$ = list5(F_FUNCTION_STATEMENT,$2,$3,NULL,NULL, $5); }
        | func_prefix FUNCTION IDENTIFIER dummy_arg_list KW result_opt
          { $$ = list5(F_FUNCTION_STATEMENT,$3,$4,NULL,$1, $6); }
        | type_spec FUNCTION IDENTIFIER dummy_arg_list KW result_opt
          { $$ = list5(F_FUNCTION_STATEMENT,$3,$4,$1,NULL, $6); }
        | type_spec func_prefix FUNCTION IDENTIFIER dummy_arg_list
          KW result_opt
          { $$ = list5(F_FUNCTION_STATEMENT,$4,$5,$1,$2, $7); }
        | func_prefix type_spec FUNCTION IDENTIFIER dummy_arg_list
          KW result_opt
          { $$ = list5(F_FUNCTION_STATEMENT,$4,$5,$2,$1, $7); }
        | ENDFUNCTION name_or_null
          { $$ = list1(F95_ENDFUNCTION_STATEMENT,$2); }
        | type_spec COL2_or_null declaration_list
          { $$ = list3(F_TYPE_DECL,$1,$3,NULL); }
        | type_spec attr_spec_list COL2 declaration_list
          { $$ = list3(F_TYPE_DECL,$1,$4,$2); }
        | ENTRY IDENTIFIER dummy_arg_list KW result_opt
          { $$ = list3(F_ENTRY_STATEMENT,$2,$3, $5); }
        | CONTAINS
          { $$ = list0(F95_CONTAINS_STATEMENT); }
        | declaration_statement
        | executable_statement
        | declaration_statement95
        | INCLUDE file_name
          { $$ = list1(F_INCLUDE_STATEMENT,$2); }
        | END
          { $$ = list0(F_END_STATEMENT); }
        | UNKNOWN
          { error("unclassifiable statement"); flush_line(); $$ = NULL; }

label:    CONSTANT      /* must be interger constant */
        ;

program_name:   /* null */
         { $$ = NULL; }
        | IDENTIFIER
        ;

result_opt:    /* null */
          { $$ = NULL; }
        | RESULT '(' name ')'
          { $$ = $3; }
        ;

intrinsic_operator: '.'
        { $$ = list0(F95_DOTOP); }
        | POWER
        { $$ = list0(F95_POWEOP); }
        | '*'
        { $$ = list0(F95_MULOP); }
        | '/'
        { $$ = list0(F95_DIVOP); }
        | '+'
        { $$ = list0(F95_PLUSOP); }
        | '-'
        { $$ = list0(F95_MINUSOP); }
        | EQ
        { $$ = list0(F95_EQOP); }
        | NE
        { $$ = list0(F95_NEOP); }
        | LT
        { $$ = list0(F95_LTOP); }
        | LE
        { $$ = list0(F95_LEOP); }
        | GE
        { $$ = list0(F95_GEOP); }
        | GT
        { $$ = list0(F95_GTOP); }
        | NOT
        { $$ = list0(F95_NOTOP); }
        | AND
        { $$ = list0(F95_ANDOP); }
        | OR
        { $$ = list0(F95_OROP); }
        | EQV
        { $$ = list0(F95_EQVOP); }
        | NEQV
        { $$ = list0(F95_NEQVOP); }
        | CONCAT
        { $$ = list0(F95_CONCATOP); }
        ;

defined_operator: intrinsic_operator
        | '.' IDENTIFIER '.'
        { $$ = list1(F95_USER_DEFINED, $2); }
        ;

generic_name:
        name
        ;

func_prefix:
          prefix_spec
        { $$ = list1(LIST,$1); need_keyword = TRUE; }
        | func_prefix prefix_spec
        { $$ = list_put_last($1,$2); need_keyword = TRUE; }
        ;

prefix_spec:
          RECURSIVE
        { $$ = list0(F95_RECURSIVE_SPEC); }
        | PURE
        { $$ = list0(F95_PURE_SPEC); }
        | ELEMENTAL
        { $$ = list0(F95_ELEMENTAL_SPEC); }
        ;

name:  IDENTIFIER;

name_or_null:
        { $$ = NULL; }
        | IDENTIFIER
        ;

dummy_arg_list:
        { $$ = NULL; }
        | '(' ')'
        { $$ = NULL; }
        | '(' dummy_args ')'
        { $$ = $2; }
        ;

dummy_args:
        dummy_arg
        { $$ = list1(LIST,$1); }
        | dummy_args ',' dummy_arg
        { $$ = list_put_last($1,$3); }
        ;

dummy_arg:
         IDENTIFIER
        | '*'
        { $$ = NULL; }
        ;

file_name:
         CONSTANT       /* must be hollerith? */
        ;

declaration_statement:
          DIMENSION COL2_or_null declaration_list
        { $$ = list1(F95_DIMENSION_DECL,$3); }
        | COMMON common_decl
        { $$ = list1(F_COMMON_DECL,$2); }
        | EXTERNAL COL2_or_null external_decl
        { $$ = list1(F_EXTERNAL_DECL, $3); }
        | INTRINSIC COL2_or_null intrinsic_decl
        { $$ = list1(F_INTRINSIC_DECL,$3); }
        | EQUIV equivalence_decl
        { $$ = list1(F_EQUIV_DECL,$2); }
        | DATA data
        { $$ = list1(F_DATA_DECL,$2); }
        | IMPLICIT_NONE /* implicit none  */
        { $$ = list1(F_IMPLICIT_DECL, NULL); }
        | IMPLICIT implicit_decl
        { $$ = list1(F_IMPLICIT_DECL, $2); }
        | NAMELIST namelist_decl
        { $$ = list1(F_NAMELIST_DECL,$2); }
        | SAVE
        { $$ = list1(F_SAVE_DECL,NULL); }
        | SAVE COL2_or_null save_list
        { $$ = list1(F_SAVE_DECL,$3); }
        | PARAMETER  '(' const_list ')'
        { $$ = list1(F_PARAM_DECL,$3); }
        | POINTER cray_pointer_list
        { $$ = list1(F_CRAY_POINTER_DECL, $2); }
        | FORMAT
        {
            if (formatString == NULL) {
                fatal("can't get format statement as string.");
            }
            $$ = list1(F_FORMAT_DECL, GEN_NODE(STRING_CONSTANT, formatString));
            formatString = NULL;
        }
        ;

declaration_statement95:
          KW_TYPE COL2_or_null IDENTIFIER
        { $$ = list2(F95_TYPEDECL_STATEMENT,$3,NULL); }
        | KW_TYPE ',' KW access_spec COL2 IDENTIFIER
        { $$ = list2(F95_TYPEDECL_STATEMENT,$6,$4); }
        | ENDTYPE
        { $$ = list1(F95_ENDTYPEDECL_STATEMENT,NULL); }
        | ENDTYPE IDENTIFIER
        { $$ = list1(F95_ENDTYPEDECL_STATEMENT,$2); }
        | OPTIONAL COL2_or_null ident_list
        { $$ = list1(F95_OPTIONAL_STATEMENT, $3); }
        | POINTER COL2_or_null array_allocation_list
        { $$ = list1(F95_POINTER_STATEMENT, $3); }
        | TARGET COL2_or_null array_allocation_list
        { $$ = list1(F95_TARGET_STATEMENT, $3); }
        | PUBLIC
        { $$ = list1(F95_PUBLIC_STATEMENT,NULL); }
        | PUBLIC access_ident_list
        { $$ = list1(F95_PUBLIC_STATEMENT, $2); }
        | PUBLIC COL2 access_ident_list
        { $$ = list1(F95_PUBLIC_STATEMENT, $3); }
        | PRIVATE
        { $$ = list1(F95_PRIVATE_STATEMENT,NULL); }
        | PRIVATE COL2_or_null access_ident_list
        { $$ = list1(F95_PRIVATE_STATEMENT, $3); }
        | PROTECTED
        { $$ = list1(F03_PROTECTED_STATEMENT,NULL); }
        | PROTECTED COL2_or_null access_ident_list
        { $$ = list1(F03_PROTECTED_STATEMENT, $3); }
        | SEQUENCE
        { $$ = list0(F95_SEQUENCE_STATEMENT); }
        | KW_USE IDENTIFIER
        { $$ = list2(F95_USE_STATEMENT,$2,NULL); }
        | KW_USE IDENTIFIER ',' KW use_rename_list
        { $$ = list2(F95_USE_STATEMENT,$2,$5); }
        | KW_USE IDENTIFIER ',' KW KW_ONLY ':' /* empty */
        { $$ = list2(F95_USE_ONLY_STATEMENT,$2, NULL); }
        | KW_USE IDENTIFIER ',' KW KW_ONLY ':' use_only_list
        { $$ = list2(F95_USE_ONLY_STATEMENT,$2,$7); }
        | INTENT '(' KW intent_spec ')' COL2_or_null ident_list
        { $$ = list2(F95_INTENT_STATEMENT, $4, $7); }
        | ALLOCATABLE COL2_or_null array_allocation_list
        { $$ = list1(F95_ALLOCATABLE_STATEMENT,$3); }
        ;

array_allocation_list:
          array_allocation
        { $$ = list1(LIST, $1); }
        | array_allocation_list ',' array_allocation
        { $$ = list_put_last($1, $3); }
        ;

array_allocation:
          IDENTIFIER
        { $$ = list5(F95_ARRAY_ALLOCATION, $1, NULL, NULL, NULL, NULL); }
        | IDENTIFIER '(' defered_shape_list ')'
        { $$ = list5(F95_ARRAY_ALLOCATION, $1, $3, NULL, NULL, NULL); }
        | IDENTIFIER '(' defered_shape_list ')' REF_OP IDENTIFIER '(' ')'
        { $$ = list5(F95_ARRAY_ALLOCATION, $1, $3, $6, NULL, NULL); }
        ;

defered_shape_list:
          defered_shape
        { $$ = list1(LIST, $1); }
        | defered_shape_list ',' defered_shape
        { $$ = list_put_last($1, $3); }
        ;

defered_shape: ':'
        { $$ = list2(LIST,NULL,NULL); }
        ;

use_rename_list:
          use_rename
        { $$ = list1(LIST,$1); }
        | use_rename_list ',' use_rename
        { $$ = list_put_last($1,$3); }
        ;

use_rename:
          IDENTIFIER REF_OP IDENTIFIER
        { $$ = list2(LIST,$1,$3); }
        ;

use_only_list:
          use_only
        { $$ = list1(LIST,$1); }
        | use_only_list ',' use_only
        { $$ = list_put_last($1,$3); }
        ;

use_only:
          use_rename
        | IDENTIFIER
        ;

COL2_or_null:
        | COL2
        ;

attr_spec_list:
          ',' KW attr_spec
        { $$ = list1(LIST,$3); }
        | attr_spec_list ',' KW attr_spec
        { $$ = list_put_last($1,$4); }
        ;

attr_spec:
          PARAMETER
        { $$ = list0(F95_PARAMETER_SPEC); }
        | access_spec
        | ALLOCATABLE
        { $$ = list0(F95_ALLOCATABLE_SPEC); }
        | DIMENSION '(' dim_list ')'
        { $$ = list1(F95_DIMENSION_SPEC,$3); }
        | CODIMENSION '[' image_dim_list ']'
        { $$ = list1(XMP_CODIMENSION_SPEC,$3); }
        | EXTERNAL
        { $$ = list0(F95_EXTERNAL_SPEC); }
        | INTENT '(' KW intent_spec ')'
        { $$ = list1(F95_INTENT_SPEC,$4); }
        | INTRINSIC
        { $$ = list0(F95_INTRINSIC_SPEC); }
        | OPTIONAL
        { $$ = list0(F95_OPTIONAL_SPEC); }
        | POINTER
        { $$ = list0(F95_POINTER_SPEC); }
        | SAVE
        { $$ = list0(F95_SAVE_SPEC); }
        | TARGET
        { $$ = list0(F95_TARGET_SPEC); }
        ;

access_spec:
          PUBLIC
        { $$ = list0(F95_PUBLIC_SPEC); }
        | PRIVATE
        { $$ = list0(F95_PRIVATE_SPEC); }
        | PROTECTED
        { $$ = list0(F03_PROTECTED_SPEC); }
        ;

intent_spec:
          KW_IN
        { $$ = list0(F95_IN_EXTENT); }
        | KW_OUT
        { $$ = list0(F95_OUT_EXTENT); }
        | KW_INOUT
        { $$ = list0(F95_INOUT_EXTENT); }
        ;

declaration_list:
         entity_decl
        { $$ = list1(LIST,$1); }
        | declaration_list ',' entity_decl
        { $$ = list_put_last($1,$3); }
        ;

entity_decl:
          IDENTIFIER  dims image_dims length_spec
        { $$ = list5(LIST,$1,$2,$4,NULL,$3); }
        | IDENTIFIER  dims image_dims length_spec '=' expr
        { $$ = list5(LIST,$1,$2,$4,$6,$3);}
        | IDENTIFIER  dims image_dims length_spec '/' data_val_list '/'
        { $$ = list5(LIST,$1,$2,$4,
                     list1(F_DATA_DECL,
                           list1(LIST,
                                 list2(LIST,
                                       list1(LIST, $1 ),
                                       $6 ))), $3);
        }
        | IDENTIFIER  dims image_dims length_spec REF_OP expr
        { $$ = list5(LIST,$1,$2,$4,$6,$3);}
        ;

type_spec: type_spec0 { $$ = $1; /* need_keyword = TRUE; */ };

type_spec0:
          KW_TYPE '(' IDENTIFIER ')'
        { $$ = $3; }
        | type_keyword kind_selector
        { $$ = list2(LIST,$1,$2); }
        | type_keyword length_spec  /* compatibility */
        { $$ = list2(LIST, $1, $2);}
        | KW_CHARACTER char_selector
        { $$ = list2(LIST,GEN_NODE(F_TYPE_NODE,TYPE_CHAR),$2); }
        | KW_DOUBLE
        { $$ = list2 (LIST, GEN_NODE(F_TYPE_NODE, TYPE_REAL),
		      GEN_NODE(INT_CONSTANT, 8)); }
		      //                      gen_default_real_kind()); }
        | KW_DCOMPLEX
        { $$ = list2 (LIST, GEN_NODE(F_TYPE_NODE, TYPE_COMPLEX),
		      GEN_NODE(INT_CONSTANT, 8)); }
		      //                      gen_default_real_kind()); }
        ;

type_keyword:
          KW_INTEGER    { $$ = GEN_NODE(F_TYPE_NODE,TYPE_INT); }
        | KW_REAL       { $$ = GEN_NODE(F_TYPE_NODE,TYPE_REAL); }
        | KW_COMPLEX        { $$ = GEN_NODE(F_TYPE_NODE,TYPE_COMPLEX); }
        | KW_LOGICAL        { $$ = GEN_NODE(F_TYPE_NODE,TYPE_LOGICAL); }
        ;

kind_selector:
        kind_or_len_selector
         { $$ = $1; }
        ;

char_selector: /* empty */
        { $$ = NULL; }
        | '(' len_spec ')'
        { $$ = list2(LIST, $2, NULL); }
        | SET_LEN  len_spec ')'
        { $$ = list2(LIST, $2, NULL); }
        | SET_LEN len_spec ',' KW kind_key_spec ')'
        { $$ = list2(LIST, $2, $5); }
        | '(' len_spec ',' KW KW_KIND '=' expr ')'
        { $$ = list2(LIST, $2, list1(F95_KIND_SELECTOR_SPEC, $7)); }
        | SET_KIND len_spec ')'
        { $$ = list2(LIST, NULL, $2); }
        | SET_KIND len_spec ',' KW len_key_spec')'
        { $$ = list2(LIST, $5, $2); }
        | length_spec_mark  expr
        { $$ = $2; }
        | length_spec_mark '(' '*' ')'
        { $$ = list0(LIST); }
        ;

len_key_spec: KW_LEN '=' expr
         { $$ = list1(F95_LEN_SELECTOR_SPEC, $3); }
        ;

len_spec: '*'
        { $$ = list1(F95_LEN_SELECTOR_SPEC, NULL); }
        | expr
        { $$ = list1(F95_LEN_SELECTOR_SPEC, $1); }
        ;

kind_key_spec: KW_KIND '=' expr
        { $$ = list1(F95_KIND_SELECTOR_SPEC, $3); }
        ;
kind_or_len_selector:
          SET_KIND expr  ')'
        { $$ = list1(F95_KIND_SELECTOR_SPEC, $2); }
        | SET_LEN '*' ')'
        { $$ = list1(F95_LEN_SELECTOR_SPEC, NULL); }
        | SET_LEN expr ')'
        { $$ = list1(F95_LEN_SELECTOR_SPEC, $2); }
        | '(' expr ')'
        { $$ = $2; }
        ;

length_spec:    /* nothing */
        { $$ = NULL; }
        | length_spec_mark  expr
        { $$ = list1(F95_LEN_SELECTOR_SPEC,$2); }
        | length_spec_mark '(' '*' ')'
        { $$ = list1(F95_LEN_SELECTOR_SPEC, NULL); }
        ;

length_spec_mark:
        '*' { need_type_len = TRUE; }
        ;

common_decl:
          common_var
        { $$ = list2(LIST, NULL, $1); }
        | common_block common_var
        { $$ = list2(LIST,$1,$2); }
        | common_decl comma_or_null common_block comma_or_null common_var
        { $$ = list_put_last(list_put_last($1,$3),$5); }
        | common_decl ',' common_var
        { $$ = list_put_last($1,$3); }
        ;

common_block:  CONCAT /* // */
        { $$ = NULL; }
        | '/' IDENTIFIER '/'
        { $$ = $2; }
        ;

common_var:  IDENTIFIER dims
        { $$ = list2(LIST,$1,$2); }
        ;

external_decl: IDENTIFIER
        { $$ = list1(LIST,$1); }
        | external_decl ',' IDENTIFIER
        { $$ = list_put_last($1,$3); }
        ;

intrinsic_decl:  IDENTIFIER
        { $$ = list1(LIST,$1); }
        | intrinsic_decl ',' IDENTIFIER
        { $$ = list_put_last($1,$3); }
        ;

equivalence_decl:
          '(' equiv_list ')'
        { $$ = list1(LIST,$2); }
        | equivalence_decl ',' '(' equiv_list ')'
        { $$ = list_put_last($1,$4); }
        ;

equiv_list:
          lhs
        { $$ = list1(LIST,$1); }
        | equiv_list ',' lhs
        { $$ = list_put_last($1,$3); }
        ;

cray_pointer_list:
        cray_pointer_pair
        { $$ = list1(LIST, $1); }
        | cray_pointer_list ',' cray_pointer_pair
        { $$ = list_put_last($1, $3); }
        ;

cray_pointer_pair:
        '(' lhs ',' cray_pointer_var ')'
        { $$ = list2(LIST, $2, $4); }
        ;

cray_pointer_var:
        IDENTIFIER
        { $$ = $1; }
        | common_var
        { $$ = list2(F_ARRAY_REF, EXPR_ARG1($1), EXPR_ARG2($1)); }
        ;

data:     data_list
        { $$ = list1(LIST,$1); }
        | data comma_or_null data_list
        { $$ = list_put_last($1,$3); }
        ;

data_list:  data_var_list '/' data_val_list '/'
        { $$ = list2(LIST,$1,$3); }
        ;

data_val_list:  data_val
        { $$ = list1(LIST,$1); }
        | data_val_list ',' data_val
        { $$ = list_put_last($1,$3); }
        ;

data_val: value
        { $$ = $1; }
        | IDENTIFIER parenthesis_arg_list
        { $$ = list2(F_ARRAY_REF,$1,$2); /* struct constructor */ }
        | simple_value '*' value
        { $$ = list2(F_DUP_DECL,$1,$3); }
        ;

value: simple_value
        | '+' simple_value
        { $$ = $2;}
        | '-' simple_value
        { $$ = list1(F_UNARY_MINUS_EXPR,$2); }
        ;

simple_value:
        IDENTIFIER
        | const
        | complex_const
        ;

save_list: save_item
        { $$ = list1(LIST,$1); }
        | save_list ',' save_item
        { $$ = list_put_last($1,$3); }
        ;

save_item: IDENTIFIER
        | common_block
        { $$ = list1(LIST,$1); } /* for identify common block name */
        ;

access_ident_list: access_ident
        { $$ = list1(LIST, $1); }
        | access_ident_list ',' access_ident
        { $$ = list_put_last($1, $3); }
        ;

access_ident: GENERIC_SPEC
        | IDENTIFIER
        ;

/*
access_ident: KW OPERATOR_P defined_operator ')'
          { $$ = list1(F95_GENERIC_SPEC, $3); }
        | KW ASSIGNMENT_P '=' ')'
          { $$ = list1(F95_GENERIC_SPEC, list0(F95_ASSIGNOP)); }
        | IDENTIFIER
        ;
*/

ident_list: IDENTIFIER
        { $$ = list1(LIST,$1); }
        | ident_list ',' IDENTIFIER
        { $$ = list_put_last($1,$3); }
        ;

const_list:  const_item
        { $$ = list1(LIST,$1); }
        | const_list ',' const_item
        { $$ = list_put_last($1,$3); }
        ;

const_item:  IDENTIFIER '=' expr
        { $$ = list2(LIST,$1,$3); }
        ;


data_var_list: data_var
        { $$ = list1(LIST,$1); }
        | data_var_list ',' data_var
        { $$ = list_put_last($1,$3); }
        ;

data_var:         lhs
        | '(' data_var_list ',' do_spec ')'
        { $$ = list2(F_IMPLIED_DO, $4, $2); }
        ;

image_dims:
        { $$ = NULL; }
        | '[' image_dim_list ']'
        { $$ = $2; }
        ;

image_dim_list:  image_dim
        { $$ = list1(LIST,$1); }
        | image_dim_list ',' image_dim
        { $$ = list_put_last($1,$3); }
        ;

image_dim:      ubound
        | expr ':' ubound
        { $$ = list2(LIST,$1,$3); }
        | ':'
        { $$ = list2(LIST,NULL,NULL); }
        ;

image_dims_alloc:
          '[' image_dim_list_alloc ']'
        { $$ = $2; }
        ;

image_dim_list_alloc:  image_dim_alloc
        { $$ = list1(LIST,$1); }
        | image_dim_list_alloc ',' image_dim_alloc
        { $$ = list_put_last($1,$3); }
        ;

image_dim_alloc:      ubound
        { $$ = list3(F95_TRIPLET_EXPR,NULL,$1,NULL); }
        | expr ':' ubound
        { $$ = list3(F95_TRIPLET_EXPR,$1,$3,NULL); }
        ;

dims:
        { $$ = NULL; }
        | '(' dim_list ')'
        { $$ = $2; }
        ;

dim_list:  dim
        { $$ = list1(LIST,$1); }
        | dim_list ',' dim
        { $$ = list_put_last($1,$3); }
        ;

dim:      ubound
        | expr ':' ubound
        { $$ = list2(LIST,$1,$3); }
        | expr ':'
        { $$ = list2(LIST,$1,NULL); }
        | ':'
        { $$ = list2(LIST,NULL,NULL); }
        ;

ubound:   '*'
        { $$ = list0(F_ASTERISK); }
        | expr
        ;

label_list: label
        { $$ = list1(LIST,$1); }
        | label_list ',' label
        { $$ = list_put_last($1,$3); }
        ;

implicit_decl:    imp_list
        { $$ = list1(LIST,$1); }
        | implicit_decl ',' imp_list
        { $$ = list_put_last($1,$3); }
        ;

/* in lexer, change the  '(' and ')'  around letter_group to [/].  */
imp_list: KW type_spec '[' letter_groups ']'
        { $$ = list2(LIST, list2(LIST, $2, $4), NULL); }
        | KW type_spec
        { $$ = list2(LIST, list2(LIST, $2,NULL), NULL); }
        ;

letter_groups: letter_group
        { $$ = list1(LIST,$1); }
        | letter_groups ',' letter_group
        { $$ = list_put_last($1,$3); }
        ;

letter_group:  IDENTIFIER
        | IDENTIFIER '-' IDENTIFIER
        { $$ = list2(LIST,$1,$3); }
        ;

namelist_decl: '/' IDENTIFIER '/' namelist_list
        { $$ = list1(LIST,list2(LIST,$2,$4)); }
        | '/' IDENTIFIER '/' namelist_list comma_or_null namelist_decl
        { $$ = list_cons(list2(LIST,$2,$4),$6); }
        ;

namelist_list:  IDENTIFIER
        { $$ = list1(LIST, $1); }
        | namelist_list ',' IDENTIFIER
        { $$ = list_put_last($1,$3); }
        ;

/*
 * executable statement
 */
executable_statement:
          action_statement
	| DO label KW_WHILE '(' expr ')'
	{ $$ = list3(F_DOWHILE_STATEMENT, $2, $5, st_name); }
        | DO label do_spec
        { $$ = list3(F_DO_STATEMENT, $2, $3, st_name); }
        | DO label ',' do_spec  /* for dusty deck */
        { $$ = list3(F_DO_STATEMENT, $2, $4, st_name); }
        | DO label
        { $$ = list3(F_DO_STATEMENT, $2, NULL, st_name); }
        | DO do_spec
        { $$ = list3(F_DO_STATEMENT,NULL, $2, st_name); }
        | DO
        { $$ = list3(F_DO_STATEMENT,NULL, NULL, st_name); }
        | ENDDO name_or_null
        { $$ = list1(F_ENDDO_STATEMENT,$2); }
        | LOGIF '(' expr ')' action_statement_key /* with keyword */
        { $$ = list2(F_IF_STATEMENT, $3, $5); }
        | LOGIF '(' expr ')' action_statement_let /* for LET... */
        { $$ = list2(F_IF_STATEMENT, $3, $5); }
        | IFTHEN '(' expr ')' KW THEN
        { $$ = list3(F_IF_STATEMENT, $3, NULL, st_name); }
        | ELSEIFTHEN '(' expr ')' KW THEN
        { $$ = list1(F_ELSEIF_STATEMENT,$3); }
        | ELSE name_or_null
        { $$ = list0(F_ELSE_STATEMENT); }
        | ENDIF name_or_null /* need to match the label in st_name?  */
        { $$ = list0(F_ENDIF_STATEMENT); }
        | DOWHILE '(' expr ')'
        { $$ = list3(F_DOWHILE_STATEMENT, NULL, $3, st_name); }
	/***
	WHERE, ELSEWHERE and ENDWHERE implimanetation is not appropriate now.
	it should be:

	 | WHERE '(' expr ')'
	 {...}
	 | ELSEWHERE '(' expr ')'
	 {...}
	 | ELSEWHERE
	 {...}
         | ENDWHERE
         {...}

	 then on compiling procedure switch cotrol-type
	 CTL_WHERE/CTL_ELSE_WHERE and treat coming statement
	 appropriately.
	 ***/
        | WHERE '(' expr ')' assign_statement_or_null
        { $$ = list2(F_WHERE_STATEMENT,$3,$5); }
        | ELSEWHERE
        { $$ = list0(F_ELSEWHERE_STATEMENT); }
        | ELSEWHERE '(' expr ')' assign_statement_or_null
        { $$ = list2(F_ELSEWHERE_STATEMENT, $3, $5); }
        | ENDWHERE
        { $$ = list0(F_ENDWHERE_STATEMENT); }
        | SELECT '(' expr ')'
        { $$ = list2(F_SELECTCASE_STATEMENT, $3, st_name); }
        | CASE '(' scene_list ')' name_or_null
        { $$ = list2(F_CASELABEL_STATEMENT, $3, $5); }
        | CASEDEFAULT name_or_null
        { $$ = list2(F_CASELABEL_STATEMENT, NULL, $2); }
        | ENDSELECT name_or_null
        { $$ = list1(F_ENDSELECT_STATEMENT,$2); }
        ;

assign_statement_or_null:
        { $$ = NULL; }
        | assign_statement
        { $$ = $1; }
        ;

assign_statement: lhs '=' expr
        { $$ = list2(F_LET_STATEMENT,$1,$3); }
        ;

do_spec:
          IDENTIFIER '=' expr ',' expr
        { $$ = list4(LIST,$1,$3,$5,NULL); }
        |  IDENTIFIER '=' expr ',' expr ',' expr
        { $$ = list4(LIST,$1,$3,$5,$7); }
        ;

/* 'ifable' statement */
action_statement: action_statement_let
        | action_statement_key
        ;

action_statement_let:
          LET assign_statement
         { $$ = $2; }
        | LET expr REF_OP expr
         { $$ = list2(F95_POINTER_SET_STATEMENT,$2,$4); }
        ;
action_statement_key: ASSIGN  label KW KW_TO IDENTIFIER
        { $$ = list2(F_ASSIGN_LABEL_STATEMENT, $2, $5); }
        | CONTINUE
        { $$ = list0(F_CONTINUE_STATEMENT); }
        | GOTO  label
        { $$ = list1(F_GOTO_STATEMENT,$2); }
        | GOTO  IDENTIFIER
        { $$ = list2(F_ASGOTO_STATEMENT,$2,NULL); }
        | GOTO  IDENTIFIER comma_or_null '(' label_list ')'
        { $$ = list2(F_ASGOTO_STATEMENT,$2,$5); }
        | GOTO  '(' label_list ')' comma_or_null expr
        { $$ = list2(F_COMPGOTO_STATEMENT,$3,$6); }
        | ARITHIF  '(' expr ')' label ',' label ',' label
        { $$ = list4(F_ARITHIF_STATEMENT,$3,$5,$7,$9); }
        | CALL IDENTIFIER
        { $$ = list2(F_CALL_STATEMENT,$2,NULL); }
        | CALL IDENTIFIER '(' ')'
        { $$ = list2(F_CALL_STATEMENT,$2,NULL); }
        | CALL IDENTIFIER '(' arg_list ')'
        { $$ = list2(F_CALL_STATEMENT,$2,$4); }
        | RETURN  expr_or_null
        { $$ = list1(F_RETURN_STATEMENT,$2); }
        | PAUSE  expr_or_null
        { $$ = list1(F_PAUSE_STATEMENT,$2); }
        | STOP  expr_or_null
        { $$ = list1(F_STOP_STATEMENT,$2); }
        | action_statement95 /* all has first key.  */
        | action_coarray_statement /* all has first key.  */
        | io_statement /* all has first key.  */
        | PRAGMA_SLINE
        {
          $$ = list1(F_PRAGMA_STATEMENT,
                     GEN_NODE(STRING_CONSTANT, pragmaString));
         pragmaString = NULL;
        }
        ;

action_statement95:
          CYCLE name_or_null
        { $$ = list1(F95_CYCLE_STATEMENT,$2); }
        | EXIT name_or_null
        { $$ = list1(F95_EXIT_STATEMENT,$2); }
        | ALLOCATE '(' allocation_list ')'
        { $$ = list1(F95_ALLOCATE_STATEMENT,$3); }
        | NULLIFY '(' allocation_list ')'
        { $$ = list1(F95_NULLIFY_STATEMENT,$3); }
        | DEALLOCATE '(' allocation_list ')'
        { $$ = list1(F95_DEALLOCATE_STATEMENT,$3); }
        ;

allocation_list:
          allocation
        { $$ = list1(LIST,$1); }
        | allocation_list ',' allocation
        { $$ = list_put_last($1,$3); }
        ;

allocation:
          lhs_alloc
        | set_expr
        ;

action_coarray_statement:
          SYNCALL
        { $$ = list1(F2008_SYNCALL_STATEMENT,NULL); }
        | SYNCALL '(' ')'
        { $$ = list1(F2008_SYNCALL_STATEMENT,NULL); }
        | SYNCALL '(' sync_stat_arg_list ')'
        { $$ = list1(F2008_SYNCALL_STATEMENT,$3); }
        | SYNCIMAGES '(' image_set ')'
        { $$ = list2(F2008_SYNCIMAGES_STATEMENT,$3, NULL); }
        | SYNCIMAGES '(' image_set ',' sync_stat_arg_list ')'
        { $$ = list2(F2008_SYNCIMAGES_STATEMENT,$3, $5); }
        | SYNCMEMORY
        { $$ = list1(F2008_SYNCMEMORY_STATEMENT,NULL); }
        | SYNCMEMORY '(' ')'
        { $$ = list1(F2008_SYNCMEMORY_STATEMENT,NULL); }
        | SYNCMEMORY '(' sync_stat_arg_list ')'
        { $$ = list1(F2008_SYNCMEMORY_STATEMENT,$3); }
        | CRITICAL
        { $$ = list1(F2008_CRITICAL_STATEMENT,st_name); }
        | ENDCRITICAL
        { $$ = list1(F2008_ENDCRITICAL_STATEMENT,NULL); }
        | ENDCRITICAL IDENTIFIER
        { $$ = list1(F2008_ENDCRITICAL_STATEMENT,$2); }
        | LOCK '(' expr ')'
        { $$ = list2(F2008_LOCK_STATEMENT,$3, NULL); }
        | LOCK '(' expr ',' sync_stat_arg_list ')'
        { $$ = list2(F2008_LOCK_STATEMENT,$3, $5); }
        | UNLOCK '(' expr ')'
        { $$ = list2(F2008_UNLOCK_STATEMENT,$3, NULL); }
        | UNLOCK '(' expr ',' sync_stat_arg_list ')'
        { $$ = list2(F2008_UNLOCK_STATEMENT,$3, $5); }
        | other_coarray_keyword parenthesis_arg_list_or_null
        { $$ = list2(F_CALL_STATEMENT,$1,$2); }
        ;


sync_stat_arg_list:
          sync_stat_arg
        { $$ = list1(LIST, $1); }
        | sync_stat_arg_list ',' sync_stat_arg
        { $$ = list_put_last($1,$3); }
        ;

sync_stat_arg:
          IDENTIFIER '=' IDENTIFIER
        { $$ = list2(F_SET_EXPR,$1,$3); }
        ;

image_set:
          expr
        { $$ = $1; }
        | '*'
        { $$ = NULL; }
        ;

other_coarray_keyword:
          ERRORSTOP
        { $$ = GEN_NODE(IDENT, find_symbol("xmpf_error_stop")); }
        ;

comma_or_null:
        | ','
        ;

parenthesis_arg_list_or_null:
        { $$ = NULL; }
        | parenthesis_arg_list
        { $$ = $1; }
        ;

parenthesis_arg_list:
          '(' arg_list ')'
        { $$ = $2; }
        ;

/* actual argument */
arg_list:
        { $$ = NULL; }
        | arg
        { $$ = list1(LIST,$1); }
        | arg_list ',' arg
        { $$ = list_put_last($1,$3); }
        ;

arg:
         expr
        | set_expr
        | '*' label
         { $$ = list1(F_LABEL_REF,$2); }
        | expr_or_null ':' expr_or_null
         { $$ = list3(F95_TRIPLET_EXPR,$1,$3,NULL); }
        | expr_or_null ':' expr_or_null ':' expr
         { $$ = list3(F95_TRIPLET_EXPR,$1,$3,$5); }
        | expr_or_null COL2 expr
         { $$ = list3(F95_TRIPLET_EXPR,$1,NULL,$3); }
        ;


image_selector:
          '[' cosubscript_list ']'
        { $$ = $2; }
        ;

cosubscript_list:
          expr
        { $$ = list1(LIST,$1); }
        | cosubscript_list ',' expr
        { $$ = list_put_last($1,$3); }
        ;
/*
 * Input/Output Statements
 */
io_statement:
          PRINT format_spec
        { $$ = list2(F_PRINT_STATEMENT,$2,NULL); }
        | PRINT format_spec ',' io_list
        { $$ = list2(F_PRINT_STATEMENT,$2,$4); }
        | WRITE_P ctl_list ')'
        { $$ = list2(F_WRITE_STATEMENT,$2,NULL); }
        | WRITE_P ctl_list ')' io_list
        { $$ = list2(F_WRITE_STATEMENT,$2,$4); }
        | READ_P ctl_list ')'
        { $$ = list2(F_READ_STATEMENT,$2,NULL); }
        | READ_P ctl_list ')' io_list
        { $$ = list2(F_READ_STATEMENT,$2,$4); }
        | READ format_spec
        { $$ = list2(F_READ1_STATEMENT,list2(LIST,NULL,$2),NULL); }
        | READ format_spec ',' io_list
        { $$ = list2(F_READ1_STATEMENT,list2(LIST,NULL,$2),$4); }
        | OPEN '(' ctl_list ')'
        { $$ = list1(F_OPEN_STATEMENT,$3); }
        | CLOSE '(' ctl_list ')'
        { $$ = list1(F_CLOSE_STATEMENT,$3); }
        | BACKSPACE_P ctl_list ')'
        { $$ = list1(F_BACKSPACE_STATEMENT,$2); }
        | BACKSPACE format_spec
        { $$ = list1(F_BACKSPACE_STATEMENT,$2); }
        | ENDFILE_P ctl_list ')'
        { $$ = list1(F_ENDFILE_STATEMENT,$2); }
        | ENDFILE format_spec
        { $$ = list1(F_ENDFILE_STATEMENT,$2); }
        | REWIND_P ctl_list ')'
        { $$ = list1(F_REWIND_STATEMENT,$2); }
        | REWIND format_spec
        { $$ = list1(F_REWIND_STATEMENT,$2); }
        | INQUIRE '(' ctl_list ')' io_list_or_null
        { $$ = list2(F_INQUIRE_STATEMENT,$3, $5); }
        ;

ctl_list: io_clause
        { $$ = list1(LIST,$1); }
        | ctl_list ',' io_clause
        { $$ = list_put_last($1,$3); }
        ;

io_clause:
         expr
        |  '*'
        { $$ = NULL; }
        | POWER /* ** */
        { $$ = list0(F_STARSTAR); }
        | IDENTIFIER '=' '*'
        { $$ = list2(F_SET_EXPR,$1,NULL); }
        | IDENTIFIER '=' POWER
        { $$ = list2(F_SET_EXPR,$1,list0(F_STARSTAR)); }
        | set_expr
        { $$ = $1; }
        ;

set_expr:
        IDENTIFIER '=' expr
        {
            /*
             * FIXME:
             *
             *	Sorry I can't let a grammer "KW KW_KIND '=' expr" work
             *	well, never even close.
             */
            if (strcasecmp(SYM_NAME(EXPR_SYM($1)), "kind") == 0) {
                $$ = list1(F95_KIND_SELECTOR_SPEC, $3);
            } else {
                $$ = list2(F_SET_EXPR, $1, $3);
            }
        }

format_spec:
          '*'
        { $$ = NULL; }
        | expr
        ;

io_list_or_null:
        { $$ = NULL; }
        | io_list
        ;


io_list: io_item
        { $$ = list1(LIST,$1); }
        | io_list ',' io_item
        { $$ = list_put_last($1,$3); }
        ;

io_item:
          expr
        | '(' expr ',' io_list ')'
        { $$ = list_cons($2,$4); }
        | '(' expr ',' do_spec ')'
        { $$ = list2(F_IMPLIED_DO,$4,list1(LIST,$2)); }
        | '(' expr ',' io_list ',' do_spec ')'
        { $$ = list2(F_IMPLIED_DO,$6,list_cons($2,$4)); }
        | '(' io_list ',' do_spec ')'
        { $$ = list2(F_IMPLIED_DO,$4,$2); }
        ;

expr:     lhs
        | L_ARRAY_CONSTRUCTOR array_constructor_list R_ARRAY_CONSTRUCTOR
        { $$ = list1(F95_ARRAY_CONSTRUCTOR, $2); }
        | '[' array_constructor_list ']'
        { $$ = list1(F95_ARRAY_CONSTRUCTOR, $2); }
        | '(' expr ')'
        { $$ = $2; }
        | complex_const
        | const
        | expr '+' expr   %prec '+'
        { $$ = list2(F_PLUS_EXPR,$1,$3); }
        | expr '-' expr   %prec '+'
        { $$ = list2(F_MINUS_EXPR,$1,$3); }
        | '+' expr
        { $$ = $2; }
        | '-' expr
        { $$ = list1(F_UNARY_MINUS_EXPR,$2); }
        | expr '*' expr
        { $$ = list2(F_MUL_EXPR,$1,$3); }
        | expr '/' expr
        { $$ = list2(F_DIV_EXPR,$1,$3); }
        | expr POWER expr
        { $$ = list2(F_POWER_EXPR,$1,$3); }
        | expr EQ expr  %prec EQ
        { $$ = list2(F_EQ_EXPR,$1,$3); }
        | expr GT expr  %prec EQ
        { $$ = list2(F_GT_EXPR,$1,$3); }
        | expr LT expr  %prec EQ
        { $$ = list2(F_LT_EXPR,$1,$3); }
        | expr GE expr  %prec EQ
        { $$ = list2(F_GE_EXPR,$1,$3); }
        | expr LE expr  %prec EQ
        { $$ = list2(F_LE_EXPR,$1,$3); }
        | expr NE expr  %prec EQ
        { $$ = list2(F_NE_EXPR,$1,$3); }
        | expr EQV expr
        { $$ = list2(F_EQV_EXPR,$1,$3); }
        | expr NEQV expr
        { $$ = list2(F_NEQV_EXPR,$1,$3); }
        | expr OR expr
        { $$ = list2(F_OR_EXPR,$1,$3); }
        | expr AND expr
        { $$ = list2(F_AND_EXPR,$1,$3); }
        | NOT expr
        { $$ = list1(F_NOT_EXPR,$2); }
        | expr CONCAT expr
        { $$ = list2(F_CONCAT_EXPR,$1,$3); }
        | expr USER_DEFINED_OP expr %prec USER_DEFINED_OP
        { $$ = list3(F95_USER_DEFINED_BINARY_EXPR, $2, $1, $3); }
        | USER_DEFINED_OP expr
        { $$ = list2(F95_USER_DEFINED_UNARY_EXPR, $1, $2); }
	| string_const_substr
        { $$ = $1; }
        ;

lhs:
          IDENTIFIER
        { $$ = $1; }
        | IDENTIFIER image_selector /* coarray */
        { $$ = list2(XMP_COARRAY_REF,$1,$2); }
        | IDENTIFIER parenthesis_arg_list
        { $$ = list2(F_ARRAY_REF,$1,$2); }
        | IDENTIFIER parenthesis_arg_list image_selector /* coarray */
        { $$ = list2(XMP_COARRAY_REF, list2(F_ARRAY_REF,$1,$2), $3); }
        | IDENTIFIER parenthesis_arg_list substring
        { $$ = list2(F_ARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3); }
        | member_ref
        { $$ = $1; }
        | member_ref image_selector /* coarray */
        { $$ = list2(XMP_COARRAY_REF,$1,$2); }
        | member_ref parenthesis_arg_list
        { $$ = list2(F_ARRAY_REF,$1,$2); }
        | member_ref parenthesis_arg_list image_selector /* coarray */
        { $$ = list2(XMP_COARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3); }
        | member_ref parenthesis_arg_list substring
        { $$ = list2(F_ARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3); }
        ;

member_ref:
          IDENTIFIER '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,$1,$3); }
        | IDENTIFIER image_selector '%' IDENTIFIER /* coarray */
        { $$ = list2(F95_MEMBER_REF,list2(XMP_COARRAY_REF,$1,$2),$4); }
        | IDENTIFIER parenthesis_arg_list '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,list2(F_ARRAY_REF,$1,$2),$4); }
        | IDENTIFIER parenthesis_arg_list image_selector '%' IDENTIFIER /* coarray */
        { $$ = list2(F95_MEMBER_REF, list2(XMP_COARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3), $5); }
        | member_ref '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,$1,$3); }
        | member_ref image_selector '%' IDENTIFIER /* coarray */
        { $$ = list2(F95_MEMBER_REF,list2(XMP_COARRAY_REF,$1,$2),$4); }
        | member_ref parenthesis_arg_list '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,list2(F_ARRAY_REF,$1,$2),$4); }
        | member_ref parenthesis_arg_list image_selector '%' IDENTIFIER /* coarray */
        { $$ = list2(F95_MEMBER_REF, list2(XMP_COARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3), $5); }
        ;

lhs_alloc:     /* For allocation list only */
          IDENTIFIER
        { $$ = $1; }
        | IDENTIFIER image_dims_alloc /* coarray */
        { $$ = list2(XMP_COARRAY_REF,$1,$2); }
        | IDENTIFIER parenthesis_arg_list
        { $$ = list2(F_ARRAY_REF,$1,$2); }
        | IDENTIFIER parenthesis_arg_list image_dims_alloc /* coarray */
        { $$ = list2(XMP_COARRAY_REF, list2(F_ARRAY_REF,$1,$2), $3); }
/*         | IDENTIFIER parenthesis_arg_list substring */
/*         { $$ = list2(F_ARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3); } */
        | member_ref_alloc
        { $$ = $1; }
        | member_ref_alloc image_dims_alloc /* coarray */
        { $$ = list2(XMP_COARRAY_REF,$1,$2); }
        | member_ref_alloc parenthesis_arg_list
        { $$ = list2(F_ARRAY_REF,$1,$2); }
        | member_ref_alloc parenthesis_arg_list image_dims_alloc /* coarray */
        { $$ = list2(XMP_COARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3); }
/*         | member_ref_alloc parenthesis_arg_list substring */
/*         { $$ = list2(F_ARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3); } */
        ;

member_ref_alloc:     /* For allocation list only */
          IDENTIFIER '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,$1,$3); }
/*         | IDENTIFIER image_dims_alloc '%' IDENTIFIER /\* coarray *\/ */
/*         { $$ = list2(F95_MEMBER_REF,list2(XMP_COARRAY_REF,$1,$2),$4); } */
        | IDENTIFIER parenthesis_arg_list '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,list2(F_ARRAY_REF,$1,$2),$4); }
/*         | IDENTIFIER parenthesis_arg_list image_dims_alloc '%' IDENTIFIER /\* coarray *\/ */
/*         { $$ = list2(F95_MEMBER_REF, list2(XMP_COARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3), $5); } */
        | member_ref_alloc '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,$1,$3); }
/*         | member_ref_alloc image_dims_alloc '%' IDENTIFIER /\* coarray *\/ */
/*         { $$ = list2(F95_MEMBER_REF,list2(XMP_COARRAY_REF,$1,$2),$4); } */
        | member_ref_alloc parenthesis_arg_list '%' IDENTIFIER
        { $$ = list2(F95_MEMBER_REF,list2(F_ARRAY_REF,$1,$2),$4); }
/*         | member_ref_alloc parenthesis_arg_list image_dims_alloc '%' IDENTIFIER /\* coarray *\/ */
/*         { $$ = list2(F95_MEMBER_REF, list2(XMP_COARRAY_REF,list2(F_ARRAY_REF,$1,$2),$3), $5); } */
        ;

array_constructor_list:
          io_item
        { $$ = list1(LIST, $1); }
        | array_constructor_list  ',' io_item
        { $$ = list_put_last($1, $3); }
        ;

/* reduce/reduce conflict between with complex const,  like (1.2, 3.4).

array_constructor: expr
        | '(' data_var_list ',' do_spec ')'
        { $$ = list2(F_IMPLIED_DO, $4, $2); }
        ;
*/

string_const_substr: const substring
	{
            if (EXPR_CODE($1) != STRING_CONSTANT) {
                error_at_node($1, "not a string constant.");
                $$ = NULL;
            } else {
                $$ = list2(F_STRING_CONST_SUBSTR, $1, $2);
            }
        }

substring:  '(' expr_or_null ':' expr_or_null ')'
        { $$ = list1(LIST, list3(F95_TRIPLET_EXPR,$2,$4,NULL)); }
        ;

expr_or_null: /* empty */
        { $$ = NULL; }
        | expr
        ;

const:    CONSTANT
        | CONSTANT '_' kind_parm
        { $$ = list2(F95_CONSTANT_WITH, $1, $3);  }
        | TRUE_CONSTANT
        { $$ = list0(F_TRUE_CONSTANT); }
        | FALSE_CONSTANT
        { $$ = list0(F_FALSE_CONSTANT); }
        | TRUE_CONSTANT '_' kind_parm
        { $$ = list1(F95_TRUE_CONSTANT_WITH, $3); }
        | FALSE_CONSTANT '_' kind_parm
        { $$ = list1(F95_FALSE_CONSTANT_WITH, $3); }
        ;

kind_parm: CONSTANT
        | IDENTIFIER
        ;

complex_const:  '(' expr ',' expr ')'
        { $$ = list2(COMPLEX_CONSTANT,$2,$4); }
        | '(' '*' ',' expr ')'
        { $$ = list2(COMPLEX_CONSTANT,NULL,$4); }
        | '(' expr ',' '*' ')'
        { $$ = list2(COMPLEX_CONSTANT,$2,NULL); }
        ;

scene_list: scene_range
        { $$ = list1(LIST, $1); }
        | scene_range ',' scene_list
        { $$ = list_cons($1, $3); }
        ;

scene_range: expr
        { $$ = list3(F_SCENE_RANGE_EXPR,$1,NULL,NULL); }
        | expr_or_null ':' expr_or_null
        { $$ = list3(F_SCENE_RANGE_EXPR,NULL,$1,$3); }
        ;

/*
 * OpenMP directives
 */
omp_directive:
	  OMPKW_PARALLEL omp_clause_option
	  { $$ = OMP_LIST(OMP_F_PARALLEL,$2); }
	| OMPKW_END OMPKW_PARALLEL
	  { $$ = OMP_LIST(OMP_F_END_PARALLEL,NULL); }
	| OMPKW_DO omp_clause_option
	  { $$ = OMP_LIST(OMP_F_DO,$2); }
	| OMPKW_END OMPKW_DO omp_nowait_option
	  { $$ = OMP_LIST(OMP_F_END_DO,$3); }
	| OMPKW_PARALLEL OMPKW_DO omp_clause_option
	  { $$ = OMP_LIST(OMP_F_PARALLEL_DO,$3); }
	| OMPKW_END OMPKW_PARALLEL OMPKW_DO omp_nowait_option
	  { $$ = OMP_LIST(OMP_F_END_PARALLEL_DO,$4); }
        | OMPKW_SIMD omp_clause_option
	{ $$ = OMP_LIST(OMP_F_SIMD,$2); }
        | OMPKW_END OMPKW_SIMD
	{ $$ = OMP_LIST(OMP_F_END_SIMD,NULL); }
        | OMPKW_DO OMPKW_SIMD omp_clause_option
	{ $$ = OMP_LIST(OMP_F_DO_SIMD,$3); }
        | OMPKW_END OMPKW_DO OMPKW_SIMD omp_nowait_option
	{ $$ = OMP_LIST(OMP_F_END_DO_SIMD,$4); }
        | OMPKW_DECLARE OMPKW_SIMD omp_clause_option
	{ $$ = OMP_LIST(OMP_F_DECLARE_SIMD,$3); }
        | OMPKW_END OMPKW_DECLARE OMPKW_SIMD
	{ $$ = OMP_LIST(OMP_F_END_DECLARE_SIMD,NULL); }
        | OMPKW_PARALLEL OMPKW_DO OMPKW_SIMD omp_clause_option
	{ $$ = OMP_LIST(OMP_F_PARALLEL_DO_SIMD,$4); }
        | OMPKW_END OMPKW_PARALLEL OMPKW_DO OMPKW_SIMD
	{ $$ = OMP_LIST(OMP_F_END_PARALLEL_DO_SIMD,NULL); }
	| OMPKW_SECTIONS omp_clause_option
	  { $$ = OMP_LIST(OMP_F_SECTIONS,$2); }
	| OMPKW_END OMPKW_SECTIONS omp_nowait_option
	  { $$ = OMP_LIST(OMP_F_END_SECTIONS,$3); }
	| OMPKW_PARALLEL OMPKW_SECTIONS omp_clause_option
	  { $$ = OMP_LIST(OMP_F_PARALLEL_SECTIONS,$3); }
	| OMPKW_END OMPKW_PARALLEL OMPKW_SECTIONS omp_nowait_option
	  { $$ = OMP_LIST(OMP_F_END_PARALLEL_SECTIONS,$4); }
	| OMPKW_SECTION
	  { $$ = OMP_LIST(OMP_F_SECTION,NULL); }
	| OMPKW_SINGLE omp_clause_option
	  { $$ = OMP_LIST(OMP_F_SINGLE,$2); }
	| OMPKW_END OMPKW_SINGLE omp_end_clause_option
	  { $$ = OMP_LIST(OMP_F_END_SINGLE,$3); }
	| OMPKW_MASTER
	  { $$ = OMP_LIST(OMP_F_MASTER,NULL); }
	| OMPKW_END OMPKW_MASTER
	  { $$ = OMP_LIST(OMP_F_END_MASTER,NULL); }
	| OMPKW_CRITICAL
	  { $$ = OMP_LIST(OMP_F_CRITICAL,NULL); }
	| OMPKW_END OMPKW_CRITICAL
	  { $$ = OMP_LIST(OMP_F_END_CRITICAL,NULL); }
	| OMPKW_CRITICAL '(' IDENTIFIER ')'
	  { $$ = OMP_LIST(OMP_F_CRITICAL,list1(LIST,$3)); }
	| OMPKW_END OMPKW_CRITICAL '(' IDENTIFIER ')'
	  { $$ = OMP_LIST(OMP_F_END_CRITICAL,list1(LIST,$4)); }
	| OMPKW_TASK omp_clause_option
	  { $$ = OMP_LIST(OMP_F_TASK,$2); }
	| OMPKW_END OMPKW_TASK omp_nowait_option
	  { $$ = OMP_LIST(OMP_F_END_TASK,NULL); }
	| OMPKW_BARRIER
	  { $$ = OMP_LIST(OMP_F_BARRIER,NULL); }
	| OMPKW_ATOMIC
	  { $$ = OMP_LIST(OMP_F_ATOMIC,NULL); }
	| OMPKW_FLUSH
	  { $$ = OMP_LIST(OMP_F_FLUSH,NULL); }
	| OMPKW_FLUSH '(' omp_list ')'
	  { $$ = OMP_LIST(OMP_F_FLUSH,$3); }
	| OMPKW_ORDERED
	  { $$ = OMP_LIST(OMP_F_ORDERED,NULL); }
	| OMPKW_END OMPKW_ORDERED
	  { $$ = OMP_LIST(OMP_F_END_ORDERED,NULL); }
	| OMPKW_THREADPRIVATE '(' omp_copyin_list ')'
 	  { $$ = OMP_LIST(OMP_F_THREADPRIVATE,$3); } /* NOTE: must be fixed */
	| OMPKW_WORKSHARE
	  { $$ = NULL; }
	| OMPKW_END OMPKW_WORKSHARE omp_nowait_option
	  { $$ = NULL; }
	| OMPKW_PARALLEL OMPKW_WORKSHARE omp_clause_option
	  { $$ = NULL; }
	| OMPKW_END OMPKW_PARALLEL OMPKW_WORKSHARE omp_nowait_option
	  { $$ = NULL; }
	;

omp_nowait_option:
	{ $$ = NULL; }
	| OMPKW_NOWAIT
	{ $$ = OMP_LIST(OMP_DIR_NOWAIT,NULL); }
	;

omp_end_clause_option:
	{ $$ = NULL; }
	| omp_end_clause_list
	;

omp_end_clause_list:
	  omp_end_clause
	 { $$ = list1(LIST,$1); }
	| omp_end_clause_list ',' omp_end_clause
	 { $$ = list_put_last($1,$3); }
	| omp_end_clause_list omp_end_clause
	 { $$ = list_put_last($1,$2); }
	;

omp_end_clause:
	  OMPKW_NOWAIT
	{ $$ = OMP_LIST(OMP_DIR_NOWAIT,NULL); }
	| OMPKW_COPYPRIVATE '(' omp_list ')'
        { $$ = OMP_LIST(OMP_DATA_COPYPRIVATE,$3); }
	;

omp_clause_option:
	{ $$ = NULL; }
	| omp_clause_list
	;

omp_clause_list:
	  omp_clause
	 { $$ = list1(LIST,$1); }
	| omp_clause_list ',' omp_clause
	 { $$ = list_put_last($1,$3); }
	| omp_clause_list omp_clause
	 { $$ = list_put_last($1,$2); }
	;

omp_clause:
	  OMPKW_PRIVATE '(' omp_list ')'
	  { $$ = OMP_LIST(OMP_DATA_PRIVATE,$3); }
	| OMPKW_SHARED '(' omp_list ')'
	  { $$ = OMP_LIST(OMP_DATA_SHARED,$3); }
	| OMPKW_DEFAULT '(' { need_keyword = TRUE; } omp_default_attr ')'
	  { $$ = OMP_LIST(OMP_DATA_DEFAULT,$4); }
	| OMPKW_FIRSTPRIVATE '(' omp_list ')'
	  { $$ = OMP_LIST(OMP_DATA_FIRSTPRIVATE,$3); }
	| OMPKW_LASTPRIVATE '(' omp_list ')'
	  { $$ = OMP_LIST(OMP_DATA_LASTPRIVATE,$3); }
	| OMPKW_COPYIN '(' omp_copyin_list ')'
	  { $$ = OMP_LIST(OMP_DATA_COPYIN,$3); }
	| OMPKW_REDUCTION '(' omp_reduction_op ':' omp_list ')'
	  { $$ = OMP_LIST($3,$5); }
	| OMPKW_IF '(' expr ')'
	  { $$ = OMP_LIST(OMP_DIR_IF,$3); }
	| OMPKW_SCHEDULE '(' { need_keyword = TRUE; } omp_schedule_arg ')'
	  { $$ = $4; }
	| OMPKW_ORDERED
	  { $$ = OMP_LIST(OMP_DIR_ORDERED,NULL); }
        | OMPKW_NUM_THREADS '(' expr ')'
	{ $$ = OMP_LIST(OMP_DIR_NUM_THREADS,$3); }
	| OMPKW_DEPEND '(' omp_depend_op ':' omp_list ')'
	{ $$ = OMP_LIST($3,$5); }
        | OMPKW_FINAL '(' expr ')'
	{ $$ = OMP_LIST(OMP_DATA_FINAL,$3); }
        | OMPKW_UNTIED
	{ $$ = OMP_LIST(OMP_DIR_UNTIED,NULL); }
        | OMPKW_MERGEABLE
	{ $$ = OMP_LIST(OMP_DIR_MERGEABLE,NULL); }
	;

omp_depend_op:
	 OMPKW_DEPEND_IN { $$ = (int) OMP_DATA_DEPEND_IN; }
	| OMPKW_DEPEND_OUT { $$ = (int) OMP_DATA_DEPEND_OUT; }
	| OMPKW_DEPEND_INOUT { $$ = (int) OMP_DATA_DEPEND_INOUT; }
	| IDENTIFIER { $$ = OMP_depend_op($1); }
	;

omp_reduction_op:
	  '+' { $$ = (int) OMP_DATA_REDUCTION_PLUS; }
	| '-' { $$ = (int) OMP_DATA_REDUCTION_MINUS; }
	| '*' { $$ = (int) OMP_DATA_REDUCTION_MUL; }
	| AND { $$ = (int) OMP_DATA_REDUCTION_LOGAND; }
	| OR  { $$ = (int) OMP_DATA_REDUCTION_LOGOR; }
	| EQV { $$ = (int) OMP_DATA_REDUCTION_EQV; }
	| NEQV { $$ = (int) OMP_DATA_REDUCTION_NEQV; }
	| IDENTIFIER { $$ = OMP_reduction_op($1); }
	;

omp_list:
	  IDENTIFIER
	  { $$ = list1(LIST,$1); }
	| omp_list ',' IDENTIFIER
	  { $$ = list_put_last($1,$3); }
	;

/*
omp_common_list:
	  '/' IDENTIFIER '/'
	 { $$ = list1(LIST,list1(LIST,$2)); }
	| omp_common_list ',' '/' IDENTIFIER '/'
	 { $$ = list_put_last($1,list1(LIST,$4)); }
	;
*/
omp_copyin_list:
	  IDENTIFIER
	  { $$ = list1(LIST,$1); }
	| omp_copyin_list ',' IDENTIFIER
	  { $$ = list_put_last($1,$3); }
	| '/' IDENTIFIER '/'
	 { $$ = list1(LIST,list1(LIST,$2)); }
	| omp_copyin_list ',' '/' IDENTIFIER '/'
	 { $$ = list_put_last($1,list1(LIST,$4)); }
	;

omp_schedule_arg:
	  omp_schedule_attr
	  { $$ = OMP_LIST(OMP_DIR_SCHEDULE,OMP_LIST($1,NULL)); }
	| omp_schedule_attr ',' expr
	  { $$ = OMP_LIST(OMP_DIR_SCHEDULE,OMP_LIST($1,$3)); }
	;

omp_schedule_attr:
	  OMPKW_STATIC { $$ = (int) OMP_SCHED_STATIC; }
	| OMPKW_DYNAMIC { $$ = (int) OMP_SCHED_DYNAMIC; }
	| OMPKW_GUIDED  { $$ = (int) OMP_SCHED_GUIDED; }
	| OMPKW_RUNTIME { $$ = (int) OMP_SCHED_RUNTIME; }
	;

omp_default_attr:
	  OMPKW_SHARED { $$ = OMP_LIST(OMP_DEFAULT_SHARED,NULL); }
	| OMPKW_PRIVATE { $$ = OMP_LIST(OMP_DEFAULT_PRIVATE,NULL); }
	| OMPKW_NONE { $$ = OMP_LIST(OMP_DEFAULT_NONE,NULL); }
	;

/*
 * XcalableMP directives
 */
xmp_directive:
	    XMPKW_NODES xmp_nodes_clause
	    { $$ = XMP_LIST(XMP_NODES,$2); }
	  | XMPKW_TEMPLATE xmp_template_clause
	    { $$ = XMP_LIST(XMP_TEMPLATE,$2); }
	  | XMPKW_DISTRIBUTE xmp_distribute_clause
	    { $$ = XMP_LIST(XMP_DISTRIBUTE,$2); }
	  | XMPKW_ALIGN xmp_align_clause
	    { $$ = XMP_LIST(XMP_ALIGN,$2); }
	  | XMPKW_SHADOW xmp_shadow_clause
	    { $$ = XMP_LIST(XMP_SHADOW,$2); }
	  | XMPKW_TEMPLATE_FIX xmp_template_fix_clause
	    { $$ = XMP_LIST(XMP_TEMPLATE_FIX,$2); }
	  | XMPKW_TASK xmp_task_clause
	    { $$ = XMP_LIST(XMP_TASK,$2); }
	  | XMPKW_END xmp_end_clause
	    { $$ = $2; }
	  | XMPKW_TASKS
	    { $$ = XMP_LIST(XMP_TASKS,NULL); }
	  /* | XMPKW_TASKS xmp_NOWAIT */
	  /*   { $$ = XMP_LIST(XMP_TASKS, */
	  /*                   GEN_NODE(INT_CONSTANT, XMP_OPT_NOWAIT)); } */
	  | XMPKW_LOOP { need_keyword = TRUE; } xmp_loop_clause
	    { $$ = XMP_LIST(XMP_LOOP,$3); }
	  | XMPKW_REFLECT xmp_reflect_clause
	    { $$ = XMP_LIST(XMP_REFLECT,$2); }
	  | XMPKW_GMOVE { need_keyword = TRUE; } xmp_gmove_clause
	    { $$ = XMP_LIST(XMP_GMOVE,$3); }
	  | XMPKW_BARRIER { need_keyword = TRUE; } xmp_barrier_clause
	    { $$ = XMP_LIST(XMP_BARRIER,$3); }
	  | XMPKW_REDUCTION xmp_reduction_clause
	    { $$ = XMP_LIST(XMP_REDUCTION,$2); }
	  | XMPKW_BCAST xmp_bcast_clause
	    { $$ = XMP_LIST(XMP_BCAST,$2); }
	  | XMPKW_ARRAY xmp_array_clause
	    { $$ = XMP_LIST(XMP_ARRAY,$2); }
          | XMPKW_LOCAL_ALIAS IDENTIFIER REF_OP IDENTIFIER
	    { $$ = XMP_LIST(XMP_LOCAL_ALIAS, list2(LIST,$2,$4)); }

          | XMPKW_SAVE_DESC xmp_save_desc_clause
	    { $$ = XMP_LIST(XMP_SAVE_DESC, $2); }

          | XMPKW_WAIT_ASYNC xmp_wait_async_clause
            { $$ = XMP_LIST(XMP_WAIT_ASYNC, $2); }

	  | XMPKW_MASTER_IO xmp_master_io_options
	    { $$ = XMP_LIST(XMP_MASTER_IO_BEGIN, $2); }
	  | XMPKW_GLOBAL_IO xmp_global_io_options
	    { $$ = XMP_LIST(XMP_GLOBAL_IO_BEGIN, $2); }

	  | XMPKW_COARRAY xmp_coarray_clause
	    { $$ = XMP_LIST(XMP_COARRAY, $2); }
	  | XMPKW_IMAGE xmp_image_clause
	    { $$ = XMP_LIST(XMP_IMAGE, $2); }
	  ;

xmp_nodes_clause:
	    IDENTIFIER '(' xmp_subscript_list ')'
	      { $$ = list3(LIST,$1,$3,NULL); }
	  | IDENTIFIER '(' xmp_subscript_list ')' '=' '*'
	    { $$ = list3(LIST,$1,$3,XMP_LIST(XMP_NODES_INHERIT_EXEC,NULL)); }
	  | IDENTIFIER '(' xmp_subscript_list ')' '=' xmp_obj_ref
	    { $$ = list3(LIST,$1,$3,XMP_LIST(XMP_NODES_INHERIT_NODES,$6)); }
	  | '(' xmp_subscript_list ')' COL2 xmp_name_list
	      { $$ = list3(LIST,$5,$2,NULL); }
	  | '(' xmp_subscript_list ')' '=' '*' COL2 xmp_name_list
	    { $$ = list3(LIST,$7,$2,XMP_LIST(XMP_NODES_INHERIT_EXEC,NULL)); }
	  | '(' xmp_subscript_list ')' '=' xmp_obj_ref COL2 xmp_name_list
	    { $$ = list3(LIST,$7,$2,XMP_LIST(XMP_NODES_INHERIT_NODES,$5)); }
  	  ;

xmp_template_clause:
	    IDENTIFIER '(' xmp_subscript_list ')'
             { $$=list2(LIST,list1(LIST,$1),$3); }
	  | '(' xmp_subscript_list ')' COL2 xmp_name_list
	     { $$=list2(LIST,$5,$2); }
	  ;

xmp_distribute_clause:
	    IDENTIFIER '(' xmp_dist_fmt_list ')' xmp_ONTO IDENTIFIER
	     { $$ = list3(LIST,list1(LIST,$1),$3,$6); }
	  | '(' xmp_dist_fmt_list ')' xmp_ONTO IDENTIFIER COL2 xmp_name_list
	     { $$ = list3(LIST,$7,$2,$5); }
	  ;

xmp_align_clause:
	    IDENTIFIER '(' xmp_subscript_list ')' xmp_WITH
  	      IDENTIFIER '(' xmp_subscript_list ')'
	    { $$ = list4(LIST,list1(LIST,$1),$3,$6,$8); }
	  | '(' xmp_subscript_list ')' xmp_WITH
  	    IDENTIFIER '(' xmp_subscript_list ')' COL2 xmp_name_list
            { $$ = list4(LIST,$10,$2,$5,$7); }
	  ;

xmp_shadow_clause:
	    IDENTIFIER '(' xmp_subscript_list ')'
	    { $$ = list2(LIST,list1(LIST,$1),$3); }
	  |  '(' xmp_subscript_list ')' COL2 xmp_name_list
            { $$ = list2(LIST,$5,$2); }
          ;

xmp_template_fix_clause:
            IDENTIFIER '(' xmp_subscript_list ')'
	    { $$ = list3(LIST,NULL,$1,$3); }
          | '(' xmp_dist_fmt_list ')' IDENTIFIER
	    { $$ = list3(LIST,$2,$4,NULL); }
          | '(' xmp_dist_fmt_list ')' IDENTIFIER '(' xmp_subscript_list ')'
	    { $$ = list3(LIST,$2,$4,$6); }
          ;

            /* '(' xmp_dist_fmt_list ')' IDENTIFIER '(' xmp_subscript_list ')' */
	    /* { $$ = list3(LIST,$2,$4,$6); } */

xmp_task_clause:
	    xmp_ON xmp_obj_ref KW xmp_nocomm_opt xmp_clause_opt
	    { $$ = list3(LIST,$2,$4,$5); }
          ;

xmp_loop_clause:
	    xmp_ON xmp_obj_ref xmp_reduction_opt xmp_clause_opt
	    { $$ = list4(LIST,NULL,$2,$3,$4); }
	  | '(' xmp_subscript_list ')' xmp_ON xmp_obj_ref
	    	xmp_reduction_opt xmp_clause_opt
	    { $$ = list4(LIST,$2,$5,$6,$7); }
	  ;

/* xmp_reflect_clause: */
/* 	   '(' xmp_expr_list ')' KW xmp_async_opt */
/*            { $$= list3(LIST,$2,NULL,$5); } */
/* 	  |'(' xmp_expr_list ')' xmp_width_opt KW xmp_async_opt */
/*            { $$= list3(LIST,$2,$4,$6); } */
/* 	   ; */

xmp_reflect_clause:
	   '(' xmp_expr_list ')' KW xmp_async_opt xmp_acc_opt
           { $$= list4(LIST,$2,NULL,$5,$6); }
	  |'(' xmp_expr_list ')' KW XMPKW_WIDTH '(' xmp_width_list ')' KW xmp_async_opt xmp_acc_opt
           { $$= list4(LIST,$2,$7,$10,$11); }
	   ;

/* xmp_gmove_clause: */
/* 	     xmp_gmove_opt xmp_clause_opt */
/* 	     { $$ = list2(LIST,$1,$2); } */
/* 	   ; */
/* xmp_gmove_clause: */
/* 	     xmp_gmove_opt KW xmp_async_opt */
/* 	     { $$ = list2(LIST,$1,$3); } */
/* 	   ; */
xmp_gmove_clause:
	    xmp_async_opt xmp_acc_opt
	    { $$ = list3(LIST, GEN_NODE(INT_CONSTANT, XMP_GMOVE_NORMAL), $1, $2); }
	  | XMPKW_IN KW xmp_async_opt xmp_acc_opt
	    { $$ = list3(LIST, GEN_NODE(INT_CONSTANT, XMP_GMOVE_IN), $3, $4); }
          | XMPKW_OUT KW xmp_async_opt xmp_acc_opt
	    { $$ = list3(LIST, GEN_NODE(INT_CONSTANT, XMP_GMOVE_OUT), $3, $4); }
          ;

xmp_barrier_clause:
	     xmp_ON xmp_obj_ref xmp_clause_opt
	      { $$ = list2(LIST,$2,$3); }
	   | xmp_clause_opt
	      { $$ = list2(LIST,NULL,$1); }
	   ;

/* xmp_bcast_clause: */
/*    	     '(' xmp_expr_list ')' xmp_FROM xmp_obj_ref xmp_clause_opt */
/* 	      { $$ = list4(LIST,$2,$5,NULL,$6); } */
/* 	   | '(' xmp_expr_list ')' xmp_ON xmp_obj_ref xmp_clause_opt */
/* 	      { $$ = list4(LIST,$2,NULL,$5,$6); } */
/*    	   | '(' xmp_expr_list ')' xmp_FROM xmp_obj_ref */
/* 	           xmp_ON xmp_obj_ref xmp_clause_opt */
/* 	      { $$ = list4(LIST,$2,$5,$7,$8); } */
/* 	   | '(' xmp_expr_list ')' xmp_clause_opt */
/* 	      { $$ = list4(LIST,$2,NULL,NULL,$4); } */
/*             ; */

xmp_bcast_clause:
   	     '(' xmp_expr_list ')' KW XMPKW_FROM xmp_obj_ref KW xmp_async_opt xmp_acc_opt
	      { $$ = list5(LIST,$2,$6,NULL,$8,$9); }
	   | '(' xmp_expr_list ')' KW XMPKW_ON xmp_obj_ref KW xmp_async_opt xmp_acc_opt
	      { $$ = list5(LIST,$2,NULL,$6,$8,$9); }
   	   | '(' xmp_expr_list ')' KW XMPKW_FROM xmp_obj_ref KW XMPKW_ON xmp_obj_ref KW xmp_async_opt xmp_acc_opt
	      { $$ = list5(LIST,$2,$6,$9,$11,$12); }
	   | '(' xmp_expr_list ')' KW xmp_async_opt xmp_acc_opt
	      { $$ = list5(LIST,$2,NULL,NULL,$5,$6); }
            ;

/* xmp_reduction_clause: */
/* 	       xmp_reduction_spec KW xmp_clause_opt */
/* 	        { $$ = list3(LIST,$1,NULL,$3); } */
/* 	     | xmp_reduction_spec KW xmp_ON xmp_obj_ref KW xmp_clause_opt */
/*                 { $$ = list3(LIST,$1,$4,$6); } */
/* 	     ; */

xmp_reduction_clause:
	       xmp_reduction_spec KW xmp_async_opt xmp_acc_opt
	        { $$ = list4(LIST,$1,NULL,$3, $4); }
	     | xmp_reduction_spec KW xmp_ON xmp_obj_ref KW xmp_async_opt xmp_acc_opt
                { $$ = list4(LIST,$1,$4,$6,$7); }
	     ;

xmp_array_clause:
	     xmp_ON xmp_obj_ref xmp_clause_opt
                { $$ = list2(LIST,$2,$3); }
	     ;

xmp_save_desc_clause:
	    IDENTIFIER
	    { $$ = list1(LIST,$1); }
	  | COL2 xmp_name_list
            { $$ = $2; }
          ;

xmp_wait_async_clause:
	     '(' xmp_expr_list ')' KW XMPKW_ON xmp_obj_ref xmp_clause_opt
	      { $$ = list2(LIST,$2,$6); }
	   | '(' xmp_expr_list ')' KW xmp_clause_opt
	      { $$ = list2(LIST,$2,NULL); }
           ;

xmp_end_clause:
            KW XMPKW_TASK { $$ = XMP_LIST(XMP_END_TASK,NULL); }
          | KW XMPKW_TASKS { $$ = XMP_LIST(XMP_END_TASKS,NULL); }
          | KW XMPKW_MASTER_IO { $$ = XMP_LIST(XMP_END_MASTER_IO,NULL); }
          | KW XMPKW_GLOBAL_IO { $$ = XMP_LIST(XMP_END_GLOBAL_IO,NULL); }
          ;

xmp_obj_ref:
	  '(' xmp_subscript ')'
	   { $$ = list2(LIST,NULL,$2); }
	  | IDENTIFIER '(' xmp_subscript_list ')'
	   { $$ = list2(LIST,$1,$3); }
          | IDENTIFIER
	   { $$ = list2(LIST,$1,NULL); }
          | '*'
	   { $$ = NULL; }
	  ;

xmp_subscript_list:
            xmp_subscript
	  { $$ = list1(LIST,$1); }
	  | xmp_subscript_list ',' xmp_subscript
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_subscript:
	    expr_or_null
	    { $$ = list3(LIST,$1,$1,GEN_NODE(INT_CONSTANT, 0)); }
	  | expr_or_null ':' expr_or_null
	    { $$ = list3(LIST,$1,$3,NULL); }
	  | expr_or_null ':' expr_or_null ':' expr
	    { $$ = list3(LIST,$1,$3,$5); }
	  | '*'
	    { $$ = NULL; }
	  ;

xmp_dist_fmt_list:
            xmp_dist_fmt
	  { $$ = list1(LIST,$1); }
	  | xmp_dist_fmt_list ',' xmp_dist_fmt
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_dist_fmt:
	   '*' { $$ = NULL; }
	  | IDENTIFIER
	    { $$ = list2(LIST,$1,NULL); }
	  | IDENTIFIER '(' expr ')'
	    { $$ = list2(LIST,$1,$3); }
	  | IDENTIFIER '(' '*' ')'
	    { $$ = list2(LIST,$1,NULL); }
	  ;

xmp_reduction_opt:
	 { need_keyword=TRUE; } xmp_reduction_opt1 { $$ = $2; }

xmp_reduction_opt1:
	     /* empty */ { $$ = NULL; }
        | XMPKW_REDUCTION xmp_reduction_spec { $$=$2; }
	;

xmp_reduction_spec:
	'(' xmp_reduction_op ':' xmp_reduction_var_list ')'
	 { $$ = list2(LIST,GEN_NODE(INT_CONSTANT,$2),$4); }
	;

xmp_reduction_op:
	  '+' { $$ = (int) XMP_DATA_REDUCE_SUM; }
	| '*' { $$ = (int) XMP_DATA_REDUCE_PROD; }
	| '-' { $$ = (int) XMP_DATA_REDUCE_SUB; }
	| AND { $$ = (int) XMP_DATA_REDUCE_LAND; }
	| OR  { $$ = (int) XMP_DATA_REDUCE_LOR; }
	| EQV { $$ = (int) XMP_DATA_REDUCE_EQV; }
	| NEQV { $$ = (int) XMP_DATA_REDUCE_NEQV; }
	| IDENTIFIER { $$ = XMP_reduction_op($1); }
	;

xmp_reduction_var_list:
          xmp_reduction_var
	  { $$ = list1(LIST,$1); }
        | xmp_reduction_var_list ',' xmp_reduction_var
	  { $$ = list_put_last($1,$3); }
	;

xmp_reduction_var:
          IDENTIFIER xmp_pos_var_list
	  { $$ = list2(LIST,$1,$2); }
        ;

xmp_pos_var_list:
	     /* empty */ { $$ = NULL; }
        | '/' ident_list '/' { $$=$2; }
	;

/* xmp_gmove_opt: */
/* 	  /\* NULL *\/ { $$= NULL; } */
/* 	 | { need_keyword=TRUE; } XMPKW_IN { $$ = GEN_NODE(INT_CONSTANT, XMP_GMOVE_IN); } */
/* 	 | { need_keyword=TRUE; } XMPKW_OUT { $$ = GEN_NODE(INT_CONSTANT, XMP_GMOVE_OUT); } */
/* 	 ; */

xmp_expr_list:
	  expr
	  { $$ = list1(LIST,$1); }
	  | xmp_expr_list ',' expr
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_name_list:
	  IDENTIFIER
	  { $$ = list1(LIST,$1); }
	  | xmp_name_list ',' IDENTIFIER
	  { $$ = list_put_last($1,$3); }
	  ;

/* xmp_width_opt: */
/*           { need_keyword=TRUE; } xmp_width_opt1 { $$ = $2; } */

/* xmp_width_opt1: */
/*         /\* empty *\/ { $$ = NULL; } */
/*         | XMPKW_WIDTH '(' xmp_width_list ')' */
/*         { $$ = $3; } */
/* 	; */

xmp_width_list:
          xmp_width
	  { $$ = list1(LIST,$1); }
	  | xmp_width_list ',' xmp_width
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_width:
	    expr_or_null
            { $$ = list3(LIST,$1,$1,GEN_NODE(INT_CONSTANT, 0)); }
	  | expr_or_null ':' expr_or_null
            { $$ = list3(LIST,$1,$3,GEN_NODE(INT_CONSTANT, 0)); }
          | XMPKW_PERIODIC expr_or_null
            { $$ = list3(LIST,$2,$2,GEN_NODE(INT_CONSTANT, 1)); }
	  | XMPKW_PERIODIC expr_or_null ':' expr_or_null
            { $$ = list3(LIST,$2,$4,GEN_NODE(INT_CONSTANT, 1)); }
	  ;

/* xmp_async_opt: */
/*           { need_keyword=TRUE; } xmp_async_opt1 { $$ = $2; } */

/* xmp_async_opt1: */
/*         /\* empty *\/ { $$ = NULL; } */
/*         | XMPKW_ASYNC '(' expr ')' */
/*         { $$ = $3; } */
/* 	; */

xmp_async_opt:
        /* empty */ { $$ = NULL; }
        | xmp_ASYNC '(' expr ')'
        { $$ = $3; }
	;

xmp_nocomm_opt:
	  /* NULL */ { $$ = GEN_NODE(INT_CONSTANT, 0); }
	 | XMPKW_NOCOMM { $$ = GEN_NODE(INT_CONSTANT, 1); }
	 ;

xmp_clause_opt:
	   /* NULL */{ $$ = NULL; }
	   | xmp_clause_list
	   ;

xmp_clause_list:
	  xmp_clause_one
	  { $$ = list1(LIST,$1); }
	  | xmp_clause_list xmp_clause_one
	  { $$ = list_put_last($1,$2); }
	  ;

xmp_clause_one:
	    xmp_ASYNC '(' IDENTIFIER ')'
	   { $$ = XMP_LIST(XMP_OPT_ASYNC, $3); }
	   ;

xmp_ON: { need_keyword = TRUE; } XMPKW_ON;
xmp_ONTO: { need_keyword = TRUE; } XMPKW_ONTO;
xmp_WITH: { need_keyword = TRUE; } XMPKW_WITH;
/*xmp_FROM: { need_keyword = TRUE; } XMPKW_FROM;*/
xmp_ASYNC: { need_keyword = TRUE; } XMPKW_ASYNC;
//xmp_NOWAIT: { need_keyword = TRUE; } XMPKW_NOWAIT;
/* xmp_REDUCTION: { need_keyword = TRUE; } XMPKW_REDUCTION; */
/* xmp_MASTER: { need_keyword = TRUE; } XMPKW_MASTER; */

/*
 * (flag, mode)
 *
 *	flag:	1: require an I/O statement.
 *		> 1: require I/O stetements.
 *
 *	mode:	NULL: master I/O.
 *		XMP_GLOBAL_IO_DIRECT: global I/O direct.
 *		XMP_GLOBAL_IO_ATOMIC: global I/O atomic.
 *		XMP_GLOBAL_IO_COLLECTIVE: global I/O collective.
 */
xmp_master_io_options:
	  /* NULL */
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, 1), NULL); }
	  | XMPKW_BEGIN
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, INT_MAX), NULL); }
	  ;

xmp_global_io_options:
	  /* NULL */
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, 1),
			 GEN_NODE(INT_CONSTANT, XMP_GLOBAL_IO_COLLECTIVE)); }
	  | XMPKW_BEGIN
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, INT_MAX),
			 GEN_NODE(INT_CONSTANT, XMP_GLOBAL_IO_COLLECTIVE)); }
	  | XMPKW_ATOMIC
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, 1),
			 GEN_NODE(INT_CONSTANT, XMP_GLOBAL_IO_ATOMIC)); }
	  | XMPKW_ATOMIC XMPKW_BEGIN
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, INT_MAX),
			 GEN_NODE(INT_CONSTANT, XMP_GLOBAL_IO_ATOMIC)); }
	  | XMPKW_DIRECT
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, 1),
			 GEN_NODE(INT_CONSTANT, XMP_GLOBAL_IO_DIRECT)); }
	  | XMPKW_DIRECT XMPKW_BEGIN
	    { $$ = list2(LIST, GEN_NODE(INT_CONSTANT, INT_MAX),
			 GEN_NODE(INT_CONSTANT, XMP_GLOBAL_IO_DIRECT)); }
	  ;

xmp_coarray_clause:
	    xmp_ON IDENTIFIER COL2 xmp_name_list
	     { $$ = list2(LIST,$2,$4); }

xmp_image_clause:
	    '(' IDENTIFIER ')'
	     { $$ = list1(LIST,$2); }

xmp_acc_opt:
	/*null*/
	{ $$ = GEN_NODE(INT_CONSTANT, 0); }
	| KW XMPKW_ACC
	{ $$ = GEN_NODE(INT_CONSTANT, 1); }
	;

/*
 * OpenACC directives
 */
acc_directive:
	  ACCKW_PARALLEL acc_parallel_clause_list
	{ $$ = ACC_LIST(ACC_PARALLEL, $2); }
	| ACCKW_DATA acc_data_clause_list
	{ $$ = ACC_LIST(ACC_DATA, $2); }
	| ACCKW_LOOP acc_loop_clause_list
	{ $$ = ACC_LIST(ACC_LOOP, $2); }
	| ACCKW_KERNELS acc_kernels_clause_list
	{ $$ = ACC_LIST(ACC_KERNELS, $2); }
	| ACCKW_PARALLEL ACCKW_LOOP acc_parallel_loop_clause_list
	{ $$ = ACC_LIST(ACC_PARALLEL_LOOP, $3); }
        | ACCKW_KERNELS ACCKW_LOOP acc_kernels_loop_clause_list
	{ $$ = ACC_LIST(ACC_KERNELS_LOOP, $3); }
	| ACCKW_ATOMIC acc_atomic_clause
	{ $$ = ACC_LIST(ACC_ATOMIC, list1(LIST,$2)); }
	| ACCKW_WAIT acc_wait_clause_list
	{ $$ = ACC_LIST(ACC_WAIT, $2); }
	| ACCKW_WAIT '(' acc_expr_list ')' acc_wait_clause_list
	{ $$ = ACC_LIST(ACC_WAIT, list_cons(ACC_LIST(ACC_CLAUSE_WAIT_ARG, $3), $5)); }
	| ACCKW_CACHE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CACHE, list1(LIST, ACC_LIST(ACC_CLAUSE_CACHE_ARG, $3))); }
	| ACCKW_ROUTINE acc_routine_clause_list
	{ $$ = ACC_LIST(ACC_ROUTINE, $2); }
	| ACCKW_ROUTINE '(' IDENTIFIER ')' acc_routine_clause_list
	{ $$ = ACC_LIST(ACC_ROUTINE, list_cons(ACC_LIST(ACC_CLAUSE_ROUTINE_ARG, $3), $5)); }
	| ACCKW_ENTER ACCKW_DATA acc_enter_data_clause_list
	{ $$ = ACC_LIST(ACC_ENTER_DATA, $3); }
	| ACCKW_EXIT ACCKW_DATA acc_exit_data_clause_list
	{ $$ = ACC_LIST(ACC_EXIT_DATA, $3); }
	| ACCKW_HOST_DATA acc_host_data_clause_list
	{ $$ = ACC_LIST(ACC_HOST_DATA, $2); }
	| ACCKW_DECLARE acc_declare_clause_list
	{ $$ = ACC_LIST(ACC_DECLARE, $2); }
	| ACCKW_UPDATE acc_update_clause_list
	{ $$ = ACC_LIST(ACC_UPDATE_D, $2); }
	| ACCKW_INIT acc_init_clause_list
	{ $$ = ACC_LIST(ACC_INIT, $2); }
	| ACCKW_SHUTDOWN acc_shutdown_clause_list
	{ $$ = ACC_LIST(ACC_SHUTDOWN, $2); }
	| ACCKW_SET acc_set_clause_list
	{ $$ = ACC_LIST(ACC_SET, $2); }
	| ACCKW_END acc_end_clause
	{ $$ = ACC_LIST($2, NULL); }
	;

/* clause separator */
acc_csep:
	      { $$ = NULL; }
	| ',' { $$ = NULL; }
	;

/* clause_lists */
/* ok */
acc_parallel_clause_list:
	{ $$ = list0(LIST); }
	| acc_parallel_clause_list acc_csep acc_parallel_clause
	{ $$ = list_put_last($1, $3); }
        ;
acc_kernels_clause_list:
	{ $$ = list0(LIST); }
	| acc_kernels_clause_list acc_csep acc_kernels_clause
	{ $$ = list_put_last($1, $3); } 
        ;
acc_parallel_loop_clause_list:
        { $$ = list0(LIST); }
	| acc_parallel_loop_clause_list acc_csep acc_parallel_loop_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_kernels_loop_clause_list:
        { $$ = list0(LIST); }
        | acc_kernels_loop_clause_list acc_csep acc_kernels_loop_clause
	{ $$ = list_put_last($1, $3); }
        ;
acc_loop_clause_list:
	{ $$ = list0(LIST); }
	| acc_loop_clause_list acc_csep acc_loop_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_routine_clause_list:
	  acc_routine_clause
	{ $$ = list1(LIST, $1); }
	| acc_routine_clause_list acc_csep acc_routine_clause
	{ $$ = list_put_last($1, $3); }
	;
/* need to rename */
acc_data_clause_list:
	{ $$ = list0(LIST); }
	| acc_data_clause_list acc_csep acc_if_clause
	{ $$ = list_put_last($1, $3); }
	| acc_data_clause_list acc_csep acc_data_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_enter_data_clause_list:
	  acc_enter_data_clause
	{ $$ = list1(LIST, $1); }
	| acc_enter_data_clause_list acc_csep acc_enter_data_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_exit_data_clause_list:
	  acc_exit_data_clause
	{ $$ = list1(LIST, $1); }
	| acc_exit_data_clause_list acc_csep acc_exit_data_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_host_data_clause_list:
	  acc_host_data_clause
	{ $$ = list1(LIST, $1); }
	| acc_host_data_clause_list acc_csep acc_host_data_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_declare_clause_list:
	  acc_declare_clause
	{ $$ = list1(LIST, $1); }
	| acc_declare_clause_list acc_csep acc_declare_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_update_clause_list:
	  acc_update_clause
	{ $$ = list1(LIST, $1); }
	| acc_update_clause_list acc_csep acc_update_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_wait_clause_list:	
        { $$ = list0(LIST); }
	| acc_wait_clause_list acc_csep acc_async_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_init_clause_list:
	{ $$ = list0(LIST); }
	| acc_init_clause_list acc_csep acc_init_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_shutdown_clause_list:
	{ $$ = list0(LIST); }
	| acc_shutdown_clause_list acc_csep acc_shutdown_clause
	{ $$ = list_put_last($1, $3); }
	;
acc_set_clause_list:
	  acc_set_clause
	{ $$ = list1(LIST, $1); }
	| acc_set_clause_list acc_csep acc_set_clause
	{ $$ = list_put_last($1, $3); }
	;

/*****************************/
/* OpenACC directive clauses */
/*****************************/
acc_compute_clause:
	  acc_async_clause
	| acc_wait_clause
	| acc_num_gangs_clause
	| acc_num_workers_clause
	| acc_vector_length_clause
	| acc_if_clause
	| acc_data_clause
	| acc_default_clause
	;
acc_parallel_clause:
	  acc_compute_clause
	| acc_reduction_clause
	| acc_private_clause
	| acc_firstprivate_clause
	| acc_device_type_clause
	;
acc_kernels_clause:
	  acc_compute_clause
	| acc_device_type_clause
	;
acc_loop_clause:
	  acc_collapse_clause
	| acc_gang_clause
	| acc_worker_clause
	| acc_vector_clause
	| acc_seq_clause
	| acc_auto_clause
	| acc_tile_clause
	| acc_device_type_clause
	| acc_independent_clause
	| acc_private_clause
	| acc_reduction_clause
	;
acc_host_data_clause:
	  acc_use_device_clause
	;
acc_init_clause:
	  acc_device_type_clause
	| acc_device_num_clause
	;
acc_shutdown_clause:
	  acc_init_clause
	;
acc_parallel_loop_clause:
	  acc_compute_clause
	| acc_loop_clause
	| acc_firstprivate_clause
	;
acc_kernels_loop_clause:
	  acc_compute_clause
	| acc_loop_clause
	;
acc_routine_clause:
	  acc_gang_clause
	| acc_worker_clause
	| acc_vector_clause
	| acc_seq_clause
	| acc_bind_clause
	| acc_device_type_clause
	| acc_nohost_clause
	;
acc_enter_data_clause:
	  acc_if_clause
	| acc_async_clause
	| acc_wait_clause
	| acc_copyin_clause
	| acc_create_clause
	| acc_present_or_copyin_clause
	| acc_present_or_create_clause
	;
acc_exit_data_clause:
	  acc_if_clause
	| acc_async_clause
	| acc_wait_clause
	| acc_copyout_clause
	| acc_delete_clause
	| acc_finalize_clause
	;
acc_set_clause:
	  acc_default_async_clause
	| acc_device_type_clause
	| acc_device_num_clause
	;
acc_atomic_clause:
				{ $$ = NULL; }
	| KW ACCKW_READ		{ $$ = ACC_LIST(ACC_CLAUSE_READ, NULL); }
	| KW ACCKW_WRITE	{ $$ = ACC_LIST(ACC_CLAUSE_WRITE, NULL); }
	| KW ACCKW_UPDATE	{ $$ = ACC_LIST(ACC_CLAUSE_UPDATE, NULL); }
	| KW ACCKW_CAPTURE	{ $$ = ACC_LIST(ACC_CLAUSE_CAPTURE, NULL); }
	;
acc_data_clause:
	  acc_copy_clause
	| acc_copyin_clause
	| acc_copyout_clause
	| acc_create_clause
	| acc_present_clause
	| acc_present_or_copy_clause
	| acc_present_or_copyin_clause
	| acc_present_or_copyout_clause
	| acc_present_or_create_clause
	| acc_deviceptr_clause
	;
acc_update_clause:
	  acc_async_clause
	| acc_wait_clause
	| acc_device_type_clause
	| acc_if_clause
	| acc_host_clause
	| acc_device_clause
	| acc_if_present_clause
	;
acc_declare_clause:
	  acc_data_clause
	| acc_device_resident_clause
	| acc_link_clause
	;
acc_end_clause:
	  ACCKW_PARALLEL		{ $$ = ACC_END_PARALLEL; }
	| ACCKW_KERNELS			{ $$ = ACC_END_KERNELS; }
	| ACCKW_DATA			{ $$ = ACC_END_DATA; }
	| ACCKW_HOST_DATA		{ $$ = ACC_END_HOST_DATA; }
	| ACCKW_ATOMIC			{ $$ = ACC_END_ATOMIC; }
	| ACCKW_PARALLEL ACCKW_LOOP	{ $$ = ACC_END_PARALLEL_LOOP; }
	| ACCKW_KERNELS  ACCKW_LOOP	{ $$ = ACC_END_KERNELS_LOOP; }
	;

/*******************/
/* OpenACC clauses */
/*******************/
acc_async_clause:
	  ACCKW_ASYNC
	{ $$ = ACC_LIST(ACC_CLAUSE_ASYNC, NULL); }
	| ACCKW_ASYNC '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_ASYNC, $3); }
	;
acc_wait_clause:
	  ACCKW_WAIT
	{ $$ = ACC_LIST(ACC_CLAUSE_WAIT, NULL); }
	| ACCKW_WAIT '(' acc_expr_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_WAIT, $3); }
	;
acc_device_type_clause:
	  ACCKW_DEVICE_TYPE '(' acc_id_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DEVICE_TYPE, $3); }
	;
acc_num_gangs_clause:
	  ACCKW_NUM_GANGS '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_NUM_GANGS, $3); }
	;
acc_num_workers_clause:
	  ACCKW_NUM_WORKERS '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_NUM_WORKERS, $3); }
	;
acc_vector_length_clause:
	  ACCKW_VECTOR_LENGTH '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_VECTOR_LENGTH, $3); }
	;
acc_reduction_clause:
	  ACCKW_REDUCTION '(' acc_reduction_op ':' acc_id_list ')'
	{ $$ = ACC_LIST($3, $5); }
	;
acc_reduction_op:
	  '+'		{ $$ = ACC_CLAUSE_REDUCTION_PLUS; }
	| '*'	       	{ $$ = ACC_CLAUSE_REDUCTION_MUL; }
	| AND		{ $$ = ACC_CLAUSE_REDUCTION_LOGAND; }
	| OR		{ $$ = ACC_CLAUSE_REDUCTION_LOGOR; }
	| EQV		{ $$ = ACC_CLAUSE_REDUCTION_EQV; }
	| NEQV		{ $$ = ACC_CLAUSE_REDUCTION_NEQV; }
	| IDENTIFIER	{ $$ = ACC_reduction_op($1); }  
	;
/*
	| ACCKW_REDUCTION_MAX	  { $$ = ACC_CLAUSE_REDUCTION_MAX; }
	| ACCKW_REDUCTION_MIN	  { $$ = ACC_CLAUSE_REDUCTION_MIN; }
	| ACCKW_REDUCTION_BITAND  { $$ = ACC_CLAUSE_REDUCTION_BITAND; }
	| ACCKW_REDUCTION_BITOR	  { $$ = ACC_CLAUSE_REDUCTION_BITOR; }
	| ACCKW_REDUCTION_BITXOR  { $$ = ACC_CLAUSE_REDUCTION_BITXOR; }
*/
acc_private_clause:
	  ACCKW_PRIVATE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_PRIVATE, $3); }
	;
acc_firstprivate_clause:
	  ACCKW_FIRSTPRIVATE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_FIRSTPRIVATE, $3); }
	;
acc_default_clause:
	  ACCKW_DEFAULT '(' acc_default_clause_arg ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DEFAULT, $3); }
	;
acc_default_clause_arg:
	  KW ACCKW_NONE     { $$ = ACC_LIST(ACC_CLAUSE_NONE, NULL); }
	| KW ACCKW_PRESENT  { $$ = ACC_LIST(ACC_CLAUSE_PRESENT, NULL); }
	;
acc_bind_clause:
	  ACCKW_BIND '(' IDENTIFIER ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_BIND, $3); }
	| ACCKW_BIND '(' CONSTANT ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_BIND, $3); }
	;
acc_nohost_clause:
	  ACCKW_NOHOST
	{ $$ = ACC_LIST(ACC_CLAUSE_NOHOST, NULL); }
	;
acc_collapse_clause:
	  ACCKW_COLLAPSE '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_COLLAPSE, $3); }
	;
acc_gang_clause:
	  ACCKW_GANG
	{ $$ = ACC_LIST(ACC_CLAUSE_GANG, NULL); }
	| ACCKW_GANG '(' acc_gang_arg_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_GANG, $3); }
	;
acc_gang_arg_list:
	  acc_gang_arg
	{ $$ = list1(LIST, $1); }
	| acc_gang_arg ',' acc_gang_arg
	{ $$ = list2(LIST, $1, $3); 
	  if((EXPR_CODE($1) != ACC_PRAGMA && EXPR_CODE($3) != ACC_PRAGMA)
	  || (EXPR_CODE($1) == ACC_PRAGMA && EXPR_CODE($3) == ACC_PRAGMA)){
	    error("gang has over one num or one static argument");
	  }
	}
	;

acc_gang_arg:
	  expr
	{ $$ = $1; }
/*
	| IDENTIFIER ':' '*'
	{ $$ = ACC_LIST(ACC_num_attr($1), NULL); ACC_check_num_attr($1, ACC_STATIC); }
*/
	| IDENTIFIER ':' acc_size_expr
	{ 
	  if(ACC_num_attr($1) == ACC_CLAUSE_STATIC){
	    $$ = ACC_LIST(ACC_num_attr($1), $3);
	  }else{
	    $$ = $3;
	  }
        }
	;
acc_worker_clause:
	  ACCKW_WORKER
	{ $$ = ACC_LIST(ACC_CLAUSE_WORKER, NULL); }
	| ACCKW_WORKER '(' acc_num_expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_WORKER, $3); }
	;
acc_num_expr:
	  expr
	| IDENTIFIER ':' expr
	{ $$ = $3; ACC_check_num_attr($1, ACC_CLAUSE_NUM_WORKERS); }
	;
acc_vector_clause:
	  ACCKW_VECTOR
	{ $$ = ACC_LIST(ACC_CLAUSE_VECTOR, NULL); }
	| ACCKW_VECTOR '(' acc_length_expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_VECTOR, $3); } 
	;
acc_length_expr: 
	  expr
	| IDENTIFIER ':' expr
	{ $$ = $3; ACC_check_num_attr($1, ACC_CLAUSE_VECTOR_LENGTH); }
	;
acc_seq_clause:
	  ACCKW_SEQ
	{ $$ = ACC_LIST(ACC_CLAUSE_SEQ, NULL); }
	;
acc_auto_clause:
	  ACCKW_AUTO
	{ $$ = ACC_LIST(ACC_CLAUSE_AUTO, NULL); }
	;
acc_tile_clause:
	  ACCKW_TILE '(' acc_size_expr_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_TILE, $3); }
	;
acc_independent_clause:
	  ACCKW_INDEPENDENT
	{ $$ = ACC_LIST(ACC_CLAUSE_INDEPENDENT, NULL); }
	;
acc_if_clause:
	  ACCKW_IF '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_IF,$3); }
	;
acc_copy_clause:
	  ACCKW_COPY '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_COPY, $3); }
	;
acc_copyin_clause:
	  ACCKW_COPYIN '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_COPYIN, $3); }
	;
acc_copyout_clause:
	  ACCKW_COPYOUT '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_COPYOUT, $3); }
	;
acc_create_clause:
	  ACCKW_CREATE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_CREATE, $3); }
	;
acc_present_clause:
	  ACCKW_PRESENT '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_PRESENT, $3); }
	;
acc_present_or_copy_clause:
	  ACCKW_PRESENT_OR_COPY '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_PRESENT_OR_COPY, $3); }
	;
acc_present_or_copyin_clause:
	  ACCKW_PRESENT_OR_COPYIN '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_PRESENT_OR_COPYIN, $3); }
	;
acc_present_or_copyout_clause:
	  ACCKW_PRESENT_OR_COPYOUT '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_PRESENT_OR_COPYOUT, $3); }
	;
acc_present_or_create_clause:
	  ACCKW_PRESENT_OR_CREATE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_PRESENT_OR_CREATE, $3); }
	;
acc_deviceptr_clause:
	  ACCKW_DEVICEPTR '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DEVICEPTR, $3); }
	;
acc_delete_clause:
	  ACCKW_DELETE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DELETE, $3); }
	;
acc_finalize_clause:
	  ACCKW_FINALIZE
	{ $$ = ACC_LIST(ACC_CLAUSE_FINALIZE, NULL); }
	;
acc_use_device_clause:
	  ACCKW_USE_DEVICE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_USE_DEVICE, $3); }
	;
acc_device_resident_clause:
	  ACCKW_DEVICE_RESIDENT '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DEVICE_RESIDENT, $3); }
	;
acc_link_clause:
	  ACCKW_LINK '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_LINK, $3); }
	;
acc_host_clause:
	  ACCKW_HOST '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_HOST, $3); }
	;
acc_device_clause:
	  ACCKW_DEVICE '(' acc_var_list ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DEVICE, $3); }
	;
acc_if_present_clause:
	  ACCKW_IF_PRESENT
	{ $$ = ACC_LIST(ACC_CLAUSE_IF_PRESENT, NULL); }
	;
acc_device_num_clause:
	  ACCKW_DEVICE_NUM '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DEVICE_NUM, $3); }
	;
acc_default_async_clause:
	  ACCKW_DEFAULT_ASYNC '(' expr ')'
	{ $$ = ACC_LIST(ACC_CLAUSE_DEFAULT_ASYNC, $3); }
	;

/***********************/
/* OpenACC other rules */
/***********************/
/* var-name, array-name, subarray, or common-block-name */
acc_var:
	  IDENTIFIER
	| '/' IDENTIFIER '/'
	{ $$ = ACC_LIST(ACC_COMMONBLOCK, $2); }
	| IDENTIFIER '(' acc_subscript_list ')'
	{ $$ = list2(F_ARRAY_REF, $1, $3); }
	;
acc_subscript_list:
	  acc_subscript
	{ $$ = list1(LIST, $1); }
	| acc_subscript_list ',' acc_subscript
	{ $$ = list_put_last($1, $3); }
	;
acc_subscript:
	  expr
	| expr_or_null ':' expr_or_null
	{ $$ = list3(F95_TRIPLET_EXPR,$1,$3,NULL); }
	;
/* list of var-name, array-name, subarray, or common-block-name */
acc_var_list:
	  acc_var
	{ $$ = list1(LIST, $1); }
	| acc_var_list ',' acc_var
	{ $$ = list_put_last($1, $3); }
	;
acc_id_list:
	  IDENTIFIER
	{ $$ = list1(LIST, $1); }
	| acc_id_list ',' IDENTIFIER
	{ $$ = list_put_last($1, $3); }
	;
acc_expr_list:
	  expr
	{ $$ = list1(LIST, $1); }
	| acc_expr_list ',' expr
	{ $$ = list_put_last($1, $3); }
	;
acc_size_expr:
	  expr
	{ $$ = $1; }
	| '*'
	{ $$ = ACC_LIST(ACC_ASTERISK, NULL); }
	;
acc_size_expr_list:
	  acc_size_expr
	{ $$ = list1(LIST, $1); }
	| acc_size_expr_list ',' acc_size_expr
	{ $$ = list_put_last($1, $3); }
	;

  
%%
#include "F95-lex.c"

/* EOF */
