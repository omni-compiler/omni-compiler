/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
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
%token OMPKW_END
%token OMPKW_PRIVATE
%token OMPKW_SHARED
%token OMPKW_DEFAULT
%token OMPKW_NONE
%token OMPKW_FIRSTPRIVATE
%token OMPKW_REDUCTION
%token OMPKW_IF
%token OMPKW_COPYIN
%token OMPKW_DO
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

%type <val> omp_directive omp_nowait_option omp_clause_option omp_clause_list omp_clause omp_list omp_common_list omp_default_attr omp_copyin_list omp_schedule_arg
%type <code> omp_schedule_attr omp_reduction_op

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

/* statement name */
expr st_name;

%}

%type <val> statement label 
%type <val> expr /*expr1*/ lhs member_ref lhs_alloc member_ref_alloc substring expr_or_null complex_const array_constructor_list
%type <val> program_name dummy_arg_list dummy_args dummy_arg file_name
%type <val> declaration_statement executable_statement action_statement action_statement_let action_statement_key assign_statement_or_null assign_statement
%type <val> declaration_list entity_decl type_spec type_spec0 length_spec common_decl
%type <val> common_block external_decl intrinsic_decl equivalence_decl
%type <val> cray_pointer_list cray_pointer_pair cray_pointer_var
%type <val> equiv_list data data_list data_val_list data_val value simple_value save_list save_item const_list const_item common_var data_var data_var_list image_dims image_dim_list image_dim image_dims_alloc image_dim_list_alloc image_dim_alloc dims dim_list dim ubound label_list implicit_decl imp_list letter_group letter_groups namelist_decl namelist_entry namelist_list ident_list access_ident_list access_ident
%type <val> do_spec arg arg_list parenthesis_arg_list image_selector cosubscript_list
%type <val> set_expr
%type <val> io_statement format_spec ctl_list io_clause io_list_or_null io_list io_item
%type <val> IDENTIFIER CONSTANT const kind_parm GENERIC_SPEC USER_DEFINED_OP

%type <val> name name_or_null generic_name defined_operator func_prefix prefix_spec
%type <val> declaration_statement95 attr_spec_list attr_spec access_spec
%type <val> intent_spec kind_selector kind_or_len_selector char_selector len_key_spec len_spec kind_key_spec array_allocation_list  array_allocation defered_shape_list defered_shape
%type <val> result_opt type_keyword
%type <val> action_statement95
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
          PROGRAM program_name
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
        | ENDINTERFACE
          { $$ = list1(F95_ENDINTERFACE_STATEMENT,NULL); }
        | MODULEPROCEDURE ident_list
          { $$ = list1(F95_MODULEPROCEDURE_STATEMENT, $2); }
        | BLOCKDATA program_name
          { $$ = list1(F_BLOCK_STATEMENT,$2); }
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

defined_operator: '.'
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
        | INTRINSIC intrinsic_decl
        { $$ = list1(F_INTRINSIC_DECL,$2); }
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
        { $$ = list5(LIST,$1,$2,$4,$6,$3); }
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
        { $$ = list2 (LIST, GEN_NODE(F_TYPE_NODE,TYPE_DREAL), NULL); }
        | KW_DCOMPLEX   
        { $$ = list2 (LIST, GEN_NODE(F_TYPE_NODE,TYPE_DCOMPLEX), NULL); }
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

namelist_decl:        namelist_entry
        { $$ = list1(LIST,$1); }
        | namelist_decl namelist_entry
        { $$ = list_put_last($1,$2); }
        ;

namelist_entry:  '/' IDENTIFIER '/' namelist_list
        { $$ = list2(LIST,$2,$4); }
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
        | WHERE '(' expr ')' assign_statement_or_null
        { $$ = list2(F_WHERE_STATEMENT,$3,$5); }
        | ELSEWHERE
        { $$ = list0(F_ELSEWHERE_STATEMENT); }
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

comma_or_null:
        | ','
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
        { $$ = list2(F_SET_EXPR, $1, $3); }

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
	| OMPKW_END OMPKW_SINGLE omp_nowait_option
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
	| OMPKW_THREADPRIVATE '(' omp_common_list ')'
	  { $$ = OMP_LIST(OMP_F_THREADPRIVATE,$3); }
	;

omp_nowait_option:
	{ $$ = NULL; }
	| OMPKW_NOWAIT
	{ $$ = OMP_LIST(OMP_DIR_NOWAIT,NULL); }
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

omp_common_list:
	  '/' IDENTIFIER '/'
	 { $$ = list1(LIST,list1(LIST,$2)); }
	| omp_common_list ',' '/' IDENTIFIER '/'
	 { $$ = list_put_last($1,list1(LIST,$4)); }
	;

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
	| OMPKW_NONE { $$ = $$ = OMP_LIST(OMP_DEFAULT_NONE,NULL); }
	;

/* 
 * XcalableMP directives 
 */
xmp_directive:
	    XMPKW_NODES IDENTIFIER '(' xmp_node_spec_list ')'
	  | XMPKW_NODES IDENTIFIER '(' xmp_node_spec_list ')' '=' '*'
	  | XMPKW_NODES IDENTIFIER '(' xmp_node_spec_list ')' '=' xmp_node_ref
	  | XMPKW_TEMPLATE IDENTIFIER '(' xmp_subscript_list ')'
	  | XMPKW_DISTRIBUTE IDENTIFIER '(' xmp_dist_fmt_list ')' 
  	    XMPKW_ON IDENTIFIER
	  | XMPKW_ALIGN IDENTIFIER '(' xmp_align_subscript_list ')' 
  	    XMPKW_WITH IDENTIFIER '(' xmp_align_subscript_list ')' 
	  | XMPKW_SHADOW IDENTIFIER '(' xmp_shadow_width_list ')' 
	  | XMPKW_TEMPLATE_FIX  /* not yet*/
	  | XMPKW_TASK XMPKW_ON xmp_node_ref
	  | XMPKW_TASK XMPKW_ON xmp_template_ref
	  | XMPKW_TASKS 
	  | XMPKW_TASKS XMPKW_NOWAIT
	  | XMPKW_LOOP XMPKW_ON xmp_on_ref xmp_reduction_opt
	  | XMPKW_LOOP '(' xmp_subscript_list ')' XMPKW_ON xmp_on_ref
	        xmp_reduction_opt
	  | XMPKW_REFLECT '(' xmp_expr_list ')' xmp_async_opt
	  | XMPKW_GMOVE  xmp_async_opt
	  | XMPKW_GMOVE  xmp_gmove_opt xmp_async_opt
	  | XMPKW_BARRIER 
	  | XMPKW_BARRIER XMPKW_ON xmp_on_ref
	  | XMPKW_REDUCTION 
	  | XMPKW_BCAST '(' xmp_expr_list ')' xmp_async_opt
	  | XMPKW_COARRAY
	  ;

xmp_node_spec_list: 
            xmp_node_spec
	  { $$ = list1(LIST,$1); }
	  | xmp_node_spec_list ',' xmp_node_spec
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_node_spec:
	   expr
	  | '*'
	  ; 

xmp_nodes_ref:
	  '(' xmp_script ')
	  | IDENTIFIER '(' xmp_script_list ')'
	  ;
		
xmp_subscript_list: 
            xmp_subscript
	  { $$ = list1(LIST,$1); }
	  | xmp_subscript_list ',' xmp_subscript
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_subscript:
	    expr_or_null
	  | expr_or_null ':' expr_or_null
	  | expr_or_null ':' expr_or_null ': expr
	  | '*'
	  ;

xmp_dist_fmt_list:
           {keyword_required = TURE;} xmp_dist_fmt
	  { $$ = list1(LIST,$1); }
	  | xmp_dist_fmt_list ',' {keyword_required = TURE;} xmp_dist_fmt
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_dist_fmt:
	   '*'
	  | XMPKW_BLOCK
	  | XMPKW_CYCLIC
	  | XMPKW_CYCLIC '(' expr ')'
	  | XMPKW_GBLOCK '(' '*' ')'
	  | XMPKW_GBLOCK '(' expr ')'
	  ;

xmp_align_subscript_list:
           xmp_align_subscript
	  { $$ = list1(LIST,$1); }
	  | xmp_align_subscrpt_list ',' xmp_align_subscript
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_align_subscript:
	   expr
	  | '*'
	  | ':'
	  ;

xmp_shadow_width_list:
           xmp_shadow_width
	  { $$ = list1(LIST,$1); }
	  | xmp_shadow_width_list ',' xmp_shadow_width
	  { $$ = list_put_last($1,$3); }
	  ;

xmp_shadow_width:
	    expr
	  | expr ':' expr 
	  | '*'
	  ;

xmp_on_ref:
	     xmp_node_ref
	   | xmp_template_ref
	   :

xmp_reduction_opt:
	     /* empty */
        | XMPKW_REDUCTION xmp_reduction_spec
	;

xmp_reduction_spec:
	'(' xmp_redution_op ':' xmp_expr_list ')'
	;

xmp_reduction_op:
	  '+' { $$ = (int) XMP_DATA_REDUCTION_PLUS; }
	| '-' { $$ = (int) XMP_DATA_REDUCTION_MINUS; }
	| '*' { $$ = (int) XMP_DATA_REDUCTION_MUL; }
	| AND { $$ = (int) XMP_DATA_REDUCTION_LOGAND; }
	| OR  { $$ = (int) XMP_DATA_REDUCTION_LOGOR; }
	| EQV { $$ = (int) XMP_DATA_REDUCTION_EQV; }
	| NEQV { $$ = (int) XMP_DATA_REDUCTION_NEQV; }
	| IDENTIFIER { $$ = XMP_reduction_op($1); }
	;

xmp_expr_list:
	  expr
	  { $$ = list1(LIST,$1); }
	  | xmp_expr_list ',' expr
	  { $$ = list_put_last($1,$3); }
	  ;



%%
#include "F95-lex.c"

/* EOF */
