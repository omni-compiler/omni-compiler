%{
/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
 
#include "c-expr.h"
#include "c-lexyacc.h"
#include "c-option.h"
#include "c-pragma.h"

#define YYDEBUG 1

int s_isParsing = 0;
PRIVATE_STATIC const CExprCodeEnum s_CUnaryOpeEnumToExprCodeEnum[]     = CUnaryOpeEnumToExprCodeEnumDef;
PRIVATE_STATIC const CExprCodeEnum s_CAssignEnumToExprCodeEnum[]       = CAssignEnumToExprCodeEnumDef;

%}

%expect 0

%start start

%union {
    CExpr               *expr;
    CTypeSpecEnum       typespec;
    CAssignEnum         assigntype;
    CUnaryOpeEnum       unop;
}

%token IDENTIFIER   TYPENAME
%token SCSPEC       STATIC      TYPESPEC        TYPEQUAL
%token CONSTANT     STRING      ELLIPSIS        COLON_SQBRACKET
%token ENUM         STRUCT      UNION
%token IF           ELSE        WHILE           DO
%token FOR          SWITCH      CASE            DEFAULT
%token BREAK        CONTINUE    RETURN          GOTO
%token SIZEOF
%token ARITH_LE     ARITH_GE    ARITH_EQ        ARITH_NE

 /* PRAGMA_ARG???*/
%token PRAGMA      PRAGMA_ARG   DIRECTIVE       PRAGMA_PACK
%token PRAGMA_EXEC PRAGMA_PREFIX
%token <expr> XMP_COARRAY_DECLARATION   XMP_CRITICAL    XMP_FUNC_CALL
%token XMP_DESC_OF

/* gcc */
%token ASSEMBLER    LABEL       REALPART     IMAGPART
%token ATTRIBUTE    EXTENSION   TYPEOF       ALIGNOF
%token BLTIN_OFFSETOF           BLTIN_VA_ARG
%token BLTIN_TYPES_COMPATIBLE_P

%nonassoc IF
%nonassoc ELSE


%left  OROR
%left  ANDAND
%left  '|'
%left  '^'
%left  '&'
%left  '<' '>' ARITH_LE ARITH_GE ARITH_EQ ARITH_NE
%left  LSHIFT RSHIFT
%left  '+' '-'
%left  '*' '/' '%'

%left  HYPERUNARY
%left  POINTSAT '.' '(' '['
%right ASSIGN '='
%right '?' ':'
%right UNARY PLUSPLUS MINUSMINUS

%type <unop>          unop
%type <assigntype>    ASSIGN

%type <expr> IDENTIFIER CONSTANT STRING TYPENAME SCSPEC TYPEQUAL TYPESPEC
%type <expr> STATIC STRUCT UNION ENUM LABEL DEFAULT
%type <expr> DIRECTIVE PRAGMA_ARG PRAGMA_PACK
%type <expr> PRAGMA_EXEC PRAGMA_PREFIX

%type <expr> ident idents string
%type <expr> typename label_idents
%type <expr> typespec_na typespec_a
%type <expr> typespec_reserved_na typespec_reserved_a
%type <expr> typespec_nonreserved_na
%type <expr> designator designators array_designator
%type <expr> primary primary_1 expr exprs opt_exprs cast_expr unary_expr
%type <expr> mul_expr plus_expr shift_expr comp_expr compeq_expr
%type <expr> band_expr bxor_expr bor_expr logand_expr logor_expr cond_expr
%type <expr> array_ref array_dimension
%type <expr> coarray_ref coarray_dimensions coarray_dimension coarray_dimension_1
%type <expr> coarray_declarations coarray_declaration coarray_declaration_1
%type <expr> declspecs_CTAA declspecs_CTAa declspecs_CTaA declspecs_CTaa
%type <expr> declspecs_CtAA declspecs_CtAa declspecs_CtaA declspecs_Ctaa
%type <expr> declspecs_cTAA declspecs_cTAa declspecs_cTaA declspecs_cTaa
%type <expr> declspecs_ctAA declspecs_ctAa declspecs_ctaA declspecs_ctaa
%type <expr> declspecs_xtxx declspecs_xTxx declspecs_xtAx declspecs_xTAx
%type <expr> declspecs_Ctxx declspecs_CTxx declspecs_Cxxx declspecs
%type <expr> scspec opt_typequals_a

%type <expr> initdecls notype_initdecls initdecl initdecl_1 notype_initdecl notype_initdecl_1
%type <expr> init opt_initializers_opt_comma initializers initializer initval

%type <expr> compstmt compstmt_end compstmt_expr_start compstmt_contents_nonempty
%type <expr> stmt stmt_1 label label_1 opt_labels stmt_nocomp

%type <expr> block_start block_labeled_stmt
%type <expr> declarator nt_declarator after_type_declarator
%type <expr> codeclarator nt_codeclarator
%type <expr> param_declarator param_declarator_ts param_declarator_nts
%type <expr> param_codeclarator
%type <expr> array_declarator
%type <expr> param param_head inner_params params params_na inner_params_1
%type <expr> params_or_idents params_or_idents_na

%type <expr> structsp_a structsp_na
%type <expr> tagname struct_a union_a enum_a
%type <expr> member_decls member_decls_1
%type <expr> member_decl members members_notype member_declarator
%type <expr> member_nt_declarator
%type <expr> enumerators enumerator
%type <expr> absdecl opt_absdecl absdecl_a absdecl_na direct_absdecl absdecl_opt_a
%type <expr> if_stmt while_stmt do_stmt for_stmt for_init_stmt switch_stmt
%type <expr> for_cond_expr for_incr_expr condition condition_1

%type <expr> ext_def ext_defs func_def data_def decl decl_1 datadecl datadecl_1 datadecls
%type <expr> opt_datadecls opt_label_decls label_decls label_decl
%type <expr> stmts_and_decls nested_func notype_nested_func
%type <expr> stmts_decls_labels_es stmts_decls_labels_ed
%type <expr> stmts_decls_labels_el stmts_decls_labels_ep
%type <expr> compstmt_or_error stmts_decls_labels stmts_decls_labels_error errstmt

%type <expr> asm_expr opt_asm_expr asm_stmt asm_argument asm_string
%type <expr> asm_def opt_asm_operands asm_operands asm_operand asm_operand_1 asm_clobbers
%type <expr> opt_attr opt_resetattrs attrs attr attr_args attr_arg attr_arg_1
%type <expr> any_word opt_volatile offsetof_member_designator

%type <expr> directive

%{

%}

%%
start:
      /* empty */
            { STAT_TRACE(("{opt_ext_defs#1}")); if(s_exprStart) freeExpr(s_exprStart);
                s_exprStart = exprNull(); EXPR_REF(s_exprStart); }
    | ext_defs
            { STAT_TRACE(("{opt_ext_defs#2}")); if(s_exprStart) freeExpr(s_exprStart);
                s_exprStart = $1; EXPR_REF(s_exprStart); }
    ;

ext_defs:
      ext_def
            { STAT_TRACE(("{ext_defs#1}")); $$ = exprList1(EC_EXT_DEFS, $1); }
    | ext_defs ext_def
            { STAT_TRACE(("{ext_defs#2}")); $$ = exprListJoin($1, $2); }
    ;

ext_def:
      func_def
            { STAT_TRACE(("{ext_def#1}")); $$ = $1; }
    | data_def
            { STAT_TRACE(("{ext_def#2}")); $$ = $1; procTypeDefInParser($1); }
    | asm_def
            { STAT_TRACE(("{ext_def#3}")); $$ = $1; }
    | EXTENSION ext_def
            { STAT_TRACE(("{ext_def#4}")); $$ = exprSetExtension($2); }
    | directive
            { STAT_TRACE(("{ext_def#5}")); $$ = $1; }
    | PRAGMA_EXEC
            { STAT_TRACE(("{ext_def#6}")); 
	      $$ = exprList(EC_COMP_STMT);((CExprOfList *)$$)->e_aux_info=$1; }
    ;

data_def:
      notype_initdecls ';'
            { STAT_TRACE(("{data_def#1}")); $$ = exprBinary(EC_DATA_DEF, NULL, $1); }
    | declspecs_xTxx notype_initdecls ';'
            { STAT_TRACE(("{data_def#2}")); $$ = exprBinary(EC_DATA_DEF, $1, $2); }
    | declspecs_xtxx initdecls ';'
            { STAT_TRACE(("{data_def#3}")); $$ = exprBinary(EC_DATA_DEF, $1, $2); }
    | declspecs ';'
            { STAT_TRACE(("{data_def#4}")); $$ = exprBinary(EC_DATA_DEF, $1, NULL); }
    | ';'
            { STAT_TRACE(("{data_def#5}")); $$ = exprNull(); }
    | error ';'
            { STAT_TRACE(("{data_def#6}")); $$ = exprBinary(EC_DATA_DEF, exprError(), NULL); }
    | error '}'
            { STAT_TRACE(("{data_def#7}")); $$ = exprBinary(EC_DATA_DEF, exprError(), NULL); }
    ;

func_def:
      declspecs_xtxx declarator
            { STAT_TRACE(("{func_def#1a}")); pushSymbolTable(); addFuncDefDeclaratorSymbolInParser($2);
                $<expr>$ = exprList2(EC_FUNC_DEF, $1, $2); }
      opt_datadecls loc dummy compstmt_or_error
            { STAT_TRACE(("{func_def#1b}")); popSymbolTable();
                $$ = exprListAdd($<expr>3, $4); exprListAdd($$, $7); }
    | declspecs_xTxx nt_declarator
            { STAT_TRACE(("{func_def#2a}")); pushSymbolTable(); addFuncDefDeclaratorSymbolInParser($2);
                $<expr>$ = exprList2(EC_FUNC_DEF, $1, $2); }
      opt_datadecls loc dummy compstmt_or_error
            { STAT_TRACE(("{func_def#2b}")); popSymbolTable();
                $$ = exprListAdd($<expr>3, $4); exprListAdd($$, $7); }
    | nt_declarator
            { STAT_TRACE(("{func_def#3a}")); pushSymbolTable(); addFuncDefDeclaratorSymbolInParser($1);
                $<expr>$ = exprList2(EC_FUNC_DEF, exprNull(), $1); }
      opt_datadecls loc dummy compstmt_or_error
            { STAT_TRACE(("{func_def#3b}")); popSymbolTable();
                $$ = exprListAdd($<expr>2, $3); exprListAdd($$, $6); }
    ;

ident:
      IDENTIFIER
            { STAT_TRACE(("{ident#1}")); $$ = $1; }
    | TYPENAME
            { STAT_TRACE(("{ident#2}")); $$ = $1; }
    ;

unop:
      '&'
            { STAT_TRACE(("{unop#1}")); $<unop>$ = UO_ADDR_OF; }
    | '-'
            { STAT_TRACE(("{unop#2}")); $<unop>$ = UO_MINUS; }
    | '+'
            { STAT_TRACE(("{unop#3}")); $<unop>$ = UO_PLUS; }
    | '~'
            { STAT_TRACE(("{unop#4}")); $<unop>$ = UO_BIT_NOT; }
    | '!'
            { STAT_TRACE(("{unop#5}")); $<unop>$ = UO_LOG_NOT; }
    | PLUSPLUS
            { STAT_TRACE(("{unop#6}")); $<unop>$ = UO_PLUSPLUS; }
    | MINUSMINUS
            { STAT_TRACE(("{unop#7}")); $<unop>$ = UO_MINUSMINUS; }
    ;

opt_exprs:
      /* empty */
            { STAT_TRACE(("{opt_exprs#1}")); $$ = NULL; }
    | exprs
            { STAT_TRACE(("{opt_exprs#2}")); $$ = $1; }
    ;

exprs:
      expr
            { STAT_TRACE(("{exprs#1}")); $$ = exprList1(EC_EXPRS, $1); }
    | exprs ',' expr
            { STAT_TRACE(("{exprs#2}")); $$ = exprListJoin($1, $3); }
    ;

unary_expr:
      primary
            { STAT_TRACE(("{unary_expr#1}")); $$ = $1; }
    | '*' cast_expr             %prec UNARY
            { STAT_TRACE(("{unary_expr#2}")); $$ = exprUnary(EC_POINTER_REF, $2); }
    | EXTENSION cast_expr       %prec UNARY
            { STAT_TRACE(("{unary_expr#3}")); $$ = exprSetExtension($2); }
    | unop cast_expr            %prec UNARY
            { STAT_TRACE(("{unary_expr#4}"));
                CExprCodeEnum ec = s_CUnaryOpeEnumToExprCodeEnum[$<unop>1];
                $$ = (ec == EC_NULL_NODE) ?  $2 :
                    ((ec == EC_ERROR_NODE) ? exprError1($2) : exprUnary(ec, $2)); }
    /* label addr pointer.  */
    | ANDAND ident
            { STAT_TRACE(("{unary_expr#5}")); $$ = exprUnary(EC_GCC_LABEL_ADDR, $2); }
    | SIZEOF unary_expr         %prec UNARY
            { STAT_TRACE(("{unary_expr#6}")); $$ = exprUnary(EC_SIZE_OF, $2); }
    | SIZEOF '(' typename ')'   %prec HYPERUNARY
            { STAT_TRACE(("{unary_expr#7}")); $$ = exprUnary(EC_SIZE_OF, $3); }
    | ALIGNOF unary_expr        %prec UNARY
            { STAT_TRACE(("{unary_expr#8}")); $$ = exprUnary(EC_GCC_ALIGN_OF, $2); }
    | ALIGNOF '(' typename ')'  %prec HYPERUNARY
            { STAT_TRACE(("{unary_expr#9}")); $$ = exprUnary(EC_GCC_ALIGN_OF, $3); }
    | REALPART cast_expr        %prec UNARY
            { STAT_TRACE(("{unary_expr#10}")); $$ = exprUnary(EC_GCC_REALPART, $2); }
    | IMAGPART cast_expr        %prec UNARY
            { STAT_TRACE(("{unary_expr#11}")); $$ = exprUnary(EC_GCC_IMAGPART, $2); }
    | XMP_DESC_OF '(' IDENTIFIER ')' 
            { STAT_TRACE(("{unary_expr#12}")); $$ = exprUnary(EC_XMP_DESC_OF, $3); }
    ;

cast_expr:
      unary_expr
            { STAT_TRACE(("{cast_expr#1}")); $$ = $1; }
    | '(' typename ')' cast_expr  %prec UNARY
            { STAT_TRACE(("{cast_expr#2}")); $$ = exprBinary(EC_CAST, $2, $4); }
    ;

mul_expr:
      cast_expr
            { STAT_TRACE(("{mul_expr#1}")); $$ = $1; }
    | mul_expr '*' cast_expr
            { STAT_TRACE(("{mul_expr#2}")); $$ = exprBinary(EC_MUL, $1, $3); }
    | mul_expr '/' cast_expr
            { STAT_TRACE(("{mul_expr#3}")); $$ = exprBinary(EC_DIV, $1, $3); }
    | mul_expr '%' cast_expr
            { STAT_TRACE(("{mul_expr#4}")); $$ = exprBinary(EC_MOD, $1, $3); }
    ;

plus_expr:
      mul_expr
            { STAT_TRACE(("{plus_expr#1}")); $$ = $1; }
    | plus_expr '+' mul_expr
            { STAT_TRACE(("{plus_expr#2}")); $$ = exprBinary(EC_PLUS, $1, $3); }
    | plus_expr '-' mul_expr
            { STAT_TRACE(("{plus_expr#3}")); $$ = exprBinary(EC_MINUS, $1, $3); }
    ;

shift_expr:
      plus_expr
            { STAT_TRACE(("{shift_expr#1}")); $$ = $1; }
    | shift_expr LSHIFT plus_expr
            { STAT_TRACE(("{shift_expr#2}")); $$ = exprBinary(EC_LSHIFT, $1, $3); }
    | shift_expr RSHIFT plus_expr
            { STAT_TRACE(("{shift_expr#3}")); $$ = exprBinary(EC_RSHIFT, $1, $3); }
    ;

comp_expr:
      shift_expr
            { STAT_TRACE(("{comp_expr#1}")); $$ = $1; }
    | comp_expr '<' shift_expr
            { STAT_TRACE(("{comp_expr#2}")); $$ = exprBinary(EC_ARITH_LT, $1, $3); }
    | comp_expr '>' shift_expr
            { STAT_TRACE(("{comp_expr#3}")); $$ = exprBinary(EC_ARITH_GT, $1, $3); }
    | comp_expr ARITH_LE shift_expr
            { STAT_TRACE(("{comp_expr#4}")); $$ = exprBinary(EC_ARITH_LE, $1, $3); }
    | comp_expr ARITH_GE shift_expr
            { STAT_TRACE(("{comp_expr#5}")); $$ = exprBinary(EC_ARITH_GE, $1, $3); }
    ;

compeq_expr:
      comp_expr
            { STAT_TRACE(("{compeq_expr#1}")); $$ = $1; }
    | compeq_expr ARITH_EQ comp_expr
            { STAT_TRACE(("{compeq_expr#2}")); $$ = exprBinary(EC_ARITH_EQ, $1, $3); }
    | compeq_expr ARITH_NE comp_expr
            { STAT_TRACE(("{compeq_expr#3}")); $$ = exprBinary(EC_ARITH_NE, $1, $3); }
    ;

band_expr:
      compeq_expr
            { STAT_TRACE(("{band_expr#1}")); $$ = $1; }
    | band_expr '&' compeq_expr
            { STAT_TRACE(("{band_expr#2}")); $$ = exprBinary(EC_BIT_AND, $1, $3); }
    ;

bxor_expr:
      band_expr
            { STAT_TRACE(("{bxor_expr#1}")); $$ = $1; }
    | bxor_expr '^' band_expr
            { STAT_TRACE(("{bxor_expr#2}")); $$ = exprBinary(EC_BIT_XOR, $1, $3); }
    ;

bor_expr:
      bxor_expr
            { STAT_TRACE(("{bor_expr#1}")); $$ = $1; }
    | bor_expr '|' bxor_expr
            { STAT_TRACE(("{bor_expr#2}")); $$ = exprBinary(EC_BIT_OR, $1, $3); }
    ;

logand_expr:
      bor_expr
            { STAT_TRACE(("{logand_expr#1}")); $$ = $1; }
    | logand_expr ANDAND bor_expr
            { STAT_TRACE(("{logand_expr#2}")); $$ = exprBinary(EC_LOG_AND, $1, $3); }
    ;

logor_expr:
      logand_expr
            { STAT_TRACE(("{logor_expr#1}")); $$ = $1; }
    | logor_expr OROR logand_expr
            { STAT_TRACE(("{logor_expr#2}")); $$ = exprBinary(EC_LOG_OR, $1, $3); }
    ;

cond_expr:
      logor_expr
            { STAT_TRACE(("{cond_expr#1}")); $$ = $1; }
    | logor_expr '?' exprs ':' cond_expr
            { STAT_TRACE(("{cond_expr#2}")); $$ = exprList3(EC_CONDEXPR, $1, $3, $5); }
    | logor_expr '?' ':' cond_expr
            { STAT_TRACE(("{cond_expr#3}")); $$ = exprList3(EC_CONDEXPR, $1, exprNull(), $4);
                EXPR_ISGCCSYNTAX($$) = 1; }
    ;

expr:
      cond_expr
            { STAT_TRACE(("{expr#1}")); $$ = $1; }
    | unary_expr '=' expr
            { STAT_TRACE(("{expr#2}")); $$ = exprBinary(EC_ASSIGN, $1, $3); }
    | unary_expr ASSIGN expr
            { STAT_TRACE(("{expr#3}"));
                $$ = exprBinary(s_CAssignEnumToExprCodeEnum[$2], $1, $3); }
    ;

string:
      STRING
            { $$ = exprList1(EC_STRINGS, $1); }
    | string STRING
            { $$ = exprListJoin($1, $2); }

primary:
      primary_1
            { STAT_TRACE(("{primary#1}")); $$ = $1; }
    | coarray_ref
            { STAT_TRACE(("{primary#2}")); $$ = $1; }
    ;

primary_1:
      IDENTIFIER
            { STAT_TRACE(("{primary_1#1}")); $$ = $1; }
    | CONSTANT
            { STAT_TRACE(("{primary_1#2}")); $$ = $1; }
    | string
            { STAT_TRACE(("{primary_1#3}")); $$ = $1; }
    | '(' typename ')' '{' dummy opt_initializers_opt_comma '}'  %prec UNARY
            { STAT_TRACE(("{primary_1#4}")); $$ = exprBinary(EC_COMPOUND_LITERAL, $2, $6); }
    | '(' exprs ')'
            { STAT_TRACE(("{primary_1#5}")); $$ = exprUnary(EC_BRACED_EXPR, $2); }
    | compstmt_expr_start compstmt_end ')'
            { STAT_TRACE(("{primary_1#6}")); $$ = exprUnary(EC_GCC_COMP_STMT_EXPR, $2); }
    | primary '(' opt_exprs ')'  %prec '.'
            { STAT_TRACE(("{primary_1#7}")); $$ = exprBinary(EC_FUNCTION_CALL, $1, $3); }
    | primary '.' ident
            { STAT_TRACE(("{primary_1#8}")); $$ = exprBinary(EC_MEMBER_REF, $1, $3); }
    | primary POINTSAT ident
            { STAT_TRACE(("{primary_1#9}")); $$ = exprBinary(EC_POINTS_AT, $1, $3); }
    | primary PLUSPLUS
            { STAT_TRACE(("{primary_1#10}")); $$ = exprUnary(EC_POST_INCR, $1); }
    | primary MINUSMINUS
            { STAT_TRACE(("{primary_1#11}")); $$ = exprUnary(EC_POST_DECR, $1); }
    | BLTIN_VA_ARG '(' expr ',' typename ')'
            { STAT_TRACE(("{primary_1#12}")); $$ = exprBinary(EC_GCC_BLTIN_VA_ARG, $3, $5); }
    | BLTIN_OFFSETOF '(' typename ',' dummy offsetof_member_designator ')'
            { STAT_TRACE(("{primary_1#13}")); $$ = exprBinary(EC_GCC_BLTIN_OFFSET_OF, $3, $6);
                EXPR_ISGCCSYNTAX($$) = 1; }
    | BLTIN_OFFSETOF '(' error ')'
            { STAT_TRACE(("{primary_1#14}")); $$ = exprError();
                EXPR_ISGCCSYNTAX($$) = 1; }
    | BLTIN_TYPES_COMPATIBLE_P '(' typename ',' typename ')'
            { STAT_TRACE(("{primary_1#15}")); $$ = exprBinary(EC_GCC_BLTIN_TYPES_COMPATIBLE_P, $3, $5);
                EXPR_ISGCCSYNTAX($$) = 1; }
    | BLTIN_TYPES_COMPATIBLE_P '(' error ')'
            { STAT_TRACE(("{primary_1#16}")); $$ = exprError();}
    | array_ref
            { STAT_TRACE(("{primary_1#17}")); $$ = $1; }
    ;

array_ref:
      primary_1 '[' array_dimension ']'  %prec '.'
            { STAT_TRACE(("{array_ref#1}"));  $$ = exprBinary(EC_ARRAY_REF, $1, $3); }
    ;

array_dimension:
      exprs
            { STAT_TRACE(("{array_dimension#1}")); $$ = exprList1(EC_ARRAY_DIMENSION, $1); }
    /* | ':' */
    /*         { STAT_TRACE(("{array_dimension#4}")); $$ = exprSubArrayDimension(NULL, NULL, NULL); } */
    /* | exprs ':' exprs */
    /*         { STAT_TRACE(("{array_dimension#2}")); $$ = exprSubArrayDimension($1, $3, NULL); } */
    /* | exprs ':' exprs ':' exprs */
    /*         { STAT_TRACE(("{array_dimension#3}")); $$ = exprSubArrayDimension($1, $3, $5); } */
    | opt_exprs ':' opt_exprs
            { STAT_TRACE(("{array_dimension#2}")); $$ = exprSubArrayDimension($1, $3, NULL); }
    | opt_exprs ':' opt_exprs ':' opt_exprs
            { STAT_TRACE(("{array_dimension#3}")); $$ = exprSubArrayDimension($1, $3, $5); }
    ;

coarray_ref:
      primary coarray_dimensions
            { STAT_TRACE(("{coarray_ref#1}")); $$ = exprCoarrayRef($1, $2); }
    ;

coarray_dimensions:
        /*
            COLON_SQBRACKET is ': ['
            If COLON_SQBRACKET is not used, it will cause r/r conflict.
        */
      COLON_SQBRACKET coarray_dimension_1
            { STAT_TRACE(("{coarray_dimensions#1}")); $$ = exprList1(EC_XMP_COARRAY_DIMENSIONS, $2); }
    | coarray_dimensions coarray_dimension
            { STAT_TRACE(("{coarray_dimensions#2}")); $$ = exprListJoin($1, $2); }
    ;

coarray_dimension_1:
      exprs ']'
            { STAT_TRACE(("{coarray_dimension_1#1}")); $$ = $1; }
    | '*' ']'
            { STAT_TRACE(("{coarray_dimension_1#2}")); $$ = (CExpr*)allocExprOfGeneralCode(EC_FLEXIBLE_STAR, 0); }
    ;

coarray_dimension:
      '[' exprs ']'
            { STAT_TRACE(("{coarray_dimension#1}")); $$ = $2; }
    | '[' '*' ']'
            { STAT_TRACE(("{coarray_dimension#2}")); $$ = (CExpr*)allocExprOfGeneralCode(EC_FLEXIBLE_STAR, 0); }
    ;

offsetof_member_designator:
      ident
            { STAT_TRACE(("{offsetof_member_designator#1}")); $$ = exprBinary(EC_GCC_OFS_MEMBER_REF, exprNull(), $1); }
    | offsetof_member_designator '.' ident
            { STAT_TRACE(("{offsetof_member_designator#2}")); $$ = exprBinary(EC_GCC_OFS_MEMBER_REF, $1, $3); }
    | offsetof_member_designator '[' exprs ']'
            { STAT_TRACE(("{offsetof_member_designator#3}")); $$ = exprBinary(EC_GCC_OFS_ARRAY_REF, $1, $3); }
    ;

opt_datadecls:
      /* empty */
            { STAT_TRACE(("{opt_datadecls#1}")); $$ = NULL; }
    | datadecls
            { STAT_TRACE(("{opt_datadecls#2}")); $$ = $1; }
    ;

datadecls:
      datadecl
            { STAT_TRACE(("{datadecls#1}")); $$ = exprList1(EC_DATA_DECLS, $1); }
    | datadecls datadecl
            { STAT_TRACE(("{datadecls#2}")); $$ = exprListJoin($1, $2); }
    ;

datadecl:
      loc datadecl_1
            { $$ = $2; }
    ;

datadecl_1:
      declspecs_xtAx initdecls ';'
            { STAT_TRACE(("{datadecl#1}")); $$ = exprBinary(EC_DATA_DECL, $1, $2); }
    | declspecs_xTAx notype_initdecls ';'
            { STAT_TRACE(("{datadecl#2}")); $$ = exprBinary(EC_DATA_DECL, $1, $2); }
    | declspecs_xtAx ';'
            { STAT_TRACE(("{datadecl#3}")); $$ = exprBinary(EC_DATA_DECL, $1, NULL); }
    | declspecs_xTAx ';'
            { STAT_TRACE(("{datadecl#4}")); $$ = exprBinary(EC_DATA_DECL, $1, NULL); }
    ;

decl:
      loc decl_1
            { $$ = $2; procTypeDefInParser($2); }
    ;

decl_1:
      declspecs_xtxx initdecls ';'
            { STAT_TRACE(("{decl_1#1}")); $$ = exprBinary(EC_DECL, $1, $2); }
    | declspecs_xTxx notype_initdecls ';'
            { STAT_TRACE(("{decl_1#2}")); $$ = exprBinary(EC_DECL, $1, $2); }
    | declspecs_xtxx nested_func
            { STAT_TRACE(("{decl_1#3}")); $$ = exprBinary(EC_DECL, $1, $2); }
    | declspecs_xTxx notype_nested_func
            { STAT_TRACE(("{decl_1#4}")); $$ = exprBinary(EC_DECL, $1, $2); }
    | declspecs ';'
            { STAT_TRACE(("{decl_1#5}")); $$ = exprBinary(EC_DECL, $1, NULL); }
    | EXTENSION decl_1
            { STAT_TRACE(("{decl_1#6}")); $$ = exprSetExtension($2); }
    ;

/*
   declaration specifiers

   naming rule of state
     declspecs_[1][2][3][4]

   1. s : storage class is included
      S : storage class is not included
      x : any
    
   2. t : type specifier is included
      T : type specifier is not included
      x : any

   3. a : starts with attr
      A : not starts with attr
      x : any

   4. a : ends with attr
      A : not ends with attr
      x : any
*/

declspecs_CTAA:
      TYPEQUAL
            { STAT_TRACE(("{declspecs_CTAA#1}")); $$ = exprList1(EC_DECL_SPECS, $1); }
    | declspecs_CTAA TYPEQUAL
            { STAT_TRACE(("{declspecs_CTAA#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTAa TYPEQUAL
            { STAT_TRACE(("{declspecs_CTAA#3}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_CTAa:
      declspecs_CTAA attrs
            { STAT_TRACE(("{declspecs_CTAa#1}")); $$ = exprSetAttrTailNode($1, $2); }
    ;

declspecs_CTaA:
      declspecs_CTaA TYPEQUAL
            { STAT_TRACE(("{declspecs_CTaA#1}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTaa TYPEQUAL
            { STAT_TRACE(("{declspecs_CTaA#2}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_CTaa:
      attrs
            { STAT_TRACE(("{declspecs_CTaa#1}")); $$ = exprList1(EC_DECL_SPECS, $1); }
    | declspecs_CTaA attrs
            { STAT_TRACE(("{declspecs_CTaa#2}")); $$ = exprSetAttrTailNode($1, $2); }
    ;

declspecs_CtAA:
      typespec_na
            { STAT_TRACE(("{declspecs_CtAA#1}")); $$ = exprList1(EC_DECL_SPECS, $1); }
    | declspecs_CtAA TYPEQUAL
            { STAT_TRACE(("{declspecs_CtAA#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtAa TYPEQUAL
            { STAT_TRACE(("{declspecs_CtAA#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtAA typespec_reserved_na
            { STAT_TRACE(("{declspecs_CtAA#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtAa typespec_reserved_na
            { STAT_TRACE(("{declspecs_CtAA#5}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTAA typespec_na
            { STAT_TRACE(("{declspecs_CtAA#6}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTAa typespec_na
            { STAT_TRACE(("{declspecs_CtAA#7}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_CtAa:
      typespec_a
            { STAT_TRACE(("{declspecs_CtAa#1}")); $$ = exprList1(EC_DECL_SPECS, $1); }
    | declspecs_CtAA attrs
            { STAT_TRACE(("{declspecs_CtAa#2}")); $$ = exprSetAttrTailNode($1, $2); }
    | declspecs_CtAA typespec_reserved_a
            { STAT_TRACE(("{declspecs_CtAa#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtAa typespec_reserved_a
            { STAT_TRACE(("{declspecs_CtAa#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTAA typespec_a
            { STAT_TRACE(("{declspecs_CtAa#5}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTAa typespec_a
            { STAT_TRACE(("{declspecs_CtAa#6}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_CtaA:
      declspecs_CtaA TYPEQUAL
            { STAT_TRACE(("{declspecs_CtaA#1}")); $$ = exprListJoin($1, $2); }
    | declspecs_Ctaa TYPEQUAL
            { STAT_TRACE(("{declspecs_CtaA#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtaA typespec_reserved_na
            { STAT_TRACE(("{declspecs_CtaA#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_Ctaa typespec_reserved_na
            { STAT_TRACE(("{declspecs_CtaA#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTaA typespec_na
            { STAT_TRACE(("{declspecs_CtaA#5}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTaa typespec_na
            { STAT_TRACE(("{declspecs_CtaA#6}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_Ctaa:
      declspecs_CtaA attrs
            { STAT_TRACE(("{declspecs_Ctaa#1}")); $$ = exprSetAttrTailNode($1, $2); }
    | declspecs_CtaA typespec_reserved_a
            { STAT_TRACE(("{declspecs_Ctaa#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_Ctaa typespec_reserved_a
            { STAT_TRACE(("{declspecs_Ctaa#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTaA typespec_a
            { STAT_TRACE(("{declspecs_Ctaa#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTaa typespec_a
            { STAT_TRACE(("{declspecs_Ctaa#5}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_cTAA:
      scspec
            { STAT_TRACE(("{declspecs_cTAA#1}")); $$ = exprList1(EC_DECL_SPECS, $1); }
    | declspecs_cTAA TYPEQUAL
            { STAT_TRACE(("{declspecs_cTAA#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTAa TYPEQUAL
            { STAT_TRACE(("{declspecs_cTAA#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTAA scspec
            { STAT_TRACE(("{declspecs_cTAA#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTAa scspec
            { STAT_TRACE(("{declspecs_cTAA#5}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTAA scspec
            { STAT_TRACE(("{declspecs_cTAA#6}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTAa scspec
            { STAT_TRACE(("{declspecs_cTAA#7}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_cTAa:
      declspecs_cTAA attrs
            { STAT_TRACE(("{declspecs_cTAa#1}")); $$ = exprSetAttrTailNode($1, $2); }
    ;

declspecs_cTaA:
      declspecs_cTaA TYPEQUAL
            { STAT_TRACE(("{declspecs_cTaA#1}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTaa TYPEQUAL
            { STAT_TRACE(("{declspecs_cTaA#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTaA scspec
            { STAT_TRACE(("{declspecs_cTaA#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_CTaa scspec
            { STAT_TRACE(("{declspecs_cTaA#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTaA scspec
            { STAT_TRACE(("{declspecs_cTaA#5}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTaa scspec
            { STAT_TRACE(("{declspecs_cTaA#6}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_cTaa:
      declspecs_cTaA attrs
            { STAT_TRACE(("{declspecs_cTaa#1}")); $$ = exprSetAttrTailNode($1, $2); }
    ;

declspecs_ctAA:
      declspecs_ctAA TYPEQUAL
            { STAT_TRACE(("{declspecs_ctAA#1}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctAa TYPEQUAL
            { STAT_TRACE(("{declspecs_ctAA#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctAA typespec_reserved_na
            { STAT_TRACE(("{declspecs_ctAA#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctAa typespec_reserved_na
            { STAT_TRACE(("{declspecs_ctAA#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTAA typespec_na
            { STAT_TRACE(("{declspecs_ctAA#5}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTAa typespec_na
            { STAT_TRACE(("{declspecs_ctAA#6}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtAA scspec
            { STAT_TRACE(("{declspecs_ctAA#7}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtAa scspec
            { STAT_TRACE(("{declspecs_ctAA#8}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctAA scspec
            { STAT_TRACE(("{declspecs_ctAA#9}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctAa scspec
            { STAT_TRACE(("{declspecs_ctAA#10}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_ctAa:
      declspecs_ctAA attrs
            { STAT_TRACE(("{declspecs_ctAa#1}")); $$ = exprSetAttrTailNode($1, $2); }
    | declspecs_ctAA typespec_reserved_a
            { STAT_TRACE(("{declspecs_ctAa#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctAa typespec_reserved_a
            { STAT_TRACE(("{declspecs_ctAa#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTAA typespec_a
            { STAT_TRACE(("{declspecs_ctAa#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTAa typespec_a
            { STAT_TRACE(("{declspecs_ctAa#5}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_ctaA:
      declspecs_ctaA TYPEQUAL
            { STAT_TRACE(("{declspecs_ctaA#1}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctaa TYPEQUAL
            { STAT_TRACE(("{declspecs_ctaA#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctaA typespec_reserved_na
            { STAT_TRACE(("{declspecs_ctaA#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctaa typespec_reserved_na
            { STAT_TRACE(("{declspecs_ctaA#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTaA typespec_na
            { STAT_TRACE(("{declspecs_ctaA#5}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTaa typespec_na
            { STAT_TRACE(("{declspecs_ctaA#6}")); $$ = exprListJoin($1, $2); }
    | declspecs_CtaA scspec
            { STAT_TRACE(("{declspecs_ctaA#7}")); $$ = exprListJoin($1, $2); }
    | declspecs_Ctaa scspec
            { STAT_TRACE(("{declspecs_ctaA#8}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctaA scspec
            { STAT_TRACE(("{declspecs_ctaA#9}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctaa scspec
            { STAT_TRACE(("{declspecs_ctaA#10}")); $$ = exprListJoin($1, $2); }
    ;

declspecs_ctaa:
      declspecs_ctaA attrs
            { STAT_TRACE(("{declspecs_ctaa#1}")); $$ = exprSetAttrTailNode($1, $2); }
    | declspecs_ctaA typespec_reserved_a
            { STAT_TRACE(("{declspecs_ctaa#2}")); $$ = exprListJoin($1, $2); }
    | declspecs_ctaa typespec_reserved_a
            { STAT_TRACE(("{declspecs_ctaa#3}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTaA typespec_a
            { STAT_TRACE(("{declspecs_ctaa#4}")); $$ = exprListJoin($1, $2); }
    | declspecs_cTaa typespec_a
            { STAT_TRACE(("{declspecs_ctaa#5}")); $$ = exprListJoin($1, $2); }
    ;

/* Particular useful classes of declspecs.  */
declspecs_xtxx:
      declspecs_CtAA
            { STAT_TRACE(("{declspecs_xtxx#1}")); $$ = $1; }
    | declspecs_CtAa
            { STAT_TRACE(("{declspecs_xtxx#2}")); $$ = $1; }
    | declspecs_CtaA
            { STAT_TRACE(("{declspecs_xtxx#3}")); $$ = $1; }
    | declspecs_Ctaa
            { STAT_TRACE(("{declspecs_xtxx#4}")); $$ = $1; }
    | declspecs_ctAA
            { STAT_TRACE(("{declspecs_xtxx#5}")); $$ = $1; }
    | declspecs_ctAa
            { STAT_TRACE(("{declspecs_xtxx#6}")); $$ = $1; }
    | declspecs_ctaA
            { STAT_TRACE(("{declspecs_xtxx#7}")); $$ = $1; }
    | declspecs_ctaa
            { STAT_TRACE(("{declspecs_xtxx#8}")); $$ = $1; }
    ;

declspecs_xTxx:
      declspecs_CTAA
            { STAT_TRACE(("{declspecs_xTxx#1}")); $$ = $1; }
    | declspecs_CTAa
            { STAT_TRACE(("{declspecs_xTxx#2}")); $$ = $1; }
    | declspecs_CTaA
            { STAT_TRACE(("{declspecs_xTxx#3}")); $$ = $1; }
    | declspecs_CTaa
            { STAT_TRACE(("{declspecs_xTxx#4}")); $$ = $1; }
    | declspecs_cTAA
            { STAT_TRACE(("{declspecs_xTxx#5}")); $$ = $1; }
    | declspecs_cTAa
            { STAT_TRACE(("{declspecs_xTxx#6}")); $$ = $1; }
    | declspecs_cTaA
            { STAT_TRACE(("{declspecs_xTxx#7}")); $$ = $1; }
    | declspecs_cTaa
            { STAT_TRACE(("{declspecs_xTxx#8}")); $$ = $1; }
    ;

declspecs_xtAx:
      declspecs_CtAA
            { STAT_TRACE(("{declspecs_xtAx#1}")); $$ = $1; }
    | declspecs_CtAa
            { STAT_TRACE(("{declspecs_xtAx#2}")); $$ = $1; }
    | declspecs_ctAA
            { STAT_TRACE(("{declspecs_xtAx#3}")); $$ = $1; }
    | declspecs_ctAa
            { STAT_TRACE(("{declspecs_xtAx#4}")); $$ = $1; }
    ;

declspecs_xTAx:
      declspecs_CTAA
            { STAT_TRACE(("{declspecs_xTAx#1}")); $$ = $1; }
    | declspecs_CTAa
            { STAT_TRACE(("{declspecs_xTAx#2}")); $$ = $1; }
    | declspecs_cTAA
            { STAT_TRACE(("{declspecs_xTAx#3}")); $$ = $1; }
    | declspecs_cTAa
            { STAT_TRACE(("{declspecs_xTAx#4}")); $$ = $1; }
    ;

declspecs_Ctxx:
      declspecs_CtAA
            { STAT_TRACE(("{declspecs_Ctxx#1}")); $$ = $1; }
    | declspecs_CtAa
            { STAT_TRACE(("{declspecs_Ctxx#2}")); $$ = $1; }
    | declspecs_CtaA
            { STAT_TRACE(("{declspecs_Ctxx#3}")); $$ = $1; }
    | declspecs_Ctaa
            { STAT_TRACE(("{declspecs_Ctxx#4}")); $$ = $1; }
    ;

declspecs_CTxx:
      declspecs_CTAA
            { STAT_TRACE(("{declspecs_CTxx#1}")); $$ = $1; }
    | declspecs_CTAa
            { STAT_TRACE(("{declspecs_CTxx#2}")); $$ = $1; }
    | declspecs_CTaA
            { STAT_TRACE(("{declspecs_CTxx#3}")); $$ = $1; }
    | declspecs_CTaa
            { STAT_TRACE(("{declspecs_CTxx#4}")); $$ = $1; }
    ;

declspecs_Cxxx:
      declspecs_CtAA
            { STAT_TRACE(("{declspecs_Cxxx#1}")); $$ = $1; }
    | declspecs_CtAa
            { STAT_TRACE(("{declspecs_Cxxx#2}")); $$ = $1; }
    | declspecs_CtaA
            { STAT_TRACE(("{declspecs_Cxxx#3}")); $$ = $1; }
    | declspecs_Ctaa
            { STAT_TRACE(("{declspecs_Cxxx#4}")); $$ = $1; }
    | declspecs_CTAA
            { STAT_TRACE(("{declspecs_Cxxx#5}")); $$ = $1; }
    | declspecs_CTAa
            { STAT_TRACE(("{declspecs_Cxxx#6}")); $$ = $1; }
    | declspecs_CTaA
            { STAT_TRACE(("{declspecs_Cxxx#7}")); $$ = $1; }
    | declspecs_CTaa
            { STAT_TRACE(("{declspecs_Cxxx#8}")); $$ = $1; }
    ;

declspecs:
      declspecs_CTAA
            { STAT_TRACE(("{declspecs#1}")); $$ = $1; }
    | declspecs_CTAa
            { STAT_TRACE(("{declspecs#2}")); $$ = $1; }
    | declspecs_CTaA
            { STAT_TRACE(("{declspecs#3}")); $$ = $1; }
    | declspecs_CTaa
            { STAT_TRACE(("{declspecs#4}")); $$ = $1; }
    | declspecs_CtAA
            { STAT_TRACE(("{declspecs#5}")); $$ = $1; }
    | declspecs_CtAa
            { STAT_TRACE(("{declspecs#6}")); $$ = $1; }
    | declspecs_CtaA
            { STAT_TRACE(("{declspecs#7}")); $$ = $1; }
    | declspecs_Ctaa
            { STAT_TRACE(("{declspecs#8}")); $$ = $1; }
    | declspecs_cTAA
            { STAT_TRACE(("{declspecs#9}")); $$ = $1; }
    | declspecs_cTAa
            { STAT_TRACE(("{declspecs#10}")); $$ = $1; }
    | declspecs_cTaA
            { STAT_TRACE(("{declspecs#11}")); $$ = $1; }
    | declspecs_cTaa
            { STAT_TRACE(("{declspecs#12}")); $$ = $1; }
    | declspecs_ctAA
            { STAT_TRACE(("{declspecs#13}")); $$ = $1; }
    | declspecs_ctAa
            { STAT_TRACE(("{declspecs#14}")); $$ = $1; }
    | declspecs_ctaA
            { STAT_TRACE(("{declspecs#15}")); $$ = $1; }
    | declspecs_ctaa
            { STAT_TRACE(("{declspecs#16}")); $$ = $1; }
    ;

opt_typequals_a:
      /* empty */
            { STAT_TRACE(("{opt_typequals_a#1}")); $$ = NULL; }
    | declspecs_CTxx
            { STAT_TRACE(("{opt_typequals_a#2}")); $$ = $1; }
    ;

typespec_na:
      typespec_reserved_na
            { STAT_TRACE(("{typespec_na#1}")); $$ = $1; }
    | typespec_nonreserved_na
            { STAT_TRACE(("{typespec_na#2}")); $$ = $1; }
    ;

typespec_a:
      typespec_reserved_a
            { STAT_TRACE(("{typespec_a#1}")); $$ = $1; }
    ;

typespec_reserved_na:
      TYPESPEC
            { STAT_TRACE(("{typespec_reserved_na#1}")); $$ = $1; }
    | structsp_na
            { STAT_TRACE(("{typespec_reserved_na#2}")); $$ = $1; }
    ;

typespec_reserved_a:
      structsp_a
            { STAT_TRACE(("{typespec_reserved_a#1}")); $$ = $1; }
    ;

typespec_nonreserved_na:
      TYPENAME
            { STAT_TRACE(("{typespec_nonreserved_na#1}")); $$ = $1; }
    | TYPEOF '(' exprs ')'
            { STAT_TRACE(("{typespec_nonreserved_na#2}")); $$ = exprUnary(EC_GCC_TYPEOF, $3); }
    | TYPEOF '(' typename ')'
            { STAT_TRACE(("{typespec_nonreserved_na#3}")); $$ = exprUnary(EC_GCC_TYPEOF, $3); }
    ;

initdecls:
      initdecl
            { STAT_TRACE(("{initdecls#1}")); $$ = exprList1(EC_INIT_DECLS, $1); }
    | initdecls ',' opt_resetattrs initdecl
            { STAT_TRACE(("{initdecls#2}"));
                exprSetAttrPre(exprListHeadData($4), $3); $$ = exprListJoin($1, $4); }
    ;

notype_initdecls:
      notype_initdecl
            { STAT_TRACE(("{notype_initdecls#1}")); $$ = exprList1(EC_INIT_DECLS, $1); }
    | notype_initdecls ',' opt_resetattrs notype_initdecl
            { STAT_TRACE(("{notype_initdecls#2}"));
                exprSetAttrPre(exprListHeadData($4), $3); $$ = exprListJoin($1, $4); }
    ;

initdecl:
      initdecl_1
            { STAT_TRACE(("{initdecl#1}")); addInitDeclSymbolInParser($1); $$ = $1; }
    ;

initdecl_1:
      declarator opt_asm_expr opt_attr '=' init
            {
                STAT_TRACE(("{initdecl_1#1}"));
                exprSetAttrPost($1, $3);
                EXPR_C($1)->e_hasInit = 1;
                $$ = exprList3(EC_INIT_DECL, $1, $2, $5);
            }
    | codeclarator opt_asm_expr opt_attr '=' init
            {
                STAT_TRACE(("{initdecl_1#1-co}"));
                exprSetAttrPost($1, $3);
                EXPR_C($1)->e_hasInit = 1;
                $$ = exprList3(EC_INIT_DECL, $1, $2, $5);
            }
    | declarator opt_asm_expr opt_attr
            {
                STAT_TRACE(("{initdecl_1#2}"));
                exprSetAttrPost($1, $3);
                $$ = exprList2(EC_INIT_DECL, $1, $2);
            }
    | codeclarator opt_asm_expr opt_attr
            {
                STAT_TRACE(("{initdecl_1#2-co}"));
                exprSetAttrPost($1, $3);
                $$ = exprList2(EC_INIT_DECL, $1, $2);
            }
    ;

notype_initdecl:
      notype_initdecl_1
            { STAT_TRACE(("{notype_initdecl#1}")); addInitDeclSymbolInParser($1); $$ = $1; }

notype_initdecl_1:
      nt_declarator opt_asm_expr opt_attr '=' init
            {
                STAT_TRACE(("{notype_initdecl_1#1}"));
                exprSetAttrPost($1, $3);
                $$ = exprList3(EC_INIT_DECL, $1, $2, $5);
            }
    | nt_codeclarator opt_asm_expr opt_attr '=' init
            {
                STAT_TRACE(("{notype_initdecl_1#1-co}"));
                exprSetAttrPost($1, $3);
                $$ = exprList3(EC_INIT_DECL, $1, $2, $5);
            }
    | nt_declarator opt_asm_expr opt_attr
            {
                STAT_TRACE(("{notype_initdecl_1#2}"));
                exprSetAttrPost($1, $3);
                $$ = exprList2(EC_INIT_DECL, $1, $2);
            }
    | nt_codeclarator opt_asm_expr opt_attr
            {
                STAT_TRACE(("{notype_initdecl_1#2-co}"));
                exprSetAttrPost($1, $3);
                $$ = exprList2(EC_INIT_DECL, $1, $2);
            }
    ;

scspec:
      STATIC
            { STAT_TRACE(("{scspec#1}")); $$ = $1; }
    | SCSPEC
            { STAT_TRACE(("{scspec#2}")); $$ = $1; }
    ;

init:
      expr
            { STAT_TRACE(("{init#1}")); $$ = exprUnary(EC_INIT, $1); }
    | '{' opt_initializers_opt_comma '}'
            { STAT_TRACE(("{init#2}")); $$ = exprUnary(EC_INIT, $2); }
    | error
            { STAT_TRACE(("{init#3}")); $$ = exprUnary(EC_INIT, exprError()); }
    ;

opt_initializers_opt_comma:
      /* empty */
            { STAT_TRACE(("{opt_initializers_opt_comma#1}")); $$ = exprList(EC_INITIALIZERS); }
    | initializers opt_comma
            { STAT_TRACE(("{opt_initializers_opt_comma#2}")); $$ = $1; }
    ;

initializers:
      initializer
            { STAT_TRACE(("{initializers#1}")); $$ = exprList1(EC_INITIALIZERS, $1); }
    | initializers ',' initializer
            { STAT_TRACE(("{initializers#2}")); $$ = exprListJoin($1, $3); }
    ;

initializer:
      designators '=' initval
            { STAT_TRACE(("{initializer#1}")); $$ = exprBinary(EC_INITIALIZER, $1, $3); }
    | array_designator initval
            { STAT_TRACE(("{initializer#2}")); $$ = exprBinary(EC_INITIALIZER, $1, $2); }
    | initval
            { STAT_TRACE(("{initializer#3}")); $$ = exprBinary(EC_INITIALIZER, NULL, $1); }
            /* 'ident : intval' means '.ident = initval' */
    | ident ':' initval
            { STAT_TRACE(("{initializer#4}")); $$ =
                    exprBinary(EC_INITIALIZER, exprList1(EC_DESIGNATORS, $1), $3);
                EXPR_ISGCCSYNTAX($$) = 1; }
    ;

initval:
      '{' opt_initializers_opt_comma '}'
            { STAT_TRACE(("{initval#1}")); $$ = $2; }
    | expr
            { STAT_TRACE(("{initval#2}")); $$ = $1; }
    | error
            { STAT_TRACE(("{initval#3}")); $$ = exprError(); }
    ;

designators:
      designator
            { STAT_TRACE(("{designators#1}")); $$ = exprList1(EC_DESIGNATORS, $1); }
    | designators designator
            { STAT_TRACE(("{designators#2}")); $$ = exprListJoin($1, $2); }
    ;

designator:
      '.' ident
            { STAT_TRACE(("{designator#1}")); $$ = $2; }
    | array_designator
            { STAT_TRACE(("{designator#2}")); $$ = $1; }
    ;

array_designator:
      '[' expr ELLIPSIS expr ']'
            { STAT_TRACE(("{array_designator#1}")); $$ = exprBinary(EC_ARRAY_DESIGNATOR, $2, $4);
                EXPR_ISGCCSYNTAX($$) = 1; }
    | '[' expr ']'
            { STAT_TRACE(("{array_designator#2}")); $$ = exprBinary(EC_ARRAY_DESIGNATOR, $2, NULL); }
    ;

nested_func:
      declarator dummy opt_datadecls loc dummy compstmt
            { STAT_TRACE(("{nested_func#1}")); $$ = exprList4(EC_FUNC_DEF, exprNull(), $1, $3, $6); }
    ;

notype_nested_func:
      nt_declarator dummy opt_datadecls loc dummy compstmt
            { STAT_TRACE(("{notype_nested_func#1}")); $$ = exprList4(EC_FUNC_DEF, exprNull(), $1, $3, $6); }
    ;

declarator:
      after_type_declarator
            { STAT_TRACE(("{declarator#1}")); $$ = $1; }
    | nt_declarator
            { STAT_TRACE(("{declarator#2}")); $$ = $1; }
    ;

codeclarator:
      declarator coarray_declarations
            { STAT_TRACE(("{codeclarator#1}"));
                $$ = exprListJoin($1, $2); }
    ;

coarray_declarations:
      COLON_SQBRACKET coarray_declaration_1
            { STAT_TRACE(("{coarray_declarations#1}"));
                //                $$ = exprList1(EC_XMP_COARRAY_DECLARATIONS, exprArrayDecl(NULL, $2));
                $$ = exprList1(EC_LDECLARATOR, $2); }
    | coarray_declarations coarray_declaration
            { STAT_TRACE(("{coarray_declarations#2}"));
                $$ = exprListJoin($1, $2); }
    ;

coarray_declaration_1:
      expr ']'
            { STAT_TRACE(("{coarray_declaration_1#1}"));
                $$ = exprArrayDecl(NULL, $1);
                $$->u.e_arrayDecl.e_common.e_exprCode = EC_COARRAY_DECL;
            }
    | '*' ']'
            { STAT_TRACE(("{coarray_declaration_1#2}"));
                // $$ = (CExpr*)allocExprOfGeneralCode(EC_FLEXIBLE_STAR, 0);
                $$ = exprArrayDecl(NULL, NULL);
                $$->u.e_arrayDecl.e_common.e_exprCode = EC_COARRAY_DECL;
            }
    ;

coarray_declaration:
      '[' expr ']'
            { STAT_TRACE(("{coarray_declaration#1}"));
                $$ = exprArrayDecl(NULL, $2);
                $$->u.e_arrayDecl.e_common.e_exprCode = EC_COARRAY_DECL;
            }
    | '[' '*' ']'
            { STAT_TRACE(("{coarray_declaration#2}"));
                // $$ = (CExpr*)allocExprOfGeneralCode(EC_FLEXIBLE_STAR, 0);
                $$ = exprArrayDecl(NULL, NULL);
                $$->u.e_arrayDecl.e_common.e_exprCode = EC_COARRAY_DECL;
            }
    ;

after_type_declarator:
      '(' opt_attr after_type_declarator ')'
            { STAT_TRACE(("{after_type_declarator#1}"));
                $$ = exprList1(EC_LDECLARATOR, exprSetAttrHeadNode($3, $2)); }
    | after_type_declarator '(' params_or_idents            %prec '.'
            { STAT_TRACE(("{after_type_declarator#2}")); $$ = exprListJoin($1, $3); }
    | after_type_declarator array_declarator                       %prec '.'
            { STAT_TRACE(("{after_type_declarator#3}")); $$ = exprListJoin($1, $2); }
    | '*' opt_typequals_a after_type_declarator                    %prec UNARY
            { STAT_TRACE(("{after_type_declarator#4}")); $$ = exprListAdd($3, allocPointerDecl($2)); }
    | TYPENAME
            { STAT_TRACE(("{after_type_declarator#5}")); $$ = exprList1(EC_LDECLARATOR, $1); }
    ;

param_declarator:
      param_declarator_ts
            { STAT_TRACE(("{param_declarator#1}")); $$ = $1; }
    | param_declarator_nts
            { STAT_TRACE(("{param_declarator#2}")); $$ = $1; }
    ;

param_codeclarator:
      param_declarator coarray_declarations
            { STAT_TRACE(("{param_cedeclarator#1}"));
                $$ = exprListJoin($1, $2); }
    ;

param_declarator_ts:
      param_declarator_ts '(' params_or_idents     %prec '.'
            { STAT_TRACE(("{param_declarator_ts#1}")); $$ = exprListJoin($1, $3); }
    | param_declarator_ts array_declarator         %prec '.'
            { STAT_TRACE(("{param_declarator_ts#2}")); $$ = exprListJoin($1, $2); }
    | TYPENAME
            { STAT_TRACE(("{param_declarator_ts#3}")); $$ = exprList1(EC_LDECLARATOR, $1); }
    ;

param_declarator_nts:
      param_declarator_nts '(' params_or_idents   %prec '.'
            { STAT_TRACE(("{param_declarator_nts#1}")); $$ = exprListJoin($1, $3); }
    | param_declarator_nts array_declarator       %prec '.'
            { STAT_TRACE(("{param_declarator_nts#2}")); $$ = exprListJoin($1, $2); }
    | '*' opt_typequals_a param_declarator_ts     %prec UNARY
            { STAT_TRACE(("{param_declarator_nts#3}"));
                $$ = exprListAdd($3, allocPointerDecl($2)); }
    | '*' opt_typequals_a param_declarator_nts    %prec UNARY
            { STAT_TRACE(("{param_declarator_nts#4}"));
                $$ = exprListAdd($3, allocPointerDecl($2)); }
    | '(' opt_attr param_declarator_nts ')'
            { STAT_TRACE(("{param_declarator_nts#5}"));
                $$ = EXPR_ISNULL($2) ? (freeExpr($2), $3) : exprListCons($2, $3); }
    ;

nt_declarator:
      nt_declarator '(' params_or_idents           %prec '.'
            { STAT_TRACE(("{nt_declarator#1}")); $$ = exprListJoin($1, $3); }
    | '(' opt_attr nt_declarator ')'
            { STAT_TRACE(("{nt_declarator#2}"));
                $$ = exprList1(EC_LDECLARATOR, $3);
                    if(EXPR_ISNULL($2) == 0) exprJoinAttrToPre($$, $2); }
    | '*' opt_typequals_a nt_declarator            %prec UNARY
            { STAT_TRACE(("{nt_declarator#3}"));
                $$ = exprListAdd($3, allocPointerDecl($2)); }
    | nt_declarator array_declarator               %prec '.'
            { STAT_TRACE(("{nt_declarator#4}")); $$ = exprListJoin($1, $2); }
    | IDENTIFIER
            { STAT_TRACE(("{nt_declarator#5}")); $$ = exprList1(EC_LDECLARATOR, $1); }
    ;

nt_codeclarator:
      nt_declarator coarray_declarations
            { STAT_TRACE(("{nt_codeclarator#1}"));
                $$ = exprListJoin($1, $2); }
    ;

tagname:
      ident
            { STAT_TRACE(("{tagname#2")); $$ = $1; }
    ;

struct_a:
      STRUCT opt_attr
            { STAT_TRACE(("{struct_a#1}")); $$ = exprList1(EC_STRUCT_TYPE, exprSetAttrPre(exprNull(), $2)); }
    ;

union_a:
      UNION opt_attr
            { STAT_TRACE(("{union_a#1}")); $$ = exprList1(EC_UNION_TYPE, exprSetAttrPre(exprNull(), $2)); }
    ;

enum_a:
      ENUM opt_attr
            { STAT_TRACE(("{enum_a#1}")); $$ = exprList1(EC_ENUM_TYPE, exprSetAttrPre(exprNull(), $2)); }
    ;

structsp_a:
      struct_a tagname
            { STAT_TRACE(("{structsp_a#1a}")); addSymbolInParser($2, ST_TAG); }
      '{' member_decls '}' opt_attr
            { STAT_TRACE(("{structsp_a#1b}"));
                $$ = exprListJoin(exprListJoin($1, $2), $5);
                exprSetAttrPost(exprListHeadData($$), $7); }
    | struct_a '{' member_decls '}' opt_attr
            { STAT_TRACE(("{structsp_a#2}"));
                $$ = exprListJoin(exprListJoin($1, exprNull()), $3);
                exprSetAttrPost(exprListHeadData($$), $5); }
    | union_a tagname
            { STAT_TRACE(("{structsp_a#3a}")); addSymbolInParser($2, ST_TAG); }
      '{' member_decls '}' opt_attr
            { STAT_TRACE(("{structsp_a#3b}"));
                $$ = exprListJoin(exprListJoin($1, $2), $5);
                exprSetAttrPost(exprListHeadData($$), $7); }
    | union_a '{' member_decls '}' opt_attr
            { STAT_TRACE(("{structsp_a#4}"));
                $$ = exprListJoin(exprListJoin($1, exprNull()), $3);
                exprSetAttrPost(exprListHeadData($$), $5); }
    | enum_a tagname
            { STAT_TRACE(("{structsp_a#5a}")); addSymbolInParser($2, ST_TAG); }
      '{' enumerators opt_comma '}' opt_attr
            { STAT_TRACE(("{structsp_a#5b}"));
                $$ = exprListJoin(exprListJoin($1, $2), $5);
                exprSetAttrPost(exprListHeadData($$), $8); }
    | enum_a '{' enumerators opt_comma '}' opt_attr
            { STAT_TRACE(("{structsp_a#6}"));
                $$ = exprListJoin(exprListJoin($1, exprNull()), $3);
                exprSetAttrPost(exprListHeadData($$), $6); }
    ;

structsp_na:
      struct_a ident
            { STAT_TRACE(("{structsp_na#1}")); $$ = exprListJoin($1, $2); EXPR_SYMBOL($2)->e_symType = ST_TAG; }
    | union_a ident
            { STAT_TRACE(("{structsp_na#2}")); $$ = exprListJoin($1, $2); EXPR_SYMBOL($2)->e_symType = ST_TAG; }
    | enum_a ident
            { STAT_TRACE(("{structsp_na#3}")); $$ = exprListJoin($1, $2); EXPR_SYMBOL($2)->e_symType = ST_TAG; }
    ;

opt_comma:
      /* empty */
            { STAT_TRACE(("{opt_comma#1}")); }
    | ','
            { STAT_TRACE(("{opt_comma#2}")); }
    ;

member_decls:
      member_decls_1
            { STAT_TRACE(("{member_decls#1}")); $$ = $1; }
    | member_decls_1 member_decl
            { STAT_TRACE(("{member_decls#2}")); $$ = exprListJoin($1, $2);
                addWarn($$, CWRN_011); }
    ;

member_decls_1:
      /* empty */
            { STAT_TRACE(("{member_decls_1#1}")); $$ = exprList(EC_MEMBER_DECLS); }
    | member_decls_1 member_decl ';'
            { STAT_TRACE(("{member_decls_1#2}")); $$ = exprListJoin($1, $2); }
    | member_decls_1 ';'
            { STAT_TRACE(("{member_decls_1#3}")); $$ = $1; }
    | member_decls_1 directive
            { STAT_TRACE(("{member_decls_1#4}")); $$ = exprListAdd($1, $2); }
    ;

member_decl:
      declspecs_Ctxx members
            { STAT_TRACE(("{member_decl#1}")); $$ = exprBinary(EC_MEMBER_DECL, $1, $2); }
    | declspecs_Ctxx
            { STAT_TRACE(("{member_decl#2}")); $$ = exprBinary(EC_MEMBER_DECL, $1, NULL); }
    | declspecs_CTxx members_notype
            { STAT_TRACE(("{member_decl#3}")); $$ = exprBinary(EC_MEMBER_DECL, $1, $2); }
    | declspecs_CTxx
            { STAT_TRACE(("{member_decl#4}")); $$ = exprBinary(EC_MEMBER_DECL, $1, NULL); }
    | EXTENSION member_decl
            { STAT_TRACE(("{member_decl#6}")); $$ = exprSetExtension($2); }
    | error
            { STAT_TRACE(("{member_decl#8}")); $$ = exprBinary(EC_MEMBER_DECL, exprError(), NULL); }
    ;

members:
      member_declarator
            { STAT_TRACE(("{members#1}")); $$ = exprList1(EC_MEMBERS, $1); }
    | members ',' opt_resetattrs member_declarator
            { STAT_TRACE(("{members#2}")); exprSetAttrPre($4, $3); $$ = exprListJoin($1, $4); }
    ;

members_notype:
      member_nt_declarator
            { STAT_TRACE(("{members_notype#1}")); $$ = exprList1(EC_MEMBERS, $1); }
    | members_notype ',' opt_resetattrs member_nt_declarator
            { STAT_TRACE(("{members_notype#2}")); exprSetAttrPre($4, $3); $$ = exprListJoin($1, $4);  }
    ;

member_declarator:
      declarator opt_attr
            { STAT_TRACE(("{member_declarator#1}"));
                if($2) exprJoinAttrToPre($1, $2);
                $$ = exprBinary(EC_MEMBER_DECLARATOR, $1, NULL); }
    | codeclarator opt_attr
            { STAT_TRACE(("{member_declarator#1-co}"));
                if($2) exprJoinAttrToPre($1, $2);
                $$ = exprBinary(EC_MEMBER_DECLARATOR, $1, NULL); }
    /* bit field */
    | declarator ':' expr opt_attr
            { STAT_TRACE(("{member_declarator#2}"));
                if($4) exprJoinAttrToPre($1, $4);
                $$ = exprBinary(EC_MEMBER_DECLARATOR, $1, $3); }
    | ':' expr opt_attr
            { STAT_TRACE(("{member_declarator#3}"));
                $$ = exprBinary(EC_MEMBER_DECLARATOR, NULL, $2);
                if($3) exprJoinAttrToPre($$, $3); }
    ;

member_nt_declarator:
      nt_declarator opt_attr
            { STAT_TRACE(("{member_nt_declarator#1}"));
                $$ = exprSetAttrPost(exprBinary(EC_MEMBER_DECLARATOR, $1, NULL), $2); }
    | nt_codeclarator opt_attr
            { STAT_TRACE(("{member_nt_declarator#1-co}"));
                $$ = exprSetAttrPost(exprBinary(EC_MEMBER_DECLARATOR, $1, NULL), $2); }
    | nt_declarator ':' expr opt_attr
            { STAT_TRACE(("{member_nt_declarator#2}"));
                $$ = exprSetAttrPost(exprBinary(EC_MEMBER_DECLARATOR, $1, $3), $4); }
    | ':' expr opt_attr
            { STAT_TRACE(("{member_nt_declarator#3}"));
                $$ = exprSetAttrPost(exprBinary(EC_MEMBER_DECLARATOR, NULL, $2), $3); }
    ;

enumerators:
      enumerator
            { STAT_TRACE(("{enumerators#1}")); $$ = exprList1(EC_ENUMERATORS, $1); }
    | enumerators ',' enumerator
            { STAT_TRACE(("{enumerators#2}")); $$ = exprListJoin($1, $3); }
    | error
            { STAT_TRACE(("{enumerators#3}")); $$ = exprList1(EC_ENUMERATORS, exprError()); }
    ;

enumerator:
      ident
            { STAT_TRACE(("{enumerator#1}")); $$ = $1; addSymbolInParser($1, ST_ENUM); }
    | ident '=' expr
            { STAT_TRACE(("{enumerator#2}")); EXPR_SYMBOL($1)->e_valueExpr = $3; EXPR_REF($3);
                addSymbolInParser($1, ST_ENUM); $$ = $1; }
    ;

typename:
      declspecs_Cxxx opt_absdecl
            { STAT_TRACE(("{typename#1}")); $$ = exprBinary(EC_TYPENAME, $1, $2); }
    ;

opt_absdecl:
      /* empty */
            { STAT_TRACE(("{opt_absdecl#1}")); $$ = NULL; }
    | absdecl
            { STAT_TRACE(("{opt_absdecl#2}")); $$ = $1; }
    ;

absdecl_opt_a:
      /* empty */
            { STAT_TRACE(("{absdecl_opt_a#1}")); $$ = NULL; }
    | absdecl
            { STAT_TRACE(("{absdecl_opt_a#2}")); $$ = $1; }
    | absdecl_na attrs
            { STAT_TRACE(("{absdecl_opt_a#3}")); $$ = exprSetAttrPost($1, $2);  }
    ;

absdecl:
      absdecl_a
            { STAT_TRACE(("{absdecl#1}")); $$ = $1; }
    | absdecl_na
            { STAT_TRACE(("{absdecl#2}")); $$ = $1; }
    ;

absdecl_na:
      direct_absdecl
            { STAT_TRACE(("{absdecl_na#1}")); $$ = $1; }
    | '*' opt_typequals_a absdecl_na
            { STAT_TRACE(("{absdecl_na#2}")); $$ = exprListAdd($3, allocPointerDecl($2)); }
    ;

absdecl_a:
      '*' opt_typequals_a
            { STAT_TRACE(("{absdecl_a#1}")); $$ = exprList1(EC_LDECLARATOR, allocPointerDecl($2)); }
    | '*' opt_typequals_a absdecl_a
            { STAT_TRACE(("{absdecl_a#2}")); $$ = exprListAdd($3, allocPointerDecl($2)); }
    ;

direct_absdecl:
      '(' opt_attr absdecl ')'
            { STAT_TRACE(("{direct_absdecl#1}"));
                /* return as EC_LDECLARATOR for inner abstract declation. see collectSpecs() */
                $$ = exprList1(EC_LDECLARATOR,
                    EXPR_ISNULL($2) ? (freeExpr($2), $3) : exprListCons($2, $3)); }
    | direct_absdecl '(' params
            { STAT_TRACE(("{direct_absdecl#2}")); $$ = exprListAdd($1, $3); }
    | direct_absdecl array_declarator
            { STAT_TRACE(("{direct_absdecl#3}")); $$ = exprListAdd($1, $2); }
    | '(' params
            { STAT_TRACE(("{direct_absdecl#4}")); $$ = exprList1(EC_LDECLARATOR, $2); }
    | array_declarator
            { STAT_TRACE(("{direct_absdecl#5}")); $$ = exprList1(EC_LDECLARATOR, $1); }
    ;

array_declarator:
      '[' opt_typequals_a expr ']'
            { STAT_TRACE(("{array_declarator#1}")); $$ = exprArrayDecl($2, $3); }
    | '[' opt_typequals_a ']'
            { STAT_TRACE(("{array_declarator#2}")); $$ = exprArrayDecl($2, NULL); }
    | '[' opt_typequals_a '*' ']'
            { STAT_TRACE(("{array_declarator#3}"));
                $$ = exprArrayDecl($2, NULL); EXPR_ARRAYDECL($$)->e_isVariable = 1; }
    | '[' STATIC opt_typequals_a expr ']'
            { STAT_TRACE(("{array_declarator#4}"));
                $$ = exprArrayDecl($3, $4); EXPR_ARRAYDECL($$)->e_isStatic = 1; freeExpr($2); }
    | '[' declspecs_CTxx STATIC expr ']'
            { STAT_TRACE(("{array_declarator#5}"));
                $$ = exprArrayDecl($2, $4); EXPR_ARRAYDECL($$)->e_isStatic = 1; freeExpr($3); }
    ;

stmts_and_decls:
      stmts_decls_labels_es
            { STAT_TRACE(("{stmts_and_decls#1}")); $$ = $1; }
    | stmts_decls_labels_ed
            { STAT_TRACE(("{stmts_and_decls#2}")); $$ = $1; }
    | stmts_decls_labels_el
            { STAT_TRACE(("{stmts_and_decls#3}")); $$ = $1;
                addError($1, CERR_003); }
    | stmts_decls_labels_ep
            { STAT_TRACE(("{stmts_and_decls#4}")); $$ = $1; }
    | stmts_decls_labels_error
            { STAT_TRACE(("{stmts_and_decls#5}")); $$ = $1; }
    ;

stmts_decls_labels_es:
      stmt
            { STAT_TRACE(("{stmts_decls_labels_es#1}")); $$ = exprList1(EC_STMTS_AND_DECLS, $1); }
    | stmts_decls_labels_es stmt
            { STAT_TRACE(("{stmts_decls_labels_es#2}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_ed stmt
            { STAT_TRACE(("{stmts_decls_labels_es#3}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_el stmt
            { STAT_TRACE(("{stmts_decls_labels_es#4}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_ep stmt
            { STAT_TRACE(("{stmts_decls_labels_es#5}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_error stmt
            { STAT_TRACE(("{stmts_decls_labels_es#6}")); $$ = exprListJoin($1, $2); }
    ;

stmts_decls_labels_ed:
      decl
            { STAT_TRACE(("{stmts_decls_labels_ed#1}")); $$ = exprList1(EC_STMTS_AND_DECLS, $1); }
    | stmts_decls_labels_es decl
            { STAT_TRACE(("{stmts_decls_labels_ed#2}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_ed decl
            { STAT_TRACE(("{stmts_decls_labels_ed#3}")); $$ = exprListJoin($1, $2); }
    /* no stmts_decls_labels_el */
    | stmts_decls_labels_ep decl
            { STAT_TRACE(("{stmts_decls_labels_ed#4}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_error decl
            { STAT_TRACE(("{stmts_decls_labels_ed#5}")); $$ = exprListJoin($1, $2); }
    ;

stmts_decls_labels_el:
      label
            { STAT_TRACE(("{stmts_decls_labels_el#1}")); $$ = exprList1(EC_STMTS_AND_DECLS, $1); }
    | stmts_decls_labels_es label
            { STAT_TRACE(("{stmts_decls_labels_el#2}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_ed label
            { STAT_TRACE(("{stmts_decls_labels_el#3}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_el label
            { STAT_TRACE(("{stmts_decls_labels_el#4}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_ep label
            { STAT_TRACE(("{stmts_decls_labels_el#5}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_error label
            { STAT_TRACE(("{stmts_decls_labels_el#6}")); $$ = exprListJoin($1, $2); }
    ;

stmts_decls_labels_ep:
      directive
            { STAT_TRACE(("{stmts_decls_labels_ep#1}")); $$ = exprList1(EC_STMTS_AND_DECLS, $1); }
    | stmts_decls_labels_es directive
            { STAT_TRACE(("{stmts_decls_labels_ep#2}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_ed directive
            { STAT_TRACE(("{stmts_decls_labels_ep#3}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_el directive
            { STAT_TRACE(("{stmts_decls_labels_ep#4}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_ep directive
            { STAT_TRACE(("{stmts_decls_labels_ep#5}")); $$ = exprListJoin($1, $2); }
    | stmts_decls_labels_error directive
            { STAT_TRACE(("{stmts_decls_labels_ep#6}")); $$ = exprListJoin($1, $2); }
    ;

stmts_decls_labels_error:
      stmts_decls_labels errstmt
            { STAT_TRACE(("{stmts_decls_labels_error#2}")); $$ = exprListJoin($1, $2); }
    | errstmt
            { STAT_TRACE(("{stmts_decls_labels_error#1}")); $$ = exprList1(EC_STMTS_AND_DECLS, $1); }
    ;

stmts_decls_labels:
      stmts_decls_labels_es
            { STAT_TRACE(("{stmts_decls_labels#1}")); $$ = $1; }
    | stmts_decls_labels_ed
            { STAT_TRACE(("{stmts_decls_labels#2}")); $$ = $1; }
    | stmts_decls_labels_el
            { STAT_TRACE(("{stmts_decls_labels#3}")); $$ = $1; }
    | stmts_decls_labels_error
            { STAT_TRACE(("{stmts_decls_labels#4}")); $$ = $1; }
    ;

errstmt:
      error ';'
            { STAT_TRACE(("{errstmt#1}")); $$ = exprError(); }
    ;

/* C99 new scope  */
block_start:
      /* empty */
            { STAT_TRACE(("{block_start#1}")); pushSymbolTable(); }
    ;

opt_label_decls:
      /* empty */
            { STAT_TRACE(("{opt_label_decls#1}")); $$ = exprNull(); }
    | label_decls
            { STAT_TRACE(("{opt_label_decls#2}")); $$ = $1; }
    ;

label_decls:
      label_decl
            { STAT_TRACE(("{label_decls#1}")); $$ = exprList1(EC_GCC_LABEL_DECLS, $1); }
    | label_decls label_decl
            { STAT_TRACE(("{label_decls#2}")); $$ = exprListJoin($1, $2); }
    ;

label_decl:
      LABEL label_idents ';'
            { STAT_TRACE(("{label_decl#1}")); $$ = $2; }
    ;

compstmt_or_error:
      compstmt
            { STAT_TRACE(("{compstmt_or_error#1}")); $$ = $1; }
    | error compstmt
            { STAT_TRACE(("{compstmt_or_error#2}")); $$ = exprListCons(exprError(), $2); }
    ;

compstmt_start:
      '{'
            { STAT_TRACE(("{compstmt_start#1}")); pushSymbolTable(); }
        ;

compstmt_end:
      '}'
            { STAT_TRACE(("{compstmt_end#1}")); $$ = exprList(EC_COMP_STMT);
                EXPR_C($$)->e_lineNumInfo = popSymbolTable(); }
    | opt_label_decls compstmt_contents_nonempty '}'
            { STAT_TRACE(("{compstmt_end#2}"));
                $$ = exprListCons($1, $2);
                EXPR_C($$)->e_lineNumInfo = popSymbolTable(); }
    ;

compstmt_contents_nonempty:
      stmts_and_decls
            { STAT_TRACE(("{compstmt_contents_nonempty#1}"));
                $$ = $1; EXPR_CODE($$) = EC_COMP_STMT; }
    | error
            { STAT_TRACE(("{compstmt_contents_nonempty#2}"));
                $$ = exprList1(EC_COMP_STMT, exprError()); }
    ;

compstmt_expr_start:
    '(' '{'
            { STAT_TRACE(("{compstmt_expr_start#1}")); pushSymbolTable(); }
        ;

compstmt: compstmt_start compstmt_end
            { STAT_TRACE(("{compstmt#1}")); $$ = $2; }
    ;

condition:
      loc condition_1
            { $$ = $2; }
    ;

condition_1:
      exprs
            { STAT_TRACE(("{condition#1}")); $$ = $1; }
    ;

if_stmt:
      IF block_start loc '(' condition ')' block_labeled_stmt                %prec IF
            { STAT_TRACE(("{if_stmt#1}")); popSymbolTable(); $$ = exprList2(EC_IF_STMT, $5, $7); }
    | IF block_start loc '(' condition ')' block_labeled_stmt ELSE block_labeled_stmt
            { STAT_TRACE(("{if_stmt#2}")); popSymbolTable(); $$ = exprList3(EC_IF_STMT, $5, $7, $9); }
    ;

while_stmt:
    WHILE block_start loc '(' condition ')' block_labeled_stmt
            { STAT_TRACE(("{while_stmt#1}")); popSymbolTable(); $$ = exprBinary(EC_WHILE_STMT, $5, $7); }
    ;

do_stmt:
    DO block_start loc block_labeled_stmt WHILE '(' condition ')' ';'
            { STAT_TRACE(("{do_stmt#1}")); popSymbolTable(); $$ = exprBinary(EC_DO_STMT, $4, $7); }
    ;

for_init_stmt:
      opt_exprs ';'
            { STAT_TRACE(("{for_init_stmt#1}")); $$ = $1; }
      /* Using "decl" causes a s/r conflict. */
    | decl_1
            { STAT_TRACE(("{for_init_stmt#2}")); $$ = $1; }
    ;

for_cond_expr:
      loc opt_exprs
            { STAT_TRACE(("{for_cond_expr#1}")); $$ = $2; }
    ;

for_incr_expr:
      opt_exprs
            { STAT_TRACE(("{for_incr_expr#1}")); $$ = $1; }
    ;

for_stmt:
    FOR block_start '(' for_init_stmt loc for_cond_expr ';' for_incr_expr ')' block_labeled_stmt
            { STAT_TRACE(("{for_stmt#1}")); popSymbolTable();
                $$ = exprList4(EC_FOR_STMT, $4, $6, $8, $10); }
    ;

switch_stmt:
    SWITCH block_start '(' exprs ')' block_labeled_stmt
            { STAT_TRACE(("{switch_stmt#1}")); popSymbolTable();
                $$ = exprBinary(EC_SWITCH_STMT, $4, $6); }
    ;

stmt_nocomp:
      exprs ';'
            { STAT_TRACE(("{stmt_nocomp#1}")); $$ = exprUnary(EC_EXPR_STMT, $1); }
    | if_stmt
            { STAT_TRACE(("{stmt_nocomp#2}")); $$ = $1; }
    | while_stmt
            { STAT_TRACE(("{stmt_nocomp#3}")); $$ = $1; }
    | do_stmt
            { STAT_TRACE(("{stmt_nocomp#4}")); $$ = $1; }
    | for_stmt
            { STAT_TRACE(("{stmt_nocomp#5}")); $$ = $1; }
    | switch_stmt
            { STAT_TRACE(("{stmt_nocomp#6}")); $$ = $1; }
    | BREAK ';'
            { STAT_TRACE(("{stmt_nocomp#7}")); $$ = exprUnary(EC_BREAK_STMT, NULL); }
    | CONTINUE ';'
            { STAT_TRACE(("{stmt_nocomp#8}")); $$ = exprUnary(EC_CONTINUE_STMT, NULL); }
    | RETURN ';'
            { STAT_TRACE(("{stmt_nocomp#9}")); $$ = exprUnary(EC_RETURN_STMT, NULL); }
    | RETURN exprs ';'
            { STAT_TRACE(("{stmt_nocomp#10}")); $$ = exprUnary(EC_RETURN_STMT, $2); }
    | asm_stmt
            { STAT_TRACE(("{stmt_nocomp#11}")); $$ = $1; }
    | GOTO ident ';'
            { STAT_TRACE(("{stmt_nocomp#12}")); $$ = exprUnary(EC_GOTO_STMT, $2);
                EXPR_SYMBOL($2)->e_symType = ST_LABEL; }
    | GOTO '*' exprs ';'
            { STAT_TRACE(("{stmt_nocomp#13}")); $$ = exprUnary(EC_GOTO_STMT, exprUnary(EC_POINTER_REF, $3)); }
    | ';'
            { STAT_TRACE(("{stmt_nocomp#14}")); $$ = exprNull(); }
    | XMP_FUNC_CALL
            { STAT_TRACE(("{stmt_nocomp#15}")); $$ = $1; }
    | XMP_CRITICAL
            { STAT_TRACE(("{stmt_nocomp#16}")); $$ = $1; }
    ;

stmt:
      loc stmt_1
        { $$ = $2; }
//    | loc PRAGMA_PREFIX stmt_1
    | loc PRAGMA_PREFIX stmt
        { $$ = exprList1(EC_COMP_STMT,$3);((CExprOfList *)$$)->e_aux_info=$2;}
        // NOTE: not check nesting of XMP and OMP pragma.
    | loc PRAGMA_EXEC
        { $$ = exprList(EC_COMP_STMT);((CExprOfList *)$$)->e_aux_info=$2;}
    ;

stmt_1:
      compstmt
            { STAT_TRACE(("{stmt#1}")); $$ = $1; }
    | stmt_nocomp
            { STAT_TRACE(("{stmt#2}")); $$ = $1; }
    ;

label:
      loc label_1
            { $$ = $2; }
    ;
 
label_1:
      CASE expr ':'
            { STAT_TRACE(("{label#1}")); $$ = exprBinary(EC_CASE_LABEL, $2, NULL); }
    | CASE expr ELLIPSIS expr ':'
            { STAT_TRACE(("{label#2}")); $$ = exprBinary(EC_CASE_LABEL, $2, $4); }
    | DEFAULT ':'
            { STAT_TRACE(("{label#3}")); $$ = $1; }
            // the only attribute it makes sense after label is 'used'.
            // it will be ignored in this parser.
    | ident ':' opt_attr
            { STAT_TRACE(("{label#4}"));
                $$ = exprSetAttrPost(exprUnary(EC_LABEL, $1), $3); addSymbolInParser($1, ST_LABEL); }
    ;

opt_labels:
      /* empty */
            { STAT_TRACE(("{opt_labels#1}")); $$ = exprNull(); }
    | opt_labels label
            { STAT_TRACE(("{opt_labels#2}"));
                if(EXPR_ISNULL($1)) {
                    freeExpr($1);
                    $1 = exprList(EC_LABELS);
                }
                $$ = exprListJoin($1, $2); }
    ;

block_labeled_stmt:
      block_start opt_labels stmt
            { STAT_TRACE(("{block_labeled_stmt#1}"));
                popSymbolTable();
                CExprCodeEnum ec = EXPR_CODE($3);
                if(EXPR_ISNULL($2)) {
                    freeExpr($2);
                    if(ec == EC_STMTS_AND_DECLS)
                        EXPR_CODE($3) = EC_COMP_STMT;
                    else if(ec != EC_COMP_STMT)
                        $3 = exprList1(EC_COMP_STMT, $3);
                    $$ = $3;
                } else {
                    if(ec == EC_STMTS_AND_DECLS)
                        EXPR_CODE($3) = EC_COMP_STMT;
                    else
                        $3 = exprList1(EC_COMP_STMT, $3);
                    $$ = exprListCons($2, $3);
                }
            }
    ;

/* function parameter */

params:
      opt_attr params_na
            { STAT_TRACE(("{params#1}")); $$ = exprSetAttrHeadNode($2, $1);  }
    ;

params_na:
      inner_params_1 ')'
            { STAT_TRACE(("{params_na#1}")); $$ = $1; }
    | inner_params ';' opt_attr params_na
            { STAT_TRACE(("{params_na#2}"));
                $$ = exprListJoin($1, exprSetAttrHeadNode($4, $3)); }
    | error ')'
            { STAT_TRACE(("{params_na#3}")); $$ = exprList1(EC_PARAMS, exprError()); }
    ;

inner_params_1:
      /* empty */
            { STAT_TRACE(("{inner_params_1#1}")); $$ = exprList(EC_PARAMS); }
    | ELLIPSIS
            { STAT_TRACE(("{inner_params_1#2}"));
                $$ = exprList1(EC_PARAMS, lexAllocExprCode(EC_ELLIPSIS, 0)); }
    | inner_params
            { STAT_TRACE(("{inner_params_1#3}")); $$ = $1; }
    | inner_params ',' ELLIPSIS
            { STAT_TRACE(("{inner_params_1#4}"));
                $$ = exprListJoin($1, lexAllocExprCode(EC_ELLIPSIS, 0)); }
    ;

inner_params:
      param_head
            { STAT_TRACE(("{inner_params#1}")); $$ = exprList1(EC_PARAMS, $1); }
    | inner_params ',' param
            { STAT_TRACE(("{inner_params#2}")); $$ = exprListJoin($1, $3); }
    ;

param:
      declspecs_xtxx param_declarator opt_attr
            { STAT_TRACE(("{param#1}")); $$ = exprSetAttrPost(exprBinary(EC_PARAM, $1, $2), $3); }
    | declspecs_xtxx param_codeclarator opt_attr
            { STAT_TRACE(("{param#1-co}")); $$ = exprSetAttrPost(exprBinary(EC_PARAM, $1, $2), $3); }
    | declspecs_xtxx nt_declarator opt_attr
            { STAT_TRACE(("{param#2}")); $$ = exprSetAttrPost(exprBinary(EC_PARAM, $1, $2), $3); }
    | declspecs_xtxx nt_codeclarator opt_attr
            { STAT_TRACE(("{param#2-co}")); $$ = exprSetAttrPost(exprBinary(EC_PARAM, $1, $2), $3); }
    | declspecs_xtxx absdecl_opt_a
            { STAT_TRACE(("{param#3}")); $$ = exprBinary(EC_PARAM, $1, $2); }
    | declspecs_xTxx nt_declarator opt_attr
            { STAT_TRACE(("{param#4}")); $$ = exprSetAttrPost(exprBinary(EC_PARAM, $1, $2), $3); }
    | declspecs_xTxx nt_codeclarator opt_attr
            { STAT_TRACE(("{param#4-co}")); $$ = exprSetAttrPost(exprBinary(EC_PARAM, $1, $2), $3); }
    | declspecs_xTxx absdecl_opt_a
            { STAT_TRACE(("{param#5}")); $$ = exprBinary(EC_PARAM, $1, $2); }
    ;

param_head:
      declspecs_xtAx param_declarator opt_attr
            { STAT_TRACE(("{param_head#1}")); $$ = exprBinary(EC_PARAM, $1, exprSetAttrPost($2, $3)); }
    | declspecs_xtAx param_codeclarator opt_attr
            { STAT_TRACE(("{param_head#1-co}")); $$ = exprBinary(EC_PARAM, $1, exprSetAttrPost($2, $3)); }
    | declspecs_xtAx nt_declarator opt_attr
            { STAT_TRACE(("{param_head#2}")); $$ = exprBinary(EC_PARAM, $1, exprSetAttrPost($2, $3)); }
    | declspecs_xtAx nt_codeclarator opt_attr
            { STAT_TRACE(("{param_head#2-co}")); $$ = exprBinary(EC_PARAM, $1, exprSetAttrPost($2, $3)); }
    | declspecs_xtAx absdecl_opt_a
            { STAT_TRACE(("{param_head#3}")); $$ = exprBinary(EC_PARAM, $1, $2); }
    | declspecs_xTAx nt_declarator opt_attr
            { STAT_TRACE(("{param_head#4}")); $$ = exprBinary(EC_PARAM, $1, exprSetAttrPost($2, $3)); }
    | declspecs_xTAx nt_codeclarator opt_attr
            { STAT_TRACE(("{param_head#4-co}")); $$ = exprBinary(EC_PARAM, $1, exprSetAttrPost($2, $3)); }
    | declspecs_xTAx absdecl_opt_a
            { STAT_TRACE(("{param_head#5}")); $$ = exprBinary(EC_PARAM, $1, $2); }
    ;

params_or_idents:
      opt_attr params_or_idents_na
            { STAT_TRACE(("{params_or_idents#1}")); $$ = exprSetAttrHeadNode($2, $1); }
    ;

params_or_idents_na:
      params_na
            { STAT_TRACE(("{params_or_idents_na#1}")); $$ = $1; }
    | idents ')'
            { STAT_TRACE(("{params_or_idents_na#2}")); $$ = $1; }
    ;

idents:
      IDENTIFIER
            { STAT_TRACE(("{idents#1}"));
                $$ = exprList1(EC_IDENTS, $1); }
    | idents ',' IDENTIFIER
            { STAT_TRACE(("{idents#2}"));
                $$ = exprListJoin($1, $3); }
    ;

label_idents:
      ident
            { STAT_TRACE(("{label_idents#1}")); $$ = exprList1(EC_GCC_LABEL_IDENTS, $1); }
    | label_idents ',' ident
            { STAT_TRACE(("{label_idents#2}")); $$ = exprListJoin($1, $3); }
    ;

/* loc and dummy inhibit r/r conflicts */
loc:
      /* empty */
    ;

dummy:
      /* empty */
    ;

/* gcc attribute */

attrs:
      attr
            { STAT_TRACE(("{attrs#1}")); $$ = exprList1(EC_GCC_ATTRS, $1); }
    | attrs attr
            { STAT_TRACE(("{attrs#2}")); $$ = exprListJoin($1, $2); }
    ;

attr:
      ATTRIBUTE '(' '(' attr_args ')' ')'
            { STAT_TRACE(("{attr#1}")); $$ = $4; }
    ;

opt_attr:
      /* empty */
            { STAT_TRACE(("{opt_attr#1}")); $$ = NULL; }
    | attrs
            { STAT_TRACE(("{opt_attr#2}")); $$ = $1; }
    ;

opt_resetattrs:
      opt_attr
            { STAT_TRACE(("{opt_resetattrs#1}")); $$ = $1; }
    ;

attr_args:
      attr_arg
            { STAT_TRACE(("{attr_args#1}")); $$ = exprList1(EC_GCC_ATTR_ARGS, $1); }
    | attr_args ',' attr_arg
            { STAT_TRACE(("{attr_args#2}")); $$ = exprListJoin($1, $3); }
    ;

attr_arg:
      attr_arg_1
            { STAT_TRACE(("{attr_arg#1}")); $$ = $1; }

attr_arg_1:
      /* empty */
            { STAT_TRACE(("{attr_arg_1#1}")); $$ = NULL; }
    | any_word
            { STAT_TRACE(("{attr_arg_1#2}")); $$ = exprBinary(EC_GCC_ATTR_ARG, $1, NULL); }
    | any_word '(' opt_exprs ')' // exprs but arguments (not comman expr)
            { STAT_TRACE(("{attr_arg_1#3}")); $$ = exprBinary(EC_GCC_ATTR_ARG, $1, $3); }

any_word:
      ident
            { STAT_TRACE(("{any_word#1}")); $$ = $1; }
    | scspec
            { STAT_TRACE(("{any_word#2}")); freeExpr($1); $$ = lexAllocSymbol(); }
    | TYPESPEC
            { STAT_TRACE(("{any_word#3}")); freeExpr($1); $$ = lexAllocSymbol(); }
    | TYPEQUAL
            { STAT_TRACE(("{any_word#4}")); freeExpr($1); $$ = lexAllocSymbol(); }
    ;

/* gcc assembler expression */

asm_expr:
      ASSEMBLER '(' asm_string ')'
            { STAT_TRACE(("{asm_expr#1}")); $$ = exprUnary(EC_GCC_ASM_EXPR, $3); }
    ;

opt_asm_expr:
      /* empty */
            { STAT_TRACE(("{opt_asm_expr#1}")); $$ = NULL; }
    | asm_expr
            { STAT_TRACE(("{opt_asm_expr#2}")); $$ = $1; }
    ;

asm_def:
      asm_expr ';'
            { STAT_TRACE(("{asm_def#1}")); $$ = $1; }
    | ASSEMBLER error ';'
            { STAT_TRACE(("{asm_def#2}")); $$ = exprUnary(EC_GCC_ASM_EXPR, exprError()); }
    ;

asm_stmt:
      ASSEMBLER opt_volatile '(' asm_argument ')' ';'
            { STAT_TRACE(("{asm_stmt#1}")); $$ = exprBinary(EC_GCC_ASM_STMT, $2, $4); }
    ;

asm_argument:
      asm_string
            { STAT_TRACE(("{asm_argument#1}"));
                $$ = exprList1(EC_GCC_ASM_ARG, $1); }
    | asm_string opt_asm_operands
            { STAT_TRACE(("{asm_argument#2}"));
                if(EXPR_ISNULL($2)) { freeExpr($2); $2 = exprList(EC_GCC_ASM_OPES); }
                $$ = exprList2(EC_GCC_ASM_ARG, $1, $2); }
    | asm_string opt_asm_operands opt_asm_operands
            { STAT_TRACE(("{asm_argument#3}"));
                if(EXPR_ISNULL($2)) { freeExpr($2); $2 = exprList(EC_GCC_ASM_OPES); }
                if(EXPR_ISNULL($3)) { freeExpr($3); $3 = exprList(EC_GCC_ASM_OPES); }
                $$ = exprList3(EC_GCC_ASM_ARG, $1, $2, $3); }
    | asm_string opt_asm_operands opt_asm_operands ':' asm_clobbers // asm_clobbers is not optional here
            { STAT_TRACE(("{asm_argument#4}"));
                if(EXPR_ISNULL($2)) { freeExpr($2); $2 = exprList(EC_GCC_ASM_OPES); }
                if(EXPR_ISNULL($3)) { freeExpr($3); $3 = exprList(EC_GCC_ASM_OPES); }
                $$ = exprList4(EC_GCC_ASM_ARG, $1, $2, $3, $5); }
    ;

opt_volatile:
      /* empty */
            { STAT_TRACE(("{opt_volatile#1}")); $$ = NULL; }
    | TYPEQUAL
            { STAT_TRACE(("{opt_volatile#2}")); $$ = $1;
                if(EXPR_GENERALCODE($1)->e_code != TQ_VOLATILE) addError($1, CERR_001); }
    ;

opt_asm_operands:
      ':'
            { STAT_TRACE(("{opt_asm_operands#1}")); $$ = NULL; }
    | asm_operands
            { STAT_TRACE(("{opt_asm_operands#2}")); $$ = $1; }
    ;

asm_operands:
      asm_operand_1
            { STAT_TRACE(("{asm_operands#1}")); $$ = exprList1(EC_GCC_ASM_OPES, $1); }
    | asm_operands ',' asm_operand
            { STAT_TRACE(("{asm_operands#2}")); $$ = exprListJoin($1, $3); }
    ;

asm_operand_1:
      ':' asm_string '(' exprs ')'
            { STAT_TRACE(("{asm_operand_1#1}")); $$ = exprList3(EC_GCC_ASM_OPE, exprNull(), $2, $4); }
    | COLON_SQBRACKET ident ']' asm_string '(' exprs ')'
            { STAT_TRACE(("{asm_operand_1#2}")); $$ = exprList3(EC_GCC_ASM_OPE, $2, $4, $6);
                EXPR_SYMBOL($2)->e_symType = ST_GCC_ASM_IDENT; }
    ;

asm_operand:
      asm_string '(' exprs ')'
            { STAT_TRACE(("{asm_operand#1}")); $$ = exprList3(EC_GCC_ASM_OPE, exprNull(), $1, $3); }
    | '[' ident ']' asm_string '(' exprs ')'
            { STAT_TRACE(("{asm_operand#2}")); $$ = exprList3(EC_GCC_ASM_OPE, $2, $4, $6);
                EXPR_SYMBOL($2)->e_symType = ST_GCC_ASM_IDENT; }
    ;

asm_clobbers:
      asm_string
            { STAT_TRACE(("{asm_clobbers#1}")); $$ = exprList1(EC_GCC_ASM_CLOBS, $1); }
    | asm_clobbers ',' asm_string
            { STAT_TRACE(("{asm_clobbers#2}")); $$ = exprListJoin($1, $3);  }
    ;

asm_string:
      string
            { STAT_TRACE(("{asm_string#1}")); $$ = $1; }
    ;

/* directive */
directive:
      DIRECTIVE
            { STAT_TRACE(("{directive#1}")); $$ = $1; }
    | PRAGMA_PACK
            { STAT_TRACE(("{directive#2}")); $$ = $1; }
    ;

%%

void
initParser()
{
}

#ifdef MTRACE
void
freeParser()
{
#ifndef YYBISON
    free(yyss);
    free(yyvs);
#endif
}
#endif

CExpr*
execParse(FILE *fp)
{
    if(s_verbose)
        printf("parsing ...\n");

    initLexer(fp);
    initParser();

    pushSymbolTable();
    s_isParsing = 1;
    yyparse();
    s_isParsing = 0;
    freeSymbolTableList();
#ifdef MTRACE
    freeParser();
    freeLexer();
#endif
    return s_exprStart;
}



