/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file C-expr.h
 */

/* 
 * internal expression data structure 
 */
#ifndef _C_EXPR_H_
#define _C_EXPR_H_

#include "C-exprcode.h"

typedef int64_t     omllint_t;
typedef long double omldouble_t;

#if ADDR_IS_64 == 0
    #define OMLL_DFMT "%lld"
    #define OMLL_XFMT "%llx"
#else
    #define OMLL_DFMT "%ld"
    #define OMLL_XFMT "%lx"
#endif

enum symbol_type {
    S_IDENT = 0,        /* default */
    S_INTR              /* intrinsic */
};

/* symbol and symbol table */
typedef struct symbol
{
    struct symbol *s_next;              /* backet chain */
    char *s_name;
    enum symbol_type s_type;
    short int s_value;
} * SYMBOL;
#define SYM_NEXT(sp)    ((sp)->s_next)
#define SYM_NAME(sp)    ((sp)->s_name)
#define SYM_TYPE(sp)    ((sp)->s_type)
#define SYM_VAL(sp)     ((sp)->s_value)

extern SYMBOL	find_symbol(const char *name);
extern SYMBOL	find_symbol_without_allocate (const char *name);

extern int endlineno_flag;

typedef struct {
    int ln_no;
    int end_ln_no;
    int file_id;
} lineno_info;

/* de-syntax program is represented by this data structure. */
typedef struct expression_node
{
    enum expr_code e_code;
    lineno_info *e_line;                /* line number this node created */
    struct type_descriptor *e_type;     /* used for expv */
    int is_rvalue;                      /* used for expv, to determine
                                           rvalue (not lhs) or not. */
    const char *keyword_opt;		/* A name of keyword option
                                         * for function/subroutine
                                         * call argument and IO
                                         * statement. Used for expv.*/
    const char *e_original_token;       /* original token in lexer for
                                         * number constant */
    struct external_symbol *entry_ext_id;
                                        /* EXT_ID for entry. Used for expv. */
    union {
        void        *e_gen;
        struct list_node *e_lp;
        SYMBOL      e_sym;      /* e_code == NAME, TYPE, ...  */
        char        *e_str;     /* e_code == STRING */
        omllint_t   e_llval;    /* e_code == INT_CONSTANT */
        omldouble_t e_lfval;    /* e_code == FLOAT_CONSTANT */
    } v;
} * expr;

#define EXPR_GEN(x)     ((x)->v.e_gen)
#define EXPR_CODE(x)    ((x)->e_code)
#define EXPR_SYM(x)     ((x)->v.e_sym)
#define EXPR_STR(x)     ((x)->v.e_str)
#define EXPR_FLOAT(x)   ((x)->v.e_lfval)
#define EXPR_LIST(x)    ((x)->v.e_lp)
#define EXPR_INT(x)     ((x)->v.e_llval)
#define EXPR_TYPE(x)    ((BASIC_DATA_TYPE)(EXPR_INT(x)))
#define EXPR_SCLASS(x)  ((STORAGE_CLASS)(EXPR_INT(x)))
#define EXPR_TYPE_QUAL(x)       ((TYPE_QUAL)(EXPR_INT(x)))
#define EXPR_LINE_NO(x) ((x)->e_line->ln_no)
#define EXPR_END_LINE_NO(x) ((x)->e_line->end_ln_no)
#define EXPR_LINE_FILE_ID(x)    ((x)->e_line->file_id)
#define EXPR_LINE(x)    ((x)->e_line)
#define EXPR_ORIGINAL_TOKEN(x)  ((x)->e_original_token)

/* list data structure, which is ended with NULL */
typedef struct list_node
{
    struct list_node *l_next;
    expr l_item;
    struct list_node *l_last;
    int l_nItems;
    struct list_node **l_array;
} *list;

#define LIST_NEXT(lp)   ((lp)->l_next)
#define LIST_ITEM(lp)   ((lp)->l_item)
#define LIST_LAST(lp)   ((lp)->l_last)
#define LIST_N_ITEMS(lp)        ((lp)->l_nItems)
#define LIST_ARRAY(lp)  ((lp)->l_array)
#define FOR_ITEMS_IN_LIST(lp,x) \
  if(x != NULL) for(lp = EXPR_LIST(x); lp != NULL ; lp = LIST_NEXT(lp))
#define EXPR_LIST1(x)   EXPR_LIST(x)
#define EXPR_LIST2(x)   LIST_NEXT(EXPR_LIST1(x))
#define EXPR_LIST3(x)   LIST_NEXT(EXPR_LIST2(x))
#define EXPR_LIST4(x)   LIST_NEXT(EXPR_LIST3(x))
#define EXPR_LIST5(x)   LIST_NEXT(EXPR_LIST4(x))
#define EXPR_ARG1(x)    LIST_ITEM(EXPR_LIST1(x))
#define EXPR_ARG2(x)    LIST_ITEM(EXPR_LIST2(x))
#define EXPR_ARG3(x)    LIST_ITEM(EXPR_LIST3(x))
#define EXPR_ARG4(x)    LIST_ITEM(EXPR_LIST4(x))
#define EXPR_ARG5(x)    LIST_ITEM(EXPR_LIST5(x))
#define EXPR_HAS_ARG1(x)    (EXPR_LIST1(x) != NULL)
#define EXPR_HAS_ARG2(x)    (EXPR_HAS_ARG1(x) && EXPR_LIST2(x) != NULL)
#define EXPR_HAS_ARG3(x)    (EXPR_HAS_ARG2(x) && EXPR_LIST3(x) != NULL)
#define EXPR_HAS_ARG4(x)    (EXPR_HAS_ARG3(x) && EXPR_LIST4(x) != NULL)
#define EXPR_HAS_ARG5(x)    (EXPR_HAS_ARG4(x) && EXPR_LIST5(x) != NULL)

/* typed value returned as a result of evaluation. */
typedef expr expv;

#define EXPV_GEN(x)     ((x)->v.e_gen)
#define EXPV_CODE(x)    ((x)->e_code)
#define EXPV_TYPE(x)    ((x)->e_type)
#define EXPV_IS_RVALUE(x)       ((x)->is_rvalue)
#define EXPV_LIST(x)    ((x)->v.e_lp)
#define EXPV_LEFT(x)    ((x)->v.e_lp->l_item)
#define EXPV_RIGHT(x) \
((x)->v.e_lp->l_next != NULL?(x)->v.e_lp->l_next->l_item:NULL)
#define EXPV_NAME(x)    ((x)->v.e_sym)
#define EXPV_STR(x)     ((x)->v.e_str)
#define EXPV_ENTRY_EXT_ID(x)	((x)->entry_ext_id)
#define EXPV_PROC_EXT_ID(x)     ((x)->entry_ext_id)
#define EXPV_INT_VALUE(x)       EXPV_LLINT_VALUE(x)
#define EXPV_FLOAT_VALUE(x)     ((x)->v.e_lfval)
#define EXPV_LLINT_VALUE(x)     ((x)->v.e_llval)
#define EXPV_ANY(t,x)           ((t)((x)->v.e_gen))
#define EXPV_KWOPT_NAME(x)      ((x)->keyword_opt)
#define EXPV_KW_IS_KIND(x) \
    ((EXPV_KWOPT_NAME(x) != NULL) && \
     (strcasecmp(EXPV_KWOPT_NAME(x), "kind") == 0))
#define EXPV_LINE(x)            ((x)->e_line)
#define EXPV_LINE_NO(x)         (EXPV_LINE(x)->ln_no)
#define EXPV_END_LINE_NO(x)     (EXPV_LINE(x)->end_ln_no)
#define EXPV_LINE_FILE_ID(x)    (EXPV_LINE(x)->file_id)
#define EXPV_ORIGINAL_TOKEN(x)  ((x)->e_original_token)
#define EXPV_COMPLEX_REAL(x)    EXPV_LEFT(x)
#define EXPV_COMPLEX_IMAG(x)    EXPV_RIGHT(x)

extern struct expr_code_info 
{
    char code_info;
    char *code_name;
    char *operator_name;
} expr_code_info[];

#define EXPR_CODE_NAME(code)    expr_code_info[code].code_name
#define EXPR_CODE_SYMBOL(code)   expr_code_info[code].operator_name

/* code info */
/* T : terminal
 * L : list
 * B : binary operator
 * U : unary opertor 
 */
#define EXPR_CODE_INFO(code)    expr_code_info[code].code_info
#define EXPR_CODE_IS_TERMINAL(code) (expr_code_info[code].code_info == 'T')
#define EXPR_CODE_IS_LIST(code) (expr_code_info[code].code_info == 'L')
#define EXPR_CODE_IS_BINARY(code) (expr_code_info[code].code_info == 'B')
#define EXPR_CODE_IS_UNARY(code) (expr_code_info[code].code_info == 'U')

#define EXPR_CODE_IS_TERMINAL_OR_CONST(code) \
((EXPR_CODE_IS_TERMINAL(code)) || \
 ((code) == COMPLEX_CONSTANT))

#define EXPR_CODE_IS_CONSTANT(x) \
    (EXPR_CODE(x) == INT_CONSTANT || EXPR_CODE(x) == FLOAT_CONSTANT || \
    EXPR_CODE(x) == F_TRUE_CONSTANT || EXPR_CODE(x) == F_FALSE_CONSTANT || \
    EXPR_CODE(x) == STRING_CONSTANT || \
    EXPR_CODE(x) == COMPLEX_CONSTANT)

#if __WORDSIZE == 64
#define ADDR_FMT  "%016lx"
#else
#define ADDR_FMT  "%08x"
#endif

#endif /* _C_EXPR_H_ */

