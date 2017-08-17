/**
 * \file F-front.h
 */

/* Fortran language front-end for Omni */

#ifndef _F_FRONT_H_
#define _F_FRONT_H_

#define OMNI_FORTRAN_ENTRY_POINT        "__Omni_FortranEntryPoint"

#include "config.h"
#include <assert.h>
#if SIZEOF_UNSIGNED_INT == 4
# define HAS_INT32 1
#endif

#if SIZEOF_UNSIGNED_SHORT == 2
# define HAS_INT16 1
#endif

#include "exc_platform.h"

#if !defined(SIZEOF_UNSIGNED_LONG_LONG)
#define SIZEOF_UNSIGNED_LONG_LONG       (SIZEOF_UNSIGNED_LONG * 2)
#endif /* !SIZEOF_UNSIGNED_LONG_LONG */

#define TRUE    1
#define FALSE   0

#ifdef SIMPLE_TYPE
extern int Addr2Uint(void *x);
#define ADDRX_PRINT_FMT  "%d"
#else /* SIMPLE_TYPE */
#define Addr2Uint(X) ((uintptr_t)(X))

#if __WORDSIZE == 64
#define ADDRX_PRINT_FMT  "%lx"
#else
#define ADDRX_PRINT_FMT  "%x"
#endif
#endif /* SIMPLE_TYPE */

/*
 * Safe for the case if iter(p) (i.e. ID_NEXT(ip), EXT_ID_NEXT(ep)) is
 * overwritten in the for clause.
 */
#define SAFE_FOREACH(p, q, list, iter)     \
    for ((p) = (list), (q) = (p)?iter(p):NULL;\
         (p) != NULL;\
         (p) = (q), (q) = (p)?iter(q):NULL)

#include "C-expr.h"
#include "F-datatype.h"
#include "F-ident.h"
#include <inttypes.h>

#include "C-OMP.h"  /* OpenMP */
#include "C-XMP.h"  /* XcalableMP */
#include "C-ACC.h"  /* OpenACC */

extern int lineno;
extern int need_keyword;
extern int enable_need_type_keyword;
extern int need_type_keyword;
extern int need_do_keyword;
extern int need_type_len;
extern int need_check_user_defined;

extern int lookup_col2(void);

extern BASIC_DATA_TYPE defaultSingleRealType;
extern BASIC_DATA_TYPE defaultDoubleRealType;
extern BASIC_DATA_TYPE defaultIntType;

extern int doImplicitUndef;
extern int nerrors;

/* max nam length */
#define MAX_NAME_LEN_F77          31    /* limitation of fortran 77, 90 */
#define MAX_NAME_LEN_F03          63    /* limitation of fortran 2003 */
#define MAX_NAME_LEN_UPPER_LIMIT 256   /* upper bound */
#define MAX_DIM  15

/* max line length */
#define DEFAULT_MAX_LINE_LEN_FIXED 72
#define DEFAULT_MAX_LINE_LEN_FREE 255

extern unsigned long int maxStackSize;
extern int debug_flag;
extern FILE *debug_fp;
extern FILE *diag_file;

/* max file path length */
#define MAX_PATH_LEN    8192

#define FILE_NAME_LEN   MAX_PATH_LEN
#define MAX_N_FILES 256
#define N_NESTED_FILE 25
extern int n_files;
extern char *file_names[];
#define FILE_NAME(id) file_names[id]

/* exit codes */
#define EXITCODE_OK             0
#define EXITCODE_ERR            (-1)

extern lineno_info *current_line;
extern lineno_info *new_line_info(int f_id,int ln);
extern int get_file_id(char *name);

extern char *source_file_name,*output_file_name;
extern FILE *source_file,*output_file;

/* max number of include search path.  */
#define MAXINCLUDEDIRV 256

extern int fixed_format_flag;
extern char *includeDirv[MAXINCLUDEDIRV + 1];
extern int includeDirvI;
extern const char * search_include_path(const char *);

/* max number of include search path.  */
#define MAXMODINCLUDEDIRV 256

extern char * xmoduleIncludeDirv;

#include <libgen.h>

/* parser states */
enum prog_state {
    OUTSIDE,
    INSIDE,
    INDCL,
    INDATA,
    INEXEC,
    INSTRUCT,
    INCONT,     /* contains */
    ININTR,      /* interface */
    IN_TYPE_PARAM_DECL, /* type parameter declarations */
    IN_TYPE_BOUND_PROCS, /* type bound procedure declarations */
};

/* procedure states */
enum procedure_state { P_DEFAULT = 0, P_SAVE = 1 };
extern enum procedure_state current_proc_state;

/* macro for parser state */
#define NOT_INDATA_YET  ((int)CURRENT_STATE < (int)INDATA)
#define INDCL_OVER      ((int)CURRENT_STATE >= (int)INDATA)

/* module visible state */
enum module_state { M_DEFAULT, M_PUBLIC, M_PRIVATE, M_PROTECTED };
extern enum module_state current_module_state;

/* control stack codes */
enum control_type {
    CTL_NONE = 0,
    CTL_DO,
    CTL_IF,
    CTL_ELSE,
    CTL_WHERE,
    CTL_ELSE_WHERE,
    CTL_SELECT,
    CTL_CASE,
    CTL_SELECT_TYPE,
    CTL_TYPE_GUARD,
    CTL_STRUCT,
    CTL_OMP,
    CTL_XMP,
    CTL_ACC,
    CTL_CRITICAL,
    CTL_BLOCK,
    CTL_INTERFACE,
    CTL_FORALL,
};

#define CONTROL_TYPE_NAMES {\
    "CTL_NONE",\
    "CTL_DO",\
    "CTL_IF",\
    "CTL_ELSE",\
    "CTL_WHERE",\
    "CTL_ELSE_WHERE",\
    "CTL_SELECT",\
    "CTL_CASE",\
    "CTL_STRUCT",\
    "CTL_OMP",\
    "CTL_XMP",\
    "CTL_ACC",\
    "CTL_CRITICAL",\
    "CTL_BLOCK",\
    "CTL_INTERFACE",\
    "CTL_FORALL",\
}

typedef struct environment {
    ID                  symbols;
    TYPE_DESC           struct_decls;
    ID                  common_symbols;
    ID                  labels;
    EXT_ID              external_symbols;
    EXT_ID              interfaces;
    BLOCK_ENV           blocks;
    expv                use_decls;

    struct environment * parent;
} *ENV;

#define ENV_SYMBOLS(l)               ((l)->symbols)
#define ENV_STRUCT_DECLS(l)          ((l)->struct_decls)
#define ENV_COMMON_SYMBOLS(l)        ((l)->common_symbols)
#define ENV_LABELS(l)                ((l)->labels)
#define ENV_EXTERNAL_SYMBOLS(l)      ((l)->external_symbols)
#define ENV_INTERFACES(l)            ((l)->interfaces)
#define ENV_BLOCKS(l)                ((l)->blocks)
#define ENV_USE_DECLS(l)             ((l)->use_decls)
#define ENV_PARENT(l)                ((l)->parent)

extern ENV current_local_env;

extern void push_env(ENV);
extern void pop_env(void);

/* control */
typedef struct control
{
    enum control_type ctltype;
    expv save;
    expv v1,v2;
    ID dolabel;
    SYMBOL dovar;

    /* FOR BLOCK STATEMENT */
    struct environment local_env;

    struct control * next;
    struct control * prev;
} * CTL;

#define CTL_TYPE(l)             ((l)->ctltype)
#define CTL_SAVE(l)             ((l)->save)
#define CTL_BLOCK(l)            ((l)->v1)
#define CTL_CLIENT(l)           ((l)->v2)

#define CTL_IF_STATEMENT(l)     ((l)->v2)
#define CTL_IF_THEN(l)          (EXPR_ARG2((l)->v2))
#define CTL_IF_ELSE(l)          (EXPR_ARG3((l)->v2))
#define CTL_DO_BODY(l)          (EXPR_ARG5(EXPR_ARG2((l)->v1)))
#define CTL_DO_LABEL(l)         ((l)->dolabel)
#define CTL_DO_VAR(l)           ((l)->dovar)
#define CTL_DO_LOCAL_ENV(l)     (&((l)->local_env))

#define CTL_STRUCT_TYPEDESC(l)  (EXPV_TYPE((l)->v1))

#define CTL_CRIT_STATEMENT(l)   ((l)->v2)
#define CTL_CRIT_BODY(l)        (EXPR_ARG1((l)->v2))
#define CTL_CRIT_CONST_NAME(l)  (EXPR_ARG2((l)->v2))

#define CTL_WHERE_STATEMENT(l)     ((l)->v2)
#define CTL_WHERE_THEN(l)          (EXPR_ARG2((l)->v2))
#define CTL_WHERE_ELSE(l)          (EXPR_ARG3((l)->v2))

#define CTL_SELECT_STATEMENT_BODY(l)    (EXPR_ARG2((l)->v1))
#define CTL_SELECT_CONST_NAME(l)        (EXPR_ARG3((l)->v1))
#define CTL_SELECT_TYPE_SELECTOR(l)     (EXPR_ARG1((l)->v1))
#define CTL_SELECT_TYPE_ASSICIATE(l)    (EXPR_ARG4((l)->v1))
#define CTL_CASE_BLOCK(l)               (EXPR_ARG2((l)->v1))
#define CTL_CASE_CONST_NAME(l)          (EXPR_ARG3((l)->v1))
#define CTL_TYPE_GUARD_BLOCK(l)         (EXPR_ARG2((l)->v1))
#define CTL_TYPE_GUARD_CONST_NAME(l)    (EXPR_ARG3((l)->v1))
#define CTL_TYPE_GUARD_LOCAL_ENV(l)     (&((l)->local_env))

#define CTL_OMP_ARG(l)	((l)->v2)
#define CTL_OMP_ARG_DIR(l) (EXPR_INT(EXPR_ARG1((l)->v2)))
#define CTL_OMP_ARG_PCLAUSE(l) (EXPR_ARG2((l)->v2))
#define CTL_OMP_ARG_DCLAUSE(l) (EXPR_ARG3((l)->v2))

#define CTL_XMP_ARG(l)	((l)->v2)
#define CTL_XMP_ARG_DIR(l) (EXPR_INT(EXPR_ARG1((l)->v2)))
#define CTL_XMP_ARG_CLAUSE(l) (EXPR_ARG2((l)->v2))

#define CTL_ACC_ARG(l)	((l)->v2)
#define CTL_ACC_ARG_DIR(l) (EXPR_INT(EXPR_ARG1((l)->v2)))
#define CTL_ACC_ARG_CLAUSE(l) (EXPR_ARG2((l)->v2))

#define CTL_BLOCK_STATEMENT(l)                   ((l)->v2)
#define CTL_BLOCK_BODY(l)                        (EXPR_ARG1((l)->v2))
#define CTL_BLOCK_CONST_NAME(l)                  (EXPR_ARG2((l)->v2))
#define CTL_BLOCK_LOCAL_ENV(l)                   (&((l)->local_env))
#define CTL_BLOCK_LOCAL_SYMBOLS(l)               ((CTL_BLOCK_LOCAL_ENV(l))->symbols)
#define CTL_BLOCK_LOCAL_STRUCT_DECLS(l)          ((CTL_BLOCK_LOCAL_ENV(l))->struct_decls)
#define CTL_BLOCK_LOCAL_COMMON_SYMBOLS(l)        ((CTL_BLOCK_LOCAL_ENV(l))->common_symbols)
#define CTL_BLOCK_LOCAL_LABELS(l)                ((CTL_BLOCK_LOCAL_ENV(l))->labels)
#define CTL_BLOCK_LOCAL_EXTERNAL_SYMBOLS(l)      ((CTL_BLOCK_LOCAL_ENV(l))->external_symbols)
#define CTL_BLOCK_LOCAL_BLOCKS(l)                ((CTL_BLOCK_LOCAL_ENV(l))->blocks)

#define CTL_FORALL_STATEMENT(l)                   ((l)->v2)
#define CTL_FORALL_HEADER(l)                      (EXPR_ARG1((l)->v1))
#define CTL_FORALL_INIT(l)                        (EXPR_ARG1(CTL_FORALL_HEADER(l)))
#define CTL_FORALL_MASK(l)                        (EXPR_ARG2(CTL_FORALL_HEADER(l)))
#define CTL_FORALL_BODY(l)                        (EXPR_ARG2((l)->v1))
#define CTL_FORALL_CONST_NAME(l)                  (EXPR_ARG3((l)->v1))
#define CTL_FORALL_LOCAL_ENV(l)                   (&((l)->local_env))
#define CTL_FORALL_LOCAL_SYMBOLS(l)               ((CTL_FORALL_LOCAL_ENV(l))->symbols)
#define CTL_FORALL_LOCAL_STRUCT_DECLS(l)          ((CTL_FORALL_LOCAL_ENV(l))->struct_decls)
#define CTL_FORALL_LOCAL_COMMON_SYMBOLS(l)        ((CTL_FORALL_LOCAL_ENV(l))->common_symbols)
#define CTL_FORALL_LOCAL_LABELS(l)                ((CTL_FORALL_LOCAL_ENV(l))->labels)
#define CTL_FORALL_LOCAL_EXTERNAL_SYMBOLS(l)      ((CTL_FORALL_LOCAL_ENV(l))->external_symbols)
#define CTL_FORALL_LOCAL_BLOCKS(l)                ((CTL_FORALL_LOCAL_ENV(l))->blocks)



#define CTL_NEXT(u)               ((u)->next)
#define CTL_PREV(u)               ((u)->prev)

/* control stack and it pointer */
#define MAX_CTL 50
extern CTL ctl_base;
extern CTL ctl_top;
extern CTL current_ctp;

#define FOR_CTLS(cp) \
    for (cp = ctl_base; \
         cp != ctl_top && \
                 (CTL_NEXT(cp) != NULL && CTL_TYPE(CTL_NEXT(cp)) != CTL_NONE); \
         cp = CTL_NEXT(cp))

#define FOR_CTLS_BACKWARD(cp) \
    for (cp = ctl_top; CTL_PREV(cp) != NULL; cp = CTL_PREV(cp))


#define MAX_REPLACE_ITEMS       100

extern struct replace_item {
    ID id;
    expv v;
} replace_stack[],*replace_sp;

struct eqv_set {
    struct eqv_list *next,*parent;
    ID id;
    int high,bottom,offset;
};

#define EQV_NEXT(ep)    ((ep)->next)
#define EQV_PARENT(ep)  ((ep)->parent)
#define EQV_LIST(ep)    ((ep)->parent)

#define EQV_ID(ep)      ((ep)->id)
#define EQV_HIGH(ep)    ((ep)->high)
#define EQV_LOW(ep)     ((ep)->low)
#define EQV_OFFSET(ep)  ((ep)->offset)

#define IMPLICIT_ALPHA_NUM      26
/* program unit control stack struct */
typedef struct {
    SYMBOL              current_proc_name;
    enum name_class     current_proc_class;
    ID                  current_procedure;
    expv                current_statements;
    int                 current_blk_level;
    EXT_ID              current_ext_id;
    enum prog_state     current_state;
    EXT_ID              current_interface;

    struct environment  local_env;

    int                 implicit_none;
    int                 implicit_type_declared;
    TYPE_DESC           implicit_types[IMPLICIT_ALPHA_NUM];
    enum storage_class  implicit_stg[IMPLICIT_ALPHA_NUM];
    expv                implicit_decls;
    expv                initialize_decls;
    expv                equiv_decls;
    /* FOR INTERFACE */
    struct control * save_ctl;
    struct control * save_ctl_base;
} *UNIT_CTL;

#define UNIT_CTL_CURRENT_PROC_NAME(u)           ((u)->current_proc_name)
#define UNIT_CTL_CURRENT_PROC_CLASS(u)          ((u)->current_proc_class)
#define UNIT_CTL_CURRENT_PROCEDURE(u)           ((u)->current_procedure)
#define UNIT_CTL_CURRENT_STATEMENTS(u)          ((u)->current_statements)
#define UNIT_CTL_CURRENT_BLK_LEVEL(u)           ((u)->current_blk_level)
#define UNIT_CTL_CURRENT_EXT_ID(u)              ((u)->current_ext_id)
#define UNIT_CTL_CURRENT_STATE(u)               ((u)->current_state)
#define UNIT_CTL_CURRENT_INTERFACE(u)           ((u)->current_interface)

#define UNIT_CTL_LOCAL_ENV(u)                   (&((u)->local_env))
#define UNIT_CTL_LOCAL_SYMBOLS(u)               ((UNIT_CTL_LOCAL_ENV(u))->symbols)
#define UNIT_CTL_LOCAL_STRUCT_DECLS(u)          ((UNIT_CTL_LOCAL_ENV(u))->struct_decls)
#define UNIT_CTL_LOCAL_COMMON_SYMBOLS(u)        ((UNIT_CTL_LOCAL_ENV(u))->common_symbols)
#define UNIT_CTL_LOCAL_LABELS(u)                ((UNIT_CTL_LOCAL_ENV(u))->labels)
#define UNIT_CTL_LOCAL_EXTERNAL_SYMBOLS(u)      ((UNIT_CTL_LOCAL_ENV(u))->external_symbols)
#define UNIT_CTL_LOCAL_BLOCKS(u)                ((UNIT_CTL_LOCAL_ENV(u))->blocks)
#define UNIT_CTL_LOCAL_INTERFACES(u)            ((UNIT_CTL_LOCAL_ENV(u))->interfaces)
#define UNIT_CTL_LOCAL_USE_DECLS(u)             ((UNIT_CTL_LOCAL_ENV(u))->use_decls)

#define UNIT_CTL_IMPLICIT_NONE(u)               ((u)->implicit_none)
#define UNIT_CTL_IMPLICIT_TYPES(u)              ((u)->implicit_types)
#define UNIT_CTL_IMPLICIT_TYPE_DECLARED(u)      ((u)->implicit_type_declared)
#define UNIT_CTL_IMPLICIT_STG(u)                ((u)->implicit_stg)
#define UNIT_CTL_IMPLICIT_DECLS(u)              ((u)->implicit_decls)
#define UNIT_CTL_INITIALIZE_DECLS(u)            ((u)->initialize_decls)
#define UNIT_CTL_EQUIV_DECLS(u)                 ((u)->equiv_decls)

#define UNIT_CTL_INTERFACE_SAVE_CTL(u)          ((u)->save_ctl)
#define UNIT_CTL_INTERFACE_SAVE_CTL_BASE(u)     ((u)->save_ctl_base)

#define MAX_UNIT_CTL            16
#define MAX_UNIT_CTL_CONTAINS   3
extern UNIT_CTL unit_ctls[];
extern int unit_ctl_level;

#define CURRENT_UNIT_CTL            unit_ctls[unit_ctl_level]
#define PARENT_UNIT_CTL             unit_ctls[unit_ctl_level-1]

#define CURRENT_PROC_NAME           UNIT_CTL_CURRENT_PROC_NAME(CURRENT_UNIT_CTL)
#define CURRENT_PROC_CLASS          UNIT_CTL_CURRENT_PROC_CLASS(CURRENT_UNIT_CTL)
#define CURRENT_PROCEDURE           UNIT_CTL_CURRENT_PROCEDURE(CURRENT_UNIT_CTL)
#define CURRENT_STATEMENTS          UNIT_CTL_CURRENT_STATEMENTS(CURRENT_UNIT_CTL)
#define CURRENT_BLK_LEVEL           UNIT_CTL_CURRENT_BLK_LEVEL(CURRENT_UNIT_CTL)
#define CURRENT_EXT_ID              UNIT_CTL_CURRENT_EXT_ID(CURRENT_UNIT_CTL)
#define CURRENT_STATE               UNIT_CTL_CURRENT_STATE(CURRENT_UNIT_CTL)
#define CURRENT_INTERFACE           UNIT_CTL_CURRENT_INTERFACE(CURRENT_UNIT_CTL)
#define CURRENT_INITIALIZE_DECLS    UNIT_CTL_INITIALIZE_DECLS(CURRENT_UNIT_CTL)

#define LOCAL_SYMBOLS               (current_local_env->symbols)
#define LOCAL_STRUCT_DECLS          (current_local_env->struct_decls)
#define LOCAL_COMMON_SYMBOLS        (current_local_env->common_symbols)
#define LOCAL_LABELS                (current_local_env->labels)
#define LOCAL_EXTERNAL_SYMBOLS      (current_local_env->external_symbols)
#define LOCAL_BLOCKS                (current_local_env->blocks)
#define LOCAL_INTERFACES            (current_local_env->interfaces)
#define LOCAL_USE_DECLS             (current_local_env->use_decls)

#define EXTERNAL_SYMBOLS            LOCAL_EXTERNAL_SYMBOLS
#define IMPLICIT_TYPES              UNIT_CTL_IMPLICIT_TYPES(CURRENT_UNIT_CTL)
#define IMPLICIT_STG                UNIT_CTL_IMPLICIT_STG(CURRENT_UNIT_CTL)

#define PARENT_EXT_ID               UNIT_CTL_CURRENT_EXT_ID(PARENT_UNIT_CTL)
#define PARENT_STATE                UNIT_CTL_CURRENT_STATE(PARENT_UNIT_CTL)
#define PARENT_CONTAINS             EXT_PROC_CONT_EXT_SYMS(PARENT_EXT_ID)
#define PARENT_INTERFACE            UNIT_CTL_CURRENT_INTERFACE(PARENT_UNIT_CTL)
#define PARENT_PROC_CLASS           UNIT_CTL_CURRENT_PROC_CLASS(PARENT_UNIT_CTL)

#define PARENT_LOCAL_SYMBOLS        (current_local_env->parent->symbols)
#define PARENT_LOCAL_STRUCT_DECLS   (current_local_env->parent->struct_decls)
#define PARENT_LOCAL_COMMON_SYMBOLS (current_local_env->parent->common_symbols)
#define PARENT_LOCAL_LABELS         (current_local_env->parent->labels)
#define PARENT_EXTERNAL_SYMBOLS     (current_local_env->parent->external_symbols)
#define PARENT_LOCAL_BLOCKS         (current_local_env->parent->blocks)
#define PARENT_LOCAL_INTERFACES     (current_local_env->parent->interfaces)
#define PARENT_LOCAL_USE_DECLS      (current_local_env->parent->use_decls)

/*
 * Language specification level. Mainly used for intrinsic table
 * initialization.
 */
#define LANGSPEC_UNKNOWN            0x0
#define LANGSPEC_F77                0x1
#define LANGSPEC_F90                0x2
#define LANGSPEC_F95                0x4
#define LANGSPEC_F2003              0x8
#define LANGSPEC_F2008              0x10
#define LANGSPEC_NONSTD             0x20

#define LANGSPEC_F77_STRICT_SET     LANGSPEC_F77
#define LANGSPEC_F90_STRICT_SET     (LANGSPEC_F77_STRICT_SET | LANGSPEC_F90)
#define LANGSPEC_F95_STRICT_SET     (LANGSPEC_F90_STRICT_SET | LANGSPEC_F95)
#define LANGSPEC_F2003_STRICT_SET   (LANGSPEC_F95_STRICT_SET | LANGSPEC_F2003)
#define LANGSPEC_F2008_STRICT_SET   (LANGSPEC_F2003_STRICT_SET | LANGSPEC_F2008)
#define LANGSPEC_F77_SET            (LANGSPEC_F77_STRICT_SET | LANGSPEC_NONSTD)
#define LANGSPEC_F90_SET            (LANGSPEC_F90_STRICT_SET | LANGSPEC_NONSTD)
#define LANGSPEC_F95_SET            (LANGSPEC_F95_STRICT_SET | LANGSPEC_NONSTD)
#define LANGSPEC_F2003_SET          (LANGSPEC_F2003_STRICT_SET | LANGSPEC_NONSTD)
#define LANGSPEC_F2008_SET          (LANGSPEC_F2008_STRICT_SET | LANGSPEC_NONSTD)
#define LANGSPEC_DEFAULT_SET        LANGSPEC_F2008_SET
extern int langSpecSet;

extern ID this_label;

extern TYPE_DESC type_REAL, type_INT, type_SUBR, type_CHAR, type_LOGICAL;
extern TYPE_DESC type_DREAL, type_COMPLEX, type_DCOMPLEX, type_CHAR_POINTER;
extern TYPE_DESC type_VOID;
extern TYPE_DESC type_MODULE;
extern TYPE_DESC type_GNUMERIC_ALL;
extern expv expv_constant_1,expv_constant_0,expv_constant_m1;
extern expv expv_float_0;

extern int OMP_flag;
extern int XMP_flag;
extern int XMP_coarray_flag;
extern int ACC_flag;
extern int PRAGMA_flag;
extern int cond_compile_enabled;
extern int leave_comment_flag;

#define EMPTY_LIST list0(LIST)

#define NAME_FOR_NONAME_PROGRAM "xmpf_main"

/*
 * I/O information specifiers
 */

#define IO_SPEC_UNIT    0
#define IO_SPEC_FMT     1
#define IO_SPEC_REC     2
#define IO_SPEC_IOSTAT  3
#define IO_SPEC_ERR     4
#define IO_SPEC_END     5

#define IO_SPEC_FILE    6
#define IO_SPEC_STATUS  7
#define IO_SPEC_ACCESS  8
#define IO_SPEC_FORM    9
#define IO_SPEC_RECL    10
#define IO_SPEC_BLANK   11

#define IO_SPEC_EXIST   12
#define IO_SPEC_OPENED  13
#define IO_SPEC_NUMBER  14
#define IO_SPEC_NAMED   15
#define IO_SPEC_NAME    16
#define IO_SPEC_SEQUENTIAL      17
#define IO_SPEC_DIRECT  18
#define IO_SPEC_FORMATTED       19
#define IO_SPEC_UNFORMATTED     20
#define IO_SPEC_NEXTREC 21
#define IO_SPEC_NML     22

#define IO_SPEC_UNKNOWN 23


#define NOT_YET() \
    fatal("%s: not implemented yet (%s:%d)", __func__, __FILE__, __LINE__)

/*
 * FIXME:
 *	SUPER BOGUS FLAG ALERT !
 */
extern int is_in_kind_compilation_flag_for_declare_ident;

/*
 * prototype
 */
extern char *   xmalloc _ANSI_ARGS_((int size));
#define XMALLOC(type, size) ((type)xmalloc(size))

extern void     error EXC_VARARGS(char *, fmt);
extern void     fatal EXC_VARARGS(char *, fmt);
extern void     warning EXC_VARARGS(char *, fmt);
extern void     warning_lineno ( lineno_info * info, char * fmt, ...);
extern void     error_at_node EXC_VARARGS(expr, x);
extern void     error_at_id EXC_VARARGS(ID, x);
extern void     warning_at_node EXC_VARARGS(expr, x);
extern void     warning_at_id EXC_VARARGS(ID, x);
extern void     debug EXC_VARARGS(char *, fmt);

extern void     initialize_lex _ANSI_ARGS_((void));
extern void     initialize_compile _ANSI_ARGS_((void));
extern void     finalize_compile _ANSI_ARGS_((void));
extern void     initialize_compile_procedure _ANSI_ARGS_((void));

extern int      output_X_file _ANSI_ARGS_((void));

extern void     expr_print _ANSI_ARGS_((expr x, FILE *fp));
extern void	expr_print_indent(expr x, int i, FILE *fp);
extern void     print_type _ANSI_ARGS_((TYPE_DESC tp, FILE *fp,
                                        int recursive));

extern void     compile_statement _ANSI_ARGS_((int st_no, expr x));
extern void     compile_statement1 _ANSI_ARGS_((int st_no, expr x));
extern void     output_statement _ANSI_ARGS_((expr v));
extern CTL      new_ctl _ANSI_ARGS_((void));
extern void     cleanup_ctl _ANSI_ARGS_((CTL));
extern void     push_ctl _ANSI_ARGS_((enum control_type ctl));
extern void     pop_ctl _ANSI_ARGS_((void));
extern void     check_INDATA _ANSI_ARGS_((void));
extern void     check_INDCL _ANSI_ARGS_((void));
extern void     check_INEXEC _ANSI_ARGS_((void));
extern void     check_NOT_INBLOCK _ANSI_ARGS_((void));
extern void     include_file(char *name, int inside_use);
extern void     push_unit_ctl _ANSI_ARGS_((enum prog_state));
extern void     pop_unit_ctl _ANSI_ARGS_((void));
extern SYMBOL   gen_temp_symbol(const char *leader);

extern void     set_implicit_type_uc _ANSI_ARGS_((UNIT_CTL uc, TYPE_DESC tp, int c1, int c2, int ignore_declared_flag));
extern void     set_implicit_type _ANSI_ARGS_((TYPE_DESC tp, int c1, int c2));
extern void     set_implicit_storage_uc _ANSI_ARGS_((UNIT_CTL uc, enum storage_class stg, int c1, int c2));
extern void     set_implicit_storage _ANSI_ARGS_((enum storage_class stg, int c1, int c2));
extern void     output_expr_statement _ANSI_ARGS_((expr v));

extern ID       declare_label _ANSI_ARGS_((int st_no, LABEL_TYPE type, int def_flag));
extern ID       declare_variable _ANSI_ARGS_((ID id));
extern void     declare_procedure
                _ANSI_ARGS_((enum name_class class,
                             expr name,
                             TYPE_DESC type, expr args, expr prefix_spec,
                             expr result_opt, expr bind_opt));
extern EXT_ID   declare_current_procedure_ext_id(void);

extern void     compile_type_decl _ANSI_ARGS_((expr typExpre, TYPE_DESC baseTp,
                                               expr decl_list, expr attributes));
extern void     compile_struct_decl _ANSI_ARGS_((expr ident, expr type, expr type_params));
extern void     compile_struct_decl_end _ANSI_ARGS_((void));
extern int      compile_type_param_values_dummy _ANSI_ARGS_((TYPE_DESC struct_type, expr type_param_args));
extern int      compile_type_param_values _ANSI_ARGS_((TYPE_DESC struct_type, expr type_param_args, expv type_param_values, ID * used));
extern ID       get_type_params _ANSI_ARGS_((TYPE_DESC struct_type));
extern void     compile_SEQUENCE_statement _ANSI_ARGS_((void));
extern void     compile_COMMON_decl _ANSI_ARGS_((expr com_list));
extern void     compile_IMPLICIT_decl _ANSI_ARGS_((expr v1,expr v2));
extern void     compile_PARAM_decl _ANSI_ARGS_((expr const_list));
extern void	postproc_PARAM_decl _ANSI_ARGS_((expr ident, expr e));

extern expv     compile_logical_expression _ANSI_ARGS_((expr x));
extern expv     compile_logical_expression_with_array _ANSI_ARGS_((expr x));
extern void     declare_statement_function _ANSI_ARGS_((ID id, expr args, expr body));
extern expv     compile_ident_expression _ANSI_ARGS_((expr x));
extern expv     compile_lhs_expression _ANSI_ARGS_((expr x));
extern expv     compile_substr_ref _ANSI_ARGS_((expr x));
extern int      substr_length(expv x);
extern int      expv_is_lvalue _ANSI_ARGS_((expv v));
extern int      expv_is_str_lvalue _ANSI_ARGS_((expv v));

extern expv     compile_terminal_node _ANSI_ARGS_((expr x));
extern expv     compile_expression _ANSI_ARGS_((expr x));
extern expv     expv_assignment _ANSI_ARGS_((expv v1, expv v2));
extern expv     compile_args _ANSI_ARGS_((expr args));
extern expv     compile_function_call _ANSI_ARGS_((ID f_id, expr args));
extern expv     compile_function_call_check_intrinsic_arg_type _ANSI_ARGS_((ID f_id, expr args, int ignoreTypeMismatch));
extern expv     compile_highorder_function_call _ANSI_ARGS_((ID f_id,
                                                             expr args,
                                                             int isCall));

extern int      type_param_values_required _ANSI_ARGS_((TYPE_DESC struct_type));
extern expv     compile_struct_constructor _ANSI_ARGS_((ID struct_id, expr type_param_argss, expr args));
extern TYPE_DESC type_apply_type_parameter _ANSI_ARGS_((TYPE_DESC tp, expv type_param_values));

extern expv     statement_function_call _ANSI_ARGS_((ID f_id, expv arglist));
extern TYPE_DESC        compile_dimensions _ANSI_ARGS_((TYPE_DESC tp, expr dims));
extern codims_desc* compile_codimensions _ANSI_ARGS_((expr dims, int is_in_alloc));
extern void     fix_array_dimensions _ANSI_ARGS_((TYPE_DESC tp));
extern TYPE_DESC        copy_array_type(TYPE_DESC tp);
extern TYPE_DESC        copy_dimension(TYPE_DESC array_tp, TYPE_DESC base);
extern expv     compare_dimensions _ANSI_ARGS_((expv lDim, expr rDim));

extern void     print_IDs _ANSI_ARGS_((ID ip, FILE *fp, int recursive));
extern void     print_EXT_IDs _ANSI_ARGS_((EXT_ID ep, FILE *fp));
extern void     print_interface_IDs _ANSI_ARGS_((ID id, FILE *fd));
extern void     print_types _ANSI_ARGS_((TYPE_DESC tp, FILE *fp));
extern void     type_output _ANSI_ARGS_((TYPE_DESC tp, FILE *fp));
extern void     expv_output _ANSI_ARGS_((expv x, FILE *fp));
extern void     print_controls _ANSI_ARGS_((FILE *fp));
extern expv     expv_char_len _ANSI_ARGS_((expv v));
extern expv     convertSubstrRefToPointerRef _ANSI_ARGS_((expv org, expv *lenVPtr));

extern ID       declare_function _ANSI_ARGS_((ID id));
extern ID       declare_ident _ANSI_ARGS_((SYMBOL s, enum name_class class));
extern ID       declare_common_ident _ANSI_ARGS_((SYMBOL s));
extern ID       find_ident_head _ANSI_ARGS_((SYMBOL s, ID head));
extern ID       find_ident _ANSI_ARGS_((SYMBOL s));
extern ID       find_ident_local _ANSI_ARGS_((SYMBOL s));
extern ID       find_ident_block_parent _ANSI_ARGS_((SYMBOL s));
extern ID       find_ident_outer_scope _ANSI_ARGS_((SYMBOL s));
extern ID       find_ident_parent _ANSI_ARGS_((SYMBOL s));
extern ID       find_ident_sibling _ANSI_ARGS_((SYMBOL s));
extern ID       find_struct_member _ANSI_ARGS_((TYPE_DESC struct_td, SYMBOL sym));
extern ID       find_struct_member_allow_private _ANSI_ARGS_((TYPE_DESC struct_td, SYMBOL sym, int allow_private));
extern int      type_is_parent_type _ANSI_ARGS_((TYPE_DESC parent, TYPE_DESC child));
extern int      type_is_unlimited_class _ANSI_ARGS_((TYPE_DESC tp));
extern int      type_is_class_of _ANSI_ARGS_((TYPE_DESC class, TYPE_DESC derived_type));
extern int      compare_derived_type_name _ANSI_ARGS_((TYPE_DESC tp1, TYPE_DESC tp2));
extern int      type_is_nopolymorphic_abstract _ANSI_ARGS_((TYPE_DESC tp));


extern ID       find_type _ANSI_ARGS_((TYPE_DESC struct_td, SYMBOL sym));
extern ID       find_external_ident_head _ANSI_ARGS_((SYMBOL s));
extern EXT_ID   find_ext_id_head _ANSI_ARGS_((SYMBOL s, EXT_ID head));
extern EXT_ID   find_ext_id _ANSI_ARGS_((SYMBOL s));
extern EXT_ID   find_ext_id_parent _ANSI_ARGS_((SYMBOL s));
extern EXT_ID   find_ext_id_sibling _ANSI_ARGS_((SYMBOL s));
extern int      char_length _ANSI_ARGS_((TYPE_DESC tp));
extern TYPE_DESC  find_struct_decl_head _ANSI_ARGS_((SYMBOL s, TYPE_DESC head));
extern TYPE_DESC  find_struct_decl _ANSI_ARGS_((SYMBOL s));
extern TYPE_DESC  find_struct_decl_parent _ANSI_ARGS_((SYMBOL s));
extern TYPE_DESC  find_struct_decl_sibling _ANSI_ARGS_((SYMBOL s));
extern int      id_link_remove _ANSI_ARGS_((ID * head, ID tobeRemoved));

extern void     initialize_intrinsic _ANSI_ARGS_((void));
extern int      is_intrinsic_function _ANSI_ARGS_((ID id));
extern expv     compile_intrinsic_call _ANSI_ARGS_((ID id,expv args));
extern expv     compile_intrinsic_call0 _ANSI_ARGS_((ID, expv, int));
extern void     generate_shape_expr _ANSI_ARGS_((TYPE_DESC tp, expv dimSpec));

extern EXT_ID   declare_external_proc_id _ANSI_ARGS_((SYMBOL s, TYPE_DESC tp,
                                                      int def_flag));
extern EXT_ID   declare_external_id _ANSI_ARGS_((SYMBOL s,
                                                 enum storage_class tag,
                                                 int def_flag));

extern void     unset_save_attr_in_dummy_args(EXT_ID ep);

extern void     declare_storage _ANSI_ARGS_((ID id, enum storage_class stg));

extern void     id_multilize _ANSI_ARGS_((ID id));
extern ID       multi_find_class  _ANSI_ARGS_((ID id, enum name_class class));


extern TYPE_DESC        compile_type _ANSI_ARGS_((expr x, int allow_predecl));
extern TYPE_DESC        compile_basic_type _ANSI_ARGS_((expr x));
extern TYPE_DESC        compile_derived_type _ANSI_ARGS_((expr x, int allow_predecl));

extern expv     compile_int_constant _ANSI_ARGS_((expr x));
extern void     compile_pragma_statement _ANSI_ARGS_((expr x));
extern void     compile_VOLATILE_statement _ANSI_ARGS_((expr id_list));
extern void     compile_VALUE_statement _ANSI_ARGS_((expr id_list));
extern void     compile_ASYNCHRONOUS_statement _ANSI_ARGS_((expr id_list));
extern void     compile_CONTIGUOUS_statement _ANSI_ARGS_((expr id_list));

extern void     compile_procedure_declaration _ANSI_ARGS_((expr x));
extern void     compile_type_bound_procedure _ANSI_ARGS_((expr x));
extern void     compile_type_bound_generic_procedure _ANSI_ARGS_((expr x));
extern void     compile_FINAL_statement _ANSI_ARGS_((expr x));
#define FINALIZER_PROCEDURE "_final"
extern void     update_type_bound_procedures_forall _ANSI_ARGS_((TYPE_DESC struct_decls, ID local_symbols));
extern int      type_bound_procedure_type_match _ANSI_ARGS_((EXT_ID f1, EXT_ID f2, int has_pass_arg));
extern int      is_procedure_acceptable _ANSI_ARGS_((EXT_ID proc, expv actual_args));

extern int      type_is_soft_compatible _ANSI_ARGS_((TYPE_DESC tp, TYPE_DESC tq));
extern int      type_is_strict_compatible _ANSI_ARGS_((TYPE_DESC tp, TYPE_DESC tq, int issue_error));
extern int      type_is_compatible_for_assignment
                    _ANSI_ARGS_((TYPE_DESC tp1, TYPE_DESC tp2));
extern int      type_is_compatible_for_allocation
                    _ANSI_ARGS_((TYPE_DESC left, TYPE_DESC right));
extern int      struct_type_is_compatible_for_assignment
                    _ANSI_ARGS_((TYPE_DESC tp1, TYPE_DESC tp2, int is_pointer_set));
extern int      type_is_specific_than
                    _ANSI_ARGS_((TYPE_DESC tp1, TYPE_DESC tp2));
extern void     function_type_udpate
                    _ANSI_ARGS_((TYPE_DESC ftp, ID idList));
extern int      function_type_is_appliable
                    _ANSI_ARGS_((TYPE_DESC ftp, expv args, int issue_error));
extern int      function_type_is_compatible
                    _ANSI_ARGS_((const TYPE_DESC ftp1, const TYPE_DESC ftp2));
extern int      type_bound_procedure_types_are_compatible
                    _ANSI_ARGS_((const TYPE_DESC tbp1, const TYPE_DESC tbp2));

extern int      check_tbp_pass_arg(TYPE_DESC stp, TYPE_DESC tbp, TYPE_DESC ftp);

extern int      procedure_has_pass_arg
                    _ANSI_ARGS_((const TYPE_DESC ftp,
                                 const SYMBOL pass_arg,
                                 const TYPE_DESC stp));
extern int      procedure_is_assignable
                    _ANSI_ARGS_((const TYPE_DESC left, const TYPE_DESC right));

extern void     replace_or_assign_type
                    _ANSI_ARGS_((TYPE_DESC *tp, const TYPE_DESC new_tp));

extern int      are_dimension_and_shape_conformant_by_type _ANSI_ARGS_((
    expr x, TYPE_DESC lt, TYPE_DESC rt, expv *shapePtr, int issue_error));

extern TYPE_DESC
	get_binary_numeric_intrinsic_operation_type(TYPE_DESC t0,
                                                    TYPE_DESC t1);
extern TYPE_DESC
	get_binary_comparative_intrinsic_operation_type(TYPE_DESC t0,
                                                        TYPE_DESC t1);
extern TYPE_DESC
	get_binary_equal_intrinsic_operation_type(TYPE_DESC t0,
                                                  TYPE_DESC t1);

extern int      type_is_possible_dreal(TYPE_DESC tp);
extern int      is_array_size_adjustable(TYPE_DESC tp);
extern int      is_array_size_const(TYPE_DESC tp);
extern expv     expv_power_expr _ANSI_ARGS_((expv left, expv right));

extern TYPE_DESC        max_type _ANSI_ARGS_((TYPE_DESC tp1, TYPE_DESC tp2));
extern expv             max_shape _ANSI_ARGS_((expv lshape, expv rshape, int select));
extern int              is_variable_shape _ANSI_ARGS_((expv shape));
extern int              score_array_spec _ANSI_ARGS_((expv aSpec));
extern expv             combine_array_specs _ANSI_ARGS_((expv l, expv r));
extern int              array_spec_size _ANSI_ARGS_((expv shape, expv dimShape,
                                                expv *whichSPtr));
extern void             set_index_range_type _ANSI_ARGS_((expv v));
extern TYPE_DESC        type_ref _ANSI_ARGS_((TYPE_DESC tp));
extern TYPE_DESC        struct_type  _ANSI_ARGS_((ID id));
extern TYPE_DESC        function_type _ANSI_ARGS_((TYPE_DESC tp));
extern TYPE_DESC        subroutine_type _ANSI_ARGS_((void));
extern TYPE_DESC        generic_procedure_type _ANSI_ARGS_((void));
extern TYPE_DESC        generic_function_type _ANSI_ARGS_((void));
extern TYPE_DESC        generic_subroutine_type _ANSI_ARGS_((void));
extern TYPE_DESC        intrinsic_function_type _ANSI_ARGS_((TYPE_DESC tp));
extern TYPE_DESC        intrinsic_subroutine_type _ANSI_ARGS_((void));
extern TYPE_DESC        program_type _ANSI_ARGS_((void));
extern TYPE_DESC        type_bound_procedure_type _ANSI_ARGS_((void));
extern TYPE_DESC        procedure_type _ANSI_ARGS_((const TYPE_DESC ftp));
extern TYPE_DESC        type_char _ANSI_ARGS_((int len));
extern TYPE_DESC        type_basic _ANSI_ARGS_((BASIC_DATA_TYPE t));
extern TYPE_DESC        array_element_type _ANSI_ARGS_((TYPE_DESC tp));

extern expv     compile_array_ref _ANSI_ARGS_((ID id, expv ary, expr args, int isLeft));
extern expv     compile_coarray_ref _ANSI_ARGS_((expr coarrayRef));
extern expv     expv_type_conversion _ANSI_ARGS_((TYPE_DESC tp, expv v));

extern expv     expv_complex_op _ANSI_ARGS_((enum expr_code op, TYPE_DESC tp, expv left, expv right));
extern expv     expv_c_cons _ANSI_ARGS_((expv left, expv right, int doInline));
extern expv     expv_z_cons _ANSI_ARGS_((expv left, expv right, int doInline));
extern expv     expv_complex_node_to_variable _ANSI_ARGS_((expv v, TYPE_DESC tp));
extern expv     expv_complex_const_reduce _ANSI_ARGS_((expv v, TYPE_DESC tp));

extern expv     expv_reduce _ANSI_ARGS_((expv v, int doParamReduce));
extern expv     expv_reduce_kind _ANSI_ARGS_((expv v));
extern expv     expv_float_reduce _ANSI_ARGS_((expv v));
extern expv     expv_reduce_conv_const _ANSI_ARGS_((TYPE_DESC tp, expv v));
extern expv     expv_inline_function _ANSI_ARGS_((expv left, expv right));
extern omllint_t power_ii(omllint_t x, omllint_t n);

extern char *   basic_type_name _ANSI_ARGS_((BASIC_DATA_TYPE t));
extern char *   name_class_name _ANSI_ARGS_((enum name_class c));
extern char *   proc_class_name _ANSI_ARGS_((enum proc_class c));
extern char *   storage_class_name _ANSI_ARGS_((enum storage_class c));
extern char *   control_type_name _ANSI_ARGS_((enum control_type c));

extern expv     expv_cons _ANSI_ARGS_((enum expr_code code, TYPE_DESC tp, expv left, expv right));
extern expv     expv_user_def_cons _ANSI_ARGS_((enum expr_code code, TYPE_DESC tp, expv id, expv left, expv right));
extern expv     expv_sym_term _ANSI_ARGS_((enum expr_code code, TYPE_DESC tp, SYMBOL name));
extern expv     expv_str_term _ANSI_ARGS_((enum expr_code code, TYPE_DESC tp, char *str));
extern expv     expv_int_term _ANSI_ARGS_((enum expr_code code, TYPE_DESC tp, omllint_t i));
extern expv     expv_any_term _ANSI_ARGS_((enum expr_code code, void *p));
extern expv     expv_float_term _ANSI_ARGS_((enum expr_code code, TYPE_DESC tp, omldouble_t d, const char *token));
extern expv     expv_retype _ANSI_ARGS_((TYPE_DESC tp, expv v));

extern expr     list0 _ANSI_ARGS_((enum expr_code code));
extern expr     list1 _ANSI_ARGS_((enum expr_code code, expr x1));
extern expr     list2 _ANSI_ARGS_((enum expr_code code, expr x1, expr x2));
extern expr     list3 _ANSI_ARGS_((enum expr_code code, expr x1, expr x2, expr x3));
extern expr     list4 _ANSI_ARGS_((enum expr_code code, expr x1, expr x2, expr x3, expr x4));
extern expr     list5 _ANSI_ARGS_((enum expr_code code, expr x1, expr x2, expr x3, expr x4, expr x5));
extern expr     list6 _ANSI_ARGS_((enum expr_code code, expr x1, expr x2, expr x3, expr x4, expr x5, expr x6));

extern expr     expr_list_get_n _ANSI_ARGS_((expr x, int n));
extern int      expr_list_set_n _ANSI_ARGS_((expr x, int n, expr val, int doOverride));
extern int      expr_list_length _ANSI_ARGS_((expr x));

extern expr     list_cons _ANSI_ARGS_((expr v, expr w));
extern expr     list_put_last _ANSI_ARGS_((expr lx, expr x));
extern expr     list_delete_item _ANSI_ARGS_((expr lx, expr x));
extern expr     list_concat _ANSI_ARGS_((expr lx, expr ly));
extern expr     list_last _ANSI_ARGS_((expr lx));
extern expr     list_replace_last _ANSI_ARGS_((expr lx, expr x));

extern void     delete_list _ANSI_ARGS_((expr lx));

extern expr     make_enode _ANSI_ARGS_((enum expr_code code, void *v));
extern expr     make_int_enode _ANSI_ARGS_((omllint_t i));
extern expr     make_float_enode _ANSI_ARGS_((enum expr_code code, omldouble_t d, const char *token));

extern ID       new_ident_desc _ANSI_ARGS_((SYMBOL sp));
extern void     id_list_put_last _ANSI_ARGS_((ID *list, ID id));
extern EXT_ID   new_external_id _ANSI_ARGS_((SYMBOL sp));
extern EXT_ID   new_external_id_for_external_decl _ANSI_ARGS_((SYMBOL sp, TYPE_DESC tp));
extern void     extid_put_last(EXT_ID base, EXT_ID to_add);
extern TYPE_DESC        new_type_desc _ANSI_ARGS_((void));

extern void     declare_id_type _ANSI_ARGS_((ID id, TYPE_DESC tp));
extern void     fix_type _ANSI_ARGS_((ID id));

extern void     compile_FORMAT_decl _ANSI_ARGS_((int st_no, expr x));
extern void     FinalizeFormat _ANSI_ARGS_((void));

extern void     compile_DATA_decl _ANSI_ARGS_((expr x));
extern void     compile_EXTERNAL_decl _ANSI_ARGS_((expr x));

extern void     compile_IO_statement _ANSI_ARGS_((expr x));
extern void     compile_OPEN_statement _ANSI_ARGS_((expr x));
extern void     compile_CLOSE_statement _ANSI_ARGS_((expr x));
extern void     compile_FPOS_statement _ANSI_ARGS_((expr x));
extern void     compile_INQUIRE_statement _ANSI_ARGS_((expr x));
extern void     compile_NAMELIST_decl _ANSI_ARGS_((expr x));
extern void     compile_WAIT_statement _ANSI_ARGS_((expr x));
extern void     compile_FLUSH_statement _ANSI_ARGS_((expr x));

extern void     compile_INTRINSIC_decl _ANSI_ARGS_((expr id_list));
extern void     compile_SAVE_decl _ANSI_ARGS_((expr id_list));
extern void     FinalizeCrayPointer _ANSI_ARGS_((void));

extern void     implicit_declaration _ANSI_ARGS_((ID id));

extern expv     NormalizeIoSpecifier _ANSI_ARGS_((expr x));
extern int      CheckIoSpecifierSanity _ANSI_ARGS_((expv ioList, int *ids, int n));
extern expr     GetIoSpecifierValue _ANSI_ARGS_((expv v, int id, int *specifiedPtr));
extern expv     Get_IOSTAT_Variable _ANSI_ARGS_((expv ioSpec, ID *vIdPtr));
extern ID       Get_ERRorEND_Label _ANSI_ARGS_((expv ioSpec, int type));
extern expv     GetIoSpecifierValueAsInteger _ANSI_ARGS_((expv ioSpec, int id, int doAddr,
                                                          int *haveItPtr));
extern expv     GetIoSpecifierValueAsIntegerVariable _ANSI_ARGS_((expv ioSpec, int id, int doAddr,
                                                                  int *haveItPtr, ID *vIdPtr));
extern expv     GetIoSpecifierValueAsLogicalVariable _ANSI_ARGS_((expv ioSpec, int id, int doAddr,
                                                                  int *haveItPtr, ID *vIdPtr));
extern expv     GetIoSpecifierValueAsString _ANSI_ARGS_((expv ioSpec, int id, int *haveItPtr,
                                                         expv *lenVPtr));
extern expv     GetIoSpecifierValueAsStringVariable _ANSI_ARGS_((expv ioSpec, int id, int *haveItPtr,
                                                                 expv *lenVPtr, ID *vIdPtr));

extern BASIC_DATA_TYPE  getBasicType _ANSI_ARGS_((TYPE_DESC tp));
extern TYPE_DESC        getBaseType _ANSI_ARGS_((TYPE_DESC tp));

extern expv     expv_get_address _ANSI_ARGS_((expv v));

extern ID       find_common_ident _ANSI_ARGS_((SYMBOL sym));
extern ID       find_common_ident_parent _ANSI_ARGS_((SYMBOL sym));
extern ID       find_common_ident_sibling _ANSI_ARGS_((SYMBOL sym));

extern ID       find_label_from_block _ANSI_ARGS_((int st_no, BLOCK_ENV block));

extern TYPE_DESC declare_struct_type_wo_component(expr ident);

extern int      expr_is_param _ANSI_ARGS_((expr x));
extern int      expr_has_param _ANSI_ARGS_((expr x));
extern int      expr_is_type_param _ANSI_ARGS_((expr x));
extern int      expr_has_type_param _ANSI_ARGS_((expr x));
extern int      expr_is_constant _ANSI_ARGS_((expr v));
extern int      expr_is_constant_typeof _ANSI_ARGS_((expr x, BASIC_DATA_TYPE bt));
extern expv     expr_constant_value _ANSI_ARGS_((expr x));
extern expv     expr_label_value _ANSI_ARGS_((expr x));
extern int      expr_is_variable _ANSI_ARGS_((expr x, int force, ID *idPtr));
extern int      expr_is_array _ANSI_ARGS_((expr x, int force, ID *idPtr));

extern expv     id_array_dimension_list _ANSI_ARGS_((ID id));
extern expv     id_array_spec_list _ANSI_ARGS_((ID id));

extern expv     expr_array_spec_list _ANSI_ARGS_((expr x, ID *idPtr));

extern int      compute_element_offset _ANSI_ARGS_((expv aSpec, expv idxV));
extern expv     expr_array_index _ANSI_ARGS_((expr x));

extern void     compile_EQUIVALENCE_decl _ANSI_ARGS_((expr x));

extern expv     ExpandImpliedDoInDATA _ANSI_ARGS_((expv spec, expv new));

extern void     compile_OMN_directive _ANSI_ARGS_((expr x));
extern void     begin_module _ANSI_ARGS_((expr name));
extern void     end_module _ANSI_ARGS_((expr name));
extern int      is_in_module(void);

extern void     begin_submodule _ANSI_ARGS_((expr name, expr module_name, expr submodule_name));
extern void     flatten_submodule_units _ANSI_ARGS_((void));
extern void     end_submodule _ANSI_ARGS_((expr name));

extern omllint_t getExprValue(expv v);

extern EXT_ID    define_external_function_id _ANSI_ARGS_((ID id));
extern ID        declare_function_result_id(SYMBOL s, TYPE_DESC tp);

/* functions for converting enum to string */
extern char *   basic_type_name(BASIC_DATA_TYPE t);
extern char *   name_class_name(enum name_class c);
extern char *   proc_class_name(enum proc_class c);
extern char *   storage_class_name(enum storage_class c);
extern expv     compile_set_expr _ANSI_ARGS_((expr x));
extern expv	compile_member_ref _ANSI_ARGS_((expr x));

extern void	expr_dump(expr x);
extern void	expv_dump(expv v);
extern void	type_dump(TYPE_DESC tp);

extern BASIC_DATA_TYPE         get_basic_type(TYPE_DESC tp);

extern TYPE_DESC               bottom_type(TYPE_DESC tp); /* bottom
                                                           * type of
                                                           * array
                                                           * type. */
extern TYPE_DESC               get_bottom_ref_type(TYPE_DESC tp);
extern int                     type_is_assumed_size_array(TYPE_DESC tp);

extern TYPE_DESC               wrap_type(TYPE_DESC tp);
extern void                    merge_attributes(TYPE_DESC tp1, TYPE_DESC tp2);
extern TYPE_DESC               type_link_add(TYPE_DESC tp,
                                             TYPE_DESC tlist,
                                             TYPE_DESC ttail);
extern TYPE_DESC               copy_type_partially(TYPE_DESC tp,
                                                   int doCopyAttr);
extern int                     type_is_omissible(TYPE_DESC tp,
                                                 uint32_t attr,
                                                 uint32_t ext);
extern void                    shrink_type(TYPE_DESC tp);
extern TYPE_DESC               reduce_type(TYPE_DESC tp);

extern int is_array_shape_assumed(TYPE_DESC tp);
//extern int is_descendant_coindexed(TYPE_DESC tp);
extern int has_coarray_component(TYPE_DESC tp);

extern void     checkTypeRef(ID id);

extern int      checkInsideUse(void);
extern void     setIsOfModule(ID id);

/* inform lexer 'FUNCITION' is appearable in next line. */
extern void set_function_disappear(void);
/* inform lexer 'FUNCITION' never appear in next line. */
extern void set_function_appearable(void);

extern int      expv_is_specification(expv x);

/* create expr hold implict declaration information. */
extern expr     create_implicit_decl_expv(TYPE_DESC tp, char * first, char * second);

extern void compile_OMP_directive(expr v);
int OMP_reduction_op(expr v);
int OMP_depend_op(expr v);

extern void compile_XMP_directive(expr v);
int XMP_reduction_op(expr v);

extern void compile_ACC_directive(expr v);
int ACC_reduction_op(expr v);
int ACC_num_attr(expr v);
void ACC_check_num_attr(expr v, enum ACC_pragma attr);

#include "xcodeml-module.h"
#include "F-module-procedure.h"

#endif /* _F_FRONT_H_ */
