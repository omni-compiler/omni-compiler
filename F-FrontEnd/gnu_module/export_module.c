
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>
#include <assert.h>

#include "export_module.h"
#include "import_module.h"

/* generated from C-exprcode.def */
struct expr_code_info expr_code_info[] = {
/* 0 */ {       'T',    "ERROR_NODE",   NULL},
/* 1 */ {       'T',    "IDENT",        NULL},
/* 2 */ {       'T',    "STRING_CONSTANT",      NULL},
/* 3 */ {       'T',    "INT_CONSTANT", NULL},
/* 4 */ {       'T',    "FLOAT_CONSTANT",       NULL},
/* 5 */ {       'T',    "BASIC_TYPE_NODE",      NULL},
/* 6 */ {       'L',    "COMPLEX_CONSTANT",     NULL},
/* 7 */ {       'L',    "LIST", NULL},
/* 8 */ {       'L',    "IF_STATEMENT", NULL},
/* 9 */ {       'L',    "GOTO_STATEMENT",       NULL},
/* 10 */        {       'L',    "STATEMENT_LABEL",      NULL},
/* 11 */        {       'L',    "DEFAULT_LABEL",        NULL},
/* 12 */        {       'L',    "EXPR_STATEMENT",       NULL},
/* 13 */        {       'B',    "PLUS_EXPR",    "+"},
/* 14 */        {       'B',    "MINUS_EXPR",   "-"},
/* 15 */        {       'U',    "UNARY_MINUS_EXPR",     "-"},
/* 16 */        {       'B',    "MUL_EXPR",     "*"},
/* 17 */        {       'B',    "DIV_EXPR",     "/"},
/* 18 */        {       'B',    "POWER_EXPR",   "**"},
/* 19 */        {       'B',    "LOG_EQ_EXPR",  "=="},
/* 20 */        {       'B',    "LOG_NEQ_EXPR", "!="},
/* 21 */        {       'B',    "LOG_GE_EXPR",  ">="},
/* 22 */        {       'B',    "LOG_GT_EXPR",  ">"},
/* 23 */        {       'B',    "LOG_LE_EXPR",  "<="},
/* 24 */        {       'B',    "LOG_LT_EXPR",  "<"},
/* 25 */        {       'B',    "LOG_AND_EXPR", "&&"},
/* 26 */        {       'B',    "LOG_OR_EXPR",  "||"},
/* 27 */        {       'U',    "LOG_NOT_EXPR", "!"},
/* 28 */        {       'L',    "F_SCENE_RANGE_EXPR",   NULL},
/* 29 */        {       'B',    "ARRAY_REF",    NULL},
/* 30 */        {       'L',    "FUNCTION_CALL",        NULL},
/* 31 */        {       'T',    "F_VAR",        NULL},
/* 32 */        {       'T',    "F_PARAM",      NULL},
/* 33 */        {       'T',    "F_FUNC",       NULL},
/* 34 */        {       'T',    "ID_LIST",      NULL},
/* 35 */        {       'L',    "VAR_DECL",     NULL},
/* 36 */        {       'L',    "EXT_DECL",     NULL},
/* 37 */        {       'L',    "F_MODULE_INTERNAL",    NULL},
/* 38 */        {       'L',    "FIRST_EXECUTION_POINT",        NULL},
/* 39 */        {       'L',    "F_PROGRAM_STATEMENT",  NULL},
/* 40 */        {       'L',    "F_BLOCK_STATEMENT",    NULL},
/* 41 */        {       'L',    "F_SUBROUTINE_STATEMENT",       NULL},
/* 42 */        {       'L',    "F_FUNCTION_STATEMENT", NULL},
/* 43 */        {       'L',    "F_ENTRY_STATEMENT",    NULL},
/* 44 */        {       'L',    "F_INCLUDE_STATEMENT",  NULL},
/* 45 */        {       'L',    "F_END_STATEMENT",      NULL},
/* 46 */        {       'L',    "F_TYPE_DECL",  NULL},
/* 47 */        {       'L',    "F_COMMON_DECL",        NULL},
/* 48 */        {       'L',    "F_EXTERNAL_DECL",      NULL},
/* 49 */        {       'L',    "F_INTRINSIC_DECL",     NULL},
/* 50 */        {       'L',    "F_EQUIV_DECL", NULL},
/* 51 */        {       'L',    "F_DATA_DECL",  NULL},
/* 52 */        {       'L',    "F_IMPLICIT_DECL",      NULL},
/* 53 */        {       'L',    "F_NAMELIST_DECL",      NULL},
/* 54 */        {       'L',    "F_SAVE_DECL",  NULL},
/* 55 */        {       'L',    "F_PARAM_DECL", NULL},
/* 56 */        {       'L',    "F_FORMAT_DECL",        NULL},
/* 57 */        {       'L',    "F_DUP_DECL",   NULL},
/* 58 */        {       'L',    "F_UNARY_MINUS",        NULL},
/* 59 */        {       'L',    "F_DO_STATEMENT",       NULL},
/* 60 */        {       'L',    "F_ENDDO_STATEMENT",    NULL},
/* 61 */        {       'L',    "F_DOWHILE_STATEMENT",  NULL},
/* 62 */        {       'L',    "F_WHERE_STATEMENT",    NULL},
/* 63 */        {       'L',    "F_ELSEWHERE_STATEMENT",        NULL},
/* 64 */        {       'L',    "F_ENDWHERE_STATEMENT", NULL},
/* 65 */        {       'L',    "F_SELECTCASE_STATEMENT",       NULL},
/* 66 */        {       'L',    "F_CASELABEL_STATEMENT",        NULL},
/* 67 */        {       'L',    "F_ENDSELECT_STATEMENT",        NULL},
/* 68 */        {       'L',    "F_IF_STATEMENT",       NULL},
/* 69 */        {       'L',    "F_ELSEIF_STATEMENT",   NULL},
/* 70 */        {       'L',    "F_ELSE_STATEMENT",     NULL},
/* 71 */        {       'L',    "F_ENDIF_STATEMENT",    NULL},
/* 72 */        {       'L',    "F_LET_STATEMENT",      NULL},
/* 73 */        {       'L',    "F_ASSIGN_LABEL_STATEMENT",     NULL},
/* 74 */        {       'L',    "F_CONTINUE_STATEMENT", NULL},
/* 75 */        {       'L',    "F_GOTO_STATEMENT",     NULL},
/* 76 */        {       'L',    "F_ASGOTO_STATEMENT",   NULL},
/* 77 */        {       'L',    "F_COMPGOTO_STATEMENT", NULL},
/* 78 */        {       'L',    "F_ARITHIF_STATEMENT",  NULL},
/* 79 */        {       'L',    "F_CALL_STATEMENT",     NULL},
/* 80 */        {       'L',    "F_RETURN_STATEMENT",   NULL},
/* 81 */        {       'L',    "F_PAUSE_STATEMENT",    NULL},
/* 82 */        {       'L',    "F_STOP_STATEMENT",     NULL},
/* 83 */        {       'L',    "F_PRINT_STATEMENT",    NULL},
/* 84 */        {       'L',    "F_WRITE_STATEMENT",    NULL},
/* 85 */        {       'L',    "F_READ_STATEMENT",     NULL},
/* 86 */        {       'L',    "F_READ1_STATEMENT",    NULL},
/* 87 */        {       'L',    "F_OPEN_STATEMENT",     NULL},
/* 88 */        {       'L',    "F_CLOSE_STATEMENT",    NULL},
/* 89 */        {       'L',    "F_INQUIRE_STATEMENT",  NULL},
/* 90 */        {       'L',    "F_BACKSPACE_STATEMENT",        NULL},
/* 91 */        {       'L',    "F_ENDFILE_STATEMENT",  NULL},
/* 92 */        {       'L',    "F_REWIND_STATEMENT",   NULL},
/* 93 */        {       'L',    "F_CRAY_POINTER_DECL",  NULL},
/* 94 */        {       'L',    "F_PRAGMA_STATEMENT",   NULL},
/* 95 */        {       'F',    "F_SET_EXPR",   NULL},
/* 96 */        {       'U',    "F_LABEL_REF",  NULL},
/* 97 */        {       'B',    "F_PLUS_EXPR",  "+"},
/* 98 */        {       'B',    "F_MINUS_EXPR", "-"},
/* 99 */        {       'B',    "F_MUL_EXPR",   "*"},
/* 100 */       {       'B',    "F_DIV_EXPR",   "/"},
/* 101 */       {       'B',    "F_POWER_EXPR", "**"},
/* 102 */       {       'U',    "F_UNARY_MINUS_EXPR",   "-"},
/* 103 */       {       'B',    "F_EQ_EXPR",    ".eq."},
/* 104 */       {       'B',    "F_GT_EXPR",    ".gt."},
/* 105 */       {       'B',    "F_GE_EXPR",    ".ge."},
/* 106 */       {       'B',    "F_LT_EXPR",    ".lt."},
/* 107 */       {       'B',    "F_LE_EXPR",    ".le."},
/* 108 */       {       'B',    "F_NE_EXPR",    ".ne."},
/* 109 */       {       'B',    "F_EQV_EXPR",   ".eqv."},
/* 110 */       {       'B',    "F_NEQV_EXPR",  ".neqv."},
/* 111 */       {       'B',    "F_OR_EXPR",    ".or."},
/* 112 */       {       'B',    "F_AND_EXPR",   ".and."},
/* 113 */       {       'U',    "F_NOT_EXPR",   ".not."},
/* 114 */       {       'B',    "F_CONCAT_EXPR",        "//"},
/* 115 */       {       'B',    "F95_USER_DEFINED_BINARY_EXPR", NULL},
/* 116 */       {       'B',    "F95_USER_DEFINED_UNARY_EXPR",  NULL},
/* 117 */       {       'B',    "F_SUBSTR_REF", NULL},
/* 118 */       {       'U',    "F_ARRAY_REF",  NULL},
/* 119 */       {       'L',    "F_STARSTAR",   NULL},
/* 120 */       {       'B',    "F95_CONSTANT_WITH",    NULL},
/* 121 */       {       'T',    "F_TRUE_CONSTANT",      NULL},
/* 122 */       {       'T',    "F_FALSE_CONSTANT",     NULL},
/* 123 */       {       'U',    "F95_TRUE_CONSTANT_WITH",       NULL},
/* 124 */       {       'U',    "F95_FALSE_CONSTANT_WITH",      NULL},
/* 125 */       {       'T',    "F_TYPE_NODE",  NULL},
/* 126 */       {       'T',    "F_DOUBLE_CONSTANT",    NULL},
/* 127 */       {       'T',    "F_QUAD_CONSTANT",      NULL},
/* 128 */       {       'L',    "F_IMPLIED_DO", NULL},
/* 129 */       {       'L',    "F_INDEX_RANGE",        NULL},
/* 130 */       {       'L',    "F95_ENDPROGRAM_STATEMENT",     NULL},
/* 131 */       {       'L',    "F95_ENDSUBROUTINE_STATEMENT",  NULL},
/* 132 */       {       'L',    "F95_ENDFUNCTION_STATEMENT",    NULL},
/* 133 */       {       'L',    "F95_MODULE_STATEMENT", NULL},
/* 134 */       {       'L',    "F95_ENDMODULE_STATEMENT",      NULL},
/* 135 */       {       'L',    "F95_INTERFACE_STATEMENT",      NULL},
/* 136 */       {       'L',    "F95_ENDINTERFACE_STATEMENT",   NULL},
/* 137 */       {       'L',    "F95_MODULEPROCEDURE_STATEMENT",        NULL},
/* 138 */       {       'L',    "F95_CONTAINS_STATEMENT",       NULL},
/* 139 */       {       'L',    "F95_RECURSIVE_SPEC",   NULL},
/* 140 */       {       'L',    "F95_PURE_SPEC",        NULL},
/* 141 */       {       'L',    "F95_ELEMENTAL_SPEC",   NULL},
/* 142 */       {       'L',    "F95_DIMENSION_DECL",   NULL},
/* 143 */       {       'L',    "F95_TYPEDECL_STATEMENT",       NULL},
/* 144 */       {       'L',    "F95_ENDTYPEDECL_STATEMENT",    NULL},
/* 145 */       {       'L',    "F95_PUBLIC_STATEMENT", NULL},
/* 146 */       {       'L',    "F95_PRIVATE_STATEMENT",        NULL},
/* 147 */       {       'L',    "F95_SEQUENCE_STATEMENT",       NULL},
/* 148 */       {       'L',    "F95_PARAMETER_SPEC",   NULL},
/* 149 */       {       'L',    "F95_ALLOCATABLE_SPEC", NULL},
/* 150 */       {       'L',    "F95_DIMENSION_SPEC",   NULL},
/* 151 */       {       'L',    "F95_EXTERNAL_SPEC",    NULL},
/* 152 */       {       'L',    "F95_INTENT_SPEC",      NULL},
/* 153 */       {       'L',    "F95_INTRINSIC_SPEC",   NULL},
/* 154 */       {       'L',    "F95_OPTIONAL_SPEC",    NULL},
/* 155 */       {       'L',    "F95_POINTER_SPEC",     NULL},
/* 156 */       {       'L',    "F95_SAVE_SPEC",        NULL},
/* 157 */       {       'L',    "F95_TARGET_SPEC",      NULL},
/* 158 */       {       'L',    "F95_PUBLIC_SPEC",      NULL},
/* 159 */       {       'L',    "F95_PRIVATE_SPEC",     NULL},
/* 160 */       {       'L',    "F95_IN_EXTENT",        NULL},
/* 161 */       {       'L',    "F95_OUT_EXTENT",       NULL},
/* 162 */       {       'L',    "F95_INOUT_EXTENT",     NULL},
/* 163 */       {       'L',    "F95_KIND_SELECTOR_SPEC",       NULL},
/* 164 */       {       'L',    "F95_LEN_SELECTOR_SPEC",        NULL},
/* 165 */       {       'L',    "F95_STAT_SPEC",        NULL},
/* 166 */       {       'U',    "F95_ARRAY_CONSTRUCTOR",        NULL},
/* 167 */       {       'U',    "F95_STRUCT_CONSTRUCTOR",       NULL},
/* 168 */       {       'L',    "F95_ASSIGNOP", "="},
/* 169 */       {       'L',    "F95_DOTOP",    NULL},
/* 170 */       {       'L',    "F95_POWEOP",   "**"},
/* 171 */       {       'L',    "F95_MULOP",    "*"},
/* 172 */       {       'L',    "F95_DIVOP",    "/"},
/* 173 */       {       'L',    "F95_PLUSOP",   "+"},
/* 174 */       {       'L',    "F95_MINUSOP",  "-"},
/* 175 */       {       'L',    "F95_EQOP",     ".eq."},
/* 176 */       {       'L',    "F95_NEOP",     ".ne."},
/* 177 */       {       'L',    "F95_LTOP",     ".lt."},
/* 178 */       {       'L',    "F95_LEOP",     ".le."},
/* 179 */       {       'L',    "F95_GEOP",     ".ge."},
/* 180 */       {       'L',    "F95_GTOP",     ".gt."},
/* 181 */       {       'L',    "F95_NOTOP",    ".not."},
/* 182 */       {       'L',    "F95_ANDOP",    ".and."},
/* 183 */       {       'L',    "F95_OROP",     ".or."},
/* 184 */       {       'L',    "F95_EQVOP",    ".eqv."},
/* 185 */       {       'L',    "F95_NEQVOP",   ".neqv."},
/* 186 */       {       'L',    "F95_CONCATOP", "//"},
/* 187 */       {       'L',    "F95_USER_DEFINED",     NULL},
/* 188 */       {       'L',    "F95_ARRAY_ALLOCATION", NULL},
/* 189 */       {       'T',    "F95_GENERIC_SPEC",     NULL},
/* 190 */       {       'L',    "F95_CYCLE_STATEMENT",  NULL},
/* 191 */       {       'L',    "F95_EXIT_STATEMENT",   NULL},
/* 192 */       {       'L',    "F95_USE_STATEMENT",    NULL},
/* 193 */       {       'L',    "F95_USE_ONLY_STATEMENT",       NULL},
/* 194 */       {       'L',    "F95_POINTER_SET_STATEMENT",    NULL},
/* 195 */       {       'L',    "F95_TRIPLET_EXPR",     NULL},
/* 196 */       {       'L',    "F95_ALLOCATE_STATEMENT",       NULL},
/* 197 */       {       'L',    "F95_DEALLOCATE_STATEMENT",     NULL},
/* 198 */       {       'L',    "F95_NULLIFY_STATEMENT",        NULL},
/* 199 */       {       'B',    "F95_MEMBER_REF",       NULL},
/* 200 */       {       'L',    "F95_OPTIONAL_STATEMENT",       NULL},
/* 201 */       {       'L',    "F95_POINTER_STATEMENT",        NULL},
/* 202 */       {       'L',    "F95_INTENT_STATEMENT", NULL},
/* 203 */       {       'L',    "F95_TARGET_STATEMENT", NULL},
/* 204 */       {       'L',    "F95_ALLOCATABLE_STATEMENT",    NULL},
/* 205 */       {       'T',    "F_ASTERISK",   NULL},
/* 206 */       {       'T',    "F_EXTFUNC",    NULL},
/* 207 */       {       'L',    "F_STRING_CONST_SUBSTR",        NULL},
/* 208 */       {       'U',    "XMP_COARRAY_REF",      NULL},
/* 209 */       {       'L',    "XMP_CODIMENSION_SPEC", NULL},
/* 210 */       {       'L',    "OMP_PRAGMA",   NULL},
/* 211 */       {       'L',    "XMP_PRAGMA",   NULL},
/* 212 */       {       'L',    "TEXT", NULL},
};

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

extern char *modincludeDirv;

int flag_module_compile  = FALSE;

static void     outx_expv(int l, expv v);
static void     collect_type_desc(expv v);
char           *xmalloc2( int size) ;

int flag_do_module_cache = TRUE;

int nerrors = 0;
int debug_flag = 0;
int XMP_flag = 0;

int unit_ctl_level = 0;
int unit_ctl_contains_level=0;

TYPE_DESC type_REAL, type_INT, type_SUBR, type_CHAR, type_LOGICAL;
TYPE_DESC type_DREAL, type_COMPLEX, type_DCOMPLEX;
TYPE_DESC type_MODULE;
TYPE_DESC type_GNUMERIC_ALL;
TYPE_DESC type_NAMELIST;
TYPE_DESC basic_type_desc[N_BASIC_TYPES];
expv expv_constant_1,expv_constant_0,expv_constant_m1;
expv expv_float_0;

static int isInFinalizer = FALSE;

generic_procedure *top_gene_pro = NULL;
generic_procedure *cur_gene_pro = NULL;
generic_procedure *end_gene_pro = NULL;


#define MAX_N_FILES  2
int n_files = 0;
char *file_names[MAX_N_FILES];

/**
 * \file module-manager.c
 */

/**
 * collection of fortran modules.
 */
struct module_manager {
    struct module * head;
    struct module * tail;
} MODULE_MANAGER;

static void
add_module(struct module * mod)
{
    if(MODULE_MANAGER.head == NULL) {
        MODULE_MANAGER.head = mod;
    }else {
        MODULE_MANAGER.tail->next = mod;
    }
    MODULE_MANAGER.tail = mod;
    MODULE_MANAGER.tail->next = NULL;
}

static void
add_module_id(struct module * mod, ID id)
{
    // NOTE: 'mid' stand for module id.
    ID mid = XMALLOC(ID, sizeof(*mid));
    *mid = *id;

    // HACK: it seems too dirty
    if(ID_CLASS(mid) == CL_PARAM && ID_STORAGE(mid) == STG_UNKNOWN)
        ID_STORAGE(mid) = STG_SAVE;

    if(mid->use_assoc == NULL) {
        mid->use_assoc = XMALLOC(struct use_assoc_info *, sizeof(*(id->use_assoc)));
        mid->use_assoc->module_name = mod->name;
        mid->use_assoc->original_name = id->name;
    }
    ID_LINK_ADD(mid, mod->head, mod->last);
}


#ifdef _HASH_
/*************
 * hash.c
 *************/
HashEntry *
FirstHashEntry(tablePtr, searchPtr)
    HashTable *tablePtr;                /* Table to search. */
    HashSearch *searchPtr;      /* Place to store information about
                                         * progress through the table. */
{
    searchPtr->tablePtr = tablePtr;
    searchPtr->nextIndex = 0;
    searchPtr->nextEntryPtr = NULL;
    return NextHashEntry(searchPtr);
}

HashEntry *
NextHashEntry(searchPtr)
    register HashSearch *searchPtr;     /* Place to store information about
                                                 * progress through the table.  Must
                                                 * have been initialized by calling
                                                 * FirstHashEntry. */
{
    HashEntry *hPtr;

    while (searchPtr->nextEntryPtr == NULL) {
        if (searchPtr->nextIndex >= searchPtr->tablePtr->numBuckets) {
            return NULL;
        }
        searchPtr->nextEntryPtr =
                searchPtr->tablePtr->buckets[searchPtr->nextIndex];
        searchPtr->nextIndex++;
    }
    hPtr = searchPtr->nextEntryPtr;
    searchPtr->nextEntryPtr = hPtr->nextPtr;
    return hPtr;
}
#endif


#if defined(__STDC__) || defined(HAVE_STDARG_H)
#   include <stdarg.h>
#   define EXC_VARARGS(type, name) (type name, ...)
#   define EXC_VARARGS_DEF(type, name) (type name, ...)
#   define EXC_VARARGS_START(type, name, list) (va_start(list, name))
#else
#   include <varargs.h>
#   ifdef __cplusplus
#       define EXC_VARARGS(type, name) (type name, ...)
#       define EXC_VARARGS_DEF(type, name) (type va_alist, ...)
#   else
#       define EXC_VARARGS(type, name) ()
#       define EXC_VARARGS_DEF(type, name) (va_alist)
#   endif
#   define EXC_VARARGS_START(type, name, list) \
        type name = (va_start(list), va_arg(list, type))
#endif

#   define EXC_VARARGS(type, name) (type name, ...)

#define bzero(p, s)     memset((p), 0, (s))

extern void     fatal EXC_VARARGS(char *, fmt);


/* compiler error: die */
/* VARARGS1 */
void
fatal EXC_VARARGS_DEF(char *, fmt)
{
    va_list args;

  /*where(current_line);*/ /*, "Fatal");*/
    fprintf(stderr, "compiler error: " );
    EXC_VARARGS_START(char *, fmt, args);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
    abort();
}

/*************
 * C-expr-mem.c
 *************/
char *
xmalloc2(size)
     int size;
{
    char *p;
    if((p = (char *)malloc(size)) == NULL)
      fatal("no memory");
    bzero(p,size);
    return(p);
}

struct list_node *cons_list(x,l)
     expr x;
     struct list_node *l;
{
    struct list_node *lp;

    lp = XMALLOC(struct list_node *,sizeof(struct list_node));
    lp->l_next = l;
    lp->l_item = x;
    lp->l_last = NULL;
    lp->l_array = NULL;
    lp->l_nItems = 1;
    return(lp);
}

expr list2(code,x1,x2)
     enum expr_code code;
     expr x1,x2;
{
    return(make_enode(code,(void *)cons_list(x1,cons_list(x2,NULL))));
}

expr list3(code,x1,x2,x3)
     enum expr_code code;
     expr x1,x2,x3;
{
    return(make_enode(code,(void *)cons_list(x1,cons_list(x2,cons_list(x3,NULL)))));
}

expr list_put_last(lx,x)
     expr lx;
     expr x;
{
    struct list_node *lp;

    if (lx == NULL) return(lx); /* error recovery in C-parser.y */
    lp = lx->v.e_lp;
    if (lp == NULL) {
      lx->v.e_lp = cons_list(x,NULL);
    } else {
        if (LIST_LAST(lp) != NULL) {
            lp = LIST_LAST(lp);
        } else {
            for (; lp->l_next != NULL; lp = lp->l_next) /* */;
        }
        lp->l_next = cons_list(x,NULL);
        LIST_LAST(lx->v.e_lp) = lp->l_next;
        LIST_N_ITEMS(lx->v.e_lp) += 1;
    }
    return(lx);
}


/*************
 * F-mem.c
 *************/
expv
expv_int_term(code, tp, i)
     enum expr_code code;
     TYPE_DESC tp;
     omllint_t i;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_INT_VALUE(v) = i;
    return(v);
}

expv
expv_any_term(code,p)
     enum expr_code code;
     void *p;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = NULL;
    EXPV_GEN(v) = p;
    return(v);
}


expv
expv_float_term(code,tp,d,token)
     enum expr_code code;
     TYPE_DESC tp;
     omldouble_t d;
     const char *token;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_FLOAT_VALUE(v) = d;
    EXPV_ORIGINAL_TOKEN(v) = token;
    return(v);
}

expr
expr_list_get_n(x, n)
     expr x;
     int n;
{
    list lp;
    int i;
    for (i = 0, lp = EXPR_LIST(x); (i < n && lp != NULL); i++, lp = LIST_NEXT(lp)) {};
    if (lp == NULL) {
        return NULL;
    }
    return LIST_ITEM(lp);
}

/*************
 * c-option.c
 *************/
#define CEXPR_OPTVAL_CHARLEN 128

/*************
 * nata_macros.h
 *************/
#define isValidString(x)        (((x) != NULL && *(x) != '\0') ? true : false)

/*************
 * F-datatype.c
 *************/
TYPE_DESC
new_type_desc()
{
    TYPE_DESC tp;
    tp = XMALLOC(TYPE_DESC,sizeof(*tp));
    return(tp);
}

TYPE_DESC
type_basic(BASIC_DATA_TYPE t)
{
    TYPE_DESC tp;
    assert(t != TYPE_CHAR);

    tp = new_type_desc();
    TYPE_BASIC_TYPE(tp) = t;
    return tp;
}

TYPE_DESC array_element_type(TYPE_DESC tp)
{
/*
    if(!IS_ARRAY_TYPE(tp)) fatal("array_element_type: not ARRAY_TYPE");
 */
    while(IS_ARRAY_TYPE(tp)) tp = TYPE_REF(tp);
    return tp;
}


/**
 * check type is omissible, such that
 * no attributes, no memebers, no indexRanage and so on.
 *
 * FIXME:
 * shrink_type() and type_is_omissible() are quick-fix.
 * see shrink_type().
 */
int
type_is_omissible(TYPE_DESC tp, uint32_t attr, uint32_t ext)
{
    // NULL or a terminal type is not omissible.
    if (tp == NULL || TYPE_REF(tp) == NULL)
        return FALSE;
    // The struct type is not omissible.
    if (IS_STRUCT_TYPE(tp))
        return FALSE;
    // The array type is not omissible.
    if (IS_ARRAY_TYPE(tp))
        return FALSE;
    // The function type is not omissible.
    if (IS_FUNCTION_TYPE(tp))
        return FALSE;
    // Co-array is not omissible.
    if (tp->codims != NULL)
        return FALSE;
    // The type has kind, leng, or size is not omissible.
    if (TYPE_KIND(tp) != NULL ||
        TYPE_LENG(tp) != NULL ||
        TYPE_CHAR_LEN(tp) != 0) {
        return FALSE;
    }

#if 0
    // The type has attributes is not omissible.
    if (TYPE_ATTR_FLAGS(tp))
        return FALSE;
    if (TYPE_EXTATTR_FLAGS(tp))
        return FALSE;
#else
    /*
     * not omissible if this type has any attributes that is not
     * included in the given attrribute flags.
     */
    if (TYPE_ATTR_FLAGS(tp) != 0) {
        if ((attr != 0 && (attr & TYPE_ATTR_FLAGS(tp)) == 0) ||
            (attr == 0)) {
            return FALSE;
        }
    }
    if (TYPE_EXTATTR_FLAGS(tp) != 0) {
        if ((ext != 0 && (ext & TYPE_EXTATTR_FLAGS(tp)) == 0) ||
            (ext == 0)) {
            return FALSE;
        }
    }
#endif

    return TRUE;
}


static TYPE_DESC
simplify_type_recursively(TYPE_DESC tp, uint32_t attr, uint32_t ext) {
    if (TYPE_REF(tp) != NULL) {
        TYPE_REF(tp) = simplify_type_recursively(TYPE_REF(tp), attr, ext);
    }
    return type_is_omissible(tp, attr, ext) ? TYPE_REF(tp) : tp;
}


/**
 * Reduce redundant type references.
 *
 *      @param  tp      A TYPE_DESC to be reduced.
 *      @return A reduced TYPE_DESC (could be the tp).
 */
TYPE_DESC
reduce_type(TYPE_DESC tp) {
    TYPE_DESC ret = NULL;

    if (tp != NULL) {
#if 0
        uint64_t f = reduce_type_attr(tp);
        uint32_t attr = (uint32_t)(f & 0xffffffffL);
        uint32_t ext = (uint32_t)((f >> 32) & 0xffffffffL);
#else
        uint32_t attr = 0;
        uint32_t ext = 0;
#endif

        ret = simplify_type_recursively(tp, attr, ext);
        if (ret == NULL) {
 /*
            fatal("%s: failure.\n", __func__);
  */
            /* not reached. */
            return NULL;
        }
#if 0
        TYPE_ATTR_FLAGS(ret) = attr;
        TYPE_EXTATTR_FLAGS(ret) = ext;
#endif
    }

    return ret;
}

int
type_is_linked(TYPE_DESC tp, TYPE_DESC tlist)
{
    if(tlist == NULL)
        return FALSE;
    do {
        if (tp == tlist)
            return TRUE;
        tlist = TYPE_LINK(tlist);
    } while (tlist != NULL);
    return FALSE;
}

TYPE_DESC
type_link_add(TYPE_DESC tp, TYPE_DESC tlist, TYPE_DESC ttail)
{
    if(ttail == NULL)
        return tp;
    TYPE_LINK(ttail) = tp;
    ttail = tp;
    while(ttail != TYPE_LINK(ttail) && TYPE_LINK(ttail) != NULL) {
        if(type_is_linked(TYPE_LINK(ttail), tlist))
            break;
        ttail = TYPE_LINK(ttail);
    }
    TYPE_LINK(ttail) = NULL;
    return ttail;
}


/*************
 * F-compile.c
 *************/
/* program unit control stack */
UNIT_CTL unit_ctls[MAX_UNIT_CTL];
int unit_ctl_level;


static UNIT_CTL
new_unit_ctl()
{
    UNIT_CTL uc;

    uc = XMALLOC(UNIT_CTL, sizeof(*uc));
   /*
    if (uc == NULL)
        fatal("memory allocation failed");
    */
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

  /*initialize_intrinsic();*/

  /*initialize_compile_procedure();*/
    initialize_unit_ctl();

    isInFinalizer = FALSE;
}

#ifdef _HASH_
/*************
 * F-module-procedure.c
 *************/
static HashTable genProcTbl;

gen_proc_t
find_generic_procedure(const char *name) {
    gen_proc_t gp = NULL;
    if (isValidString(name)) {
        HashEntry *hPtr = FindHashEntry(&genProcTbl, name);
        if (hPtr != NULL) {
            gp = (gen_proc_t)GetHashValue(hPtr);
        }
    }
    return gp;
}

static void
colloct_module_procedure_types(mod_proc_t mp, expr l) {
    if (mp != NULL) {
        list lp;
        expv v;
        TYPE_DESC tp;

        if ((mp)->retType != NULL) {
            v = list3(LIST,
                      expv_int_term(INT_CONSTANT, type_INT, 1),
                      expv_any_term(IDENT, (void *)(mp)->retType),
                      expv_any_term(IDENT, (void *)MOD_PROC_EXT_ID(mp)));
            list_put_last(l, v);
        }

       /*
        FOR_ITEMS_IN_LIST(lp, MOD_PROC_ARGS(mp)) {
        */
        FOR_ITEMS_IN_LIST(lp, (mp)->args) {
            v = LIST_ITEM(lp);
            if (v != NULL) {
                tp = EXPV_TYPE(v);
                if (tp != NULL) {
                    v = list2(LIST,
                              expv_int_term(INT_CONSTANT, type_INT, 0),
                              expv_any_term(IDENT, (void *)tp));
                    list_put_last(l, v);
                }
            }
        }
    }
}


static void
collect_module_procedures_types(gen_proc_t gp, expr l) {
    if (gp != NULL) {
        HashTable *tPtr = GEN_PROC_MOD_TABLE(gp);
        if (tPtr != NULL) {
            HashEntry *hPtr;
            HashSearch sCtx;

            FOREACH_IN_HASH(hPtr, &sCtx, tPtr) {
                colloct_module_procedure_types((mod_proc_t)GetHashValue(hPtr),
                                               l);
            }
        }
    }
}

expr
collect_all_module_procedures_types(void) {
    expr ret = list0(LIST);
    HashEntry *hPtr;
    HashSearch sCtx;

    FOREACH_IN_HASH(hPtr, &sCtx, &genProcTbl) {
        collect_module_procedures_types((gen_proc_t)GetHashValue(hPtr), ret);
    }

    return ret;
}
#endif /* _HASH_ */

/*************
 * F-output-xcodeml.c
 *************/
#define CHAR_BUF_SIZE 65536

#define ARRAY_LEN(a)    (sizeof(a) / sizeof(a[0]))

static void     outx_functionDefinition(int l, EXT_ID ep);
static void     outx_interfaceDecl(int l, EXT_ID ep);

char s_xmlIndent[CEXPR_OPTVAL_CHARLEN] = "  ";

EXT_ID current_function_stack[MAX_UNIT_CTL_CONTAINS+1];
int current_function_top = 0;
#define CRT_FUNCEP          current_function_stack[current_function_top]
#define CRT_FUNCEP_PUSH(ep) current_function_top++;                    \
                            current_function_stack[current_function_top] = (ep)
#define CRT_FUNCEP_POP      current_function_stack[current_function_top] = (ep); \
                            current_function_top--


typedef struct type_ext_id {
    EXT_ID ep;
    struct type_ext_id *next;
} *TYPE_EXT_ID;

#define FOREACH_TYPE_EXT_ID(te, headte) \
    for ((te) = (headte); (te) != NULL ; (te) = (te)->next)

#define FUNC_EXT_LINK_ADD(te, list, tail) \
    { if((list) == NULL || (tail) == NULL) (list) = (te); \
      else (tail)->next = (te); \
      (tail) = (te); }

static TYPE_DESC    type_list,type_list_tail;
static TYPE_EXT_ID  type_module_proc_list, type_module_proc_last;
static TYPE_EXT_ID  type_ext_id_list, type_ext_id_last;
static FILE         *print_fp;
static char         s_charBuf[CHAR_BUF_SIZE];
static int          is_outputed_module = FALSE;

#define GET_EXT_LINE(ep) \
    (EXT_LINE(ep) ? EXT_LINE(ep) : \
    EXT_PROC_ID_LIST(ep) ? ID_LINE(EXT_PROC_ID_LIST(ep)) : NULL)

static int is_emitting_module = FALSE;


static void
set_module_emission_mode(int mode) {
    is_emitting_module = mode;
}

int
is_emitting_xmod(void) {
    return is_emitting_module;
}


static const char*
xtag(enum expr_code code)
{
    switch(code) {

    /*
     * constants
     */
    case STRING_CONSTANT:           return "FcharacterConstant";
    case INT_CONSTANT:              return "FintConstant";
    case FLOAT_CONSTANT:            return "FrealConstant";
    case COMPLEX_CONSTANT:          return "FcomplexConstant";

    /*
     * declarations
     */
    case F_DATA_DECL:               return "FdataDecl";
    case F_EQUIV_DECL:              return "FequivalenceDecl";
    case F_COMMON_DECL:             return "FcommonDecl";

    /*
     * general statements
     */
    case EXPR_STATEMENT:            return "exprStatement";
    case IF_STATEMENT:              return "FifStatement";
    case F_DO_STATEMENT:            return "FdoStatement";
    case F_DOWHILE_STATEMENT:       return "FdoWhileStatement";
    case F_SELECTCASE_STATEMENT:    return "FselectCaseStatement";
    case F_CASELABEL_STATEMENT:     return "FcaseLabel";
    case F_WHERE_STATEMENT:         return "FwhereStatement";
    case F_RETURN_STATEMENT:        return "FreturnStatement";
    case F_CONTINUE_STATEMENT:      return "continueStatement";
    case GOTO_STATEMENT:
    case F_COMPGOTO_STATEMENT:      return "gotoStatement";
    case STATEMENT_LABEL:           return "statementLabel";
    case F_FORMAT_DECL:             return "FformatDecl";
    case F_STOP_STATEMENT:          return "FstopStatement";
    case F_PAUSE_STATEMENT:         return "FpauseStatement";
    case F_PRAGMA_STATEMENT:        return "FpragmaStatement";
    case F_LET_STATEMENT:           return "FassignStatement";
    case F95_CYCLE_STATEMENT:       return "FcycleStatement";
    case F95_EXIT_STATEMENT:        return "FexitStatement";
    case F_ENTRY_STATEMENT:         return "FentryDecl";

    /*
     * IO statements
     */
    case F_WRITE_STATEMENT:         return "FwriteStatement";
    case F_PRINT_STATEMENT:         return "FprintStatement";
    case F_READ_STATEMENT:          return "FreadStatement";
    case F_READ1_STATEMENT:         return "FreadStatement";
    case F_OPEN_STATEMENT:          return "FopenStatement";
    case F_CLOSE_STATEMENT:         return "FcloseStatement";
    case F_BACKSPACE_STATEMENT:     return "FbackspaceStatement";
    case F_ENDFILE_STATEMENT:       return "FendFileStatement";
    case F_REWIND_STATEMENT:        return "FrewindStatement";
    case F_INQUIRE_STATEMENT:       return "FinquireStatement";

    /*
     * F90/95 Pointer related
     */
    case F95_POINTER_SET_STATEMENT: return "FpointerAssignStatement";
    case F95_ALLOCATE_STATEMENT:    return "FallocateStatement";
    case F95_DEALLOCATE_STATEMENT:  return "FdeallocateStatement";
    case F95_NULLIFY_STATEMENT:     return "FnullifyStatement";


    /*
     * expressions
     */
    case FUNCTION_CALL:             return "functionCall";
    case F_FUNC:                    return "Ffunction";
    case IDENT:
    case F_VAR:
    case F_PARAM:                   return "Var";
    case F95_MEMBER_REF:            return "FmemberRef";
    case ARRAY_REF:                 return "FarrayRef";
    case F_SUBSTR_REF:              return "FcharacterRef";
    case F95_ARRAY_CONSTRUCTOR:     return "FarrayConstructor";
    case F95_STRUCT_CONSTRUCTOR:    return "FstructConstructor";
    case XMP_COARRAY_REF:           return "FcoArrayRef";

    /*
     * operators
     */
    case PLUS_EXPR:                 return "plusExpr";
    case MINUS_EXPR:                return "minusExpr";
    case UNARY_MINUS_EXPR:          return "unaryMinusExpr";
    case MUL_EXPR:                  return "mulExpr";
    case DIV_EXPR:                  return "divExpr";
    case POWER_EXPR:                return "FpowerExpr";
    case F_CONCAT_EXPR:             return "FconcatExpr";
    case LOG_EQ_EXPR:               return "logEQExpr";
    case LOG_NEQ_EXPR:              return "logNEQExpr";
    case LOG_GE_EXPR:               return "logGEExpr";
    case LOG_GT_EXPR:               return "logGTExpr";
    case LOG_LE_EXPR:               return "logLEExpr";
    case LOG_LT_EXPR:               return "logLTExpr";
    case LOG_AND_EXPR:              return "logAndExpr";
    case LOG_OR_EXPR:               return "logOrExpr";
    case LOG_NOT_EXPR:              return "logNotExpr";
    case F_EQV_EXPR:                return "logEQVExpr";
    case F_NEQV_EXPR:               return "logNEQVExpr";

    case F95_USER_DEFINED_BINARY_EXPR:          return "userBinaryExpr";
    case F95_USER_DEFINED_UNARY_EXPR:           return "userUnaryExpr";

    /*
     * misc.
     */
    case F_IMPLIED_DO:              return "FdoLoop";
    case F_INDEX_RANGE:             return "indexRange";

    /*
     * module.
     */
    case F95_USE_STATEMENT:         return "FuseDecl";
    case F95_USE_ONLY_STATEMENT:    return "FuseOnlyDecl";

    /*
     * invalid or no corresponding tag
     */
    case ERROR_NODE:
    case LIST:
    case DEFAULT_LABEL:
    case F_SCENE_RANGE_EXPR:
    case ID_LIST:
    case VAR_DECL:
    case EXT_DECL:
    case FIRST_EXECUTION_POINT:
    case F_PROGRAM_STATEMENT:
    case F_BLOCK_STATEMENT:
    case F_SUBROUTINE_STATEMENT:
    case F_FUNCTION_STATEMENT:
    case F_INCLUDE_STATEMENT:
    case F_END_STATEMENT:
    case F_TYPE_DECL:
    case F_EXTERNAL_DECL:
    case F_INTRINSIC_DECL:
    case F_IMPLICIT_DECL:
    case F_NAMELIST_DECL:
    case F_SAVE_DECL:
    case F_PARAM_DECL:
    case F_DUP_DECL:
    case F_ENDDO_STATEMENT:
    case F_ELSEWHERE_STATEMENT:
    case F_ENDWHERE_STATEMENT:
    case F_ENDSELECT_STATEMENT:
    case F_IF_STATEMENT:
    case F_ELSEIF_STATEMENT:
    case F_ELSE_STATEMENT:
    case F_ENDIF_STATEMENT:
    case F_ASSIGN_LABEL_STATEMENT:
    case F_GOTO_STATEMENT:
    case F_ASGOTO_STATEMENT:
    case F_ARITHIF_STATEMENT:
    case F_CALL_STATEMENT:
    case F_CRAY_POINTER_DECL:
    case F_SET_EXPR:
    case F_LABEL_REF:
    case F_PLUS_EXPR:
    case F_MINUS_EXPR:
    case F_MUL_EXPR:
    case F_DIV_EXPR:
    case F_UNARY_MINUS_EXPR:
    case F_POWER_EXPR:
    case F_EQ_EXPR:
    case F_GT_EXPR:
    case F_GE_EXPR:
    case F_LT_EXPR:
    case F_LE_EXPR:
    case F_NE_EXPR:
    case F_OR_EXPR:
    case F_AND_EXPR:
    case F_NOT_EXPR:
    case F_UNARY_MINUS:
    case BASIC_TYPE_NODE:
    case F_TYPE_NODE:
    case F_TRUE_CONSTANT:
    case F_FALSE_CONSTANT:
    case F_ARRAY_REF:
    case F_STARSTAR:
    case F95_CONSTANT_WITH:
    case F95_TRUE_CONSTANT_WITH:
    case F95_FALSE_CONSTANT_WITH:
    case F95_ENDPROGRAM_STATEMENT:
    case F95_ENDSUBROUTINE_STATEMENT:
    case F95_ENDFUNCTION_STATEMENT:
    case F95_MODULE_STATEMENT:
    case F95_ENDMODULE_STATEMENT:
    case F95_INTERFACE_STATEMENT:
    case F95_ENDINTERFACE_STATEMENT:
    case F95_CONTAINS_STATEMENT:
    case F95_RECURSIVE_SPEC:
    case F95_PURE_SPEC:
    case F95_ELEMENTAL_SPEC:
    case F95_DIMENSION_DECL:
    case F95_TYPEDECL_STATEMENT:
    case F95_ENDTYPEDECL_STATEMENT:
    case F95_PRIVATE_STATEMENT:
    case F95_SEQUENCE_STATEMENT:
    case F95_PARAMETER_SPEC:
    case F95_ALLOCATABLE_SPEC:
    case F95_DIMENSION_SPEC:
    case F95_EXTERNAL_SPEC:
    case F95_INTENT_SPEC:
    case F95_INTRINSIC_SPEC:
    case F95_OPTIONAL_SPEC:
    case F95_POINTER_SPEC:
    case F95_SAVE_SPEC:
    case F95_TARGET_SPEC:
    case F95_PUBLIC_SPEC:
    case F95_PRIVATE_SPEC:
    case F95_IN_EXTENT:
    case F95_OUT_EXTENT:
    case F95_INOUT_EXTENT:
    case F95_KIND_SELECTOR_SPEC:
    case F95_LEN_SELECTOR_SPEC:
    case F95_STAT_SPEC:
    case F95_TRIPLET_EXPR:
    case F95_PUBLIC_STATEMENT:
    case F95_OPTIONAL_STATEMENT:
    case F95_POINTER_STATEMENT:
    case F95_INTENT_STATEMENT:
    case F95_TARGET_STATEMENT:
    case F_ASTERISK:
    case F_EXTFUNC:
    case F_DOUBLE_CONSTANT:
    case F95_ASSIGNOP:
    case F95_DOTOP:
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
    case F95_MODULEPROCEDURE_STATEMENT:
    case F95_ARRAY_ALLOCATION:
    case F95_ALLOCATABLE_STATEMENT:
    case F95_GENERIC_SPEC:
    case F95_USER_DEFINED:
    case XMP_CODIMENSION_SPEC:
    case EXPR_CODE_END:
/*
        fatal("invalid exprcode : %s", EXPR_CODE_NAME(code));
 */
    case OMP_PRAGMA:
      return "OMPPragma";

    case XMP_PRAGMA:
      return "XMPPragma";

    default:
/*
      fatal("unknown exprcode : %d", code);
 */
      ;
    }

    return NULL;
}

#define XTAG(v)  xtag(EXPR_CODE(v))

static void
outx_indent(int l)
{
    int i;
    for(i = 0; i < l; ++i)
        fputs(s_xmlIndent, print_fp);
}


static void
outx_printi(int l, const char *fmt, ...)
{
    outx_indent(l);
    va_list args;
    va_start(args, fmt);
    vfprintf(print_fp, fmt, args);
    va_end(args);
}

#define         outx_print(...)    outx_printi(0, __VA_ARGS__)

static void
outx_puts(const char *s)
{
    fputs(s, print_fp);
}


/**
 * output any tag with format
 */
static void
outx_tag(int l, const char *tagAndFmt, ...)
{
    sprintf(s_charBuf, "<%s>\n", tagAndFmt);
    outx_indent(l);
    va_list args;
    va_start(args, tagAndFmt);
    vfprintf(print_fp, s_charBuf, args);
    va_end(args);
}


/**
 * output any closed tag
 */
static void
outx_close(int l, const char *tag)
{
    outx_printi(l, "</%s>\n", tag);
}

#define outx_expvClose(l, v)    outx_close(l, XTAG(v))


static void
xstrcat(char **p, const char *s)
{
    int len = strlen(s);
    strcpy(*p, s);
    *p += len;
}



static const char*
getXmlEscapedStr(const char *s)
{
    static char x[CHAR_BUF_SIZE];
    const char *p = s;
    char *px = x;
    char c;
    x[0] = '\0';

    while((c = *p++)) {
        switch(c) {
        case '<':  xstrcat(&px, "&lt;"); break;
        case '>':  xstrcat(&px, "&gt;"); break;
        case '&':  xstrcat(&px, "&amp;"); break;
        case '"':  xstrcat(&px, "&quot;"); break;
        case '\'': xstrcat(&px, "&apos;"); break;
        /* \002 used as quote in F95-lexer.c */
        case '\002':  xstrcat(&px, "&quot;"); break;
        case '\r': xstrcat(&px, "\\r"); break;
        case '\n': xstrcat(&px, "\\n"); break;
        case '\t': xstrcat(&px, "\\t"); break;
        default:
            if(c <= 31 || c >= 127) {
                char buf[16];
                sprintf(buf, "&#x%x;", (unsigned int)(c & 0xFF));
                xstrcat(&px, buf);
            } else {
                *px++ = c;
            }
            break;
        }
    }

    *px = 0;

    return x;
}


/**
 * output expv enclosed with tag
 */
static void
outx_childrenWithTag(int l, const char *tag, expv v)
{
    outx_printi(l, "<%s>\n", tag);
    outx_expv(l + 1, v);
    outx_printi(l, "</%s>\n", tag);
}


/**
 * output number as intConstatnt
 */
static void
outx_intAsConst(int l, omllint_t n)
{
    outx_printi(l, "<FintConstant type=\"Fint\">%lld</FintConstant>\n", n);
}


#define STR_HAS_NO_QUOTE        0
#define STR_HAS_DBL_QUOTE       1
#define STR_HAS_SGL_QUOTE       2
/**
 * Check the string has quote character(s) or not.
 * @param str a string.
 * @return STR_HAS_NO_QUOTE: no quote character contained.
 *      <br/> STR_HAS_DBL_QUOTE: contains at least a double quote.
 *      <br/> STR_HAS_SGL_QUOTE: contains at least a single quote.
 */
static int
hasStringQuote(const char *str)
{
    char *s = strchr(str, '"');
    if (s != NULL) {
        return STR_HAS_DBL_QUOTE;
    } else if ((s = strchr(str, '\'')) != NULL) {
        return STR_HAS_SGL_QUOTE;
    } else {
        return STR_HAS_NO_QUOTE;
    }
}


static const char*
getRawString(expv v)
{
    static char buf[CHAR_BUF_SIZE];

    switch (EXPV_CODE(v)) {
    case INT_CONSTANT:
        snprintf(buf, CHAR_BUF_SIZE, OMLL_DFMT, EXPV_INT_VALUE(v));
        break;
    case STRING_CONSTANT: {
        if (EXPV_STR(v) != NULL) {
            int quote = hasStringQuote(EXPV_STR(v));
            switch (quote) {
                case STR_HAS_SGL_QUOTE:
                case STR_HAS_NO_QUOTE: {
                    snprintf(buf, CHAR_BUF_SIZE,
                             "&quot;%s&quot;", getXmlEscapedStr(EXPV_STR(v)));
                    break;
                }
                case STR_HAS_DBL_QUOTE: {
                    snprintf(buf, CHAR_BUF_SIZE,
                             "&#39;%s&#39;", getXmlEscapedStr(EXPV_STR(v)));
                    break;
                }
                default: {
                   /*
                    fatal("%s: Unknown quote status??", __func__);
                    */
                    break;
                }
            }
        } else {
            snprintf(buf, CHAR_BUF_SIZE, "\"\"");
        }
        break;
    }
    case IDENT:
    case F_VAR:
    case F_PARAM:
    case F_FUNC:
        snprintf(buf, CHAR_BUF_SIZE,
                 "%s", getXmlEscapedStr(SYM_NAME(EXPV_NAME(v))));
        break;
    case ARRAY_REF:
        snprintf(buf, CHAR_BUF_SIZE,
                 "%s", getXmlEscapedStr(SYM_NAME(EXPV_NAME(EXPR_ARG1(v)))));
        break;
    default:
        abort();
    }

    return buf;
}


static void
outx_true(int cond, const char *flagname)
{
    if(cond)
        outx_print(" %s=\"true\"", flagname);
}


#define TOPT_TYPEONLY   (1 << 0)
#define TOPT_NEXTLINE   (1 << 1)
#define TOPT_CLOSE      (1 << 2)
#define TOPT_INTRINSIC  (1 << 3)

static int
has_attribute_except_func_attrs(TYPE_DESC tp)
{
    return
        /* TYPE_IS_EXTERNAL(tp) || */
        /* TYPE_IS_INTRINSIC(tp) || */
        /* TYPE_IS_RECURSIVE(tp) || */
        /* TYPE_IS_PURE(tp) || */
        /* TYPE_IS_ELEMENTAL(tp) || */
        TYPE_IS_PARAMETER(tp) ||
        TYPE_IS_ALLOCATABLE(tp) ||
        TYPE_IS_OPTIONAL(tp) ||
        TYPE_IS_POINTER(tp) ||
        TYPE_IS_SAVE(tp) ||
        TYPE_IS_TARGET(tp) ||
        TYPE_IS_PUBLIC(tp) ||
        TYPE_IS_PRIVATE(tp) ||
        TYPE_IS_SEQUENCE(tp) ||
        TYPE_IS_INTERNAL_PRIVATE(tp) ||
        TYPE_IS_INTENT_IN(tp) ||
        TYPE_IS_INTENT_OUT(tp) ||
        TYPE_IS_INTENT_INOUT(tp) ||
        tp->codims;
}

static int
has_attribute(TYPE_DESC tp)
{
    return
        has_attribute_except_func_attrs(tp) ||
        TYPE_IS_EXTERNAL(tp) ||
        TYPE_IS_INTRINSIC(tp) ||
        TYPE_IS_RECURSIVE(tp) ||
        TYPE_IS_PURE(tp) ||
        TYPE_IS_ELEMENTAL(tp);
}

static int
has_attribute_except_private_public(TYPE_DESC tp)
{
    int ret;
    int is_public = TYPE_IS_PUBLIC(tp);
    int is_private = TYPE_IS_PRIVATE(tp);
    TYPE_UNSET_PUBLIC(tp);
    TYPE_UNSET_PRIVATE(tp);
    ret = has_attribute(tp);
    if(is_private)
        TYPE_SET_PRIVATE(tp);
    if(is_public)
        TYPE_SET_PUBLIC(tp);
    return ret;
}

static const char *
getBasicTypeID(BASIC_DATA_TYPE t)
{
    const char *tid = NULL;
    switch(t) {
    case TYPE_INT:          tid = "Fint"; break;
    case TYPE_CHAR:         tid = "Fcharacter"; break;
    case TYPE_LOGICAL:      tid = "Flogical"; break;
    case TYPE_REAL:         /* fall through */
    case TYPE_DREAL:        tid = "Freal"; break;
    case TYPE_COMPLEX:      /* fall through */
    case TYPE_DCOMPLEX:     tid = "Fcomplex"; break;
    case TYPE_GNUMERIC:     tid = "Fnumeric"; break;
    case TYPE_GENERIC:      /* fall through */
    case TYPE_LHS:          /* fall through too */
    case TYPE_GNUMERIC_ALL: tid = "FnumericAll"; break;
    case TYPE_SUBR:         /* fall through */
    case TYPE_MODULE:       tid = "Fvoid"; break;
    case TYPE_NAMELIST:     tid = "Fnamelist"; break;
    default: abort();
    }
    return tid;
}


#define checkBasic(tp)                          \
    ((tp == NULL) || (                          \
    (has_attribute_except_func_attrs(tp) == FALSE) && \
    (TYPE_KIND(tp) == NULL) &&                  \
    (IS_DOUBLED_TYPE(tp) == FALSE) &&           \
    (!tp->codims) &&                            \
    ((IS_NUMERIC(tp) ||                         \
      IS_LOGICAL(tp) ||                         \
      IS_SUBR(tp) ||                            \
      IS_MODULE(tp) ||                          \
      (IS_CHAR(tp) && TYPE_CHAR_LEN(tp) == 1)))))
/**
 * get typeID
 */
static const char*
getTypeID(TYPE_DESC tp)
{
    static char buf[256];

    if (checkBasic(tp) && checkBasic(TYPE_REF(tp))) {
        strcpy(buf, getBasicTypeID(TYPE_BASIC_TYPE(tp)));
    } else {
        char pfx;

        switch(TYPE_BASIC_TYPE(tp)) {
        case TYPE_INT:          pfx = 'I'; break;
        case TYPE_CHAR:         pfx = 'C'; break;
        case TYPE_LOGICAL:      pfx = 'L'; break;
        case TYPE_REAL:         /* fall through */
        case TYPE_DREAL:        pfx = 'R'; break;
        case TYPE_COMPLEX:      /* fall through */
        case TYPE_DCOMPLEX:     pfx = 'P'; break;
        case TYPE_FUNCTION:     /* fall through */
        case TYPE_SUBR:         pfx = 'F'; break;
        case TYPE_ARRAY:        pfx = 'A'; break;
        case TYPE_STRUCT:       pfx = 'S'; break;
        case TYPE_GNUMERIC:     pfx = 'U'; break;
        case TYPE_GENERIC:      /* fall through */
        case TYPE_LHS:          /* fall through too */
        case TYPE_GNUMERIC_ALL: pfx = 'V'; break;
        case TYPE_NAMELIST:     pfx = 'N'; break;
        default: abort();
        }

        sprintf(buf, "%c" ADDRX_PRINT_FMT, pfx, Addr2Uint(tp));
    }

    return buf;
}


static char*
genFunctionTypeID(EXT_ID ep)
{
    char buf[128];
    sprintf(buf, "F" ADDRX_PRINT_FMT, Addr2Uint(ep));
    return strdup(buf);
}


/**
 * output tag and type attribute
 */
static void
outx_typeAttrs(int l, TYPE_DESC tp, const char *tag, int options)
{
    if(tp == NULL) {
        outx_printi(l,"<%s", tag);
        return;
    }

    /* Output type as generic numeric type when type is not fixed */
    if(TYPE_IS_NOT_FIXED(tp) && TYPE_BASIC_TYPE(tp) == TYPE_UNKNOWN) {
        outx_printi(l,"<%s type=\"%s\"", tag, "FnumericAll");
    } else {
        outx_printi(l,"<%s type=\"%s\"", tag, getTypeID(tp));
    }

    if((options & TOPT_TYPEONLY) == 0) {

        if(TYPE_HAS_INTENT(tp)) {
            const char *intent;
            if(TYPE_IS_INTENT_IN(tp))
                intent = "in";
            else if(TYPE_IS_INTENT_OUT(tp))
                intent = "out";
            else
                intent = "inout";
            outx_print(" intent=\"%s\"", intent);
        }

        /*
         * FIXME:
         *      Actually we want this assertions.
         */
        assert(TYPE_IS_RECURSIVE(tp) == FALSE);
        assert(TYPE_IS_EXTERNAL(tp) == FALSE);
        assert(TYPE_IS_INTRINSIC(tp) == FALSE);

        outx_true(TYPE_IS_PUBLIC(tp),           "is_public");
        outx_true(TYPE_IS_PRIVATE(tp),          "is_private");
        outx_true(TYPE_IS_POINTER(tp),          "is_pointer");
        outx_true(TYPE_IS_TARGET(tp),           "is_target");
        outx_true(TYPE_IS_OPTIONAL(tp),         "is_optional");
        outx_true(TYPE_IS_SAVE(tp),             "is_save");
        outx_true(TYPE_IS_PARAMETER(tp),        "is_parameter");
        outx_true(TYPE_IS_ALLOCATABLE(tp),      "is_allocatable");
        outx_true(TYPE_IS_SEQUENCE(tp),         "is_sequence");
        outx_true(TYPE_IS_INTERNAL_PRIVATE(tp), "is_internal_private");
    }

    if((options & TOPT_INTRINSIC) > 0)
        outx_true(1, "is_intrinsic");

    if((options & TOPT_NEXTLINE) > 0)
        outx_print(">\n");
    if((options & TOPT_CLOSE) > 0)
        outx_print("/>\n");
}


#define outx_typeAttrOnly_EXPR(l, v, tag) \
    outx_typeAttrs((l), ((v) ? EXPV_TYPE(v) : NULL), tag, TOPT_TYPEONLY)

#define outx_typeAttrOnly_ID(l, id, tag) \
    outx_typeAttrs((l), ((id) ? ID_TYPE(id) : NULL), tag, TOPT_TYPEONLY)

static void
outx_typeAttrOnly_functionType(int l, EXT_ID ep, const char *tag)
{
    char *tid = genFunctionTypeID(ep);
    outx_printi(l,"<%s type=\"%s\"", tag, tid);
}


static void
outx_typeAttrOnly_functionTypeWithResultVar(
    int l, EXT_ID ep, const char *tag)
{
    outx_typeAttrOnly_functionType(l, ep, tag);
    if (EXT_PROC_RESULTVAR(ep) != NULL) {
        expv res = EXT_PROC_RESULTVAR(ep);
        outx_print(" result_name=\"%s\"",
                   SYM_NAME(EXPV_NAME(res)));
    }
}


static void
outx_lineno(lineno_info *li)
{
    if(li) {
        outx_print(" lineno=\"%d\"", li->ln_no);
        if (li->end_ln_no) outx_print(" endlineno=\"%d\"", li->end_ln_no);
        outx_print(" file=\"%s\"", getXmlEscapedStr(FILE_NAME(li->file_id)));
    }
}


/**
 * output tag with format and lineno attribute
 */
static void
outx_vtagLineno(int l, const char *tag, lineno_info *li, va_list args)
{
    static char buf1[CHAR_BUF_SIZE], buf2[CHAR_BUF_SIZE];
    snprintf(buf1, sizeof(buf1), "<%s", tag);
    if (args != NULL) {
        vsnprintf(buf2, sizeof(buf2), buf1, args);
    } else {
        strncpy(buf2, buf1, sizeof(buf2));
    }

    outx_indent(l);
    outx_puts(buf2);
    outx_lineno(li);
}


static void
outx_vtagOfDecl(int l, const char *tag, lineno_info *li, va_list args)
{
    outx_vtagLineno(l, tag, li, args);
}


static void
outx_tagOfDecl1(int l, const char *tag, lineno_info *li, ...)
{
    va_list args;
    va_start(args, li);
    outx_vtagOfDecl(l, tag, li, args);
    va_end(args);
    outx_puts(">\n");
}


static void
outx_tagOfDecl(int l, const char *tag, ID id)
{
    outx_tagOfDecl1(l, tag, ID_LINE(id));
}


/**
 * output tag which has only text symbol value
 */
static void
outx_tagText(int l, const char *tag, const char *s)
{
    outx_printi(l, "<%s>%s</%s>\n", tag, s, tag);
}


#define outx_symbolName(l, s)   outx_tagText(l, "name", SYM_NAME(s))
#define outx_expvName(l, v)     outx_symbolName(l, EXPV_NAME(v))


/**
 * output a symbol with function type (not a return type)
 */
static void
outx_symbolNameWithFunctionType(int l, EXT_ID ep)
{
    outx_typeAttrOnly_functionType(l, ep, "name");
    outx_print(">%s</name>\n", SYM_NAME(EXT_SYM(ep)));
}


/**
 * output an identifier symbol with type
 */
static void
outx_symbolNameWithType_ID(int l, ID id)
{
    outx_typeAttrs(l, ID_TYPE(id), "name", TOPT_TYPEONLY);
    outx_print(">%s</name>\n", SYM_NAME(ID_SYM(id)));
}


/**
 * output an expr as name with type
 */
static void
outx_expvNameWithType(int l, expv v)
{
    if(EXPV_PROC_EXT_ID(v)) {
        // for high order function
        outx_typeAttrOnly_functionType(l, EXPV_PROC_EXT_ID(v), "name");
    } else {
        outx_typeAttrs(l, EXPV_TYPE(v), "name", TOPT_TYPEONLY);
    }
    outx_print(">%s</name>\n", SYM_NAME(EXPV_NAME(v)));
}




static const char*
get_sclass(ID id)
{
    switch(ID_CLASS(id)) {
    case CL_LABEL:
        return "flabel"; /* unused */
    case CL_PARAM:
        return "flocal";
    case CL_COMMON:
        return "fcommon_name";
        break;
    case CL_NAMELIST:
        return "fnamelist_name";
    default:
        if(ID_EQUIV_ID(id)) {
            return get_sclass(ID_EQUIV_ID(id));
        }

        switch(ID_STORAGE(id)) {
        case STG_ARG:
            return "fparam";
        case STG_EXT:
            /*
            if(PROC_CLASS(id) == P_INTRINSIC || PROC_CLASS(id) == P_EXTERNAL)
                return "extern";
            else
                return "extern_def";
            */
            return "ffunc";
        case STG_COMEQ:
        case STG_COMMON:
            return "fcommon";
        case STG_SAVE:
            return "fsave";
        case STG_AUTO:
        case STG_EQUIV:
            return "flocal";
        case STG_TAGNAME:
            return "ftype_name";
        case STG_UNKNOWN:
        case STG_NONE:
/*
            fatal("%s: illegal storage class: symbol=%s", __func__, ID_NAME(id));
 */
            abort();
        }
        break;
    }

    return NULL;
}


/**
 * get scope of id
 */
static const char*
getScope(expv v)
{
    const char *scope;
    switch(EXPV_CODE(v)) {
    case F_FUNC:
    case IDENT:
    case F_VAR:     scope = "local"; break;
    case F_PARAM:   scope = "param"; break;
    default: abort();
    }

    return scope;
}


/**
 * output Var/Ffunction
 */
static void
outx_varOrFunc(int l, expv v)
{
    const char *scope = getScope(v);
    outx_typeAttrOnly_EXPR(l, v, XTAG(v));
    if(scope)
        outx_print(" scope=\"%s\"", scope);
    outx_print(">%s</%s>\n", SYM_NAME(EXPV_NAME(v)), XTAG(v));
}


/**
 * output Var of ident
 */
static void
outx_var_ID(int l, ID id)
{
    const char *scope = "local";
    const char *tag = xtag(F_VAR);
    outx_typeAttrOnly_ID(l, id, tag);
    if(scope)
        outx_print(" scope=\"%s\"", scope);
    outx_print(">%s</%s>\n", ID_NAME(id), tag);
}


/**
 * output expression tag which has type attribute
 */
static void
outx_tagOfExpression(int l, expv v)
{
    outx_typeAttrs(l, EXPV_TYPE(v), XTAG(v),
        TOPT_TYPEONLY|TOPT_NEXTLINE);
}

/**
 * output id for ID
 */
static void
outx_id(int l, ID id)
{
    if(ID_STORAGE(id) == STG_EXT && PROC_EXT_ID(id) == NULL) {
/*
        fatal("outx_id: PROC_EXT_ID is NULL: symbol=%s", ID_NAME(id));
 */
    }

    if(ID_CLASS(id) == CL_PROC && PROC_EXT_ID(id)) {
        outx_typeAttrOnly_functionType(l, PROC_EXT_ID(id), "id");
    } else {
        outx_typeAttrOnly_ID(l, id, "id");
    }

    const char *sclass = get_sclass(id);
    outx_print(" sclass=\"%s\"", sclass);
    if(ID_IS_OFMODULE(id))
        outx_print(" declared_in=\"%s\"",
                   ID_USEASSOC_INFO(id)->module_name->s_name);
    outx_print(">\n");
    outx_symbolName(l + 1, ID_SYM(id));
    outx_close(l, "id");
}


static void
outx_valueList(int l, expv v)
{
    struct list_node *lp;
    int l1 = l + 1, l2 = l1 + 1;
    expv val0, val;

    outx_tag(l, "valueList");

    FOR_ITEMS_IN_LIST(lp, v) {
        val0 = LIST_ITEM(lp);
        outx_tag(l1, "value");
        if(EXPV_CODE(val0) == F_DUP_DECL)
            val = EXPR_ARG2(val0);
        else
            val = val0;
        outx_expv(l2, val);

        if(EXPV_CODE(val0) == F_DUP_DECL) {
            outx_tag(l2, "repeat_count");
            outx_expv(l2 + 1, EXPR_ARG1(val0));
            outx_close(l2, "repeat_count");
        }

        outx_close(l1, "value");
    }

    outx_close(l, "valueList");
}


/**
 * output indexRange
 */
static void
outx_indexRange0(int l,
    ARRAY_ASSUME_KIND assumeKind,
    ARRAY_ASSUME_KIND defaultAssumeKind,
    expv lower, expv upper, expv step)
{
    const int l1 = l + 1;

    outx_printi(l, "<%s", xtag(F_INDEX_RANGE));

    if(assumeKind == ASSUMED_NONE && upper == NULL) {
        assumeKind = defaultAssumeKind;
    }

    switch(assumeKind) {
    case ASSUMED_NONE:
        break;
    case ASSUMED_SIZE:
        outx_true(TRUE, "is_assumed_size");
        break;
    case ASSUMED_SHAPE:
        outx_true(TRUE, "is_assumed_shape");
        break;
    }

    outx_print(">\n");
    if(lower)
        outx_childrenWithTag(l1, "lowerBound", lower);
    if(upper)
        outx_childrenWithTag(l1, "upperBound", upper);
    if(step)
        outx_childrenWithTag(l1, "step", step);


      outx_close(l, xtag(F_INDEX_RANGE));
}

#define outx_indexRange(l, lower, upper, step) \
    outx_indexRange0((l), \
    ASSUMED_NONE, ASSUMED_SHAPE, (lower), (upper), (step))


/**
 * output indexRange for list element
 */
static void
outx_indexRangeInList(int l, expr v)
{
    outx_indexRange(l, EXPR_ARG1(v),
        expr_list_get_n(v, 1), expr_list_get_n(v, 2));
}


/**
 * output indexRange for TYPE_DESC which represents dimension
 */
static void
outx_indexRangeOfType(int l, TYPE_DESC tp)
{
    TYPE_DESC rtp = TYPE_REF(tp);
    if(IS_ARRAY_TYPE(rtp)) outx_indexRangeOfType(l, rtp);
    outx_indexRange0(l, TYPE_ARRAY_ASSUME_KIND(tp), ASSUMED_SIZE,
    TYPE_DIM_LOWER(tp), TYPE_DIM_UPPER(tp), TYPE_DIM_STEP(tp));
}


/**
 * output FstructDecl
 */
static void
outx_structDecl(int l, ID id)
{
    outx_tagOfDecl(l, "FstructDecl", id);
    outx_symbolNameWithType_ID(l + 1, id);
    outx_close(l, "FstructDecl");
}


/**
 * output varRef of ident_descriptor
 */
static void
outx_varRef(int l, expv v, ID id)
{
    const int l1 = l + 1;
    if(v) {
        outx_typeAttrOnly_EXPR(l, v, "varRef");
        outx_print(">\n");
        outx_expv(l1, v);
    } else if(id) {
        outx_typeAttrOnly_ID(l, id, "varRef");
        outx_print(">\n");
        outx_var_ID(l1, id);
    } else {
        abort();
    }
    outx_printi(l, "</varRef>\n");
}


#define outx_varRef_EXPR(l, v)    outx_varRef(l, v, 0)
#define outx_varRef_ID(l, id)     outx_varRef(l, 0, id)


/**
 * output value
 */
static void
outx_value(int l, expv v)
{
    if(v == NULL) return;
    outx_childrenWithTag(l, "value", v);
}



/**
 * output values in list element
 */
static void
outx_valueInList(int l, expv v)
{
    list lp;

    if(v == NULL) return;
    FOR_ITEMS_IN_LIST(lp, v)
        outx_value(l, LIST_ITEM(lp));
}


/**
 * output FdoLoop
 */
static void
outx_doLoop(int l, expv v)
{
    const int l1 = l + 1;

    expv vl = EXPR_ARG1(v);
    expv vr = EXPR_ARG2(v);
    expv var   = EXPR_ARG1(vl);
    expv lower = EXPR_ARG2(vl);
    expv upper = EXPR_ARG3(vl);
    expv step  = EXPR_ARG4(vl);

    outx_tag(l, XTAG(v));
    outx_expv(l1, var);
    outx_indexRange(l1, lower, upper, step);
    outx_valueInList(l1, vr);

    outx_expvClose(l, v);
}



/**
 * output varList
 */
static void
outx_varList(int l, expv v, const char *name)
{
    const int l1 = l + 1;
    char buf[256];
    list lp;

    if(name)
        sprintf(buf, " name=\"%s\"", name);
    else
        buf[0] = '\0';

    outx_tag(l, "varList%s", buf);
    FOR_ITEMS_IN_LIST(lp, v) {
        expv x = LIST_ITEM(lp);
        if(EXPR_CODE(x) == F_IMPLIED_DO)
            outx_doLoop(l1, x);
        else
            outx_varRef_EXPR(l1, x);
    }
    outx_close(l, "varList");
}


/**
 * output namelistDecl
 */
static void
outx_namelistDecl(int l, ID nlId)
{
    const int l1 = l + 1;

    if(nlId == NULL)
        return;

    outx_tagOfDecl(l, "FnamelistDecl", nlId);
    outx_varList(l1, NL_LIST(nlId), SYM_NAME(ID_SYM(nlId)));
    outx_close(l, "FnamelistDecl");
}


/**
 * output FdataDecl
 */
static void
outx_dataDecl(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfDecl1(l, XTAG(v), EXPR_LINE(v));
    outx_varList(l1, EXPR_ARG1(v), NULL);
    outx_valueList(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output varDecl
 */
static void
outx_varDecl(int l, ID id)
{
    assert(id);
    const int l1 = l + 1;

    outx_tagOfDecl(l, "varDecl", id);

    if(PROC_EXT_ID(id)) {
        /* high order func */
        outx_symbolNameWithFunctionType(l1, PROC_EXT_ID(id));
    } else {
        outx_symbolNameWithType_ID(l1, id);
        outx_value(l1, VAR_INIT_VALUE(id));
    }

    outx_close(l, "varDecl");
}


static const char*
getKindParameter(TYPE_DESC tp)
{
    static char buf[256];
    expv v = TYPE_KIND(tp);

    if(IS_DOUBLED_TYPE(tp)) {
        sprintf(buf, "%d", KIND_PARAM_DOUBLE);
    } else if(v) {
        strcpy(buf, getRawString(v));
    } else {
        return NULL;
    }

    return buf;
}


static TYPE_DESC
getLargeIntType()
{
    static TYPE_DESC tp = NULL;
    if(tp) return tp;

    tp = type_basic(TYPE_INT);
    TYPE_KIND(tp) = expv_int_term(INT_CONSTANT, type_INT, 8);

    return tp;
}

/**
 * output FintConstant/FcharacterConstant/FlogicalConstant/FcomplexConstant
 */
static void
outx_constants(int l, expv v)
{
    static char buf[CHAR_BUF_SIZE];
    const int l1 = l + 1;
    const char *tid;
    const char *tag = XTAG(v);
    TYPE_DESC tp = EXPV_TYPE(v);
    const char *kind;

    switch(EXPV_CODE(v)) {
    case INT_CONSTANT:
        if(IS_LOGICAL(EXPV_TYPE(v))) {
            sprintf(buf, ".%s.", EXPV_INT_VALUE(v) ? "TRUE" : "FALSE");
            tag = "FlogicalConstant";
        } else {
            omllint_t n = EXPV_INT_VALUE(v);
            if(n > INT_MAX)
                tp = getLargeIntType();
            sprintf(buf, OMLL_DFMT, n);
        }
        if(tp == NULL)
            tp = type_INT;
        tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
        goto print_constant;

    case FLOAT_CONSTANT:
        if(EXPV_ORIGINAL_TOKEN(v))
            strcpy(buf, EXPV_ORIGINAL_TOKEN(v));
        else
            sprintf(buf, "%Lf", EXPV_FLOAT_VALUE(v));
        if(tp == NULL)
            tp = type_REAL;
        tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
        goto print_constant;

    case STRING_CONSTANT:
        strcpy(buf, getXmlEscapedStr(EXPV_STR(v)));
        assert(tp);
        tid = getTypeID(tp);
        goto print_constant;

    print_constant:
        outx_printi(l, "<%s type=\"%s\"", tag, tid);
        if(EXPV_CODE(v) != STRING_CONSTANT &&
            (kind = getKindParameter(tp)) != NULL)
            outx_print(" kind=\"%s\"", kind);
        outx_print(">%s</%s>\n", buf, tag);
        break;

    case COMPLEX_CONSTANT:
        assert(tp);
        outx_tagOfExpression(l, v);
        outx_expv(l1, EXPR_ARG1(v));
        outx_expv(l1, EXPR_ARG2(v));
        outx_expvClose(l, v);
        break;

    default:
        abort();
    }
}

/**
 * output rename
 */
static void
outx_useRenamable(int l, expv local, expv use)
{
    assert(use != NULL);
    outx_printi(l, "<renamable");

    if (local != NULL)
        outx_printi(0," local_name=\"%s\"", getRawString(local));

    outx_printi(0, " use_name=\"%s\"/>\n", getRawString(use));
}

/**
 * output FuseOnlyDecl
 */
static void
outx_useOnlyDecl(int l, expv v)
{
    list lp;
    const char *mod_name = SYM_NAME(EXPV_NAME(EXPR_ARG1(v)));

    outx_tagOfDecl1(l, "%s name=\"%s\"", EXPR_LINE(v), XTAG(v), mod_name);

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)) {
        expv x = LIST_ITEM(lp);
        outx_useRenamable(l+1, EXPR_ARG1(x), EXPR_ARG2(x));
    }

    outx_expvClose(l, v);
}

static void
outx_expv(int l, expv v)
{
    enum expr_code code;
    if(v == NULL)
        return;
    code = EXPV_CODE(v);

    switch(code) {
    /*
     * identifiers
     */
    case F_FUNC:            outx_varOrFunc(l, v); break;
    case IDENT:
    case F_VAR:
    case F_PARAM:           outx_varOrFunc(l, v); break;

    /*
     * constants
     */
    case INT_CONSTANT:
    case STRING_CONSTANT:
    case FLOAT_CONSTANT:
    case COMPLEX_CONSTANT:  outx_constants(l, v); break;

#if 0
    /*
     * declarations
     */
    case F_FORMAT_DECL:     outx_formatDecl(l, v); break;

    /*
     * general statements
     */
    case EXPR_STATEMENT:            outx_exprStatement(l, v); break;
    case F_DO_STATEMENT:            outx_doStatement(l, v); break;
    case F_DOWHILE_STATEMENT:       outx_doWhileStatement(l, v); break;
    case F_SELECTCASE_STATEMENT:    outx_selectStatement(l, v); break;
    case IF_STATEMENT:
    case F_WHERE_STATEMENT:         outx_IFWHERE_Statement(l, v); break;
    case F_RETURN_STATEMENT:        outx_returnStatement(l, v); break;
    case F_CONTINUE_STATEMENT:      outx_continueStatement(l, v); break;
    case GOTO_STATEMENT:            outx_gotoStatement(l, v); break;
    case F_COMPGOTO_STATEMENT:      outx_compgotoStatement(l, v); break;
    case STATEMENT_LABEL:           outx_labeledStatement(l, v); break;
    case F_CASELABEL_STATEMENT:     outx_caseLabel(l, v); break;
    case F_STOP_STATEMENT:
    case F_PAUSE_STATEMENT:         outx_STOPPAUSE_statement(l, v); break;
    case F_LET_STATEMENT:           outx_assignStatement(l, v); break;
    case F_PRAGMA_STATEMENT:        outx_pragmaStatement(l, v); break;
    case F95_CYCLE_STATEMENT:
    case F95_EXIT_STATEMENT:        outx_EXITCYCLE_statement(l, v); break;
    case F_ENTRY_STATEMENT:         outx_entryDecl(l, v); break;

    /*
     * IO statements
     */
    case F_PRINT_STATEMENT:         outx_printStatement(l, v); break;
    case F_READ_STATEMENT:
    case F_WRITE_STATEMENT:
    case F_INQUIRE_STATEMENT:       outx_RW_statement(l, v); break;
    case F_READ1_STATEMENT:
    case F_OPEN_STATEMENT:
    case F_CLOSE_STATEMENT:
    case F_BACKSPACE_STATEMENT:
    case F_ENDFILE_STATEMENT:
    case F_REWIND_STATEMENT:        outx_IO_statement(l, v); break;

    /*
     * F90/95 Pointer related.
     */
    case F95_POINTER_SET_STATEMENT: outx_pointerAssignStatement(l, v); break;
    case F95_ALLOCATE_STATEMENT:
    case F95_DEALLOCATE_STATEMENT:  outx_ALLOCDEALLOC_statement(l, v); break;
    case F95_NULLIFY_STATEMENT:     outx_nullifyStatement(l, v); break;

    /*
     * expressions
     */
    case FUNCTION_CALL:     outx_functionCall(l, v); break;
    case F95_MEMBER_REF:    outx_memberRef(l, v); break;
    case ARRAY_REF:         outx_arrayRef(l, v); break;
    case F_SUBSTR_REF:      outx_characterRef(l, v); break;
    case F95_ARRAY_CONSTRUCTOR:     outx_arrayConstructor(l, v); break;
    case F95_STRUCT_CONSTRUCTOR:    outx_structConstructor(l, v); break;

    case XMP_COARRAY_REF:   outx_coarrayRef(l, v); break;
    /*
     * operators
     */
    case PLUS_EXPR:
    case MINUS_EXPR:
    case MUL_EXPR:
    case DIV_EXPR:
    case POWER_EXPR:
    case LOG_EQ_EXPR:
    case LOG_NEQ_EXPR:
    case LOG_GE_EXPR:
    case LOG_GT_EXPR:
    case LOG_LE_EXPR:
    case LOG_LT_EXPR:
    case LOG_AND_EXPR:
    case LOG_OR_EXPR:
    case F_EQV_EXPR:
    case F_NEQV_EXPR:
    case F_CONCAT_EXPR:     outx_binaryOp(l, v); break;
    case LOG_NOT_EXPR:
    case UNARY_MINUS_EXPR:  outx_unaryOp(l, v); break;


    case F95_USER_DEFINED_BINARY_EXPR:
    case F95_USER_DEFINED_UNARY_EXPR: outx_udefOp(l, v); break;

    /*
     * misc.
     */
    case F_IMPLIED_DO:          outx_doLoop(l, v); break;
    case F_INDEX_RANGE:         outx_indexRangeInList(l, v); break;
    case F_SCENE_RANGE_EXPR:    outx_sceneRange(l, v); break;
    case F_MODULE_INTERNAL:
        /*
         * When using other module, F_MODULE_INTERNAL is set as dummy
         * expression insted of real value defined in module.
         * We emit dummy FintConstant for F_Back.
         */
        outx_printi(l, "<!-- dummy value for imported type -->\n");
        outx_intAsConst(l, -INT_MAX);
        break;

    /*
     * elements to skip
     */
    case F_DATA_DECL:
    case F_EQUIV_DECL:
    case F95_TYPEDECL_STATEMENT:
    case FIRST_EXECUTION_POINT:
    case F95_INTERFACE_STATEMENT:
    case F95_USE_STATEMENT:
    case F95_USE_ONLY_STATEMENT:
        break;

    /*
     * child elements
     */
    case LIST:
        {
            list lp;
            FOR_ITEMS_IN_LIST(lp, v)
                outx_expv(l, LIST_ITEM(lp));
        }
        break;

    /*
     * invalid or no corresponding tag
     */
    case ERROR_NODE:
    case BASIC_TYPE_NODE:
    case DEFAULT_LABEL:
    case ID_LIST:
    case VAR_DECL:
    case EXT_DECL:
    case F_PROGRAM_STATEMENT:
    case F_BLOCK_STATEMENT:
    case F_SUBROUTINE_STATEMENT:
    case F_FUNCTION_STATEMENT:
    case F_INCLUDE_STATEMENT:
    case F_END_STATEMENT:
    case F_TYPE_DECL:
    case F_COMMON_DECL:
    case F_EXTERNAL_DECL:
    case F_INTRINSIC_DECL:
    case F_IMPLICIT_DECL:
    case F_NAMELIST_DECL:
    case F_SAVE_DECL:
    case F_PARAM_DECL:
    case F_DUP_DECL:
    case F_UNARY_MINUS:
    case F_ENDDO_STATEMENT:
    case F_ELSEWHERE_STATEMENT:
    case F_ENDWHERE_STATEMENT:
    case F_ENDSELECT_STATEMENT:
    case F_IF_STATEMENT:
    case F_ELSEIF_STATEMENT:
    case F_ELSE_STATEMENT:
    case F_ENDIF_STATEMENT:
    case F_ASSIGN_LABEL_STATEMENT:
    case F_GOTO_STATEMENT:
    case F_ASGOTO_STATEMENT:
    case F_ARITHIF_STATEMENT:
    case F_CALL_STATEMENT:
    case F_CRAY_POINTER_DECL:
    case F_SET_EXPR:
    case F_LABEL_REF:
    case F_PLUS_EXPR:
    case F_MINUS_EXPR:
    case F_MUL_EXPR:
    case F_DIV_EXPR:
    case F_UNARY_MINUS_EXPR:
    case F_POWER_EXPR:
    case F_EQ_EXPR:
    case F_GT_EXPR:
    case F_GE_EXPR:
    case F_LT_EXPR:
    case F_LE_EXPR:
    case F_NE_EXPR:
    case F_OR_EXPR:
    case F_AND_EXPR:
    case F_NOT_EXPR:
    case F_ARRAY_REF:
    case F_STARSTAR:
    case F_TRUE_CONSTANT:
    case F_FALSE_CONSTANT:
    case F_TYPE_NODE:
    case F95_CONSTANT_WITH:
    case F95_TRUE_CONSTANT_WITH:
    case F95_FALSE_CONSTANT_WITH:
    case F95_ENDPROGRAM_STATEMENT:
    case F95_ENDSUBROUTINE_STATEMENT:
    case F95_ENDFUNCTION_STATEMENT:
    case F95_MODULE_STATEMENT:
    case F95_ENDMODULE_STATEMENT:
    case F95_ENDINTERFACE_STATEMENT:
    case F95_CONTAINS_STATEMENT:
    case F95_RECURSIVE_SPEC:
    case F95_PURE_SPEC:
    case F95_ELEMENTAL_SPEC:
    case F95_DIMENSION_DECL:
    case F95_ENDTYPEDECL_STATEMENT:
    case F95_PRIVATE_STATEMENT:
    case F95_SEQUENCE_STATEMENT:
    case F95_PARAMETER_SPEC:
    case F95_ALLOCATABLE_SPEC:
    case F95_DIMENSION_SPEC:
    case F95_EXTERNAL_SPEC:
    case F95_INTENT_SPEC:
    case F95_INTRINSIC_SPEC:
    case F95_OPTIONAL_SPEC:
    case F95_POINTER_SPEC:
    case F95_SAVE_SPEC:
    case F95_TARGET_SPEC:
    case F95_PUBLIC_SPEC:
    case F95_PRIVATE_SPEC:
    case F95_IN_EXTENT:
    case F95_OUT_EXTENT:
    case F95_INOUT_EXTENT:
    case F95_KIND_SELECTOR_SPEC:
    case F95_LEN_SELECTOR_SPEC:
    case F95_STAT_SPEC:
    case F95_TRIPLET_EXPR:
    case F95_PUBLIC_STATEMENT:
    case F95_OPTIONAL_STATEMENT:
    case F95_POINTER_STATEMENT:
    case F95_INTENT_STATEMENT:
    case F95_TARGET_STATEMENT:
    case F_ASTERISK:
    case F_EXTFUNC:
    case F_DOUBLE_CONSTANT:
    case F95_ASSIGNOP:
    case F95_DOTOP:
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
    case F95_USER_DEFINED:
    case F95_MODULEPROCEDURE_STATEMENT:
    case F95_ARRAY_ALLOCATION:
    case F95_ALLOCATABLE_STATEMENT:
    case F95_GENERIC_SPEC:
    case XMP_CODIMENSION_SPEC:
    case EXPR_CODE_END:

        if(debug_flag)
            expv_output(v, stderr);
        fatal("invalid exprcode : %s", EXPR_CODE_NAME(code));
        abort();

    case OMP_PRAGMA:
      outx_OMP_pragma(l, v);
      break;

    case XMP_PRAGMA:
      outx_XMP_pragma(l, v);
      break;
#endif
    default:
/*
        fatal("unkown exprcode : %d", code);
*/
        printf(" outx_expv  code : %d\n",code);
        fflush(stdout);
        abort();
    }
}

static void mark_type_desc_in_structure(TYPE_DESC tp);
//static void check_type_desc(TYPE_DESC tp);


static void
mark_type_desc(TYPE_DESC tp)
{
    if (tp == NULL || TYPE_IS_REFERENCED(tp) || IS_MODULE(tp))
        return;

    if (TYPE_REF(tp) != NULL) {
        TYPE_DESC sTp = NULL;
        if (IS_ARRAY_TYPE(tp)){
            mark_type_desc(array_element_type(tp));
        }
        sTp = reduce_type(TYPE_REF(tp));
        mark_type_desc(sTp);
        TYPE_REF(tp) = sTp;
    }

    collect_type_desc(TYPE_KIND(tp));
    collect_type_desc(TYPE_LENG(tp));
    collect_type_desc(TYPE_DIM_SIZE(tp));
    collect_type_desc(TYPE_DIM_UPPER(tp));
    collect_type_desc(TYPE_DIM_LOWER(tp));
    collect_type_desc(TYPE_DIM_STEP(tp));

    TYPE_LINK_ADD(tp, type_list, type_list_tail);
    TYPE_IS_REFERENCED(tp) = 1;

    if (IS_STRUCT_TYPE(tp))
        mark_type_desc_in_structure(tp);
}

static void
mark_type_desc_in_structure(TYPE_DESC tp)
{
    ID id;
    TYPE_DESC itp, siTp;

    FOREACH_MEMBER(id, tp) {
        itp = ID_TYPE(id);
        siTp = reduce_type(itp);
        mark_type_desc(siTp);
        ID_TYPE(id) = siTp;
        if(IS_STRUCT_TYPE(itp))
            mark_type_desc_in_structure(itp);
        if (VAR_INIT_VALUE(id) != NULL)
            collect_type_desc(VAR_INIT_VALUE(id));
    }
}

static void
collect_type_desc(expv v)
{
    list lp;
    TYPE_DESC sTp;

    if (v == NULL) return;
    sTp = reduce_type(EXPV_TYPE(v));
    mark_type_desc(sTp);
    EXPV_TYPE(v) = sTp;
    if (EXPR_CODE_IS_TERMINAL(EXPV_CODE(v))) return;

    FOR_ITEMS_IN_LIST(lp, v)
        collect_type_desc(LIST_ITEM(lp));
}


void
add_type_ext_id(EXT_ID ep)
{
    TYPE_EXT_ID te = (TYPE_EXT_ID)malloc(sizeof(struct type_ext_id));
    bzero(te, sizeof(struct type_ext_id));
    te->ep = ep;
    FUNC_EXT_LINK_ADD(te, type_ext_id_list, type_ext_id_last);
}


static void
mark_type_desc_in_id_list(ID ids)
{
    ID id;
    TYPE_DESC sTp;
    FOREACH_ID(id, ids) {
        sTp = reduce_type(ID_TYPE(id));
        mark_type_desc(sTp);
        ID_TYPE(id) = sTp;
        collect_type_desc(ID_ADDR(id));
        switch(ID_CLASS(id)) {
        case CL_PARAM:
            collect_type_desc(VAR_INIT_VALUE(id));
            break;
        case CL_VAR:
            collect_type_desc(VAR_INIT_VALUE(id));
            /* fall through */
        case CL_PROC:
            if (PROC_EXT_ID(id) && EXT_TAG(PROC_EXT_ID(id)) != STG_UNKNOWN &&
                (PROC_CLASS(id) == P_INTRINSIC ||
                 PROC_CLASS(id) == P_EXTERNAL ||
                 PROC_CLASS(id) == P_DEFINEDPROC)) {
                /* symbol declared as intrinsic */
                add_type_ext_id(PROC_EXT_ID(id));
                sTp = reduce_type(EXT_PROC_TYPE(PROC_EXT_ID(id)));
                mark_type_desc(sTp);
                EXT_PROC_TYPE(PROC_EXT_ID(id)) = sTp;
                /*
                 * types of argmument below may be verbose(not used).
                 * But to pass consistency check in backend, we choose to
                 * output these types.
                 */
                collect_type_desc(EXT_PROC_ARGS(PROC_EXT_ID(id)));
            }
            // TODO
            if (id->use_assoc != NULL) {
                TYPE_EXT_ID te =
                    (TYPE_EXT_ID)malloc(sizeof(struct type_ext_id));
                bzero(te, sizeof(struct type_ext_id));
                te->ep = PROC_EXT_ID(id);
                FUNC_EXT_LINK_ADD(te, type_module_proc_list,
                                  type_module_proc_last);
            }
            break;
        default:
            break;
        }
    }
}


outx_kind(int l, TYPE_DESC tp)
{
    static expv doubledKind = NULL;
    expv vkind;

    if(doubledKind == NULL)
        doubledKind = expv_int_term(INT_CONSTANT, type_INT, KIND_PARAM_DOUBLE);

    if(IS_DOUBLED_TYPE(tp))
        vkind = doubledKind;
    else if(tp && TYPE_KIND(tp))
        vkind = TYPE_KIND(tp);
    else
        return;

    outx_childrenWithTag(l, "kind", vkind);
}


/**
 * output basicType of coarray
 */
static void
outx_coShape(int l, TYPE_DESC tp)
{
  list lp;
  codims_desc *codims = tp->codims;

  outx_printi(l, "<coShape>\n");

  FOR_ITEMS_IN_LIST(lp, codims->cobound_list){
    expr cobound = LIST_ITEM(lp);
    expr upper = EXPR_ARG2(cobound);
    ARRAY_ASSUME_KIND defaultAssumeKind = ASSUMED_SHAPE;

    if (upper && EXPR_CODE(upper) == F_ASTERISK){
      upper = NULL;
      defaultAssumeKind = ASSUMED_SIZE;
    }

    outx_indexRange0(l+1, ASSUMED_NONE, defaultAssumeKind,
                     EXPR_ARG1(cobound), upper, EXPR_ARG3(cobound));
  }

  outx_close(l, "coShape");
}


/**
 * output basicType of character
 */
static void
outx_characterType(int l, TYPE_DESC tp)
{
    const int l1 = l + 1, l2 = l1 + 1;
    int charLen = TYPE_CHAR_LEN(tp);
    expv vcharLen = TYPE_LENG(tp);
    const char *tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
    TYPE_DESC tRef = TYPE_REF(tp);

    outx_typeAttrs(l, tp, "FbasicType", 0);

    if(tRef && checkBasic(tRef)) {
        tp = tRef;
        tRef = NULL;
    }

    if (tRef) {
        outx_print(" ref=\"C" ADDRX_PRINT_FMT "\"/>\n", Addr2Uint(tRef));
    }
    else if (TYPE_KIND(tp) || charLen != 1 || vcharLen != NULL || tp->codims){
        outx_print(" ref=\"%s\">\n", tid);
        outx_kind(l1, tp);

        if(charLen != 1|| vcharLen != NULL) {
            outx_tag(l1, "len");
            if(IS_CHAR_LEN_UNFIXED(tp) == FALSE) {
                if(vcharLen != NULL)
                    outx_expv(l2, vcharLen);
                else
                    outx_intAsConst(l2, TYPE_CHAR_LEN(tp));
            }
            outx_close(l1, "len");
        }
        if (tp->codims) outx_coShape(l+1, tp);
        outx_close(l, "FbasicType");
    } else {
        outx_print(" ref=\"%s\"/>\n", tid);
    }
}

/**
 * output basicType except character, array
 */
static void
outx_basicTypeNoCharNoAry(int l, TYPE_DESC tp)
{
    TYPE_DESC rtp = TYPE_REF(tp);
    assert(rtp);
    outx_typeAttrs(l, tp, "FbasicType", 0);
    if (tp->codims){
      outx_print(" ref=\"%s\">\n", getTypeID(rtp));
      outx_coShape(l+1, tp);
      outx_close(l ,"FbasicType");
    }
    else
      outx_print(" ref=\"%s\"/>\n", getTypeID(rtp));
}


/**
 * output basicType except character, array, without TYPE_REF.
 */
static void
outx_basicTypeNoCharNoAryNoRef(int l, TYPE_DESC tp)
{

  /* TYPE_FUNCTION comes here in the following case (maybe for the reference
     in the argument list). This is only an ad-hoc fix.

      subroutine sub(subsub)
      implicit none
      external subsub
      end

  */
  if (IS_FUNCTION_TYPE(tp)) return;


    outx_typeAttrs(l, tp, "FbasicType", 0);

    /* Output type as generic numeric type when type is not fixed */
    if (TYPE_IS_NOT_FIXED(tp) && TYPE_BASIC_TYPE(tp) == TYPE_UNKNOWN) {
        outx_print(" ref=\"%s\"", "FnumericAll");
    } else {
        /* tp is basic data type */
        outx_print(" ref=\"%s\"", getBasicTypeID(TYPE_BASIC_TYPE(tp)));
    }

    if (TYPE_KIND(tp) || IS_DOUBLED_TYPE(tp) || tp->codims){
        outx_print(">\n");
        outx_kind(l + 1, tp);
        if (tp->codims) outx_coShape(l+1, tp);
        outx_close(l, "FbasicType");
    } else {
        outx_print("/>\n");
    }
}


/**
 * output basicType of array
 */
static void
outx_arrayType(int l, TYPE_DESC tp)
{
    const int l1 = l + 1;

      outx_typeAttrs(l, tp, "FbasicType", 0);
      outx_print(" ref=\"%s\">\n", getTypeID(array_element_type(tp)));

      outx_indexRangeOfType(l1, tp);

      if (tp->codims) outx_coShape(l1, tp);

      outx_close(l ,"FbasicType");
}


/**
 * output functionType of external symbol
 */
static void
outx_functionType_EXT(int l, EXT_ID ep)
{
    list lp;
    TYPE_DESC tp;
    const int l1 = l + 1, l2 = l1 + 1;
    const char *rtid;

    if(EXT_PROC_IS_OUTPUT(ep))
        return;

    EXT_PROC_IS_OUTPUT(ep) = TRUE;

    tp = EXT_PROC_TYPE(ep);
    outx_typeAttrOnly_functionTypeWithResultVar(l, ep, "FfunctionType");

    if(tp) {
        if(IS_SUBR(tp))
            rtid = "Fvoid";
        else if(IS_FUNCTION_TYPE(tp))
            rtid = getTypeID(TYPE_REF(tp));
        else
            rtid = getTypeID(tp);
    } else {
        rtid = "Fvoid";
    }

    outx_print(" return_type=\"%s\"", rtid);
    outx_true(EXT_PROC_IS_PROGRAM(ep), "is_program");
    outx_true(EXT_PROC_IS_INTRINSIC(ep), "is_intrinsic");

    if (tp) {
        outx_true(TYPE_IS_RECURSIVE(tp), "is_recursive");
        outx_true(TYPE_IS_PURE(tp), "is_pure");
        outx_true(TYPE_IS_ELEMENTAL(tp), "is_elemental");

        if (TYPE_IS_EXTERNAL(tp) ||
            (XMP_flag && !TYPE_IS_FOR_FUNC_SELF(tp) &&
             !EXT_PROC_IS_INTRINSIC(ep) && !EXT_PROC_IS_MODULE_PROCEDURE(ep) && !EXT_PROC_IS_INTERNAL(ep))){
          outx_true(TRUE, "is_external");
        }

        outx_true(TYPE_IS_PUBLIC(tp), "is_public");
        outx_true(TYPE_IS_PRIVATE(tp), "is_private");
    }

    if(EXT_PROC_ARGS(ep) == NULL) {
        outx_print("/>\n");
    } else {
        outx_print(">\n");
        outx_tag(l1, "params");
        FOR_ITEMS_IN_LIST(lp, EXT_PROC_ARGS(ep))
            outx_expvNameWithType(l2, EXPR_ARG1(LIST_ITEM(lp)));
        outx_close(l1, "params");
        outx_close(l, "FfunctionType");
    }
}


/**
 * output FstructType
 */
static void
outx_structType(int l, TYPE_DESC tp)
{
    ID id;
    int l1 = l + 1, l2 = l1 + 1, l3 = l2 + 1;

    outx_typeAttrs(l, tp ,"FstructType", TOPT_NEXTLINE);
    outx_tag(l1, "symbols");

    FOREACH_MEMBER(id, tp) {
        outx_printi(l2, "<id type=\"%s\">\n", getTypeID(ID_TYPE(id)));
        outx_symbolName(l3, ID_SYM(id));
        if (VAR_INIT_VALUE(id) != NULL) {
            outx_value(l3, VAR_INIT_VALUE(id));
        }
        outx_close(l2, "id");
    }

    outx_close(l1,"symbols");
    outx_close(l,"FstructType");
}


/**
 * output types in typeTable
 */
static void
outx_type(int l, TYPE_DESC tp)
{
    TYPE_DESC tRef = TYPE_REF(tp);

    if (IS_SUBR(tp)) {
        /* output nothing */
    } else if(IS_CHAR(tp)) {
        if(checkBasic(tp) == FALSE || checkBasic(tRef) == FALSE)
            outx_characterType(l, tp);
    } else if(IS_ARRAY_TYPE(tp)) {
        outx_arrayType(l, tp);
    } else if(IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL) {
        outx_structType(l, tp);
    } else if (tRef != NULL) {
        if (has_attribute_except_func_attrs(tp) ||
            TYPE_KIND(tRef) ||
            IS_DOUBLED_TYPE(tRef) ||
            IS_STRUCT_TYPE(tRef)) {
            outx_basicTypeNoCharNoAry(l, tp);
        }
    } else if (tRef == NULL) {
        if(checkBasic(tp) == FALSE)
            outx_basicTypeNoCharNoAryNoRef(l, tp);
    } else {
        ;
/*
        fatal("%s: type not covered yet.", __func__);
 */
    }
}


/**
 * distinguish the id is to be output
 */
static int
id_is_visibleVar(ID id)
{
    if (ID_DEFINED_BY(id) != NULL) {
        TYPE_DESC tp = ID_TYPE(id);
        if (tp == NULL) {
            return FALSE;
        }
        if (IS_GENERIC_TYPE(tp)) {
            /* The generic function visibility is differed from
             * function and subroutine.
             */
            return TRUE;
        }
        if (ID_CLASS(id) == CL_PROC &&
            CRT_FUNCEP != NULL &&
            CRT_FUNCEP != PROC_EXT_ID(id)) {
            return FALSE;
        }
        if ((is_outputed_module && CRT_FUNCEP == NULL)
            && (TYPE_IS_PUBLIC(tp) || TYPE_IS_PRIVATE(tp))) {
            return TRUE;
        }
        return FALSE;
    }

    switch(ID_CLASS(id)) {
    case CL_VAR:
        if(VAR_IS_IMPLIED_DO_DUMMY(id))
            return FALSE;
        break;
    case CL_PARAM:
        return TRUE;
    case CL_NAMELIST:
        return TRUE;
    case CL_PROC:
        if(PROC_CLASS(id) == P_DEFINEDPROC) {
            /* this id is of function.
               Checkes if this id is of the current function or not. */
            if(CRT_FUNCEP == PROC_EXT_ID(id)) {
                return TRUE;
            } else {
                return FALSE;
            }
        }
    default:
        switch(ID_STORAGE(id)) {
        case STG_TAGNAME:
            return TRUE;
        case STG_UNKNOWN:
        case STG_NONE:
            return FALSE;
        default:
            break;
        }
    }

    return TRUE;
}


/**
 * output symbols in FfunctionDefinition
 */
static void
outx_definition_symbols(int l, EXT_ID ep)
{
    ID id;
    const int l1 = l + 1;

    outx_tag(l, "symbols");

    FOREACH_ID(id, EXT_PROC_ID_LIST(ep)) {
        if(id_is_visibleVar(id) && IS_MODULE(ID_TYPE(id)) == FALSE)
            outx_id(l1, id);
    }

    /* print common ids */
    FOREACH_ID(id, EXT_PROC_COMMON_ID_LIST(ep)) {
        if(IS_MODULE(ID_TYPE(id)) == FALSE)
            outx_id(l1, id);
    }

    /* print label */
    /* decided not to output labels
    FOREACH_ID(id, EXT_PROC_LABEL_LIST(ep)) {
        outx_id(l1, id);
    }
    */

    outx_close(l, "symbols");
}


/**
 * output FcommonDecl
 */
static void
outx_commonDecl(int l, ID cid)
{
    list lp;
    expv var;
    char buf[256];
    const int l1 = l + 1, l2 = l1 + 1;

    outx_tagOfDecl1(l, xtag(F_COMMON_DECL), ID_LINE(cid));

    if(COM_IS_BLANK_NAME(cid) == FALSE)
        sprintf(buf, " name=\"%s\"", ID_NAME(cid));
    else
        buf[0] = '\0';

    outx_tag(l1, "varList%s", buf);

    FOR_ITEMS_IN_LIST(lp, COM_VARS(cid)) {
        var = LIST_ITEM(lp);
        outx_varRef_EXPR(l2, var);
    }

    outx_close(l1, "varList");
    outx_close(l, xtag(F_COMMON_DECL));
}

/**
 * output FequivalenceDecl
 */
static void
outx_equivalenceDecl(int l, expv v)
{
    const int l1 = l + 1;
    expv v1 = v;

    outx_tagOfDecl1(l, xtag(F_EQUIV_DECL), EXPV_LINE(v));
    outx_varRef_EXPR(l1, EXPR_ARG1(v1));
    outx_varList(l1, EXPR_ARG2(v1), NULL);
    outx_close(l, xtag(F_EQUIV_DECL));
}



static int
qsort_compare_id(const void *v1, const void *v2)
{
    int o1 = ID_ORDER(*(ID*)v1);
    int o2 = ID_ORDER(*(ID*)v2);

    return (o1 == o2) ? 0 : ((o1 < o2) ? -1 : 1);
}


/**
 * sort id by order value
 */
ID*
genSortedIDs(ID ids, int *retnIDs)
{
    ID id, *sortedIDs;
    int i = 0, nIDs = 0;

    if(ids == NULL)
        return NULL;

    FOREACH_ID(id, ids)
        ++nIDs;

    if(nIDs == 0)
        return NULL;

    sortedIDs = (ID*)malloc(nIDs * sizeof(ID));

    FOREACH_ID(id, ids)
        sortedIDs[i++] = id;

    qsort((void*)sortedIDs, nIDs, sizeof(ID), qsort_compare_id);
    *retnIDs = nIDs;

    return sortedIDs;
}



#define IS_NO_PROC_OR_DECLARED_PROC(id) \
    ((ID_CLASS(id) != CL_PROC || \
      PROC_CLASS(id) == P_EXTERNAL || \
      (ID_TYPE(id) && TYPE_IS_EXTERNAL(ID_TYPE(id))) || \
      (ID_TYPE(id) && TYPE_IS_INTRINSIC(ID_TYPE(id))) ||        \
      PROC_CLASS(id) == P_UNDEFINEDPROC || \
      PROC_CLASS(id) == P_DEFINEDPROC) \
  && (PROC_EXT_ID(id) == NULL ||             \
      PROC_CLASS(id) == P_UNDEFINEDPROC || \
      PROC_CLASS(id) == P_DEFINEDPROC || ( \
      EXT_PROC_IS_INTERFACE(PROC_EXT_ID(id)) == FALSE && \
      EXT_PROC_IS_INTERFACE_DEF(PROC_EXT_ID(id)) == FALSE)) \
  && (ID_TYPE(id) \
      && IS_MODULE(ID_TYPE(id)) == FALSE            \
      && (IS_SUBR(ID_TYPE(id)) == FALSE || \
          has_attribute_except_private_public(ID_TYPE(id)))))


static int
is_id_used_in_struct_member(ID id, TYPE_DESC sTp)
{
    /*
     * FIXME:
     *  Actually, it is not checked if id is used in sTp's member.
     *  Instead, just checking line number.
     */
    ID mId;

    if (ID_LINE(id) == NULL) {
        return FALSE;
    }

    FOREACH_MEMBER(mId, sTp) {
        if (ID_LINE(mId) == NULL) {
            continue;
        }
        if (ID_LINE_FILE_ID(id) == ID_LINE_FILE_ID(mId)) {
            if (ID_LINE_NO(id) < ID_LINE_NO(mId)) {
                return TRUE;
            }
        }
    }

    return FALSE;
}


static void
emit_decl(int l, ID id)
{
    if (ID_IS_EMITTED(id) == TRUE) {
        return;
    }
    if (ID_IS_OFMODULE(id) == TRUE) {
        return;
    }

    switch(ID_CLASS(id)) {
    case CL_NAMELIST:
        outx_namelistDecl(l, id);
        break;

    case CL_PARAM:
        if (id_is_visibleVar(id))
            outx_varDecl(l, id);
        break;

    case CL_ENTRY:
        break;

    default:
        switch (ID_STORAGE(id)) {
            case STG_ARG:
            case STG_SAVE:
            case STG_AUTO:
            case STG_EQUIV:
            case STG_COMEQ:
            case STG_COMMON:
                if (id_is_visibleVar(id) &&
                    IS_NO_PROC_OR_DECLARED_PROC(id)) {
                    outx_varDecl(l, id);
                }
                break;

            case STG_TAGNAME:
                break;

            case STG_EXT:
                if (id_is_visibleVar(id) &&
                    IS_NO_PROC_OR_DECLARED_PROC(id)) {
                    outx_varDecl(l, id);
                }
                break;

            case STG_UNKNOWN:
            case STG_NONE:
                break;
        }
        break;
    }
}


/**
 * output declarations with pragmas
 */
static void
outx_declarations1(int l, EXT_ID parent_ep, int outputPragmaInBody)
{
    const int l1 = l + 1;
    list lp;
    ID id, *ids;
    EXT_ID ep;
    expv v;
    int i, nIDs;
    int hasResultVar = (EXT_PROC_RESULTVAR(parent_ep) != NULL) ? TRUE : FALSE;
    const char *myName = SYM_NAME(EXT_SYM(parent_ep));
    TYPE_DESC tp;

    outx_tag(l, "declarations");

    /*
     * FuseDecl
     */
/*
    FOR_ITEMS_IN_LIST(lp, EXT_PROC_BODY(parent_ep)) {
        v = LIST_ITEM(lp);
        switch(EXPV_CODE(v)) {
        case F95_USE_STATEMENT:
            outx_useDecl(l1, v);
            break;
        case F95_USE_ONLY_STATEMENT:
            outx_useOnlyDecl(l1, v);
            break;
        default:
            break;
        }
    }
*/

    ids = genSortedIDs(EXT_PROC_ID_LIST(parent_ep), &nIDs);

    if (ids) {
        /*
         * Firstly emit struct base type (TYPE_REF(tp) == NULL).
         * ex) type(x)::p = x(1)
         */
        for (i = 0; i < nIDs; i++) {
            id = ids[i];

            if (ID_CLASS(id) != CL_TAGNAME) {
                continue;
            }
            if (ID_IS_EMITTED(id) == TRUE) {
                continue;
            }

            if (ID_IS_OFMODULE(id) == TRUE) {
                continue;
            }

            tp = ID_TYPE(id);
            if (IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL) {
                int j;
                for (j = 0; j < nIDs; j++) {
                    if (i != j) {
                        if (ID_IS_EMITTED(ids[j]) == TRUE) {
                            continue;
                        }
                        if (is_id_used_in_struct_member(ids[j], tp) == TRUE) {
                            emit_decl(l1, ids[j]);
                            ID_IS_EMITTED(ids[j]) = TRUE;
                        }
                    }
                }
                outx_structDecl(l1, id);
                ID_IS_EMITTED(id) = TRUE;
            }
        }

        /*
         * varDecl except structDecl, namelistDecl
         */
        for (i = 0; i < nIDs; ++i) {
            id = ids[i];

            if (hasResultVar == TRUE &&
                strcasecmp(myName, SYM_NAME(ID_SYM(id))) == 0) {
                continue;
            }

            emit_decl(l1, id);
            ID_IS_EMITTED(id) = TRUE;
        }
        free(ids);
    }


    /*
     * FdataDecl / FequivalenceDecl
     */
    FOR_ITEMS_IN_LIST(lp, EXT_PROC_BODY(parent_ep)) {
        v = LIST_ITEM(lp);
        switch(EXPV_CODE(v)) {
        case F_DATA_DECL:
            outx_dataDecl(l1, v);
            break;
        case F_EQUIV_DECL:
            outx_equivalenceDecl(l1, v);
            break;
        default:
            break;
        }
    }

    /*
     * FcommonDecl
     */
    FOREACH_ID(id, EXT_PROC_COMMON_ID_LIST(parent_ep)) {
        outx_commonDecl(l1, id);
    }

    /*
     * FinterfaceDecl
     */
    FOREACH_EXT_ID(ep, EXT_PROC_INTERFACES(parent_ep)) {
        outx_interfaceDecl(l1, ep);
    }

    /*
     * FpragmaStatement
     */
#if 0
    if(outputPragmaInBody) {
        FOR_ITEMS_IN_LIST(lp, EXT_PROC_BODY(parent_ep)) {
            v = LIST_ITEM(lp);
            switch(EXPV_CODE(v)) {
            case F_PRAGMA_STATEMENT:
                // for FmoduleDefinition-declarations
                outx_pragmaStatement(l1, v);
                break;
            case XMP_PRAGMA:
                outx_XMP_pragma(l1, v);
                break;
            default:
                break;
            }
        }
    }
#endif
    outx_close(l, "declarations");
}


/**
 * output declarations
 */
static void
outx_declarations(int l, EXT_ID parent_ep)
{
    outx_declarations1(l, parent_ep, FALSE);
}




/**
 * output FmoduleProcedureDecl
 */
static void
outx_moduleProcedureDecl(int l, EXT_ID parent_ep, SYMBOL parentName)
{
    const int l1 = l + 1;
    int hasModProc = FALSE;
    EXT_ID ep;

    FOREACH_EXT_ID(ep, parent_ep) {
        if(EXT_TAG(ep) == STG_EXT &&
            EXT_PROC_IS_MODULE_PROCEDURE(ep)) {
            hasModProc = TRUE;
            break;
        }
    }

    if(hasModProc == FALSE)
        return;

    if (EXT_PROC_IS_MODULE_SPECIFIED(parent_ep)) {
        outx_tagOfDecl1(l, "FmoduleProcedureDecl is_module_specified=\"true\"",
            GET_EXT_LINE(ep));
    } else {
        outx_tagOfDecl1(l, "FmoduleProcedureDecl", GET_EXT_LINE(ep));
    }

    if (is_emitting_xmod() == FALSE) {
        FOREACH_EXT_ID(ep, parent_ep) {
            if (EXT_TAG(ep) == STG_EXT &&
                EXT_PROC_IS_MODULE_PROCEDURE(ep) &&
                !EXT_IS_OFMODULE(ep)) {
                outx_symbolName(l1, EXT_SYM(ep));
            }
        }
    } else {
        if (parentName != NULL) {
            if (top_gene_pro != NULL) {
               cur_gene_pro = top_gene_pro;
               while (cur_gene_pro != NULL) {
                  if(strcasecmp(parentName->s_name,cur_gene_pro->belongProcName)==0) {
                     outx_symbolNameWithFunctionType(l1, cur_gene_pro->eId);
                  }
                  cur_gene_pro = cur_gene_pro->next;
               }
            }
        } else {
            fatal("module procedure w/o generic name.");
            /* not reached. */
            return;
        }
    }

    outx_close(l, "FmoduleProcedureDecl");
}

/**
 * output FfunctionDecl
 */
static void
outx_functionDecl(int l, EXT_ID ep)
{
    const int l1 = l + 1;
    CRT_FUNCEP_PUSH(ep);
    outx_tagOfDecl1(l, "FfunctionDecl", GET_EXT_LINE(ep));
    outx_symbolNameWithFunctionType(l1, ep);
    outx_declarations(l1, ep);
    outx_close(l, "FfunctionDecl");
    CRT_FUNCEP_POP;
}


static void
outx_innerDefinitions(int l, EXT_ID extids, SYMBOL parentName, int asDefOrDecl)
{
    EXT_ID ep;

    FOREACH_EXT_ID(ep, extids) {
        if (EXT_TAG(ep) != STG_EXT)
            continue;
        if (EXT_PROC_IS_ENTRY(ep) == TRUE)
            continue;
        if (!EXT_PROC_BODY(ep) && !EXT_PROC_ID_LIST(ep))
            continue;
        if (EXT_PROC_TYPE(ep) != NULL
            && TYPE_BASIC_TYPE(EXT_PROC_TYPE(ep)) == TYPE_MODULE)
            continue;

        if(asDefOrDecl) {
            outx_functionDefinition(l, ep);
        } else {
            if(EXT_PROC_IS_MODULE_PROCEDURE(ep) == FALSE)
                outx_functionDecl(l, ep);
        }
    }

    outx_moduleProcedureDecl(l, extids, parentName);
}


/**
 * output FcontainsStatement
 */
static void
outx_contains(int l, EXT_ID parent)
{
    EXT_ID contains;
    lineno_info *contains_line;

    contains = EXT_PROC_CONT_EXT_SYMS(parent);
    contains_line = EXT_PROC_CONT_EXT_LINE(parent);
    if (contains == NULL) {
        return;
    }
    assert(contains_line != NULL);
    outx_tagOfDecl1(l, "FcontainsStatement", contains_line);
    outx_innerDefinitions(l + 1, contains, NULL, TRUE);
    outx_close(l, "FcontainsStatement");
}


/**
 * output FinterfaceDecl
 */
static void
outx_interfaceDecl(int l, EXT_ID ep)
{
    EXT_ID extids;

    extids = EXT_PROC_INTR_DEF_EXT_IDS(ep);
    if(extids == NULL)
        return;

    if(EXT_IS_OFMODULE(ep) == TRUE)
        return;

    CRT_FUNCEP_PUSH(NULL);
    outx_printi(l, "<FinterfaceDecl");

    switch(EXT_PROC_INTERFACE_CLASS(ep)) {
        case INTF_DECL:
            if(EXT_IS_BLANK_NAME(ep) == FALSE)
                outx_printi(0, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));
            break;
        case INTF_ASSINGMENT:
            outx_true(TRUE, "is_assignment");
            break;
        case INTF_OPERATOR:
        case INTF_USEROP:
            outx_printi(0, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));
            outx_true(TRUE, "is_operator");
            break;
        default:
            /* never reach. here*/
            break;
    }

    outx_lineno(EXT_LINE(ep));
    outx_printi(0,">\n");
    outx_innerDefinitions(l + 1, extids, EXT_SYM(ep), FALSE);
    outx_close(l, "FinterfaceDecl");
    CRT_FUNCEP_POP;
}


/**
 * output FfunctionDefinition
 */
static void
outx_functionDefinition(int l, EXT_ID ep)
{
    const int l1 = l + 1, l2 = l + 2;

    CRT_FUNCEP_PUSH(ep);

    outx_tagOfDecl1(l, "FfunctionDefinition", GET_EXT_LINE(ep));
    outx_symbolNameWithFunctionType(l1, ep);
    outx_definition_symbols(l1, ep);
    outx_declarations(l1, ep);
    outx_tag(l1, "body");
    outx_expv(l2, EXT_PROC_BODY(ep));
    outx_contains(l2, ep);
    outx_close(l1, "body");
    outx_close(l, "FfunctionDefinition");

    CRT_FUNCEP_POP;
}


/**
 * recursively collect TYPE_DESC to type_list
 */
static void
collect_types1(EXT_ID extid)
{
    EXT_ID ep;
    TYPE_DESC tp, sTp;

    if(extid == NULL)
        return;

    /* collect used types */
    FOREACH_EXT_ID(ep, extid) {
        if((EXT_TAG(ep) != STG_EXT &&
            EXT_TAG(ep) != STG_COMMON) ||
            EXT_IS_DEFINED(ep) == FALSE) {
            /* STG_EXT
             *      is program, subroutine, function, module, interface.
             * STG_COMMON
             *      is block data.
             * EXT_IS_DEFINED
             *      is FALSE when the id is not declared
             *      but called as function or subroutine.
             */
            continue;
        }

        if (EXT_TAG(ep) == STG_EXT &&
            IS_MODULE(EXT_PROC_TYPE(ep)) == FALSE &&
            EXT_PROC_IS_INTERFACE_DEF(ep) == FALSE &&
            (EXT_PROC_IS_INTERFACE(ep) == FALSE ||
            EXT_IS_BLANK_NAME(ep) == FALSE)) {
            add_type_ext_id(ep);
        }

        /* symbols in CONTAINS */
        collect_types1(EXT_PROC_CONT_EXT_SYMS(ep));
        /* INTERFACE symbols */
        collect_types1(EXT_PROC_INTERFACES(ep));
        /* symbols in INTERFACE */
        collect_types1(EXT_PROC_INTR_DEF_EXT_IDS(ep));

        sTp = reduce_type(EXT_PROC_TYPE(ep));
        mark_type_desc(sTp);
        EXT_PROC_TYPE(ep) = sTp;
        collect_type_desc(EXT_PROC_ARGS(ep));
        mark_type_desc_in_id_list(EXT_PROC_ID_LIST(ep));
        collect_type_desc(EXT_PROC_BODY(ep));

        for(tp = EXT_PROC_STRUCT_DECLS(ep); tp != NULL; tp = TYPE_SLINK(tp)) {
            if(TYPE_IS_DECLARED(tp)) {
                mark_type_desc(tp);
                mark_type_desc_in_structure(tp);
            }
        }
    }
}


/**
 * output typeTable
 */
static void
outx_typeTable(int l)
{
    const int l1 = l + 1;
    TYPE_DESC tp;
    TYPE_EXT_ID te;

    outx_tag(l, "typeTable");

    for (tp = type_list; tp != NULL; tp = TYPE_LINK(tp)){
        outx_type(l1, tp);
    }

    FOREACH_TYPE_EXT_ID(te, type_module_proc_list) {
        if(te->ep != NULL)
            outx_functionType_EXT(l1, te->ep);
    }

    FOREACH_TYPE_EXT_ID(te, type_ext_id_list) {
        assert(EXT_TAG(te->ep) == STG_EXT);
        outx_functionType_EXT(l1, te->ep);
    }

    outx_close(l, "typeTable");
}


/**
 * output <id> node
 */
static void
outx_id_mod(int l, ID id)
{
    if(ID_STORAGE(id) == STG_EXT && PROC_EXT_ID(id) == NULL) {
/*
        fatal("outx_id: PROC_EXT_ID is NULL: symbol=%s", ID_NAME(id));
 */
    }

    if ((ID_CLASS(id) == CL_PROC || ID_CLASS(id) == CL_ENTRY) &&
        PROC_EXT_ID(id)) {
        outx_typeAttrOnly_functionType(l, PROC_EXT_ID(id), "id");
    } else {
        outx_typeAttrOnly_ID(l, id, "id");
    }

    const char *sclass = get_sclass(id);

    outx_print(" sclass=\"%s\"", sclass);
    outx_print(" original_name=\"%s\"", SYM_NAME(id->use_assoc->original_name));
    outx_print(" declared_in=\"%s\"", SYM_NAME(id->use_assoc->module_name));
    if(ID_IS_AMBIGUOUS(id))
        outx_print(" is_ambiguous=\"true\"");
    outx_print(">\n", SYM_NAME(id->use_assoc->module_name));
    outx_symbolName(l + 1, ID_SYM(id));
    outx_close(l, "id");
}

/**
 * output <identifiers> node
 */
static void
outx_identifiers(int l, ID ids)
{
    const int l1 = l + 1;
    ID id;

    outx_tag(l, "identifiers");

    FOREACH_ID(id, ids) {
        outx_id_mod(l1, id);
    }

    outx_close(l, "identifiers");
}


/**
 * output <interfaceDecls> node
 */
static void
outx_interfaceDecls(int l, ID ids)
{
    const int l1 = l + 1;
    ID id;
    EXT_ID ep;

    outx_tag(l, "interfaceDecls");

    FOREACH_ID(id, ids) {
        ep = PROC_EXT_ID(id);
        if (ep != NULL) {
            outx_interfaceDecl(l1, ep);
        }
    }

    outx_close(l, "interfaceDecls");
}

/**
 * output <XcalablempFortranModule> node
 */
static void
outx_module(struct module * mod)
{
    const int l = 0, l1 = l + 1, l2 = l1 + 1;
    struct depend_module * mp;
    outx_tag(l, "OmniFortranModule version=\"%s\"", F_MODULE_VER);

    outx_printi(l1, "<name>%s</name>\n", SYM_NAME(mod->name));

    mp = mod->depend.head;

    outx_tag(l1, "depends");

    while (mp != NULL) {
        outx_printi(l2, "<name>%s</name>\n", SYM_NAME(mp->module_name));
        mp = mp->next;
    }

    outx_close(l1, "depends");

    outx_typeTable(l1);

    outx_identifiers(l1, mod->head);

    outx_interfaceDecls(l1, mod->head);

    /* output pragmas etc in CURRENT_STATEMENTS */
    outx_tag(l1,"aux_info");
/*  if(CURRENT_STATEMENTS != NULL){ */
/*  if(UNIT_CTL_CURRENT_STATEMENTS(unit_ctls[unit_ctl_level]) != NULL){ */
    if((unit_ctls[unit_ctl_level])->current_statements != NULL) {
        list lp;
        expv v;
     /* FOR_ITEMS_IN_LIST(lp,CURRENT_STATEMENTS){ */
        FOR_ITEMS_IN_LIST(lp,(unit_ctls[unit_ctl_level])->current_statements){
         /* v = LIST_ITEM(lp); */
            v = (lp)->l_item;
            outx_expv(l1+1,v);
        }
    }
    outx_close(l1,"aux_info");

    outx_close(l, "OmniFortranModule");
}








/**
 * output module to .xmod file
 */
void
output_module_file(struct module * mod)
{
  char filename[255] = {0};
    ID id;
    EXT_ID ep;
    TYPE_EXT_ID te;
    TYPE_DESC sTp;
    int oEmitMode;
    expr modTypeList;
    list lp;
    expv v;

    unit_ctl_level = 0;

    if(flag_module_compile) {
        print_fp = stdout;
    } else {
        char tmp[255];
      /*if (modincludeDirv) snprintf(filename, sizeof(filename), "%s/", modincludeDirv);*/
        snprintf(tmp, sizeof(tmp), "%s.xmod", SYM_NAME(mod->name));
        strncat(filename, tmp, sizeof(filename) - strlen(filename) - 1);
        if ((print_fp = fopen(filename, "w")) == NULL) {
 /*
            fatal("could'nt open module file to write.");
 */
            return;
        }
    }
  
    oEmitMode = is_emitting_xmod();
    set_module_emission_mode(TRUE);
  
    type_list = NULL;

    type_module_proc_list = NULL;
    type_module_proc_last = NULL;
    type_ext_id_list = NULL;
    type_ext_id_last = NULL;

    mark_type_desc_in_id_list(mod->head);
/*  FOREACH_ID(id, mod->head) { */
    for ((id) = (mod->head); (id) != NULL ; (id) = (id)->next) {
        ep = PROC_EXT_ID(id);
        if (ep != NULL) {
            collect_types1(ep);
/*          FOREACH_TYPE_EXT_ID(te, type_ext_id_list) { */
            for ((te) = (type_ext_id_list); (te) != NULL ; (te) = (te)->next) {
                TYPE_DESC tp = EXT_PROC_TYPE(te->ep);
                if (tp && EXT_TAG(te->ep) == STG_EXT) {
                    sTp = reduce_type(EXT_PROC_TYPE(te->ep));
                    mark_type_desc(sTp);
                    EXT_PROC_TYPE(te->ep) = sTp;
                }
            }
        }
    }

#ifdef _HASH_
    modTypeList = collect_all_module_procedures_types();
#else
    modTypeList = NULL;
#endif
/*  FOR_ITEMS_IN_LIST(lp, modTypeList) { */
    if(modTypeList != NULL) {
    for(lp = (modTypeList)->v.e_lp; lp != NULL ; lp = (lp)->l_next){
/*      v = LIST_ITEM(lp); */
        v = (lp)->l_item;
        if (EXPV_INT_VALUE(EXPR_ARG1(v)) == 1) {
            if (EXPR_ARG3(v) != NULL) {
                ep = EXPV_ANY(EXT_ID, EXPR_ARG3(v));
                collect_types1(ep);
                sTp = reduce_type(EXT_PROC_TYPE(ep));
                mark_type_desc(sTp);
                EXT_PROC_TYPE(ep) = sTp;
            }
        }
    }
    }

    outx_module(mod);
    if(!flag_module_compile)
        fclose(print_fp);
}


// TODO check this
#define AVAILABLE_ID(id)                           \
    ID_CLASS(id)   == CL_NAMELIST ||               \
    (!TYPE_IS_PRIVATE(ID_TYPE(id))                 \
    && (      ID_CLASS(id)   == CL_VAR             \
           || ID_CLASS(id)   == CL_ENTRY           \
           || ID_CLASS(id)   == CL_PARAM           \
           || ID_CLASS(id)   == CL_CONTAINS        \
           || ID_CLASS(id)   == CL_TAGNAME         \
           || ID_CLASS(id)   == CL_UNKNOWN         \
           || ID_CLASS(id)   == CL_GENERICS        \
           || ID_CLASS(id)   == CL_PROC            \
       ) \
    && (      ID_STORAGE(id) != STG_NONE           \
        ))

/**
 * export public identifiers to module-manager.
 */
int
export_module(SYMBOL sym, ID ids, expv use_decls)
{
    ID id;
    list lp;
    struct depend_module * dep;
    struct module * mod = XMALLOC(struct module *, sizeof(struct module));
    extern int flag_do_module_cache;

    *mod = (struct module){0};
    mod->name = sym;

    for ((id) = (ids); (id) != NULL ; (id) = (id)->next) {
        if(AVAILABLE_ID(id))
            add_module_id(mod, id);
    }

    FOR_ITEMS_IN_LIST(lp, use_decls) {
        dep = XMALLOC(struct depend_module *, sizeof(struct depend_module));
        dep->module_name = EXPR_SYM(LIST_ITEM(lp));
        if (mod->depend.last == NULL) {
            mod->depend.head = dep;
            mod->depend.last = dep;
        } else {
            mod->depend.last->next = dep;
            mod->depend.last = dep;
        }
    }

    if (flag_do_module_cache == TRUE)
        add_module(mod);

    if (nerrors == 0)
        output_module_file(mod);

    return TRUE;
}
