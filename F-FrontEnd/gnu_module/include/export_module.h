/*
 *
 */
#ifndef _EXPORT_MODULE_H_
#define _EXPORT_MODULE_H_

#include <stdint.h>
#ifdef _HASH_
#include "hash.h"
#endif
#include "C-expr.h"
#include "F-datatype.h"
#include "F-ident.h"

#define F_MODULE_VER  "1.0"

#define FOREACH_IN_HASH(h, s, tab) \
    for (h = FirstHashEntry(tab, s); \
         h != NULL; \
         h = NextHashEntry(s))

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

#define TRUE  1
#define FALSE 0
#define true  1
#define false 0

#define XMALLOC(type, size) ((type)xmalloc2(size))
#define MEMCLEAN( m ) memset( m , 0x00 , sizeof(*(m)) )

extern void     error EXC_VARARGS(char *, fmt);
extern void     fatal EXC_VARARGS(char *, fmt);
extern void     warning EXC_VARARGS(char *, fmt);

#ifdef _HASH_
/*************
 * F-module-procedure.h
 *************/
struct generic_procedure_info_record {
    const char *genericProcName;        /* A name of a generic
                                         * procedure. Also used as a
                                         * hash key. */
    HashEntry *hPtr;                    /* A hash entry. */
    HashTable *modProcTbl;              /* A hash table of module
                                         * procedures which this
                                         * generic procedure consists
                                         * of. */
};
typedef struct generic_procedure_info_record *gen_proc_t;

struct module_procedure_info_record {
    const char *modProcName;    /* A name of a module procedure. Also
                                 * used as a hash key. */

    gen_proc_t belongsTo;       /* A generic procedure which consists
                                 * of this module procedure. */
    HashEntry *hPtr;            /* A hash entry. */
    EXT_ID eId;                 /* external id. */
    TYPE_DESC retType;          /* The return type of this procedure. */
    expv args;                  /* A dummy arguments list of this
                                 * procedure. */
    /*
     * Note:
     *  The retType and EXPV_TYPE() of each element in argSpec must
     *  exist in regular manner.
     */
};
typedef struct module_procedure_info_record *mod_proc_t;
#endif /* _HASH_ */

#define GEN_PROC_MOD_TABLE(gp)    ((gp)->modProcTbl)
#define MOD_PROC_EXT_ID(mp)       ((mp)->eId)

/*************
 * module-manager.h
 *************/

/* /\** */
/*  * use association information about ID. */
/*  *\/ */
/* struct use_assoc_info { */
/*     SYMBOL module_name;       /\* name of module which the ID declared. *\/ */
/*     SYMBOL original_name;     /\* original name of the ID. *\/ */
/* }; */

/**
 * list of module name.
 */
struct depend_module {
    struct depend_module * next;
    SYMBOL module_name;
};

/**
 * fortran module.
 */
struct module {
    struct module * next;
    struct {
        struct depend_module * head;
        struct depend_module * last;
    } depend;                 /* list of module name which this module depends on. */
    SYMBOL name;              /* name of this module. */
    ID head;                  /* public elements of this module. */
    ID last;
};


/*************
 * F-front.h
 *************/

/* max file path length */
#define MAX_PATH_LEN    8192

#define FILE_NAME_LEN   MAX_PATH_LEN
#define MAX_N_FILES 256
#define N_NESTED_FILE 25
extern int n_files;
extern char *file_names[];
#define FILE_NAME(id) file_names[id]

#define MAX_UNIT_CTL_CONTAINS   3

/* parser states */
enum prog_state {
    OUTSIDE,
    INSIDE,
    INDCL,
    INDATA,
    INEXEC,
    INSTRUCT,
    INCONT,     /* contains */
    ININTR      /* interface */
};


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

    ID                  local_symbols;
    TYPE_DESC           local_struct_decls;
    ID                  local_common_symbols;
    ID                  local_labels;
    EXT_ID              local_external_symbols;

    int                 implicit_none;
    int                 implicit_type_declared;
    TYPE_DESC           implicit_types[IMPLICIT_ALPHA_NUM];
    enum storage_class  implicit_stg[IMPLICIT_ALPHA_NUM];
    expv                implicit_decls;
    expv                initialize_decls;
    expv                equiv_decls;
    expv                use_decls;
} *UNIT_CTL;

#define UNIT_CTL_CURRENT_PROC_NAME(u)           ((u)->current_proc_name)
#define UNIT_CTL_CURRENT_PROC_CLASS(u)          ((u)->current_proc_class)
#define UNIT_CTL_CURRENT_PROCEDURE(u)           ((u)->current_procedure)
#define UNIT_CTL_CURRENT_STATEMENTS(u)          ((u)->current_statements)
#define UNIT_CTL_CURRENT_BLK_LEVEL(u)           ((u)->current_blk_level)
#define UNIT_CTL_CURRENT_EXT_ID(u)              ((u)->current_ext_id)
#define UNIT_CTL_CURRENT_STATE(u)               ((u)->current_state)
#define UNIT_CTL_CURRENT_INTERFACE(u)           ((u)->current_interface)
#define UNIT_CTL_LOCAL_SYMBOLS(u)               ((u)->local_symbols)
#define UNIT_CTL_LOCAL_STRUCT_DECLS(u)          ((u)->local_struct_decls)
#define UNIT_CTL_LOCAL_COMMON_SYMBOLS(u)        ((u)->local_common_symbols)
#define UNIT_CTL_LOCAL_LABELS(u)                ((u)->local_labels)
#define UNIT_CTL_LOCAL_EXTERNAL_SYMBOLS(u)      ((u)->local_external_symbols)
#define UNIT_CTL_IMPLICIT_NONE(u)               ((u)->implicit_none)
#define UNIT_CTL_IMPLICIT_TYPES(u)              ((u)->implicit_types)
#define UNIT_CTL_IMPLICIT_TYPE_DECLARED(u)      ((u)->implicit_type_declared)
#define UNIT_CTL_IMPLICIT_STG(u)                ((u)->implicit_stg)
#define UNIT_CTL_IMPLICIT_DECLS(u)              ((u)->implicit_decls)
#define UNIT_CTL_INITIALIZE_DECLS(u)            ((u)->initialize_decls)
#define UNIT_CTL_EQUIV_DECLS(u)                 ((u)->equiv_decls)
#define UNIT_CTL_USE_DECLS(u)                   ((u)->use_decls)

#define MAX_UNIT_CTL            16
#define MAX_UNIT_CTL_CONTAINS   3
extern UNIT_CTL unit_ctls[];
extern int unit_ctl_level;

extern TYPE_DESC type_REAL, type_INT, type_SUBR, type_CHAR, type_LOGICAL;
extern TYPE_DESC type_DREAL, type_COMPLEX, type_DCOMPLEX, type_CHAR_POINTER;
extern TYPE_DESC type_MODULE;
extern TYPE_DESC type_GNUMERIC_ALL;

#define CURRENT_UNIT_CTL            unit_ctls[unit_ctl_level]
#define LOCAL_SYMBOLS               UNIT_CTL_LOCAL_SYMBOLS(CURRENT_UNIT_CTL)



#define EMPTY_LIST list0(LIST)





typedef struct generic_procedure_list
{
    const char *modProcName;
    const char *belongProcName;
    EXT_ID eId;                 /* external id. */
    TYPE_DESC retType;          /* The return type of this procedure. */
    expv args;                  /* A dummy arguments list of this procedure. */
    struct generic_procedure_list  *next;
}
generic_procedure;


#endif /* _F_DATATYPE_H_ */

