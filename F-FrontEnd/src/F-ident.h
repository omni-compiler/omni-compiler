/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-ident.h
 */

#ifndef _F_IDENT_H_
#define _F_IDENT_H_

/* Fortran name class */
enum name_class {
    CL_UNKNOWN = 0,
    CL_PARAM,     /* parameter */
    CL_VAR,       /* variable name */
    CL_ENTRY,     /* entry name */
    CL_MAIN,      /* program name */
    CL_MODULE,    /* module name */
    CL_SUBMODULE, /* submodule name */
    CL_CONTAINS,  /* contains entry */
    CL_BLOCK,     /* data block name */
    CL_PROC,      /* procedure name (subroutine, function, statement func,..) */
    CL_LABEL,     /* label entry */
    CL_FORMAT,    /* format entry */
    CL_TAGNAME,   /* derived type name  */
    CL_NAMELIST,  /* name list name (not implmented) */
    CL_COMMON,    /* common block */
    CL_ELEMENT,   /* structure element name  */
    CL_GENERICS,  /* generics name */
    CL_TYPE_PARAM, /* type parameter name */
    CL_TYPE_BOUND_PROC, /* type bound procedure */
    CL_MULTI,     /* Both the derived type name and the generic procedure */
};

extern char *name_class_names[];
#define NAME_CLASS_NAMES \
{ \
  "CL_UNKNOWN", \
  "CL_PARAM",   \
  "CL_VAR",     \
  "CL_ENTRY",   \
  "CL_MAIN",    \
  "CL_MODULE",  \
  "CL_CONTAINS",\
  "CL_BLOCK",   \
  "CL_PROC",    \
  "CL_LABEL",   \
  "CL_FORMAT",  \
  "CL_TAGNAME", \
  "CL_NAMELIST", \
  "CL_COMMON",  \
  "CL_ELEMENT", \
  "CL_GENERICS", \
  "CL_TYPE_PARAM", \
  "CL_TYPE_BOUND_PROCS", \
  "CL_MULTI",   \
}

/* for CL_PROC  */
enum proc_class {
    P_UNKNOWN,
    P_EXTERNAL,
    P_INTRINSIC,
    P_STFUNCT,
    P_THISPROC,
    P_DEFINEDPROC,
    /* defined_proc is a class for the procedure
       which is defined in the file. */
    P_UNDEFINEDPROC,
    /* unddefined proc is a class for the procedure
       which is not defined, but used as function. */
};

extern char *proc_class_names[];
#define PROC_CLASS_NAMES \
{ \
   "P_UNKNOWN",        \
   "P_EXTERNAL",       \
   "P_INTRINSIC",      \
   "P_STFUNCT",        \
   "P_THISPROC",       \
   "P_DEFINEDPROC",    \
   "P_UNDEFINEDPROC",  \
   "P_EXT_DEFINEDPROC" \
}

/* storage class */
enum storage_class {
    STG_UNKNOWN = 0,
    STG_ARG,    /* dummy argument */
    STG_AUTO,   /* auto variable */
    STG_SAVE,   /* save attr */
    STG_EXT,    /* program, subroutine, function, module, interface */
    STG_COMMON, /* allocated in common */
    STG_EQUIV,  /* allocated in equive */
    STG_COMEQ,  /* allocated in common and equive */
    STG_TAGNAME, /* derived type name  */
    STG_NONE,    /* for intrinsic, stfunction */
    STG_TYPE_PARAM, /* type parameter */
    STG_INDEX, /* indexes of forall */

};

extern char *storage_class_names[];
#define STORAGE_CLASS_NAMES \
{\
 "STG_UNKNOWN",\
 "STG_ARG",     \
 "STG_AUTO",    \
 "STG_SAVE",     \
 "STG_EXT",     \
 "STG_COMMON",  \
 "STG_EQUIV",   \
 "STG_COMEQ",   \
 "STG_TAGNAME", \
 "STG_NONE",    \
 "STG_TYPE_PARAM", \
 "STG_INDEX", \
}

/* statement label for CL_LABEL */
typedef enum statement_label_type { 
    LAB_UNKNOWN = 0,    /* yet unknown type */
    LAB_EXEC,           /* label at exectuable statment */
    LAB_FORMAT          /* label to format statment */
} LABEL_TYPE;

/* FORTRAN identifier structure */
typedef struct ident_descriptor
{
    struct ident_descriptor *next;      /* linked list */
    enum name_class class;              /* name class */
    char is_declared;
    char is_associative;                /* ASSOCIATE and associate-name */
    char could_be_implicitly_typed;     /* id is declared in current scope */
                                        /* and could be typed implicity. */
                                        /* never changed to FALSE from TRUE */
    enum storage_class stg;
    SYMBOL name;                        /* key */
    struct type_descriptor *type;       /* its type */
    expv addr;
    lineno_info *line;                  /* line number this node created */
    struct type_attr attr;              /* type attribute for ID */
    int order;                          /* order in local_symbols */
    struct ident_descriptor *defined_by;/* if this ID is defined parents
                                           then point it, otherwise NULL */
    struct external_symbol *extID;      /* external symbol which is
                                         * represented by this ID */

    struct external_symbol *interfaceId;/* interface id which referes
                                         * this ID. */

    int is_varDeclEmitted;              /* varDecl for this ID is emitted
                                         * or not. */
    struct ident_descriptor *equivID;

    int    use_assoc_conflicted;        /* if TRUE then id ID is
                                           conflicted with use
                                           associated id. */
    struct use_assoc_info *use_assoc;   /* use association infomation
                                           of this ID, otherwise
                                           NULL */
    int from_parent_module;             /* ID is imported from parent module  */

    union {
        struct {
            enum proc_class pclass;   /* for CL_PROC */
            expr args;
            expr result;              /* temporary storage for EXT_ID's
                                       * EXT_PROC_RESULTVAR(). At the
                                       * time of declare_procedure(),
                                       * the return type of the
                                       * procedure might be not
                                       * determined. So the 'expr
                                       * result' must be compiled in
                                       * end_declaration(). */
            int is_recursive;         /* temporary flag.
                                         copy to TYPE_DESC's is_recursive
                                         after type is decided */
            int is_pure;              /* like above. */
            int is_elemental;         /* like above. */
            int is_module;            /* like above. */
            int is_dummy;             /* if TRUE, declared as dummy
                                       * arg in the parent scope. */
            int is_func_subr_ambiguous;
                                      /* if TRUE, the id is still
                                       * ambiguous function or
                                       * subroutine when pclass ==
                                       * P_EXTERNAL. */
            int has_bind;             /* if TRUE, proc uses BIND feature */
            expr bind;                /* temporary storage for bind
                                       * information */
        } proc_info;
        struct {

            /* for STG_COMMON */
            struct ident_descriptor *common_id;

            /* for STG_PTRBASE and pointer variable */
            struct ident_descriptor *ptrPair;

            /* for CL_VAR */
            expv value;                 /* an initial value */
            expv initValues;            /* initial value list */
            expv arrayInfo;             /* If variable is array, not NULL. */
            int initType;
#define VAR_INIT_NEVER          0
#define VAR_INIT_WHOLE          1
#define VAR_INIT_SUBSTR         2
#define VAR_INIT_PARTIAL        3
#define VAR_INIT_EQUIV          4
            int isImpDoDummy;
            expr unCompiledTypeDecl;    /* A type declaration BEFORE
                                         * the type is fixed. */
            int isUnCompiledArray;
            int isUnCompiled;
            int isUsedAsHighOrder;      /* Once used as a function. */

            struct ident_descriptor * ref_proc; /* for a procedure variable, refer to a procedure name */
        } var_info;
        struct {
            LABEL_TYPE lab_type;
            char lab_blklevel;
            char lab_cannot_jmp;        /* can jump or not */
            char lab_is_used;
            int lab_st_no;
        } label_info;
        struct {
            /* for CL_FORMAT */
            expv formatStrV;
        } format_info;
        struct {
            /* for CL_NAMELIST */
            expr list;
        } nl_info;
        struct {
            /* for CL_COMMON */
            expr vars;              /* all members for common block */
            char is_save;           /* save attribute */
            char is_blank_name;     /* blank name */
        } common_info;
        struct {
            /* for CL_TYPE_BOUND_PROCS */
            struct ident_descriptor * binding; /* binding */
            struct ident_descriptor * pass_arg; /* pass argument */
            uint32_t type_bound_attrs;
#define TYPE_BOUND_PROCEDURE_IS_GENERIC            0x0001
#define TYPE_BOUND_PROCEDURE_PASS                  0x0002
#define TYPE_BOUND_PROCEDURE_NOPASS                0x0004
#define TYPE_BOUND_PROCEDURE_NON_OVERRIDABLE       0x0008
#define TYPE_BOUND_PROCEDURE_DEFERRED              0x0010
#define TYPE_BOUND_PROCEDURE_IS_OPERATOR           0x0020
#define TYPE_BOUND_PROCEDURE_IS_ASSIGNMENT         0x0040
#define TYPE_BOUND_PROCEDURE_WRITE                 0x0080
#define TYPE_BOUND_PROCEDURE_READ                  0x0100
#define TYPE_BOUND_PROCEDURE_FORMATTED             0x0200
#define TYPE_BOUND_PROCEDURE_UNFORMATTED           0x0400
        } tbp_info;
        struct {
            /* for CL_MULTI */
            struct ident_descriptor * id_list;
        } multi_info;
    } info;
} *ID;

#define ID_NEXT(id)     ((id)->next)
#define ID_CLASS(id)    ((id)->class)
#define ID_STORAGE(id)  ((id)->stg)
#define ID_SYM(id)      ((id)->name)
#define ID_NAME(id)     SYM_NAME((id)->name)
#define ID_TYPE(id)     ((id)->type)
#define ID_IS_DECLARED(id) ((id)->is_declared)
#define ID_IS_ASSOCIATIVE(id) ((id)->is_associative)
#define ID_COULD_BE_IMPLICITLY_TYPED(id) ((id)->could_be_implicitly_typed)
#define ID_ADDR(id)     ((id)->addr)
#define ID_LINE_NO(x)   ((x)->line->ln_no)
#define ID_END_LINE_NO(id)    ((id)->line->end_ln_no)
#define ID_LINE_FILE_ID(x)    ((x)->line->file_id)
#define ID_LINE(id)     ((id)->line)
#define ID_ORDER(id)    ((id)->order)
#define ID_DEFINED_BY(id)       ((id)->defined_by)

#define ID_INTF(id)     ((id)->interfaceId)

/**
 * use association information about ID.
 */
struct use_assoc_info {
    struct module * module;   /* module. */
    SYMBOL module_name;       /* name of module which the ID declared. */
    SYMBOL original_name;     /* original name of the ID. */
};

#define ID_USEASSOC_INFO(id) ((id)->use_assoc)
#define ID_MODULE(id) ((ID_USEASSOC_INFO(id))->module)
#define ID_MODULE_NAME(id) ((ID_USEASSOC_INFO(id))->module_name)
#define ID_ORIGINAL_NAME(id) ((ID_USEASSOC_INFO(id))->original_name)
#define ID_IS_OFMODULE(id)  ((id)->use_assoc != NULL)
#define ID_IS_AMBIGUOUS(id) ((id)->use_assoc_conflicted)

#define ID_IS_FROM_PARENT_MOD(id) ((id)->from_parent_module)

#define ID_IS_EMITTED(id)   ((id)->is_varDeclEmitted)
#define ID_EQUIV_ID(id)     ((id)->equivID)

#define ID_MAY_HAVE_ACCECIBILITY(id) \
    (ID_STORAGE(id) != STG_ARG && \
     (ID_CLASS(id) == CL_UNKNOWN || \
      ID_CLASS(id) == CL_PARAM || \
      ID_CLASS(id) == CL_VAR || \
      ID_CLASS(id) == CL_ENTRY || \
      ID_CLASS(id) == CL_PROC || \
      ID_CLASS(id) == CL_TAGNAME || \
      ID_CLASS(id) == CL_NAMELIST || \
      ID_CLASS(id) == CL_GENERICS))

#define ID_LINK_ADD(id, list, tail) \
    { if((list) == NULL || (tail) == NULL) (list) = (id); \
      else ID_NEXT(tail) = (id); \
      (tail) = (id); ID_NEXT(id) = NULL; }

#define FOREACH_ID(/* ID */ idp, /* ID */ headp) \
  for ((idp) = (headp); (idp) != NULL ; (idp) = ID_NEXT(idp))

#define SAFE_FOREACH_ID(ip, iq, headp)\
    SAFE_FOREACH(ip, iq, headp, ID_NEXT)

/* for CL_PROC */
#define PROC_CLASS(id)  ((id)->info.proc_info.pclass)
#define PROC_ARGS(id)   ((id)->info.proc_info.args)
#define PROC_RESULTVAR(id)    ((id)->info.proc_info.result)
#define PROC_STBODY(id) ((id)->addr)
#define PROC_IS_RECURSIVE(id) ((id)->info.proc_info.is_recursive)
#define PROC_IS_PURE(id) ((id)->info.proc_info.is_pure)
#define PROC_IS_ELEMENTAL(id) ((id)->info.proc_info.is_elemental)
#define PROC_IS_MODULE(id) ((id)->info.proc_info.is_module)
#define PROC_IS_DUMMY_ARG(id) ((id)->info.proc_info.is_dummy)
#define PROC_IS_FUNC_SUBR_AMBIGUOUS(id) \
    ((id)->info.proc_info.is_func_subr_ambiguous)
#define PROC_HAS_BIND(id) ((id)->info.proc_info.has_bind)
#define PROC_BIND(id)   ((id)->info.proc_info.bind)

#define ID_IS_DUMMY_ARG(id) \
    ((ID_STORAGE((id)) == STG_ARG) || \
     (ID_CLASS((id)) == CL_PROC && PROC_IS_DUMMY_ARG((id)) == TRUE))

#define ID_IS_TYPE_PARAM(id) \
    ((ID_CLASS((id)) == CL_TYPE_PARAM))

/* for CL_VAR */
#define VAR_COM_ID(id)          ((id)->info.var_info.common_id)

#define VAR_POINTER_ID(id)      ((id)->info.var_info.ptrPair)
#define VAR_POINTER_BASE_ID(id) ((id)->info.var_info.ptrPair)

#define VAR_ARRAY_INFO(id)      ((id)->info.var_info.arrayInfo)

#define VAR_INIT_VALUE(id)      ((id)->info.var_info.value)
#define VAR_INIT_LIST(id)       ((id)->info.var_info.initValues)
#define VAR_INIT_TYPE(id)       ((id)->info.var_info.initType)
#define VAR_IS_IMPLIED_DO_DUMMY(id)     ((id)->info.var_info.isImpDoDummy)
#define VAR_UNCOMPILED_DECL(id) ((id)->info.var_info.unCompiledTypeDecl)
#define VAR_IS_UNCOMPILED(id)   ((id)->info.var_info.isUnCompiled)
#define VAR_IS_UNCOMPILED_ARRAY(id)     ((id)->info.var_info.isUnCompiledArray)
#define VAR_IS_USED_AS_FUNCTION(id)     ((id)->info.var_info.isUsedAsHighOrder)
#define VAR_REF_PROC(id)        ((id)->info.var_info.ref_proc)

/* for CL_PROC/CL_VAR */
#define PROC_EXT_ID(id) ((id)->extID)

/* for CL_LABEL */
#define LAB_TYPE(l)     ((l)->info.label_info.lab_type)
#define LAB_ST_NO(l)    ((l)->info.label_info.lab_st_no)
#define LAB_BLK_LEVEL(l)        ((l)->info.label_info.lab_blklevel)
#define LAB_IS_USED(l)  ((l)->info.label_info.lab_is_used)
#define LAB_CANNOT_JUMP(l)      ((l)->info.label_info.lab_cannot_jmp)
#define LAB_IS_DEFINED(l)       ((l)->is_declared)

/* for CL_FORMAT */
#define FORMAT_STR(id)  ((id)->info.format_info.formatStrV)
/* for CL_NAMELIST */
#define NL_LIST(id)     ((id)->info.nl_info.list)

/* for CL_COMMON */
#define COM_VARS(id)            ((id)->info.common_info.vars)
#define COM_IS_SAVE(id)         ((id)->info.common_info.is_save)
#define COM_IS_BLANK_NAME(id)   ((id)->info.common_info.is_blank_name)

/* for CL_TYPE_BOUND_PROCS */
#define TBP_BINDING(id)         ((id)->info.tbp_info.binding)
#define TBP_BINDING_ATTRS(id)   ((id)->info.tbp_info.type_bound_attrs)
#define TBP_PASS_ARG(id)        ((id)->info.tbp_info.pass_arg)

#define TBP_IS_OPERATOR(id) \
    (ID_CLASS(id) == CL_TYPE_BOUND_PROC && \
     TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_IS_OPERATOR)

#define TBP_IS_ASSIGNMENT(id) \
    (ID_CLASS(id) == CL_TYPE_BOUND_PROC && \
     TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_IS_ASSIGNMENT)

#define TBP_IS_DEFINED_IO(id) \
    (ID_CLASS(id) == CL_TYPE_BOUND_PROC && (                    \
        TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_WRITE ||   \
        TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_READ))


#define MULTI_ID_LIST(id)     ((id)->info.multi_info.id_list)


struct interface_info {
    enum {
        INTF_ASSIGNMENT,  /* for assignment interface */
        INTF_OPERATOR,    /* for operator (override) interface */
        INTF_USEROP,      /* for user defined operator interface */
        INTF_GENERICS,    /* for generics interface but not yet defined */
        INTF_GENERIC_FUNC,/* for generic 'functions' interface */
        INTF_GENERIC_SUBR,/* for generic 'subroutines' interface */
        INTF_GENERIC_WRITE_FORMATTED,/* for generic 'WRITE(FORMATTED)' interface */
        INTF_GENERIC_WRITE_UNFORMATTED,/* for generic 'WRITE(UNFORMATTED)' interface */
        INTF_GENERIC_READ_FORMATTED,/* for generic 'READ(FORMATTED)' interface */
        INTF_GENERIC_READ_UNFORMATTED,/* for generic 'READ(UNFORMATTED)' interface */
        INTF_DECL         /* for interface not above cases. (interface for function prottype)*/
    } class;
};

enum ext_proc_class {
    EP_UNKNOWN,
    EP_PROGRAM,
    EP_PROC,
    EP_ENTRY,
    EP_INTERFACE,
    EP_INTERFACE_DEF,
    EP_MODULE_PROCEDURE,
    EP_INTRINSIC
};

/* external symbol */
typedef struct external_symbol
{
    struct external_symbol *next;
    SYMBOL name;                /* key */
    enum storage_class stg;     /* STG_UNKNOWN, STG_EXT, STG_COMMON, STG_LIB */
    char is_defined;            /* defined or not */
    lineno_info *line;          /* line number this node created */
    char isHighOrderDummy;      /* Dummy sub program entry for high order. */
    char is_blank_name;         /* name is blank or dummy */
    int is_ofModule;            /* If TRUE(1) indicates this ID is
                                 * declared in USE inclusion. */
    union {
        struct {
            TYPE_DESC type;     /* type in Fortran */
            expr args;
            expv result;        /* result var for functions/entries */
            expv body;          /* body of defined procedure */
            ID id_list;
            ID label_list;
            ID common_id_list;  /* common block ids */
            TYPE_DESC struct_decls; /* derived types in Fortran90 */
            lineno_info *contains_line;
            struct block_env * blocks; /* block constructs */
            struct external_symbol *contains_external_symbols;
            struct external_symbol *interface_external_symbols;
            struct interface_info * interface_info;
            struct external_symbol *intr_def_external_symbols;
            enum ext_proc_class ext_proc_class;
            char is_output;
            char is_module_specified; /* module procedure with keyword 'module' */
            char is_internal; /* internal procedure */
            int is_procedureDecl; /* procedure declaration */

            struct { /* For submodule */
                int is_submodule;
                SYMBOL ancestor;
                SYMBOL parent;
            } extends;
        } proc_info;
    } info;
} *EXT_ID;

#define EXT_NEXT(ep)    ((ep)->next)
#define EXT_SYM(ep)     ((ep)->name)
#define EXT_TAG(ep)     ((ep)->stg)
#define EXT_IS_DEFINED(ep)      ((ep)->is_defined)
#define EXT_LINE_NO(ep)   ((ep)->line->ln_no)
#define EXT_END_LINE_NO(ep)     ((ep)->line->end_ln_no)
#define EXT_LINE_FILE_ID(ep)    ((ep)->line->file_id)
#define EXT_LINE(ep)     ((ep)->line)
#define EXT_IS_DUMMY(ep) ((ep)->isHighOrderDummy)
#define EXT_IS_BLANK_NAME(ep) ((ep)->is_blank_name)
#define EXT_IS_OFMODULE(ep)  ((ep)->is_ofModule)

#define EXT_PROC_TYPE(ep)       ((ep)->info.proc_info.type)
#define EXT_PROC_BODY(ep)       ((ep)->info.proc_info.body)
#define EXT_PROC_ARGS(ep)       ((ep)->info.proc_info.args)
#define EXT_PROC_RESULTVAR(ep)  ((ep)->info.proc_info.result)
#define EXT_PROC_ID_LIST(ep)    ((ep)->info.proc_info.id_list)
#define EXT_PROC_LABEL_LIST(ep) ((ep)->info.proc_info.label_list)
#define EXT_PROC_STRUCT_DECLS(ep) ((ep)->info.proc_info.struct_decls)
#define EXT_PROC_BLOCKS(ep)    ((ep)->info.proc_info.blocks)
#define EXT_PROC_CONT_EXT_SYMS(ep) ((ep)->info.proc_info.contains_external_symbols)
#define EXT_PROC_CONT_EXT_LINE(ep) ((ep)->info.proc_info.contains_line)
#define EXT_PROC_INTERFACES(ep) ((ep)->info.proc_info.interface_external_symbols)
#define EXT_PROC_INTERFACE_INFO(ep) ((ep)->info.proc_info.interface_info)
#define EXT_PROC_INTERFACE_CLASS(ep) ((ep)->info.proc_info.interface_info->class)
#define EXT_PROC_INTR_DEF_EXT_IDS(ep) \
                                ((ep)->info.proc_info.intr_def_external_symbols)
#define EXT_PROC_CLASS(ep)      ((ep)->info.proc_info.ext_proc_class)
#define EXT_PROC_IS_PROGRAM(ep) (EXT_PROC_CLASS(ep) == EP_PROGRAM)
#define EXT_PROC_IS_ENTRY(ep)   (EXT_PROC_CLASS(ep) == EP_ENTRY)
#define EXT_PROC_IS_INTERFACE(ep)   (EXT_PROC_CLASS(ep) == EP_INTERFACE)
#define EXT_PROC_IS_INTERFACE_DEF(ep) \
                                (EXT_PROC_CLASS(ep) == EP_INTERFACE_DEF)

#define EXT_PROC_IS_MODULE_PROCEDURE(ep) \
                                (EXT_PROC_CLASS(ep) == EP_MODULE_PROCEDURE)
#define EXT_PROC_IS_MODULE_SPECIFIED(ep) \
                                ((ep)->info.proc_info.is_module_specified)
#define EXT_PROC_IS_INTRINSIC(ep)   (EXT_PROC_CLASS(ep) == EP_INTRINSIC)
#define EXT_PROC_COMMON_ID_LIST(ep) ((ep)->info.proc_info.common_id_list)
#define EXT_PROC_IS_OUTPUT(ep)  ((ep)->info.proc_info.is_output)
#define EXT_PROC_IS_INTERNAL(ep)  ((ep)->info.proc_info.is_internal)
#define EXT_PROC_IS_PROCEDUREDECL(ep)  ((ep)->info.proc_info.is_procedureDecl)

#define EXT_MODULE_IS_SUBMODULE(ep) ((ep)->info.proc_info.extends.is_submodule)
#define EXT_MODULE_PARENT(ep) ((ep)->info.proc_info.extends.parent)
#define EXT_MODULE_ANCESTOR(ep) ((ep)->info.proc_info.extends.ancestor)

#define FOREACH_EXT_ID(/* EXT_ID */ ep, /* EXT_ID */ headp) \
  for ((ep) = (headp); (ep) != NULL ; (ep) = EXT_NEXT(ep))

#define SAFE_FOREACH_EXT_ID(ep, eq, headp)\
    SAFE_FOREACH(ep, eq, headp, EXT_NEXT)

#define EXT_LINK_ADD(ep, list, tail) \
    { if((list) == NULL || (tail) == NULL) (list) = (ep); \
      else EXT_NEXT(tail) = (ep); \
      (tail) = (ep); }

#define BLANK_COMMON_NAME       "_____BLANK_COMMON_____"

typedef struct block_env
{
    struct block_env *next;
    struct block_env *blocks;
    ID id_list;
    ID label_list;
    TYPE_DESC struct_decls; /* derived types in Fortran90 */
    struct external_symbol *interfaces;
    struct external_symbol *external_symbols;
} *BLOCK_ENV;

#define BLOCK_NEXT(bp) ((bp)->next)
#define BLOCK_CHILDREN(bp) ((bp)->blocks)
#define BLOCK_LOCAL_LABELS(bp) ((bp)->label_list)
#define BLOCK_LOCAL_SYMBOLS(bp) ((bp)->id_list)
#define BLOCK_LOCAL_STRUCT_DECLS(bp) ((bp)->struct_decls)
#define BLOCK_LOCAL_INTERFACES(bp) ((bp)->interfaces)
#define BLOCK_LOCAL_EXTERNAL_SYMBOLS(bp) ((bp)->external_symbols)

#define FOREACH_BLOCKS(bp, headp) \
    for ((bp) = (headp); (bp) != NULL ; (bp) = BLOCK_NEXT(bp))

#define BLOCK_LINK_ADD(bp, list, tail) \
    { if((list) == NULL || (tail) == NULL) (list) = (bp); \
      else BLOCK_NEXT(tail) = (bp); \
      (tail) = (bp); }


#endif /* _F_IDENT_H_ */

