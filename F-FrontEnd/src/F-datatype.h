/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-datatype.h
 */

#ifndef _F_DATATYPE_H_
#define _F_DATATYPE_H_

#include <stdint.h>

/* f77 data types */
typedef enum datatype {
    TYPE_UNKNOWN=0,     /* 0 undef or pointer */
    TYPE_INT,           /* 1 */
    TYPE_REAL,          /* 2 */
    TYPE_DREAL,         /* 3 */
    TYPE_COMPLEX,       /* 4 */
    TYPE_DCOMPLEX,      /* 5 */
    TYPE_LOGICAL,       /* 6 */
    TYPE_CHAR,          /* 7 */
    TYPE_SUBR,          /* 8 */
    TYPE_ARRAY,         /* 9 extended */
    TYPE_FUNCTION,      /* 10 function */
    TYPE_STRUCT,        /* 11 derived type */
    TYPE_GNUMERIC,      /* 12 general numeric (integer or real) */
    TYPE_GNUMERIC_ALL,  /* 13 general all numeric (integer or real or
                         * complex) */
    TYPE_MODULE,        /* 14 module */
    TYPE_GENERIC,       /* 15 generic type for interface */
    TYPE_NAMELIST,      /* 16 type for namelist */
    TYPE_LHS,           /* 17 type for intrinsic null(), always
                         * comforms to the type of the left hand
                         * expression. */
    TYPE_VOID,          /* 18 type of subroutine call */
    TYPE_END
} BASIC_DATA_TYPE;

typedef enum array_assume_kind {
    ASSUMED_NONE,
    ASSUMED_SIZE,
    ASSUMED_SHAPE
} ARRAY_ASSUME_KIND;

#define N_BASIC_TYPES ((int)TYPE_END)

#define KIND_PARAM_DOUBLE   8

extern char *basic_type_names[];
#define BASIC_TYPE_NAMES \
{ \
 "*undef*",     \
 "integer",     \
 "real",        \
 "double_real", \
 "complex",     \
 "double_complex",\
 "logical",     \
 "character",   \
 "subroutine",  \
 "*array*",     \
 "*function*",\
 "*type*", \
 "*numeric*", \
 "*numeric_all*", \
 "module", \
 "*generic*", \
 "*namelist*", \
 "*comforms_to_the_lefthand*", \
}

typedef struct _codims_desc {
  int corank;
  expr cobound_list;
} codims_desc;


/* FORTRAN 77 type descriptor */
/* FORTRAN 77 does not have nested data structure */
/* pointer type, TYPE_UNKNOWN and ref != NULL */
/* array type, TYPE_ARRAY && ref != NULL */
/* but, Fortran 90 has nested data structure */
typedef struct type_descriptor
{
    struct type_descriptor *link;        /* global linked list */
    struct type_descriptor *struct_link; /* struct linked list */
    BASIC_DATA_TYPE basic_type;
    struct type_descriptor *ref;         /* reference to other */
    struct ident_descriptor *tagname;    /* derived type tagname */
    char is_referenced;
    expv kind;                 /* kind parameter */
    expv leng;                 /* len parameter */
    int size;                  /* for TYPE_CHAR char length */
    int is_declared;           /* boolean for type has declared.
                                  (only used by struct type) */
    int is_modified;           /* modified with VOLATILE or ASYNCHRONOUS */
    expv bind_name;            /* ISO BIND C name attribute */

    struct type_attr {
#define TYPE_ATTR_PARAMETER         0x00000001
#define TYPE_ATTR_ALLOCATABLE       0x00000002
#define TYPE_ATTR_EXTERNAL          0x00000004
#define TYPE_ATTR_INTRINSIC         0x00000008
#define TYPE_ATTR_OPTIONAL          0x00000010
#define TYPE_ATTR_POINTER           0x00000020
#define TYPE_ATTR_SAVE              0x00000040
#define TYPE_ATTR_TARGET            0x00000080
#define TYPE_ATTR_PUBLIC            0x00000100
#define TYPE_ATTR_PRIVATE           0x00000200
#define TYPE_ATTR_INTENT_IN         0x00000400
#define TYPE_ATTR_INTENT_OUT        0x00000800
#define TYPE_ATTR_INTENT_INOUT      0x00001000
#define TYPE_ATTR_SEQUENCE          0x00002000
#define TYPE_ATTR_INTERNAL_PRIVATE  0x00004000
#define TYPE_ATTR_RECURSIVE         0x00008000
#define TYPE_ATTR_PURE              0x00010000
#define TYPE_ATTR_ELEMENTAL         0x00020000
#define TYPE_ATTR_PROTECTED         0x00040000
#define TYPE_ATTR_VOLATILE          0x00080000
#define TYPE_ATTR_KIND              0x00100000
#define TYPE_ATTR_LEN               0x00200000
#define TYPE_ATTR_CLASS             0x00400000
#define TYPE_ATTR_BIND              0x00800000
#define TYPE_ATTR_VALUE             0x01000000
#define TYPE_ATTR_MODULE            0x02000000 /* for module function/subroutine */
#define TYPE_ATTR_PROCEDURE         0x04000000 /* for procedure variables */
        uint32_t type_attr_flags;
#define TYPE_EXFLAGS_IMPLICIT       0x00000001 /* implicitly defined or not */
#define TYPE_EXFLAGS_OVERRIDDEN     0x00000002 /* type is overridden by child */
#define TYPE_EXFLAGS_USED_EXPLICIT  0x00000004 /* type is used explicitly (used for function/subroutine call) */
#define TYPE_EXFLAGS_NOT_FIXED      0x00000008 /* type is not fixed, since expression
                                                  contains undefined function. */
#define TYPE_EXFLAGS_FOR_FUNC_SELF  0x00000010 /* type is for the function itself */
#define TYPE_EXFLAGS_UNCHANGABLE    0x00000020 /* type is not able to change */
        uint32_t exflags;
    } attr; /* FbasicType */
    struct {
        char n_dim;            /* dimension (max 15) */
        char dim_fixed;        /* fixed or not */
        char dim_fixing;
        ARRAY_ASSUME_KIND assume_kind; /* represents assumed size or shape */
        expv dim_size;
        expv dim_upper, dim_lower, dim_step; /* dimension subscripts */
    } array_info; /* FOR FbasicType for Array */
    struct ident_descriptor *parent;  /* represents super-class of this derived type.  */
    struct ident_descriptor *type_parameters; /* type parameters for derived type */
                                              /* For the parameterized derived-type, it works as dummy arguments */
                                              /* For the instance of the parameterized derived-type, it works as actual arguments */
    expv type_param_values; /* type parameter values */
    /* struct ident_descriptor *type_parameters_used; /\* TODO: write nice comment *\/ */
    struct ident_descriptor *members; /* all members for derived type */
    codims_desc *codims;
    int is_reshaped_type;       /* A bool flag to specify this type is
                                 * genereted by reshape() intrinsic. */

    struct {
        struct type_descriptor * return_type;
        SYMBOL result;
        int has_explicit_arguments;
        struct ident_descriptor * args;
        int is_program;                 /* for the type of the program */
        int is_generic;                 /* for the type of generic function/subroutine */
        int is_tbp;                     /* function/subroutine is type-bound procedure */
        int is_defined;                 /* function/subroutine has a definition (For submodule only) */
        int is_internal;                /* for internal subprograms (function/subroutine in the contain block)*/
        int is_module_procedure;        /* used as a module procedure */ /* may not be required */
        int is_visible_intrinsic;       /* TRUE if non standard intrinsic */

        int has_binding_arg;
        int has_pass_arg;                   /* for the function type of procedure variable OR type-bound procedure */
        struct ident_descriptor * pass_arg; /* for the function type of procedure variable OR type-bound procedure */
        struct type_descriptor * pass_arg_type; /* for the function type of procedure variable OR type-bound procedure */

        struct {
            struct ident_descriptor * generics; /* for the function type of type-bound generic */
        } type_bound_proc_info;
    } proc_info;

} *TYPE_DESC;

struct type_attr_check {
    uint32_t flag;
    uint32_t acceptable_flags;
    char *flag_name;
};

extern struct type_attr_check type_attr_checker[];

extern TYPE_DESC basic_type_desc[];
#define BASIC_TYPE_DESC(t) basic_type_desc[(int)t]

#define TYPE_LINK(tp)           ((tp)->link)
#define TYPE_SLINK(tp)          ((tp)->struct_link)
#define TYPE_IS_DECLARED(tp)    ((tp)->is_declared)
#define TYPE_IS_COINDEXED(tp)   (tp != NULL && (tp)->codims)
#define TYPE_BASIC_TYPE(tp)     ((tp)->basic_type)
#define TYPE_REF(tp)            ((tp)->ref)
#define TYPE_TAGNAME(tp)        ((tp)->tagname)
#define TYPE_IS_REFERENCED(tp)  ((tp)->is_referenced)
#define TYPE_CODIMENSION(tp)    ((tp)->codims)
#define TYPE_LINK_ADD(tp, tlist, ttail) \
    { if((tlist) == NULL) (tlist) = (tp); \
      ttail = type_link_add(tp, tlist, ttail);  \
    }
#define TYPE_SLINK_ADD(tp, tlist, ttail) \
    { if((tlist) == NULL) (tlist) = (tp); \
      else TYPE_SLINK(ttail) = (tp); \
      (ttail) = (tp); }

/* F95 type attribute macros */
#define TYPE_ATTR_FLAGS(tp)         ((tp)->attr.type_attr_flags)

#define TYPE_IS_PARAMETER(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_PARAMETER)
#define TYPE_SET_PARAMETER(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_PARAMETER)
#define TYPE_UNSET_PARAMETER(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_PARAMETER)
#define TYPE_IS_ALLOCATABLE(tp)     ((tp)->attr.type_attr_flags &   TYPE_ATTR_ALLOCATABLE)
#define TYPE_SET_ALLOCATABLE(tp)    ((tp)->attr.type_attr_flags |=  TYPE_ATTR_ALLOCATABLE)
#define TYPE_UNSET_ALLOCATABLE(tp)  ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_ALLOCATABLE)
#define TYPE_IS_EXTERNAL(tp)        ((tp)->attr.type_attr_flags &   TYPE_ATTR_EXTERNAL)
#define TYPE_SET_EXTERNAL(tp)       ((tp)->attr.type_attr_flags |=  TYPE_ATTR_EXTERNAL)
#define TYPE_UNSET_EXTERNAL(tp)     ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_EXTERNAL)
#define TYPE_IS_INTRINSIC(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_INTRINSIC)
#define TYPE_SET_INTRINSIC(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_INTRINSIC)
#define TYPE_UNSET_INTRINSIC(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_INTRINSIC)
#define TYPE_IS_OPTIONAL(tp)        ((tp)->attr.type_attr_flags &   TYPE_ATTR_OPTIONAL)
#define TYPE_SET_OPTIONAL(tp)       ((tp)->attr.type_attr_flags |=  TYPE_ATTR_OPTIONAL)
#define TYPE_UNSET_OPTIONAL(tp)     ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_OPTIONAL)
#define TYPE_IS_POINTER(tp)         ((tp)->attr.type_attr_flags &   TYPE_ATTR_POINTER)
#define TYPE_SET_POINTER(tp)        ((tp)->attr.type_attr_flags |=  TYPE_ATTR_POINTER)
#define TYPE_UNSET_POINTER(tp)      ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_POINTER)
#define TYPE_IS_SAVE(tp)            ((tp)->attr.type_attr_flags &   TYPE_ATTR_SAVE)
#define TYPE_SET_SAVE(tp)           ((tp)->attr.type_attr_flags |=  TYPE_ATTR_SAVE)
#define TYPE_UNSET_SAVE(tp)         ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_SAVE)
#define TYPE_IS_TARGET(tp)          ((tp)->attr.type_attr_flags &   TYPE_ATTR_TARGET)
#define TYPE_SET_TARGET(tp)         ((tp)->attr.type_attr_flags |=  TYPE_ATTR_TARGET)
#define TYPE_UNSET_TARGET(tp)       ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_TARGET)
#define TYPE_IS_PUBLIC(tp)          ((tp)->attr.type_attr_flags &   TYPE_ATTR_PUBLIC)
#define TYPE_SET_PUBLIC(tp)         ((tp)->attr.type_attr_flags |=  TYPE_ATTR_PUBLIC)
#define TYPE_UNSET_PUBLIC(tp)       ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_PUBLIC)
#define TYPE_IS_PRIVATE(tp)         ((tp)->attr.type_attr_flags &   TYPE_ATTR_PRIVATE)
#define TYPE_SET_PRIVATE(tp)        ((tp)->attr.type_attr_flags |=  TYPE_ATTR_PRIVATE)
#define TYPE_UNSET_PRIVATE(tp)      ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_PRIVATE)
#define TYPE_IS_PROTECTED(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_PROTECTED)
#define TYPE_SET_PROTECTED(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_PROTECTED)
#define TYPE_UNSET_PROTECTED(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_PROTECTED)
#define TYPE_IS_SEQUENCE(tp)        ((tp)->attr.type_attr_flags &   TYPE_ATTR_SEQUENCE)
#define TYPE_SET_SEQUENCE(tp)       ((tp)->attr.type_attr_flags |=  TYPE_ATTR_SEQUENCE)
#define TYPE_UNSET_SEQUENCE(tp)     ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_SEQUENCE)
#define TYPE_IS_INTERNAL_PRIVATE(tp) ((tp)->attr.type_attr_flags &  TYPE_ATTR_INTERNAL_PRIVATE)
#define TYPE_SET_INTERNAL_PRIVATE(tp)      ((tp)->attr.type_attr_flags |= TYPE_ATTR_INTERNAL_PRIVATE)
#define TYPE_UNSET_INTERNAL_PRIVATE(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_INTERNAL_PRIVATE)
#define TYPE_IS_RECURSIVE(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_RECURSIVE)
#define TYPE_SET_RECURSIVE(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_RECURSIVE)
#define TYPE_UNSET_RECURSIVE(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_RECURSIVE)
#define TYPE_IS_PURE(tp)            ((tp)->attr.type_attr_flags &   TYPE_ATTR_PURE)
#define TYPE_SET_PURE(tp)           ((tp)->attr.type_attr_flags |=  TYPE_ATTR_PURE)
#define TYPE_UNSET_PURE(tp)         ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_PURE)
#define TYPE_IS_ELEMENTAL(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_ELEMENTAL)
#define TYPE_SET_ELEMENTAL(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_ELEMENTAL)
#define TYPE_UNSET_ELEMENTAL(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_ELEMENTAL)
#define TYPE_IS_MODULE(tp)          ((tp)->attr.type_attr_flags &   TYPE_ATTR_MODULE)
#define TYPE_SET_MODULE(tp)         ((tp)->attr.type_attr_flags |=  TYPE_ATTR_MODULE)
#define TYPE_UNSET_MODULE(tp)       ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_MODULE)
#define TYPE_IS_PROCEDURE(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_PROCEDURE)
#define TYPE_SET_PROCEDURE(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_PROCEDURE)
#define TYPE_UNSET_PROCEDURE(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_PROCEDURE)
#define TYPE_IS_INTENT_IN(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_INTENT_IN)
#define TYPE_SET_INTENT_IN(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_INTENT_IN)
#define TYPE_UNSET_INTENT_IN(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_INTENT_IN)
#define TYPE_IS_INTENT_OUT(tp)      ((tp)->attr.type_attr_flags &   TYPE_ATTR_INTENT_OUT)
#define TYPE_SET_INTENT_OUT(tp)     ((tp)->attr.type_attr_flags |=  TYPE_ATTR_INTENT_OUT)
#define TYPE_UNSET_INTENT_OUT(tp)   ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_INTENT_OUT)
#define TYPE_IS_INTENT_INOUT(tp)    ((tp)->attr.type_attr_flags &   TYPE_ATTR_INTENT_INOUT)
#define TYPE_SET_INTENT_INOUT(tp)   ((tp)->attr.type_attr_flags |=  TYPE_ATTR_INTENT_INOUT)
#define TYPE_UNSET_INTENT_INOUT(tp) ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_INTENT_INOUT)
#define TYPE_IS_VOLATILE(tp)        ((tp)->attr.type_attr_flags &   TYPE_ATTR_VOLATILE)
#define TYPE_SET_VOLATILE(tp)       ((tp)->attr.type_attr_flags |=  TYPE_ATTR_VOLATILE)
#define TYPE_UNSET_VOLATILE(tp)     ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_VOLATILE)
#define TYPE_IS_KIND(tp)            ((tp)->attr.type_attr_flags &   TYPE_ATTR_KIND)
#define TYPE_SET_KIND(tp)           ((tp)->attr.type_attr_flags |=  TYPE_ATTR_KIND)
#define TYPE_UNSET_KIND(tp)         ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_KIND)
#define TYPE_IS_LEN(tp)             ((tp)->attr.type_attr_flags &   TYPE_ATTR_LEN)
#define TYPE_SET_LEN(tp)            ((tp)->attr.type_attr_flags |=  TYPE_ATTR_LEN)
#define TYPE_UNSET_LEN(tp)          ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_LEN)
#define TYPE_IS_CLASS(tp)           ((tp)->attr.type_attr_flags &   TYPE_ATTR_CLASS)
#define TYPE_SET_CLASS(tp)          ((tp)->attr.type_attr_flags |=  TYPE_ATTR_CLASS)
#define TYPE_UNSET_CLASS(tp)        ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_CLASS)
#define TYPE_HAS_BIND(tp)           ((tp)->attr.type_attr_flags &   TYPE_ATTR_BIND)
#define TYPE_SET_BIND(tp)           ((tp)->attr.type_attr_flags |=  TYPE_ATTR_BIND)
#define TYPE_UNSET_BIND(tp)         ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_BIND)
#define TYPE_IS_VALUE(tp)           ((tp)->attr.type_attr_flags &   TYPE_ATTR_VALUE)
#define TYPE_SET_VALUE(tp)          ((tp)->attr.type_attr_flags |=  TYPE_ATTR_VALUE)
#define TYPE_UNSET_VALUE(tp)        ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_VALUE)

#define TYPE_EXTATTR_FLAGS(tp)      ((tp)->attr.exflags)
#define TYPE_IS_IMPLICIT(tp)        ((tp)->attr.exflags &   TYPE_EXFLAGS_IMPLICIT)
#define TYPE_SET_IMPLICIT(tp)       ((tp)->attr.exflags |=  TYPE_EXFLAGS_IMPLICIT)
#define TYPE_UNSET_IMPLICIT(tp)     ((tp)->attr.exflags &= ~TYPE_EXFLAGS_IMPLICIT)
#define TYPE_IS_EXPLICIT(tp)        (!TYPE_IS_IMPLICIT(tp))
#define TYPE_IS_OVERRIDDEN(tp)      ((tp)->attr.exflags &   TYPE_EXFLAGS_OVERRIDDEN)
#define TYPE_SET_OVERRIDDEN(tp)     ((tp)->attr.exflags |=  TYPE_EXFLAGS_OVERRIDDEN)
#define TYPE_UNSET_OVERRIDDEN(tp)   ((tp)->attr.exflags &= ~TYPE_EXFLAGS_OVERRIDDEN)
#define TYPE_IS_USED_EXPLICIT(tp)        ((tp)->attr.exflags &   TYPE_EXFLAGS_USED_EXPLICIT)
#define TYPE_SET_USED_EXPLICIT(tp)       ((tp)->attr.exflags |=  TYPE_EXFLAGS_USED_EXPLICIT)
#define TYPE_UNSET_USED_EXPLICIT(tp)     ((tp)->attr.exflags &= ~TYPE_EXFLAGS_USED_EXPLICIT)
#define TYPE_IS_NOT_FIXED(tp)       ((tp)->attr.exflags &   TYPE_EXFLAGS_NOT_FIXED)
#define TYPE_IS_FIXED(tp)           (!TYPE_IS_NOT_FIXED(tp))
#define TYPE_SET_NOT_FIXED(tp)      ((tp)->attr.exflags |=  TYPE_EXFLAGS_NOT_FIXED)
#define TYPE_UNSET_NOT_FIXED(tp)    ((tp)->attr.exflags &= ~TYPE_EXFLAGS_NOT_FIXED)

#define TYPE_IS_FOR_FUNC_SELF(tp)   ((tp)->attr.exflags &   TYPE_EXFLAGS_FOR_FUNC_SELF)
#define TYPE_SET_FOR_FUNC_SELF(tp)  ((tp)->attr.exflags |=  TYPE_EXFLAGS_FOR_FUNC_SELF)
#define TYPE_UNSET_FOR_FUNC_SELF(tp) ((tp)->attr.exflags &= ~TYPE_EXFLAGS_FOR_FUNC_SELF)

#define TYPE_IS_UNCHANGABLE(tp)   ((tp)->attr.exflags &   TYPE_EXFLAGS_UNCHANGABLE)
#define TYPE_SET_UNCHANGABLE(tp)  ((tp)->attr.exflags |=  TYPE_EXFLAGS_UNCHANGABLE)
#define TYPE_UNSET_UNCHANGABLE(tp) ((tp)->attr.exflags &= ~TYPE_EXFLAGS_UNCHANGABLE)

#define TYPE_ATTR_FOR_COMPARE \
    (TYPE_ATTR_PARAMETER |                      \
     TYPE_ATTR_ALLOCATABLE |                    \
     TYPE_ATTR_EXTERNAL |                       \
     TYPE_ATTR_INTRINSIC |                      \
     TYPE_ATTR_OPTIONAL |                       \
     TYPE_ATTR_POINTER |                        \
     TYPE_ATTR_TARGET |                         \
     TYPE_ATTR_INTENT_IN |                      \
     TYPE_ATTR_INTENT_OUT |                     \
     TYPE_ATTR_INTENT_INOUT |                   \
     TYPE_ATTR_SEQUENCE |                       \
     TYPE_ATTR_VOLATILE |                       \
     TYPE_ATTR_CLASS |                          \
     TYPE_ATTR_BIND |                           \
     TYPE_ATTR_VALUE)


#define TYPE_HAS_INTENT(tp)      (TYPE_IS_INTENT_IN(tp) || \
                TYPE_IS_INTENT_OUT(tp) || TYPE_IS_INTENT_INOUT(tp))
// TODO PROTECTED 
#define IS_TYPE_PUBLICORPRIVATE(tp)  \
                ((TYPE_IS_PUBLIC(tp)) || (TYPE_IS_PRIVATE(tp)))

#define TYPE_HAS_ACCESSIBILITY_FLAGS(tp) \
         ((TYPE_ATTR_FLAGS(tp) & TYPE_ATTR_PRIVATE) || \
          (TYPE_ATTR_FLAGS(tp) & TYPE_ATTR_PUBLIC)  || \
          (TYPE_ATTR_FLAGS(tp) & TYPE_ATTR_PROTECTED))

#define TYPE_HAS_NON_ACCESSIBILITY_FLAGS(tp) \
         (TYPE_ATTR_FLAGS(tp)  != 0 &&  \
          (TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_PRIVATE) && \
          (TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_PUBLIC)  && \
          (TYPE_ATTR_FLAGS(tp) != TYPE_ATTR_PROTECTED))


#define TYPE_N_DIM(tp)          ((tp)->array_info.n_dim)
#define TYPE_DIM_FIXED(tp)      ((tp)->array_info.dim_fixed)
#define TYPE_DIM_FIXING(tp)     ((tp)->array_info.dim_fixing)
#define TYPE_DIM_SIZE(tp)       ((tp)->array_info.dim_size)
#define TYPE_DIM_UPPER(tp)      ((tp)->array_info.dim_upper)
#define TYPE_DIM_LOWER(tp)      ((tp)->array_info.dim_lower)
#define TYPE_DIM_STEP(tp)       ((tp)->array_info.dim_step)
#define TYPE_IS_SCALAR(tp)      (((tp)->array_info.n_dim == 0))
#define TYPE_MEMBER_LIST(tp)    ((tp)->members)
#define TYPE_TYPE_PARAMS(tp)    ((tp)->type_parameters)
#define TYPE_TYPE_ACTUAL_PARAMS(tp)    ((tp)->type_parameters)
#define TYPE_TYPE_PARAM_VALUES(tp)    ((tp)->type_param_values)
#define TYPE_HAS_TYPE_PARAMS(tp) (((tp)->type_parameters) != NULL)

#define TYPE_CHAR_LEN(tp)       ((tp)->size)
#define TYPE_KIND(tp)           ((tp)->kind)
#define TYPE_LENG(tp)           ((tp)->leng)
#define TYPE_PARENT(tp)         ((tp)->parent)
#define TYPE_PARENT_TYPE(tp)    (TYPE_PARENT(tp)->type)
#define TYPE_BIND_NAME(tp)      ((tp)->bind_name)

#define TYPE_ARRAY_ASSUME_KIND(tp) ((tp)->array_info.assume_kind)

#define TYPE_IS_RESHAPED(tp)            ((tp)->is_reshaped_type)

#define TYPE_IS_ARRAY_ASSUMED_SIZE(tp) \
                (TYPE_ARRAY_ASSUME_KIND(tp) == ASSUMED_SIZE)
#define TYPE_IS_ARRAY_ASSUMED_SHAPE(tp) \
                (TYPE_ARRAY_ASSUME_KIND(tp) == ASSUMED_SHAPE)
#define TYPE_IS_ARRAY_ADJUSTABLE(tp) \
                (TYPE_IS_ARRAY_ASSUMED_SIZE(tp) || TYPE_IS_ARRAY_ASSUMED_SHAPE(tp))

#define TYPE_IS_RESHAPED_ARRAY(tp)

#define TYPE_HAVE_KIND(tp) \
                ((tp) != NULL && TYPE_KIND(tp) != NULL)

#define CHAR_LEN_UNFIXED (-1)

#define CHAR_LEN_ALLOCATABLE (-2)

/* macros distinguishing type */
#define IS_STRUCT_TYPE(tp) \
                ((tp) != NULL && TYPE_BASIC_TYPE(tp) == TYPE_STRUCT)
#define IS_ARRAY_TYPE(tp) \
                ((tp) != NULL && TYPE_BASIC_TYPE(tp) == TYPE_ARRAY)
#define IS_ELEMENT_TYPE(tp) \
                ((tp) != NULL && (tp)->ref == NULL)
#define IS_FUNCTION_TYPE(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_FUNCTION))
#define IS_SUBR(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_SUBR))
#define IS_VOID(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_VOID))
#define IS_PROCEDURE_TYPE(tp) \
                (IS_FUNCTION_TYPE(tp) || IS_SUBR(tp))
#define IS_GENERIC_PROCEDURE_TYPE(tp) \
                (IS_PROCEDURE_TYPE(tp) && FUNCTION_TYPE_IS_GENERIC(tp))
#define IS_COMPLEX(tp) \
                ((tp) != NULL && \
                (TYPE_BASIC_TYPE(tp) == TYPE_COMPLEX || \
                TYPE_BASIC_TYPE(tp) == TYPE_DCOMPLEX))
#define IS_REAL(tp) \
                ((tp) != NULL && \
                (TYPE_BASIC_TYPE(tp) == TYPE_REAL || \
                TYPE_BASIC_TYPE(tp) == TYPE_DREAL))
#define IS_INT(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_INT))
#define IS_INT_OR_REAL(tp) \
                (IS_INT(tp) || IS_REAL(tp) || IS_GNUMERIC(tp))
#define IS_NUMERIC(tp)  \
                ((tp) != NULL && (IS_COMPLEX(tp)||IS_REAL(tp)|| \
                IS_INT(tp)||IS_GNUMERIC(tp)||IS_GNUMERIC_ALL(tp)))
#define IS_NUMERIC_OR_LOGICAL(tp)  \
                (IS_NUMERIC(tp) || IS_LOGICAL(tp))
#define IS_CHAR(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_CHAR))
#define IS_CHAR_LEN_UNFIXED(tp) \
                ((tp) != NULL && (TYPE_CHAR_LEN(tp) == CHAR_LEN_UNFIXED))
#define IS_CHAR_LEN_ALLOCATABLE(tp) \
                ((tp) != NULL && (TYPE_CHAR_LEN(tp) == CHAR_LEN_ALLOCATABLE))
#define IS_LOGICAL(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_LOGICAL))
#define IS_INT_CONST_V(v) \
                (IS_INT(EXPV_TYPE(v)) && expr_is_constant(v))
#define IS_INT_PARAM_V(v) \
                (IS_INT(EXPV_TYPE(v)) && expr_is_param(v))
#define IS_REAL_CONST_V(v) \
                (IS_REAL(EXPV_TYPE(v)) && expr_is_constant(v))
#define IS_COMPLEX_CONST_V(v) \
                (IS_COMPLEX(EXPV_TYPE(v)) && expr_is_constant(v))
#define IS_NUMERIC_CONST_V(v)   \
                ((IS_INT_CONST_V(v)) || (IS_REAL_CONST_V(v)) || \
                (IS_COMPLEX_CONST_V(v)))
#define IS_GNUMERIC(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_GNUMERIC))
#define IS_GNUMERIC_ALL(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_GNUMERIC_ALL))
#define IS_GENERIC_TYPE(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_GENERIC))
#define IS_DOUBLED_TYPE(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_DREAL || \
                TYPE_BASIC_TYPE(tp) == TYPE_DCOMPLEX))
#define IS_MODULE(tp) \
                ((tp) != NULL && TYPE_BASIC_TYPE(tp) == TYPE_MODULE)

#define IS_NAMELIST(tp) \
                ((tp) != NULL && TYPE_BASIC_TYPE(tp) == TYPE_NAMELIST)

#define IS_REFFERENCE(tp) \
                ((tp) != NULL && TYPE_N_DIM(tp) == 0 && TYPE_REF(tp) != NULL)

#define TYPE_IS_MODIFIED(tp) \
                ((tp) != NULL && (tp)->is_modified)

#define SET_MODIFIED(tp) \
    ((tp != NULL) && ((tp)->is_modified = TRUE))

#define UNSET_MODIFIED(tp) \
    ((tp != NULL) && ((tp)->is_modified = FALSE))


#define FOREACH_STRUCTDECLS(/* TYPE_DESC */ tp, /* TYPE_DESC */stp) \
    if ((stp) !=NULL) \
        for ((tp) = (stp); (tp) != NULL; (tp) = TYPE_SLINK(tp))

#define SAFE_FOREACH_STRUCTDECLS(tp, tq, headp)\
    SAFE_FOREACH(tp, tq, headp, TYPE_SLINK)

#define FOREACH_MEMBER(/* ID */ mp, /* TYPE_DESC */ tp) \
    if ((tp) != NULL && TYPE_MEMBER_LIST(tp) != NULL) \
        FOREACH_ID(mp, TYPE_MEMBER_LIST(tp))

#define FOREACH_TYPE_PARAMS(/* ID */ mp, /* TYPE_DESC */ tp) \
    if ((tp) != NULL && TYPE_TYPE_PARAMS(tp) != NULL) \
        FOREACH_ID(mp, TYPE_TYPE_PARAMS(tp))

#define FOREACH_TYPE_BOUND_PROCEDURE(/* ID */ mp, /* TYPE_DESC */ tp) \
    FOREACH_MEMBER(mp, tp) \
    if (ID_CLASS(mp) == CL_TYPE_BOUND_PROC && \
        !(TBP_BINDING_ATTRS(mp) & TYPE_BOUND_PROCEDURE_IS_GENERIC))

#define FOREACH_TYPE_BOUND_GENERIC(/* ID */ mp, /* TYPE_DESC */ tp) \
    FOREACH_MEMBER(mp, tp) \
    if (ID_CLASS(mp) == CL_TYPE_BOUND_PROC && \
        (TBP_BINDING_ATTRS(mp) & TYPE_BOUND_PROCEDURE_IS_GENERIC))

#define FUNCTION_TYPE_RETURN_TYPE(tp) ((tp)->proc_info.return_type)
#define FUNCTION_TYPE_HAS_EXPLICIT_ARGS(tp) ((tp)->proc_info.has_explicit_arguments)
#define FUNCTION_TYPE_ARGS(tp) ((tp)->proc_info.args)
#define FUNCTION_TYPE_RESULT(tp) ((tp)->proc_info.result)

#define FUNCTION_TYPE_HAS_IMPLICIT_RETURN_TYPE(tp) \
    (IS_FUNCTION_TYPE(tp) && TYPE_IS_IMPLICIT(FUNCTION_TYPE_RETURN_TYPE(tp)))

#define FUNCTION_TYPE_HAS_UNKNOWN_RETURN_TYPE(tp) \
    (IS_FUNCTION_TYPE(tp) && \
     (TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp)) == TYPE_UNKNOWN))

#define FUNCTION_TYPE_HAS_EXPLICT_INTERFACE(tp) \
    (FUNCTION_TYPE_RETURN_TYPE(tp) != NULL && FUNCTION_TYPE_HAS_EXPLICIT_ARGS(tp))

#define FUNCTION_TYPE_IS_PROGRAM(tp) ((tp)->proc_info.is_program)
#define FUNCTION_TYPE_SET_PROGRAM(tp) ((tp)->proc_info.is_program = TRUE)
#define FUNCTION_TYPE_UNSET_PROGRAM(tp) ((tp)->proc_info.is_program = FALSE)

#define FUNCTION_TYPE_IS_TYPE_BOUND(tp) ((tp)->proc_info.is_tbp == TRUE)
#define FUNCTION_TYPE_SET_TYPE_BOUND(tp) ((tp)->proc_info.is_tbp = TRUE)
#define FUNCTION_TYPE_UNSET_TYPE_BOUND(tp) ((tp)->proc_info.is_tbp = FALSE)

#define FUNCTION_TYPE_IS_GENERIC(tp) ((tp)->proc_info.is_generic == TRUE)
#define FUNCTION_TYPE_SET_GENERIC(tp) ((tp)->proc_info.is_generic = TRUE)
#define FUNCTION_TYPE_UNSET_GENERIC(tp) ((tp)->proc_info.is_generic = FALSE)

#define FUNCTION_TYPE_IS_DEFINED(tp) ((tp)->proc_info.is_defined == TRUE)
#define FUNCTION_TYPE_SET_DEFINED(tp) ((tp)->proc_info.is_defined = TRUE)
#define FUNCTION_TYPE_UNSET_DEFINED(tp) ((tp)->proc_info.is_defined = FALSE)

/* For is_external attribute */
#define FUNCTION_TYPE_IS_INTERNAL(tp) ((tp)->proc_info.is_internal == TRUE)
#define FUNCTION_TYPE_SET_INTERNAL(tp) ((tp)->proc_info.is_internal = TRUE)
#define FUNCTION_TYPE_UNSET_INTERNAL(tp) ((tp)->proc_info.is_internal = FALSE)

/* For is_external attribute */
#define FUNCTION_TYPE_IS_MOUDLE_PROCEDURE(tp) ((tp)->proc_info.is_module_procedure == TRUE)
#define FUNCTION_TYPE_SET_MOUDLE_PROCEDURE(tp) ((tp)->proc_info.is_module_procedure = TRUE)
#define FUNCTION_TYPE_UNSET_MOUDLE_PROCEDURE(tp) ((tp)->proc_info.is_module_procedure = FALSE)

#define FUNCTION_TYPE_IS_VISIBLE_INTRINSIC(tp) ((tp)->proc_info.is_visible_intrinsic == TRUE)
#define FUNCTION_TYPE_SET_VISIBLE_INTRINSIC(tp) ((tp)->proc_info.is_visible_intrinsic = TRUE)
#define FUNCTION_TYPE_UNSET_VISIBLE_INTRINSIC(tp) ((tp)->proc_info.is_visible_intrinsic = FALSE)

#define TYPE_BOUND_GENERIC_TYPE_GENERICS(tp) ((tp)->proc_info.type_bound_proc_info.generics)
#define TYPE_BOUND_PROCEDURE_TYPE_HAS_PASS_ARG(tp) ((tp)->proc_info.has_pass_arg)
#define TYPE_BOUND_PROCEDURE_TYPE_PASS_ARG(tp) ((tp)->proc_info.pass_arg)

#define FUNCTION_TYPE_HAS_BINDING_ARG(tp) ((tp)->proc_info.has_binding_arg)
#define FUNCTION_TYPE_HAS_PASS_ARG(tp) ((tp)->proc_info.has_pass_arg)
#define FUNCTION_TYPE_PASS_ARG(tp) ((tp)->proc_info.pass_arg)
#define FUNCTION_TYPE_PASS_ARG_TYPE(tp) ((tp)->proc_info.pass_arg_type)


#endif /* _F_DATATYPE_H_ */
