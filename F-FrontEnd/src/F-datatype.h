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
    TYPE_LHS,		/* 17 type for intrinsic null(), always
                         * comforms to the type of the left hand
                         * expression. */
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
        uint32_t type_attr_flags;
#define TYPE_EXFLAGS_IMPLICIT       0x00000001 /* implicitly defined or not */
#define TYPE_EXFLAGS_OVERRIDDEN     0x00000002 /* type is overridden by child */
#define TYPE_EXFLAGS_USED_EXPLICIT  0x00000004 /* OBSOLETE: not used anymore */
#define TYPE_EXFLAGS_NOT_FIXED      0x00000008 /* type is not fixed, since expression
                                                  contains undefined function. */
#define TYPE_EXFLAGS_FOR_FUNC_SELF  0x00000010 /* type is for the function itself */
        uint32_t exflags;
    } attr; /* FbasicType */
    struct {
        char n_dim;            /* dimension (max 7) */
        char dim_fixed;        /* fixed or not */
        char dim_fixing;
        ARRAY_ASSUME_KIND assume_kind; /* represents assumed size or shape */
        expv dim_size;
        expv dim_upper, dim_lower, dim_step; /* dimension subscripts */
    } array_info; /* FOR FbasicType for Array */
    struct ident_descriptor *members; /* all members for derived type */
    codims_desc *codims;
    int is_reshaped_type;       /* A bool flag to specify this type is
                                 * genereted by reshape() intrinsic. */
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
#if 0
#define TYPE_LINK_ADD(tp, tlist, ttail) \
    { if((tlist) == NULL) (tlist) = (tp); \
      else TYPE_LINK(ttail) = (tp); \
      (ttail) = (tp); }
#else
#define TYPE_LINK_ADD(tp, tlist, ttail) \
    { if((tlist) == NULL) (tlist) = (tp); \
      ttail = type_link_add(tp, tlist, ttail);  \
    }
#endif
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
#define TYPE_UNSET_REDURSIVE(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_RECURSIVE)
#define TYPE_IS_PURE(tp)            ((tp)->attr.type_attr_flags &   TYPE_ATTR_PURE)
#define TYPE_SET_PURE(tp)           ((tp)->attr.type_attr_flags |=  TYPE_ATTR_PURE)
#define TYPE_UNSET_PURE(tp)         ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_PURE)
#define TYPE_IS_ELEMENTAL(tp)       ((tp)->attr.type_attr_flags &   TYPE_ATTR_ELEMENTAL)
#define TYPE_SET_ELEMENTAL(tp)      ((tp)->attr.type_attr_flags |=  TYPE_ATTR_ELEMENTAL)
#define TYPE_UNSET_ELEMENTAL(tp)    ((tp)->attr.type_attr_flags &= ~TYPE_ATTR_ELEMENTAL)
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
#define TYPE_SET_NOT_FIXED(tp)      ((tp)->attr.exflags |=  TYPE_EXFLAGS_NOT_FIXED)
#define TYPE_UNSET_NOT_FIXED(tp)    ((tp)->attr.exflags &= ~TYPE_EXFLAGS_NOT_FIXED)

#define TYPE_IS_FOR_FUNC_SELF(tp)   ((tp)->attr.exflags &   TYPE_EXFLAGS_FOR_FUNC_SELF)
#define TYPE_SET_FOR_FUNC_SELF(tp)  ((tp)->attr.exflags |=  TYPE_EXFLAGS_FOR_FUNC_SELF)
#define TYPE_UNSET_FOR_FUNC_SELF(tp) ((tp)->attr.exflags &= ~TYPE_EXFLAGS_FOR_FUNC_SELF)

#define TYPE_HAS_INTENT(tp)      (TYPE_IS_INTENT_IN(tp) || \
                TYPE_IS_INTENT_OUT(tp) || TYPE_IS_INTENT_INOUT(tp))
// TODO PROTECTED 
#define IS_TYPE_PUBLICORPRIVATE(tp)  \
                ((TYPE_IS_PUBLIC(tp)) || (TYPE_IS_PRIVATE(tp)))

#define TYPE_N_DIM(tp)          ((tp)->array_info.n_dim)
#define TYPE_DIM_FIXED(tp)      ((tp)->array_info.dim_fixed)
#define TYPE_DIM_FIXING(tp)     ((tp)->array_info.dim_fixing)
#define TYPE_DIM_SIZE(tp)       ((tp)->array_info.dim_size)
#define TYPE_DIM_UPPER(tp)      ((tp)->array_info.dim_upper)
#define TYPE_DIM_LOWER(tp)      ((tp)->array_info.dim_lower)
#define TYPE_DIM_STEP(tp)       ((tp)->array_info.dim_step)
#define TYPE_IS_SCALAR(tp)      (((tp)->array_info.n_dim == 0))
#define TYPE_MEMBER_LIST(tp)    ((tp)->members)

#define TYPE_CHAR_LEN(tp)       ((tp)->size)
#define TYPE_KIND(tp)           ((tp)->kind)
#define TYPE_LENG(tp)            ((tp)->leng)

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

/* macros distinguishing type */
#define IS_STRUCT_TYPE(tp) \
                ((tp) != NULL && TYPE_BASIC_TYPE(tp) == TYPE_STRUCT)
#define IS_ARRAY_TYPE(tp) \
                ((tp) != NULL && TYPE_BASIC_TYPE(tp) == TYPE_ARRAY)
#define IS_ELEMENT_TYPE(tp) \
                ((tp) != NULL && (tp)->ref == NULL)
#define IS_FUNCTION_TYPE(tp) \
                ((tp) != NULL && TYPE_BASIC_TYPE(tp) == TYPE_FUNCTION)
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
#define IS_LOGICAL(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_LOGICAL))
#define IS_SUBR(tp) \
                ((tp) != NULL && (TYPE_BASIC_TYPE(tp) == TYPE_SUBR))
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

#define FOREACH_MEMBER(/* ID */ mp, /* TYPE_DESC */ tp) \
    if ((tp) != NULL && TYPE_MEMBER_LIST(tp) != NULL) \
        FOREACH_ID(mp, TYPE_MEMBER_LIST(tp))

#if 0
typedef enum {
    PRAGMA_NOT_IN_SCOPE = 0,	/* The sentinel is not appeared, yet. */
    PRAGMA_ENTER_SCOPE,		/* The sentinel just appeared. */
    PRAGMA_LEAVE_SCOPE		/* The sentinel got enough block(s) and
                                 * the scope is needed to be
                                 * closed. */
} pragma_status_t;
#endif

#endif /* _F_DATATYPE_H_ */
