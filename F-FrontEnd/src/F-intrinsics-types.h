/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-intrinsics-types.h
 */

#ifndef _F_INTRINSICS_TYPES_H_
#define _F_INTRINSICS_TYPES_H_


typedef enum {
    INTR_TYPE_NONE = 0,



    INTR_TYPE_INT,
    INTR_TYPE_REAL,
    INTR_TYPE_DREAL,
    INTR_TYPE_ALL_REAL,
    INTR_TYPE_COMPLEX,
    INTR_TYPE_DCOMPLEX,
    INTR_TYPE_ALL_COMPLEX,
    INTR_TYPE_CHAR,
    INTR_TYPE_LOGICAL,

    INTR_TYPE_ANY,

    INTR_TYPE_NUMERICS,         /* INTR_TYPE_INT, INTR_TYPE_REAL or INTR_TYPE_DREAL. */
    INTR_TYPE_ALL_NUMERICS,     /* INTR_TYPE_INT, INTR_TYPE_REAL, INTR_TYPE_DREAL or INTR_TYPE_COMPLEX. */



    INTR_TYPE_INT_ARRAY,
    INTR_TYPE_REAL_ARRAY,
    INTR_TYPE_DREAL_ARRAY,
    INTR_TYPE_ALL_REAL_ARRAY,
    INTR_TYPE_COMPLEX_ARRAY,
    INTR_TYPE_DCOMPLEX_ARRAY,
    INTR_TYPE_ALL_COMPLEX_ARRAY,
    INTR_TYPE_CHAR_ARRAY,
    INTR_TYPE_LOGICAL_ARRAY,

    INTR_TYPE_ANY_ARRAY,

    INTR_TYPE_NUMERICS_ARRAY,
    INTR_TYPE_ALL_NUMERICS_ARRAY,



    /* for Array reduction functions. */
    INTR_TYPE_INT_DYNAMIC_ARRAY,
    INTR_TYPE_REAL_DYNAMIC_ARRAY,
    INTR_TYPE_DREAL_DYNAMIC_ARRAY,
    INTR_TYPE_ALL_REAL_DYNAMIC_ARRAY,
    INTR_TYPE_COMPLEX_DYNAMIC_ARRAY,
    INTR_TYPE_DCOMPLEX_DYNAMIC_ARRAY,
    INTR_TYPE_ALL_COMPLEX_DYNAMIC_ARRAY,
    INTR_TYPE_CHAR_DYNAMIC_ARRAY,
    INTR_TYPE_LOGICAL_DYNAMIC_ARRAY,

    INTR_TYPE_ANY_DYNAMIC_ARRAY,

    INTR_TYPE_NUMERICS_DYNAMIC_ARRAY,
    INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,



    /* Others. */
    INTR_TYPE_POINTER, 
    INTR_TYPE_TARGET,

    INTR_TYPE_ANY_ARRAY_ALLOCATABLE,
    INTR_TYPE_ANY_OPTIONAL

} INTR_DATA_TYPE;


typedef enum {
    INTR_UNKNOWN = 0,

    /* Numerical functions. */
    INTR_ABS,
    INTR_AIMAG,
    INTR_AINT,
    INTR_CMPLX,
    INTR_DCMPLX,
    INTR_CONJG,
    INTR_DCONJG,
    INTR_DABS,
    INTR_DBLE,
    INTR_DIMAG,
    INTR_INT,
    INTR_MAX,
    INTR_MIN,
    INTR_MOD,
    INTR_NINT,
    INTR_REAL,
    INTR_SIGN,

    /* Mathematical functions. */
    INTR_ACOS,
    INTR_ATAN,
    INTR_COS,
    INTR_EXP,
    INTR_LOG,
    INTR_LOG10,
    INTR_SIN,
    INTR_SINH,
    INTR_SQRT,

    /* Character functions. */
    INTR_CHAR,
    INTR_ICHAR,
    INTR_INDEX,

    /* Character inquiry functions. */
    INTR_LEN,

    /* F90 numeric functions. */
    INTR_CEILING,
    INTR_FLOOR,

    /* F90 character functions. */
    INTR_ACHAR,
    INTR_ADJUSTL,
    INTR_IACHAR,
    INTR_LEN_TRIM,
    INTR_SCAN,
    INTR_TRIM,
    INTR_VERIFY,

    /* F90 kind functions. */
    INTR_KIND,
    INTR_SELECTED_INT_KIND,
    INTR_SELECTED_REAL_KIND,

    /* F90 numeric inquiry functions. */
    INTR_DIGITS,
    INTR_EPSILON,
    INTR_HUGE,
    INTR_MAXEXPONENT,
    INTR_TINY,

    /* F90 bit inquiry functions. */
    INTR_BIT_SIZE,
    INTR_BTEST,
    INTR_IAND,
    INTR_IBITS,
    INTR_IBSET,
    INTR_IOR,
    INTR_ISHFT,

    /* F90 transfer functions. */
    INTR_TRANSFER,

    /* F90 floating-point manipulation functions. */
    INTR_EXPONENT,
    INTR_SCALE,
    INTR_SPACING,

    /* F90 vector and matrix multiply functions. */
    INTR_DOT_PRODUCT,
    INTR_MATMUL,

    /* F90 array reduction functions. */
    INTR_ALL,
    INTR_ANY,
    INTR_COUNT,
    INTR_MAXVAL,
    INTR_MINVAL,
    INTR_PRODUCT,
    INTR_SUM,

    /* F90 array inquiry functions. */
    INTR_ALLOCATED,
    INTR_LBOUND,
    INTR_SHAPE,
    INTR_SIZE,
    INTR_UBOUND,

    /* F90 array construction functions. */
    INTR_MERGE,
    INTR_SPREAD,

    /* F90 array reshape functions. */
    INTR_RESHAPE,

    /* F90 array manipulation functions. */
    INTR_CSHIFT,
    INTR_TRANSPOSE,

    /* F90 array location functions. */
    INTR_MINLOC,

    /* F90 pointer association status functions. */
    INTR_ASSOCIATED,

    /* F90 intrinsic subroutines. */
    INTR_DATE_AND_TIME,
    INTR_SYSTEM_CLOCK,

    /* F95 intrinsic functions. */
    INTR_PRESENT,
    INTR_EOSHIFT,

    /* F95 intrinsic subroutines. */
    INTR_CPU_TIME,

    INTR_END

} INTR_OPS;


typedef enum {
    INTR_NAME_GENERIC = 0,
    INTR_NAME_SPECIFIC,
    INTR_NAME_SPECIFIC_NA
} INTR_NAME_TYPE;


typedef struct {
    INTR_OPS ops;
    INTR_NAME_TYPE nameType;
    const char *name;
    int hasKind;
    INTR_DATA_TYPE argsType[10];
    INTR_DATA_TYPE returnType;
    int nArgs;

    int retTypeSameAs;  /* greater than/equals zero (n) : return type
                         * is equals to (n)th arg's type. */

                        /* -1 : return type completely differs to any
                            args. */

                        /* -2 : return type is the BASIC_TYPE of the
                            first arg. */

                        /* -3 : return type is a single dimension
                            array of integer, in which having elements
                            equals to the first arg's dimension. */

                        /* -4 : return type is transpose of the first
                           arg (two dimension/matrix). */

                        /* -5 : BASIC_TYPE of return type is 'returnType'
                            and kind of return type is same as first
                            arg. */

                        /* -6 : return type completely differs to any
                            args and always scalar type. */

    int langSpec;
} intrinsic_entry;
#define INTR_OP(ep)             ((ep)->ops)
#define INTR_NAMETYPE(ep)       ((ep)->nameType)
#define INTR_IS_GENERIC(ep)     (INTR_NAMETYPE(ep) == INTR_NAME_GENERIC)
#define INTR_NAME(ep)           ((ep)->name)
#define INTR_ARG_TYPE(ep)       ((ep)->argsType)
#define INTR_KIND(ep)           ((ep)->hasKind)
#define INTR_HAS_KIND_ARG(ep)   (INTR_KIND(ep) == 1)
#define INTR_RETURN_TYPE(ep)    ((ep)->returnType)
#define INTR_N_ARGS(ep)         ((ep)->nArgs)
#define INTR_RETURN_TYPE_SAME_AS(ep)    ((ep)->retTypeSameAs)

#define INTR_IS_RETURN_TYPE_DYNAMIC(ep) \
    (INTR_RETURN_TYPE(ep) == INTR_TYPE_INT_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_REAL_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_DREAL_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_ALL_REAL_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_COMPLEX_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_DCOMPLEX_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_ALL_COMPLEX_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_CHAR_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_LOGICAL_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_ANY_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_NUMERICS_DYNAMIC_ARRAY || \
     INTR_RETURN_TYPE(ep) == INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY)

#define INTR_IS_ARG_TYPE0_ARRAY(ep) \
    (INTR_ARG_TYPE(ep)[0] == INTR_TYPE_INT_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_REAL_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_DREAL_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_ALL_REAL_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_COMPLEX_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_DCOMPLEX_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_ALL_COMPLEX_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_CHAR_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_LOGICAL_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_ANY_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_NUMERICS_ARRAY || \
    INTR_ARG_TYPE(ep)[0] == INTR_TYPE_ALL_NUMERICS_ARRAY)

/*
 * NOTE:
 *
 *      If INTR_KIND(ep) == 1, INTR_RETURN_TYPE_SAME_AS(ep) must be
 *      -1.
 */

extern intrinsic_entry intrinsic_table[];


#endif /* _F_INTRINSICS_TYPES_H_ */
