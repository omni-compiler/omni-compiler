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



    /* for coarray functions. */
    INTR_TYPE_COARRAY_ANY,
    INTR_TYPE_COARRAY_INT,
    INTR_TYPE_COARRAY_REAL,
    INTR_TYPE_COARRAY_LOGICAL,
    INTR_TYPE_SCALAR_COARRAY_ANY,
    INTR_TYPE_SCALAR_COARRAY_INT,
    INTR_TYPE_SCALAR_COARRAY_REAL,
    INTR_TYPE_SCALAR_COARRAY_LOGICAL,


    /* Others. */
    INTR_TYPE_POINTER, 
    INTR_TYPE_TARGET,

    INTR_TYPE_ANY_ARRAY_ALLOCATABLE,
    INTR_TYPE_ANY_OPTIONAL,


    /* For NULL(void) */
    INTR_TYPE_LHS

} INTR_DATA_TYPE;


typedef enum {
    INTR_UNKNOWN = 0,

    /* Numerical functions. */
    INTR_ABS,
    INTR_AIMAG,
    INTR_AINT,
    INTR_ANINT,
    INTR_CMPLX,
    INTR_DCMPLX,
    INTR_CONJG,
    INTR_DCONJG,
    INTR_DABS,
    INTR_DBLE,
    INTR_DIM,
    INTR_DIMAG,
    INTR_DPROD,
    INTR_DREAL,
    INTR_INT,
    INTR_MAX,
    INTR_MIN,
    INTR_MOD,
    INTR_NINT,
    INTR_REAL,
    INTR_SIGN,

    /* Mathematical functions. */
    INTR_ACOS,
    INTR_ASIN,
    INTR_ATAN,
    INTR_ATAN2,
    INTR_COS,
    INTR_COSH,
    INTR_EXP,
    INTR_LOG,
    INTR_LOG10,
    INTR_SIN,
    INTR_SINH,
    INTR_SQRT,
    INTR_TAN,
    INTR_TANH,

    /* Character functions. */
    INTR_CHAR,
    INTR_ICHAR,
    INTR_INDEX,
    INTR_LGE,
    INTR_LGT,
    INTR_LLE,
    INTR_LLT,

    /* Character inquiry functions. */
    INTR_LEN,

    /* F77 non-standard*/
    INTR_LOC,

    /* F90 numeric functions. */
    INTR_CEILING,
    INTR_FLOOR,
    INTR_MODULO,

    /* F90 character functions. */
    INTR_ACHAR,
    INTR_ADJUSTL,
    INTR_ADJUSTR,
    INTR_IACHAR,
    INTR_LEN_TRIM,
    INTR_REPEAT,
    INTR_SCAN,
    INTR_TRIM,
    INTR_VERIFY,

    /* F90 kind functions. */
    INTR_KIND,
    INTR_SELECTED_INT_KIND,
    INTR_SELECTED_REAL_KIND,
    /* F2003 kind function */
    INTR_SELECTED_CHAR_KIND,

    /* 7. Logical function */

    INTR_LOGICAL,

    /* F90 numeric inquiry functions. */
    INTR_DIGITS,
    INTR_EPSILON,
    INTR_HUGE,
    INTR_MAXEXPONENT,
    INTR_MINEXPONENT,
    INTR_PRECISION,
    INTR_RADIX,
    INTR_RANGE,
    INTR_TINY,

    /* F90 bit inquiry functions. */
    INTR_BIT_SIZE,
    INTR_BTEST,
    INTR_IAND,
    INTR_IBCLR,
    INTR_IBITS,
    INTR_IBSET,
    INTR_IEOR,
    INTR_IOR,
    INTR_ISHFT,
    INTR_ISHFTC,
    INTR_NOT,

    /* F90 transfer functions. */
    INTR_TRANSFER,

    /* F90 floating-point manipulation functions. */
    INTR_EXPONENT,
    INTR_FRACTION,
    INTR_NEAREST,
    INTR_RRSPACING,
    INTR_SCALE,
    INTR_SET_EXPONENT,
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
    INTR_PACK,
    INTR_SPREAD,
    INTR_UNPACK,

    /* F90 array reshape functions. */
    INTR_RESHAPE,

    /* F90 array manipulation functions. */
    INTR_CSHIFT,
    INTR_TRANSPOSE,

    /* F90 array location functions. */
    INTR_MINLOC,
    INTR_MAXLOC,

    /* F90 pointer association status functions. */
    INTR_ASSOCIATED,

    /* F90 intrinsic subroutines. */
    INTR_DATE_AND_TIME,
    INTR_MVBITS,
    INTR_RANDOM_NUMBER,
    INTR_RANDOM_SEED,
    INTR_SYSTEM_CLOCK,

    /* F95 intrinsic functions. */
    INTR_PRESENT,
    INTR_EOSHIFT,

    /* 20. Pointer association status functions */
    INTR_NULL,

    /* F95 intrinsic subroutines. */
    INTR_CPU_TIME,

    /* CAF1.0 (F2008) intrinsic functions */
    INTR_NUM_IMAGES,
    INTR_THIS_IMAGE,
    INTR_IMAGE_INDEX,
    INTR_LCOBOUND,
    INTR_UCOBOUND,

    INTR_ATOMIC_DEFINE,
    INTR_ATOMIC_REF,

    /* F03 intrinsic subroutins */
    INTR_IS_IOSTAT_END,
    INTR_IS_IOSTAT_EOR,

    INTR_EXTENDS_TYPE_OF,
    INTR_SAME_TYPE_AS,
    INTR_MOVE_ALLOC,

    /* F08 intrinsic subroutines */
    INTR_COMMAND_ARUGMENT_COUNT,
    INTR_GET_COMMAND,
    INTR_GET_COMMAND_ARUGMENT,
    INTR_GET_ENVIRONMENT_VARIABLE,
    INTR_GAMMA,
    INTR_LOGGAMMA,

    INTR_COARRAY_MALLOC_BYTES,       // hidden interface
    INTR_COARRAY_ALLOCATED_BYTES,    // hidden interface
    INTR_COARRAY_GARBAGE_BYTES,      // hidden interface

    /* XMP/F */
    INTR_DESC_OF,
    INTR_GET_MPI_COMM,
    INTR_NUM_NODES,
    INTR_NODE_NUM,
    INTR_ALL_NUM_NODES,
    INTR_ALL_NODE_NUM,
    INTR_WTIME,
    INTR_WTICK,

    INTR_ARRAY_NDIMS,
    INTR_ARRAY_LBOUND,
    INTR_ARRAY_UBOUND,
    INTR_ARRAY_LSIZE,
    INTR_ARRAY_USHADOW,
    INTR_ARRAY_LSHADOW,
    INTR_ARRAY_LEAD_DIM,
    INTR_ARRAY_GTOL,
    INTR_ALIGN_AXIS,
    INTR_ALIGN_OFFSET,
    INTR_ALIGN_REPLICATED,
    INTR_ALIGN_TEMPLATE,
    INTR_TEMPLATE_FIXED,
    INTR_TEMPLATE_NDIMS,
    INTR_TEMPLATE_LBOUND,
    INTR_TEMPLATE_UBOUND,
    INTR_DIST_FORMAT,
    INTR_DIST_BLOCKSIZE,
    INTR_DIST_GBLOCKMAP,
    INTR_DIST_NODES,
    INTR_DIST_AXIS,
    INTR_NODES_NDIMS,
    INTR_NODES_INDEX,
    INTR_NODES_SIZE,
    INTR_NODES_EQUIV,

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

                        /* -7 : return type always comforms to the
                            left hand. */

                        /* -8 : return type is external. XMP original 
                            intrinsic functions need to use this value. */

                        /* -9 : return type is anything. returnType 
                            may assume INTR_TYPE_INT. */

    int langSpec;
    int intrinsicClass;
#define INTRINSIC_CLASS_NONE           0x0000
#define INTRINSIC_CLASS_ATOMIC         0x0001
#define INTRINSIC_CLASS_ELEMENTAL_FUN  0x0002
#define INTRINSIC_CLASS_ELEMENTAL_SUB  0x0004
#define INTRINSIC_CLASS_INQUIRY        0x0008
#define INTRINSIC_CLASS_PURE_SUB       0x0010
#define INTRINSIC_CLASS_SUB            0x0020
#define INTRINSIC_CLASS_TRANS          0x0040

#define INTR_CLASS_N       INTRINSIC_CLASS_NONE
#define INTR_CLASS_A       INTRINSIC_CLASS_ATOMIC
#define INTR_CLASS_E       INTRINSIC_CLASS_ELEMENTAL_FUN
#define INTR_CLASS_ES      INTRINSIC_CLASS_ELEMENTAL_SUB
#define INTR_CLASS_I       INTRINSIC_CLASS_INQUIRY
#define INTR_CLASS_PS      INTRINSIC_CLASS_PURE_SUB
#define INTR_CLASS_S       INTRINSIC_CLASS_SUB
#define INTR_CLASS_T       INTRINSIC_CLASS_TRANS

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
#define INTR_CLASS(ep)          ((ep)->intrinsicClass)

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
