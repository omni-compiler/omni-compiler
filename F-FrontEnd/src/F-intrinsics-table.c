/**
 * \file F-intrinsics-table.c
 */

#include "F-intrinsics-types.h"
#include "F-front.h"

/*
 * NOTE:
 *
 *      If INTR_KIND(ep) == 1, INTR_RETURN_TYPE_SAME_AS(ep) must be
 *      -1.
 */

intrinsic_entry intrinsic_table[] = {
    /*
     * FORTRAN77 intrinsic functions
     */

    /* 2. Numeric functions */

    // ABS (A)
    { INTR_ABS,         INTR_NAME_GENERIC,      "abs",          0,      {INTR_TYPE_NUMERICS},   INTR_TYPE_NUMERICS,     1,  0, LANGSPEC_F77 },
    { INTR_ABS,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_REAL,         1, -1, LANGSPEC_F77 },
    { INTR_ABS,         INTR_NAME_SPECIFIC,     "dabs",         0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1,  0, LANGSPEC_F77 },
    { INTR_ABS,         INTR_NAME_SPECIFIC,     "iabs",         0,      {INTR_TYPE_INT},        INTR_TYPE_INT,          1,  0, LANGSPEC_F77 },
    { INTR_ABS,         INTR_NAME_SPECIFIC,     "cabs",         0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_REAL,         1, -1, LANGSPEC_F77 },

    // AIMAG (Z)
    { INTR_AIMAG,       INTR_NAME_SPECIFIC,     "aimag",        0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_REAL,         1, -5, LANGSPEC_F77 },
    { INTR_AIMAG,       INTR_NAME_SPECIFIC,     "",             0,      {INTR_TYPE_DCOMPLEX},   INTR_TYPE_DREAL,        1, -5, LANGSPEC_F77 },

    // DIMAG (Z)
    { INTR_DIMAG,       INTR_NAME_GENERIC,      "dimag",        0,      {INTR_TYPE_DCOMPLEX},   INTR_TYPE_DREAL,        1, -5, LANGSPEC_NONSTD },
    { INTR_DIMAG,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_REAL,         1, -5, LANGSPEC_NONSTD },

    // AINT (A  [, KIND])
    { INTR_AINT,        INTR_NAME_GENERIC,      "aint",         1,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, -1, LANGSPEC_F77 },
    { INTR_AINT,        INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, -1, LANGSPEC_F77 },
    { INTR_AINT,        INTR_NAME_SPECIFIC,     "dint",         1,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, -1, LANGSPEC_F77 },

    // ANINT (A [, KIND])
    { INTR_ANINT,       INTR_NAME_GENERIC,      "anint",        1,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, -1, LANGSPEC_F77 },
    { INTR_ANINT,       INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, -1, LANGSPEC_F77 },
    { INTR_ANINT,       INTR_NAME_SPECIFIC,     "dnint",        1,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, -1, LANGSPEC_F77 },

    // CMPLX (X [, Y][, KIND])
    { INTR_CMPLX,       INTR_NAME_GENERIC,      "cmplx",        1,      {INTR_TYPE_ALL_NUMERICS},                       INTR_TYPE_COMPLEX,      1, -1, LANGSPEC_F77 },
    { INTR_CMPLX,       INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS},       INTR_TYPE_COMPLEX,      2, -1, LANGSPEC_F77 },
    { INTR_CMPLX,       INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS},       INTR_TYPE_COMPLEX,      3, -1, LANGSPEC_F77 },
    { INTR_DCMPLX,      INTR_NAME_SPECIFIC,     "dcmplx",       1,      {INTR_TYPE_NUMERICS},                           INTR_TYPE_DCOMPLEX,     1, -1, LANGSPEC_NONSTD },
    { INTR_DCMPLX,      INTR_NAME_SPECIFIC,     "",             1,      {INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS},       INTR_TYPE_DCOMPLEX,     2, -1, LANGSPEC_NONSTD },

    // CONJG (Z)  ???
    { INTR_CONJG,       INTR_NAME_SPECIFIC,     "conjg",        0,      {INTR_TYPE_COMPLEX},                    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },

    // DCONJG (Z) ???
    { INTR_DCONJG,      INTR_NAME_SPECIFIC,     "dconjg",       0,      {INTR_TYPE_COMPLEX},                   INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },

    // DBLE (A)
    { INTR_DBLE,        INTR_NAME_GENERIC,      "dble",         0,      {INTR_TYPE_ALL_NUMERICS},               INTR_TYPE_DREAL,        1, -1, LANGSPEC_F77 },

    // DREAL (A)
    { INTR_REAL,        INTR_NAME_GENERIC,      "dreal",         0,      {INTR_TYPE_ALL_NUMERICS},               INTR_TYPE_DREAL,        1, -1, LANGSPEC_NONSTD },

    // DIM (X, Y)
    { INTR_DIM,         INTR_NAME_GENERIC,      "dim",          0,      {INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS},       INTR_TYPE_NUMERICS,     2, 0, LANGSPEC_F77 },
    { INTR_DIM,         INTR_NAME_SPECIFIC,     "idim",         0,      {INTR_TYPE_INT, INTR_TYPE_INT},                 INTR_TYPE_INT,          2, 0, LANGSPEC_F77 },
    { INTR_DIM,         INTR_NAME_SPECIFIC,     "ddim",         0,      {INTR_TYPE_DREAL, INTR_TYPE_DREAL},             INTR_TYPE_DREAL,        2, 0, LANGSPEC_F77 },

    // DPROD (X, Y)
    { INTR_DPROD,       INTR_NAME_GENERIC,      "dprod",        0,      {INTR_TYPE_REAL, INTR_TYPE_REAL},               INTR_TYPE_DREAL,        2, -1, LANGSPEC_F77 },
    { INTR_DPROD,       INTR_NAME_SPECIFIC,     "",             0,      {INTR_TYPE_REAL, INTR_TYPE_REAL},               INTR_TYPE_DREAL,        2, -1, LANGSPEC_F77 },

    // INT (A [, KIND])
    { INTR_INT,         INTR_NAME_GENERIC,      "int",          1,      {INTR_TYPE_ALL_NUMERICS},               INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },
    { INTR_INT,         INTR_NAME_SPECIFIC_NA,  "",             1,      {INTR_TYPE_REAL},                       INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },
    { INTR_INT,         INTR_NAME_SPECIFIC_NA,  "ifix",         1,      {INTR_TYPE_REAL},                       INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },
    { INTR_INT,         INTR_NAME_SPECIFIC_NA,  "idint",        1,      {INTR_TYPE_DREAL},                      INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },

    // MAX (A1, A2 [, A3,...])
    { INTR_MAX,         INTR_NAME_GENERIC,      "max",          0,      {INTR_TYPE_NUMERICS},                   INTR_TYPE_NUMERICS,  -1, 0, LANGSPEC_F77 },

    { INTR_MAX,         INTR_NAME_SPECIFIC_NA,  "max0",         0,      {INTR_TYPE_INT},                        INTR_TYPE_INT,  -1, 0, LANGSPEC_F77 },
    { INTR_MAX,         INTR_NAME_SPECIFIC_NA,  "amax1",        0,      {INTR_TYPE_REAL},                       INTR_TYPE_REAL, -1, 0, LANGSPEC_F77 },
    { INTR_MAX,         INTR_NAME_SPECIFIC_NA,  "dmax1",        0,      {INTR_TYPE_DREAL},                      INTR_TYPE_DREAL,-1, 0, LANGSPEC_F77 },
    { INTR_MAX,         INTR_NAME_SPECIFIC_NA,  "amax0",        0,      {INTR_TYPE_INT},                        INTR_TYPE_REAL, -1, 0, LANGSPEC_F77 },
    { INTR_MAX,         INTR_NAME_SPECIFIC_NA,  "max1",         0,      {INTR_TYPE_REAL},                       INTR_TYPE_INT,  -1, 0, LANGSPEC_F77 },

    // MIN (A1, A2 [, A3,...])
    { INTR_MIN,         INTR_NAME_GENERIC,      "min",          0,      {INTR_TYPE_NUMERICS},                   INTR_TYPE_NUMERICS,  -1, 0, LANGSPEC_F77 },

    { INTR_MIN,         INTR_NAME_SPECIFIC_NA,  "min0",         0,      {INTR_TYPE_INT},                        INTR_TYPE_INT,  -1, 0, LANGSPEC_F77 },
    { INTR_MIN,         INTR_NAME_SPECIFIC_NA,  "amin1",        0,      {INTR_TYPE_REAL},                       INTR_TYPE_REAL, -1, 0, LANGSPEC_F77 },
    { INTR_MIN,         INTR_NAME_SPECIFIC_NA,  "dmin1",        0,      {INTR_TYPE_DREAL},                      INTR_TYPE_DREAL,-1, 0, LANGSPEC_F77 },
    { INTR_MIN,         INTR_NAME_SPECIFIC_NA,  "amin0",        0,      {INTR_TYPE_INT},                        INTR_TYPE_REAL, -1, 0, LANGSPEC_F77 },
    { INTR_MIN,         INTR_NAME_SPECIFIC_NA,  "min1",         0,      {INTR_TYPE_REAL},                       INTR_TYPE_INT,  -1, 0, LANGSPEC_F77 },

    // MOD (A, P)
    { INTR_MOD,         INTR_NAME_GENERIC,      "mod",          0,      {INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS},       INTR_TYPE_NUMERICS,     2, 0, LANGSPEC_F77 },
    { INTR_MOD,         INTR_NAME_SPECIFIC,     "amod",         0,      {INTR_TYPE_REAL, INTR_TYPE_REAL},               INTR_TYPE_REAL,         2, 0, LANGSPEC_F77 },
    { INTR_MOD,         INTR_NAME_SPECIFIC,     "dmod",         0,      {INTR_TYPE_DREAL, INTR_TYPE_DREAL},             INTR_TYPE_DREAL,        2, 0, LANGSPEC_F77 },

    // NINT (A [, KIND])
    { INTR_NINT,        INTR_NAME_GENERIC,      "nint",         1,      {INTR_TYPE_REAL},                       INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },
    { INTR_NINT,        INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_DREAL},                      INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },
    { INTR_NINT,        INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_ALL_NUMERICS},               INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },
    { INTR_NINT,        INTR_NAME_SPECIFIC,     "idnint",       1,      {INTR_TYPE_DREAL},                      INTR_TYPE_INT,  1, -1, LANGSPEC_F77 },

    // REAL (A [, KIND])
    { INTR_REAL,        INTR_NAME_GENERIC,      "real",         1,      {INTR_TYPE_ALL_NUMERICS},               INTR_TYPE_REAL,  1, -1, LANGSPEC_F77 },
    { INTR_REAL,        INTR_NAME_SPECIFIC_NA,  "float",        1,      {INTR_TYPE_INT},                        INTR_TYPE_REAL,  1, -1, LANGSPEC_F77 },
    { INTR_REAL,        INTR_NAME_SPECIFIC_NA,  "dfloat",       1,      {INTR_TYPE_INT},                        INTR_TYPE_DREAL, 1, -1, LANGSPEC_NONSTD },	/* non-standard */
    { INTR_REAL,        INTR_NAME_SPECIFIC_NA,  "sngl",         1,      {INTR_TYPE_DREAL},                      INTR_TYPE_REAL,  1, -1, LANGSPEC_F77 },

    // SIGN (A, B)
    { INTR_SIGN,        INTR_NAME_GENERIC,      "sign",         0,      {INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS},       INTR_TYPE_NUMERICS,     2, 0, LANGSPEC_F77 },
    { INTR_SIGN,        INTR_NAME_SPECIFIC,     "isign",        0,      {INTR_TYPE_INT, INTR_TYPE_INT},                 INTR_TYPE_INT,          2, 0, LANGSPEC_F77 },
    { INTR_SIGN,        INTR_NAME_SPECIFIC,     "dsign",        0,      {INTR_TYPE_DREAL, INTR_TYPE_DREAL},             INTR_TYPE_DREAL,        2, 0, LANGSPEC_F77 },



    /* 3. Mathematical functions */

    // ACOS (X)
    { INTR_ACOS,        INTR_NAME_GENERIC,      "acos",         0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_ACOS,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_ACOS,        INTR_NAME_SPECIFIC,     "dacos",        0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },

    // ASIN (X)
    { INTR_ASIN,        INTR_NAME_GENERIC,      "asin",         0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_ASIN,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_ASIN,        INTR_NAME_SPECIFIC,     "dasin",        0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },

    // ATAN (X)
    { INTR_ATAN,        INTR_NAME_GENERIC,      "atan",         0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_ATAN,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_ATAN,        INTR_NAME_SPECIFIC,     "datan",        0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },

    // ATAN2 (Y, X)
    { INTR_ATAN2,       INTR_NAME_GENERIC,      "atan2",        0,      {INTR_TYPE_REAL, INTR_TYPE_REAL},       INTR_TYPE_REAL,         2, 0, LANGSPEC_F77 },
    { INTR_ATAN2,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL, INTR_TYPE_DREAL},     INTR_TYPE_DREAL,        2, 0, LANGSPEC_F77 },
    { INTR_ATAN2,       INTR_NAME_SPECIFIC,     "datan2",       0,      {INTR_TYPE_DREAL, INTR_TYPE_DREAL},     INTR_TYPE_DREAL,        2, 0, LANGSPEC_F77 },

    // COS (X)
    { INTR_COS,         INTR_NAME_GENERIC,      "cos",          0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_COS,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_COS,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },
    { INTR_COS,         INTR_NAME_SPECIFIC,     "dcos",         0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_COS,         INTR_NAME_SPECIFIC,     "ccos",         0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },

    // COSH (X)
    { INTR_COSH,        INTR_NAME_GENERIC,      "cosh",         0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_COSH,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_COSH,        INTR_NAME_SPECIFIC,     "dcosh",        0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },

    // EXP (X)
    { INTR_EXP,         INTR_NAME_GENERIC,      "exp",          0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_EXP,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_EXP,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },
    { INTR_EXP,         INTR_NAME_SPECIFIC,     "dexp",         0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_EXP,         INTR_NAME_SPECIFIC,     "cexp",         0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },

    // LOG (X)
    { INTR_LOG,         INTR_NAME_GENERIC,      "log",          0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_LOG,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_LOG,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },
    { INTR_LOG,         INTR_NAME_SPECIFIC,     "alog",         0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_LOG,         INTR_NAME_SPECIFIC,     "dlog",         0,      {INTR_TYPE_DREAL},      INTR_TYPE_DREAL,        1, 0, LANGSPEC_F77 },
    { INTR_LOG,         INTR_NAME_SPECIFIC,     "clog",         0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },

    // LOG10 (X)
    { INTR_LOG10,       INTR_NAME_GENERIC,      "log10",        0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_LOG10,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_LOG10,       INTR_NAME_SPECIFIC,     "alog10",       0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_LOG10,       INTR_NAME_SPECIFIC,     "dlog10",       0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },

    // SIN (X)
    { INTR_SIN,         INTR_NAME_GENERIC,      "sin",          0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_SIN,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_SIN,         INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },
    { INTR_SIN,         INTR_NAME_SPECIFIC,     "dsin",         0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_SIN,         INTR_NAME_SPECIFIC,     "csin",         0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },

    // SINH (X)
    { INTR_SINH,        INTR_NAME_GENERIC,      "sinh",         0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_SINH,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_SINH,        INTR_NAME_SPECIFIC,     "dsinh",        0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },

    // SQRT (X)
    { INTR_SQRT,        INTR_NAME_GENERIC,      "sqrt",         0,      {INTR_TYPE_REAL},       INTR_TYPE_REAL,         1, 0, LANGSPEC_F77 },
    { INTR_SQRT,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_SQRT,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },
    { INTR_SQRT,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ALL_NUMERICS},INTR_TYPE_COMPLEX,     1, 0, LANGSPEC_F77 },
    { INTR_SQRT,        INTR_NAME_SPECIFIC,     "dsqrt",        0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_SQRT,        INTR_NAME_SPECIFIC,     "csqrt",        0,      {INTR_TYPE_COMPLEX},    INTR_TYPE_COMPLEX,      1, 0, LANGSPEC_F77 },

    // TAN (X)
    { INTR_TAN,        INTR_NAME_GENERIC,       "tan",          0,      {INTR_TYPE_REAL},      INTR_TYPE_REAL,          1, 0, LANGSPEC_F77 },
    { INTR_TAN,        INTR_NAME_GENERIC,       "",             0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_TAN,        INTR_NAME_SPECIFIC,      "dtan",         0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },

    // TANH (X)
    { INTR_TANH,        INTR_NAME_GENERIC,      "tanh",         0,      {INTR_TYPE_REAL},      INTR_TYPE_REAL,          1, 0, LANGSPEC_F77 },
    { INTR_TANH,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },
    { INTR_TANH,        INTR_NAME_SPECIFIC,     "dtanh",        0,      {INTR_TYPE_DREAL},     INTR_TYPE_DREAL,         1, 0, LANGSPEC_F77 },



    /* 4. Character functions */

    // CHAR (I [, KIND])
    { INTR_CHAR,        INTR_NAME_GENERIC,      "char",         1,      {INTR_TYPE_INT},        INTR_TYPE_CHAR,         1, -1, LANGSPEC_F77 },
    { INTR_CHAR,        INTR_NAME_SPECIFIC_NA,  "",             1,      {INTR_TYPE_INT},        INTR_TYPE_CHAR,         1, -1, LANGSPEC_F77 },

    // ICHAR (C)
    { INTR_ICHAR,       INTR_NAME_GENERIC,      "ichar",        0,      {INTR_TYPE_CHAR},       INTR_TYPE_INT,          1, -1, LANGSPEC_F77 },
    { INTR_ICHAR,       INTR_NAME_SPECIFIC_NA,  "",             0,      {INTR_TYPE_CHAR},       INTR_TYPE_INT,          1, -1, LANGSPEC_F77 },

    // INDEX (STRING, SUBSTRING [, BACK])
    { INTR_INDEX,       INTR_NAME_GENERIC,      "index",        0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_INT,  2, -1, LANGSPEC_F77 },
    { INTR_INDEX,       INTR_NAME_SPECIFIC,     "",             0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_INT,  2, -1, LANGSPEC_F77 },
    { INTR_INDEX,       INTR_NAME_SPECIFIC,     "",             0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR, INTR_TYPE_LOGICAL},    INTR_TYPE_INT,  3, -1, LANGSPEC_F77 },

    // LGE (STRING_A, STRING_B)
    { INTR_LGE,         INTR_NAME_GENERIC,      "lge",          0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },
    { INTR_LGE,         INTR_NAME_SPECIFIC_NA,  "",             0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },

    // LGT (STRING_A, STRING_B)
    { INTR_LGT,         INTR_NAME_GENERIC,      "lgt",          0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },
    { INTR_LGT,         INTR_NAME_SPECIFIC_NA,  "",             0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },

    // LLE (STRING_A, STRING_B)
    { INTR_LLE,         INTR_NAME_GENERIC,      "lle",          0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },
    { INTR_LLE,         INTR_NAME_SPECIFIC_NA,  "",             0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },

    // LLT (STRING_A, STRING_B)
    { INTR_LLT,         INTR_NAME_GENERIC,      "llt",          0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },
    { INTR_LLT,         INTR_NAME_SPECIFIC_NA,  "",             0,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F77 },



    /* 5. Character inquiry function */

    // LEN (STRING)
    { INTR_LEN,         INTR_NAME_GENERIC,      "len",          0,      {INTR_TYPE_CHAR},       INTR_TYPE_INT,  1, -6, LANGSPEC_F77 },

    /* 6. Fortran77 non-standard */
    { INTR_LOC,         INTR_NAME_GENERIC,      "loc",          0,      {INTR_TYPE_ANY},       INTR_TYPE_INT,  1, -9, LANGSPEC_NONSTD },


    /*
     * Fortran90 intrinsic
     */

    /* 2. Numeric functions */

    /* Entries of ceiling and floor for DREAL added (by Hitoshi Murai). */

    // CEILING (A  [, KIND])
    { INTR_CEILING,     INTR_NAME_GENERIC,      "ceiling",      1,      {INTR_TYPE_REAL},       INTR_TYPE_INT,  1, -1, LANGSPEC_F90 },
    { INTR_CEILING,     INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_DREAL},      INTR_TYPE_INT,  1, -1, LANGSPEC_F90 },

    // FLOOR (A  [, KIND])
    { INTR_FLOOR,       INTR_NAME_GENERIC,      "floor",        1,      {INTR_TYPE_REAL},       INTR_TYPE_INT,  1, -1, LANGSPEC_F90 },
    { INTR_FLOOR,       INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_DREAL},      INTR_TYPE_INT,  1, -1, LANGSPEC_F90 },

    // MODULO (A, P)
    { INTR_MODULO,         INTR_NAME_GENERIC,   "modulo",       0,      {INTR_TYPE_NUMERICS, INTR_TYPE_NUMERICS},       INTR_TYPE_NUMERICS,     2, 0, LANGSPEC_F90 },



    /* 4. Character functions */

    // ACHAR (I)
    { INTR_ACHAR,       INTR_NAME_GENERIC,      "achar",        0,      {INTR_TYPE_INT},        INTR_TYPE_CHAR, 1, -1, LANGSPEC_F90 },

    // ADJUSTL (STRING)
    { INTR_ADJUSTL,     INTR_NAME_GENERIC,      "adjustl",      0,      {INTR_TYPE_CHAR},       INTR_TYPE_CHAR, 1, 0, LANGSPEC_F90 },

    // ADJUSTR (STRING)
    { INTR_ADJUSTR,     INTR_NAME_GENERIC,      "adjustr",      0,      {INTR_TYPE_CHAR},       INTR_TYPE_CHAR, 1, 0, LANGSPEC_F90 },

    // IACHAR (C)
    { INTR_IACHAR,      INTR_NAME_GENERIC,      "iachar",       0,      {INTR_TYPE_CHAR},       INTR_TYPE_INT,  1, -1, LANGSPEC_F90 },

    // LEN_TRIM (STRING)
    { INTR_LEN_TRIM,    INTR_NAME_GENERIC,      "len_trim",     0,      {INTR_TYPE_CHAR},       INTR_TYPE_INT,  1, -1, LANGSPEC_F90 },

    // REPEAT (STRING, NCOPIES)
    { INTR_REPEAT,  INTR_NAME_GENERIC,          "repeat",       0,      {INTR_TYPE_CHAR, INTR_TYPE_INT},       INTR_TYPE_INT,  2, 0, LANGSPEC_F90 },

    // SCAN (STRING, SET [, BACK, KIND ])
    { INTR_SCAN,        INTR_NAME_GENERIC,      "scan",         1,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_INT,  2, -1, LANGSPEC_F90 },
    { INTR_SCAN,        INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR, INTR_TYPE_LOGICAL},    INTR_TYPE_INT,  3, -1, LANGSPEC_F90 },

    // TRIM (STRING)
    { INTR_TRIM,        INTR_NAME_GENERIC,      "trim",         0,      {INTR_TYPE_CHAR},       INTR_TYPE_CHAR, 1, 0, LANGSPEC_F90 },

    // VERIFY (STRING, SET [, BACK, KIND])
    { INTR_VERIFY,      INTR_NAME_GENERIC,      "verify",       1,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR},                       INTR_TYPE_INT,      2, -1, LANGSPEC_F90 },
    { INTR_VERIFY,      INTR_NAME_GENERIC,      "",             1,      {INTR_TYPE_CHAR, INTR_TYPE_CHAR, INTR_TYPE_LOGICAL},    INTR_TYPE_INT,      3, -1, LANGSPEC_F90 },



    /* 6. Kind functions */

    // KIND (X)
    { INTR_KIND,        INTR_NAME_GENERIC,      "kind",         0,      {INTR_TYPE_ANY},                        INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // SELECTED_INT_KIND (R)
    { INTR_SELECTED_INT_KIND,   INTR_NAME_GENERIC,      "selected_int_kind",    0,      {INTR_TYPE_INT},        INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // SELECTED_REAL_KIND ([P, R])
    { INTR_SELECTED_REAL_KIND,  INTR_NAME_GENERIC,      "selected_real_kind",   0,      {INTR_TYPE_NONE},               INTR_TYPE_INT,  0, -6, LANGSPEC_F90 },
    { INTR_SELECTED_REAL_KIND,  INTR_NAME_GENERIC,      "",                     0,      {INTR_TYPE_INT},                INTR_TYPE_INT,  1, 0, LANGSPEC_F90 },
    { INTR_SELECTED_REAL_KIND,  INTR_NAME_GENERIC,      "",                     0,      {INTR_TYPE_INT, INTR_TYPE_INT}, INTR_TYPE_INT,  2, 0, LANGSPEC_F90 },



    /* 7. Logical function */

    // LOGICAL (L [, KIND])
    { INTR_LOGICAL,     INTR_NAME_GENERIC,      "logical",      1,      {INTR_TYPE_LOGICAL},            INTR_TYPE_LOGICAL,  1, -1, LANGSPEC_F90 },



    /* 8. Numeric inquiry functions */

    // DIGITS (X)
    { INTR_DIGITS,      INTR_NAME_GENERIC,      "digits",       0,      {INTR_TYPE_NUMERICS_ARRAY},     INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_DIGITS,      INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_NUMERICS},           INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // EPSILON (X)
    { INTR_EPSILON,     INTR_NAME_GENERIC,      "epsilon",      0,      {INTR_TYPE_REAL_ARRAY},         INTR_TYPE_REAL,         1, -6, LANGSPEC_F90 },
    { INTR_EPSILON,     INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL_ARRAY},        INTR_TYPE_DREAL,        1, -6, LANGSPEC_F90 },
    { INTR_EPSILON,     INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL},               INTR_TYPE_REAL,         1, 0, LANGSPEC_F90 },
    { INTR_EPSILON,     INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},              INTR_TYPE_DREAL,        1, 0, LANGSPEC_F90 },

    // HUGE (X)
    { INTR_HUGE,        INTR_NAME_GENERIC,      "huge",         0,      {INTR_TYPE_INT_ARRAY},          INTR_TYPE_INT,          1, -6, LANGSPEC_F90 },
    { INTR_HUGE,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL_ARRAY},         INTR_TYPE_REAL,         1, -6, LANGSPEC_F90 },
    { INTR_HUGE,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL_ARRAY},        INTR_TYPE_DREAL,        1, -6, LANGSPEC_F90 },
    { INTR_HUGE,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_INT},                INTR_TYPE_INT,          1, 0, LANGSPEC_F90 },
    { INTR_HUGE,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL},               INTR_TYPE_REAL,         1, 0, LANGSPEC_F90 },
    { INTR_HUGE,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},              INTR_TYPE_DREAL,        1, 0, LANGSPEC_F90 },

    // MAXEXPONENT (X)
    { INTR_MAXEXPONENT, INTR_NAME_GENERIC,      "maxexponent",  0,      {INTR_TYPE_REAL_ARRAY},         INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_MAXEXPONENT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL_ARRAY},        INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_MAXEXPONENT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL},               INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_MAXEXPONENT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},              INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // MINEXPONENT (X)
    { INTR_MINEXPONENT, INTR_NAME_GENERIC,      "minexponent",  0,      {INTR_TYPE_REAL_ARRAY},         INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_MINEXPONENT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL_ARRAY},        INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_MINEXPONENT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL},               INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_MINEXPONENT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},              INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // PRECISION (X)
    { INTR_PRECISION,   INTR_NAME_GENERIC,      "precision",    0,      {INTR_TYPE_ALL_REAL_ARRAY},     INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_PRECISION,   INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ALL_COMPLEX_ARRAY},  INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_PRECISION,   INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ALL_REAL},           INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_PRECISION,   INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ALL_COMPLEX},        INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // RADIX (X)
    { INTR_RADIX,       INTR_NAME_GENERIC,      "radix",        0,      {INTR_TYPE_ALL_REAL_ARRAY},     INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_RADIX,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_INT_ARRAY},          INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_RADIX,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ALL_REAL},           INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_RADIX,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_INT},                INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // RANGE (X)
    { INTR_RANGE,       INTR_NAME_GENERIC,      "range",        0,      {INTR_TYPE_ALL_NUMERICS_ARRAY},      INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_RANGE,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ALL_NUMERICS},            INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // TINY (X)
    { INTR_TINY,        INTR_NAME_GENERIC,      "tiny",         0,      {INTR_TYPE_REAL_ARRAY},         INTR_TYPE_REAL,         1, -6, LANGSPEC_F90 },
    { INTR_TINY,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL_ARRAY},        INTR_TYPE_DREAL,        1, -6, LANGSPEC_F90 },
    { INTR_TINY,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL},               INTR_TYPE_REAL,         1, 0, LANGSPEC_F90 },
    { INTR_TINY,        INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},              INTR_TYPE_DREAL,        1, 0, LANGSPEC_F90 },



    /* 9. Bit inquiry function */

    // BIT_SIZE (I)
    { INTR_BIT_SIZE,    INTR_NAME_GENERIC,      "bit_size",     0,      {INTR_TYPE_INT},                INTR_TYPE_INT,  1, 0, LANGSPEC_F90 },



    /* 10. Bit manipulation functions */

    // BTEST (I, POS)
    { INTR_BTEST,       INTR_NAME_GENERIC,      "btest",        0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_LOGICAL,  2, -1, LANGSPEC_F90 | LANGSPEC_NONSTD },

    // IAND (I, J)
    { INTR_IAND,        INTR_NAME_GENERIC,      "iand",         0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  2, 0, LANGSPEC_F90 | LANGSPEC_NONSTD },

    // IBCLR (I, POS)
    { INTR_IBCLR,       INTR_NAME_GENERIC,      "ibclr",        0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  2, 0, LANGSPEC_F90 },

    // IBITS (I, POS, LEN)
    { INTR_IBITS,       INTR_NAME_GENERIC,      "ibits",        0,      {INTR_TYPE_INT, INTR_TYPE_INT, INTR_TYPE_INT},  INTR_TYPE_INT,  3, 0, LANGSPEC_F90 },

    // IBSET (I, POS)
    { INTR_IBSET,       INTR_NAME_GENERIC,      "ibset",        0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  2, 0, LANGSPEC_F90 },

    // IEOR (I, J)
    { INTR_IEOR,        INTR_NAME_GENERIC,      "ieor",         0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  2, 0, LANGSPEC_F90 | LANGSPEC_NONSTD },

    // IOR (I, J)
    { INTR_IOR,         INTR_NAME_GENERIC,      "ior",          0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  2, 0, LANGSPEC_F90 },

    // ISHFT (I, SHIFT)
    { INTR_ISHFT,       INTR_NAME_GENERIC,      "ishft",        0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  2, 0, LANGSPEC_F90 },

    // ISHFTC (I, SHIFT [, SIZE])
    { INTR_ISHFTC,       INTR_NAME_GENERIC,      "ishftc",      0,      {INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  2, 0, LANGSPEC_F90 },
    { INTR_ISHFTC,       INTR_NAME_GENERIC,      "ishftc",      0,      {INTR_TYPE_INT, INTR_TYPE_INT, INTR_TYPE_INT},         INTR_TYPE_INT,  3, 0, LANGSPEC_F90 },

    // NOT (I)
    { INTR_NOT,          INTR_NAME_GENERIC,      "not",         0,      {INTR_TYPE_INT},         INTR_TYPE_INT,  1, 0, LANGSPEC_F90 },



    /* 11. Transfer function */

    // TRANSFER (SOURCE, MOLD [, SIZE])
    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "transfer",     0,      {INTR_TYPE_ANY_ARRAY, INTR_TYPE_ANY_ARRAY},             INTR_TYPE_ANY_ARRAY, 2, 1, LANGSPEC_F90 },
    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ANY_ARRAY, INTR_TYPE_ANY},                   INTR_TYPE_ANY,  2, 1, LANGSPEC_F90 },
    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ANY, INTR_TYPE_ANY_ARRAY},                   INTR_TYPE_ANY_ARRAY,    2, 1, LANGSPEC_F90 },
    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ANY, INTR_TYPE_ANY},                         INTR_TYPE_ANY,  2, 1, LANGSPEC_F90 },

    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ANY_ARRAY, INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT},      INTR_TYPE_ANY_ARRAY, 3, 1, LANGSPEC_F90 },
    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ANY_ARRAY, INTR_TYPE_ANY, INTR_TYPE_INT},    INTR_TYPE_ANY,  3, 1, LANGSPEC_F90 },
    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ANY, INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT},    INTR_TYPE_ANY_ARRAY, 3, 1, LANGSPEC_F90 },
    { INTR_TRANSFER,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_ANY, INTR_TYPE_ANY, INTR_TYPE_INT},          INTR_TYPE_ANY,  3, 1, LANGSPEC_F90 },



    /* 12. Floating-point manipulation functions */

    // EXPONENT (X)
    { INTR_EXPONENT,    INTR_NAME_GENERIC,      "exponent",     0,      {INTR_TYPE_REAL},              INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },
    { INTR_EXPONENT,    INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},             INTR_TYPE_INT,  1, -6, LANGSPEC_F90 },

    // FRACTION (X)
    { INTR_FRACTION,    INTR_NAME_GENERIC,      "fraction",     0,      {INTR_TYPE_ALL_REAL},          INTR_TYPE_ALL_REAL,  1, 0, LANGSPEC_F90 },

    // NEAREST (X, S)
    { INTR_NEAREST,     INTR_NAME_GENERIC,      "nearest",      0,      {INTR_TYPE_ALL_REAL, INTR_TYPE_ALL_REAL},        INTR_TYPE_ALL_REAL,         2, 0, LANGSPEC_F90 },

    // RRSPACING (X)
    { INTR_RRSPACING,   INTR_NAME_GENERIC,      "rrspacing",    0,      {INTR_TYPE_ALL_REAL},          INTR_TYPE_ALL_REAL,  1, 0, LANGSPEC_F90 },

    // SCALE (X, I)
    { INTR_SCALE,       INTR_NAME_GENERIC,      "scale",        0,      {INTR_TYPE_REAL, INTR_TYPE_INT},        INTR_TYPE_REAL,         2, 0, LANGSPEC_F90 },
    { INTR_SCALE,       INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL, INTR_TYPE_INT},       INTR_TYPE_DREAL,        2, 0, LANGSPEC_F90 },

    // SET_EXPONENT (X, I)
    { INTR_SET_EXPONENT, INTR_NAME_GENERIC,     "set_exponent", 0,      {INTR_TYPE_ALL_REAL, INTR_TYPE_INT},        INTR_TYPE_ALL_REAL,         2, 0, LANGSPEC_F90 },

    // SPACING (X)
    { INTR_SPACING,     INTR_NAME_GENERIC,      "spacing",      0,      {INTR_TYPE_REAL},               INTR_TYPE_REAL,         1, 0, LANGSPEC_F90 },
    { INTR_SPACING,     INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL},              INTR_TYPE_DREAL,        1, 0, LANGSPEC_F90 },



    /* 13. Vector and matrix multiply functions */

    // DOT_PRODUCT (VECTOR_A, VECTOR_B)
    { INTR_DOT_PRODUCT, INTR_NAME_GENERIC,      "dot_product",  0,      {INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_ALL_NUMERICS_ARRAY},             INTR_TYPE_ALL_NUMERICS,    2, -6, LANGSPEC_F90 },
    { INTR_DOT_PRODUCT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_LOGICAL_ARRAY},                       INTR_TYPE_LOGICAL,         2, -6, LANGSPEC_F90 },
#if 0
    { INTR_DOT_PRODUCT, INTR_NAME_GENERIC,      "dot_product",  0,      {INTR_TYPE_INT_ARRAY, INTR_TYPE_INT_ARRAY},             INTR_TYPE_INT_ARRAY,    2, 0, LANGSPEC_F90 },
    { INTR_DOT_PRODUCT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL_ARRAY, INTR_TYPE_REAL_ARRAY},           INTR_TYPE_REAL_ARRAY,   2, 0, LANGSPEC_F90 },
    { INTR_DOT_PRODUCT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL_ARRAY, INTR_TYPE_DREAL_ARRAY},         INTR_TYPE_DREAL_ARRAY, 2, 0, LANGSPEC_F90 },
    { INTR_DOT_PRODUCT, INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX_ARRAY, INTR_TYPE_COMPLEX_ARRAY},     INTR_TYPE_COMPLEX_ARRAY,        2, 0, LANGSPEC_F90 },
#endif

    // MATMUL (MATRIX_A, MATRIX_B)
    { INTR_MATMUL,      INTR_NAME_GENERIC,      "matmul",             0,      {INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_ALL_NUMERICS_ARRAY},           INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,   2, -1, LANGSPEC_F90 },
    { INTR_MATMUL,      INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_LOGICAL_ARRAY,INTR_TYPE_LOGICAL_ARRAY},      INTR_TYPE_LOGICAL_DYNAMIC_ARRAY,        2, -1, LANGSPEC_F90 },

#if 0
    { INTR_MATMUL,      INTR_NAME_GENERIC,      "matmul",       0,      {INTR_TYPE_INT_ARRAY, INTR_TYPE_INT_ARRAY},             INTR_TYPE_INT_ARRAY,    2, 1, LANGSPEC_F90 },
    { INTR_MATMUL,      INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_REAL_ARRAY, INTR_TYPE_REAL_ARRAY},           INTR_TYPE_REAL_ARRAY,   2, 1, LANGSPEC_F90 },
    { INTR_MATMUL,      INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_DREAL_ARRAY, INTR_TYPE_DREAL_ARRAY},         INTR_TYPE_REAL_ARRAY,   2, 1, LANGSPEC_F90 },
    { INTR_MATMUL,      INTR_NAME_GENERIC,      "",             0,      {INTR_TYPE_COMPLEX_ARRAY,INTR_TYPE_COMPLEX_ARRAY},      INTR_TYPE_COMPLEX_ARRAY,        2, 1, LANGSPEC_F90 },
#endif


    /* 14. Array reduction functions */

    //ALL (MASK [, DIM])
    { INTR_ALL,         INTR_NAME_GENERIC,      "all",          0,      { INTR_TYPE_LOGICAL_ARRAY },                    INTR_TYPE_LOGICAL,                      1, -1, LANGSPEC_F90 },
    { INTR_ALL,         INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_INT },     INTR_TYPE_LOGICAL_DYNAMIC_ARRAY,        2, -1, LANGSPEC_F90 },

    //ANY (MASK [, DIM])
    { INTR_ANY,         INTR_NAME_GENERIC,      "any",          0,      { INTR_TYPE_LOGICAL_ARRAY },                        INTR_TYPE_LOGICAL,                      1, -1, LANGSPEC_F90 },
    { INTR_ANY,         INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_INT },         INTR_TYPE_LOGICAL_DYNAMIC_ARRAY,        2, -1, LANGSPEC_F90 },

    // COUNT (MASK [, DIM])
    { INTR_COUNT,       INTR_NAME_GENERIC,      "count",        0,      { INTR_TYPE_LOGICAL_ARRAY },                    INTR_TYPE_INT,                  1, -1, LANGSPEC_F90 },
    { INTR_COUNT,       INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_INT},      INTR_TYPE_INT_DYNAMIC_ARRAY,    2, -1, LANGSPEC_F90 },

    // MAXVAL (ARRAY [, MASK])
    { INTR_MAXVAL,      INTR_NAME_GENERIC,      "maxval",       0,      { INTR_TYPE_ALL_NUMERICS_ARRAY },                       INTR_TYPE_ALL_NUMERICS,             1, -2, LANGSPEC_F90 },
    { INTR_MAXVAL,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY,  INTR_TYPE_LOGICAL_ARRAY }, INTR_TYPE_ALL_NUMERICS,             2, -2, LANGSPEC_F90 },
    // MAXVAL (ARRAY, DIM [, MASK])
    { INTR_MAXVAL,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_INT },        INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,   2, -1, LANGSPEC_F90 },
    { INTR_MAXVAL,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },   INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,   3, -1, LANGSPEC_F90 },

    // MINVAL (ARRAY [, MASK])
    { INTR_MINVAL,      INTR_NAME_GENERIC,      "minval",       0,      { INTR_TYPE_NUMERICS_ARRAY },                           INTR_TYPE_NUMERICS,             1, -2, LANGSPEC_F90 },
    { INTR_MINVAL,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_NUMERICS_ARRAY,  INTR_TYPE_LOGICAL_ARRAY }, INTR_TYPE_NUMERICS,             2, -2, LANGSPEC_F90 },
    // MINVAL (ARRAY, DIM [, MASK])
    { INTR_MINVAL,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_NUMERICS_ARRAY, INTR_TYPE_INT },                            INTR_TYPE_NUMERICS_DYNAMIC_ARRAY,       2, -1, LANGSPEC_F90 },
    { INTR_MINVAL,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_NUMERICS_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },   INTR_TYPE_NUMERICS_DYNAMIC_ARRAY,       3, -1, LANGSPEC_F90 },

    // PRODUCT (ARRAY [, MASK])
    { INTR_PRODUCT,     INTR_NAME_GENERIC,      "product",      0,      { INTR_TYPE_ALL_NUMERICS_ARRAY },                               INTR_TYPE_ALL_NUMERICS,         1, -2, LANGSPEC_F90 },
    { INTR_PRODUCT,     INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY,  INTR_TYPE_LOGICAL_ARRAY },     INTR_TYPE_ALL_NUMERICS,         2, -2, LANGSPEC_F90 },
    // PRODUCT (ARRAY, DIM [, MASK])
    { INTR_PRODUCT,     INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_INT },                                INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,   2, -1, LANGSPEC_F90 },
    { INTR_PRODUCT,     INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },       INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,   3, -1, LANGSPEC_F90 },

    // SUM (ARRAY [, MASK])
    { INTR_SUM, INTR_NAME_GENERIC,      "sum",          0,      { INTR_TYPE_ALL_NUMERICS_ARRAY },                               INTR_TYPE_ALL_NUMERICS,         1, -2, LANGSPEC_F90 },
    { INTR_SUM, INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY,  INTR_TYPE_LOGICAL_ARRAY },     INTR_TYPE_ALL_NUMERICS,         2, -2, LANGSPEC_F90 },
    // SUM (ARRAY, DIM [, MASK])
    { INTR_SUM, INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_INT },                                INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,   2, -1, LANGSPEC_F90 },
    { INTR_SUM, INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ALL_NUMERICS_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },       INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY,   3, -1, LANGSPEC_F90 },



    /* 15. Array inquiry functions */

    // ALLOCATED (ARRAY)
    { INTR_ALLOCATED,   INTR_NAME_GENERIC,      "allocated",    0,      { INTR_TYPE_ANY_ARRAY_ALLOCATABLE },                    INTR_TYPE_LOGICAL,      1, -6, LANGSPEC_F90 },

    // LBOUND (ARRAY [, DIM])
    { INTR_LBOUND,      INTR_NAME_GENERIC,      "lbound",       0,      { INTR_TYPE_ANY_ARRAY },                                INTR_TYPE_INT_ARRAY,    1, -3, LANGSPEC_F90 },
    { INTR_LBOUND,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT },                 INTR_TYPE_INT,          2, -1, LANGSPEC_F90 },

    // SHAPE (SOURCE)
    { INTR_SHAPE,       INTR_NAME_GENERIC,      "shape",        0,      { INTR_TYPE_ANY_ARRAY },                                INTR_TYPE_INT_ARRAY,    1, -3, LANGSPEC_F90 },

    // SIZE (ARRAY [, DIM])
    { INTR_SIZE,        INTR_NAME_GENERIC,      "size",         0,      { INTR_TYPE_ANY_ARRAY },                                INTR_TYPE_INT,  1, -1, LANGSPEC_F90 },
    { INTR_SIZE,        INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT },                 INTR_TYPE_INT,  2, -1, LANGSPEC_F90 },

    // UBOUND (ARRAY [, DIM])
    { INTR_UBOUND,      INTR_NAME_GENERIC,      "ubound",       0,      { INTR_TYPE_ANY_ARRAY },                                INTR_TYPE_INT_ARRAY,    1, -3, LANGSPEC_F90 },
    { INTR_UBOUND,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT },                 INTR_TYPE_INT,          2, -1, LANGSPEC_F90 },



    /* 16. Array construction functions */

    // MERGE (TSOURCE, FSOURCE, MASK)
    { INTR_MERGE,       INTR_NAME_GENERIC,      "merge",        0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_ANY_ARRAY, INTR_TYPE_LOGICAL },        INTR_TYPE_ANY_ARRAY,    3, 0, LANGSPEC_F90 },
    { INTR_MERGE,       INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY, INTR_TYPE_ANY, INTR_TYPE_LOGICAL },                    INTR_TYPE_ANY,          3, 0, LANGSPEC_F90 },

    // PACK (ARRAY, MASK [, VECTOR])
    //    { INTR_PACK,        INTR_NAME_GENERIC,      "pack",         0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_LOGICAL_ARRAY, },              INTR_TYPE_ANY_DYNAMIC_ARRAY,            2, -1, LANGSPEC_F90 },
    //    { INTR_PACK,        INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_ANY_ARRAY},              INTR_TYPE_ANY_DYNAMIC_ARRAY,            3, -1, LANGSPEC_F90 },
    { INTR_PACK,        INTR_NAME_GENERIC,      "pack",         0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_LOGICAL, },              INTR_TYPE_ANY_DYNAMIC_ARRAY,            2, -1, LANGSPEC_F90 },
    { INTR_PACK,        INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_LOGICAL, INTR_TYPE_ANY_ARRAY},              INTR_TYPE_ANY_DYNAMIC_ARRAY,            3, -1, LANGSPEC_F90 },

    // SPREAD (SOURCE, DIM, NCOPIES)
    { INTR_SPREAD,      INTR_NAME_GENERIC,      "spread",       0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT, INTR_TYPE_INT },          INTR_TYPE_ANY_DYNAMIC_ARRAY,    3, -1, LANGSPEC_F90 },
    { INTR_SPREAD,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT },                INTR_TYPE_ANY_DYNAMIC_ARRAY,    3, -1, LANGSPEC_F90 },

    // UNPACK (VECTOR, MASK, FIELD)
    { INTR_UNPACK,      INTR_NAME_GENERIC,      "unpack",       0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_ANY_ARRAY},              INTR_TYPE_ANY_DYNAMIC_ARRAY,            3, -1, LANGSPEC_F90 },
    { INTR_UNPACK,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_ANY},              INTR_TYPE_ANY_DYNAMIC_ARRAY,            3, -1, LANGSPEC_F90 },



    /* 17. Array reshape function */

    // RESHAPE (SOURCE, SHAPE [, PAD, ORDER])
    { INTR_RESHAPE,     INTR_NAME_GENERIC,      "reshape",      0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY },   INTR_TYPE_ANY_DYNAMIC_ARRAY,    2, -1, LANGSPEC_F90 },
    { INTR_RESHAPE,     INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_INT_ARRAY },      INTR_TYPE_ANY_DYNAMIC_ARRAY,    3, -1, LANGSPEC_F90 },
    { INTR_RESHAPE,     INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_ANY_ARRAY },      INTR_TYPE_ANY_DYNAMIC_ARRAY,    3, -1, LANGSPEC_F90 },
    { INTR_RESHAPE,     INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY }, INTR_TYPE_ANY_DYNAMIC_ARRAY,    4, -1, LANGSPEC_F90 },



    /* 18. Array manipulation functions */

    // CSHIFT (ARRAY, SHIFT [, DIM])
    { INTR_CSHIFT,      INTR_NAME_GENERIC,      "cshift",       0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY },                   INTR_TYPE_ANY_ARRAY,    2, 0, LANGSPEC_F90 },
    { INTR_CSHIFT,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT },                         INTR_TYPE_ANY_ARRAY,    2, 0, LANGSPEC_F90 },
    { INTR_CSHIFT,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_INT },    INTR_TYPE_ANY_ARRAY,    3, 0, LANGSPEC_F90 },
    { INTR_CSHIFT,      INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT, INTR_TYPE_INT },          INTR_TYPE_ANY_ARRAY,    3, 0, LANGSPEC_F90 },

    // TRANSPOSE (MATRIX)
    { INTR_TRANSPOSE,   INTR_NAME_GENERIC,      "transpose",    0,      { INTR_TYPE_ANY_ARRAY },                                INTR_TYPE_ANY_ARRAY,    1, -4, LANGSPEC_F90 },



    /* 19. Array location functions */

    // MINLOC (ARRAY [, MASK, KIND])
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "minloc",       1,      { INTR_TYPE_INT_ARRAY },                                INTR_TYPE_INT_ARRAY,    1, -3, LANGSPEC_F90 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY },                           INTR_TYPE_INT_ARRAY,    1, -3, LANGSPEC_F90 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_LOGICAL_ARRAY },       INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F90 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_LOGICAL_ARRAY },  INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F90 },

    // MINLOC (ARRAY [, DIM, MASK, KIND]) (Fortran 95)
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_INT },       INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F95 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_INT },  INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F95 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_INT },       INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },       INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_INT },  INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },
    { INTR_MINLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },  INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },

    // MAXLOC (ARRAY [, MASK, KIND])
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "maxloc",       1,      { INTR_TYPE_INT_ARRAY },                                INTR_TYPE_INT_ARRAY,    1, -3, LANGSPEC_F90 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY },                           INTR_TYPE_INT_ARRAY,    1, -3, LANGSPEC_F90 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_LOGICAL_ARRAY },       INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F90 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_LOGICAL_ARRAY },  INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F90 },

    // MAXLOC (ARRAY [, DIM, MASK, KIND]) (Fortran 95)
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_INT },       INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F95 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_INT },  INTR_TYPE_INT_ARRAY,    2, -3, LANGSPEC_F95 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_INT },       INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_INT_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },       INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_LOGICAL_ARRAY, INTR_TYPE_INT },  INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },
    { INTR_MAXLOC,      INTR_NAME_GENERIC,      "",             1,      { INTR_TYPE_ALL_REAL_ARRAY, INTR_TYPE_INT, INTR_TYPE_LOGICAL_ARRAY },  INTR_TYPE_INT_ARRAY,    3, -3, LANGSPEC_F95 },


    /* 20. Pointer association status functions */

    // ASSOCIATED (POINTER [, TARGET])
    { INTR_ASSOCIATED,  INTR_NAME_GENERIC,      "associated",   0,      { INTR_TYPE_POINTER },                          INTR_TYPE_LOGICAL,      1, -6, LANGSPEC_F90 },
    { INTR_ASSOCIATED,  INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_POINTER, INTR_TYPE_POINTER },       INTR_TYPE_LOGICAL,      2, -6, LANGSPEC_F90 },
    { INTR_ASSOCIATED,  INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_POINTER, INTR_TYPE_TARGET },        INTR_TYPE_LOGICAL,      2, -6, LANGSPEC_F90 },


    /* 21. Intrinsic subroutines */

    // DATE_AND_TIME ([DATE, TIME, ZONE, VALUES])
    { INTR_DATE_AND_TIME,       INTR_NAME_GENERIC,      "date_and_time",        0,      { INTR_TYPE_NONE },                                     INTR_TYPE_NONE, 0, -1, LANGSPEC_F90 },
    { INTR_DATE_AND_TIME,       INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_CHAR },                                     INTR_TYPE_NONE, 1, -1, LANGSPEC_F90 },
    { INTR_DATE_AND_TIME,       INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_CHAR, INTR_TYPE_CHAR },                     INTR_TYPE_NONE, 2, -1, LANGSPEC_F90 },
    { INTR_DATE_AND_TIME,       INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_CHAR, INTR_TYPE_CHAR, INTR_TYPE_CHAR },     INTR_TYPE_NONE, 3, -1, LANGSPEC_F90 },
    { INTR_DATE_AND_TIME,       INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_CHAR, INTR_TYPE_CHAR, INTR_TYPE_CHAR, INTR_TYPE_INT_ARRAY },        INTR_TYPE_NONE, 4, -1, LANGSPEC_F90 },

    // MVBITS (FROM, FROMPOS, LEN, TO, TOPOS)
    { INTR_MVBITS,              INTR_NAME_GENERIC,      "mvbits",               0,      { INTR_TYPE_INT, INTR_TYPE_INT, INTR_TYPE_INT, INTR_TYPE_INT, INTR_TYPE_INT },        INTR_TYPE_NONE, 5, -1, LANGSPEC_F90 },

    // RANDOM_NUMBER (HARVEST)
    { INTR_RANDOM_NUMBER,      INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_ALL_REAL_ARRAY },                           INTR_TYPE_NONE, 1, -1, LANGSPEC_F90 },
    { INTR_RANDOM_NUMBER,      INTR_NAME_GENERIC,      "random_number",        0,      { INTR_TYPE_ALL_REAL },                                 INTR_TYPE_NONE, 1, -1, LANGSPEC_F90 },

    // RANDOM_SEED ([SIZE, PUT, GET])
    { INTR_RANDOM_SEED,         INTR_NAME_GENERIC,      "random_seed",          0,      { INTR_TYPE_NONE },                                     INTR_TYPE_NONE, 0, -1, LANGSPEC_F90 },
    { INTR_RANDOM_SEED,         INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_INT },                                      INTR_TYPE_NONE, 1, -1, LANGSPEC_F90 },
    { INTR_RANDOM_SEED,         INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_INT, INTR_TYPE_INT_ARRAY },                  INTR_TYPE_NONE, 2, -1, LANGSPEC_F90 },
    { INTR_RANDOM_SEED,         INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_INT, INTR_TYPE_INT_ARRAY, INTR_TYPE_INT_ARRAY },     INTR_TYPE_NONE, 3, -1, LANGSPEC_F90 },



    // SYSTEM_CLOCK ([COUNT, COUNT_RATE, COUNT_MAX])
    { INTR_SYSTEM_CLOCK,        INTR_NAME_GENERIC,      "system_clock",         0,      { INTR_TYPE_NONE },                             INTR_TYPE_NONE, 0, -6, LANGSPEC_F90 },
    { INTR_SYSTEM_CLOCK,        INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_INT },                              INTR_TYPE_NONE, 1, -6, LANGSPEC_F90 },
    { INTR_SYSTEM_CLOCK,        INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_INT, INTR_TYPE_INT },               INTR_TYPE_NONE, 2, -6, LANGSPEC_F90 },
    { INTR_SYSTEM_CLOCK,        INTR_NAME_GENERIC,      "",                     0,      { INTR_TYPE_INT, INTR_TYPE_INT, INTR_TYPE_INT },        INTR_TYPE_NONE, 3, -6, LANGSPEC_F90 },

    /*
     * Fortran90 intrinsics
     */

    /* 1. Argument presence inquiry function */

    // PRESENT (A)

    { INTR_PRESENT,    INTR_NAME_GENERIC,      "present",       0,      { INTR_TYPE_ANY_OPTIONAL },                            INTR_TYPE_LOGICAL, 1, -6, LANGSPEC_F95 },

    /* 18. Array manipulation functions */

    // EOSHIFT (ARRAY, SHIFT [, BOUNDARY, DIM])

    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "eoshift",       0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY },                               INTR_TYPE_ANY_ARRAY, 2, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT },                                     INTR_TYPE_ANY_ARRAY, 2, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_ANY_ARRAY },          INTR_TYPE_ANY_ARRAY, 3, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_ANY },                INTR_TYPE_ANY_ARRAY, 3, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT, INTR_TYPE_ANY_ARRAY },                INTR_TYPE_ANY_ARRAY, 3, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT, INTR_TYPE_ANY },                      INTR_TYPE_ANY_ARRAY, 3, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT }, INTR_TYPE_ANY_ARRAY, 4, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_ANY, INTR_TYPE_INT }, INTR_TYPE_ANY_ARRAY, 4, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT, INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT }, INTR_TYPE_ANY_ARRAY, 4, 0, LANGSPEC_F95 },
    { INTR_EOSHIFT,    INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY, INTR_TYPE_INT, INTR_TYPE_ANY, INTR_TYPE_INT },       INTR_TYPE_ANY_ARRAY, 4, 0, LANGSPEC_F95 },

    /*
     * Fortran95 intrinsic
     */

    /* 20. Pointer association status functions */

    // NULL ([MOLD])
    { INTR_NULL,       INTR_NAME_GENERIC,      "null",          0,      { INTR_TYPE_NONE },                     INTR_TYPE_LHS, 0, -7, LANGSPEC_F95 },
    { INTR_NULL,       INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_ANY_ARRAY_ALLOCATABLE },    INTR_TYPE_TARGET, 1, 0,  LANGSPEC_F95 },
    { INTR_NULL,       INTR_NAME_GENERIC,      "",              0,      { INTR_TYPE_POINTER },                  INTR_TYPE_TARGET, 1, 0,  LANGSPEC_F95 },

    /* 21. Intrinsic subroutines */

    // CPU_TIME (TIME)
    { INTR_CPU_TIME,    INTR_NAME_GENERIC,      "cpu_time",     0,      { INTR_TYPE_REAL },                                    INTR_TYPE_NONE, 1, -6, LANGSPEC_F95 },
    { INTR_CPU_TIME,    INTR_NAME_GENERIC,      "",             0,      { INTR_TYPE_DREAL },                                   INTR_TYPE_NONE, 1, -6, LANGSPEC_F95 },

    /*
     * CAF1.0 (subset of Fortran2008) intrinic functions
     * This list should match with XcodeML-Exc-Tools/src/exc/xmpF/XMPtransCoarrayRun.java
     */

    { INTR_NUM_IMAGES,    INTR_NAME_GENERIC,   "num_images",        0,      {INTR_TYPE_NONE},                 INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },
    { INTR_THIS_IMAGE,    INTR_NAME_GENERIC,   "this_image",        0,      {INTR_TYPE_ANY},                  INTR_TYPE_INT, -1, -8, LANGSPEC_NONSTD },
    { INTR_IMAGE_INDEX,   INTR_NAME_GENERIC,   "image_index",       0,      {INTR_TYPE_ANY},                  INTR_TYPE_INT, -1, -8, LANGSPEC_NONSTD },
    { INTR_LCOBOUND,      INTR_NAME_GENERIC,   "lcobound",          0,      {INTR_TYPE_ANY},                  INTR_TYPE_INT, -1, -8, LANGSPEC_NONSTD },
    { INTR_UCOBOUND,      INTR_NAME_GENERIC,   "ucobound",          0,      {INTR_TYPE_ANY},                  INTR_TYPE_INT, -1, -8, LANGSPEC_NONSTD },

    /* hidden interfaces for debugging */
    { INTR_COARRAY_ALLOCATED_BYTES,   INTR_NAME_GENERIC,   "xmpf_coarray_allocated_bytes",     0,   {INTR_TYPE_NONE},   INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },
    { INTR_COARRAY_GARBAGE_BYTES,     INTR_NAME_GENERIC,   "xmpf_coarray_garbage_bytes",       0,   {INTR_TYPE_NONE},   INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },


    /* XMP/F */

    { INTR_DESC_OF,        INTR_NAME_GENERIC,      "xmp_desc_of",         0,      {INTR_TYPE_ANY},       INTR_TYPE_DREAL,  1, -8, LANGSPEC_NONSTD },

    { INTR_GET_MPI_COMM,        INTR_NAME_GENERIC,      "xmp_get_mpi_comm",         0,      {INTR_TYPE_NONE},       INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },

    { INTR_NUM_NODES,        INTR_NAME_GENERIC,      "xmp_num_nodes",         0,      {INTR_TYPE_NONE},       INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },

    { INTR_NODE_NUM,         INTR_NAME_GENERIC,      "xmp_node_num",          0,      {INTR_TYPE_NONE},       INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },

    { INTR_ALL_NUM_NODES,        INTR_NAME_GENERIC,      "xmp_all_num_nodes",         0,      {INTR_TYPE_NONE},       INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },

    { INTR_ALL_NODE_NUM,         INTR_NAME_GENERIC,      "xmp_all_node_num",          0,      {INTR_TYPE_NONE},       INTR_TYPE_INT,  0, -8, LANGSPEC_NONSTD },

    { INTR_WTIME,         INTR_NAME_GENERIC,      "xmp_wtime",          0,      {INTR_TYPE_NONE},       INTR_TYPE_DREAL,  0, -8, LANGSPEC_NONSTD },

    { INTR_WTICK,         INTR_NAME_GENERIC,      "xmp_wtick",          0,      {INTR_TYPE_NONE},       INTR_TYPE_DREAL,  0, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_NDIMS,         INTR_NAME_GENERIC,      "xmp_array_ndims",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT},       INTR_TYPE_INT, 2, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_LBOUND,         INTR_NAME_GENERIC,      "xmp_array_lbound",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_UBOUND,         INTR_NAME_GENERIC,      "xmp_array_ubound",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_LSIZE,         INTR_NAME_GENERIC,      "xmp_array_lsize",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_USHADOW,         INTR_NAME_GENERIC,      "xmp_array_ushadow",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_LSHADOW,         INTR_NAME_GENERIC,      "xmp_array_lshadow",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_LEAD_DIM,         INTR_NAME_GENERIC,      "xmp_array_lead_dim",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT_ARRAY},       INTR_TYPE_INT, 2, -8, LANGSPEC_NONSTD },

    { INTR_ARRAY_GTOL,         INTR_NAME_GENERIC,      "xmp_array_gtol",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT_ARRAY, INTR_TYPE_INT_ARRAY},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ALIGN_AXIS,         INTR_NAME_GENERIC,      "xmp_align_axis",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ALIGN_OFFSET,         INTR_NAME_GENERIC,      "xmp_align_offset",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ALIGN_REPLICATED,         INTR_NAME_GENERIC,      "xmp_align_replicated",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_LOGICAL},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_ALIGN_TEMPLATE,         INTR_NAME_GENERIC,      "xmp_align_template",          0,      {INTR_TYPE_ANY, INTR_TYPE_ANY},       INTR_TYPE_INT, 2, -8, LANGSPEC_NONSTD },

    { INTR_TEMPLATE_FIXED,         INTR_NAME_GENERIC,      "xmp_template_fixed",          0,      {INTR_TYPE_ANY, INTR_TYPE_LOGICAL},       INTR_TYPE_INT, 2, -8, LANGSPEC_NONSTD },

    { INTR_TEMPLATE_NDIMS,         INTR_NAME_GENERIC,      "xmp_template_ndims",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT},       INTR_TYPE_INT, 2, -8, LANGSPEC_NONSTD },

    { INTR_TEMPLATE_LBOUND,         INTR_NAME_GENERIC,      "xmp_template_lbound",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_TEMPLATE_UBOUND,         INTR_NAME_GENERIC,      "xmp_template_ubound",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_DIST_FORMAT,         INTR_NAME_GENERIC,      "xmp_dist_format",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_DIST_BLOCKSIZE,         INTR_NAME_GENERIC,      "xmp_dist_blocksize",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_DIST_GBLOCKMAP,         INTR_NAME_GENERIC,      "xmp_dist_gblockmap",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT_ARRAY},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_DIST_NODES,         INTR_NAME_GENERIC,      "xmp_dist_nodes",          0,      {INTR_TYPE_ANY, INTR_TYPE_ANY},       INTR_TYPE_INT, 2, -8, LANGSPEC_NONSTD },

    { INTR_DIST_AXIS,         INTR_NAME_GENERIC,      "xmp_dist_axis",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_NODES_NDIMS,         INTR_NAME_GENERIC,      "xmp_nodes_ndims",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT},       INTR_TYPE_INT, 2, -8, LANGSPEC_NONSTD },

    { INTR_NODES_INDEX,         INTR_NAME_GENERIC,      "xmp_nodes_index",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_NODES_SIZE,         INTR_NAME_GENERIC,      "xmp_nodes_size",          0,      {INTR_TYPE_ANY, INTR_TYPE_INT, INTR_TYPE_INT},       INTR_TYPE_INT, 3, -8, LANGSPEC_NONSTD },

    { INTR_NODES_EQUIV,         INTR_NAME_GENERIC,      "xmp_nodes_equiv",          0,      {INTR_TYPE_ANY, INTR_TYPE_ANY, INTR_TYPE_INT_ARRAY, INTR_TYPE_INT_ARRAY, INTR_TYPE_INT_ARRAY},       INTR_TYPE_INT, 5, -8, LANGSPEC_NONSTD },

    { INTR_END }
};
