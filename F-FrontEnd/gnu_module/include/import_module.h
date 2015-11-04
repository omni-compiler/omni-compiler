/********************************************************************/
/********************************************************************/

#ifndef IMPORT_MODULE_H
#define IMPORT_MODULE_H


#define CHAR_BIT 8
#define UCHAR_MAX 255
#define bool unsigned char
#define size_t int

#ifdef ENABLE_RUNTIME_CHECKING
#define gcc_assert(EXPR) ((void)(!(EXPR) ? abort (), 0 : 0))
#else
/* Include EXPR, so that unused variable warnings do not occur.  */
#define gcc_assert(EXPR) ((void)(0 && (EXPR)))
#endif

/* Use gcc_unreachable() to mark unreachable locations (like an
   unreachable default case of a switch.  Do not use gcc_assert(0).  */
#define gcc_unreachable() (abort ())

/* The size of `int', as computed by sizeof. */
#ifndef USED_FOR_TARGET
#define SIZEOF_INT 4
#endif


/* The size of `long', as computed by sizeof. */
#ifndef USED_FOR_TARGET
#define SIZEOF_LONG 8
#endif


/* The size of `long long', as computed by sizeof. */
#ifndef USED_FOR_TARGET
#define SIZEOF_LONG_LONG 8
#endif


/* The size of `short', as computed by sizeof. */
#ifndef USED_FOR_TARGET
#define SIZEOF_SHORT 2
#endif

/* This describes the machine the compiler is hosted on.  */
#define HOST_BITS_PER_CHAR  CHAR_BIT
#define HOST_BITS_PER_SHORT (CHAR_BIT * SIZEOF_SHORT)
#define HOST_BITS_PER_INT   (CHAR_BIT * SIZEOF_INT)
#define HOST_BITS_PER_LONG  (CHAR_BIT * SIZEOF_LONG)

/* Be conservative and only use enum bitfields with C++ or GCC.
   FIXME: provide a complete autoconf test for buggy enum bitfields.  */

/*
#ifdef __cplusplus
#define ENUM_BITFIELD(TYPE) enum TYPE
#elif (GCC_VERSION > 2000)
#define ENUM_BITFIELD(TYPE) __extension__ enum TYPE
#else
#define ENUM_BITFIELD(TYPE) unsigned int
#endif
*/


/* Define to the widest efficient host integer type at least as wide as the
   target's size_t type. */
#define HOST_WIDE_INT long

#define true    1
#define false   0


/* These macros provide a K&R/C89/C++-friendly way of allocating structures
   with nice encapsulation.  The XDELETE*() macros are technically
   superfluous, but provided here for symmetry.  Using them consistently
   makes it easier to update client code to use different allocators such
   as new/delete and new[]/delete[].  */

/* Scalar allocators.  */

#define XALLOCA(T)              ((T *) alloca (sizeof (T)))
#define XNEW(T)                 ((T *) xmalloc (sizeof (T)))
#define XCNEW(T)                ((T *) xcalloc (1, sizeof (T)))
#define XDUP(T, P)              ((T *) xmemdup ((P), sizeof (T), sizeof (T)))
#define XDELETE(P)              free ((void*) (P))

/* Array allocators.  */

#define XALLOCAVEC(T, N)        ((T *) alloca (sizeof (T) * (N)))
#define XNEWVEC(T, N)           ((T *) xmalloc (sizeof (T) * (N)))
#define XCNEWVEC(T, N)          ((T *) xcalloc ((N), sizeof (T)))
#define XDUPVEC(T, P, N)        ((T *) xmemdup ((P), sizeof (T) * (N), sizeof (T) * (N)))
#define XRESIZEVEC(T, P, N)     ((T *) xrealloc ((void *) (P), sizeof (T) * (N)))
#define XDELETEVEC(P)           free ((void*) (P))

/* Allocators for variable-sized structures and raw buffers.  */

#define XALLOCAVAR(T, S)        ((T *) alloca ((S)))
#define XNEWVAR(T, S)           ((T *) xmalloc ((S)))
#define XCNEWVAR(T, S)          ((T *) xcalloc (1, (S)))
#define XDUPVAR(T, P, S1, S2)   ((T *) xmemdup ((P), (S1), (S2)))
#define XRESIZEVAR(T, P, S)     ((T *) xrealloc ((P), (S)))

/* Type-safe obstack allocator.  */

#define XOBNEW(O, T)            ((T *) obstack_alloc ((O), sizeof (T)))
#define XOBNEWVEC(O, T, N)      ((T *) obstack_alloc ((O), sizeof (T) * (N)))
#define XOBNEWVAR(O, T, S)      ((T *) obstack_alloc ((O), (S)))
#define XOBFINISH(O, T)         ((T) obstack_finish ((O)))

/*******************************************************************/

#include "ansidecl.h"

/*********************/
/***** tsystem.h *****/
/*********************/

#define CONST_CAST2(TOTYPE,FROMTYPE,X) ((__extension__(union {FROMTYPE _q; TOTYPE _nq;})(X))._nq)
#define CONST_CAST(TYPE,X) CONST_CAST2(TYPE, const TYPE, (X))

/*********************/
/***** options.h *****/
/*********************/

enum opt_code
{
  OPT____ = 0,                               /* -### */
  /* OPT__CLASSPATH = 1, */                  /* --CLASSPATH */
  /* OPT__all_warnings = 2, */               /* --all-warnings */
  /* OPT__ansi = 3, */                       /* --ansi */
  /* OPT__assemble = 4, */                   /* --assemble */
  /* OPT__assert = 5, */                     /* --assert */
  /* OPT__assert_ = 6, */                    /* --assert= */
  /* OPT__bootclasspath = 7, */              /* --bootclasspath */
  /* OPT__classpath = 8, */                  /* --classpath */
  /* OPT__comments = 9, */                   /* --comments */
  /* OPT__comments_in_macros = 10, */        /* --comments-in-macros */
  /* OPT__compile = 11, */                   /* --compile */
  /* OPT__coverage = 12, */                  /* --coverage */
  /* OPT__debug = 13, */                     /* --debug */
  /* OPT__define_macro = 14, */              /* --define-macro */
  /* OPT__define_macro_ = 15, */             /* --define-macro= */
  /* OPT__dependencies = 16, */              /* --dependencies */
  /* OPT__dump = 17, */                      /* --dump */
  /* OPT__dump_ = 18, */                     /* --dump= */
  /* OPT__dumpbase = 19, */                  /* --dumpbase */
  /* OPT__dumpdir = 20, */                   /* --dumpdir */
  /* OPT__encoding = 21, */                  /* --encoding */
  /* OPT__entry = 22, */                     /* --entry */
  /* OPT__entry_ = 23, */                    /* --entry= */
  /* OPT__extdirs = 24, */                   /* --extdirs */
  /* OPT__extra_warnings = 25, */            /* --extra-warnings */
  /* OPT__for_assembler = 26, */             /* --for-assembler */
  /* OPT__for_assembler_ = 27, */            /* --for-assembler= */
  /* OPT__for_linker = 28, */                /* --for-linker */
  /* OPT__for_linker_ = 29, */               /* --for-linker= */
  /* OPT__force_link = 30, */                /* --force-link */
  /* OPT__force_link_ = 31, */               /* --force-link= */
  OPT__help = 32,                            /* --help */
  OPT__help_ = 33,                           /* --help= */
  /* OPT__imacros = 34, */                   /* --imacros */
  /* OPT__imacros_ = 35, */                  /* --imacros= */
  /* OPT__include = 36, */                   /* --include */
  /* OPT__include_barrier = 37, */           /* --include-barrier */
  /* OPT__include_directory = 38, */         /* --include-directory */
  /* OPT__include_directory_after = 39, */   /* --include-directory-after */
  /* OPT__include_directory_after_ = 40, */  /* --include-directory-after= */
  /* OPT__include_directory_ = 41, */        /* --include-directory= */
  /* OPT__include_prefix = 42, */            /* --include-prefix */
  /* OPT__include_prefix_ = 43, */           /* --include-prefix= */
  /* OPT__include_with_prefix = 44, */       /* --include-with-prefix */
  /* OPT__include_with_prefix_after = 45, */ /* --include-with-prefix-after */
  /* OPT__include_with_prefix_after_ = 46, *//* --include-with-prefix-after= */
  /* OPT__include_with_prefix_before = 47, *//* --include-with-prefix-before */
  /* OPT__include_with_prefix_before_ = 48, *//* --include-with-prefix-before= */
  /* OPT__include_with_prefix_ = 49, */      /* --include-with-prefix= */
  /* OPT__include_ = 50, */                  /* --include= */
  /* OPT__language = 51, */                  /* --language */
  /* OPT__language_ = 52, */                 /* --language= */
  /* OPT__library_directory = 53, */         /* --library-directory */
  /* OPT__library_directory_ = 54, */        /* --library-directory= */
  /* OPT__no_canonical_prefixes = 55, */     /* --no-canonical-prefixes */
  /* OPT__no_integrated_cpp = 56, */         /* --no-integrated-cpp */
  /* OPT__no_line_commands = 57, */          /* --no-line-commands */
  /* OPT__no_standard_includes = 58, */      /* --no-standard-includes */
  /* OPT__no_standard_libraries = 59, */     /* --no-standard-libraries */
  /* OPT__no_warnings = 60, */               /* --no-warnings */
  /* OPT__optimize = 61, */                  /* --optimize */
  /* OPT__output = 62, */                    /* --output */
  /* OPT__output_class_directory = 63, */    /* --output-class-directory */
  /* OPT__output_class_directory_ = 64, */   /* --output-class-directory= */
  OPT__output_pch_ = 65,                     /* --output-pch= */
  /* OPT__output_ = 66, */                   /* --output= */
  OPT__param = 67,                           /* --param */
  /* OPT__param_ = 68, */                    /* --param= */
  /* OPT__pass_exit_codes = 69, */           /* --pass-exit-codes */
  /* OPT__pedantic = 70, */                  /* --pedantic */
  /* OPT__pedantic_errors = 71, */           /* --pedantic-errors */
  /* OPT__pie = 72, */                       /* --pie */
  /* OPT__pipe = 73, */                      /* --pipe */
  /* OPT__prefix = 74, */                    /* --prefix */
  /* OPT__prefix_ = 75, */                   /* --prefix= */
  /* OPT__preprocess = 76, */                /* --preprocess */
  /* OPT__print_file_name = 77, */           /* --print-file-name */
  /* OPT__print_file_name_ = 78, */          /* --print-file-name= */
  /* OPT__print_libgcc_file_name = 79, */    /* --print-libgcc-file-name */
  /* OPT__print_missing_file_dependencies = 80, *//* --print-missing-file-dependencies */
  /* OPT__print_multi_directory = 81, */     /* --print-multi-directory */
  /* OPT__print_multi_lib = 82, */           /* --print-multi-lib */
  /* OPT__print_multi_os_directory = 83, */  /* --print-multi-os-directory */
  /* OPT__print_prog_name = 84, */           /* --print-prog-name */
  /* OPT__print_prog_name_ = 85, */          /* --print-prog-name= */
  /* OPT__print_search_dirs = 86, */         /* --print-search-dirs */
  /* OPT__print_sysroot = 87, */             /* --print-sysroot */
  /* OPT__print_sysroot_headers_suffix = 88, *//* --print-sysroot-headers-suffix */
  /* OPT__profile = 89, */                   /* --profile */
  /* OPT__resource = 90, */                  /* --resource */
  /* OPT__resource_ = 91, */                 /* --resource= */
  /* OPT__save_temps = 92, */                /* --save-temps */
  /* OPT__shared = 93, */                    /* --shared */
  /* OPT__specs = 94, */                     /* --specs */
  /* OPT__specs_ = 95, */                    /* --specs= */
  /* OPT__static = 96, */                    /* --static */
  /* OPT__symbolic = 97, */                  /* --symbolic */
  /* OPT__sysroot = 98, */                   /* --sysroot */
  OPT__sysroot_ = 99,                        /* --sysroot= */
  OPT__target_help = 100,                    /* --target-help */
  /* OPT__time = 101, */                     /* --time */
  /* OPT__trace_includes = 102, */           /* --trace-includes */
  /* OPT__traditional = 103, */              /* --traditional */
  /* OPT__traditional_cpp = 104, */          /* --traditional-cpp */
  /* OPT__trigraphs = 105, */                /* --trigraphs */
  /* OPT__undefine_macro = 106, */           /* --undefine-macro */
  /* OPT__undefine_macro_ = 107, */          /* --undefine-macro= */
  /* OPT__user_dependencies = 108, */        /* --user-dependencies */
  /* OPT__verbose = 109, */                  /* --verbose */
  OPT__version = 110,                        /* --version */
  /* OPT__write_dependencies = 111, */       /* --write-dependencies */
  /* OPT__write_user_dependencies = 112, */  /* --write-user-dependencies */
  OPT_A = 113,                               /* -A */
  OPT_B = 114,                               /* -B */
  OPT_C = 115,                               /* -C */
  OPT_CC = 116,                              /* -CC */
  /* OPT_CLASSPATH = 117, */                 /* -CLASSPATH */
  OPT_D = 118,                               /* -D */
  OPT_E = 119,                               /* -E */
  OPT_F = 120,                               /* -F */
  OPT_H = 121,                               /* -H */
  OPT_I = 122,                               /* -I */
  OPT_J = 123,                               /* -J */
  OPT_L = 124,                               /* -L */
  OPT_M = 125,                               /* -M */
  OPT_MD = 126,                              /* -MD */
  OPT_MD_ = 127,                             /* -MD_ */
  OPT_MF = 128,                              /* -MF */
  OPT_MG = 129,                              /* -MG */
  OPT_MM = 130,                              /* -MM */
  OPT_MMD = 131,                             /* -MMD */
  OPT_MMD_ = 132,                            /* -MMD_ */
  OPT_MP = 133,                              /* -MP */
  OPT_MQ = 134,                              /* -MQ */
  OPT_MT = 135,                              /* -MT */
  OPT_N = 136,                               /* -N */
  OPT_O = 137,                               /* -O */
  OPT_Ofast = 138,                           /* -Ofast */
  OPT_Os = 139,                              /* -Os */
  OPT_P = 140,                               /* -P */
  OPT_Q = 141,                               /* -Q */
  OPT_Qn = 142,                              /* -Qn */
  OPT_Qy = 143,                              /* -Qy */
  OPT_R = 144,                               /* -R */
  OPT_S = 145,                               /* -S */
  OPT_T = 146,                               /* -T */
  OPT_Tbss = 147,                            /* -Tbss */
  OPT_Tbss_ = 148,                           /* -Tbss= */
  OPT_Tdata = 149,                           /* -Tdata */
  OPT_Tdata_ = 150,                          /* -Tdata= */
  OPT_Ttext = 151,                           /* -Ttext */
  OPT_Ttext_ = 152,                          /* -Ttext= */
  OPT_U = 153,                               /* -U */
  /* OPT_W = 154, */                         /* -W */
  OPT_Wa_ = 155,                             /* -Wa, */
  OPT_Wabi = 156,                            /* -Wabi */
  OPT_Waddress = 157,                        /* -Waddress */
  OPT_Waggregate_return = 158,               /* -Waggregate-return */
  OPT_Waliasing = 159,                       /* -Waliasing */
  OPT_Walign_commons = 160,                  /* -Walign-commons */
  OPT_Wall = 161,                            /* -Wall */
  OPT_Wall_deprecation = 162,                /* -Wall-deprecation */
  OPT_Wall_javadoc = 163,                    /* -Wall-javadoc */
  OPT_Wampersand = 164,                      /* -Wampersand */
  OPT_Warray_bounds = 165,                   /* -Warray-bounds */
  OPT_Warray_temporaries = 166,              /* -Warray-temporaries */
  OPT_Wassert_identifier = 167,              /* -Wassert-identifier */
  OPT_Wassign_intercept = 168,               /* -Wassign-intercept */
  OPT_Wattributes = 169,                     /* -Wattributes */
  OPT_Wbad_function_cast = 170,              /* -Wbad-function-cast */
  OPT_Wboxing = 171,                         /* -Wboxing */
  OPT_Wbuiltin_macro_redefined = 172,        /* -Wbuiltin-macro-redefined */
  OPT_Wc___compat = 173,                     /* -Wc++-compat */
  OPT_Wc__0x_compat = 174,                   /* -Wc++0x-compat */
  /* OPT_Wc__11_compat = 175, */             /* -Wc++11-compat */
  OPT_Wcast_align = 176,                     /* -Wcast-align */
  OPT_Wcast_qual = 177,                      /* -Wcast-qual */
  OPT_Wchar_concat = 178,                    /* -Wchar-concat */
  OPT_Wchar_subscripts = 179,                /* -Wchar-subscripts */
  OPT_Wcharacter_truncation = 180,           /* -Wcharacter-truncation */
  OPT_Wclobbered = 181,                      /* -Wclobbered */
  OPT_Wcomment = 182,                        /* -Wcomment */
  /* OPT_Wcomments = 183, */                 /* -Wcomments */
  OPT_Wcondition_assign = 184,               /* -Wcondition-assign */
  OPT_Wconstructor_name = 185,               /* -Wconstructor-name */
  OPT_Wconversion = 186,                     /* -Wconversion */
  OPT_Wconversion_extra = 187,               /* -Wconversion-extra */
  OPT_Wconversion_null = 188,                /* -Wconversion-null */
  OPT_Wcoverage_mismatch = 189,              /* -Wcoverage-mismatch */
  OPT_Wcpp = 190,                            /* -Wcpp */
  OPT_Wctor_dtor_privacy = 191,              /* -Wctor-dtor-privacy */
  OPT_Wdeclaration_after_statement = 192,    /* -Wdeclaration-after-statement */
  OPT_Wdelete_non_virtual_dtor = 193,        /* -Wdelete-non-virtual-dtor */
  OPT_Wdep_ann = 194,                        /* -Wdep-ann */
  OPT_Wdeprecated = 195,                     /* -Wdeprecated */
  OPT_Wdeprecated_declarations = 196,        /* -Wdeprecated-declarations */
  OPT_Wdisabled_optimization = 197,          /* -Wdisabled-optimization */
  OPT_Wdiscouraged = 198,                    /* -Wdiscouraged */
  OPT_Wdiv_by_zero = 199,                    /* -Wdiv-by-zero */
  OPT_Wdouble_promotion = 200,               /* -Wdouble-promotion */
  OPT_Weffc__ = 201,                         /* -Weffc++ */
  OPT_Wempty_block = 202,                    /* -Wempty-block */
  OPT_Wempty_body = 203,                     /* -Wempty-body */
  OPT_Wendif_labels = 204,                   /* -Wendif-labels */
  OPT_Wenum_compare = 205,                   /* -Wenum-compare */
  OPT_Wenum_identifier = 206,                /* -Wenum-identifier */
  OPT_Wenum_switch = 207,                    /* -Wenum-switch */
  OPT_Werror = 208,                          /* -Werror */
  /* OPT_Werror_implicit_function_declaration = 209, *//* -Werror-implicit-function-declaration */
  OPT_Werror_ = 210,                         /* -Werror= */
  OPT_Wextra = 211,                          /* -Wextra */
  OPT_Wextraneous_semicolon = 212,           /* -Wextraneous-semicolon */
  OPT_Wfallthrough = 213,                    /* -Wfallthrough */
  OPT_Wfatal_errors = 214,                   /* -Wfatal-errors */
  OPT_Wfield_hiding = 215,                   /* -Wfield-hiding */
  OPT_Wfinal_bound = 216,                    /* -Wfinal-bound */
  OPT_Wfinally = 217,                        /* -Wfinally */
  OPT_Wfloat_equal = 218,                    /* -Wfloat-equal */
  OPT_Wforbidden = 219,                      /* -Wforbidden */
  OPT_Wformat = 220,                         /* -Wformat */
  OPT_Wformat_contains_nul = 221,            /* -Wformat-contains-nul */
  OPT_Wformat_extra_args = 222,              /* -Wformat-extra-args */
  OPT_Wformat_nonliteral = 223,              /* -Wformat-nonliteral */
  OPT_Wformat_security = 224,                /* -Wformat-security */
  OPT_Wformat_y2k = 225,                     /* -Wformat-y2k */
  OPT_Wformat_zero_length = 226,             /* -Wformat-zero-length */
  OPT_Wformat_ = 227,                        /* -Wformat= */
  OPT_Wframe_larger_than_ = 228,             /* -Wframe-larger-than= */
  OPT_Wfree_nonheap_object = 229,            /* -Wfree-nonheap-object */
  OPT_Wfunction_elimination = 230,           /* -Wfunction-elimination */
  OPT_Whiding = 231,                         /* -Whiding */
  OPT_Wignored_qualifiers = 232,             /* -Wignored-qualifiers */
  OPT_Wimplicit = 233,                       /* -Wimplicit */
  OPT_Wimplicit_function_declaration = 234,  /* -Wimplicit-function-declaration */
  OPT_Wimplicit_int = 235,                   /* -Wimplicit-int */
  OPT_Wimplicit_interface = 236,             /* -Wimplicit-interface */
  OPT_Wimplicit_procedure = 237,             /* -Wimplicit-procedure */
  /* OPT_Wimport = 238, */                   /* -Wimport */
  OPT_Windirect_static = 239,                /* -Windirect-static */
  OPT_Winit_self = 240,                      /* -Winit-self */
  OPT_Winline = 241,                         /* -Winline */
  OPT_Wint_to_pointer_cast = 242,            /* -Wint-to-pointer-cast */
  OPT_Wintf_annotation = 243,                /* -Wintf-annotation */
  OPT_Wintf_non_inherited = 244,             /* -Wintf-non-inherited */
  OPT_Wintrinsic_shadow = 245,               /* -Wintrinsic-shadow */
  OPT_Wintrinsics_std = 246,                 /* -Wintrinsics-std */
  OPT_Winvalid_memory_model = 247,           /* -Winvalid-memory-model */
  OPT_Winvalid_offsetof = 248,               /* -Winvalid-offsetof */
  OPT_Winvalid_pch = 249,                    /* -Winvalid-pch */
  OPT_Wjavadoc = 250,                        /* -Wjavadoc */
  OPT_Wjump_misses_init = 251,               /* -Wjump-misses-init */
  OPT_Wl_ = 252,                             /* -Wl, */
  /* OPT_Wlarger_than_ = 253, */             /* -Wlarger-than- */
  OPT_Wlarger_than_ = 254,                   /* -Wlarger-than= */
  OPT_Wline_truncation = 255,                /* -Wline-truncation */
  OPT_Wlocal_hiding = 256,                   /* -Wlocal-hiding */
  OPT_Wlogical_op = 257,                     /* -Wlogical-op */
  OPT_Wlong_long = 258,                      /* -Wlong-long */
  OPT_Wmain = 259,                           /* -Wmain */
  OPT_Wmasked_catch_block = 260,             /* -Wmasked-catch-block */
  OPT_Wmaybe_uninitialized = 261,            /* -Wmaybe-uninitialized */
  OPT_Wmissing_braces = 262,                 /* -Wmissing-braces */
  OPT_Wmissing_declarations = 263,           /* -Wmissing-declarations */
  OPT_Wmissing_field_initializers = 264,     /* -Wmissing-field-initializers */
  OPT_Wmissing_format_attribute = 265,       /* -Wmissing-format-attribute */
  OPT_Wmissing_include_dirs = 266,           /* -Wmissing-include-dirs */
  OPT_Wmissing_noreturn = 267,               /* -Wmissing-noreturn */
  OPT_Wmissing_parameter_type = 268,         /* -Wmissing-parameter-type */
  OPT_Wmissing_prototypes = 269,             /* -Wmissing-prototypes */
  OPT_Wmudflap = 270,                        /* -Wmudflap */
  OPT_Wmultichar = 271,                      /* -Wmultichar */
  OPT_Wnarrowing = 272,                      /* -Wnarrowing */
  OPT_Wnested_externs = 273,                 /* -Wnested-externs */
  OPT_Wnls = 274,                            /* -Wnls */
  OPT_Wno_effect_assign = 275,               /* -Wno-effect-assign */
  OPT_Wnoexcept = 276,                       /* -Wnoexcept */
  OPT_Wnon_template_friend = 277,            /* -Wnon-template-friend */
  OPT_Wnon_virtual_dtor = 278,               /* -Wnon-virtual-dtor */
  OPT_Wnonnull = 279,                        /* -Wnonnull */
  OPT_Wnormalized_ = 280,                    /* -Wnormalized= */
  OPT_Wnull = 281,                           /* -Wnull */
  OPT_Wold_style_cast = 282,                 /* -Wold-style-cast */
  OPT_Wold_style_declaration = 283,          /* -Wold-style-declaration */
  OPT_Wold_style_definition = 284,           /* -Wold-style-definition */
  OPT_Wout_of_date = 285,                    /* -Wout-of-date */
  OPT_Wover_ann = 286,                       /* -Wover-ann */
  OPT_Woverflow = 287,                       /* -Woverflow */
  OPT_Woverlength_strings = 288,             /* -Woverlength-strings */
  OPT_Woverloaded_virtual = 289,             /* -Woverloaded-virtual */
  OPT_Woverride_init = 290,                  /* -Woverride-init */
  OPT_Wp_ = 291,                             /* -Wp, */
  OPT_Wpacked = 292,                         /* -Wpacked */
  OPT_Wpacked_bitfield_compat = 293,         /* -Wpacked-bitfield-compat */
  OPT_Wpadded = 294,                         /* -Wpadded */
  OPT_Wparam_assign = 295,                   /* -Wparam-assign */
  OPT_Wparentheses = 296,                    /* -Wparentheses */
  OPT_Wpkg_default_method = 297,             /* -Wpkg-default-method */
  OPT_Wpmf_conversions = 298,                /* -Wpmf-conversions */
  OPT_Wpointer_arith = 299,                  /* -Wpointer-arith */
  OPT_Wpointer_sign = 300,                   /* -Wpointer-sign */
  OPT_Wpointer_to_int_cast = 301,            /* -Wpointer-to-int-cast */
  OPT_Wpragmas = 302,                        /* -Wpragmas */
  OPT_Wproperty_assign_default = 303,        /* -Wproperty-assign-default */
  OPT_Wprotocol = 304,                       /* -Wprotocol */
  OPT_Wpsabi = 305,                          /* -Wpsabi */
  OPT_Wraw = 306,                            /* -Wraw */
  OPT_Wreal_q_constant = 307,                /* -Wreal-q-constant */
  OPT_Wredundant_decls = 308,                /* -Wredundant-decls */
  OPT_Wredundant_modifiers = 309,            /* -Wredundant-modifiers */
  OPT_Wreorder = 310,                        /* -Wreorder */
  OPT_Wreturn_type = 311,                    /* -Wreturn-type */
  OPT_Wselector = 312,                       /* -Wselector */
  OPT_Wsequence_point = 313,                 /* -Wsequence-point */
  OPT_Wserial = 314,                         /* -Wserial */
  OPT_Wshadow = 315,                         /* -Wshadow */
  OPT_Wsign_compare = 316,                   /* -Wsign-compare */
  OPT_Wsign_conversion = 317,                /* -Wsign-conversion */
  OPT_Wsign_promo = 318,                     /* -Wsign-promo */
  OPT_Wspecial_param_hiding = 319,           /* -Wspecial-param-hiding */
  OPT_Wstack_protector = 320,                /* -Wstack-protector */
  OPT_Wstack_usage_ = 321,                   /* -Wstack-usage= */
  OPT_Wstatic_access = 322,                  /* -Wstatic-access */
  OPT_Wstatic_receiver = 323,                /* -Wstatic-receiver */
  OPT_Wstrict_aliasing = 324,                /* -Wstrict-aliasing */
  OPT_Wstrict_aliasing_ = 325,               /* -Wstrict-aliasing= */
  OPT_Wstrict_null_sentinel = 326,           /* -Wstrict-null-sentinel */
  OPT_Wstrict_overflow = 327,                /* -Wstrict-overflow */
  OPT_Wstrict_overflow_ = 328,               /* -Wstrict-overflow= */
  OPT_Wstrict_prototypes = 329,              /* -Wstrict-prototypes */
  OPT_Wstrict_selector_match = 330,          /* -Wstrict-selector-match */
  OPT_Wsuggest_attribute_const = 331,        /* -Wsuggest-attribute=const */
  OPT_Wsuggest_attribute_noreturn = 332,     /* -Wsuggest-attribute=noreturn */
  OPT_Wsuggest_attribute_pure = 333,         /* -Wsuggest-attribute=pure */
  OPT_Wsuppress = 334,                       /* -Wsuppress */
  OPT_Wsurprising = 335,                     /* -Wsurprising */
  OPT_Wswitch = 336,                         /* -Wswitch */
  OPT_Wswitch_default = 337,                 /* -Wswitch-default */
  OPT_Wswitch_enum = 338,                    /* -Wswitch-enum */
  OPT_Wsync_nand = 339,                      /* -Wsync-nand */
  OPT_Wsynth = 340,                          /* -Wsynth */
  OPT_Wsynthetic_access = 341,               /* -Wsynthetic-access */
  OPT_Wsystem_headers = 342,                 /* -Wsystem-headers */
  OPT_Wtabs = 343,                           /* -Wtabs */
  OPT_Wtasks = 344,                          /* -Wtasks */
  OPT_Wtraditional = 345,                    /* -Wtraditional */
  OPT_Wtraditional_conversion = 346,         /* -Wtraditional-conversion */
  OPT_Wtrampolines = 347,                    /* -Wtrampolines */
  OPT_Wtrigraphs = 348,                      /* -Wtrigraphs */
  OPT_Wtype_hiding = 349,                    /* -Wtype-hiding */
  OPT_Wtype_limits = 350,                    /* -Wtype-limits */
  OPT_Wuncheck = 351,                        /* -Wuncheck */
  OPT_Wundeclared_selector = 352,            /* -Wundeclared-selector */
  OPT_Wundef = 353,                          /* -Wundef */
  OPT_Wunderflow = 354,                      /* -Wunderflow */
  OPT_Wuninitialized = 355,                  /* -Wuninitialized */
  OPT_Wunknown_pragmas = 356,                /* -Wunknown-pragmas */
  OPT_Wunnecessary_else = 357,               /* -Wunnecessary-else */
  OPT_Wunqualified_field = 358,              /* -Wunqualified-field */
  /* OPT_Wunreachable_code = 359, */         /* -Wunreachable-code */
  OPT_Wunsafe_loop_optimizations = 360,      /* -Wunsafe-loop-optimizations */
  OPT_Wunsuffixed_float_constants = 361,     /* -Wunsuffixed-float-constants */
  OPT_Wunused = 362,                         /* -Wunused */
  OPT_Wunused_argument = 363,                /* -Wunused-argument */
  OPT_Wunused_but_set_parameter = 364,       /* -Wunused-but-set-parameter */
  OPT_Wunused_but_set_variable = 365,        /* -Wunused-but-set-variable */
  OPT_Wunused_dummy_argument = 366,          /* -Wunused-dummy-argument */
  OPT_Wunused_function = 367,                /* -Wunused-function */
  OPT_Wunused_import = 368,                  /* -Wunused-import */
  OPT_Wunused_label = 369,                   /* -Wunused-label */
  OPT_Wunused_local = 370,                   /* -Wunused-local */
  OPT_Wunused_local_typedefs = 371,          /* -Wunused-local-typedefs */
  OPT_Wunused_macros = 372,                  /* -Wunused-macros */
  OPT_Wunused_parameter = 373,               /* -Wunused-parameter */
  OPT_Wunused_private = 374,                 /* -Wunused-private */
  OPT_Wunused_result = 375,                  /* -Wunused-result */
  OPT_Wunused_thrown = 376,                  /* -Wunused-thrown */
  OPT_Wunused_value = 377,                   /* -Wunused-value */
  OPT_Wunused_variable = 378,                /* -Wunused-variable */
  OPT_Wuseless_type_check = 379,             /* -Wuseless-type-check */
  OPT_Wvarargs_cast = 380,                   /* -Wvarargs-cast */
  OPT_Wvariadic_macros = 381,                /* -Wvariadic-macros */
  OPT_Wvector_operation_performance = 382,   /* -Wvector-operation-performance */
  OPT_Wvla = 383,                            /* -Wvla */
  OPT_Wvolatile_register_var = 384,          /* -Wvolatile-register-var */
  OPT_Wwarning_token = 385,                  /* -Wwarning-token */
  OPT_Wwrite_strings = 386,                  /* -Wwrite-strings */
  OPT_Wzero_as_null_pointer_constant = 387,  /* -Wzero-as-null-pointer-constant */
  OPT_Xassembler = 388,                      /* -Xassembler */
  OPT_Xlinker = 389,                         /* -Xlinker */
  OPT_Xpreprocessor = 390,                   /* -Xpreprocessor */
  OPT_Z = 391,                               /* -Z */
  OPT_ansi = 392,                            /* -ansi */
  OPT_aux_info = 393,                        /* -aux-info */
  /* OPT_aux_info_ = 394, */                 /* -aux-info= */
  OPT_auxbase = 395,                         /* -auxbase */
  OPT_auxbase_strip = 396,                   /* -auxbase-strip */
  /* OPT_bootclasspath = 397, */             /* -bootclasspath */
  OPT_c = 398,                               /* -c */
  /* OPT_classpath = 399, */                 /* -classpath */
  OPT_coverage = 400,                        /* -coverage */
  OPT_cpp = 401,                             /* -cpp */
  OPT_cpp_ = 402,                            /* -cpp= */
  OPT_d = 403,                               /* -d */
  OPT_dumpbase = 404,                        /* -dumpbase */
  OPT_dumpdir = 405,                         /* -dumpdir */
  OPT_dumpmachine = 406,                     /* -dumpmachine */
  OPT_dumpspecs = 407,                       /* -dumpspecs */
  OPT_dumpversion = 408,                     /* -dumpversion */
  OPT_e = 409,                               /* -e */
  /* OPT_encoding = 410, */                  /* -encoding */
  OPT_export_dynamic = 411,                  /* -export-dynamic */
  OPT_extdirs = 412,                         /* -extdirs */
  /* OPT_fCLASSPATH_ = 413, */               /* -fCLASSPATH= */
  OPT_fPIC = 414,                            /* -fPIC */
  OPT_fPIE = 415,                            /* -fPIE */
  OPT_fRTS_ = 416,                           /* -fRTS= */
  OPT_fabi_version_ = 417,                   /* -fabi-version= */
  OPT_faccess_control = 418,                 /* -faccess-control */
  OPT_faggressive_function_elimination = 419,/* -faggressive-function-elimination */
  OPT_falign_commons = 420,                  /* -falign-commons */
  OPT_falign_functions = 421,                /* -falign-functions */
  OPT_falign_functions_ = 422,               /* -falign-functions= */
  OPT_falign_jumps = 423,                    /* -falign-jumps */
  OPT_falign_jumps_ = 424,                   /* -falign-jumps= */
  OPT_falign_labels = 425,                   /* -falign-labels */
  OPT_falign_labels_ = 426,                  /* -falign-labels= */
  OPT_falign_loops = 427,                    /* -falign-loops */
  OPT_falign_loops_ = 428,                   /* -falign-loops= */
  OPT_fall_intrinsics = 429,                 /* -fall-intrinsics */
  /* OPT_fall_virtual = 430, */              /* -fall-virtual */
  OPT_fallow_leading_underscore = 431,       /* -fallow-leading-underscore */
  OPT_fallow_parameterless_variadic_functions = 432,/* -fallow-parameterless-variadic-functions */
  /* OPT_falt_external_templates = 433, */   /* -falt-external-templates */
  /* OPT_fargument_alias = 434, */           /* -fargument-alias */
  /* OPT_fargument_noalias = 435, */         /* -fargument-noalias */
  /* OPT_fargument_noalias_anything = 436, *//* -fargument-noalias-anything */
  /* OPT_fargument_noalias_global = 437, */  /* -fargument-noalias-global */
  OPT_fasm = 438,                            /* -fasm */
  OPT_fassert = 439,                         /* -fassert */
  OPT_fassociative_math = 440,               /* -fassociative-math */
  OPT_fassume_compiled = 441,                /* -fassume-compiled */
  OPT_fassume_compiled_ = 442,               /* -fassume-compiled= */
  OPT_fasynchronous_unwind_tables = 443,     /* -fasynchronous-unwind-tables */
  OPT_fauto_inc_dec = 444,                   /* -fauto-inc-dec */
  OPT_fautomatic = 445,                      /* -fautomatic */
  OPT_faux_classpath = 446,                  /* -faux-classpath */
  OPT_fbackslash = 447,                      /* -fbackslash */
  OPT_fbacktrace = 448,                      /* -fbacktrace */
  OPT_fblas_matmul_limit_ = 449,             /* -fblas-matmul-limit= */
  OPT_fbootclasspath_ = 450,                 /* -fbootclasspath= */
  OPT_fbootstrap_classes = 451,              /* -fbootstrap-classes */
  OPT_fbounds_check = 452,                   /* -fbounds-check */
  OPT_fbranch_count_reg = 453,               /* -fbranch-count-reg */
  OPT_fbranch_probabilities = 454,           /* -fbranch-probabilities */
  OPT_fbranch_target_load_optimize = 455,    /* -fbranch-target-load-optimize */
  OPT_fbranch_target_load_optimize2 = 456,   /* -fbranch-target-load-optimize2 */
  OPT_fbtr_bb_exclusive = 457,               /* -fbtr-bb-exclusive */
  OPT_fbuilding_libgcc = 458,                /* -fbuilding-libgcc */
  OPT_fbuiltin = 459,                        /* -fbuiltin */
  OPT_fbuiltin_ = 460,                       /* -fbuiltin- */
  OPT_fcall_saved_ = 461,                    /* -fcall-saved- */
  OPT_fcall_used_ = 462,                     /* -fcall-used- */
  OPT_fcaller_saves = 463,                   /* -fcaller-saves */
  OPT_fcheck_array_temporaries = 464,        /* -fcheck-array-temporaries */
  OPT_fcheck_data_deps = 465,                /* -fcheck-data-deps */
  OPT_fcheck_new = 466,                      /* -fcheck-new */
  OPT_fcheck_references = 467,               /* -fcheck-references */
  OPT_fcheck_ = 468,                         /* -fcheck= */
  OPT_fclasspath_ = 469,                     /* -fclasspath= */
  OPT_fcoarray_ = 470,                       /* -fcoarray= */
  OPT_fcombine_stack_adjustments = 471,      /* -fcombine-stack-adjustments */
  OPT_fcommon = 472,                         /* -fcommon */
  OPT_fcompare_debug = 473,                  /* -fcompare-debug */
  OPT_fcompare_debug_second = 474,           /* -fcompare-debug-second */
  OPT_fcompare_debug_ = 475,                 /* -fcompare-debug= */
  OPT_fcompare_elim = 476,                   /* -fcompare-elim */
  OPT_fcompile_resource_ = 477,              /* -fcompile-resource= */
  OPT_fcond_mismatch = 478,                  /* -fcond-mismatch */
  OPT_fconserve_space = 479,                 /* -fconserve-space */
  OPT_fconserve_stack = 480,                 /* -fconserve-stack */
  OPT_fconstant_string_class_ = 481,         /* -fconstant-string-class= */
  OPT_fconstexpr_depth_ = 482,               /* -fconstexpr-depth= */
  OPT_fconvert_big_endian = 483,             /* -fconvert=big-endian */
  OPT_fconvert_little_endian = 484,          /* -fconvert=little-endian */
  OPT_fconvert_native = 485,                 /* -fconvert=native */
  OPT_fconvert_swap = 486,                   /* -fconvert=swap */
  OPT_fcprop_registers = 487,                /* -fcprop-registers */
  OPT_fcray_pointer = 488,                   /* -fcray-pointer */
  OPT_fcrossjumping = 489,                   /* -fcrossjumping */
  OPT_fcse_follow_jumps = 490,               /* -fcse-follow-jumps */
  /* OPT_fcse_skip_blocks = 491, */          /* -fcse-skip-blocks */
  OPT_fcx_fortran_rules = 492,               /* -fcx-fortran-rules */
  OPT_fcx_limited_range = 493,               /* -fcx-limited-range */
  OPT_fd_lines_as_code = 494,                /* -fd-lines-as-code */
  OPT_fd_lines_as_comments = 495,            /* -fd-lines-as-comments */
  OPT_fdata_sections = 496,                  /* -fdata-sections */
  OPT_fdbg_cnt_list = 497,                   /* -fdbg-cnt-list */
  OPT_fdbg_cnt_ = 498,                       /* -fdbg-cnt= */
  OPT_fdce = 499,                            /* -fdce */
  OPT_fdebug_cpp = 500,                      /* -fdebug-cpp */
  OPT_fdebug_prefix_map_ = 501,              /* -fdebug-prefix-map= */
  OPT_fdebug_types_section = 502,            /* -fdebug-types-section */
  OPT_fdeduce_init_list = 503,               /* -fdeduce-init-list */
  OPT_fdefault_double_8 = 504,               /* -fdefault-double-8 */
  /* OPT_fdefault_inline = 505, */           /* -fdefault-inline */
  OPT_fdefault_integer_8 = 506,              /* -fdefault-integer-8 */
  OPT_fdefault_real_8 = 507,                 /* -fdefault-real-8 */
  OPT_fdefer_pop = 508,                      /* -fdefer-pop */
  OPT_fdelayed_branch = 509,                 /* -fdelayed-branch */
  OPT_fdelete_null_pointer_checks = 510,     /* -fdelete-null-pointer-checks */
  OPT_fdevirtualize = 511,                   /* -fdevirtualize */
  OPT_fdiagnostics_show_location_ = 512,     /* -fdiagnostics-show-location= */
  OPT_fdiagnostics_show_option = 513,        /* -fdiagnostics-show-option */
  OPT_fdirectives_only = 514,                /* -fdirectives-only */
  OPT_fdisable_ = 515,                       /* -fdisable- */
  OPT_fdisable_assertions = 516,             /* -fdisable-assertions */
  OPT_fdisable_assertions_ = 517,            /* -fdisable-assertions= */
  OPT_fdollar_ok = 518,                      /* -fdollar-ok */
  OPT_fdollars_in_identifiers = 519,         /* -fdollars-in-identifiers */
  OPT_fdse = 520,                            /* -fdse */
  OPT_fdump_ = 521,                          /* -fdump- */
  /* OPT_fdump_core = 522, */                /* -fdump-core */
  OPT_fdump_final_insns = 523,               /* -fdump-final-insns */
  OPT_fdump_final_insns_ = 524,              /* -fdump-final-insns= */
  OPT_fdump_fortran_optimized = 525,         /* -fdump-fortran-optimized */
  OPT_fdump_fortran_original = 526,          /* -fdump-fortran-original */
  OPT_fdump_go_spec_ = 527,                  /* -fdump-go-spec= */
  OPT_fdump_noaddr = 528,                    /* -fdump-noaddr */
  OPT_fdump_parse_tree = 529,                /* -fdump-parse-tree */
  OPT_fdump_passes = 530,                    /* -fdump-passes */
  OPT_fdump_unnumbered = 531,                /* -fdump-unnumbered */
  OPT_fdump_unnumbered_links = 532,          /* -fdump-unnumbered-links */
  OPT_fdwarf2_cfi_asm = 533,                 /* -fdwarf2-cfi-asm */
  OPT_fearly_inlining = 534,                 /* -fearly-inlining */
  OPT_felide_constructors = 535,             /* -felide-constructors */
  OPT_feliminate_dwarf2_dups = 536,          /* -feliminate-dwarf2-dups */
  OPT_feliminate_unused_debug_symbols = 537, /* -feliminate-unused-debug-symbols */
  OPT_feliminate_unused_debug_types = 538,   /* -feliminate-unused-debug-types */
  OPT_femit_class_debug_always = 539,        /* -femit-class-debug-always */
  OPT_femit_class_file = 540,                /* -femit-class-file */
  OPT_femit_class_files = 541,               /* -femit-class-files */
  OPT_femit_struct_debug_baseonly = 542,     /* -femit-struct-debug-baseonly */
  OPT_femit_struct_debug_detailed_ = 543,    /* -femit-struct-debug-detailed= */
  OPT_femit_struct_debug_reduced = 544,      /* -femit-struct-debug-reduced */
  OPT_fenable_ = 545,                        /* -fenable- */
  OPT_fenable_assertions = 546,              /* -fenable-assertions */
  OPT_fenable_assertions_ = 547,             /* -fenable-assertions= */
  OPT_fencoding_ = 548,                      /* -fencoding= */
  OPT_fenforce_eh_specs = 549,               /* -fenforce-eh-specs */
  /* OPT_fenum_int_equiv = 550, */           /* -fenum-int-equiv */
  OPT_fexceptions = 551,                     /* -fexceptions */
  OPT_fexcess_precision_ = 552,              /* -fexcess-precision= */
  OPT_fexec_charset_ = 553,                  /* -fexec-charset= */
  OPT_fexpensive_optimizations = 554,        /* -fexpensive-optimizations */
  OPT_fextdirs_ = 555,                       /* -fextdirs= */
  OPT_fextended_identifiers = 556,           /* -fextended-identifiers */
  OPT_fexternal_blas = 557,                  /* -fexternal-blas */
  /* OPT_fexternal_templates = 558, */       /* -fexternal-templates */
  OPT_ff2c = 559,                            /* -ff2c */
  OPT_ffast_math = 560,                      /* -ffast-math */
  OPT_ffat_lto_objects = 561,                /* -ffat-lto-objects */
  OPT_ffilelist_file = 562,                  /* -ffilelist-file */
  OPT_ffinite_math_only = 563,               /* -ffinite-math-only */
  OPT_ffixed_ = 564,                         /* -ffixed- */
  OPT_ffixed_form = 565,                     /* -ffixed-form */
  OPT_ffixed_line_length_ = 566,             /* -ffixed-line-length- */
  OPT_ffixed_line_length_none = 567,         /* -ffixed-line-length-none */
  OPT_ffloat_store = 568,                    /* -ffloat-store */
  OPT_ffor_scope = 569,                      /* -ffor-scope */
  /* OPT_fforce_addr = 570, */               /* -fforce-addr */
  OPT_fforce_classes_archive_check = 571,    /* -fforce-classes-archive-check */
  OPT_fforward_propagate = 572,              /* -fforward-propagate */
  OPT_ffp_contract_ = 573,                   /* -ffp-contract= */
  OPT_ffpe_trap_ = 574,                      /* -ffpe-trap= */
  OPT_ffree_form = 575,                      /* -ffree-form */
  OPT_ffree_line_length_ = 576,              /* -ffree-line-length- */
  OPT_ffree_line_length_none = 577,          /* -ffree-line-length-none */
  OPT_ffreestanding = 578,                   /* -ffreestanding */
  OPT_ffriend_injection = 579,               /* -ffriend-injection */
  OPT_ffrontend_optimize = 580,              /* -ffrontend-optimize */
  OPT_ffunction_cse = 581,                   /* -ffunction-cse */
  OPT_ffunction_sections = 582,              /* -ffunction-sections */
  OPT_fgcse = 583,                           /* -fgcse */
  OPT_fgcse_after_reload = 584,              /* -fgcse-after-reload */
  OPT_fgcse_las = 585,                       /* -fgcse-las */
  OPT_fgcse_lm = 586,                        /* -fgcse-lm */
  OPT_fgcse_sm = 587,                        /* -fgcse-sm */
  OPT_fgnu_keywords = 588,                   /* -fgnu-keywords */
  OPT_fgnu_runtime = 589,                    /* -fgnu-runtime */
  OPT_fgnu_tm = 590,                         /* -fgnu-tm */
  OPT_fgnu89_inline = 591,                   /* -fgnu89-inline */
  OPT_fgo_dump_ = 592,                       /* -fgo-dump- */
  OPT_fgo_optimize_ = 593,                   /* -fgo-optimize- */
  OPT_fgo_prefix_ = 594,                     /* -fgo-prefix= */
  OPT_fgraphite = 595,                       /* -fgraphite */
  OPT_fgraphite_identity = 596,              /* -fgraphite-identity */
  OPT_fguess_branch_probability = 597,       /* -fguess-branch-probability */
  /* OPT_fguiding_decls = 598, */            /* -fguiding-decls */
  /* OPT_fhandle_exceptions = 599, */        /* -fhandle-exceptions */
  OPT_fhash_synchronization = 600,           /* -fhash-synchronization */
  /* OPT_fhelp = 601, */                     /* -fhelp */
  /* OPT_fhelp_ = 602, */                    /* -fhelp= */
  /* OPT_fhonor_std = 603, */                /* -fhonor-std */
  OPT_fhosted = 604,                         /* -fhosted */
  /* OPT_fhuge_objects = 605, */             /* -fhuge-objects */
  OPT_fident = 606,                          /* -fident */
  OPT_fif_conversion = 607,                  /* -fif-conversion */
  OPT_fif_conversion2 = 608,                 /* -fif-conversion2 */
  OPT_fimplement_inlines = 609,              /* -fimplement-inlines */
  OPT_fimplicit_inline_templates = 610,      /* -fimplicit-inline-templates */
  OPT_fimplicit_none = 611,                  /* -fimplicit-none */
  OPT_fimplicit_templates = 612,             /* -fimplicit-templates */
  OPT_findirect_classes = 613,               /* -findirect-classes */
  OPT_findirect_dispatch = 614,              /* -findirect-dispatch */
  OPT_findirect_inlining = 615,              /* -findirect-inlining */
  OPT_finhibit_size_directive = 616,         /* -finhibit-size-directive */
  OPT_finit_character_ = 617,                /* -finit-character= */
  OPT_finit_integer_ = 618,                  /* -finit-integer= */
  OPT_finit_local_zero = 619,                /* -finit-local-zero */
  OPT_finit_logical_ = 620,                  /* -finit-logical= */
  OPT_finit_real_ = 621,                     /* -finit-real= */
  OPT_finline = 622,                         /* -finline */
  OPT_finline_atomics = 623,                 /* -finline-atomics */
  OPT_finline_functions = 624,               /* -finline-functions */
  OPT_finline_functions_called_once = 625,   /* -finline-functions-called-once */
  /* OPT_finline_limit_ = 626, */            /* -finline-limit- */
  OPT_finline_limit_ = 627,                  /* -finline-limit= */
  OPT_finline_small_functions = 628,         /* -finline-small-functions */
  OPT_finput_charset_ = 629,                 /* -finput-charset= */
  OPT_finstrument_functions = 630,           /* -finstrument-functions */
  OPT_finstrument_functions_exclude_file_list_ = 631,/* -finstrument-functions-exclude-file-list= */
  OPT_finstrument_functions_exclude_function_list_ = 632,/* -finstrument-functions-exclude-function-list= */
  OPT_finteger_4_integer_8 = 633,            /* -finteger-4-integer-8 */
  OPT_fintrinsic_modules_path = 634,         /* -fintrinsic-modules-path */
  OPT_fipa_cp = 635,                         /* -fipa-cp */
  OPT_fipa_cp_clone = 636,                   /* -fipa-cp-clone */
  OPT_fipa_matrix_reorg = 637,               /* -fipa-matrix-reorg */
  OPT_fipa_profile = 638,                    /* -fipa-profile */
  OPT_fipa_pta = 639,                        /* -fipa-pta */
  OPT_fipa_pure_const = 640,                 /* -fipa-pure-const */
  OPT_fipa_reference = 641,                  /* -fipa-reference */
  OPT_fipa_sra = 642,                        /* -fipa-sra */
  /* OPT_fipa_struct_reorg = 643, */         /* -fipa-struct-reorg */
  OPT_fira_algorithm_ = 644,                 /* -fira-algorithm= */
  OPT_fira_loop_pressure = 645,              /* -fira-loop-pressure */
  OPT_fira_region_ = 646,                    /* -fira-region= */
  OPT_fira_share_save_slots = 647,           /* -fira-share-save-slots */
  OPT_fira_share_spill_slots = 648,          /* -fira-share-spill-slots */
  OPT_fira_verbose_ = 649,                   /* -fira-verbose= */
  OPT_fivopts = 650,                         /* -fivopts */
  OPT_fjni = 651,                            /* -fjni */
  OPT_fjump_tables = 652,                    /* -fjump-tables */
  OPT_fkeep_inline_dllexport = 653,          /* -fkeep-inline-dllexport */
  OPT_fkeep_inline_functions = 654,          /* -fkeep-inline-functions */
  OPT_fkeep_static_consts = 655,             /* -fkeep-static-consts */
  /* OPT_flabels_ok = 656, */                /* -flabels-ok */
  OPT_flax_vector_conversions = 657,         /* -flax-vector-conversions */
  OPT_fleading_underscore = 658,             /* -fleading-underscore */
  OPT_floop_block = 659,                     /* -floop-block */
  OPT_floop_flatten = 660,                   /* -floop-flatten */
  OPT_floop_interchange = 661,               /* -floop-interchange */
  /* OPT_floop_optimize = 662, */            /* -floop-optimize */
  OPT_floop_parallelize_all = 663,           /* -floop-parallelize-all */
  OPT_floop_strip_mine = 664,                /* -floop-strip-mine */
  OPT_flto = 665,                            /* -flto */
  OPT_flto_compression_level_ = 666,         /* -flto-compression-level= */
  OPT_flto_partition_1to1 = 667,             /* -flto-partition=1to1 */
  OPT_flto_partition_balanced = 668,         /* -flto-partition=balanced */
  OPT_flto_partition_none = 669,             /* -flto-partition=none */
  OPT_flto_report = 670,                     /* -flto-report */
  OPT_flto_ = 671,                           /* -flto= */
  OPT_fltrans = 672,                         /* -fltrans */
  OPT_fltrans_output_list_ = 673,            /* -fltrans-output-list= */
  OPT_fmain_ = 674,                          /* -fmain= */
  OPT_fmath_errno = 675,                     /* -fmath-errno */
  OPT_fmax_array_constructor_ = 676,         /* -fmax-array-constructor= */
  OPT_fmax_errors_ = 677,                    /* -fmax-errors= */
  OPT_fmax_identifier_length_ = 678,         /* -fmax-identifier-length= */
  OPT_fmax_stack_var_size_ = 679,            /* -fmax-stack-var-size= */
  OPT_fmax_subrecord_length_ = 680,          /* -fmax-subrecord-length= */
  OPT_fmem_report = 681,                     /* -fmem-report */
  OPT_fmerge_all_constants = 682,            /* -fmerge-all-constants */
  OPT_fmerge_constants = 683,                /* -fmerge-constants */
  OPT_fmerge_debug_strings = 684,            /* -fmerge-debug-strings */
  OPT_fmessage_length_ = 685,                /* -fmessage-length= */
  OPT_fmodule_private = 686,                 /* -fmodule-private */
  OPT_fmodulo_sched = 687,                   /* -fmodulo-sched */
  OPT_fmodulo_sched_allow_regmoves = 688,    /* -fmodulo-sched-allow-regmoves */
  OPT_fmove_loop_invariants = 689,           /* -fmove-loop-invariants */
  OPT_fms_extensions = 690,                  /* -fms-extensions */
  OPT_fmudflap = 691,                        /* -fmudflap */
  OPT_fmudflapir = 692,                      /* -fmudflapir */
  OPT_fmudflapth = 693,                      /* -fmudflapth */
  /* OPT_fname_mangling_version_ = 694, */   /* -fname-mangling-version- */
  /* OPT_fnew_abi = 695, */                  /* -fnew-abi */
  OPT_fnext_runtime = 696,                   /* -fnext-runtime */
  OPT_fnil_receivers = 697,                  /* -fnil-receivers */
  OPT_fnon_call_exceptions = 698,            /* -fnon-call-exceptions */
  OPT_fnonansi_builtins = 699,               /* -fnonansi-builtins */
  /* OPT_fnonnull_objects = 700, */          /* -fnonnull-objects */
  OPT_fnothrow_opt = 701,                    /* -fnothrow-opt */
  OPT_fobjc_abi_version_ = 702,              /* -fobjc-abi-version= */
  OPT_fobjc_call_cxx_cdtors = 703,           /* -fobjc-call-cxx-cdtors */
  OPT_fobjc_direct_dispatch = 704,           /* -fobjc-direct-dispatch */
  OPT_fobjc_exceptions = 705,                /* -fobjc-exceptions */
  OPT_fobjc_gc = 706,                        /* -fobjc-gc */
  OPT_fobjc_nilcheck = 707,                  /* -fobjc-nilcheck */
  OPT_fobjc_sjlj_exceptions = 708,           /* -fobjc-sjlj-exceptions */
  OPT_fobjc_std_objc1 = 709,                 /* -fobjc-std=objc1 */
  OPT_fomit_frame_pointer = 710,             /* -fomit-frame-pointer */
  OPT_fopenmp = 711,                         /* -fopenmp */
  OPT_foperator_names = 712,                 /* -foperator-names */
  OPT_foptimize_register_move = 713,         /* -foptimize-register-move */
  OPT_foptimize_sibling_calls = 714,         /* -foptimize-sibling-calls */
  OPT_foptimize_static_class_initialization = 715,/* -foptimize-static-class-initialization */
  OPT_foptimize_strlen = 716,                /* -foptimize-strlen */
  /* OPT_foptional_diags = 717, */           /* -foptional-diags */
  OPT_foutput_class_dir_ = 718,              /* -foutput-class-dir= */
  OPT_fpack_derived = 719,                   /* -fpack-derived */
  OPT_fpack_struct = 720,                    /* -fpack-struct */
  OPT_fpack_struct_ = 721,                   /* -fpack-struct= */
  OPT_fpartial_inlining = 722,               /* -fpartial-inlining */
  OPT_fpcc_struct_return = 723,              /* -fpcc-struct-return */
  OPT_fpch_deps = 724,                       /* -fpch-deps */
  OPT_fpch_preprocess = 725,                 /* -fpch-preprocess */
  OPT_fpeel_loops = 726,                     /* -fpeel-loops */
  OPT_fpeephole = 727,                       /* -fpeephole */
  OPT_fpeephole2 = 728,                      /* -fpeephole2 */
  OPT_fpermissive = 729,                     /* -fpermissive */
  OPT_fpic = 730,                            /* -fpic */
  OPT_fpie = 731,                            /* -fpie */
  OPT_fplan9_extensions = 732,               /* -fplan9-extensions */
  OPT_fplugin_arg_ = 733,                    /* -fplugin-arg- */
  OPT_fplugin_ = 734,                        /* -fplugin= */
  OPT_fpost_ipa_mem_report = 735,            /* -fpost-ipa-mem-report */
  OPT_fpre_ipa_mem_report = 736,             /* -fpre-ipa-mem-report */
  OPT_fpredictive_commoning = 737,           /* -fpredictive-commoning */
  OPT_fprefetch_loop_arrays = 738,           /* -fprefetch-loop-arrays */
  OPT_fpreprocessed = 739,                   /* -fpreprocessed */
  OPT_fpretty_templates = 740,               /* -fpretty-templates */
  OPT_fprofile = 741,                        /* -fprofile */
  OPT_fprofile_arcs = 742,                   /* -fprofile-arcs */
  OPT_fprofile_correction = 743,             /* -fprofile-correction */
  OPT_fprofile_dir_ = 744,                   /* -fprofile-dir= */
  OPT_fprofile_generate = 745,               /* -fprofile-generate */
  OPT_fprofile_generate_ = 746,              /* -fprofile-generate= */
  OPT_fprofile_use = 747,                    /* -fprofile-use */
  OPT_fprofile_use_ = 748,                   /* -fprofile-use= */
  OPT_fprofile_values = 749,                 /* -fprofile-values */
  OPT_fprotect_parens = 750,                 /* -fprotect-parens */
  OPT_frandom_seed = 751,                    /* -frandom-seed */
  OPT_frandom_seed_ = 752,                   /* -frandom-seed= */
  OPT_frange_check = 753,                    /* -frange-check */
  OPT_freal_4_real_10 = 754,                 /* -freal-4-real-10 */
  OPT_freal_4_real_16 = 755,                 /* -freal-4-real-16 */
  OPT_freal_4_real_8 = 756,                  /* -freal-4-real-8 */
  OPT_freal_8_real_10 = 757,                 /* -freal-8-real-10 */
  OPT_freal_8_real_16 = 758,                 /* -freal-8-real-16 */
  OPT_freal_8_real_4 = 759,                  /* -freal-8-real-4 */
  OPT_frealloc_lhs = 760,                    /* -frealloc-lhs */
  OPT_freciprocal_math = 761,                /* -freciprocal-math */
  OPT_frecord_gcc_switches = 762,            /* -frecord-gcc-switches */
  OPT_frecord_marker_4 = 763,                /* -frecord-marker=4 */
  OPT_frecord_marker_8 = 764,                /* -frecord-marker=8 */
  OPT_frecursive = 765,                      /* -frecursive */
  OPT_freduced_reflection = 766,             /* -freduced-reflection */
  OPT_free = 767,                            /* -free */
  OPT_freg_struct_return = 768,              /* -freg-struct-return */
  OPT_fregmove = 769,                        /* -fregmove */
  OPT_frename_registers = 770,               /* -frename-registers */
  OPT_freorder_blocks = 771,                 /* -freorder-blocks */
  OPT_freorder_blocks_and_partition = 772,   /* -freorder-blocks-and-partition */
  OPT_freorder_functions = 773,              /* -freorder-functions */
  OPT_frepack_arrays = 774,                  /* -frepack-arrays */
  OPT_freplace_objc_classes = 775,           /* -freplace-objc-classes */
  OPT_frepo = 776,                           /* -frepo */
  OPT_frequire_return_statement = 777,       /* -frequire-return-statement */
  OPT_frerun_cse_after_loop = 778,           /* -frerun-cse-after-loop */
  /* OPT_frerun_loop_opt = 779, */           /* -frerun-loop-opt */
  OPT_freschedule_modulo_scheduled_loops = 780,/* -freschedule-modulo-scheduled-loops */
  OPT_fresolution_ = 781,                    /* -fresolution= */
  OPT_frounding_math = 782,                  /* -frounding-math */
  OPT_frtti = 783,                           /* -frtti */
  OPT_fsaw_java_file = 784,                  /* -fsaw-java-file */
  OPT_fsched_critical_path_heuristic = 785,  /* -fsched-critical-path-heuristic */
  OPT_fsched_dep_count_heuristic = 786,      /* -fsched-dep-count-heuristic */
  OPT_fsched_group_heuristic = 787,          /* -fsched-group-heuristic */
  OPT_fsched_interblock = 788,               /* -fsched-interblock */
  OPT_fsched_last_insn_heuristic = 789,      /* -fsched-last-insn-heuristic */
  OPT_fsched_pressure = 790,                 /* -fsched-pressure */
  OPT_fsched_rank_heuristic = 791,           /* -fsched-rank-heuristic */
  OPT_fsched_spec = 792,                     /* -fsched-spec */
  OPT_fsched_spec_insn_heuristic = 793,      /* -fsched-spec-insn-heuristic */
  OPT_fsched_spec_load = 794,                /* -fsched-spec-load */
  OPT_fsched_spec_load_dangerous = 795,      /* -fsched-spec-load-dangerous */
  OPT_fsched_stalled_insns = 796,            /* -fsched-stalled-insns */
  OPT_fsched_stalled_insns_dep = 797,        /* -fsched-stalled-insns-dep */
  OPT_fsched_stalled_insns_dep_ = 798,       /* -fsched-stalled-insns-dep= */
  OPT_fsched_stalled_insns_ = 799,           /* -fsched-stalled-insns= */
  OPT_fsched_verbose_ = 800,                 /* -fsched-verbose= */
  OPT_fsched2_use_superblocks = 801,         /* -fsched2-use-superblocks */
  /* OPT_fsched2_use_traces = 802, */        /* -fsched2-use-traces */
  OPT_fschedule_insns = 803,                 /* -fschedule-insns */
  OPT_fschedule_insns2 = 804,                /* -fschedule-insns2 */
  OPT_fsecond_underscore = 805,              /* -fsecond-underscore */
  OPT_fsection_anchors = 806,                /* -fsection-anchors */
  /* OPT_fsee = 807, */                      /* -fsee */
  OPT_fsel_sched_pipelining = 808,           /* -fsel-sched-pipelining */
  OPT_fsel_sched_pipelining_outer_loops = 809,/* -fsel-sched-pipelining-outer-loops */
  OPT_fsel_sched_reschedule_pipelined = 810, /* -fsel-sched-reschedule-pipelined */
  OPT_fselective_scheduling = 811,           /* -fselective-scheduling */
  OPT_fselective_scheduling2 = 812,          /* -fselective-scheduling2 */
  OPT_fshort_double = 813,                   /* -fshort-double */
  OPT_fshort_enums = 814,                    /* -fshort-enums */
  OPT_fshort_wchar = 815,                    /* -fshort-wchar */
  OPT_fshow_column = 816,                    /* -fshow-column */
  OPT_fshrink_wrap = 817,                    /* -fshrink-wrap */
  OPT_fsign_zero = 818,                      /* -fsign-zero */
  OPT_fsignaling_nans = 819,                 /* -fsignaling-nans */
  OPT_fsigned_bitfields = 820,               /* -fsigned-bitfields */
  OPT_fsigned_char = 821,                    /* -fsigned-char */
  OPT_fsigned_zeros = 822,                   /* -fsigned-zeros */
  OPT_fsingle_precision_constant = 823,      /* -fsingle-precision-constant */
  OPT_fsource_filename_ = 824,               /* -fsource-filename= */
  OPT_fsource_ = 825,                        /* -fsource= */
  OPT_fsplit_ivs_in_unroller = 826,          /* -fsplit-ivs-in-unroller */
  OPT_fsplit_stack = 827,                    /* -fsplit-stack */
  OPT_fsplit_wide_types = 828,               /* -fsplit-wide-types */
  /* OPT_fsquangle = 829, */                 /* -fsquangle */
  OPT_fstack_arrays = 830,                   /* -fstack-arrays */
  /* OPT_fstack_check = 831, */              /* -fstack-check */
  OPT_fstack_check_ = 832,                   /* -fstack-check= */
  OPT_fstack_limit = 833,                    /* -fstack-limit */
  OPT_fstack_limit_register_ = 834,          /* -fstack-limit-register= */
  OPT_fstack_limit_symbol_ = 835,            /* -fstack-limit-symbol= */
  OPT_fstack_protector = 836,                /* -fstack-protector */
  OPT_fstack_protector_all = 837,            /* -fstack-protector-all */
  OPT_fstack_usage = 838,                    /* -fstack-usage */
  OPT_fstats = 839,                          /* -fstats */
  OPT_fstore_check = 840,                    /* -fstore-check */
  /* OPT_fstrength_reduce = 841, */          /* -fstrength-reduce */
  OPT_fstrict_aliasing = 842,                /* -fstrict-aliasing */
  OPT_fstrict_enums = 843,                   /* -fstrict-enums */
  OPT_fstrict_overflow = 844,                /* -fstrict-overflow */
  /* OPT_fstrict_prototype = 845, */         /* -fstrict-prototype */
  OPT_fstrict_volatile_bitfields = 846,      /* -fstrict-volatile-bitfields */
  OPT_fsyntax_only = 847,                    /* -fsyntax-only */
  OPT_ftabstop_ = 848,                       /* -ftabstop= */
  /* OPT_ftarget_help = 849, */              /* -ftarget-help */
  OPT_ftarget_ = 850,                        /* -ftarget= */
  /* OPT_ftemplate_depth_ = 851, */          /* -ftemplate-depth- */
  OPT_ftemplate_depth_ = 852,                /* -ftemplate-depth= */
  OPT_ftest_coverage = 853,                  /* -ftest-coverage */
  /* OPT_fthis_is_variable = 854, */         /* -fthis-is-variable */
  OPT_fthread_jumps = 855,                   /* -fthread-jumps */
  OPT_fthreadsafe_statics = 856,             /* -fthreadsafe-statics */
  OPT_ftime_report = 857,                    /* -ftime-report */
  OPT_ftls_model_ = 858,                     /* -ftls-model= */
  OPT_ftoplevel_reorder = 859,               /* -ftoplevel-reorder */
  OPT_ftracer = 860,                         /* -ftracer */
  OPT_ftrack_macro_expansion = 861,          /* -ftrack-macro-expansion */
  OPT_ftrack_macro_expansion_ = 862,         /* -ftrack-macro-expansion= */
  OPT_ftrapping_math = 863,                  /* -ftrapping-math */
  OPT_ftrapv = 864,                          /* -ftrapv */
  OPT_ftree_bit_ccp = 865,                   /* -ftree-bit-ccp */
  OPT_ftree_builtin_call_dce = 866,          /* -ftree-builtin-call-dce */
  OPT_ftree_ccp = 867,                       /* -ftree-ccp */
  OPT_ftree_ch = 868,                        /* -ftree-ch */
  OPT_ftree_copy_prop = 869,                 /* -ftree-copy-prop */
  OPT_ftree_copyrename = 870,                /* -ftree-copyrename */
  OPT_ftree_cselim = 871,                    /* -ftree-cselim */
  OPT_ftree_dce = 872,                       /* -ftree-dce */
  OPT_ftree_dominator_opts = 873,            /* -ftree-dominator-opts */
  OPT_ftree_dse = 874,                       /* -ftree-dse */
  OPT_ftree_forwprop = 875,                  /* -ftree-forwprop */
  OPT_ftree_fre = 876,                       /* -ftree-fre */
  OPT_ftree_loop_distribute_patterns = 877,  /* -ftree-loop-distribute-patterns */
  OPT_ftree_loop_distribution = 878,         /* -ftree-loop-distribution */
  OPT_ftree_loop_if_convert = 879,           /* -ftree-loop-if-convert */
  OPT_ftree_loop_if_convert_stores = 880,    /* -ftree-loop-if-convert-stores */
  OPT_ftree_loop_im = 881,                   /* -ftree-loop-im */
  OPT_ftree_loop_ivcanon = 882,              /* -ftree-loop-ivcanon */
  /* OPT_ftree_loop_linear = 883, */         /* -ftree-loop-linear */
  OPT_ftree_loop_optimize = 884,             /* -ftree-loop-optimize */
  OPT_ftree_lrs = 885,                       /* -ftree-lrs */
  OPT_ftree_parallelize_loops_ = 886,        /* -ftree-parallelize-loops= */
  OPT_ftree_phiprop = 887,                   /* -ftree-phiprop */
  OPT_ftree_pre = 888,                       /* -ftree-pre */
  OPT_ftree_pta = 889,                       /* -ftree-pta */
  OPT_ftree_reassoc = 890,                   /* -ftree-reassoc */
  /* OPT_ftree_salias = 891, */              /* -ftree-salias */
  OPT_ftree_scev_cprop = 892,                /* -ftree-scev-cprop */
  OPT_ftree_sink = 893,                      /* -ftree-sink */
  OPT_ftree_slp_vectorize = 894,             /* -ftree-slp-vectorize */
  OPT_ftree_sra = 895,                       /* -ftree-sra */
  /* OPT_ftree_store_ccp = 896, */           /* -ftree-store-ccp */
  /* OPT_ftree_store_copy_prop = 897, */     /* -ftree-store-copy-prop */
  OPT_ftree_switch_conversion = 898,         /* -ftree-switch-conversion */
  OPT_ftree_tail_merge = 899,                /* -ftree-tail-merge */
  OPT_ftree_ter = 900,                       /* -ftree-ter */
  OPT_ftree_vect_loop_version = 901,         /* -ftree-vect-loop-version */
  OPT_ftree_vectorize = 902,                 /* -ftree-vectorize */
  OPT_ftree_vectorizer_verbose_ = 903,       /* -ftree-vectorizer-verbose= */
  OPT_ftree_vrp = 904,                       /* -ftree-vrp */
  OPT_funderscoring = 905,                   /* -funderscoring */
  OPT_funit_at_a_time = 906,                 /* -funit-at-a-time */
  OPT_funroll_all_loops = 907,               /* -funroll-all-loops */
  OPT_funroll_loops = 908,                   /* -funroll-loops */
  OPT_funsafe_loop_optimizations = 909,      /* -funsafe-loop-optimizations */
  OPT_funsafe_math_optimizations = 910,      /* -funsafe-math-optimizations */
  OPT_funsigned_bitfields = 911,             /* -funsigned-bitfields */
  OPT_funsigned_char = 912,                  /* -funsigned-char */
  OPT_funswitch_loops = 913,                 /* -funswitch-loops */
  OPT_funwind_tables = 914,                  /* -funwind-tables */
  OPT_fuse_atomic_builtins = 915,            /* -fuse-atomic-builtins */
  OPT_fuse_boehm_gc = 916,                   /* -fuse-boehm-gc */
  OPT_fuse_cxa_atexit = 917,                 /* -fuse-cxa-atexit */
  OPT_fuse_cxa_get_exception_ptr = 918,      /* -fuse-cxa-get-exception-ptr */
  OPT_fuse_divide_subroutine = 919,          /* -fuse-divide-subroutine */
  OPT_fuse_linker_plugin = 920,              /* -fuse-linker-plugin */
  OPT_fvar_tracking = 921,                   /* -fvar-tracking */
  OPT_fvar_tracking_assignments = 922,       /* -fvar-tracking-assignments */
  OPT_fvar_tracking_assignments_toggle = 923,/* -fvar-tracking-assignments-toggle */
  OPT_fvar_tracking_uninit = 924,            /* -fvar-tracking-uninit */
  OPT_fvariable_expansion_in_unroller = 925, /* -fvariable-expansion-in-unroller */
  OPT_fvect_cost_model = 926,                /* -fvect-cost-model */
  OPT_fverbose_asm = 927,                    /* -fverbose-asm */
  /* OPT_fversion = 928, */                  /* -fversion */
  OPT_fvisibility_inlines_hidden = 929,      /* -fvisibility-inlines-hidden */
  OPT_fvisibility_ms_compat = 930,           /* -fvisibility-ms-compat */
  OPT_fvisibility_ = 931,                    /* -fvisibility= */
  OPT_fvpt = 932,                            /* -fvpt */
  /* OPT_fvtable_gc = 933, */                /* -fvtable-gc */
  /* OPT_fvtable_thunks = 934, */            /* -fvtable-thunks */
  OPT_fweak = 935,                           /* -fweak */
  OPT_fweb = 936,                            /* -fweb */
  OPT_fwhole_file = 937,                     /* -fwhole-file */
  OPT_fwhole_program = 938,                  /* -fwhole-program */
  OPT_fwide_exec_charset_ = 939,             /* -fwide-exec-charset= */
  OPT_fworking_directory = 940,              /* -fworking-directory */
  OPT_fwpa = 941,                            /* -fwpa */
  OPT_fwrapv = 942,                          /* -fwrapv */
  /* OPT_fxref = 943, */                     /* -fxref */
  /* OPT_fzee = 944, */                      /* -fzee */
  OPT_fzero_initialized_in_bss = 945,        /* -fzero-initialized-in-bss */
  OPT_fzero_link = 946,                      /* -fzero-link */
  OPT_g = 947,                               /* -g */
  OPT_gant = 948,                            /* -gant */
  OPT_gcoff = 949,                           /* -gcoff */
  OPT_gdwarf_ = 950,                         /* -gdwarf- */
  OPT_gen_decls = 951,                       /* -gen-decls */
  OPT_ggdb = 952,                            /* -ggdb */
  OPT_gnat = 953,                            /* -gnat */
  OPT_gnatO = 954,                           /* -gnatO */
  OPT_gno_record_gcc_switches = 955,         /* -gno-record-gcc-switches */
  OPT_gno_strict_dwarf = 956,                /* -gno-strict-dwarf */
  OPT_grecord_gcc_switches = 957,            /* -grecord-gcc-switches */
  OPT_gstabs = 958,                          /* -gstabs */
  OPT_gstabs_ = 959,                         /* -gstabs+ */
  OPT_gstrict_dwarf = 960,                   /* -gstrict-dwarf */
  OPT_gtoggle = 961,                         /* -gtoggle */
  OPT_gvms = 962,                            /* -gvms */
  OPT_gxcoff = 963,                          /* -gxcoff */
  OPT_gxcoff_ = 964,                         /* -gxcoff+ */
  OPT_h = 965,                               /* -h */
  OPT_idirafter = 966,                       /* -idirafter */
  OPT_imacros = 967,                         /* -imacros */
  OPT_imultilib = 968,                       /* -imultilib */
  OPT_include = 969,                         /* -include */
  OPT_iplugindir_ = 970,                     /* -iplugindir= */
  OPT_iprefix = 971,                         /* -iprefix */
  OPT_iquote = 972,                          /* -iquote */
  OPT_isysroot = 973,                        /* -isysroot */
  OPT_isystem = 974,                         /* -isystem */
  OPT_iwithprefix = 975,                     /* -iwithprefix */
  OPT_iwithprefixbefore = 976,               /* -iwithprefixbefore */
  OPT_k8 = 977,                              /* -k8 */
  OPT_l = 978,                               /* -l */
  OPT_lang_asm = 979,                        /* -lang-asm */
  OPT_m128bit_long_double = 980,             /* -m128bit-long-double */
  OPT_m32 = 981,                             /* -m32 */
  OPT_m3dnow = 982,                          /* -m3dnow */
  OPT_m3dnowa = 983,                         /* -m3dnowa */
  OPT_m64 = 984,                             /* -m64 */
  OPT_m80387 = 985,                          /* -m80387 */
  OPT_m8bit_idiv = 986,                      /* -m8bit-idiv */
  OPT_m96bit_long_double = 987,              /* -m96bit-long-double */
  OPT_mabi_ = 988,                           /* -mabi= */
  OPT_mabm = 989,                            /* -mabm */
  OPT_maccumulate_outgoing_args = 990,       /* -maccumulate-outgoing-args */
  OPT_maes = 991,                            /* -maes */
  OPT_malign_double = 992,                   /* -malign-double */
  OPT_malign_functions_ = 993,               /* -malign-functions= */
  OPT_malign_jumps_ = 994,                   /* -malign-jumps= */
  OPT_malign_loops_ = 995,                   /* -malign-loops= */
  OPT_malign_stringops = 996,                /* -malign-stringops */
  OPT_mandroid = 997,                        /* -mandroid */
  OPT_march_ = 998,                          /* -march= */
  OPT_masm_ = 999,                           /* -masm= */
  OPT_mavx = 1000,                           /* -mavx */
  OPT_mavx2 = 1001,                          /* -mavx2 */
  OPT_mavx256_split_unaligned_load = 1002,   /* -mavx256-split-unaligned-load */
  OPT_mavx256_split_unaligned_store = 1003,  /* -mavx256-split-unaligned-store */
  OPT_mbionic = 1004,                        /* -mbionic */
  OPT_mbmi = 1005,                           /* -mbmi */
  OPT_mbmi2 = 1006,                          /* -mbmi2 */
  OPT_mbranch_cost_ = 1007,                  /* -mbranch-cost= */
  OPT_mcld = 1008,                           /* -mcld */
  OPT_mcmodel_ = 1009,                       /* -mcmodel= */
  /* OPT_mcpu_ = 1010, */                    /* -mcpu= */
  OPT_mcrc32 = 1011,                         /* -mcrc32 */
  OPT_mcx16 = 1012,                          /* -mcx16 */
  OPT_mdispatch_scheduler = 1013,            /* -mdispatch-scheduler */
  OPT_mf16c = 1014,                          /* -mf16c */
  OPT_mfancy_math_387 = 1015,                /* -mfancy-math-387 */
  OPT_mfentry = 1016,                        /* -mfentry */
  OPT_mfma = 1017,                           /* -mfma */
  OPT_mfma4 = 1018,                          /* -mfma4 */
  OPT_mforce_drap = 1019,                    /* -mforce-drap */
  OPT_mfp_ret_in_387 = 1020,                 /* -mfp-ret-in-387 */
  OPT_mfpmath_ = 1021,                       /* -mfpmath= */
  OPT_mfsgsbase = 1022,                      /* -mfsgsbase */
  /* OPT_mfused_madd = 1023, */              /* -mfused-madd */
  OPT_mglibc = 1024,                         /* -mglibc */
  OPT_mhard_float = 1025,                    /* -mhard-float */
  OPT_mieee_fp = 1026,                       /* -mieee-fp */
  OPT_mincoming_stack_boundary_ = 1027,      /* -mincoming-stack-boundary= */
  OPT_minline_all_stringops = 1028,          /* -minline-all-stringops */
  OPT_minline_stringops_dynamically = 1029,  /* -minline-stringops-dynamically */
  /* OPT_mintel_syntax = 1030, */            /* -mintel-syntax */
  OPT_mlarge_data_threshold_ = 1031,         /* -mlarge-data-threshold= */
  OPT_mlwp = 1032,                           /* -mlwp */
  OPT_mlzcnt = 1033,                         /* -mlzcnt */
  OPT_mmmx = 1034,                           /* -mmmx */
  OPT_mmovbe = 1035,                         /* -mmovbe */
  OPT_mms_bitfields = 1036,                  /* -mms-bitfields */
  OPT_mno_align_stringops = 1037,            /* -mno-align-stringops */
  OPT_mno_fancy_math_387 = 1038,             /* -mno-fancy-math-387 */
  OPT_mno_push_args = 1039,                  /* -mno-push-args */
  OPT_mno_red_zone = 1040,                   /* -mno-red-zone */
  OPT_mno_sse4 = 1041,                       /* -mno-sse4 */
  OPT_momit_leaf_frame_pointer = 1042,       /* -momit-leaf-frame-pointer */
  OPT_mpc32 = 1043,                          /* -mpc32 */
  OPT_mpc64 = 1044,                          /* -mpc64 */
  OPT_mpc80 = 1045,                          /* -mpc80 */
  OPT_mpclmul = 1046,                        /* -mpclmul */
  OPT_mpopcnt = 1047,                        /* -mpopcnt */
  OPT_mprefer_avx128 = 1048,                 /* -mprefer-avx128 */
  OPT_mpreferred_stack_boundary_ = 1049,     /* -mpreferred-stack-boundary= */
  OPT_mpush_args = 1050,                     /* -mpush-args */
  OPT_mrdrnd = 1051,                         /* -mrdrnd */
  OPT_mrecip = 1052,                         /* -mrecip */
  OPT_mrecip_ = 1053,                        /* -mrecip= */
  OPT_mred_zone = 1054,                      /* -mred-zone */
  OPT_mregparm_ = 1055,                      /* -mregparm= */
  OPT_mrtd = 1056,                           /* -mrtd */
  OPT_msahf = 1057,                          /* -msahf */
  OPT_msoft_float = 1058,                    /* -msoft-float */
  OPT_msse = 1059,                           /* -msse */
  OPT_msse2 = 1060,                          /* -msse2 */
  OPT_msse2avx = 1061,                       /* -msse2avx */
  OPT_msse3 = 1062,                          /* -msse3 */
  OPT_msse4 = 1063,                          /* -msse4 */
  OPT_msse4_1 = 1064,                        /* -msse4.1 */
  OPT_msse4_2 = 1065,                        /* -msse4.2 */
  OPT_msse4a = 1066,                         /* -msse4a */
  /* OPT_msse5 = 1067, */                    /* -msse5 */
  OPT_msseregparm = 1068,                    /* -msseregparm */
  OPT_mssse3 = 1069,                         /* -mssse3 */
  OPT_mstack_arg_probe = 1070,               /* -mstack-arg-probe */
  OPT_mstackrealign = 1071,                  /* -mstackrealign */
  OPT_mstringop_strategy_ = 1072,            /* -mstringop-strategy= */
  OPT_mtbm = 1073,                           /* -mtbm */
  OPT_mtls_dialect_ = 1074,                  /* -mtls-dialect= */
  OPT_mtls_direct_seg_refs = 1075,           /* -mtls-direct-seg-refs */
  OPT_mtune_ = 1076,                         /* -mtune= */
  OPT_muclibc = 1077,                        /* -muclibc */
  OPT_mveclibabi_ = 1078,                    /* -mveclibabi= */
  OPT_mvect8_ret_in_mem = 1079,              /* -mvect8-ret-in-mem */
  OPT_mvzeroupper = 1080,                    /* -mvzeroupper */
  OPT_mx32 = 1081,                           /* -mx32 */
  OPT_mxop = 1082,                           /* -mxop */
  OPT_n = 1083,                              /* -n */
  OPT_no_canonical_prefixes = 1084,          /* -no-canonical-prefixes */
  OPT_no_integrated_cpp = 1085,              /* -no-integrated-cpp */
  OPT_nocpp = 1086,                          /* -nocpp */
  OPT_nodefaultlibs = 1087,                  /* -nodefaultlibs */
  OPT_nostartfiles = 1088,                   /* -nostartfiles */
  OPT_nostdinc = 1089,                       /* -nostdinc */
  OPT_nostdinc__ = 1090,                     /* -nostdinc++ */
  OPT_nostdlib = 1091,                       /* -nostdlib */
  OPT_o = 1092,                              /* -o */
  OPT_p = 1093,                              /* -p */
  OPT_pass_exit_codes = 1094,                /* -pass-exit-codes */
  OPT_pedantic = 1095,                       /* -pedantic */
  OPT_pedantic_errors = 1096,                /* -pedantic-errors */
  OPT_pg = 1097,                             /* -pg */
  OPT_pie = 1098,                            /* -pie */
  OPT_pipe = 1099,                           /* -pipe */
  OPT_posix = 1100,                          /* -posix */
  OPT_print_file_name_ = 1101,               /* -print-file-name= */
  OPT_print_libgcc_file_name = 1102,         /* -print-libgcc-file-name */
  OPT_print_multi_directory = 1103,          /* -print-multi-directory */
  OPT_print_multi_lib = 1104,                /* -print-multi-lib */
  OPT_print_multi_os_directory = 1105,       /* -print-multi-os-directory */
  OPT_print_objc_runtime_info = 1106,        /* -print-objc-runtime-info */
  OPT_print_prog_name_ = 1107,               /* -print-prog-name= */
  OPT_print_search_dirs = 1108,              /* -print-search-dirs */
  OPT_print_sysroot = 1109,                  /* -print-sysroot */
  OPT_print_sysroot_headers_suffix = 1110,   /* -print-sysroot-headers-suffix */
  OPT_profile = 1111,                        /* -profile */
  OPT_pthread = 1112,                        /* -pthread */
  OPT_quiet = 1113,                          /* -quiet */
  OPT_r = 1114,                              /* -r */
  OPT_rdynamic = 1115,                       /* -rdynamic */
  OPT_remap = 1116,                          /* -remap */
  OPT_s = 1117,                              /* -s */
  OPT_s_bc_abi = 1118,                       /* -s-bc-abi */
  OPT_save_temps = 1119,                     /* -save-temps */
  OPT_save_temps_ = 1120,                    /* -save-temps= */
  OPT_shared = 1121,                         /* -shared */
  OPT_shared_libgcc = 1122,                  /* -shared-libgcc */
  /* OPT_specs = 1123, */                    /* -specs */
  OPT_specs_ = 1124,                         /* -specs= */
  OPT_static = 1125,                         /* -static */
  OPT_static_libgcc = 1126,                  /* -static-libgcc */
  OPT_static_libgcj = 1127,                  /* -static-libgcj */
  OPT_static_libgfortran = 1128,             /* -static-libgfortran */
  OPT_static_libgo = 1129,                   /* -static-libgo */
  OPT_static_libstdc__ = 1130,               /* -static-libstdc++ */
  /* OPT_std_c__03 = 1131, */                /* -std=c++03 */
  /* OPT_std_c__0x = 1132, */                /* -std=c++0x */
  OPT_std_c__11 = 1133,                      /* -std=c++11 */
  OPT_std_c__98 = 1134,                      /* -std=c++98 */
  OPT_std_c11 = 1135,                        /* -std=c11 */
  /* OPT_std_c1x = 1136, */                  /* -std=c1x */
  /* OPT_std_c89 = 1137, */                  /* -std=c89 */
  OPT_std_c90 = 1138,                        /* -std=c90 */
  OPT_std_c99 = 1139,                        /* -std=c99 */
  /* OPT_std_c9x = 1140, */                  /* -std=c9x */
  OPT_std_f2003 = 1141,                      /* -std=f2003 */
  OPT_std_f2008 = 1142,                      /* -std=f2008 */
  OPT_std_f2008ts = 1143,                    /* -std=f2008ts */
  OPT_std_f95 = 1144,                        /* -std=f95 */
  OPT_std_gnu = 1145,                        /* -std=gnu */
  /* OPT_std_gnu__03 = 1146, */              /* -std=gnu++03 */
  /* OPT_std_gnu__0x = 1147, */              /* -std=gnu++0x */
  OPT_std_gnu__11 = 1148,                    /* -std=gnu++11 */
  OPT_std_gnu__98 = 1149,                    /* -std=gnu++98 */
  OPT_std_gnu11 = 1150,                      /* -std=gnu11 */
  /* OPT_std_gnu1x = 1151, */                /* -std=gnu1x */
  /* OPT_std_gnu89 = 1152, */                /* -std=gnu89 */
  OPT_std_gnu90 = 1153,                      /* -std=gnu90 */
  OPT_std_gnu99 = 1154,                      /* -std=gnu99 */
  /* OPT_std_gnu9x = 1155, */                /* -std=gnu9x */
  /* OPT_std_iso9899_1990 = 1156, */         /* -std=iso9899:1990 */
  OPT_std_iso9899_199409 = 1157,             /* -std=iso9899:199409 */
  /* OPT_std_iso9899_1999 = 1158, */         /* -std=iso9899:1999 */
  /* OPT_std_iso9899_199x = 1159, */         /* -std=iso9899:199x */
  /* OPT_std_iso9899_2011 = 1160, */         /* -std=iso9899:2011 */
  OPT_std_legacy = 1161,                     /* -std=legacy */
  OPT_symbolic = 1162,                       /* -symbolic */
  OPT_t = 1163,                              /* -t */
  OPT_time = 1164,                           /* -time */
  OPT_time_ = 1165,                          /* -time= */
  OPT_tno_android_cc = 1166,                 /* -tno-android-cc */
  OPT_tno_android_ld = 1167,                 /* -tno-android-ld */
  OPT_traditional = 1168,                    /* -traditional */
  OPT_traditional_cpp = 1169,                /* -traditional-cpp */
  OPT_trigraphs = 1170,                      /* -trigraphs */
  OPT_u = 1171,                              /* -u */
  OPT_undef = 1172,                          /* -undef */
  OPT_v = 1173,                              /* -v */
  OPT_version = 1174,                        /* -version */
  OPT_w = 1175,                              /* -w */
  OPT_wrapper = 1176,                        /* -wrapper */
  OPT_x = 1177,                              /* -x */
  OPT_z = 1178,                              /* -z */
  N_OPTS,
  OPT_SPECIAL_unknown,
  OPT_SPECIAL_ignore,
  OPT_SPECIAL_program_name,
  OPT_SPECIAL_input_file
};

/* Structure describing the result of decoding an option.  */

struct cl_decoded_option
{
  /* The index of this option, or an OPT_SPECIAL_* value for
     non-options and unknown options.  */
  size_t opt_index;

  /* Any warning to give for use of this option, or NULL if none.  */
  const char *warn_message;

  /* The string argument, or NULL if none.  For OPT_SPECIAL_* cases,
     the option or non-option command-line argument.  */
  const char *arg;

  /* The original text of option plus arguments, with separate argv
     elements concatenated into one string with spaces separating
     them.  This is for such uses as diagnostics and
     -frecord-gcc-switches.  */
  const char *orig_option_with_args_text;

  /* The canonical form of the option and its argument, for when it is
     necessary to reconstruct argv elements (in particular, for
     processing specs and passing options to subprocesses from the
     driver).  */
  const char *canonical_option[4];

  /* The number of elements in the canonical form of the option and
     arguments; always at least 1.  */
  size_t canonical_option_num_elements;

  /* For a boolean option, 1 for the true case and 0 for the "no-"
     case.  For an unsigned integer option, the value of the
     argument.  1 in all other cases.  */
  int value;

  /* Any flags describing errors detected in this option.  */
  int errors;
};




/**********/
/* intl.h */
/**********/


#ifndef _
# define _(msgid) gettext (msgid)
#endif


#endif /* IMPORT_MODULE_H  */
