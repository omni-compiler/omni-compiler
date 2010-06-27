/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-gcctype.h
 */

#ifndef _C_GCCATTR_H_
#define _C_GCCATTR_H_

/**
 * GCC attribute kind
 */

/**
 * \brief
 * GCC attribute kinds
 */
typedef enum {
    GAK_UNDEF       = 0,
    GAK_FUNC        = 1,
    GAK_VAR         = 1 << 1,
    GAK_TYPE        = 1 << 2,
    GAK_FUNC_DEF    = 1 << 3,
} CGccAttrKindEnum;

#define GAK_ALL (GAK_FUNC|GAK_VAR|GAK_TYPE)

#define CGccAttrKindEnumNamesDef {\
    "UNDEF",\
    "FUNC",\
    "VAR",\
    "FUNC|VAR",\
    "TYPE",\
    "TYPE|FUNC",\
    "TYPE|VAR",\
    "TYPE|VAR|FUNC",\
    "FUNC_DEF",\
    "END",\
}

/**
 * \brief
 * GCC attribute ids
 */

typedef enum {
    GA_ALIAS,
    GA_ALLOC_SIZE,
    GA_ALWAYS_INLINE,
    GA_ARTIFICIAL,
    GA_COLD,
    GA_CONST,
    GA_CONSTRUCTOR,
    GA_DESTRUCTOR,
    GA_ERROR,
    GA_EXTERNALLY_VISIBLE,
    GA_FLATTEN,
    GA_FORMAT,
    GA_FORMAT_ARG,
    GA_GNU_INLINE,
    GA_HOT,
    GA_MALLOC,
    GA_NO_INSTRUMENT_FUNCTION,
    GA_NOINLINE,
    GA_NONNULL,
    GA_NORETURN,
    GA_NOTHROW,
    GA_PURE,
    GA_RETURNS_TWICE,
    GA_SENTINEL,
    GA_WARN_UNUSED_RESULT,
    GA_WARNING,
    GA_WEAKREF,
    GA_CDECL,
    GA_EIGHTBIT_DATA,
    GA_EXCEPTION_HANDLER,
    GA_FAR,
    GA_FASTCALL,
    GA_FORCE_ALIGN_ARG_POINTER,
    GA_FUNCTION_VECTOR,
    GA_INTERRUPT,
    GA_INTERRUPT_HANDLER,
    GA_INTERRUPT_THREAD,
    GA_KSPISUSP,
    GA_L1_TEXT,
    GA_LONG_CALL,
    GA_LONGCALL,
    GA_MIPS16,
    GA_NAKED,
    GA_NEAR,
    GA_NESTING,
    GA_NMI_HANDLER,
    GA_NOMIPS16,
    GA_NOTSHARED,
    GA_REGPARM,
    GA_SAVEALL,
    GA_SHORT_CALL,
    GA_SHORTCALL,
    GA_SIGNAL,
    GA_SP_SWITCH,
    GA_SSEREGPARM,
    GA_STDCALL,
    GA_TINY_DATA,
    GA_TRAP_EXIT,
    GA_VERSION_ID,
    GA_CLEANUP,
    GA_COMMON,
    GA_NOCOMMON,
    GA_MODE,
    GA_SHARED,
    GA_TLS_MODEL,
    GA_VECTOR_SIZE,
    GA_SELECTANY,
    GA_L1_DATA,
    GA_L1_DATA_A,
    GA_L1_DATA_B,
    GA_MODEL,
    GA_BELOW100,
    GA_PROGMEM,
    GA_TRANSPARENT_UNION,
    GA_VISIBILITY,
    GA_MAY_ALIAS,
    GA_SECTION,
    GA_USED,
    GA_WEAK,
    GA_DLLIMPORT,
    GA_DLLEXPORT,
    GA_PACKED,
    GA_ALIGNED,
    GA_UNUSED,
    GA_DEPRECATED,
    GA_SPU_VECTOR,
    GA_GCC_STRUCT,
    GA_MS_STRUCT,
    GA_ALTIVEC
} CGccAttrIdEnum;

/**
 * \brief
 * GCC attribute info
 */
typedef struct CGccAttrInfo {
    //! kind
    CGccAttrKindEnum    ga_kind;
    //! symbol
    const char          *ga_symbol;
    //! symbol ID
    CGccAttrIdEnum      ga_symbolId;
} CGccAttrInfo;

/**
 * \brief
 * GCC builtin function's return types
 */
typedef enum CGccBuiltinFuncTypeEnum {
    GBF_FIXED,
    GBF_ARG,
    GBF_ARG_PTRREF,
    GBF_INT64,
    GBF_VOIDPTR,
    GBF_CHARPTR,
} CGccBuiltinFuncTypeEnum;

/**
 * \brief
 * GCC builtin function info
 */
typedef struct CGccBuiltinFuncInfo {
    //! function name
    const char              *gbf_name;
    //! is const function
    unsigned int            gbf_isConst;
    //! return type for basic type
    CBasicTypeEnum          gbf_basicType;
    //! function type
    CGccBuiltinFuncTypeEnum gbf_type;

} CGccBuiltinFuncInfo;

//! GCC builtin type info array
extern const char           *s_gccBuiltinTypes[];
//! GCC builtin function info array
extern CGccBuiltinFuncInfo  s_gccBuiltinFuncs[];


#endif /* _C_GCCATTR_H_ */

