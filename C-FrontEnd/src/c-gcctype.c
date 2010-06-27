/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "c-expr.h"
#include "c-option.h"
#include "c-comp.h"

/**
 * \brief
 * GCC attributes (GCC 4.3.2)
 */
CGccAttrInfo s_gccAttrInfos[] = {
    { GAK_FUNC                 , "alloc_size"             , GA_ALLOC_SIZE },
    { GAK_FUNC                 , "always_inline"          , GA_ALWAYS_INLINE },
    { GAK_FUNC                 , "artificial"             , GA_ARTIFICIAL },
    { GAK_FUNC                 , "cold"                   , GA_COLD },
    { GAK_FUNC                 , "const"                  , GA_CONST },
    { GAK_FUNC                 , "constructor"            , GA_CONSTRUCTOR },
    { GAK_FUNC                 , "destructor"             , GA_DESTRUCTOR },
    { GAK_FUNC                 , "error"                  , GA_ERROR },
    { GAK_FUNC                 , "externally_visible"     , GA_EXTERNALLY_VISIBLE },
    { GAK_FUNC                 , "flatten"                , GA_FLATTEN },
    { GAK_FUNC                 , "format"                 , GA_FORMAT },
    { GAK_FUNC                 , "format_arg"             , GA_FORMAT_ARG },
    // gnu_inline is treated as GA_TYPE. see exprFixAttr() and
    // checkAndMergeIdent().
    { GAK_FUNC                 , "gnu_inline"             , GA_GNU_INLINE },
    { GAK_FUNC                 , "hot"                    , GA_HOT },
    { GAK_FUNC                 , "malloc"                 , GA_MALLOC },
    { GAK_FUNC                 , "no_instrument_function" , GA_NO_INSTRUMENT_FUNCTION },
    { GAK_FUNC                 , "noinline"               , GA_NOINLINE },
    { GAK_FUNC                 , "nonnull"                , GA_NONNULL },
    { GAK_FUNC                 , "noreturn"               , GA_NORETURN },
    { GAK_FUNC                 , "nothrow"                , GA_NOTHROW },
    { GAK_FUNC                 , "pure"                   , GA_PURE },
    { GAK_FUNC                 , "returns_twice"          , GA_RETURNS_TWICE },
    { GAK_FUNC                 , "sentinel"               , GA_SENTINEL },
    { GAK_FUNC                 , "warn_unused_result"     , GA_WARN_UNUSED_RESULT },
    { GAK_FUNC                 , "warning"                , GA_WARNING },
    { GAK_FUNC                 , "weakref"                , GA_WEAKREF },
    { GAK_FUNC                 , "cdecl"                  , GA_CDECL },
    { GAK_FUNC                 , "eightbit_data"          , GA_EIGHTBIT_DATA },
    { GAK_FUNC                 , "exception_handler"      , GA_EXCEPTION_HANDLER },
    { GAK_FUNC                 , "far"                    , GA_FAR },
    { GAK_FUNC                 , "fastcall"               , GA_FASTCALL },
    { GAK_FUNC                 , "force_align_arg_pointer", GA_FORCE_ALIGN_ARG_POINTER },
    { GAK_FUNC                 , "function_vector"        , GA_FUNCTION_VECTOR },
    { GAK_FUNC                 , "interrupt"              , GA_INTERRUPT },
    { GAK_FUNC                 , "interrupt_handler"      , GA_INTERRUPT_HANDLER },
    { GAK_FUNC                 , "interrupt_thread"       , GA_INTERRUPT_THREAD },
    { GAK_FUNC                 , "kspisusp"               , GA_KSPISUSP },
    { GAK_FUNC                 , "l1_text"                , GA_L1_TEXT },
    { GAK_FUNC                 , "long_call"              , GA_LONG_CALL },
    { GAK_FUNC                 , "longcall"               , GA_LONGCALL },
    { GAK_FUNC                 , "mips16"                 , GA_MIPS16 },
    { GAK_FUNC                 , "naked"                  , GA_NAKED },
    { GAK_FUNC                 , "near"                   , GA_NEAR },
    { GAK_FUNC                 , "nesting"                , GA_NESTING },
    { GAK_FUNC                 , "nmi_handler"            , GA_NMI_HANDLER },
    { GAK_FUNC                 , "nomips16"               , GA_NOMIPS16 },
    { GAK_FUNC                 , "notshared"              , GA_NOTSHARED },
    { GAK_FUNC                 , "saveall"                , GA_SAVEALL },
    { GAK_FUNC                 , "short_call"             , GA_SHORT_CALL },
    { GAK_FUNC                 , "shortcall"              , GA_SHORTCALL },
    { GAK_FUNC                 , "signal"                 , GA_SIGNAL },
    { GAK_FUNC                 , "sp_switch"              , GA_SP_SWITCH },
    { GAK_FUNC                 , "sseregparm"             , GA_SSEREGPARM },
    { GAK_FUNC                 , "stdcall"                , GA_STDCALL },
    { GAK_FUNC                 , "tiny_data"              , GA_TINY_DATA },
    { GAK_FUNC                 , "trap_exit"              , GA_TRAP_EXIT },
    { GAK_FUNC                 , "version_id"             , GA_VERSION_ID },
    { GAK_VAR                  , "cleanup"                , GA_CLEANUP },
    { GAK_VAR                  , "common"                 , GA_COMMON },
    { GAK_VAR                  , "nocommon"               , GA_NOCOMMON },
    { GAK_VAR                  , "shared"                 , GA_SHARED },
    { GAK_VAR                  , "tls_model"              , GA_TLS_MODEL },
    { GAK_VAR                  , "vector_size"            , GA_VECTOR_SIZE },
    { GAK_VAR                  , "selectany"              , GA_SELECTANY },
    { GAK_VAR                  , "l1_data"                , GA_L1_DATA },
    { GAK_VAR                  , "l1_data_A"              , GA_L1_DATA_A },
    { GAK_VAR                  , "l1_data_B"              , GA_L1_DATA_B },
    { GAK_VAR                  , "model"                  , GA_MODEL },
    { GAK_VAR                  , "below100"               , GA_BELOW100 },
    { GAK_VAR                  , "progmem"                , GA_PROGMEM },
    { GAK_TYPE                 , "transparent_union"      , GA_TRANSPARENT_UNION },
    { GAK_TYPE                 , "visibility"             , GA_VISIBILITY },
    { GAK_TYPE                 , "may_alias"              , GA_MAY_ALIAS },
    { GAK_FUNC|GAK_VAR         , "alias"                  , GA_ALIAS },
    { GAK_FUNC|GAK_VAR         , "section"                , GA_SECTION },
    { GAK_FUNC|GAK_VAR         , "used"                   , GA_USED },
    { GAK_FUNC|GAK_VAR         , "weak"                   , GA_WEAK },
    { GAK_FUNC|GAK_VAR         , "dllimport"              , GA_DLLIMPORT },
    { GAK_FUNC|GAK_VAR         , "dllexport"              , GA_DLLEXPORT },
    { GAK_FUNC|GAK_TYPE        , "regparm"                , GA_REGPARM },
    { GAK_VAR|GAK_TYPE         , "mode"                   , GA_MODE },
    { GAK_VAR|GAK_TYPE         , "packed"                 , GA_PACKED },
    { GAK_VAR|GAK_TYPE         , "aligned"                , GA_ALIGNED },
    { GAK_VAR|GAK_TYPE         , "gcc_struct"             , GA_GCC_STRUCT },
    { GAK_VAR|GAK_TYPE         , "ms_struct"              , GA_MS_STRUCT },
    { GAK_VAR|GAK_TYPE         , "spu_vector"             , GA_SPU_VECTOR },
    { GAK_VAR|GAK_TYPE         , "altivec"                , GA_ALTIVEC },
    { GAK_FUNC|GAK_VAR|GAK_TYPE, "unused"                 , GA_UNUSED },
    { GAK_FUNC|GAK_VAR|GAK_TYPE, "deprecated"             , GA_DEPRECATED },
    { GAK_UNDEF, 0, 0 }
};


/**
 * \brief
 * GCC builtin types which are supported in XcodeML
 */
const char *s_gccBuiltinTypes[] = {
    "__builtin_va_list",
    NULL
};


/**
 * \brief
 * GCC builtin functions which are architecture-independent (GCC 4.3.2)
 * 
 * This list does not include defined tokens following.
 *   - __builtin_va_arg : EC_GCC_BLTIN_VA_ARG
 *   - __builtin_offsetof : EC_GCC_BLTIN_OFFSET_OF
 *   - __builtin_types_compatible_p : EC_GCC_BLTIN_TYPES_COMPATIBLE_P
 */
CGccBuiltinFuncInfo s_gccBuiltinFuncs[] = {
    { "__sync_fetch_and_add",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_fetch_and_sub",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_fetch_and_or",                0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_fetch_and_and",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_fetch_and_xor",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_fetch_and_nand",              0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_add_and_fetch",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_sub_and_fetch",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_or_and_fetch",                0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_and_and_fetch",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_xor_and_fetch",               0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_nand_and_fetch",              0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_val_compare_and_swap",        0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__sync_lock_test_and_set",           0, BT_UNDEF,               GBF_ARG_PTRREF },
    { "__builtin_choose_expr",              1, BT_UNDEF,               GBF_ARG },
    { "__builtin_va_arg",                   0, BT_INT,                 GBF_FIXED },
    { "__builtin_va_arg_pack",              0, BT_INT,                 GBF_FIXED },
    { "__builtin_va_arg_pack_len",          0, BT_INT,                 GBF_FIXED },
    { "__builtin_va_start",                 0, BT_VOID,                GBF_FIXED },
    { "__builtin_va_end",                   0, BT_VOID,                GBF_FIXED },
    { "__builtin_va_copy",                  0, BT_VOID,                GBF_FIXED },
    { "__builtin_apply",                    0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_apply_args",               0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_return",                   0, BT_VOID,                GBF_FIXED },
    { "__builtin_return_address",           0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_frame_address",            0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_extract_return_addr",      0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_memset",                   0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_memcpy",                   0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin___memcpy_chk",             0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin___sprintf_chk",            0, BT_INT,                 GBF_FIXED },
    { "__builtin___snprintf_chk",           0, BT_INT,                 GBF_FIXED },
    { "__builtin___vsprintf_chk",           0, BT_INT,                 GBF_FIXED },
    { "__builtin___vsnprintf_chk",          0, BT_INT,                 GBF_FIXED },
    //returns size_t                       
    { "__builtin_object_size",              0, BT_INT,                 GBF_FIXED },
    //returns bool                         
    { "__sync_bool_compare_and_swap",       0, BT_UNSIGNED_INT,        GBF_ARG_PTRREF },
    { "__sync_synchronize",                 0, BT_INT,                 GBF_FIXED },
    { "__sync_lock_release",                0, BT_VOID,                GBF_FIXED },
    { "__builtin_constant_p",               1, BT_INT,                 GBF_FIXED },
    { "__builtin_trap",                     0, BT_VOID,                GBF_FIXED },
    { "__builtin___clear_cache",            0, BT_VOID,                GBF_FIXED },
    { "__builtin_prefetch",                 0, BT_VOID,                GBF_FIXED },
    { "__builtin_expect",                   0, BT_LONG,                GBF_FIXED },
    { "__builtin_huge_valf",                0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_inff",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_nanf",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_nansf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_powif",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_huge_val",                 0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_inf",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_nan",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_nans",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_powi",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_huge_vall",                0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_infl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_nanl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_nansl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_powil",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_exit",                     0, BT_VOID,                GBF_FIXED },
    { "__builtin_abs",                      0, BT_INT,                 GBF_FIXED },
    { "__builtin_labs",                     0, BT_LONG,                GBF_FIXED },
    { "__builtin_llabs",                    0, BT_LONGLONG,            GBF_FIXED },

    { "__builtin_acos",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_asin",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_atan",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_atan2",                    0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_ceil",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_cosh",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_cos",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_exp",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_fabs",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_floor",                    0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_fmod",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_ldexp",                    0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_log",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_pow",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_sqrt",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_sin",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_sinh",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_tan",                      0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_tanh",                     0, BT_DOUBLE,              GBF_FIXED },
    { "__builtin_acosf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_asinf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_atanf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_atan2f",                   0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_ceilf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_coshf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_cosf",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_expf",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_fabsf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_floorf",                   0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_fmodf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_ldexpf",                   0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_logf",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_powf",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_sqrtf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_sinf",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_sinhf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_tanf",                     0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_tanhf",                    0, BT_FLOAT,               GBF_FIXED },
    { "__builtin_acosl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_asinl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_atanl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_atan2l",                   0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_ceill",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_coshl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_cosl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_expl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_fabsl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_floorl",                   0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_fmodl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_ldexpl",                   0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_logl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_powl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_sqrtl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_sinl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_sinhl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_tanl",                     0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_tanhl",                    0, BT_LONGDOUBLE,          GBF_FIXED },
    //returns _Decimal32                   
    { "__builtin_infd32",                   0, BT_FLOAT,               GBF_FIXED },
    //returns _Decimal32                   
    { "__builtin_nand32",                   0, BT_FLOAT,               GBF_FIXED },
    //returns _Decimal64                   
    { "__builtin_infd64",                   0, BT_DOUBLE,              GBF_FIXED },
    //returns _Decimal64                   
    { "__builtin_nand64",                   0, BT_DOUBLE,              GBF_FIXED },
    //returns _Decimal128                  
    { "__builtin_infd128",                  0, BT_LONGDOUBLE,          GBF_FIXED },
    //returns _Decimal128                  
    { "__builtin_nand128",                  0, BT_LONGDOUBLE,          GBF_FIXED },
    { "__builtin_ffs",                      0, BT_INT,                 GBF_FIXED },
    { "__builtin_clz",                      0, BT_INT,                 GBF_FIXED },
    { "__builtin_ctz",                      0, BT_INT,                 GBF_FIXED },
    { "__builtin_popcount",                 0, BT_INT,                 GBF_FIXED },
    { "__builtin_parity",                   0, BT_INT,                 GBF_FIXED },
    { "__builtin_ffsl",                     0, BT_INT,                 GBF_FIXED },
    { "__builtin_clzl",                     0, BT_INT,                 GBF_FIXED },
    { "__builtin_ctzl",                     0, BT_INT,                 GBF_FIXED },
    { "__builtin_popcountl",                0, BT_INT,                 GBF_FIXED },
    { "__builtin_parityl",                  0, BT_INT,                 GBF_FIXED },
    { "__builtin_ffsll",                    0, BT_INT,                 GBF_FIXED },
    { "__builtin_clzll",                    0, BT_INT,                 GBF_FIXED },
    { "__builtin_ctzll",                    0, BT_INT,                 GBF_FIXED },
    { "__builtin_popcountll",               0, BT_INT,                 GBF_FIXED },
    { "__builtin_parityll",                 0, BT_INT,                 GBF_FIXED },
    { "__builtin_bswap32",                  0, BT_INT,                 GBF_FIXED },
    { "__builtin_bswap64",                  0, BT_UNDEF,               GBF_INT64 },
    { "__builtin_alloca",                   0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_calloc",                   0, BT_UNDEF,               GBF_VOIDPTR },
    // returns char pointer
    { "__builtin_index",                    0, BT_UNDEF,               GBF_CHARPTR },
    { "__builtin_rindex",                   0, BT_UNDEF,               GBF_CHARPTR },
    { "__builtin_strstr",                   0, BT_UNDEF,               GBF_CHARPTR },
    { "__builtin_strchr",                   0, BT_UNDEF,               GBF_CHARPTR },
    { "__builtin_strrchr",                  0, BT_UNDEF,               GBF_CHARPTR },
    { "__builtin_strpbrk",                  0, BT_UNDEF,               GBF_CHARPTR },
    { "__builtin_strtok",                   0, BT_UNDEF,               GBF_CHARPTR },
    { "__builtin_memchr",                   0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_memrchr",                  0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_memcmp",                   0, BT_INT,                 GBF_FIXED },
    { "__builtin_memcpy",                   0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_memset",                   0, BT_UNDEF,               GBF_VOIDPTR },
    { "__builtin_strcmp",                   0, BT_INT,                 GBF_FIXED },
    { "__builtin_strlen",                   0, BT_INT,                 GBF_FIXED },
    { "__builtin_strncmp",                  0, BT_INT,                 GBF_FIXED },
    { "__builtin_strspn",                   0, BT_UNSIGNED_INT,        GBF_FIXED },
    { "__builtin_strcspn",                  0, BT_UNSIGNED_INT,        GBF_FIXED },
};



#ifdef CEXPR_DEBUG_GCCATTR

/**
 * \brief
 * add gcc attributes to attributes list
 */
PRIVATE_STATIC void
addGccAttrArgs(GccAttrCheckInfo *gac, CExpr *attrs) {
    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, attrs) {
        CCOL_SL_CONS(&gac->gac_attrArgs, EXPR_L_DATA(ite));
    }
}


/**
 * \brief
 * collect gcc attributes in AST
 */
PRIVATE_STATIC void
collectGccAttrCheckInfo(GccAttrCheckInfo *gac, CExpr *expr)
{
    if(expr == NULL)
        return;

    if(EXPR_CODE(expr) == EC_GCC_ATTRS) {
        addGccAttrArgs(gac, expr);
        return;
    }

    CExprCommon *ec = EXPR_C(expr);
    if(EXPR_ISNULL(ec->e_gccAttrPre) == 0) {
        addGccAttrArgs(gac, ec->e_gccAttrPre);
    }
    if(EXPR_ISNULL(ec->e_gccAttrPost) == 0) {
        addGccAttrArgs(gac, ec->e_gccAttrPost);
    }
    gac->gac_extCount += ec->e_gccExtension;

    switch(EXPR_STRUCT(expr)) {

    case STRUCT_CExprOfUnaryNode:
        collectGccAttrCheckInfo(gac, EXPR_U(expr)->e_node);
        break;

    case STRUCT_CExprOfBinaryNode:
        for(int i = 0; i < 2; ++i) {
            CExpr *node = EXPR_B(expr)->e_nodes[i];
            collectGccAttrCheckInfo(gac, node);
        }
        break;

    case STRUCT_CExprOfList:
        {
            CCOL_DListNode *ite;
            EXPR_FOREACH(ite, expr) {
                CExpr *node = EXPR_L_DATA(ite);
                collectGccAttrCheckInfo(gac, node);
            }
        }
        break;

    default:
        break;
    }
}


/**
 * \brief
 * check gcc attributes not to be lacked from AST
 */
void
startCheckGccAttr(CExpr *expr)
{
    if(s_gccAttrCheckInfos == NULL) {
        s_gccAttrCheckInfos = XALLOC(CCOL_SList);
        memset(s_gccAttrCheckInfos, 0, sizeof(CCOL_SList));
    }
    GccAttrCheckInfo *gac = XALLOC(GccAttrCheckInfo);
    memset(gac, 0, sizeof(GccAttrCheckInfo));
    CCOL_SL_CONS(s_gccAttrCheckInfos, gac);
    collectGccAttrCheckInfo(gac, expr);
}


/**
 * complete checking gcc attributes
 */
void
endCheckGccAttr(CExpr *expr)
{
    assert(s_gccAttrCheckInfos != NULL);
    GccAttrCheckInfo ngac, *gac = (GccAttrCheckInfo*)CCOL_SL_REMOVE_HEAD(s_gccAttrCheckInfos);
    memset(&ngac, 0, sizeof(GccAttrCheckInfo));
    assert(gac != NULL);

    if(CCOL_SL_SIZE(s_gccAttrCheckInfos) == 0) {
        CCOL_SL_CLEAR(s_gccAttrCheckInfos);
        ccol_Free(s_gccAttrCheckInfos);
        s_gccAttrCheckInfos = NULL;
    }

    collectGccAttrCheckInfo(&ngac, expr);

    if(ngac.gac_extCount != gac->gac_extCount) {
        DBGPRINT(("\ngcc extension count=%d -> %d\n", gac->gac_extCount, ngac.gac_extCount));
        ABORT();
    }

    CCOL_SListNode *ite1, *ite2;
    CCOL_SL_FOREACH(ite1, &gac->gac_attrArgs) {
        int exists = 0;
        CExpr *attr1 = (CExpr*)CCOL_SL_DATA(ite1);
        CCOL_SL_FOREACH(ite2, &ngac.gac_attrArgs) {
            CExpr *attr2 = (CExpr*)CCOL_SL_DATA(ite2);
            if(attr1 == attr2) {
                exists = 1;
                break;
            }
        }
        if(exists == 0) {
            DBGPRINT(("\nlost GCC_ATTR_ARGS@" ADDR_PRINT_FMT "\n", (uintptr_t)attr1));
            ABORT();
        }
    }

    CCOL_SL_CLEAR(&ngac.gac_attrArgs);
    CCOL_SL_CLEAR(&gac->gac_attrArgs);
    XFREE(gac);
}

#endif /* CEXPR_DEBUG_GCCATTR */


#ifdef CEXPR_DEBUG

/**
 * \brief
 * check gcc attributes are all output
 */
void
checkGccAttrOutput0(CExpr *e)
{
    if(e == NULL)
        return;
    if(EXPR_CODE(e) == EC_GCC_ATTR_ARG &&
        EXPR_B(e)->e_gccAttrIgnored == 0 &&
        EXPR_B(e)->e_gccAttrOutput == 0) {
        CGccAttrInfo *gai = EXPR_B(e)->e_gccAttrInfo;
        addError(e, CFTL_007, gai ? gai->ga_symbol : "(not fixed)");
        //not abort
    }

    CExprIterator ite;
    EXPR_FOREACH_MULTI(ite, e) {
        checkGccAttrOutput0(ite.node);
    }
}


/**
 * \brief
 * check gcc attributes are all output
 */
void
checkGccAttrOutput()
{
    CCOL_DListNode *ite;

    CCOL_DL_FOREACH(ite, &s_typeDescList) {
        CExprOfTypeDesc *td = EXPR_T(CCOL_DL_DATA(ite));
        if(td->e_isGccAttrDuplicated == 0) {
            checkGccAttrOutput0(EXPR_C(td)->e_gccAttrPre);
            checkGccAttrOutput0(EXPR_C(td)->e_gccAttrPost);
        }
    }
}

#endif /* CEXPR_DEBUG */


/**
 * \brief
 * set gcc attribute to node in prefix position
 *
 * @param expr
 *      target node
 * @param attr
 *      gcc attributes
 * @return
 *      expr
 */
CExpr*
exprSetAttrPre(CExpr *expr, CExpr *attr)
{
    if(attr == NULL)
        return expr;
    if(EXPR_ISNULL(attr)) {
        freeExpr(attr);
        return expr;
    }

    assertYYLineno(expr != NULL);
    assertExpr(expr, EXPR_C(expr)->e_gccAttrPre == NULL);
    assertExprCode(attr, EC_GCC_ATTRS);

    EXPR_C(expr)->e_gccAttrPre = attr;
    EXPR_REF(attr);
    return expr;
}


/**
 * \brief
 * set gcc attribute to node in postfix position
 *
 * @param expr
 *      target node
 * @param attr
 *      gcc attributes
 * @return
 *      expr
 */
CExpr*
exprSetAttrPost(CExpr *expr, CExpr *attr)
{
    if(attr == NULL)
        return expr;
    if(EXPR_ISNULL(attr)) {
        freeExpr(attr);
        return expr;
    }

    assertYYLineno(expr != NULL);
    assertExpr(expr, EXPR_C(expr)->e_gccAttrPost == NULL);
    assertExprCode(attr, EC_GCC_ATTRS);

    EXPR_C(expr)->e_gccAttrPost = attr;
    EXPR_REF(attr);
    return expr;
}


/**
 * \brief
 * set gcc attribute to head node in prefix position
 *
 * @param expr
 *      target node list
 * @param attr
 *      gcc attributes
 * @return
 *      expr
 */
CExpr*
exprSetAttrHeadNode(CExpr *expr, CExpr *attr)
{
    if(attr == NULL)
        return expr;
    if(EXPR_ISNULL(attr)) {
        freeExpr(attr);
        return expr;
    }

    assertYYLineno(expr != NULL);
    assertExpr(expr, EXPR_STRUCT(expr) == STRUCT_CExprOfList);
    assertExpr(expr, CCOL_DL_SIZE(EXPR_DLIST(expr)) > 0);
    assertExprCode(attr, EC_GCC_ATTRS);

    CExpr *headExpr = exprListHeadData(expr);
    exprSetAttrPre(headExpr, attr);

    return expr;
}


/**
 * \brief
 * set gcc attribute to tail node in postfix position
 *
 * @param expr
 *      target node list
 * @param attr
 *      gcc attributes
 * @return
 *      expr
 */
CExpr*
exprSetAttrTailNode(CExpr *expr, CExpr *attr)
{
    if(attr == NULL)
        return expr;
    if(EXPR_ISNULL(attr)) {
        freeExpr(attr);
        return expr;
    }

    assertYYLineno(expr != NULL);
    assertExprStruct(expr, STRUCT_CExprOfList);
    assertExpr(expr, CCOL_DL_SIZE(EXPR_DLIST(expr)) > 0);
    assertExprCode(attr, EC_GCC_ATTRS);

    CExpr *tailExpr = exprListTailData(expr);
    exprSetAttrPost(tailExpr, attr);

    return expr;
}


/**
 * \brief
 * do shallow copy gcc attributes
 *
 * @param dst
 *      destination
 * @param src
 *      source
 */
void
exprCopyAttr(CExpr *dst, CExpr *src)
{
    CExprCommon *cdst = EXPR_C(dst);
    CExprCommon *csrc = EXPR_C(src);

    assertYYLineno(EXPR_ISNULL(cdst->e_gccAttrPre));
    assertYYLineno(EXPR_ISNULL(cdst->e_gccAttrPost));
    assertYYLineno(cdst->e_gccExtension == 0);

    if(cdst->e_gccAttrPre) {
        freeExpr(cdst->e_gccAttrPre);
    }

    if(cdst->e_gccAttrPost) {
        freeExpr(cdst->e_gccAttrPost);
    }

    cdst->e_gccAttrPre   = csrc->e_gccAttrPre;
    cdst->e_gccAttrPost  = csrc->e_gccAttrPost;
    cdst->e_gccExtension = csrc->e_gccExtension;
}


/**
 * \brief
 * join gcc attributes list
 *
 * @param dst
 *      destination
 * @param src
 *      source
 */
void
exprJoinAttr(CExpr *dst, CExpr *src)
{
    assert(dst != NULL);
    assert(src != NULL);

    CExprCommon *cdst = EXPR_C(dst);

    if(EXPR_CODE(src) == EC_GCC_ATTRS) {
        CExpr *attrs;
        if(EXPR_ISNULL(cdst->e_gccAttrPre)) {
            freeExpr(cdst->e_gccAttrPre);
            attrs = exprList(EC_GCC_ATTRS);
            EXPR_REF(attrs);
            cdst->e_gccAttrPre = attrs;
        } else {
            attrs = cdst->e_gccAttrPre;
        }
        exprListJoin(attrs, src);
        return;
    }

    CExprCommon *csrc = EXPR_C(src);

    if(EXPR_ISNULL(csrc->e_gccAttrPre) == 0) {
        CExpr *attrs;
        if(EXPR_ISNULL(cdst->e_gccAttrPre)) {
            freeExpr(cdst->e_gccAttrPre);
            attrs = exprList(EC_GCC_ATTRS);
            EXPR_REF(attrs);
            cdst->e_gccAttrPre = attrs;
        } else {
            attrs = cdst->e_gccAttrPre;
        }
        exprListJoin(attrs, csrc->e_gccAttrPre);
        csrc->e_gccAttrPre = NULL;
    }

    if(EXPR_ISNULL(csrc->e_gccAttrPost) == 0) {
        CExpr *attrs;
        if(EXPR_ISNULL(cdst->e_gccAttrPost)) {
            freeExpr(cdst->e_gccAttrPost);
            attrs = exprList(EC_GCC_ATTRS);
            EXPR_REF(attrs);
            cdst->e_gccAttrPost = attrs;
        } else {
            attrs = cdst->e_gccAttrPost;
        }
        exprListJoin(attrs, csrc->e_gccAttrPost);
        csrc->e_gccAttrPost = NULL;
    }

    cdst->e_gccExtension |= csrc->e_gccExtension;
}


/**
 * \brief
 * join gcc attributes to prefix position
 *
 * @param dst
 *      destination
 * @param src
 *      source
 */
void
exprJoinAttrToPre(CExpr *dst, CExpr *src)
{
    assert(dst);
    assert(src);

    CExprCommon *cdst = EXPR_C(dst);

    if(EXPR_CODE(src) == EC_GCC_ATTRS) {
        if(EXPR_ISNULL(cdst->e_gccAttrPre)) {
            freeExpr(cdst->e_gccAttrPre);
        } else {
            exprListJoin(src, cdst->e_gccAttrPre);
        }
        cdst->e_gccAttrPre = src;
        EXPR_REF(src);
    } else {
        CExprCommon *csrc = EXPR_C(src);

        if(EXPR_ISNULL(csrc->e_gccAttrPre) == 0) {
            if(EXPR_ISNULL(cdst->e_gccAttrPre)) {
                freeExpr(cdst->e_gccAttrPre);
            } else {
                exprListJoin(csrc->e_gccAttrPre, cdst->e_gccAttrPre);
            }
            cdst->e_gccAttrPre = csrc->e_gccAttrPre;
            csrc->e_gccAttrPre = NULL;
        }

        if(EXPR_ISNULL(csrc->e_gccAttrPost) == 0) {
            if(EXPR_ISNULL(cdst->e_gccAttrPre)) {
                freeExpr(cdst->e_gccAttrPre);
            } else {
                exprListJoin(csrc->e_gccAttrPost, cdst->e_gccAttrPre);
            }
            cdst->e_gccAttrPre = csrc->e_gccAttrPost;
            csrc->e_gccAttrPost = NULL;
        }

        cdst->e_gccExtension |= csrc->e_gccExtension;
    }
}


/**
 * \brief
 * add gcc attribute to prefix position
 *
 * @param td
 *      type descriptor
 * @param arg
 *      gcc attribute
 */
void
exprAddAttrToPre(CExprOfTypeDesc *td, CExprOfBinaryNode *arg)
{
    assert(td);
    assert(arg);

    CExpr *args = exprList1(EC_GCC_ATTR_ARGS, (CExpr*)arg);
    CExpr *attrs = EXPR_C(td)->e_gccAttrPre;

    if(EXPR_ISNULL(attrs)) {
        freeExpr(attrs);
        attrs = exprList(EC_GCC_ATTRS);
        EXPR_REF(attrs);
        EXPR_C(td)->e_gccAttrPre = attrs;
    }

    exprListAdd(attrs, args);
}


/**
 * \brief
 * copy gcc attributes and join attributes list 
 *
 * @param dst
 *      destination
 * @param src
 *      srouce
 */
void
exprJoinDuplicatedAttr(CExprOfTypeDesc *dst, CExprOfTypeDesc *src)
{
    if(EXPR_L_ISNULL(EXPR_C(src)->e_gccAttrPost) == 0) {
        CExpr *attrs = (CExpr*)duplicateExprOfList(
            EXPR_L(EXPR_C(src)->e_gccAttrPost));
        exprJoinAttr((CExpr*)dst, attrs);
        src->e_isGccAttrDuplicated = 1;
    }

    if(EXPR_L_ISNULL(EXPR_C(src)->e_gccAttrPre) == 0) {
        CExpr *attrs = (CExpr*)duplicateExprOfList(
            EXPR_L(EXPR_C(src)->e_gccAttrPre));
        exprJoinAttr((CExpr*)dst, attrs);
        src->e_isGccAttrDuplicated = 1;
    }
}


/**
 * \brief
 * judge expr has gcc attribute in prefix/postfix position
 *
 * @param expr
 *      target node
 * @return
 *      0:no, 1:yes
 */
int
exprHasGccAttr(CExpr *expr)
{
    return
        (EXPR_ISNULL(EXPR_C(expr)->e_gccAttrPre) == 0 &&
        EXPR_L_SIZE(EXPR_C(expr)->e_gccAttrPre) > 0) ||
        (EXPR_ISNULL(EXPR_C(expr)->e_gccAttrPost) == 0 &&
        EXPR_L_SIZE(EXPR_C(expr)->e_gccAttrPost) > 0);
}


/**
 * \brief
 * add warning for ignoring attributes
 *
 * @param expr
 *      error node
 * @param attrName
 *      attribute symbol
 */
PRIVATE_STATIC void
addWarnAttrIgnore(CExpr *expr, const char *attrName)
{
    addWarn(expr, CWRN_012, attrName);
}


/**
 * \brief
 * split multiple GCC_ATTR_ARG in GCC_ATTR_ARGS
 *
 * @param expr
 *      target node
 * @return
 *      0:no attributes, 1:ok
 */
PRIVATE_STATIC int
splitAttrArg(CExpr *expr)
{
    CExprCommon *cmn = EXPR_C(expr);
    if(EXPR_ISNULL(cmn->e_gccAttrPost) == 0) {
        if(EXPR_ISNULL(cmn->e_gccAttrPre) == 0)
            exprListJoin(cmn->e_gccAttrPost, cmn->e_gccAttrPre);
        cmn->e_gccAttrPre = cmn->e_gccAttrPost;
        cmn->e_gccAttrPost = NULL;
    }

    CExpr *attrs = cmn->e_gccAttrPre;

    if(EXPR_L_ISNULL(attrs))
        return 0;

    CCOL_DListNode *ite, *ite2, *ite2n;

    EXPR_FOREACH(ite, attrs) {
        CExpr *args = EXPR_L_DATA(ite);
        if(EXPR_L_SIZE(args) <= 1)
            continue;
        int idx = 0;
        EXPR_FOREACH_SAFE(ite2, ite2n, args) {
            if(idx++ == 0)
                continue;
            CExpr *arg = exprListRemove(args, ite2);
            assert(arg);
            CExprOfList *nargs = allocExprOfList1(EC_GCC_ATTR_ARGS, arg);
            CCOL_DL_INSERT_NEXT(EXPR_DLIST(attrs), nargs, ite);
            EXPR_REF(nargs);
        }
    }

    return 1;
}


/**
 * \brief
 * set unknown symbol's type as ST_GCC_BUILTIN
 *
 * @param e
 *      target node
 */
PRIVATE_STATIC void
setUnknownSymbolGccBultin(CExpr *e)
{
    if(EXPR_CODE(e) == EC_IDENT) {
        int ignore = 0;
        resolveType_ident(e, 0, &ignore);
        if(EXPR_SYMBOL(e)->e_symType == ST_UNDEF) {
            EXPR_SYMBOL(e)->e_symType = ST_GCC_BUILTIN;
            exprSetExprsType(e, &s_undefTypeDesc);
        }
    } else {
        CExprIterator ite;
        EXPR_FOREACH_MULTI(ite, e) {
            if(ite.node)
                setUnknownSymbolGccBultin(ite.node);
        }
    }
}


/**
 * \brief
 * set gcc attribute kind to attributes in td
 *
 * @param td
 *      type descriptor
 * @param declr
 *      declarator
 * @param declrContext
 *      declarator context
 * @param isTypeDef
 *      set to 1 when declaration is typedef
 */
void
exprFixAttr(CExprOfTypeDesc *td, CExprOfBinaryNode *declr,
    CDeclaratorContext declrContext, int isTypeDef)
{
    splitAttrArg((CExpr*)td);

    int isTypeName = (declrContext == DC_IN_TYPENAME);
    CExpr *typeExpr = td->e_typeExpr;

    if(typeExpr) {
        if(EXPR_CODE(typeExpr) == EC_TYPE_DESC) {
            exprFixAttr(EXPR_T(typeExpr), declr, declrContext, isTypeDef);
        } else if(ETYP_IS_TAGGED(td)) {
            CExpr *nullExpr = exprListHeadData(typeExpr);
            if(splitAttrArg((CExpr*)nullExpr))
                exprJoinAttrToPre((CExpr*)td, nullExpr);
            if(splitAttrArg((CExpr*)typeExpr))
                exprJoinAttrToPre((CExpr*)td, typeExpr);
        }
    }

    CExpr *attrs = EXPR_C(td)->e_gccAttrPre;

    if(EXPR_L_ISNULL(attrs))
        return;

    CExprOfTypeDesc *declrTd = NULL, *tdo = NULL;
    if(declr) {
        declrTd = EXPR_T(declr->e_nodes[0]);
        tdo = getRefTypeWithoutFix(declrTd);
    } else if(isTypeName) {
        tdo = getRefTypeWithoutFix(td);
    }

    int isFunc = tdo && ETYP_IS_FUNC(tdo) && (isTypeDef == 0);
    int isFuncDef = declr && (EXPR_CODE(EXPR_PARENT(declr)) == EC_FUNC_DEF);
    int isVar = 0, isFuncPtr = 0;
    if(tdo && isFunc == 0 && isTypeDef == 0) {
        if(ETYP_IS_POINTER(tdo)) {
            //check if function pointer
            CExprOfTypeDesc *rtd = EXPR_T(tdo->e_typeExpr);
            CExprOfTypeDesc *rtdo = getRefTypeWithoutFix(rtd);
            isFunc = isFuncPtr = ETYP_IS_FUNC(rtdo);
        }
        isVar = !isFunc;
    }

    int isParams = (declrContext == DC_IN_PARAMS);

    CCOL_DListNode *ite, *iten;

    // set attribute id
    EXPR_FOREACH_SAFE(ite, iten, attrs) {
        CExpr *args = EXPR_L_DATA(ite);
        CExprOfBinaryNode *arg = EXPR_B(exprListHeadData(args));

        if(arg->e_gccAttrIgnored || arg->e_gccAttrKind != GAK_UNDEF)
            continue;

        const char *attrName = EXPR_SYMBOL(arg->e_nodes[0])->e_symName;
        CExpr *exprs = arg->e_nodes[1];
        CGccAttrInfo *gai = getGccAttrInfo(attrName);

        if(gai == NULL) {
            addWarnAttrIgnore(args, attrName);
            arg->e_gccAttrIgnored = 1;
            continue;
        }

        CGccAttrKindEnum gak = gai->ga_kind;

        if( //typedef but not GAK_TYPE
            (isTypeDef && (gak & GAK_TYPE) == 0) ||
            //func declaration but not GAK_FUNC|GAK_TYPE
            (isFunc && (gak & (GAK_FUNC|GAK_TYPE)) == 0) ||
            //var declaration but not GAK_VAR|GAK_TYPE
            (isVar && (gak & (GAK_VAR|GAK_TYPE)) == 0)) {

            //if DC_IN_NO_MEMBER_DECL, estimate later
            if(declrContext != DC_IN_NO_MEMBER_DECL) {
                addWarnAttrIgnore(args, attrName);
                arg->e_gccAttrIgnored = 1;
            }
            continue;
        }

        if(isTypeDef || isTypeName) {
            gak = GAK_TYPE;
        } else {
            if(isParams)
                gak = GAK_TYPE;
            else if(isFunc && isFuncPtr == 0)
                gak = isFuncDef ? GAK_FUNC_DEF : GAK_FUNC;
            else
                gak = GAK_VAR;
        }

        if(isFunc && (gak == GAK_FUNC || gak == GAK_FUNC_DEF)) {
            switch(gai->ga_symbolId) {
            case GA_CONST:
                declrTd->e_isGccConst = 1;
                break;
            case GA_GNU_INLINE:
                gak = GAK_TYPE;
                break;
            default:
                break;
            }
        }

        arg->e_gccAttrKind = gak;
        arg->e_gccAttrInfo = gai;

        if(EXPR_ISNULL(exprs) == 0 && EXPR_ISERROR(exprs) == 0) {
            setUnknownSymbolGccBultin(exprs);
            compile1(exprs, (CExpr*)arg);
        }
    }

    if(EXPR_L_SIZE(attrs) == 0) {
        freeExpr(attrs);
        EXPR_C(td)->e_gccAttrPre = NULL;
    }
}


/**
 * \brief
 * get gcc attribute info by attribute symbol
 *
 * @param symbol
 *      attribute symbol
 * @return
 *      NULL or gcc attribute info
 */
CGccAttrInfo*
getGccAttrInfo(const char *symbol)
{
    char *symbol1 = s_charBuf[0];
    int len = strlen(symbol);

    if(len < 2)
        return NULL;

    if(len > 4) {
        int u1 = (symbol[0] == '_' && symbol[1] == '_' );
        int u2 = (symbol[len - 2] == '_' && symbol[len - 1] == '_');

        if(u1 && u2 == 0) {
            memcpy(symbol1, symbol + 2, len - 2);
            symbol1[len - 2] = 0;
        } else if(u1 && u2) {
            memcpy(symbol1, symbol + 2, len - 4);
            symbol1[len - 4] = 0;
        } else {
            strcpy(symbol1, symbol);
        }
    } else {
        strcpy(symbol1, symbol);
    }

    for(CGccAttrInfo *gai = s_gccAttrInfos; gai->ga_kind != GAK_UNDEF; ++gai) {
        if(strcmp(gai->ga_symbol, symbol1) == 0)
            return gai;
    }

    return NULL;
}


/**
 * \brief
 * sub function of hasGccAttr()
 */
PRIVATE_STATIC int
hasGccAttr0(CExpr *attrs, CGccAttrKindEnum gak)
{
    if(EXPR_ISNULL(attrs))
        return 0;

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, attrs) {
        CExpr *arg = exprListHeadData(EXPR_L_DATA(ite));
        assertExpr(attrs, arg);
        int k = EXPR_B(arg)->e_gccAttrKind;
        if((k & gak) > 0)
            return 1;
    }

    return 0;
}


/**
 * \brief
 * judge td has gcc attribute of specified kind.
 * 
 * @param td
 *      type descriptor
 * @param gak
 *      attribute kind
 * @return
 *      0:no, 1:yes
 */
int
hasGccAttr(CExprOfTypeDesc *td, CGccAttrKindEnum gak)
{
    if(td == NULL || EXPR_CODE(td) != EC_TYPE_DESC ||
        td->e_isGccAttrDuplicated)
        return 0;

    if(hasGccAttr0(EXPR_C(td)->e_gccAttrPre, gak))
        return 1;
    if(hasGccAttr0(EXPR_C(td)->e_gccAttrPost, gak))
        return 1;

    return 0;
}


/**
 * \brief
 * judge td has gcc attribute of specified kind.
 * if td does not have gcc attribute, check original reference type.
 *
 * @param td
 *      type descriptor
 * @param gak
 *      attribute kind
 * @return
 *      0:no, 1:yes
 */
int
hasGccAttrDerived(CExprOfTypeDesc *td, CGccAttrKindEnum gak)
{
    if(hasGccAttr(td, gak))
        return 1;

    if(td->e_refType)
        return hasGccAttr(td->e_refType, gak);

    return 0;
}


/**
 * \brief
 * collect gcc attributes of specified kind
 *
 * @param[out] attrs
 *      collected gcc attributes
 * @param td
 *      type descriptor
 * @param gak
 *      attribute kind
 */
void
getGccAttrRecurse(CExpr *attrs, CExprOfTypeDesc *td, CGccAttrKindEnum gak)
{
    CExpr *tdAttrs = EXPR_C(td)->e_gccAttrPre;

    switch(td->e_tdKind) {
    case TD_FUNC:
    case TD_POINTER:
    case TD_ARRAY:
    case TD_COARRAY:
        if(td->e_typeExpr)
            getGccAttrRecurse(attrs, EXPR_T(td->e_typeExpr), gak);
        break;
    case TD_DERIVED:
        //not derive attributes attached to var/func
        if(td->e_refType) {
            if(gak & GAK_TYPE)
                getGccAttrRecurse(attrs, td->e_refType, GAK_TYPE);
        }
        break;
    default:
        break;
    }

    if(td->e_isGccAttrDuplicated == 0 && EXPR_ISNULL(tdAttrs) == 0) {
        CCOL_DListNode *ite;
        EXPR_FOREACH_REVERSE(ite, tdAttrs) {
            CExpr *args = (CExpr*)EXPR_L_DATA(ite);
            CExprOfBinaryNode *arg = EXPR_B(exprListHeadData(args));
            assertExpr(args, arg);
            if((arg->e_gccAttrKind & gak) > 0) {
                exprListCons(args, attrs);
            }
        }
    }
}


/**
 * \brief
 * judge function is gcc builtin type
 *
 * @param typeName
 *      type name
 * @return
 *      0:no, 1:yes
 */
int
isGccBuiltinType(const char *typeName)
{
    if(s_supportGcc == 0)
        return 0;

    for(const char **p = s_gccBuiltinTypes; *p != NULL; ++p) {
        if(strcmp(*p, typeName) == 0)
            return 1;
    }

    return 0;
}


/**
 * \brief
 * get gcc builtin function's return type
 *
 * @param funcName
 *      function name
 * @param argTd
 *      1st argument type descriptor
 * @param errExpr
 *      node as error node when error occurrs
 * @param[out] isConst
 *      will set to 1 when function is const
 * @return
 *      return type descriptor
 */
CExprOfTypeDesc*
getGccBuiltinFuncReturnType(const char *funcName, CExprOfTypeDesc *argTd,
    CExpr *errExpr, int *isConst)
{
    if(s_supportGcc == 0)
        return 0;

    if(strncmp(funcName, "__builtin_", 10) != 0 &&
        strncmp(funcName, "__sync_", 7) != 0)
        return 0;

    CExprOfTypeDesc *rtd = NULL;
    int count = sizeof(s_gccBuiltinFuncs) / sizeof(CGccBuiltinFuncInfo);

    for(int i = 0; i < count; ++i) {
        CGccBuiltinFuncInfo *gbf = &s_gccBuiltinFuncs[i];
        if(strcmp(gbf->gbf_name, funcName) == 0) {
            *isConst = gbf->gbf_isConst;
            switch(gbf->gbf_type) {
            case GBF_FIXED:
                rtd = &s_numTypeDescs[gbf->gbf_basicType];
                rtd->e_isUsed = 1;
                return rtd;
            case GBF_INT64:
                rtd = &s_numTypeDescs[s_int64Type];
                rtd->e_isUsed = 1;
                return rtd;
            case GBF_VOIDPTR:
                rtd = &s_voidPtrTypeDesc;
                rtd->e_isUsed = 1;
                return rtd;
            case GBF_ARG:
                if(argTd == NULL) {
                    s_voidPtrTypeDesc.e_isUsed = 1;
                    return &s_voidPtrTypeDesc; // for function pointer
                }
                return argTd;
            case GBF_ARG_PTRREF:
                if(argTd == NULL || ETYP_IS_POINTER(argTd) == 0) {
                    s_voidPtrTypeDesc.e_isUsed = 1;
                    return &s_voidPtrTypeDesc; // for function pointer
                }
                return EXPR_T(argTd->e_typeExpr);
            case GBF_CHARPTR:
                rtd = &s_charPtrTypeDesc;
                rtd->e_isUsed = 1;
                return rtd;
            default:
                ABORT();
            }
        }
    }

    return NULL;
}


/**
 * \brief
 * choice and remove gcc attributes which are duplicated
 *
 * @param attrs
 *      gcc attributes
 */
void
exprChoiceAttr(CExpr *attrs)
{
    if(EXPR_L_ISNULL(attrs))
        return;
    CCOL_DListNode *ite, *iten;
    int del = 0;
    CExprOfBinaryNode *preArgs[sizeof(s_gccAttrInfos) / sizeof(CGccAttrInfo)];
    memset(preArgs, 0, sizeof(preArgs));

    EXPR_FOREACH(ite, attrs) {
        int inhibitDup = 0;
        CExpr *args = EXPR_L_DATA(ite);
        CExprOfBinaryNode *arg = EXPR_B(exprListHeadData(args));
        CGccAttrInfo *gai = arg->e_gccAttrInfo;
        if(arg->e_gccAttrIgnored || gai == NULL)
            continue;
        switch(gai->ga_symbolId) {
        case GA_SECTION:
        case GA_ALIAS:
            inhibitDup = 1;
            break;
        default:
            break;
        }

        if(inhibitDup || EXPR_ISNULL(arg->e_nodes[1])) {
            //use one of __section__ last defined 
            CExprOfBinaryNode *preArg = preArgs[gai->ga_symbolId];
            if(preArg) {
                EXPR_ISDELETING(preArg) = del = 1;
                preArg->e_gccAttrIgnored = 1;
            }
            preArgs[gai->ga_symbolId] = arg;
        }
    }

    if(del == 0)
        return;

    EXPR_FOREACH_SAFE(ite, iten, attrs) {
        CExpr *args = EXPR_L_DATA(ite);
        CExprOfBinaryNode *arg = EXPR_B(exprListHeadData(args));
        
        if(EXPR_ISDELETING(arg)) {
            EXPR_REF(args);
            exprListRemove(attrs, ite);
            freeExpr(args);
        }
    }
}


/**
 * \brief
 * returns if e has attrId attribute.
 *
 * @param e
 *      target expr
 * @param id
 *      gcc attribute symbol id
 */
int
hasGccAttrId(CExpr *e, CGccAttrIdEnum id)
{
    CExpr *args = EXPR_C(e)->e_gccAttrPre;
    if(args == NULL)
        return 0;

    CCOL_DListNode *ite;
    EXPR_FOREACH(ite, args) {
        CExprOfBinaryNode *arg = EXPR_B(exprListHeadData(EXPR_L_DATA(ite)));
        if(arg->e_gccAttrInfo && arg->e_gccAttrInfo->ga_symbolId == id)
            return 1;
    }

    return 1;
}

