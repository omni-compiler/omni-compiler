/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file omdriver.h
 */

#ifndef __OMDRIVER_H__
#define __OMDRIVER_H__


/**
 * @define
 */

/** for build */
#define OM_OBJFILE_EXT       "." OBJEXT        /** object file extension */
#define OM_TMPDIR_NAME       "/__omni_tmp__"   /** path to temporary file */

/** for driver process */
#define MINIMUM_ARGUMENT        2     /** maximum number of argument */
#define CODE_NEW_LINE           '\n'  /** new line */
#define CODE_SHARP              '#'   /** sharp */
#define CODE_SPACE              " "   /** space */
#define CODE_HYPHEN             '-'   /** hyphen */
#define CODE_PERIOD             '.'   /** period */
#define CODE_END_OF_STR         '\0'  /** end of string */
#define CODE_CURRENT_DIR        "./"  /** current directory */
#define CODE_PATH_DELIM         "/"   /** delimiter */
#define CODE_SUBSTITUTE         '='   /** substitution code */
#define MAX_OPTION_CNT          1024  /** the number of option user can set */
#define MAX_CONF_LEN            8192  /** length of configuration value */
#define MAX_INPUT_FILE          65536 /** the number of file can be compiled */
#define OPT_BUF                 65536 /** buffer for unrecognized options */
#define MAX_INPUT_FILE_PATH     65536 /** length of input file path */
#define STR_BUF                 8192
#define MAX_TRANSLATORS         128
#define CNT_WITH_W              6     /** option code contains "W" */
#define CNT_CONFIG              13    /** configuration value */
#define MAX_LEN_EXTENSION       6     /** max length of extension */

/** default parent temporary directory */
#define DEFAULT_TEMP_DIR     "/tmp"


/** language ID */
#define LANGID_C        0     /** C Language */
#define LANGID_F        1     /** FORTRUN */
#define LANGCODE_C      "c"   /** C Language */
#define LANGCODE_F      "f"   /** FORTRUN */

/** configuration file */
#define CONF_FILE_C          "omc.conf"
#define CONF_FILE_F          "omf.conf"

/** remove string */
#define SYSTEM_REMOVE        "rm -rf "
#define SYSTEM_MKDIR         "mkdir "
/** defualt output file */
#define DEFAULT_OUT_FILE     "a.out"

/** module name */
#define MODULE_PROCESSOR     "Preprocessor"
#define MODULE_L2X           "L2X"
#define MODULE_LX2X          "LX2X"
#define MODULE_X2L           "X2L"
#define MODULE_NATIVE        "NativeCompiler"
#define MODULE_LINKER        "Linker"

/** extension */
#define EXTENSION_C          ".c"
#define EXTENSION_O          OM_OBJFILE_EXT
#define EXTENSION_A          ".a"
#define EXTENSION_I          ".i"
#define EXTENSION_F_LOWER    ".f"
#define EXTENSION_F_UPPER    ".F"
#define EXTENSION_F90_LOWER  ".f90"
#define EXTENSION_F90_UPPER  ".F90"
#define EXTENSION_XML        ".xml"

/** exit code */
#define EXIT_SUCCESS         0
#define EXIT_ERROR_GENERAL   1
#define EXIT_ERROR_PROCESSOR 2
#define EXIT_ERROR_L2X       3
#define EXIT_ERROR_LX2X      4
#define EXIT_ERROR_X2L       5
#define EXIT_ERROR_NATIVE    6
#define EXIT_ERROR_LINKER    7

/** configuration ID */
#define CONFIG_PATH_PP       "Preprocessor"
#define CONFIG_PATH_L2X      "L2X"
#define CONFIG_PATH_LX2X     "LX2X"
#define CONFIG_PATH_X2L      "X2L"
#define CONFIG_PATH_NTV      "NativeCompiler"
#define CONFIG_PATH_LNK      "Linker"
#define CONFIG_OPT_PP        "PreprocessorOption"
#define CONFIG_OPT_L2X       "L2XOption"
#define CONFIG_OPT_LX2X      "LX2XOption"
#define CONFIG_OPT_X2L       "X2LOption"
#define CONFIG_OPT_NTV       "NativeCompilerOption"
#define CONFIG_OPT_LNK       "LinkerOption"
#define CONFIG_PATH_TEMP     "TempDir"
#define CONFIG_VALUE_DEFAULT "INVALID"
#define CONFIG_KIND_PATH     0
#define CONFIG_KIND_OPT      1



/** for lower module */
/** enviromental value */
#define ENV_OM_OPTIONS          "OM_OPTIONS"
#define ENV_OM_USE_XMP          "OM_USE_XMP"
#define ENV_OM_USE_OPENMP       "OM_USE_OPENMP"
#define ENV_OM_USE_ACC          "OM_USE_ACC"
#define ENV_OM_VERBOSE          "OM_VERBOSE"
#define ENV_OM_LANGID           "OM_LANGID"
#define ENV_OM_USE              "OM_USE_%s"
#define ENV_OM_TRANSLATORS      "OM_TRANSLATORS"
#define ENV_OMNI_HOME           "OMNI_HOME"
#define ENV_OM_DRIVER_CONF_DIR  "OM_DRIVER_CONF_DIR"

/** option identifier */
/** for Preprocessor */
#define OPT_PP_INCPATH          "-I"
#define OPT_PP_D_MACRO          "-D"
#define OPT_PP_U_MACRO          "-U"
#define OPT_PP_P                "--Wp"
/** for L2X */
#define OPT_L2X_M32             "--m32"
#define OPT_L2X_M64             "--m64"
#define OPT_L2X_SAVE            "--save"
#define OPT_L2X_SAVEEQ          "--save="
#define OPT_L2X_F               "--Wf"
#define OPT_L2X_FOPENMP   	"-fopenmp"
/** for LX2X */
#define OPT_LX2X_X              "--Wx"
/** for X2L */
#define OPT_X2L_B               "--Wb"
/** for X2L */
#define OPT_LX2X_TRANS          "-T"
#define OPT_NTV_N               "--Wn"
#define OPT_LNK_L               "--Wl"
/** for Driver */
#define OPT_INVALID_CODE        "-"
#define OPT_DRV_DONT_LINK       "-c"
#define OPT_LNK_OUTPUT          "-o"
#define OPT_DRV_DO_CPP          "-cpp"
#define OPT_DRV_NO_CPP          "-nocpp"
#define OPT_DRV_LANGID          "--x"
#define OPT_DRV_CONF            "--conf"
#define OPT_DRV_TEMP            "--tempdir"
#define OPT_DRV_STAY            "--stay-tempdir"
#define OPT_DRV_FIXED           "--fixed-tempdir"
#define OPT_DRV_DEBUG           "--debug"
#define OPT_DRV_VERBOSE         "--verbose"
#define OPT_DRV_VERSION         "--version"
#define OPT_DRV_HELP            "--help"


/** for output message */
/** normal message */
#define NORMAL_MSG_SUCCESS   "succeeded."
#define NORMAL_MSG_COMPLETED "completed."
#define NORMAL_MSG_COMMAND   "omdriver"
/** error message */
#define ERROR_MSG_GENERAL    "error:"
#define ERROR_MSG_FATAL      "fatal Error."
#define ERROR_MSG_ARGUMENT   "type over 2 arguments."
#define ERROR_MSG_PROCESSOR  "Preprocessor Error."
#define ERROR_MSG_L2X        "L2X Error."
#define ERROR_MSG_LX2X       "LX2X Error."
#define ERROR_MSG_X2L        "X2L Error."
#define ERROR_MSG_NATIVE     "Native Compiler Error."
#define ERROR_MSG_LINKER     "Linker Error."
#define ERROR_MSG_CANT_OPEN  "can not open conffile."
#define ERROR_MSG_UNREC_FILE "file format not recognized."
#define ERROR_MSG_UNREC_LANG "unrecognized language id."
#define ERROR_MSG_FAIL_SYS   "executing module."
#define ERROR_MSG_TEMP_DIR   "failed to create temporary directory."
#define ERROR_MSG_NO_INPFILE "no input files."
#define ERROR_MSG_INVALID_LANGID "invalid language id specified."



/** for general code */
/** boolean value */
#define TRUE                 1
#define FALSE                0

/** result value */
#define SUCCESS              0
#define FAILED               -1




/**
 * \brief
 * lower module IDs
 */
typedef enum e_option_applier {
    MOD_PP = 0,
    MOD_L2X,
    MOD_LX2X,
    MOD_X2L,
    MOD_NTV,
    MOD_LNK,
    MOD_DRV
} opt_applier;



/**
 * \brief
 * directory path where module script is put
 */
typedef struct st_module_paths {
    char *l2x;
    char *lx2x;
    char *x2l;
    char *native;
    char *linker;
} module_paths;

/**
 * \brief
 * option which module scripts use
 */
typedef struct st_module_options {
    char *l2x;
    char *lx2x;
    char *x2l;
    char *native;
    char *linker;
} module_opts;

/**
 * \brief
 * container of module options and paths
 */
typedef struct st_module_configuration {
    module_paths paths;
    module_opts opts;
} module_conf;


/**
 * \brief
 * for driver, pair of option
 */
typedef struct st_option_pair {
    char *opt_value;
    opt_applier opt_applier;
    unsigned char is_multiple;
    unsigned char req_arg;
    unsigned char need_exclusion;
    unsigned char allow_concat_arg;
    char *exclusion_opt;
} opt_pair;

/**
 * \brief
 * set of option
 */
typedef struct st_option_set {
    char *opt_value[MAX_OPTION_CNT];
    char *opt_argument[MAX_OPTION_CNT];
    unsigned char is_applied;
    unsigned char apply_cnt;
} opt_set;

/**
 * \brief
 * each in/out file
 */
typedef struct st_inout_file {
    char *in_file;
    unsigned char is_preped;
    unsigned char is_valid;
    unsigned char is_compiled;
/** reserved */
} io_file;

/**
 * \brief
 * input file names
 */
typedef struct st_input_files {
    io_file info[MAX_INPUT_FILE];
    int     next_index;
} om_file;

/**
 * \brief
 * config items in configuration file
 */
typedef struct st_config {
    char *config_id;
    opt_applier module_id;
    int config_kind;
    char config_value[MAX_CONF_LEN];
    unsigned char is_applied;
} om_config;

/**
 * \brief
 * management information for driver
 */
typedef struct st_driver_manage_info {
    int     opt_cnt;
    char    **opt_values;
    om_file file;
    opt_set *options;

    struct {
        int     proc_lang;
        int     dont_link;
        int     do_cpp;
        char    *conffile;
        char    *tempdir;
        int     dont_del_temp;
        int     fixed_temp;
        int     verbose;
    } private;
} driver_manage_info;



/**
 * @macro function
 */
/** accessor to driver private option */
#define GET_DONT_LINK() g_manage_info.private.dont_link
#define SET_DONT_LINK( flg ) g_manage_info.private.dont_link = flg
#define GET_DO_CPP() g_manage_info.private.do_cpp
#define SET_DO_CPP( flg ) g_manage_info.private.do_cpp = flg
#define GET_CONFFILE() g_manage_info.private.conffile
#define SET_CONFFILE( file ) g_manage_info.private.conffile = file
#define GET_TEMPDIR() g_manage_info.private.tempdir
#define SET_TEMPDIR( dir ) g_manage_info.private.tempdir = dir
#define GET_DONT_DEL_TEMP() g_manage_info.private.dont_del_temp
#define SET_DONT_DEL_TEMP( flg ) g_manage_info.private.dont_del_temp = flg
#define GET_FIXED_TEMP() g_manage_info.private.fixed_temp
#define SET_FIXED_TEMP( flg ) g_manage_info.private.fixed_temp = flg
#define GET_VERBOSE() g_manage_info.private.verbose
#define SET_VERBOSE( flg ) g_manage_info.private.verbose = flg

/** accessor to option table */
#define GET_OPT_VALUE_TBL( idx_opt, idx_value ) \
    g_manage_info.options[idx_opt].opt_value[idx_value]
#define GET_OPT_ARG_TBL( idx_opt, idx_arg ) \
    g_manage_info.options[idx_opt].opt_argument[idx_arg]
#define GET_OPT_IS_APPLIED_TBL( idx ) g_manage_info.options[idx].is_applied
#define GET_OPT_APPLY_CNT_TBL( idx ) g_manage_info.options[idx].apply_cnt

#endif /** __OMDRIVER_H__ */
