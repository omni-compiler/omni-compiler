/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file omdriver.c
 */

#define _BSD_SOURCE 1
#include "exc_platform.h"
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <libgen.h>

#include "omdriver.h"

static driver_manage_info g_manage_info;
static char g_unrec_opt_str[OPT_BUF] = { 0 };
static int  g_num_in_file = 0;
static int  g_lang_id = LANGID_C;
static int  g_is_temp_dir_created = FALSE;
static char g_cmd_buf[MAX_INPUT_FILE_PATH];
static char g_module_path[MAX_INPUT_FILE_PATH];
static char g_option_buf[OPT_BUF];
static char g_outfile_buf[MAX_INPUT_FILE_PATH];
static char g_infile_buf[MAX_INPUT_FILE_PATH];
static char g_all_obj[MAX_INPUT_FILE];


void exe_system(char *shcmd);

/**
 * option table
 * */
static const opt_pair opt_pair_table[] = {
    /* option id          module id multi,      other option
     *                              req_arg,       
     *                              exclusion,       
     *                              concat,    
     * |                  |         |           |
     */
    { OPT_PP_INCPATH,     MOD_PP,   1, 1, 0, 1, OPT_INVALID_CODE },
    { OPT_PP_D_MACRO,     MOD_PP,   1, 1, 0, 1, OPT_INVALID_CODE },
    { OPT_PP_U_MACRO,     MOD_PP,   1, 1, 0, 1, OPT_INVALID_CODE },
    { OPT_PP_P,           MOD_PP,   1, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_L2X_M32,        MOD_L2X,  0, 0, 1, 0, OPT_L2X_M64 },
    { OPT_L2X_M64,        MOD_L2X,  0, 0, 1, 0, OPT_L2X_M32 },
    { OPT_L2X_SAVE,       MOD_L2X,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_L2X_SAVEEQ,     MOD_L2X,  0, 1, 0, 1, OPT_INVALID_CODE },
    { OPT_L2X_F,          MOD_L2X,  1, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_LX2X_X,         MOD_LX2X, 1, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_X2L_B,          MOD_X2L,  1, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_NTV_N,          MOD_NTV,  1, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_LNK_OUTPUT,     MOD_LNK,  0, 1, 0, 0, OPT_INVALID_CODE },
    { OPT_LNK_L,          MOD_LNK,  1, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_LX2X_TRANS,     MOD_DRV , 1, 1, 0, 1, OPT_INVALID_CODE },
    { OPT_DRV_LANGID,     MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_DONT_LINK,  MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_CONF,       MOD_DRV,  0, 1, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_TEMP,       MOD_DRV,  0, 1, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_STAY,       MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_FIXED,      MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_DEBUG,      MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_VERBOSE,    MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_VERSION,    MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
    { OPT_DRV_HELP,       MOD_DRV,  0, 0, 0, 0, OPT_INVALID_CODE },
};

#define NUM_OPT_KINDS   (sizeof(opt_pair_table) / sizeof(opt_pair))


/**
 * config table
 * */
static om_config config_table[CNT_CONFIG] = {
    { CONFIG_PATH_PP,     MOD_PP,   CONFIG_KIND_PATH, CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_PATH_L2X,    MOD_L2X,  CONFIG_KIND_PATH, CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_PATH_LX2X,   MOD_LX2X, CONFIG_KIND_PATH, CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_PATH_X2L,    MOD_X2L,  CONFIG_KIND_PATH, CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_PATH_NTV,    MOD_NTV,  CONFIG_KIND_PATH, CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_PATH_LNK,    MOD_LNK,  CONFIG_KIND_PATH, CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_OPT_PP,      MOD_PP,   CONFIG_KIND_OPT,  CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_OPT_L2X,     MOD_L2X,  CONFIG_KIND_OPT,  CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_OPT_LX2X,    MOD_LX2X, CONFIG_KIND_OPT,  CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_OPT_X2L,     MOD_X2L,  CONFIG_KIND_OPT,  CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_OPT_NTV,     MOD_NTV,  CONFIG_KIND_OPT,  CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_OPT_LNK,     MOD_LNK,  CONFIG_KIND_OPT,  CONFIG_VALUE_DEFAULT, 0 },
    { CONFIG_PATH_TEMP,   MOD_DRV,  CONFIG_KIND_PATH, CONFIG_VALUE_DEFAULT, 0 }
};

static int opt_idx(const char *opt)
{
    int i;
    for(i = 0; i < NUM_OPT_KINDS; ++i) {
        if(strcmp(opt_pair_table[i].opt_value, opt) == 0)
            return i;
    }

    return -1;
}


/**
 * compare option value
 */
static int
cmp_opt_value( char *opt_value, char *opt_src )
{
    int idx_opt_value = 0;
    int opt_src_len = strlen(opt_src);
    int opt_value_len = strlen(opt_value);

    while ((idx_opt_value < opt_value_len) &&
            (idx_opt_value < opt_src_len)) {
        if (*(opt_value+idx_opt_value) != *(opt_src+idx_opt_value)) {
            return FALSE;
        }
        idx_opt_value++;
    }

    return TRUE;
}


/**
 * concat i/o file name with space
 *
 * */
void concat_io_file( char *script, char *in, char *out )
{
    strcat( script, CODE_SPACE );
    strcat( script, in );
    strcat( script, CODE_SPACE );
    strcat( script, out );
}


/**
 * change file's extension
 *
 * */
void change_extension( char *src, char *extension )
{
    int i;
    int len = strlen( src );

    for ( i=len-1; i>0; i--) {
        if (*(src + i) == CODE_PERIOD) {
            strcpy( (src + i), extension );
            break;
        }
    }
}


/**
 * check configuration is applied
 *
 * */
int config_is_applied( char *key )
{
    int ret = FALSE;
    int i;
    om_config *configs = config_table;

    for (i = 0; i<CNT_CONFIG; i++) {
        if (strcmp( key, (configs + i)->config_id ) == 0) {
            ret = (configs + i)->is_applied;
            break;
        }
    }

    return ret;
}


/**
 * get configuration from table
 * 
 */
int get_config( char *value, char *key )
{
    int ret = SUCCESS;
    int i;
    om_config *config = config_table;

    /** search configuration table with key */
    for (i = 0; i<CNT_CONFIG; i++) {
        if (strcmp( (config+i)->config_id, key ) == 0) {
            strcpy( value, (config+i)->config_value );
            ret = SUCCESS;
            return ret;
        } else {
            ret = FAILED;
        }
    }

    return ret;
}


/**
 * get temporary directory
 *
 * */
int get_tempdir( char *dir )
{
    int ret = SUCCESS;
    char *tempdir;

    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_TEMP)) && GET_TEMPDIR()) {
        tempdir = GET_TEMPDIR();
        strcpy( dir, tempdir );
    } else if (config_is_applied( CONFIG_PATH_TEMP )) {
        ret = get_config( dir, CONFIG_PATH_TEMP );
    } else {
        strcpy( dir, DEFAULT_TEMP_DIR );
    }

    strcat(dir, OM_TMPDIR_NAME);

    if (GET_FIXED_TEMP() == FALSE) {
        char buf[MAX_INPUT_FILE_PATH];
        sprintf( buf, "%s.%d", dir, getpid());
        strcpy( dir, buf );
    }

    return ret;
}


/**
 * @brief delete temp files and exit
 * 
 * @param[in] exitCode exit code
 */
static void
clean_exit(int exitCode)
{
    if (g_is_temp_dir_created && GET_DONT_DEL_TEMP() == FALSE) {
        char tempdir[MAX_INPUT_FILE_PATH];
        char shcmd[MAX_INPUT_FILE_PATH];
        get_tempdir( tempdir );
        strcpy( shcmd, SYSTEM_REMOVE );
        strcat( shcmd, tempdir );
        exe_system( shcmd );
    }

    exit( exitCode );
}


/**
 * output message
 *
 * @brief output message to stdout
 *
 * @param[in] msg :message want to output to stdout.
 *
 * */
void msg_normal( const char *msg, ... )
{
    /** check if message is enabled */
    if (GET_VERBOSE() == FALSE) {
        return;
    }

    /** check if null pointer exists */
    if (msg == NULL) {
        fprintf( stderr, "%s%c", ERROR_MSG_FATAL, CODE_NEW_LINE );
    }

    /** output message */
    va_list args;
    va_start( args, msg );
    vfprintf( stdout, msg, args );
    va_end( args );

    fprintf( stdout, "%c", CODE_NEW_LINE );
}


/**
 * output error message
 *
 * @brief output error message to stderr
 *
 * @param[in] msg :message want to output to stderr.
 *
 * */
void msg_error( const char *msg, const char *arg )
{

    /** check if null pointer exists */
    if (msg == NULL) {
        fprintf( stderr, "%s%c", ERROR_MSG_FATAL, CODE_NEW_LINE );
        return;
    }

    /** output error message */
    fprintf( stderr, "%s%s", ERROR_MSG_GENERAL, CODE_SPACE );
    if (arg != NULL) {
        fprintf( stderr, "%s%s(:%s)%c", msg, CODE_SPACE, arg, CODE_NEW_LINE );
    } else {
        fprintf( stderr, "%s%c", msg, CODE_NEW_LINE );
    }
}


/**
 * execute sysmte function
 *
 * */
void exe_system( char *shcmd )
{
    int ret = 0;

    ret = system( shcmd );
    if (ret == -1) {
        msg_error( ERROR_MSG_FAIL_SYS, shcmd );
        clean_exit( EXIT_ERROR_GENERAL );
    } else {
        ret = WEXITSTATUS( ret );
        if (ret != SUCCESS) {
            if (GET_VERBOSE())
                msg_error( ERROR_MSG_FAIL_SYS, shcmd );
            clean_exit( EXIT_ERROR_GENERAL );
        }
    }
}


/**
 * output version information
 *
 * */
void disp_version( void )
{
    printf( "%s %s\n", PACKAGE_NAME, PACKAGE_VERSION );
}


/**
 * output help information
 *
 * */
void disp_help( void )
{
    const char *usages[] = {
        "",
        "Compile Driver Options",
        "",
        "    -c                  not run the Linker.",
        "    --conf [conffile]   read conffile as configuration file.",
        "    --tempdir [tempdir] set parent of temporary directory.",
        "    --stay-tempdir      do not delete temporary directory.",
        "    --fixed-tempdir     use fixed name for temporary directory.",
        "    --debug             same as '--tempdir . --stay-tempdir",
        "                        --fixed-tempdir'.",
        "    --verbose           print processing status.",
        "    --version           print version.",
        "    -h, --help          print usage.",
        "",
        "Preprocessor Options",
        "",
        "    -I[incpath]         add include path.",
        "    -D[macro]<=value>   define macro.",
        "    -U[macro]           undefine macro.",
        "    --Wp[option]        pass [option] as an option to the Preprocessor.",
        "",
        "Frontend Options",
        "",
        "    --Wf[option]        pass [option] as an option to the Frontend.",
        "",
        "c2c Frontend Options",
        "",
        "    --m32               set long and pointer to 32 bits.",
        "    --m64               set long and pointer to 64 bits.",
        "",
        "f2f Frontend Options",
        "",
        "    --save[=n]          add save attribute to local variable larger",
        "                        than n kbytes except in a recursive function",
        "                        and common variables. (default n=1)",
        "",
        "Xcode Processor Options",
        "",
        "    -T[translator]      execute specified XcodeML to XcodeML translator.",
        "    --Wx[option]        pass [option] as an option to the Xcode Processor.",
        "",
        "Backend Options",
        "",
        "    --Wb[option]        pass [option] as an option to the Backend.",
        "",
        "Native Compiler Options",
        "",
        "    -o [outputfile]     specify output file path.",
        "    --Wn[option]        pass [option] as an option to the Native Compiler.",
        "",
        "Linker Options",
        "",
        "    -o [outputfile]     specify output file path.",
        "    --Wl[option]        pass [option] as an option to the Linker.",
        "",
        "XcalableMP Options",
        "",
        "    -tmp                output parallel code (__omni_tmp_SRCNAME.c)",
        "",
        "    -profile            profile specified directives",
        "    -allprofile         profile all directives",
        "    -with-scalasca      output results in a scalasca format",
        "    -with-tlog          output results in a tlog format",
        "",
    };

    printf( "usage: %s <OPTIONS> <INPUTFILE> ...\n", NORMAL_MSG_COMMAND );

    for(int i = 0; i < sizeof(usages) / sizeof(usages[0]); ++i) {
        fputs( usages[i], stdout );
        fputs( "\n", stdout );
    }
}


/**
 * set driver private option
 *
 * @brief this function set all compile driver option.
 *
 * */
int set_driver_option( void )
{
    int ret = SUCCESS;
    const char *lang_code;

    /** langid */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_LANGID))) {
        lang_code = GET_OPT_VALUE_TBL(opt_idx(OPT_DRV_LANGID), 0) +
            strlen(OPT_DRV_LANGID);
        if (strcmp(lang_code, LANGCODE_C) == 0) {
            g_lang_id = LANGID_C;
        } else if(strcmp(lang_code, LANGCODE_F) == 0) {
            g_lang_id = LANGID_F;
        } else {
            msg_error( ERROR_MSG_INVALID_LANGID, NULL );
            clean_exit( EXIT_ERROR_GENERAL );
        }
    }

    /** dont link */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_DONT_LINK))) {
        SET_DONT_LINK( TRUE );
    }

    /** set conffile */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_CONF))) {
        SET_CONFFILE(GET_OPT_ARG_TBL(opt_idx(OPT_DRV_CONF), 0));
    }

    /** set tempdir */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_TEMP))) {
        SET_TEMPDIR(GET_OPT_ARG_TBL(opt_idx(OPT_DRV_TEMP), 0));
    }

    /** stay tempfile */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_STAY))) {
        SET_DONT_DEL_TEMP( TRUE );
    }

    /** fixed tempdir */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_FIXED))) {
        SET_FIXED_TEMP( TRUE );
    }

    /** debug */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_DEBUG))) {
        SET_DONT_DEL_TEMP( TRUE );
        SET_FIXED_TEMP( TRUE );

        if(GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_TEMP)) == FALSE) {
            GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_TEMP)) = TRUE;
            SET_TEMPDIR(".");
        }
    }

    /** verbose */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_VERBOSE))) {
        SET_VERBOSE( TRUE );
    }

    /** version */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_VERSION))) {
        disp_version();
        clean_exit( EXIT_SUCCESS );
    }

    /** help */
    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_HELP))) {
        disp_help();
        clean_exit( EXIT_SUCCESS );
    }

    return ret;
}


/**
 * set option value
 *
 * */
int set_option( char **opt_value) {
    int found = 0;
    int idx_opt_table = 0;
    int apply_cnt = 0;
    const opt_pair *pair;
    opt_set  *set;
    char     *value = *opt_value;

    while (idx_opt_table < NUM_OPT_KINDS) {
        pair = &opt_pair_table[idx_opt_table];
        set = &g_manage_info.options[idx_opt_table];

        /** compare table value and argument */
        if (cmp_opt_value( value, pair->opt_value)) {
            found = 1;
            break;
        }
        idx_opt_table++;
    }

    if (found == 0)
        return 0;

    set->opt_value[set->apply_cnt] = value;
    apply_cnt++;
    set->is_applied = TRUE;

    if (pair->req_arg &&
        (pair->allow_concat_arg == FALSE ||
        strlen(value) == strlen(pair->opt_value))) {
        set->opt_argument[set->apply_cnt] = *(opt_value + 1);
        apply_cnt++;
    }

    if (pair->is_multiple)
        set->apply_cnt++;

    return apply_cnt;
}


/**
 * add input file name
 *
 * */
int add_infile( char *src )
{
    int ret = SUCCESS;
    om_file *ofile = &(g_manage_info.file);

    ofile->info[ofile->next_index].in_file = src;
    ofile->next_index++;

    ++g_num_in_file;

    return ret;
}


/**
 * set commandline options
 *
 * @param[in] opt_cnt :the number of commandline option
 * @param[in] options :values of options
 */
int set_options( int opt_cnt, char** options )
{
    int loop = 0;
    int ret = SUCCESS;

    while (loop < opt_cnt) {
        ret = set_option( (options + loop));
        if (ret >= 1) {
            loop += ret;
        } else {

            /** exit with error code if the option is unexpected */
            if (**(options + loop) == CODE_HYPHEN) {
                strcat(g_unrec_opt_str, *(options + loop));
                strcat(g_unrec_opt_str, " ");
            } else {
                /** add value to input files */
                add_infile( (char*)*(options + loop));
            }

            loop++;
        }
    }

    return ret;
}


/**
 * get extension from source file
 *
 * */
int get_extension( char *ext, char *src )
{
    int i;
    int len = strlen( src );

    for ( i=len-1; i>0; i--) {
        if (*(src + i) == CODE_PERIOD) {
            strcpy( ext, (src + i));
            return TRUE;
        }
    }

    ext[0] = 0;

    return FALSE;
}



/**
 * validate all source files
 *
 * */
int validate_src( void )
{
    int ret = SUCCESS;
    int i;
    io_file *io = &(g_manage_info.file.info[0]);
    om_file *om = &(g_manage_info.file);
    char extension[MAX_LEN_EXTENSION];

    for (i = 0; i<om->next_index; i++) {
        if (get_extension( extension, io->in_file ) == FALSE) {
            io->is_valid = FALSE;
        } else {
            /** check extension */
            switch (g_lang_id) {

            case LANGID_C:
                if (strcmp( extension, EXTENSION_C ) == 0) {
                    io->is_valid = TRUE;
                    io->is_preped = FALSE;
                } else if (strcmp( extension, EXTENSION_I ) == 0) {
                    io->is_valid = TRUE;
                    io->is_preped = TRUE;
                } else if (strcmp( extension, EXTENSION_O ) == 0) {
                    io->is_valid = TRUE;
                    io->is_compiled = TRUE;
                } else if (strcmp( extension, EXTENSION_A ) == 0) {
                    io->is_valid = TRUE;
                    io->is_compiled = TRUE;
                } else {
                    io->is_valid = FALSE;
                }
                break;

            case LANGID_F:
                if (( strcmp( extension, EXTENSION_F_UPPER ) == 0 ) ||
                     ( strcmp( extension, EXTENSION_F90_UPPER ) == 0 )) {
                    io->is_valid = TRUE;
                    io->is_preped = FALSE;
                } else if (( strcmp( extension, EXTENSION_F_LOWER ) == 0 ) ||
                            ( strcmp( extension, EXTENSION_F90_LOWER ) == 0 )) {
                    io->is_valid = TRUE;
                    io->is_preped = TRUE;
                } else if (strcmp( extension, EXTENSION_O ) == 0) {
                    io->is_valid = TRUE;
                    io->is_compiled = TRUE;
                } else if (strcmp( extension, EXTENSION_A ) == 0) {
                    io->is_valid = TRUE;
                    io->is_compiled = TRUE;
                } else {
                    io->is_valid = FALSE;
                }
                break;
            default:
                break;
            }
        }

        if (io->is_valid == FALSE) {
            msg_error( ERROR_MSG_UNREC_FILE, io->in_file );
            clean_exit( EXIT_ERROR_GENERAL );
        }

        io++;
    }

    return ret;
}


/**
 * get file name from source file
 *
 * */
int get_filename( char *file_name, char *src )
{
    int ret = SUCCESS;
    char *name = basename( src );
    strcat( file_name, name );
    return ret;
}


/**
 * judge directory exists
 */
int dir_exists( const char *dir )
{
    struct stat st;
    return ( stat( dir, &st ) == 0 && S_ISDIR( st.st_mode ) == 1 );
}


/**
 * get output file name
 *
 * */
int get_out_file( opt_applier module_id, char *out, char *in )
{
    int ret = SUCCESS;
    unsigned char is_f90 = FALSE;
    char ext[MAX_LEN_EXTENSION];

    get_extension( ext, in );
    if (( strcmp( ext, EXTENSION_F90_LOWER ) == 0 ) ||
         ( strcmp( ext, EXTENSION_F90_UPPER ) == 0 )) {
        is_f90 = TRUE;
    }

    /** select output directory */
    get_tempdir( out );
    strcat( out, CODE_PATH_DELIM );

    /** get output file name from input file name */
    ret = get_filename( out, in );
    switch (module_id) {
    case MOD_PP:
        switch (g_lang_id) {
        case LANGID_C:
            change_extension( out, EXTENSION_I );
            break;
        case LANGID_F:
            if (is_f90) {
                change_extension( out, EXTENSION_F90_LOWER );
            } else {
                change_extension( out, EXTENSION_F_LOWER );
            }
            break;
        default:
            /** do nothing */
            break;
        } /** switch (lang_id ) */
        break;
    case MOD_L2X:
        change_extension( out, EXTENSION_XML );
        break;
    case MOD_LX2X:
        change_extension( out, EXTENSION_XML );
        break;
    case MOD_X2L:
        switch (g_lang_id) {
        case LANGID_C:
            change_extension( out, EXTENSION_C );
            break;
        case LANGID_F:
            change_extension( out, EXTENSION_F90_UPPER );
            break;
        }
        break;
    case MOD_NTV:
        if (GET_DONT_LINK() &&
            GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_LNK_OUTPUT))) {
            strcpy( out, GET_OPT_ARG_TBL(opt_idx(OPT_LNK_OUTPUT), 0 ));
        } else {
            change_extension( out, EXTENSION_O );
            strcpy( out, basename(out));
        }
        break;
    case MOD_LNK:
        /** do nothing */
        break;
    default:
        break;
    }

    return ret;
}


/**
 * get input file name
 *
 * */
int get_in_file( opt_applier module_id, char *in, char *out, io_file *io )
{
    int ret = SUCCESS;
    char temp_in[MAX_INPUT_FILE_PATH];
    unsigned char is_f90 = FALSE;
    char ext[MAX_LEN_EXTENSION];

    get_extension( ext, io->in_file );
    if (( strcmp( ext, EXTENSION_F90_LOWER ) == 0 ) ||
         ( strcmp( ext, EXTENSION_F90_UPPER ) == 0 )) {
        is_f90 = TRUE;
    }

    /** select output directory */
    get_tempdir( in );
    strcat( in, CODE_PATH_DELIM );

    /** get input file name from output file name */
    memset( temp_in, 0x00, MAX_INPUT_FILE_PATH );
    ret = get_filename( temp_in, out );
    strcat( in, temp_in );
    switch (module_id) {
    case MOD_PP:
        /** do nothing */
        break;
    case MOD_L2X:
        switch (g_lang_id) {
        case LANGID_C:
            change_extension( in, EXTENSION_I );
            break;
        case LANGID_F:
            if (is_f90) {
                change_extension( in, EXTENSION_F90_LOWER );
            } else {
                change_extension( in, EXTENSION_F_LOWER );
            }
            break;
        default:
            break;
            /** do nothing */
        }
        /** replace path if the source file is preprocessed */
        if (io->is_preped) {
            strcpy( in, io->in_file );
        }
        break;
    case MOD_LX2X:
        change_extension( in, EXTENSION_XML );
        break;
    case MOD_X2L:
        change_extension( in, EXTENSION_XML );
        break;
    case MOD_NTV:
        switch (g_lang_id) {
        case LANGID_C:
            change_extension( in, EXTENSION_C );
            break;
        case LANGID_F:
            change_extension( in, EXTENSION_F90_UPPER );
            break;
        default:
            /** do nothing */
            break;
        }
        break;
    case MOD_LNK:
        /** do nothing */
        break;
    default:
        break;
    }

    return ret;
}


/**
 * concat module name for each modules
 *
 * */
int get_module_name( char *dst, opt_applier module )
{
    int ret = SUCCESS;
    char config[MAX_CONF_LEN];

    switch (module) {
    case MOD_PP:
        ret = get_config( config, CONFIG_PATH_PP );
        break;
    case MOD_L2X:
        ret = get_config( config, CONFIG_PATH_L2X );
        break;
    case MOD_LX2X:
        ret = get_config( config, CONFIG_PATH_LX2X );
        break;
    case MOD_X2L:
        ret = get_config( config, CONFIG_PATH_X2L );
        break;
    case MOD_NTV:
        ret = get_config( config, CONFIG_PATH_NTV );
        break;
    case MOD_LNK:
        ret = get_config( config, CONFIG_PATH_LNK );
        break;
    default :
        /** do nothing */
        ret = FAILED;
        break;
    }

    if (ret == SUCCESS) {
        strcpy( dst, config );
    }

    return ret;
}


/**
 * concat option string except "W"
 *
 * */
int get_option_without_w( char *dst, char *src, int opt_tbl_idx )
{
    int ret = SUCCESS;
    unsigned char w_pp     = FALSE;
    unsigned char w_l2x    = FALSE;
    unsigned char w_lx2x   = FALSE;
    unsigned char w_x2l    = FALSE;
    unsigned char w_native = FALSE;
    unsigned char w_linker = FALSE;
    opt_pair *pair;

    pair = (opt_pair*)&(opt_pair_table[opt_tbl_idx]);
    w_pp = (strcmp( pair->opt_value, OPT_PP_P ) == 0);
    w_l2x = (strcmp( pair->opt_value, OPT_L2X_F ) == 0);
    w_lx2x = (strcmp( pair->opt_value, OPT_LX2X_X ) == 0);
    w_x2l = (strcmp( pair->opt_value, OPT_X2L_B ) == 0);
    w_native = (strcmp( pair->opt_value, OPT_NTV_N ) == 0);
    w_linker = (strcmp( pair->opt_value, OPT_LNK_L ) == 0);
    if (w_pp || w_l2x || w_lx2x || w_x2l || w_native || w_linker) {
        strcat( dst, src + strlen(OPT_PP_P));
    } else {
        strcat( dst, src );
    }
    strcat(dst, " ");

    ret = strlen( dst );

    return ret;
}


/**
 * get configuration from table using module id & config kind
 * 
 * */
int get_config_with_id( char *value, opt_applier module_id, int config_kind )
{
    int ret = SUCCESS;
    int i;
    om_config *config = config_table;

    for (i = 0; i < CNT_CONFIG; i++) {
        if ((module_id == (config+i)->module_id) &&
             (config_kind == (config+i)->config_kind)) {
            strcpy(value, (config+i)->config_value);
            ret = SUCCESS;
            return ret;
        } else {
            ret = FAILED;
        }
    }

    return ret;
}


/**
 * concat option string for each modules
 *
 * */
int get_option_each_module( char *dst, opt_applier module )
{
    int ret = SUCCESS;
    int i, j;
    int next_opt_idx = 0, apply_cnt;
    char *option = dst;
    char pri_opt[MAX_INPUT_FILE_PATH];
    const opt_pair *pair;
    opt_set *set;

    memset( option, 0x00, OPT_BUF );
    ret = get_config_with_id( pri_opt, module, CONFIG_KIND_OPT );
    strcpy( option, pri_opt );
    strcat( option, CODE_SPACE );
    next_opt_idx += strlen( pri_opt ) + 1;

    for (i = 0; i < NUM_OPT_KINDS; i++) {

        pair = &opt_pair_table[i];
        set  = &g_manage_info.options[i];

        if(set->is_applied == FALSE)
            continue;

        if (pair->opt_applier != module) {
            /* pass -I to preprocessor and l2x */

            if (g_lang_id != LANGID_F || module != MOD_L2X)
                continue;
            if(cmp_opt_value(
                pair->opt_value, OPT_PP_INCPATH) == FALSE) {
                continue;
            }
        }

        apply_cnt = pair->is_multiple ? set->apply_cnt : 1;

        for (j = 0; j < apply_cnt; j++) {
            ret = get_option_without_w( option + next_opt_idx,
                                        set->opt_value[j], i );
            next_opt_idx += ret;
            if (set->opt_argument[j] != NULL) {
                strcat( option + next_opt_idx, CODE_SPACE );
                strcat( option + next_opt_idx + 1,
                        set->opt_argument[j] );
                next_opt_idx += strlen( set->opt_argument[j] ) + 1;
            }
        }

        strcat( option + next_opt_idx, CODE_SPACE );
        next_opt_idx++;
    }

    switch (module) {
    case MOD_PP:
    case MOD_NTV:
    case MOD_LNK:
        strcat( option, g_unrec_opt_str );
        break;
    default:
        break;
    }

    return ret;
}


/**
 * get all object file name
 *
 * */
int get_all_obj_files( char *dst )
{
    int ret = SUCCESS;
    int i;
    int in_file_cnt = g_manage_info.file.next_index;
    io_file *io = &(g_manage_info.file.info[0]);
    int next_src_idx = 0;
    char ext[MAX_LEN_EXTENSION];

    for (i = 0; i<in_file_cnt; i++) {
        get_extension( ext, io->in_file );

        if ((strcmp( ext, EXTENSION_O ) == 0) ||
            (strcmp( ext, EXTENSION_A ) == 0)) {
            strcat( dst + next_src_idx, io->in_file );
            next_src_idx += strlen( io->in_file );
        } else {
            strcat( dst + next_src_idx, basename( io->in_file ));
            change_extension( dst + next_src_idx, OM_OBJFILE_EXT );
            next_src_idx += (strlen( basename( io->in_file )) - strlen(ext) + strlen(OM_OBJFILE_EXT));
        }
        strcat( dst + next_src_idx, CODE_SPACE );
        next_src_idx++;
        io++;
    }

    return ret;
}


/**
 * execute specific lower module
 *
 */
int exec_module( opt_applier id )
{
    int ret = SUCCESS;
    int i;
    int cnt_file;
    io_file *io = &(g_manage_info.file.info[0]);

    /** get module string and options */
    ret = get_module_name( g_module_path, id );

    msg_normal("executing %s ...", g_module_path);

    ret = get_option_each_module( g_option_buf, id );

    /** execute setenv */
    ret = setenv( ENV_OM_OPTIONS, g_option_buf, TRUE );

    /** execute lower module for each file */
    cnt_file = g_manage_info.file.next_index;
    switch (id) {
    case MOD_PP:
        for (i = 0; i < cnt_file; i++) {
            if (( io->is_valid )
                 && (io->is_preped == FALSE )
                 && (io->is_compiled == FALSE )) {
                strcpy( g_cmd_buf, g_module_path );
                memset( g_outfile_buf, 0x00, MAX_INPUT_FILE_PATH );
                ret = get_out_file( id, g_outfile_buf, io->in_file );
                concat_io_file( g_cmd_buf, g_outfile_buf, io->in_file );
                exe_system( g_cmd_buf );
            }
            io++;
        }
        break;
    case MOD_L2X:
        /** do same proccess */
    case MOD_LX2X:
        /** do same proccess */
    case MOD_X2L:
        /** do same proccess */
    case MOD_NTV:
        for (i = 0; i<cnt_file; i++) {
            if (( io->is_valid ) && ( io->is_compiled == FALSE )) {
                strcpy( g_cmd_buf, g_module_path );
                memset( g_outfile_buf, 0x00, MAX_INPUT_FILE_PATH );
                memset( g_infile_buf, 0x00, MAX_INPUT_FILE_PATH );
                ret = get_out_file( id, g_outfile_buf, io->in_file );
                ret = get_in_file( id, g_infile_buf, io->in_file, io );
                concat_io_file( g_cmd_buf, g_outfile_buf, g_infile_buf );
                exe_system( g_cmd_buf );
            }
            io++;
        }
        break;
    case MOD_LNK:
        strcpy( g_cmd_buf, g_module_path );
        strcat( g_cmd_buf, CODE_SPACE );
        if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_LNK_OUTPUT))) {
            strcat( g_cmd_buf, GET_OPT_ARG_TBL(opt_idx(OPT_LNK_OUTPUT), 0 ));
        } else {
            strcat( g_cmd_buf, DEFAULT_OUT_FILE );
        }
        memset( g_all_obj, 0x00, MAX_INPUT_FILE );
        ret = get_all_obj_files( g_all_obj );
        strcat( g_cmd_buf, CODE_SPACE );
        strcat( g_cmd_buf, g_all_obj );
        exe_system( g_cmd_buf );
        break;
    case MOD_DRV:
        /** do nothing */
        break;
    }

    return ret;
}


/**
 * execute all lower modules
 */
int exec_all_module( void )
{
    int ret = SUCCESS;
    char tempdir[MAX_INPUT_FILE_PATH];

    /** make temporary directory */
    ret = get_tempdir( tempdir );
    if (g_is_temp_dir_created == FALSE && dir_exists( tempdir ) == FALSE) {
        if (mkdir( tempdir, 0777 ) != 0) {
            msg_error( ERROR_MSG_TEMP_DIR, tempdir );
            clean_exit( EXIT_ERROR_GENERAL );
        }
        g_is_temp_dir_created = TRUE;
    }

    ret = exec_module( MOD_PP );
    if (ret != SUCCESS) {
        clean_exit( EXIT_ERROR_PROCESSOR );
    }

    ret = exec_module( MOD_L2X );
    if (ret != SUCCESS) {
        clean_exit( EXIT_ERROR_L2X );
    }

    ret = exec_module( MOD_LX2X );
    if (ret != SUCCESS) {
        clean_exit( EXIT_ERROR_LX2X );
    }

    ret = exec_module( MOD_X2L );
    if (ret != SUCCESS) {
        clean_exit( EXIT_ERROR_X2L );
    }

    ret = exec_module( MOD_NTV );
    if (ret != SUCCESS) {
        clean_exit( EXIT_ERROR_NATIVE );
    }

    if (GET_DONT_LINK()) {
        /** dont execute linker */
    } else {
        ret = exec_module( MOD_LNK );
        if (ret != SUCCESS) {
            clean_exit( EXIT_ERROR_LINKER );
        }
    }

    return ret;
}



/**
 * set configuration from file
 *
 * */
int set_config( void )
{
    int ret = SUCCESS;
    int j, blen;
    FILE *fp = NULL;
    char conf_path[MAX_INPUT_FILE_PATH];
    char buf[MAX_CONF_LEN];
    char config_id[MAX_CONF_LEN];
    char config_value[MAX_CONF_LEN];
    om_config *configs = config_table;

    if (GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_CONF))) {
        strcpy( conf_path,  GET_CONFFILE());
    } else {
        /** edit path to configuration file */
        strcpy( conf_path, OM_DRIVER_CONF_DIR );
        strcat( conf_path, CODE_PATH_DELIM );
        if (g_lang_id == LANGID_C) {
            strcat( conf_path, CONF_FILE_C );
        } else {
            strcat( conf_path, CONF_FILE_F );
        }
    }

    /** open configuration file */
    fp = fopen( conf_path, "r" );
    if (fp == NULL) {
        msg_error( ERROR_MSG_CANT_OPEN, NULL );
        clean_exit( EXIT_ERROR_GENERAL );
    }

    /** set configuration value to table from configuration file */
    while (fgets( buf, MAX_CONF_LEN, fp ) != NULL) {
        if (buf[0] == CODE_SHARP || buf[0] == CODE_NEW_LINE) {
            continue;
        }
        blen = strlen(buf);
        for (j = 0; j<blen; j++) {
            if (buf[j] == CODE_SUBSTITUTE) {
                strcpy( config_id, buf );
                config_id[j] = CODE_END_OF_STR;
                strcpy( config_value, (buf + (j + 1)));
                *(config_value + (strlen(config_value)) - 1) = CODE_END_OF_STR;
                break;
            }
        }
        for (j = 0; j<CNT_CONFIG; j++) {
            if (strcmp( config_id, (configs + j)->config_id ) == 0) {
                strcpy( (configs + j)->config_value, config_value );
                (configs + j)->is_applied = TRUE;
                break;
            }
        }
    }

    return ret;
}



const char* get_lang_code()
{
    switch(g_lang_id) {
    case LANGID_C:
        return LANGCODE_C;
    case LANGID_F:
        return LANGCODE_F;
    default:
        abort();
        return NULL;
    }
}

/**
 * set all environmental value
 *
 * */
int set_all_env( void )
{
    char trans_line[STR_BUF] = {0};
    char *trans_ids[MAX_TRANSLATORS];
    int i, idx_opt, args;
    int ret = SUCCESS;

    setenv( ENV_OMNI_HOME, OMNI_HOME, TRUE );
    const char *verbose = GET_OPT_IS_APPLIED_TBL(opt_idx(OPT_DRV_VERBOSE)) ? "1" : "0";
    setenv(ENV_OM_VERBOSE, verbose, TRUE);
    setenv(ENV_OM_LANGID, get_lang_code(), TRUE);

    /* set env by translator option */
    idx_opt = opt_idx(OPT_LX2X_TRANS);
    args = GET_OPT_APPLY_CNT_TBL(idx_opt);
    memset(trans_ids, 0, sizeof(trans_ids));

    for(i = 0; i < args; ++i) {
        char *p, *tid, *tidopt, *envname;
        const char *arg;
        int j, exists;
        arg = GET_OPT_VALUE_TBL(idx_opt, i);
        tidopt = strdup(arg);
        tid = &tidopt[2];

        exists = 0;
        for(j = 0; j < MAX_TRANSLATORS; ++j) {
            if(trans_ids[j] == NULL) {
                trans_ids[j] = strdup(tid);
                break;
            }
            if(strcmp(trans_ids[j], tid) == 0) {
                exists = 1;
                break;
            }
        }

        if(exists)
            continue;
        if(trans_line[0] != '\0')
            strcat(trans_line, " ");
        strcat(trans_line, tid);
        envname = (char*)malloc(strlen(arg) + 16);

        for(p = tid; *p != '\0'; ++p)
            *p = toupper(*p);
        sprintf(envname, ENV_OM_USE, tid);
        setenv(envname, "1", TRUE);
    }

    setenv(ENV_OM_TRANSLATORS, trans_line, TRUE);

    return ret;
}


/**
 * initialize
 *
 * @brief this function set compile driver option.
 *
 * */
int init( void )
{
    int ret = SUCCESS;

    /** set option values */
    int optionsSz = sizeof(opt_set) * NUM_OPT_KINDS;
    g_manage_info.options = (opt_set*)malloc( optionsSz );
    memset( g_manage_info.options, 0, optionsSz );
    ret = set_options( (g_manage_info.opt_cnt - 1),
                       (g_manage_info.opt_values + 1));

    /** set driver private option */
    ret = set_driver_option();

    if (g_num_in_file == 0) {
        msg_error( ERROR_MSG_NO_INPFILE, NULL );
        clean_exit( EXIT_ERROR_GENERAL );
    }

    /** set all environmental value */
    ret = set_all_env();

    /** set configuration values */
    ret = set_config();

    return ret;
}


/**
 * main
 *
 * @brief this function execute initialization, reading source file,
 *        all lower module.
 *
 * @param[in] argc the number of argument (contains execute file name)
 * @param[in] argv values of arguments
 *
 * */
int main( int argc, char** argv )
{
    int ret = SUCCESS;

    /** check the number of argument */
    if (argc < MINIMUM_ARGUMENT) {
        SET_VERBOSE( TRUE );
        msg_error( ERROR_MSG_ARGUMENT, NULL );
        disp_help();
        clean_exit( EXIT_ERROR_GENERAL );
    }

    /** set argument to management informantion */
    g_manage_info.opt_cnt = argc;
    g_manage_info.opt_values = argv;

    /** execute initialization */
    ret = init();

    /** validate input source file */
    ret = validate_src();

    /** execute all modules */
    ret = exec_all_module();

    /** exit successfully */
    msg_normal( NORMAL_MSG_COMPLETED );
    clean_exit( EXIT_SUCCESS );

} /** main */

