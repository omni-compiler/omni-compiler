/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F95-main.c
 */

/* Fortran lanuage front-end */

#include "F-front.h"
#include "F-output-xcodeml.h"
#include <math.h>

/* for debug */
int debug_flag = 0;
FILE *debug_fp;
FILE *diag_file;

/* default variable type */
BASIC_DATA_TYPE defaultIntType = TYPE_INT;
BASIC_DATA_TYPE defaultSingleRealType = TYPE_REAL;
BASIC_DATA_TYPE defaultDoubleRealType = TYPE_DREAL;

/* Treat implicit typed variable as undefined. */
int doImplicitUndef = FALSE;

/* the number of errors */
int nerrors;

char *original_source_file_name = NULL;
char *source_file_name = NULL;
char *output_file_name = NULL;
FILE *source_file,*output_file;

/* -save=? */
int auto_save_attr_kb = -1;

int endlineno_flag = 0;
int ocl_flag = 0;
int max_name_len = -1;
int dollar_ok = 0; // accept '$' in identifier or not. 

extern int      yyparse _ANSI_ARGS_((void));
static void     check_nerrors _ANSI_ARGS_((void));

static int
getVarSize(str)
char *str;
{
    int ret = 0;
    char *ePtr = NULL;

    if (str == NULL || *str == '\0') {
        return 0;
    }
    ret = strtol(str, &ePtr, 10);
    if (ePtr != str) {
        return ret;
    } else {
        return 0;
    }
}

static void
cmd_error_exit EXC_VARARGS_DEF(char *, fmt)
{
    va_list args;

    fprintf(stderr, "error: " );
    EXC_VARARGS_START(char *, fmt, args);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
    check_nerrors();
    exit(EXITCODE_ERR);
}

static void
cmd_warning EXC_VARARGS_DEF(char *, fmt)
{ 
    va_list args;

    fprintf(stderr, "warning: " );
    EXC_VARARGS_START(char *, fmt, args);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
    check_nerrors();
}

/* xmodule path */
char xmodule_path[MAX_PATH_LEN] = { 0 };

/* module compile(-MC=) arg.  */
int mcLn_no = -1;
long mcStart, mcEnd;

/* for fork/exec or system in module compile.  */;
char *myName;
/* has termination element.  */
char *includeDirv[MAXINCLUDEDIRV + 1];
int includeDirvI = 0;

/* -MC?  */
int flag_module_compile = FALSE;

int flag_do_module_cache = TRUE;

static void
usage()
{
    const char *usages[] = {
        "",
        "OPTIONS",
        "",
        /* "-d", */
        /* "-yd", */
        "-o [outputfile]           specify output file path.",
        "-I [dirpath]              specify include directory path.",
        "-fopenmp                  enable openmp translation.",
        "-fxmp                     enable XcalableMP translation.",
        "-Kscope-omp               enable conditional compilation.",
        "-force-fixed-format       read file as fixed format.",
        "-force-free-format        read file as free format.",
        "-max-line-length[=n]      set max columns in a line.",
        "                          (default n=72 in fixed format,",
        "                                   n=132 in free format.)",
        "-max-cont-line[=n]        set max number of continuation lines.",
        "                          (default n=255)",
        "-force-c-comment          enable 'c' comment in free format.",
        "-f77                      use F77 spec intrinsic.",
        "-f90                      use F90 spec intrinsic.",
        "-f95                      use F95 spec intrinsic.",
        "-u                        use no implicit type.",
        "-r[N]                     set double precision size (default N=8).",
        "--save[=n]                add save attribute than n kbytes except",
        "                          in a recursive function and common "
                                  "variables.",
        "                          (default n=1)",
        "-max_name_len=n           set maximum identifier name length.",
        "-fdollar-ok               enable using \'$\' in identifier.",
	"-fleave-comment           leave comment in xcodeml file.",
	"-endlineno                output the endlineno attribute.",
        "",
        "internal options:",
        "-d                        enable debug mode.",
        "-no-module-cache          always load module from file.",

        NULL
    };
    const char * const *p = usages;

    fprintf(stderr, "usage: %s <OPTIONS> <INPUT_FILE>\n", myName);

    while (*p != NULL) {
        fprintf(stderr, "%s\n", *(p++));
    }
}

int
main(argc, argv) 
int argc; 
char *argv[]; 
{ 
    extern int fixed_format_flag;
    extern int max_line_len;
    extern int max_cont_line;
    extern int flag_force_c_comment;
#if YYDEBUG != 0
    extern int yydebug;
#endif

    int parseError = 0;
    int flag_force_fixed_format = -1; /* unset */

#ifdef HAVE_SETLOCALE
    (void)setlocale(LC_ALL, "C");
#endif /* HAVE_SETLOCALE */

    char message_str[128];

    myName = argv[0];
    source_file_name = NULL;
    output_file_name = NULL;

    --argc;
    ++argv;

    /* parse command line */
    while(argc > 0 && argv[0] != NULL) {
        if (argv[0][0] != '-' && argv[0][0] != '\0') {
            if(source_file_name != NULL)
                cmd_error_exit("too many arguments");

            source_file_name = argv[0];
            original_source_file_name = source_file_name;

            if((source_file = fopen(source_file_name,"r")) == NULL)
                cmd_error_exit("cannot open file : %s",source_file_name);
        } else if (strcmp(argv[0],"-d") == 0) {
            ++debug_flag;
        } else if (strcmp(argv[0], "-yd") == 0) {
#if YYDEBUG != 0
            yydebug = 1;
#endif
        } else if (strcmp(argv[0], "-o") == 0) {
            argc--;
            argv++;
            if((argc == 0) || (argv[0] == NULL) || (argv[0] == '\0')) {
                cmd_error_exit("output file name not specified.");
            }
            output_file_name = argv[0];
            if((output_file = fopen(output_file_name,"w")) == NULL)
                cmd_error_exit("cannot open file : %s",output_file_name);
            if(xmodule_path[0] == '\0') {
                char *dir = strdup(output_file_name);
                strcpy(xmodule_path, dirname(dir));
            }
        } else if (strcmp(argv[0],"-fopenmp") == 0){
            OMP_flag = TRUE;   /* enable openmp */
        } else if (strcmp(argv[0],"-fxmp") == 0){
	    XMP_flag = TRUE;   /* enable XcalableMP */
        } else if (strcmp(argv[0],"-Kscope-omp") == 0){
	    cond_compile_enabled = TRUE;
        } else if (strcmp(argv[0],"-fleave-comment") == 0){
	    leave_comment_flag = TRUE;
        } else if (strncmp(argv[0], "-max-line-length=", 17) == 0) {
            max_line_len = atoi(argv[0] + 17);
        } else if (strncmp(argv[0], "-max-cont-line=", 15) == 0) {
            max_cont_line = atoi(argv[0] + 15);
        } else if (strcmp(argv[0], "-u") == 0) {
            doImplicitUndef = TRUE;
        } else if (strcmp(argv[0], "-C") == 0) {
            cmd_warning("Array range check is not supported, just ignore this option.");
        } else if (strncmp(argv[0], "-r", 2) == 0) {
            int sz = getVarSize(argv[0] + 2);
            switch (sz) {
            case SIZEOF_FLOAT:      defaultSingleRealType = TYPE_REAL; break;
            case SIZEOF_DOUBLE:     defaultSingleRealType = TYPE_DREAL; break;
            default: {
                cmd_error_exit(
                    "invalid single-real size %d, must be %d or %d.",
                    sz, SIZEOF_FLOAT, SIZEOF_DOUBLE);
            }
            }
        } else if (strncmp(argv[0], "-d", 2) == 0) {
            int sz = getVarSize(argv[0] + 2);
            switch (sz) {
            case SIZEOF_FLOAT:      defaultDoubleRealType = TYPE_REAL; break;
            case SIZEOF_DOUBLE:     defaultDoubleRealType = TYPE_DREAL; break;
            default: {
                cmd_error_exit(
                    "invalid double-real size %d, must be %d or %d.",
                    sz, SIZEOF_FLOAT, SIZEOF_DOUBLE);
            }
            }
        } else if (strcmp(argv[0], "-force-fixed-format") == 0) {
            if (flag_force_fixed_format != -1)
                cmd_warning("it seems to be set both of -force-fixed-format and -force-free-format.");
            /* do not file name checking for fixed-format.  */
            flag_force_fixed_format = TRUE;
        } else if (strcmp(argv[0], "-force-free-format") == 0) {
            if (flag_force_fixed_format != -1)
                cmd_warning("it seems to be set both of -force-fixed-format and -force-free-format.");
            flag_force_fixed_format = FALSE;
        } else if (strncmp(argv[0], "-TD=", 4) == 0) {
            /* -TD=directoryPath */
            strcpy(xmodule_path, argv[0] + 4);
            if(strlen(xmodule_path) == 0)
                cmd_error_exit("invalid path after -TD.");
        } else if (strcmp(argv[0], "-module-compile") == 0) {
            flag_module_compile = TRUE;
#if 0
        } else if (strncmp(argv[0], "-MC=", 4) == 0) {
            /* -MC=fileName:N:StartSeekPoint:EndSeekPoint */
            /* internal option for module compile.  */
            flag_module_compile = TRUE;
            if (sscanf (argv[0] + 4, "%d:%ld:%ld",
                        &mcLn_no, &mcStart, &mcEnd) != 3) {
                cmd_error_exit ("internal error on internal command option, does not match: -MC=fileName:N:StartSeekPoint:EndSeekPoint");
            }
#endif
        } else if (strncmp(argv[0], "-I", 2) == 0) {
            /* -I <anotherDir> or -I<anotherDir> */
            char *path;
            if (strlen(argv[0]) == 2) {
                /* -I <anotherDir> */
                if (--argc <= 0)
                    cmd_error_exit("no arg for -I.");
                argv++;
                path = argv[0];
            } else {
                /* -I<anotherDir> */
                path = argv[0] + 2;
            }

            if (includeDirvI < 256) {
                includeDirv[includeDirvI++] = path;
            } else {
                cmd_error_exit(
                    "over the maximum include search dir. vector, %d",
                    MAXINCLUDEDIRV);
            }
        } else if (strcmp(argv[0], "-f77") == 0) {
            langSpecSet = LANGSPEC_F77_SET;
        } else if (strcmp(argv[0], "-f90") == 0) {
            langSpecSet = LANGSPEC_F90_SET;
        } else if (strcmp(argv[0], "-f95") == 0) {
            langSpecSet = LANGSPEC_F95_SET;
        } else if (strcmp(argv[0], "-force-c-comment") == 0) {
            /* enable c comment in free format.  */
            flag_force_c_comment = TRUE;
            if (flag_force_fixed_format == 1)  {
                cmd_warning("no need option for enable c comment(-force-c-comment) in fixed format mode(.f or .F).");
            }
#if 0
        } else if (strcmp(argv[0], "-xmod") == 0) {
            char *path;
            symbol_filter * filter;
            if (--argc <= 0)
                cmd_error_exit("no arg for -xmod.");
            argv++;
            path = argv[0];

            filter = push_new_filter();

            FILTER_USAGE(filter) = RENAME;

            return use_module_to(path, stdout) ? EXITCODE_OK : EXITCODE_ERR;
#endif
        } else if (strcmp(argv[0], "--save") == 0) {
            auto_save_attr_kb = 1; // 1kbytes
        } else if (strncmp(argv[0], "--save=", 7) == 0) {
            auto_save_attr_kb = atoi(argv[0] + 7);
            if (auto_save_attr_kb < 0)
                cmd_error_exit("invalid value after -save.");
        } else if (strncmp(argv[0], "-max-name-len=", 14) == 0) {
            max_name_len = atoi(argv[0]+14);
            if (max_name_len < MAX_NAME_LEN_F77){
                max_name_len = MAX_NAME_LEN_F77;
                sprintf(message_str, "attempt to set too small value for max_name_len. use %d.", 
                       MAX_NAME_LEN_F77);
                cmd_warning(message_str);
            }
            if (max_name_len > MAX_NAME_LEN_UPPER_LIMIT){
                max_name_len = MAX_NAME_LEN_UPPER_LIMIT;
                sprintf(message_str, "attempt to set too large value for max_name_len. use %d.", 
                        MAX_NAME_LEN_UPPER_LIMIT);
                cmd_warning(message_str);
            }
            if( debug_flag ){
                sprintf(message_str, "max_name_len = %d", max_name_len);
                cmd_warning(message_str);
            }
        } else if (strcmp(argv[0],"-fdollar-ok") == 0){
	    dollar_ok = 1;   /* enable using '$' in identifier */
        } else if (strcmp(argv[0], "-endlineno") == 0) {
 	    endlineno_flag = 1;
	} else if (strcmp(argv[0], "-ocl") == 0) {
 	    ocl_flag = 1;
        } else if (strcmp(argv[0], "--help") == 0) {
            usage();
            exit(0);
#if 0
        } else if (strncmp(argv[0], "-m", 2) == 0) {
            cmd_warning("quad/multiple precision is not supported.");
#endif
        } else if (strcmp(argv[0], "-no-module-cache") == 0) {
            flag_do_module_cache = FALSE;
        } else {
            cmd_error_exit("unknown option : %s",argv[0]);
        }
        --argc;
        ++argv;
    }

    if (source_file_name == NULL) {
        source_file = stdin;
        /* set this as option.  */
        if (flag_force_fixed_format != -1) {
            fixed_format_flag = flag_force_fixed_format;
        }
    } else {
        /* file name checking for fixed-format.  */
        if (flag_force_fixed_format == -1) { /* unset?  */
            const char *dotPos = strrchr(source_file_name, '.');
            if (dotPos != NULL &&
                (strcasecmp(dotPos, ".f") == 0 ||
                 strcasecmp(dotPos, ".f77") == 0)) {
                fixed_format_flag = TRUE;
            }
        } else {
            fixed_format_flag = flag_force_fixed_format;
        }
    }

    if(output_file_name == NULL) {
        output_file = stdout;
        if(getcwd(xmodule_path, MAX_PATH_LEN) == NULL) {
            cmd_error_exit("cannot get current directory");
        }
    }

    if( max_line_len < 0 ){ /* unset */
        max_line_len = fixed_format_flag ? DEFAULT_MAX_LINE_LEN_FIXED :
                       DEFAULT_MAX_LINE_LEN_FREE;
    }

    if( max_name_len < 0 ){ /* unset */
        max_name_len = fixed_format_flag?MAX_NAME_LEN_F77:MAX_NAME_LEN_F03;
    }

    /* DEBUG */
    debug_fp = stderr;
    diag_file = stderr;

    initialize_lex();
    initialize_compile();

    /* start processing */
    parseError = yyparse();
    if (nerrors != 0 ||
        parseError != 0) {
        goto Done;
    }
    nerrors = 0;

    /* end compile */
    if (unit_ctl_level != 0) {
        error("contains stack is not closed properly");
    }
    finalize_compile();
    if (nerrors != 0) {
        goto Done;
    }

    final_fixup();
    if (nerrors != 0) {
        goto Done;
    }
    
    /* output XcodeML/Fortran code */
    output_XcodeML_file();

Done:
    if (nerrors != 0) {
        if (output_file_name != NULL) {
            fclose(output_file);
            (void)unlink(output_file_name);
        }
    }

    return (nerrors ? EXITCODE_ERR : EXITCODE_OK);
}

const char *
search_include_path(const char * filename)
{
    int i;
    int length;
    static char path[MAX_PATH_LEN];
    FILE * fp;

    if ((fp = fopen(filename, "r")) != NULL) {
        fclose(fp);
        return filename;
    }

    length = strlen(filename);

    if (includeDirvI <= 0 ||
        (length >= 1 && strncmp("/", filename, 1) == 0) ||
        (length >= 2 && strncmp("./", filename, 2) == 0) ||
        (length >= 3 && strncmp("../", filename, 3) == 0)) {
        return filename;
    }

    for (i = 0; i < includeDirvI; i++) {
        strcpy(path, includeDirv[i]);
        strcat(path, "/");
        strcat(path, filename);

        if ((fp = fopen(path, "r")) != NULL) {
            fclose(fp);
            return path;
        }
    }

    return NULL;
}

void
where(lineno_info *ln)
{ 
    extern char *current_module_name;

    /* print location of error  */
    if (ln != NULL) {
        if (current_module_name == NULL)
            fprintf(stderr, "\"%s\", line %d: ",FILE_NAME(ln->file_id), ln->ln_no);
        else
            fprintf(stderr, "\"%s:%s\", line %d: ",
                    FILE_NAME(ln->file_id),
                    current_module_name,
                    ln->ln_no);
    } else {
        fprintf(stderr, "\"??\", line ??: ");
    }
}

/* nonfatal error message */
/* VARARGS0 */
void
error EXC_VARARGS_DEF(char *, fmt)
{ 
    va_list args;

    ++nerrors;
    where(current_line);
    EXC_VARARGS_START(char *, fmt, args);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
    check_nerrors();
}


/* VARARGS0 */
void
error_at_node EXC_VARARGS_DEF(expr, x)
{
    va_list args;
    char *fmt;

    ++nerrors;
    EXC_VARARGS_START(expr, x, args);
    where(EXPR_LINE(x));
    fmt = va_arg(args, char *);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
    check_nerrors();
}


/* VARARGS0 */
void
error_at_id EXC_VARARGS_DEF(ID, x)
{
    va_list args;
    char *fmt;

    ++nerrors;
    EXC_VARARGS_START(ID, x, args);
    where(ID_LINE(x));
    fmt = va_arg(args, char *);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
    check_nerrors();
}




/* VARARGS0 */
void
warning_at_node EXC_VARARGS_DEF(expr, x)
{ 
    va_list args;
    char *fmt;

    where(EXPR_LINE(x)); /*, "WarnAtNode"); */
    fprintf(stderr,"warning: ");
    EXC_VARARGS_START(expr, x, args);
    fmt = va_arg(args, char *);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
}


void
warning_at_id EXC_VARARGS_DEF(ID, x)
{ 
    va_list args;
    char *fmt;

    where(ID_LINE(x));
    fprintf(stderr,"warning: ");
    EXC_VARARGS_START(ID, x, args);
    fmt = va_arg(args, char *);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
}


static void
check_nerrors()
{
    if(nerrors > 30)
    {
        /* give the compiler the benefit of the doubt */
        fprintf(stderr, 
                "too many error, cannot recover from earlier errors: goodbye!\n" );
        exit(EXITCODE_ERR);
    }
}

/* compiler error: die */
/* VARARGS1 */
void
fatal EXC_VARARGS_DEF(char *, fmt)
{ 
    va_list args;
    
    where(current_line); /*, "Fatal");*/
    fprintf(stderr, "compiler error: " );
    EXC_VARARGS_START(char *, fmt, args);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
    abort();
}

int warning_flag = FALSE; 

/* warning with lineno_info */
void
warning_lineno (lineno_info * info, char * fmt, ...)
{
    va_list args;

    if (warning_flag) return;
    where(info); /*, "Warn");*/
    EXC_VARARGS_START(char *, fmt, args);
    fprintf(stderr, "warning: " );
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
}

/* warning */
void
warning EXC_VARARGS_DEF(char *, fmt)
{
    va_list args;

    if (warning_flag) return;
    where(current_line); /*, "Warn");*/
    EXC_VARARGS_START(char *, fmt, args);
    fprintf(stderr, "warning: " );
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n" );
    fflush(stderr);
}
