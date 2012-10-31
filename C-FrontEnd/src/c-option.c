/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-option.c
 */

#include <libgen.h>
#include <time.h>
#include "c-option.h"

#define OFFSETOF(type, member)        ((size_t)&((type*)0)->member)
#define ALIGNOF_SAMPLE_STRUCT(type)   struct { char a; type b; }
#define ALIGNOF(type)                 OFFSETOF(ALIGNOF_SAMPLE_STRUCT(type), b)

unsigned int s_sizeAddr         = SIZEOF_VOID_P;
unsigned int s_sizeChar         = SIZEOF_UNSIGNED_CHAR;
unsigned int s_sizeWChar        = SIZEOF_UNSIGNED_LONG;
unsigned int s_sizeShort        = SIZEOF_UNSIGNED_SHORT;
unsigned int s_sizeInt          = SIZEOF_UNSIGNED_INT;
unsigned int s_sizeLong         = SIZEOF_UNSIGNED_LONG;
unsigned int s_sizeLongLong     = SIZEOF_UNSIGNED_LONG_LONG;
unsigned int s_sizeFloat        = SIZEOF_FLOAT;
unsigned int s_sizeDouble       = SIZEOF_DOUBLE;
unsigned int s_sizeLongDouble   = SIZEOF_LONG_DOUBLE;
unsigned int s_sizeBool         = SIZEOF__BOOL;
unsigned int s_basicTypeSizeOf  = BT_UNSIGNED_INT;

unsigned int s_alignAddr        = ALIGNOF_VOID_P;
unsigned int s_alignChar        = ALIGNOF_UNSIGNED_CHAR;
unsigned int s_alignWChar       = ALIGNOF_UNSIGNED_LONG;
unsigned int s_alignShort       = ALIGNOF_UNSIGNED_SHORT;
unsigned int s_alignInt         = ALIGNOF_UNSIGNED_INT;
unsigned int s_alignLong        = ALIGNOF_UNSIGNED_LONG;
unsigned int s_alignLongLong    = ALIGNOF_UNSIGNED_LONG_LONG;
unsigned int s_alignFloat       = ALIGNOF_FLOAT;
unsigned int s_alignDouble      = ALIGNOF_DOUBLE;
unsigned int s_alignLongDouble  = ALIGNOF_LONG_DOUBLE;
unsigned int s_alignBool        = ALIGNOF__BOOL;

unsigned int s_verbose                  = 0;
unsigned int s_rawlineNo                = 0;
unsigned int s_suppressSameTypes        = 0;
unsigned int s_xoutputInfo              = 1;
unsigned int s_supportGcc               = 1;
unsigned int s_useBuiltinWchar          = 0;
unsigned int s_useShortWchar            = 0;
unsigned int s_useIntWchar              = 0;
unsigned int s_transFuncInInit          = 0;
unsigned int s_useXMP                   = 0;
unsigned int s_useACC                   = 0;
unsigned int s_debugSymbol		= 0;
unsigned int s_arrayToPointer           = 0;

#define CEXPR_OPTVAL_CHARLEN 128

#define COPT_SIZE       "--size-"
#define COPT_ALIGN      "--align-"
#define COPT_NO_WARN    "--no-warn-"


const char *s_inFile = NULL;
const char *s_outFile = NULL;
char s_anonymousCompositePrefix[CEXPR_OPTVAL_CHARLEN]       = "anon_type_";
char s_anonymousMemberPrefix[CEXPR_OPTVAL_CHARLEN]          = "anon_mem_";
char s_tmpVarPrefix[CEXPR_OPTVAL_CHARLEN]                   = "tmp_var_";
char s_gccLocalLabelPrefix[CEXPR_OPTVAL_CHARLEN]            = "local_label_";
char s_xmlIndent[CEXPR_OPTVAL_CHARLEN]                      = "  ";
char s_xmlEncoding[CEXPR_OPTVAL_CHARLEN]                    = "ISO-8859-1";
char s_sourceFileName[CEXPR_OPTVAL_CHARLEN]                 = "<stdin>";

char s_timeStamp[128];
CCOL_SList s_noWarnIds;


/**
 * \brief
 * free static data for processing options
 */
void freeStaticOptionData()
{
    CCOL_SListNode *site;
    CCOL_SL_FOREACH(site, &s_noWarnIds) {
        free(CCOL_SL_DATA(site));
    }
}


/**
 * \brief
 * judge msg has ID which is specified to ignore 
 *
 * @param msg
 *      message
 * @return
 *      0:no, 1:yes
 */
int isWarnableId(const char *msg)
{
    CCOL_SListNode *site;
    CCOL_SL_FOREACH(site, &s_noWarnIds) {
        const char *msgId = CCOL_SL_DATA(site);
        int n = strlen(msgId);
        if(strncmp(msg, msgId, n) == 0 && msg[n] == ':')
            return 0;
    }

    return 1;
}


/**
 * \brief
 * set timestamp
 */
void setTimestamp()
{
    const time_t t = time(NULL);
    struct tm *ltm = localtime(&t);
    strftime(s_timeStamp, sizeof(s_timeStamp), "%F %T", ltm);
}


/**
 * \brief
 * post process of procOptions()
 */
PRIVATE_STATIC void
afterProcOptions()
{
    if(s_useShortWchar) {
        s_sizeWChar = s_sizeShort;
        s_alignWChar = s_alignShort;
        s_wcharType = BT_SHORT;
    } else if (s_useIntWchar) {
        s_sizeWChar = s_sizeInt;
        s_alignWChar = s_alignInt;
        s_wcharType = BT_INT;
    }
}


/**
 * \brief
 * show usage
 */
PRIVATE_STATIC void
usage(const char *argv0)
{
    const char *usages[] = {
        "",
        "OPTIONS",
        "",
        "-o [outputfile]           specify output file path.",
        "--m32                     set long and pointer to 32 bits.",
        "--m64                     set long and pointer to 64 bits.",
        "--size-[type]=[n]         set size of type to n.",
        "                          type: char, wchar, short, int, long, longlong,",
        "                                float, double, longdouble, bool",
        "--align-[type]=[n]        set alignment of size to n.",
        "                          this option has no affect.",
        "--no-warn-[WID]           suppress warnings specified ID.",
        "--builtin-wchar           treats wchar_t as built-in type.",
        "--short-wchar             set size of wchar_t to short size.",
        "                          default size is long size.",
        "--trans-func-in-init      enable translation for function call in",
        "                          initializer.",
        "--rawlineno               output raw line number in preprocessed code",
        "                          to XcodeML tags and error message.",
        "--anon-type-prefix=[pfx]  specify prefix for anonymous types.",
        "--anon-mem-prefix=[pfx]   specify prefix for anonymous members.",
        "--tmp-var-prefix          specify prefix for variables which will be",
        "                          generated at syntax tree transformation.",
        "--local-label-prefix      specify prefix for label name of gcc's __label__",
        "                          transformation.",
        "--no-gcc-support          treat gcc built-in keywords as undefined keywords.",
        "--suppress-typeid         suppress same type in XcodeML typeTable.",
        "                          (this is heavy process)",
        "--xindent=[n]             set indent width n.",
        "--no-xinfo                suppress XcodeProgram attributes and line number",
        "                          attributes.",
        "-fxmp                     translate xmp pragma directive.",
        "--verbose                 print syntax tree transformation statistics and",
        "                          processing status.",
        "",
    };

    char buf[MAX_NAME_SIZ];
    strcpy(buf, argv0);
    char *progname = basename(buf);

    fprintf(stdout, "usage: %s <OPTIONS> <INPUT_FILE>\n", progname);

    for(int i = 0; i < sizeof(usages) / sizeof(usages[0]); ++i) {
        fputs( usages[i], stdout );
        fputs( "\n", stdout );
    }
}


/**
 * \brief
 * show version info
 */
PRIVATE_STATIC void
version()
{
    fputs(C_FRONTEND_NAME " " C_FRONTEND_VER "\n\n", stdout);
}


/**
 * \brief
 * set option '--xindent'
 *
 * @param val
 *      indent
 */
PRIVATE_STATIC void
setXmlIndent(const char *val)
{
    int indent = atoi(val);
    if(indent < 0 || indent > 16)
        return;

    for(int i = 0; i < indent; ++i)
        s_xmlIndent[i] = ' ';
    s_xmlIndent[indent] = 0;
}


//! type size sets
typedef enum CSizeSetEnum {
    CSIZESET_M32,
    CSIZESET_M64,
} CSizeSetEnum;



/**
 * \brief
 * set option '--m32', '--m64'
 *
 * @param szSet
 *      type size set
 * @return
 *      0:failed, 1:ok
 */
PRIVATE_STATIC void
setOptSizeSet(CSizeSetEnum szSet)
{
    switch(szSet) {
    case CSIZESET_M32:
        s_sizeAddr         = 4;
        s_sizeLong         = 4;
        s_sizeLongLong     = 8;
        s_sizeLongDouble   = 12;

        s_alignAddr        = 4;
        s_alignLong        = 4;
        s_alignLongLong    = 8;
        s_alignLongDouble  = 4;

        s_basicTypeSizeOf  = BT_UNSIGNED_INT;
        break;
    case CSIZESET_M64:
        s_sizeAddr         = 8;
        s_sizeLong         = 8;
        s_sizeLongLong     = 8;
        s_sizeLongDouble   = 16;

        s_alignAddr        = 8;
        s_alignLong        = 8;
        s_alignLongLong    = 8;
        s_alignLongDouble  = 16;

        s_basicTypeSizeOf  = BT_UNSIGNED_LONG;
        break;
    }
}


/**
 * \brief
 * set option '--size-*'
 *
 * @param typeName
 *      type name
 * @param val
 *      size
 * @return
 *      0:failed, 1:ok
 */
PRIVATE_STATIC int
setOptSize(const char *typeName, const char *val)
{
    int v = atoi(val);
    if(v < 0 || v > 0x10000) {
        fprintf(stderr, CERR_506, COPT_SIZE);
        return 0;
    }

    if(strcmp(typeName, "voidptr") == 0) {
        s_sizeAddr = v;
        s_basicTypeSizeOf  = (v <= 4) ? BT_UNSIGNED_INT : BT_UNSIGNED_LONG;
    } else if(strcmp(typeName, "char") == 0) {
        s_sizeChar = v;
    } else if(strcmp(typeName, "wchar") == 0) {
        s_sizeWChar = v;
    } else if(strcmp(typeName, "short") == 0) {
        s_sizeShort = v;
    } else if(strcmp(typeName, "int") == 0) {
        s_sizeInt = v;
    } else if(strcmp(typeName, "long") == 0) {
        s_sizeLong = v;
    } else if(strcmp(typeName, "longlong") == 0) {
        s_sizeLongLong = v;
    } else if(strcmp(typeName, "float") == 0) {
        s_sizeFloat = v;
    } else if(strcmp(typeName, "double") == 0) {
        s_sizeDouble = v;
    } else if(strcmp(typeName, "longdouble") == 0) {
        s_sizeLongDouble = v;
    } else if(strcmp(typeName, "bool") == 0) {
        s_sizeBool = v;
    } else {
        fprintf(stderr, CERR_507, COPT_SIZE);
        return 0;
    }

    return 1;
}


/**
 * \brief
 * set option '--align-*'
 *
 * @param typeName
 *      type name
 * @param val
 *      alignment
 * @return
 *      0:failed, 1:ok
 */
PRIVATE_STATIC int
setOptAlign(const char *typeName, const char *val)
{
    int v = atoi(val);
    if(v < 0 || v > 0x10000) {
        fprintf(stderr, CERR_506, COPT_SIZE);
        return 0;
    }

    if(strcmp(typeName, "voidptr") == 0) {
        s_alignAddr = v;
    } else if(strcmp(typeName, "char") == 0) {
        s_alignChar = v;
    } else if(strcmp(typeName, "wchar") == 0) {
        s_alignWChar = v;
    } else if(strcmp(typeName, "short") == 0) {
        s_alignShort = v;
    } else if(strcmp(typeName, "int") == 0) {
        s_alignInt = v;
    } else if(strcmp(typeName, "long") == 0) {
        s_alignLong = v;
    } else if(strcmp(typeName, "longlong") == 0) {
        s_alignLongLong = v;
    } else if(strcmp(typeName, "float") == 0) {
        s_alignFloat = v;
    } else if(strcmp(typeName, "double") == 0) {
        s_alignDouble = v;
    } else if(strcmp(typeName, "longdouble") == 0) {
        s_alignLongDouble = v;
    } else if(strcmp(typeName, "bool") == 0) {
        s_alignBool = v;
    } else {
        fprintf(stderr, CERR_507, COPT_SIZE);
        return 0;
    }

    return 1;
}


/**
 * \brief
 * set option '--no-warn-*'
 *
 * @param msgId
 *      message ID
 * @return
 *      0:failed, 1:ok
 */
PRIVATE_STATIC int
setOptNoWarn(const char *msgId)
{
    CCOL_SL_CONS(&s_noWarnIds, ccol_strdup(msgId, MAX_NAME_SIZ));
    return 1;
}


/**
 * \brief
 * process options
 *
 * @param argc
 *      argc of main function
 * @param argv
 *      argv of main function
 * @return
 *      0:error, 1:ok
 */
int
procOptions(int argc, char **argv)
{
    #define NEEDS_ARGUMENT(opt)\
        if(narg == NULL) {\
            fprintf(stderr, CERR_503, opt);\
            fprintf(stderr, "\n");\
            return 0;\
        }

    #define NEEDS_VALUE(opt)\
        if(val == NULL) {\
            fprintf(stderr, CERR_505, opt);\
            fprintf(stderr, "\n");\
            return 0;\
        }

    const int lenOptSize = strlen(COPT_SIZE);
    const int lenOptAlign = strlen(COPT_ALIGN);
    const int lenOptNoWarn = strlen(COPT_NO_WARN);

    memset(&s_noWarnIds, 0, sizeof(s_noWarnIds));

    for(int i = 1; i < argc; ++i) {
        char *arg, *narg = NULL, *val = NULL;
        arg = argv[i];
        if(arg[0] == 0)
            continue;
        if(i < argc - 1)
            narg = argv[i + 1];

        if(arg[0] != '-') {
            if(s_inFile == NULL)
                s_inFile = arg;
            else {
                fprintf(stderr, CERR_504, arg);
                fprintf(stderr, "\n");
                return 0;
            }
        } else {
            for(char *p = arg; *p; ++p) {
                if(*p == '=') {
                    *p = 0;
                    val = ++p;
                    break;
                }
            }

            if(strcmp(arg, "-o") == 0) {
                NEEDS_ARGUMENT("-o");
                s_outFile = narg;
                ++i;
            } else if(strcmp(arg, "--builtin-wchar") == 0) {
                s_useBuiltinWchar = 1;
            } else if(strcmp(arg, "--short-wchar") == 0) {
                s_useShortWchar = 1;
            } else if(strcmp(arg, "--int-wchar") == 0) {
                s_useIntWchar = 1;
            } else if(strcmp(arg, "--no-gcc-support") == 0) {
                s_supportGcc = 0;
            } else if(strcmp(arg, "--verbose") == 0) {
                s_verbose = 1;
            } else if(strcmp(arg, "--suppress-typeid") == 0) {
                s_suppressSameTypes = 1;
            } else if(strcmp(arg, "--trans-func-in-init") == 0) {
                s_transFuncInInit = 0;
            } else if(strcmp(arg, "--rawlineno") == 0) {
                s_rawlineNo = 1;
            } else if(strcmp(arg, "--no-xinfo") == 0) {
                s_xoutputInfo = 0;
            } else if(strcmp(arg, "--anon-type-prefix") == 0) {
                NEEDS_VALUE("--anon-type-prefix");
                strcpy(s_anonymousCompositePrefix, val);
            } else if(strcmp(arg, "--anon-mem-prefix") == 0) {
                NEEDS_VALUE("--anon-mem-prefix");
                strcpy(s_anonymousMemberPrefix, val);
            } else if(strcmp(arg, "--tmp-var-prefix") == 0) {
                NEEDS_VALUE("--tmp-var-prefix");
                strcpy(s_tmpVarPrefix, val);
            } else if(strcmp(arg, "--local-label-prefix") == 0) {
                NEEDS_VALUE("--local-label-prefix");
                strcpy(s_gccLocalLabelPrefix, val);
            } else if(strcmp(arg, "--xindent") == 0) {
                NEEDS_VALUE("--xindent");
                setXmlIndent(val);
            } else if(strcmp(arg, "--m32") == 0) {
                setOptSizeSet(CSIZESET_M32);
            } else if(strcmp(arg, "--m64") == 0) {
                setOptSizeSet(CSIZESET_M64);
            } else if(strcmp(arg, "-fxmp") == 0) {
                // s_useXMP = 1;
            } else if(strcmp(arg, "-facc") == 0) {
                s_useACC = 1;
            } else if(strcmp(arg, "-fopenmp") == 0) {
                /* accept but no action */
	    } else if(strcmp(arg, "--array-to-pointer") == 0) {
	        s_arrayToPointer = 1;
            } else if(strncmp(arg, COPT_SIZE, lenOptSize) == 0) {
                NEEDS_VALUE(COPT_SIZE);
                if(setOptSize(arg + lenOptSize, val) == 0)
                    return 0;
            } else if(strncmp(arg, COPT_ALIGN, lenOptAlign) == 0) {
                NEEDS_VALUE(COPT_ALIGN);
                if(setOptAlign(arg + lenOptAlign, val) == 0)
                    return 0;
            } else if(strncmp(arg, COPT_NO_WARN, lenOptNoWarn) == 0) {
                if(setOptNoWarn(arg + lenOptNoWarn) == 0)
                    return 0;
            } else if(strcmp(arg, "--version") == 0) {
                version();
                exit(0);
            } else if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
                usage(argv[0]);
                exit(0);
            } else {
                fprintf(stderr, CERR_500, arg);
                return 0;
            }
        }
    }

    afterProcOptions();

    return 1;
}


/**
 * \brief
 * get basic type size
 *
 * @param bt
 *      basic type
 * @return
 *      size
 */
unsigned int
getBasicTypeSize(CBasicTypeEnum bt)
{
    switch(bt) {
    case BT_VOID:
    case BT_CHAR:
    case BT_UNSIGNED_CHAR:
        return s_sizeChar;
    case BT_WCHAR:
        return s_sizeWChar;
    case BT_SHORT:
    case BT_UNSIGNED_SHORT:
        return s_sizeShort;
    case BT_INT:
    case BT_UNSIGNED_INT:
        return s_sizeInt;
    case BT_LONG:
    case BT_UNSIGNED_LONG:
        return s_sizeLong;
    case BT_LONGLONG:
    case BT_UNSIGNED_LONGLONG:
        return s_sizeLongLong;
    case BT_FLOAT:
    case BT_FLOAT_IMAGINARY:
        return s_sizeFloat;
    case BT_DOUBLE:
    case BT_DOUBLE_IMAGINARY:
        return s_sizeDouble;
    case BT_LONGDOUBLE:
    case BT_LONGDOUBLE_IMAGINARY:
        return s_sizeLongDouble;
    case BT_BOOL:
        return s_sizeBool;
    case BT_FLOAT_COMPLEX:
        return (s_sizeFloat * 2);
    case BT_DOUBLE_COMPLEX:
        return (s_sizeDouble * 2);
    case BT_LONGDOUBLE_COMPLEX:
        return (s_sizeLongDouble * 2);
    default:
        ABORT();
        return 0;
    }
}


/**
 * \brief
 * get basic type alignment
 *
 * @param bt
 *      basic type
 * @return
 *      alignment     
 */
unsigned int
getBasicTypeAlign(CBasicTypeEnum bt)
{ 
    switch(bt) {
    case BT_VOID:
    case BT_CHAR:
    case BT_UNSIGNED_CHAR:
        return s_alignChar;
    case BT_WCHAR:
        return s_alignWChar;
    case BT_SHORT:
    case BT_UNSIGNED_SHORT:
        return s_alignShort;
    case BT_INT:
    case BT_UNSIGNED_INT:
        return s_alignInt;
    case BT_LONG:
    case BT_UNSIGNED_LONG:
        return s_alignLong;
    case BT_LONGLONG:
    case BT_UNSIGNED_LONGLONG:
        return s_alignLongLong;
    case BT_FLOAT:
    case BT_FLOAT_COMPLEX:
    case BT_FLOAT_IMAGINARY:
        return s_alignFloat;
    case BT_DOUBLE:
    case BT_DOUBLE_COMPLEX:
    case BT_DOUBLE_IMAGINARY:
        return s_alignDouble;
    case BT_LONGDOUBLE:
    case BT_LONGDOUBLE_COMPLEX:
    case BT_LONGDOUBLE_IMAGINARY:
        return s_alignLongDouble;
    case BT_BOOL:
        return s_alignBool;
    default:
        ABORT();
    }
    return 0;
}

