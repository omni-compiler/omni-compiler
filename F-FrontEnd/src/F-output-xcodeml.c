/**
 * \file F-output-xcodeml.c
 */

#include "F-front.h"
#include "F-output-xcodeml.h"
#include "module-manager.h"

#define CHAR_BUF_SIZE 65536

#define ARRAY_LEN(a)    (sizeof(a) / sizeof(a[0]))

#ifdef SIMPLE_TYPE
#define ADDR2UINT_TABLE_SIZE 1024
static void *addr2uint_table[ADDR2UINT_TABLE_SIZE];
static int addr2uint_idx = 0;
int Addr2Uint(void *x)
{
    int i;
    for (i = 0; i < addr2uint_idx; ++i) {
        if (addr2uint_table[i] == x)
            return i;
    }
    if (addr2uint_idx >= ADDR2UINT_TABLE_SIZE)
        fatal("type too much");
    addr2uint_table[addr2uint_idx++] = x;
    return addr2uint_idx - 1;
}
#endif

int is_emitting_for_submodule;

extern int      flag_module_compile;

static void     outx_expv(int l, expv v);
static void     outx_functionDefinition(int l, EXT_ID ep);
static void     outx_interfaceDecl(int l, EXT_ID ep);
static void     outx_definition_symbols(int l, EXT_ID ep);
static void     outx_declarations(int l, EXT_ID parent_ep);
static void     outx_id_declarations(int l, ID id_list, int expectResultVar, const char *functionName);
static void     collect_types(EXT_ID extid);
static void     collect_types_inner(EXT_ID extid);
static void     collect_type_desc(expv v);
static int      id_is_visibleVar(ID id);
static int      id_is_visibleVar_for_symbols(ID id);
static void     mark_type_desc_in_id_list(ID ids);


char s_timestamp[CEXPR_OPTVAL_CHARLEN] = { 0 };
char s_xmlIndent[CEXPR_OPTVAL_CHARLEN] = "  ";

EXT_ID current_function_stack[MAX_UNIT_CTL_CONTAINS+1];
int current_function_top = 0;
#define CRT_FUNCEP          current_function_stack[current_function_top]
#define CRT_FUNCEP_PUSH(ep) current_function_top++;                    \
                            current_function_stack[current_function_top] = (ep)
#define CRT_FUNCEP_POP      current_function_stack[current_function_top] = (ep); \
                            current_function_top--

typedef struct type_ext_id {
    EXT_ID ep;
    struct type_ext_id *next;
} *TYPE_EXT_ID;

#define FOREACH_TYPE_EXT_ID(te, headte) \
    for ((te) = (headte); (te) != NULL ; (te) = (te)->next)

#define FUNC_EXT_LINK_ADD(te, list, tail) \
    { if((list) == NULL || (tail) == NULL) (list) = (te); \
      else (tail)->next = (te); \
      (tail) = (te); }

static TYPE_DESC    type_list,type_list_tail;
static TYPE_DESC    tbp_list,tbp_list_tail;
static TYPE_EXT_ID  type_module_proc_list, type_module_proc_last;
static TYPE_EXT_ID  type_ext_id_list, type_ext_id_last;
static FILE         *print_fp;
static char         s_charBuf[CHAR_BUF_SIZE];
static int          is_outputed_module = FALSE;

#define GET_EXT_LINE(ep) \
    (EXT_LINE(ep) ? EXT_LINE(ep) : \
    EXT_PROC_ID_LIST(ep) ? ID_LINE(EXT_PROC_ID_LIST(ep)) : NULL)

static int is_emitting_module = FALSE;

static void
set_module_emission_mode(int mode) {
    is_emitting_module = mode;
}

int
is_emitting_xmod(void) {
    return is_emitting_module;
}

static const char*
xtag(enum expr_code code)
{
    switch(code) {

    /*
     * constants
     */
    case STRING_CONSTANT:           return "FcharacterConstant";
    case INT_CONSTANT:              return "FintConstant";
    case FLOAT_CONSTANT:            return "FrealConstant";
    case COMPLEX_CONSTANT:          return "FcomplexConstant";

    /*
     * declarations
     */
    case F_DATA_DECL:               return "FdataDecl";
    case F_EQUIV_DECL:              return "FequivalenceDecl";
    case F_COMMON_DECL:             return "FcommonDecl";

    /*
     * general statements
     */
    case EXPR_STATEMENT:            return "exprStatement";
    case IF_STATEMENT:              return "FifStatement";
    case F_DO_STATEMENT:            return "FdoStatement";
    case F_DOWHILE_STATEMENT:       return "FdoWhileStatement";
    case F_SELECTCASE_STATEMENT:    return "FselectCaseStatement";
    case F03_SELECTTYPE_STATEMENT:  return "selectTypeStatement";
    case F_CASELABEL_STATEMENT:     return "FcaseLabel";
    case F03_CLASSIS_STATEMENT:     return "typeGuard";
    case F03_TYPEIS_STATEMENT:      return "typeGuard";
    case F_WHERE_STATEMENT:         return "FwhereStatement";
    case F_RETURN_STATEMENT:        return "FreturnStatement";
    case F_CONTINUE_STATEMENT:      return "continueStatement";
    case GOTO_STATEMENT:
    case F_COMPGOTO_STATEMENT:      return "gotoStatement";
    case STATEMENT_LABEL:           return "statementLabel";
    case F_FORMAT_DECL:             return "FformatDecl";
    case F_STOP_STATEMENT:          return "FstopStatement";
    case F_PAUSE_STATEMENT:         return "FpauseStatement";
    case F_PRAGMA_STATEMENT:        return "FpragmaStatement";
    case F_LET_STATEMENT:           return "FassignStatement";
    case F95_CYCLE_STATEMENT:       return "FcycleStatement";
    case F95_EXIT_STATEMENT:        return "FexitStatement";
    case F_ENTRY_STATEMENT:         return "FentryDecl";
    case F_FORALL_STATEMENT:        return "forallStatement";

    /*
     * IO statements
     */
    case F_WRITE_STATEMENT:         return "FwriteStatement";
    case F_PRINT_STATEMENT:         return "FprintStatement";
    case F_READ_STATEMENT:          return "FreadStatement";
    case F_READ1_STATEMENT:         return "FreadStatement";
    case F_OPEN_STATEMENT:          return "FopenStatement";
    case F_CLOSE_STATEMENT:         return "FcloseStatement";
    case F_BACKSPACE_STATEMENT:     return "FbackspaceStatement";
    case F_ENDFILE_STATEMENT:       return "FendFileStatement";
    case F_REWIND_STATEMENT:        return "FrewindStatement";
    case F_INQUIRE_STATEMENT:       return "FinquireStatement";

    case F03_FLUSH_STATEMENT:       return "FflushStatement";

    /*
     * F90/95 Pointer related
     */
    case F95_POINTER_SET_STATEMENT: return "FpointerAssignStatement";
    case F95_ALLOCATE_STATEMENT:    return "FallocateStatement";
    case F95_DEALLOCATE_STATEMENT:  return "FdeallocateStatement";
    case F95_NULLIFY_STATEMENT:     return "FnullifyStatement";

    /*
     * expressions
     */
    case FUNCTION_CALL:             return "functionCall";
    case F_FUNC:                    return "Ffunction";
    case IDENT:
    case F_VAR:
    case F_PARAM:                   return "Var";
    case F95_MEMBER_REF:            return "FmemberRef";
    case ARRAY_REF:                 return "FarrayRef";
    case F_SUBSTR_REF:              return "FcharacterRef";
    case F95_ARRAY_CONSTRUCTOR:     return "FarrayConstructor";
    case F03_TYPED_ARRAY_CONSTRUCTOR: return "FarrayConstructor";
    case F95_STRUCT_CONSTRUCTOR:    return "FstructConstructor";
    case XMP_COARRAY_REF:           return "FcoArrayRef";

    /*
     * operators
     */
    case PLUS_EXPR:                 return "plusExpr";
    case MINUS_EXPR:                return "minusExpr";
    case UNARY_MINUS_EXPR:          return "unaryMinusExpr";
    case MUL_EXPR:                  return "mulExpr";
    case DIV_EXPR:                  return "divExpr";
    case POWER_EXPR:                return "FpowerExpr";
    case F_CONCAT_EXPR:             return "FconcatExpr";
    case LOG_EQ_EXPR:               return "logEQExpr";
    case LOG_NEQ_EXPR:              return "logNEQExpr";
    case LOG_GE_EXPR:               return "logGEExpr";
    case LOG_GT_EXPR:               return "logGTExpr";
    case LOG_LE_EXPR:               return "logLEExpr";
    case LOG_LT_EXPR:               return "logLTExpr";
    case LOG_AND_EXPR:              return "logAndExpr";
    case LOG_OR_EXPR:               return "logOrExpr";
    case LOG_NOT_EXPR:              return "logNotExpr";
    case F_EQV_EXPR:                return "logEQVExpr";
    case F_NEQV_EXPR:               return "logNEQVExpr";

    case F95_USER_DEFINED_BINARY_EXPR:          return "userBinaryExpr";
    case F95_USER_DEFINED_UNARY_EXPR:           return "userUnaryExpr";

    case F2008_SYNCALL_STATEMENT:    return "syncAllStatement";
    case F2008_SYNCIMAGES_STATEMENT: return "syncImagesStatement";
    case F2008_SYNCMEMORY_STATEMENT: return "syncMemoryStatement";
    case F2008_CRITICAL_STATEMENT:   return "criticalStatement";
    case F2008_LOCK_STATEMENT:       return "lockStatement";
    case F2008_UNLOCK_STATEMENT:     return "unlockStatement";

    case F2008_BLOCK_STATEMENT:      return "blockStatement";

    /*                          
     * misc.                    
     */                         
    case F_IMPLIED_DO:              return "FdoLoop";
    case F_INDEX_RANGE:             return "indexRange";

    /*
     * module.
     */
    case F03_USE_INTRINSIC_STATEMENT: 
    case F95_USE_STATEMENT:         return "FuseDecl";
    case F03_USE_ONLY_INTRINSIC_STATEMENT:
    case F95_USE_ONLY_STATEMENT:    return "FuseOnlyDecl";

    /*
     * F2003 statement
     */
    case F03_IMPORT_STATEMENT:      return "FimportDecl";
    case F03_WAIT_STATEMENT:        return "FwaitStatement";

    /*
     * invalid or no corresponding tag
     */
    case ERROR_NODE:
    case LIST:
    case DEFAULT_LABEL:
    case F_SCENE_RANGE_EXPR:
    case ID_LIST:
    case VAR_DECL:
    case EXT_DECL:
    case FIRST_EXECUTION_POINT:
    case F_PROGRAM_STATEMENT:
    case F_BLOCK_STATEMENT:
    case F_SUBROUTINE_STATEMENT:
    case F_FUNCTION_STATEMENT:
    case F_INCLUDE_STATEMENT:
    case F_END_STATEMENT:
    case F_TYPE_DECL:
    case F_EXTERNAL_DECL:
    case F_INTRINSIC_DECL:
    case F_IMPLICIT_DECL:
    case F_NAMELIST_DECL:
    case F_SAVE_DECL:
    case F_PARAM_DECL:
    case F_DUP_DECL:
    case F_ENDDO_STATEMENT:
    case F_ELSEWHERE_STATEMENT:
    case F_ENDWHERE_STATEMENT:
    case F_ENDSELECT_STATEMENT:
    case F_IF_STATEMENT:
    case F_ELSEIF_STATEMENT:
    case F_ELSE_STATEMENT:
    case F_ENDIF_STATEMENT:
    case F_ASSIGN_LABEL_STATEMENT:
    case F_GOTO_STATEMENT:
    case F_ASGOTO_STATEMENT:
    case F_ARITHIF_STATEMENT:
    case F_CALL_STATEMENT:
    case F_CRAY_POINTER_DECL:
    case F_SET_EXPR:
    case F_LABEL_REF:
    case F_PLUS_EXPR:
    case F_MINUS_EXPR:
    case F_MUL_EXPR:
    case F_DIV_EXPR:
    case F_UNARY_MINUS_EXPR:
    case F_POWER_EXPR:
    case F_EQ_EXPR:
    case F_GT_EXPR:
    case F_GE_EXPR:
    case F_LT_EXPR:
    case F_LE_EXPR:
    case F_NE_EXPR:
    case F_OR_EXPR:
    case F_AND_EXPR:
    case F_NOT_EXPR:
    case F_UNARY_MINUS:
    case BASIC_TYPE_NODE:
    case F_TYPE_NODE:
    case F_TRUE_CONSTANT:
    case F_FALSE_CONSTANT:
    case F_ARRAY_REF:
    case F_STARSTAR:
    case F_ENDFORALL_STATEMENT:
    case F95_CONSTANT_WITH:
    case F95_TRUE_CONSTANT_WITH:
    case F95_FALSE_CONSTANT_WITH:
    case F95_ENDPROGRAM_STATEMENT:
    case F95_ENDSUBROUTINE_STATEMENT:
    case F95_ENDFUNCTION_STATEMENT:
    case F95_MODULE_STATEMENT:
    case F95_ENDMODULE_STATEMENT:
    case F95_INTERFACE_STATEMENT:
    case F95_ENDINTERFACE_STATEMENT:
    case F95_CONTAINS_STATEMENT:
    case F95_RECURSIVE_SPEC:
    case F95_PURE_SPEC:
    case F95_ELEMENTAL_SPEC:
    case F95_DIMENSION_DECL:
    case F95_TYPEDECL_STATEMENT:
    case F95_ENDTYPEDECL_STATEMENT:
    case F95_PRIVATE_STATEMENT:
    case F03_PROTECTED_STATEMENT:
    case F95_SEQUENCE_STATEMENT:
    case F95_PARAMETER_SPEC:
    case F95_ALLOCATABLE_SPEC:
    case F95_DIMENSION_SPEC:
    case F95_EXTERNAL_SPEC:
    case F95_INTENT_SPEC:
    case F95_INTRINSIC_SPEC:
    case F95_OPTIONAL_SPEC:
    case F95_POINTER_SPEC:
    case F95_SAVE_SPEC:
    case F95_TARGET_SPEC:
    case F95_PUBLIC_SPEC:
    case F95_PRIVATE_SPEC:
    case F03_PROTECTED_SPEC:
    case F03_BIND_SPEC:
    case F03_VALUE_SPEC:
    case F95_IN_EXTENT:
    case F95_OUT_EXTENT:
    case F95_INOUT_EXTENT:
    case F95_KIND_SELECTOR_SPEC:
    case F95_LEN_SELECTOR_SPEC:
    case F95_STAT_SPEC:
    case F95_TRIPLET_EXPR:
    case F95_PUBLIC_STATEMENT:
    case F95_OPTIONAL_STATEMENT:
    case F95_POINTER_STATEMENT:
    case F95_INTENT_STATEMENT:
    case F95_TARGET_STATEMENT:
    case F_ASTERISK:
    case F_EXTFUNC:
    case F_DOUBLE_CONSTANT:
    case F95_ASSIGNOP:
    case F95_DOTOP:
    case F95_POWEOP:
    case F95_MULOP:
    case F95_DIVOP:
    case F95_PLUSOP:
    case F95_MINUSOP:
    case F95_EQOP:
    case F95_NEOP:
    case F95_LTOP:
    case F95_LEOP:
    case F95_GEOP:
    case F95_GTOP:
    case F95_NOTOP:
    case F95_ANDOP:
    case F95_OROP:
    case F95_EQVOP:
    case F95_NEQVOP:
    case F95_CONCATOP:
    case F95_MODULEPROCEDURE_STATEMENT:
    case F95_ARRAY_ALLOCATION:
    case F95_ALLOCATABLE_STATEMENT:
    case F95_GENERIC_SPEC:
    case F95_USER_DEFINED:
    case XMP_CODIMENSION_SPEC:
    case EXPR_CODE_END:
    case F2008_ENDCRITICAL_STATEMENT:
    case F2008_ENDBLOCK_STATEMENT:

        fatal("invalid exprcode : %s", EXPR_CODE_NAME(code));

    case OMP_PRAGMA:
      return "OMPPragma";

    case XMP_PRAGMA:
      return "XMPPragma";

    case ACC_PRAGMA:
      return "ACCPragma";

    default:
      fatal("unknown exprcode : %d", code);
    }

    return NULL;
}

#define XTAG(v)  xtag(EXPR_CODE(v))

static void
outx_indent(int l)
{
    int i;
    for(i = 0; i < l; ++i)
        fputs(s_xmlIndent, print_fp);
}


static void
outx_printi(int l, const char *fmt, ...)
{
    outx_indent(l);
    va_list args;
    va_start(args, fmt);
    vfprintf(print_fp, fmt, args);
    va_end(args);
}

#define         outx_print(...)    outx_printi(0, __VA_ARGS__)

static void
outx_puts(const char *s)
{
    fputs(s, print_fp);
}



/**
 * output any tag with format
 */
static void
outx_tag(int l, const char *tagAndFmt, ...)
{
    sprintf(s_charBuf, "<%s>\n", tagAndFmt);
    outx_indent(l);
    va_list args;
    va_start(args, tagAndFmt);
    vfprintf(print_fp, s_charBuf, args);
    va_end(args);
}


/**
 * output any closed tag
 */
static void
outx_close(int l, const char *tag)
{
    outx_printi(l, "</%s>\n", tag);
}


#define outx_expvClose(l, v)    outx_close(l, XTAG(v))


static void
xstrcat(char **p, const char *s)
{
    int len = strlen(s);
    strcpy(*p, s);
    *p += len;
}


static const char*
getXmlEscapedStr(const char *s)
{
    static char x[CHAR_BUF_SIZE];
    const char *p = s;
    char *px = x;
    char c;
    x[0] = '\0';

    while((c = *p++)) {
        switch(c) {
        case '<':  xstrcat(&px, "&lt;"); break;
        case '>':  xstrcat(&px, "&gt;"); break;
        case '&':  xstrcat(&px, "&amp;"); break;
        case '"':  xstrcat(&px, "&quot;"); break;
        case '\'': xstrcat(&px, "&apos;"); break;
        /* \002 used as quote in F95-lexer.c */
        case '\002':  xstrcat(&px, "&quot;"); break;
        case '\r': xstrcat(&px, "\\r"); break;
        case '\n': xstrcat(&px, "\\n"); break;
        case '\t': xstrcat(&px, "\\t"); break;
        default:
            if(c <= 31 || c >= 127) {
                char buf[16];
                sprintf(buf, "&#x%x;", (unsigned int)(c & 0xFF));
                xstrcat(&px, buf);
            } else {
                *px++ = c;
            }
            break;
        }
    }

    *px = 0;

    return x;
}


/**
 * output expv enclosed with tag
 */
static void
outx_childrenWithTag(int l, const char *tag, expv v)
{
    outx_printi(l, "<%s>\n", tag);
    outx_expv(l + 1, v);
    outx_printi(l, "</%s>\n", tag);
}


/**
 * output number as intConstatnt
 */
static void
outx_intAsConst(int l, omllint_t n)
{
    outx_printi(l, "<FintConstant type=\"Fint\">%lld</FintConstant>\n", n);
}


#define STR_HAS_NO_QUOTE	0
#define STR_HAS_DBL_QUOTE	1
#define STR_HAS_SGL_QUOTE	2
/**
 * Check the string has quote character(s) or not.
 * @param str a string.
 * @return STR_HAS_NO_QUOTE: no quote character contained.
 *	<br/> STR_HAS_DBL_QUOTE: contains at least a double quote.
 *	<br/> STR_HAS_SGL_QUOTE: contains at least a single quote.
 */
static int
hasStringQuote(const char *str)
{
    char *s = strchr(str, '"');
    if (s != NULL) {
        return STR_HAS_DBL_QUOTE;
    } else if ((s = strchr(str, '\'')) != NULL) {
        return STR_HAS_SGL_QUOTE;
    } else {
        return STR_HAS_NO_QUOTE;
    }
}


static const char*
getRawString(expv v)
{
    static char buf[CHAR_BUF_SIZE];

    switch (EXPV_CODE(v)) {
    case INT_CONSTANT:
        snprintf(buf, CHAR_BUF_SIZE, OMLL_DFMT, EXPV_INT_VALUE(v));
        break;
    case STRING_CONSTANT: {
        if (EXPV_STR(v) != NULL) {
            int quote = hasStringQuote(EXPV_STR(v));
            switch (quote) {
                case STR_HAS_SGL_QUOTE:
                case STR_HAS_NO_QUOTE: {
                    snprintf(buf, CHAR_BUF_SIZE,
                             "&quot;%s&quot;", getXmlEscapedStr(EXPV_STR(v)));
                    break;
                }
                case STR_HAS_DBL_QUOTE: {
                    snprintf(buf, CHAR_BUF_SIZE,
                             "&#39;%s&#39;", getXmlEscapedStr(EXPV_STR(v)));
                    break;
                }
                default: {
                    fatal("%s: Unknown quote status??", __func__);
                    break;
                }
            }
        } else {
            snprintf(buf, CHAR_BUF_SIZE, "\"\"");
        }
        break;
    }
    case IDENT:
    case F_VAR:
    case F_PARAM:
    case F_FUNC:
        snprintf(buf, CHAR_BUF_SIZE,
                 "%s", getXmlEscapedStr(SYM_NAME(EXPV_NAME(v))));
        break;
    case ARRAY_REF:
        snprintf(buf, CHAR_BUF_SIZE,
                 "%s", getXmlEscapedStr(SYM_NAME(EXPV_NAME(EXPR_ARG1(v)))));
        break;
    default:
        abort();
    }

    return buf;
}


static void
outx_true(int cond, const char *flagname)
{
    if(cond)
        outx_print(" %s=\"true\"", flagname);
}


#define TOPT_TYPEONLY   (1 << 0)
#define TOPT_NEXTLINE   (1 << 1)
#define TOPT_CLOSE      (1 << 2)
#define TOPT_INTRINSIC  (1 << 3)


static int
has_attribute_except_func_attrs(TYPE_DESC tp)
{
    return
        /* TYPE_IS_EXTERNAL(tp) || */
        /* TYPE_IS_INTRINSIC(tp) || */
        /* TYPE_IS_RECURSIVE(tp) || */
        /* TYPE_IS_PURE(tp) || */
        /* TYPE_IS_ELEMENTAL(tp) || */
        TYPE_IS_PARAMETER(tp) ||
        TYPE_IS_ALLOCATABLE(tp) ||
        TYPE_IS_OPTIONAL(tp) ||
        TYPE_IS_POINTER(tp) ||
        TYPE_IS_SAVE(tp) ||
        TYPE_IS_TARGET(tp) ||
        TYPE_IS_PUBLIC(tp) ||
        TYPE_IS_PRIVATE(tp) ||
        TYPE_IS_PROTECTED(tp) ||
        TYPE_IS_SEQUENCE(tp) ||
        TYPE_IS_INTERNAL_PRIVATE(tp) ||
        TYPE_IS_INTENT_IN(tp) ||
        TYPE_IS_INTENT_OUT(tp) ||
        TYPE_IS_INTENT_INOUT(tp) ||
        TYPE_IS_VOLATILE(tp) ||
        TYPE_IS_VALUE(tp) ||
        TYPE_IS_CLASS(tp) ||
        TYPE_IS_PROCEDURE(tp) ||
        TYPE_IS_ASYNCHRONOUS(tp) ||
        tp->codims;
}


static int
has_attribute(TYPE_DESC tp)
{
    return
        has_attribute_except_func_attrs(tp) ||
        TYPE_IS_EXTERNAL(tp) ||
        TYPE_IS_INTRINSIC(tp) ||
        TYPE_IS_RECURSIVE(tp) ||
        TYPE_IS_PURE(tp) ||
        TYPE_IS_ELEMENTAL(tp) ||
        TYPE_IS_MODULE(tp);
}

static int
has_attribute_except_private_public(TYPE_DESC tp)
{
    int ret;
    int is_public = TYPE_IS_PUBLIC(tp);
    int is_private = TYPE_IS_PRIVATE(tp);
    int is_protected = TYPE_IS_PROTECTED(tp);
    TYPE_UNSET_PUBLIC(tp);
    TYPE_UNSET_PRIVATE(tp);
    TYPE_UNSET_PROTECTED(tp);
    ret = has_attribute(tp);
    if(is_private)
        TYPE_SET_PRIVATE(tp);
    if(is_public)
        TYPE_SET_PUBLIC(tp);
    if(is_protected)
        TYPE_SET_PROTECTED(tp);
    return ret;
}

static const char *
getBasicTypeID(BASIC_DATA_TYPE t)
{
    const char *tid = NULL;
    switch(t) {
    case TYPE_INT:          tid = "Fint"; break;
    case TYPE_CHAR:         tid = "Fcharacter"; break;
    case TYPE_LOGICAL:      tid = "Flogical"; break;
    case TYPE_REAL:         /* fall through */
    case TYPE_DREAL:        tid = "Freal"; break;
    case TYPE_COMPLEX:      /* fall through */
    case TYPE_DCOMPLEX:     tid = "Fcomplex"; break;
    case TYPE_GNUMERIC:     tid = "Fnumeric"; break;
    case TYPE_GENERIC:      /* fall through */
    case TYPE_LHS:          /* fall through too */
    case TYPE_GNUMERIC_ALL: tid = "FnumericAll"; break;
    case TYPE_MODULE:       tid = "Fvoid"; break;
    case TYPE_NAMELIST:     tid = "Fnamelist"; break;
    case TYPE_VOID:         tid = "Fvoid"; break;
    default: abort();
    }
    return tid;
}


#define checkBasic(tp)                          \
    ((tp == NULL) || (                          \
    (has_attribute_except_func_attrs(tp) == FALSE) && \
    (TYPE_KIND(tp) == NULL) &&                  \
    (IS_DOUBLED_TYPE(tp) == FALSE) &&           \
    (!tp->codims) &&                            \
    ((IS_NUMERIC(tp) ||                         \
      IS_LOGICAL(tp) ||                         \
      IS_VOID(tp) ||                            \
      IS_MODULE(tp) ||                          \
      (IS_CHAR(tp) && TYPE_CHAR_LEN(tp) == 1)))))
/**
 * get typeID
 */
static const char*
getTypeID(TYPE_DESC tp)
{
    static char buf[256];

    if (checkBasic(tp) && checkBasic(TYPE_REF(tp))) {
        strcpy(buf, getBasicTypeID(TYPE_BASIC_TYPE(tp)));
    } else {
        char pfx;

        switch(TYPE_BASIC_TYPE(tp)) {
        case TYPE_INT:          pfx = 'I'; break;
        case TYPE_CHAR:         pfx = 'C'; break;
        case TYPE_LOGICAL:      pfx = 'L'; break;
        case TYPE_REAL:         /* fall through */
        case TYPE_DREAL:        pfx = 'R'; break;
        case TYPE_COMPLEX:      /* fall through */
        case TYPE_DCOMPLEX:     pfx = 'P'; break;
        case TYPE_FUNCTION:     /* fall through */
        case TYPE_SUBR:         pfx = 'F'; break;
        case TYPE_ARRAY:        pfx = 'A'; break;
        case TYPE_STRUCT:       pfx = 'S'; break;
        case TYPE_GNUMERIC:     pfx = 'U'; break;
        case TYPE_GENERIC:      /* fall through */
        case TYPE_LHS:		/* fall through too */
        case TYPE_GNUMERIC_ALL: pfx = 'V'; break;
        case TYPE_NAMELIST:     pfx = 'N'; break;
        default: abort();
        }

        sprintf(buf, "%c" ADDRX_PRINT_FMT, pfx, Addr2Uint(tp));
    }

    return buf;
}

/**
 * output tag and type attribute
 */
static void
outx_typeAttrs(int l, TYPE_DESC tp, const char *tag, int options)
{
    if(tp == NULL) {
        outx_printi(l,"<%s", tag);
        return;
    }

    /* Output type as generic numeric type when type is not fixed */
    if(TYPE_IS_NOT_FIXED(tp) && TYPE_BASIC_TYPE(tp) == TYPE_UNKNOWN) {
        outx_printi(l,"<%s type=\"%s\"", tag, "FnumericAll");
    } else {
        outx_printi(l,"<%s type=\"%s\"", tag, getTypeID(tp));
    }

    if((options & TOPT_TYPEONLY) == 0) {

        if(TYPE_HAS_INTENT(tp)) {
            const char *intent;
            if(TYPE_IS_INTENT_IN(tp))
                intent = "in";
            else if(TYPE_IS_INTENT_OUT(tp))
                intent = "out";
            else
                intent = "inout";
            outx_print(" intent=\"%s\"", intent);
        }

        outx_true(TYPE_IS_PUBLIC(tp),           "is_public");
        outx_true(TYPE_IS_PRIVATE(tp),          "is_private");
        outx_true(TYPE_IS_PROTECTED(tp),        "is_protected");
        outx_true(TYPE_IS_POINTER(tp),          "is_pointer");
        outx_true(TYPE_IS_TARGET(tp),           "is_target");
        outx_true(TYPE_IS_OPTIONAL(tp),         "is_optional");
        outx_true(TYPE_IS_SAVE(tp),             "is_save");
        outx_true(TYPE_IS_PARAMETER(tp),        "is_parameter");
        outx_true(TYPE_IS_ALLOCATABLE(tp),      "is_allocatable");
        outx_true(TYPE_IS_SEQUENCE(tp),         "is_sequence");
        outx_true(TYPE_IS_INTERNAL_PRIVATE(tp), "is_internal_private");
        outx_true(TYPE_IS_VOLATILE(tp),         "is_volatile");
        outx_true(TYPE_IS_VALUE(tp),            "is_value");
        outx_true(TYPE_IS_PROCEDURE(tp),        "is_procedure");
        outx_true(TYPE_IS_ASYNCHRONOUS(tp),     "is_asynchronous");

        if (TYPE_PARENT(tp)) {
            outx_print(" extends=\"%s\"", getTypeID(TYPE_PARENT_TYPE(tp)));
        }
        outx_true(TYPE_IS_CLASS(tp),            "is_class");

        if (IS_STRUCT_TYPE(tp)) {
            /*
             * function/subroutine type can be abstract,
             * but XcodeML schema does not allow it.
             */
            outx_true(TYPE_IS_ABSTRACT(tp) ,    "is_abstract");
        }

        if(TYPE_HAS_BIND(tp)){
            outx_print(" bind=\"%s\"", "C"); // Only C for the moment
            if(TYPE_BIND_NAME(tp)){
                outx_print(" bind_name=\"%s\"", EXPR_STR(TYPE_BIND_NAME(tp)));
            }
        }


    }

    if((options & TOPT_INTRINSIC) > 0)
        outx_true(1, "is_intrinsic");

    if((options & TOPT_NEXTLINE) > 0)
        outx_print(">\n");
    if((options & TOPT_CLOSE) > 0)
        outx_print("/>\n");
}


#define outx_typeAttrOnly_EXPR(l, v, tag) \
    outx_typeAttrs((l), ((v) ? EXPV_TYPE(v) : NULL), tag, TOPT_TYPEONLY)

#define outx_typeAttrOnly_ID(l, id, tag) \
    outx_typeAttrs((l), ((id) ? ID_TYPE(id) : NULL), tag, TOPT_TYPEONLY)


static void
outx_typeAttrOnly_functionType(int l, TYPE_DESC tp, const char *tag)
{
    const char *tid = getTypeID(tp);
    outx_printi(l,"<%s type=\"%s\"", tag, tid);
}


static void
outx_lineno(lineno_info *li)
{
    if(li) {
        outx_print(" lineno=\"%d\"", li->ln_no);
	if (li->end_ln_no) outx_print(" endlineno=\"%d\"", li->end_ln_no);
        outx_print(" file=\"%s\"", getXmlEscapedStr(FILE_NAME(li->file_id)));
    }
}


/**
 * output tag with format and lineno attribute
 */
static void
outx_vtagLineno(int l, const char *tag, lineno_info *li, va_list args)
{
    static char buf1[CHAR_BUF_SIZE], buf2[CHAR_BUF_SIZE];
    snprintf(buf1, sizeof(buf1), "<%s", tag);
    if (args != NULL) {
        vsnprintf(buf2, sizeof(buf2), buf1, args);
    } else {
        strncpy(buf2, buf1, sizeof(buf2));
    }

    outx_indent(l);
    outx_puts(buf2);
    outx_lineno(li);
}


static void
outx_tagOfStatement(int l, expv v)
{
    outx_vtagLineno(l, XTAG(v), EXPR_LINE(v), NULL);
    outx_puts(">\n");
}


static void
outx_tagOfStatement1(int l, expv v, const char *attrs, ...)
{
    sprintf(s_charBuf, "%s%s", XTAG(v), attrs ? attrs : "");
    va_list args;
    va_start(args, attrs);
    outx_vtagLineno(l, s_charBuf, EXPR_LINE(v), args);
    va_end(args);
    outx_puts(">\n");
}


static void
outx_tagOfStatement2(int l, expv v)
{
    outx_vtagLineno(l, XTAG(v), EXPR_LINE(v), NULL);
    outx_print(">");
}


static void
outx_tagOfStatement3(int l, const char *tag, lineno_info *li)
{
    outx_vtagLineno(l, tag, li, NULL);
    outx_print(">\n");
}


/**
 * output tag of statement with construct name
 */
static void
outx_tagOfStatementWithConstructName(int l, expv v, expv cv, int hasChild)
{
    if(cv)
        sprintf(s_charBuf, "%s construct_name=\"%s\"",
            XTAG(v), SYM_NAME(EXPV_NAME(cv)));
    else
        strcpy(s_charBuf, XTAG(v));

    outx_vtagLineno(l, s_charBuf, EXPR_LINE(v), NULL);
    if(hasChild)
        outx_puts(">\n");
    else
        outx_puts("/>\n");
}


static void
outx_tagOfStatementNoChild(int l, expv v, const char *attrs, ...)
{
    sprintf(s_charBuf, "%s%s", XTAG(v), attrs ? attrs : "");
    va_list args;
    va_start(args, attrs);
    outx_vtagLineno(l, s_charBuf, EXPR_LINE(v), args);
    va_end(args);
    outx_puts("/>\n");
}


static void
outx_vtagOfDecl(int l, const char *tag, lineno_info *li, va_list args)
{
    outx_vtagLineno(l, tag, li, args);
}


static void
outx_tagOfDecl1(int l, const char *tag, lineno_info *li, ...)
{
    va_list args;
    va_start(args, li);
    outx_vtagOfDecl(l, tag, li, args);
    va_end(args);
    outx_puts(">\n");
}


static void
outx_tagOfDeclNoClose(int l, const char *tag, lineno_info *li, ...)
{
    va_list args;
    va_start(args, li);
    outx_vtagOfDecl(l, tag, li, args);
    va_end(args);
}


static void
outx_tagOfDeclNoChild(int l, const char *tag, lineno_info *li, ...)
{
    va_list args;
    va_start(args, li);
    outx_vtagOfDecl(l, tag, li, args);
    va_end(args);
    outx_puts("/>\n");
}


static void
outx_tagOfDecl(int l, const char *tag, ID id)
{
    outx_tagOfDecl1(l, tag, ID_LINE(id));
}


/**
 * output tag which has only text symbol value
 */
static void
outx_tagText(int l, const char *tag, const char *s)
{
    outx_printi(l, "<%s>%s</%s>\n", tag, s, tag);
}


#define outx_symbolName(l, s)   outx_tagText(l, "name", SYM_NAME(s))
#define outx_expvName(l, v)     outx_symbolName(l, EXPV_NAME(v))


/**
 * output a symbol with function type (not a return type)
 */
static void
outx_symbolNameWithFunctionType(int l, expv v)
{
    outx_typeAttrOnly_functionType(l, EXPV_TYPE(v), "name");
    outx_print(">%s</name>\n", SYM_NAME(EXPR_SYM(v)));
}


/**
 * output a symbol with function type (not a return type)
 */
static void
outx_symbolNameWithFunctionType_EXT(int l, EXT_ID ep)
{
    outx_typeAttrOnly_functionType(l, EXT_PROC_TYPE(ep), "name");
    outx_print(">%s</name>\n", SYM_NAME(EXT_SYM(ep)));
}


/**
 * output an identifier symbol with type
 */
static void
outx_symbolNameWithType_ID(int l, ID id)
{
    outx_typeAttrs(l, ID_TYPE(id), "name", TOPT_TYPEONLY);
    outx_print(">%s</name>\n", SYM_NAME(ID_SYM(id)));
}


/**
 * output name as statement label
 */
static void
outx_linenoNameWithFconstant(int l, expv fconst)
{
    outx_tagText(l, "name", getRawString(fconst));
}


/**
 * get scope of id
 */
static const char*
getScope(expv v)
{
    const char *scope;
    switch(EXPV_CODE(v)) {
    case F_FUNC:
    case IDENT:
    case F_VAR:     scope = "local"; break;
    case F_PARAM:   scope = "param"; break;
    default: abort();
    }

    return scope;
}


/**
 * output Var/Ffunction
 */
static void
outx_varOrFunc(int l, expv v)
{
    const char *scope = getScope(v);
    outx_typeAttrOnly_EXPR(l, v, XTAG(v));
    if(scope)
        outx_print(" scope=\"%s\"", scope);
    outx_print(">%s</%s>\n", SYM_NAME(EXPV_NAME(v)), XTAG(v));
}


/**
 * output Var of ident
 */
static void
outx_var_ID(int l, ID id)
{
    const char *scope = "local";
    const char *tag = xtag(F_VAR);
    outx_typeAttrOnly_ID(l, id, tag);
    if(scope)
        outx_print(" scope=\"%s\"", scope);
    outx_print(">%s</%s>\n", ID_NAME(id), tag);
}


/**
 * output expression tag which has type attribute
 */
static void
outx_tagOfExpression(int l, expv v)
{
    outx_typeAttrs(l, EXPV_TYPE(v), XTAG(v),
        TOPT_TYPEONLY|TOPT_NEXTLINE);
}


/**
 * output expression tag which has type attribute
 */
static void
outx_tagOfExpression1(int l, expv v, int addOptions)
{
    outx_typeAttrs(l, EXPV_TYPE(v), XTAG(v),
        TOPT_TYPEONLY|TOPT_NEXTLINE|addOptions);
}


/**
 * output id for EXT_ID
 */
static void
outx_ext_id(int l, EXT_ID ep)
{
    const char *sclass;
    assert(EXT_TAG(ep) == STG_EXT || EXT_TAG(ep) == STG_COMMON);

    if(EXT_TAG(ep) == STG_COMMON)
        return;

    /*
    sclass = (EXT_IS_DEFINED(ep) && EXT_PROC_IS_INTRINSIC(ep) == FALSE &&
        (EXT_PROC_BODY(ep) || EXT_PROC_ID_LIST(ep))) ?
            "extern_def" : "extern";
    */
    sclass = "ffunc";

    if(EXT_TAG(ep) == STG_COMMON ||
        EXT_PROC_IS_INTERFACE(ep) ||
        IS_MODULE(EXT_PROC_TYPE(ep)))
        outx_printi(l, "<id");
    else
        outx_typeAttrOnly_functionType(l, EXT_PROC_TYPE(ep), "id");
    outx_print(" sclass=\"%s\">\n",sclass);
    outx_symbolName(l + 1, EXT_SYM(ep));
    outx_printi(l,"</id>\n");
}


static const char*
get_sclass(ID id)
{
    switch(ID_CLASS(id)) {
    case CL_LABEL:
        return "flabel"; /* unused */
    case CL_PARAM:
        return "flocal";
    case CL_COMMON:
        return "fcommon_name";
        break;
    case CL_NAMELIST:
        return "fnamelist_name";
    default:
        if(ID_EQUIV_ID(id)) {
            return get_sclass(ID_EQUIV_ID(id));
        }

        switch(ID_STORAGE(id)) {
        case STG_ARG:
        case STG_TYPE_PARAM:
            return "fparam";
        case STG_EXT:
            /*
            if(PROC_CLASS(id) == P_INTRINSIC || PROC_CLASS(id) == P_EXTERNAL)
                return "extern";
            else
                return "extern_def";
            */
            return "ffunc";
        case STG_COMEQ:
        case STG_COMMON:
            return "fcommon";
        case STG_SAVE:
            return "fsave";
        case STG_AUTO:
        case STG_EQUIV:
        case STG_INDEX:
            return "flocal";
        case STG_TAGNAME:
            return "ftype_name";
        case STG_UNKNOWN:
        case STG_NONE:
            fatal("%s: illegal storage class: symbol=%s", __func__, ID_NAME(id));
            abort();
        }
        break;
    }

    return NULL;
}


/**
 * output id for ID
 */
static void
outx_id(int l, ID id)
{
    if(ID_STORAGE(id) == STG_EXT && !IS_PROCEDURE_TYPE(ID_TYPE(id)) &&  PROC_EXT_ID(id) == NULL) {
        fatal("outx_id: PROC_EXT_ID is NULL: symbol=%s", ID_NAME(id));
    }

    if (ID_CLASS(id) == CL_PROC && IS_PROCEDURE_TYPE(ID_TYPE(id))) {
        outx_typeAttrOnly_functionType(l, ID_TYPE(id), "id");
    } else if(ID_CLASS(id) == CL_PROC && PROC_EXT_ID(id)) {
        outx_typeAttrOnly_functionType(l, EXT_PROC_TYPE(PROC_EXT_ID(id)), "id");
    } else {
        outx_typeAttrOnly_ID(l, id, "id");
    }

    const char *sclass = get_sclass(id);
    outx_print(" sclass=\"%s\"", sclass);
    if(ID_IS_OFMODULE(id))
        outx_print(" declared_in=\"%s\"",
                   ID_USEASSOC_INFO(id)->module_name->s_name);
    outx_print(">\n");
    outx_symbolName(l + 1, ID_SYM(id));
    outx_close(l, "id");
}


static void
outx_namedValue(int l, const char *name, expv v, const char *defaultVal)
{
    outx_printi(l, "<namedValue name=\"%s\"", name);

    if(v && EXPV_CODE(v) != IDENT) {
        outx_print(">\n");
        outx_expv(l + 1, v);
        outx_close(l, "namedValue");
    } else {
        const char *val = v ? SYM_NAME(EXPV_NAME(v)) : defaultVal;
        outx_print(" value=\"%s\"/>\n", val);
    }
}


static void
outx_namedValueList(int l, const char **names, int namesLen,
    expv v, const char *defaultVal)
{
    struct list_node *lp;
    int i = 0;
    int l1 = l + 1;

    outx_tag(l, "namedValueList");

    FOR_ITEMS_IN_LIST(lp, v) {
        expv item = LIST_ITEM(lp);
        expv vval = NULL;
        const char *name = NULL;

        if(i < namesLen) {
            name = names[i];
        } else if(item == NULL) {
            fatal("outx_namedValueList: abbreviated argument name");
        }

        if(item) {
            if(EXPV_CODE(item) == F_SET_EXPR) {
                name = SYM_NAME(EXPV_NAME(EXPR_ARG1(item)));
                vval = EXPR_ARG2(item);
            } else {
                vval = item;
            }
        }

        outx_namedValue(l1, name, vval, defaultVal);
        i++;
    }

    outx_close(l, "namedValueList");
}


static void
outx_valueList(int l, expv v)
{
    struct list_node *lp;
    int l1 = l + 1, l2 = l1 + 1;
    expv val0, val;

    outx_tag(l, "valueList");

    FOR_ITEMS_IN_LIST(lp, v) {
        val0 = LIST_ITEM(lp);
        outx_tag(l1, "value");
        if(EXPV_CODE(val0) == F_DUP_DECL)
            val = EXPR_ARG2(val0);
        else
            val = val0;
        outx_expv(l2, val);

        if(EXPV_CODE(val0) == F_DUP_DECL) {
            outx_tag(l2, "repeat_count");
            outx_expv(l2 + 1, EXPR_ARG1(val0));
            outx_close(l2, "repeat_count");
        }

        outx_close(l1, "value");
    }

    outx_close(l, "valueList");
}


/**
 * output indexRange
 */
static void
outx_indexRange0(int l,
    ARRAY_ASSUME_KIND assumeKind,
    ARRAY_ASSUME_KIND defaultAssumeKind,
    expv lower, expv upper, expv step)
{
    const int l1 = l + 1;

    outx_printi(l, "<%s", xtag(F_INDEX_RANGE));

    if(assumeKind == ASSUMED_NONE && upper == NULL) {
        assumeKind = defaultAssumeKind;
    }

    switch(assumeKind) {
    case ASSUMED_NONE:
        break;
    case ASSUMED_SIZE:
        outx_true(TRUE, "is_assumed_size");
        break;
    case ASSUMED_SHAPE:
        outx_true(TRUE, "is_assumed_shape");
        break;
    }

    outx_print(">\n");
    if(lower)
        outx_childrenWithTag(l1, "lowerBound", lower);
    if(upper)
        outx_childrenWithTag(l1, "upperBound", upper);
    if(step)
        outx_childrenWithTag(l1, "step", step);


      outx_close(l, xtag(F_INDEX_RANGE));
}

#define outx_indexRange(l, lower, upper, step) \
    outx_indexRange0((l), \
    ASSUMED_NONE, ASSUMED_SHAPE, (lower), (upper), (step))


/**
 * output indexRange for list element
 */
static void
outx_indexRangeInList(int l, expr v)
{
    outx_indexRange(l, EXPR_ARG1(v),
        expr_list_get_n(v, 1), expr_list_get_n(v, 2));
}


/**
 * output indexRange for TYPE_DESC which represents dimension
 */
static void
outx_indexRangeOfType(int l, TYPE_DESC tp)
{
    TYPE_DESC rtp = TYPE_REF(tp);
    if(IS_ARRAY_TYPE(rtp)) outx_indexRangeOfType(l, rtp);
    outx_indexRange0(l, TYPE_ARRAY_ASSUME_KIND(tp), ASSUMED_SIZE,
    TYPE_DIM_LOWER(tp), TYPE_DIM_UPPER(tp), TYPE_DIM_STEP(tp));
}


/**
 * output FstructDecl
 */
static void
outx_structDecl(int l, ID id)
{
    outx_tagOfDecl(l, "FstructDecl", id);
    outx_symbolNameWithType_ID(l + 1, id);
    outx_close(l, "FstructDecl");
}


/**
 * output varRef of ident_descriptor
 */
static void
outx_varRef(int l, expv v, ID id)
{
    const int l1 = l + 1;
    if(v) {
        outx_typeAttrOnly_EXPR(l, v, "varRef");
        outx_print(">\n");
        outx_expv(l1, v);
    } else if(id) {
        outx_typeAttrOnly_ID(l, id, "varRef");
        outx_print(">\n");
        outx_var_ID(l1, id);
    } else {
        abort();
    }
    outx_printi(l, "</varRef>\n");
}


#define outx_varRef_EXPR(l, v)    outx_varRef(l, v, 0)
#define outx_varRef_ID(l, id)     outx_varRef(l, 0, id)


/**
 * output value
 */
static void
outx_value(int l, expv v)
{
    if(v == NULL) return;
    outx_childrenWithTag(l, "value", v);
}


/**
 * output values in list element
 */
static void
outx_valueInList(int l, expv v)
{
    list lp;

    if(v == NULL) return;
    FOR_ITEMS_IN_LIST(lp, v)
        outx_value(l, LIST_ITEM(lp));
}


/**
 * output FdoLoop
 */
static void
outx_doLoop(int l, expv v)
{
    const int l1 = l + 1;

    expv vl = EXPR_ARG1(v);
    expv vr = EXPR_ARG2(v);
    expv var   = EXPR_ARG1(vl);
    expv lower = EXPR_ARG2(vl);
    expv upper = EXPR_ARG3(vl);
    expv step  = EXPR_ARG4(vl);

    outx_tag(l, XTAG(v));
    outx_expv(l1, var);
    outx_indexRange(l1, lower, upper, step);
    outx_valueInList(l1, vr);

    outx_expvClose(l, v);
}


/**
 * output varList
 */
static void
outx_varList(int l, expv v, const char *name)
{
    const int l1 = l + 1;
    char buf[256];
    list lp;

    if(name)
        sprintf(buf, " name=\"%s\"", name);
    else
        buf[0] = '\0';

    outx_tag(l, "varList%s", buf);
    FOR_ITEMS_IN_LIST(lp, v) {
        expv x = LIST_ITEM(lp);
        if(EXPR_CODE(x) == F_IMPLIED_DO)
            outx_doLoop(l1, x);
        else
            outx_varRef_EXPR(l1, x);
    }
    outx_close(l, "varList");
}


/**
 * output namelistDecl
 */
static void
outx_namelistDecl(int l, ID nlId)
{
    const int l1 = l + 1;

    if(nlId == NULL)
        return;

    outx_tagOfDecl(l, "FnamelistDecl", nlId);
    outx_varList(l1, NL_LIST(nlId), SYM_NAME(ID_SYM(nlId)));
    outx_close(l, "FnamelistDecl");
}


/**
 * output body
 */
static void
outx_body(int l, expv v)
{
    outx_childrenWithTag(l, "body", v);
}


/**
 * output condition
 */
static void
outx_condition(int l, expv v)
{
    outx_childrenWithTag(l, "condition", v);
}


/**
 * output FdataDecl
 */
static void
outx_dataDecl(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfDecl1(l, XTAG(v), EXPR_LINE(v));
    outx_varList(l1, EXPR_ARG1(v), NULL);
    outx_valueList(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output varDecl
 */
static void
outx_varDecl(int l, ID id)
{
    assert(id);
    const int l1 = l + 1;

    if(ID_IS_ASSOCIATIVE(id))
        return;

    outx_tagOfDecl(l, "varDecl", id);

    outx_symbolNameWithType_ID(l1, id);
    outx_value(l1, VAR_INIT_VALUE(id));

    outx_close(l, "varDecl");
}


static void
outx_type_bound_procedure_call0(int l, expv v)
{
    const int l1 = l + 1, l2 = l1 + 1;
    list lp;
    expv arg, v2;

    assert (EXPV_CODE(EXPR_ARG1(v)) == F95_MEMBER_REF);

    outx_tagOfExpression1(l, v, 0);
    outx_expv(l1, EXPR_ARG1(v));

    assert(EXPR_HAS_ARG2(v)); /* make sure ARG2 exists */
    v2 = EXPR_ARG2(v);
    if(EXPR_LIST(v2)) {
        outx_tag(l1, "arguments");
        FOR_ITEMS_IN_LIST(lp, v2) {
            arg = LIST_ITEM(lp);
            if(EXPV_KWOPT_NAME(arg))
                outx_namedValue(l2, EXPV_KWOPT_NAME(arg), arg, NULL);
            else
                outx_expv(l2, arg);
        }
        outx_close(l1, "arguments");
    }
    outx_expvClose(l, v);
}


static void
outx_functionCall0(int l, expv v)
{
    const int l1 = l + 1, l2 = l1 + 1;
    list lp;
    expv arg, v2;
    int opt = 0;
    TYPE_DESC tp;
    int isIntrinsic;

    if (EXPV_CODE(EXPR_ARG1(v)) == F95_MEMBER_REF) {
        outx_type_bound_procedure_call0(l, v);
        return;
    }

    isIntrinsic = (SYM_TYPE(EXPV_NAME(EXPR_ARG1(v))) == S_INTR);

    if (isIntrinsic && (tp = EXPV_TYPE(EXPR_ARG1(v)))){
        isIntrinsic = !TYPE_IS_EXTERNAL(tp);
    }

    if (isIntrinsic) {
        opt |= TOPT_INTRINSIC;
    }
    outx_tagOfExpression1(l, v, opt);

    if (isIntrinsic)
        outx_expvName(l1, EXPR_ARG1(v));
    else {
        assert(EXPR_ARG1(v));
        outx_symbolNameWithFunctionType(l1, EXPR_ARG1(v));
    }

    assert(LIST_NEXT(EXPV_LIST(v))); /* make sure ARG2 exists */
    v2 = EXPR_ARG2(v);
    if(EXPR_LIST(v2)) {
        outx_tag(l1, "arguments");
        FOR_ITEMS_IN_LIST(lp, v2) {
            arg = LIST_ITEM(lp);
            if(EXPV_KWOPT_NAME(arg))
                outx_namedValue(l2, EXPV_KWOPT_NAME(arg), arg, NULL);
            else
                outx_expv(l2, arg);
        }
        outx_close(l1, "arguments");
    }
    outx_expvClose(l, v);
}


static void
outx_subroutineCall(int l, expv v)
{
    outx_tagOfStatement3(l, xtag(EXPR_STATEMENT), EXPR_LINE(v));
    outx_functionCall0(l + 1, v);
    outx_close(l, xtag(EXPR_STATEMENT));
}


static void
outx_functionCall(int l, expv v)
{
    if(IS_VOID(EXPV_TYPE(v)) || IS_VOID(TYPE_REF(EXPV_TYPE(v))))
        outx_subroutineCall(l, v);
    else
        outx_functionCall0(l, v);
}


/**
 * output exprStatement
 */
static void
outx_exprStatement(int l, expv v)
{
    outx_tagOfStatement(l, v);
    outx_expv(l + 1, EXPR_ARG1(v));
    outx_expvClose(l, v);
}


/**
 * output FdoStatement
 */
static void
outx_doStatement(int l, expv v)
{
    const int l1 = l + 1;
    expv vl, vr, v1, v2, v3, v4, v5;

    vl = EXPR_ARG1(v);
    vr = EXPR_ARG2(v);
    v1 = EXPR_ARG1(vr);
    v2 = EXPR_ARG2(vr);
    v3 = EXPR_ARG3(vr);
    v4 = EXPR_ARG4(vr);
    v5 = EXPR_ARG5(vr);

    outx_tagOfStatementWithConstructName(l, v, vl, 1);
    outx_expv(l1, v1);
    if(v2 || v3 || v4)
        outx_indexRange(l1, v2, v3, v4);
    outx_body(l1, v5);
    outx_expvClose(l, v);
}


/**
 * output scena range for FcaseLabel
 */
static void
outx_sceneRange(int l, expv v)
{
    if(EXPR_ARG1(v))
        outx_value(l, EXPR_ARG1(v));
    else
        outx_indexRange(l, EXPR_ARG2(v), EXPR_ARG3(v), NULL);
}

static void
outx_arrayConstructor(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfExpression(l, v);
    outx_expv(l1, EXPR_ARG1(v));
    outx_expvClose(l, v);
}

static void
outx_typedArrayConstructor(int l, expv v)
{
    TYPE_DESC element_tp = array_element_type(EXPV_TYPE(v));
    const int l1 = l + 1;
    EXPV_TYPE(v) = EXPV_TYPE(v);
    outx_typeAttrs(l, EXPV_TYPE(v), XTAG(v), TOPT_TYPEONLY);
    outx_print(" element_type=\"%s\">\n", getTypeID(element_tp));
    outx_expv(l1, EXPR_ARG1(v));
    outx_expvClose(l, v);
}

static void
outx_typeParamValues(int l, expv type_param_values)
{
  list lp;
  int l1 = l + 1;

  if (type_param_values == NULL) {
      return;
  }

  outx_printi(l, "<typeParamValues>\n");

  FOR_ITEMS_IN_LIST(lp, type_param_values){
      expv item = LIST_ITEM(lp);
      if(EXPV_KWOPT_NAME(item)) {
          outx_namedValue(l1, EXPV_KWOPT_NAME(item), item, NULL);
      } else {
          outx_expv(l1, item);
      }
  }
  outx_close(l, "typeParamValues");
}

static void
outx_structConstructorComponents(int l, expv components)
{
  list lp;

  if (components == NULL) {
      return;
  }

  FOR_ITEMS_IN_LIST(lp, components){
      expv item = LIST_ITEM(lp);
      if(EXPV_KWOPT_NAME(item)) {
          outx_namedValue(l, EXPV_KWOPT_NAME(item), item, NULL);
      } else {
          outx_expv(l, item);
      }
  }
}

static void
outx_structConstructor(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfExpression(l, v);
    if (EXPR_ARG1(v)) {
        outx_typeParamValues(l1, EXPR_ARG1(v));
    }
    outx_structConstructorComponents(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}

/**
 * output FcaseLabel
 */
static void
outx_caseLabel(int l, expv v)
{
    const int l1 = l + 1;

    outx_tagOfStatementWithConstructName(l, v, EXPR_ARG3(v), 1);
    outx_expv(l1, EXPR_ARG1(v));
    outx_body(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}

/**
 * output typeGuard
 */
static void
outx_typeGuard(int l, expv v, int is_class)
{
    const int l1 = l + 1;
    outx_vtagLineno(l, XTAG(v), EXPR_LINE(v), NULL);
    if(EXPR_ARG3(v) != NULL) { // construct name
        outx_print(" construct_name=\"%s\"", SYM_NAME(EXPV_NAME(EXPR_ARG3(v))));
    }
    if(is_class){ // CLASS IS and CLASS DEFAULT
        if(EXPR_ARG1(v) == NULL){ // 
            outx_print(" kind=\"CLASS_DEFAULT\">\n");
        } else {
            outx_print(" kind=\"CLASS_IS\" type=\"%s\">\n", getTypeID(EXPV_TYPE(EXPR_ARG1(v))));
        }
    } else { // TYPE IS
        outx_print(" kind=\"TYPE_IS\" type=\"%s\">\n", getTypeID(EXPV_TYPE(EXPR_ARG1(v))));
    }

    outx_body(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output FselectStatement
 */
static void
outx_selectStatement(int l, expv v)
{
    const int l1 = l + 1;
    list lp = EXPR_LIST(v);

    outx_tagOfStatementWithConstructName(l, v, EXPR_ARG3(v), 1);
    outx_value(l1, LIST_ITEM(lp));

    if(LIST_NEXT(lp) && LIST_ITEM(LIST_NEXT(lp))) {
        FOR_ITEMS_IN_LIST(lp, LIST_ITEM(LIST_NEXT(lp)))
            outx_expv(l1, LIST_ITEM(lp));
    }

    outx_expvClose(l, v);
}

/**
 * output selectTypeStatement for SELECT TYPE statement
 */
static void
outx_selectTypeStatement(int l, expv v)
{
    const int l1 = l + 1;
    list lp = EXPR_LIST(v);
    outx_tagOfStatementWithConstructName(l, v, EXPR_ARG3(v), 1);

    outx_printi(l1, "<id>\n"); 
    if(EXPR_ARG4(v) != NULL){
        outx_printi(l1+1, "<name>%s</name>\n", SYM_NAME(EXPR_SYM(EXPR_ARG4(v))));
    } else {
        outx_printi(l1+1, "<name></name>\n");
    }
    outx_value(l1+1, LIST_ITEM(lp));
    outx_printi(l1, "</id>\n");
    if(LIST_NEXT(lp) && LIST_ITEM(LIST_NEXT(lp))) {
        FOR_ITEMS_IN_LIST(lp, LIST_ITEM(LIST_NEXT(lp)))
            outx_expv(l1, LIST_ITEM(lp));
    }
    outx_expvClose(l, v);
}


/**
 * output FdoWhileStatement
 */
static void
outx_doWhileStatement(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfStatementWithConstructName(l, v, EXPR_ARG3(v), 1);
    outx_condition(l1, EXPR_ARG1(v));
    outx_body(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output FifStatement/FwhereStatement
 */
static void
outx_IFWHERE_Statement(int l, expv v)
{
    const int l1 = l + 1, l2 = l1 + 1;
    expv cv = expr_list_get_n(v, 3);

    outx_tagOfStatementWithConstructName(l, v, cv, 1);
    outx_condition(l1, EXPR_ARG1(v));
    outx_tag(l1, "then");
    outx_body(l2, EXPR_ARG2(v));
    outx_close(l1, "then");

    if(EXPR_ARG3(v)) {
      if (EXPR_ARG5(v)){
	outx_vtagLineno(l1, "else", EXPR_LINE(EXPR_ARG5(v)), NULL);
	outx_puts(">\n");
      }
      else {
	outx_tag(l1, "else");
      }
      outx_body(l2, EXPR_ARG3(v));
      outx_close(l1, "else");
    }

    outx_expvClose(l, v);
}


/**
 * output FreturnStatement
 */
static void
outx_returnStatement(int l, expv v)
{
    outx_tagOfStatementNoChild(l, v, NULL);
}

/**
 * output FimportStatement
 */
static void
outx_importStatement(int l, expv v) {
    const int l1 = l + 1;
    expv ident_list, arg;
    list lp;
    outx_tagOfStatement(l, v);
    ident_list = EXPR_ARG1(v);
    if(EXPR_LIST(ident_list)) {
        FOR_ITEMS_IN_LIST(lp, ident_list) {
            arg = LIST_ITEM(lp);
            if(EXPR_CODE(arg) == IDENT){
                outx_printi(l1, "<name>%s</name>\n", getRawString(arg));
            }            
        }
    }
    outx_expvClose(l, v);
}


/**
 * output tag with label_name attribute
 */
static void
outx_labeledStatement(int l, expv v)
{
    outx_tagOfStatementNoChild(l, v,
        " label_name=\"%s\"", SYM_NAME(EXPV_NAME(EXPR_ARG1(v))));
}


/**
 * output gotoStatement
 */
static void
outx_gotoStatement(int l, expv v)
{
    outx_labeledStatement(l, v);
}

/**
 * output gotoStatement
 */
static void
outx_compgotoStatement(int l, expv v)
{
    list lp;
    const int l1 = l + 1, l2 = l1 + 1;

    /* computed goto (params, value) */
    outx_tagOfStatement(l, v);
    outx_tag(l1, "params");
    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(v))
        outx_linenoNameWithFconstant(l + 2, LIST_ITEM(lp));
    outx_close(l1, "params");
    outx_tag(l1, "value");
    outx_expv(l2, EXPR_ARG2(v));
    outx_close(l1, "value");
    outx_expvClose(l, v);
}


/**
 * output FcontinueStatement
 */
static void
outx_continueStatement(int l, expv v)
{
    outx_tagOfStatementNoChild(l, v, NULL);
}


/**
 * output FstopStatement/FpauseStatement
 */
static void
outx_STOPPAUSE_statement(int l, expv v)
{
    char buf[CHAR_BUF_SIZE];
    expv x1 = EXPR_ARG1(v);
    buf[0] = '\0';

    if(x1) {
        switch(EXPV_CODE(x1)) {
        case INT_CONSTANT:
            sprintf(buf, " code=\""OMLL_DFMT"\"", EXPV_INT_VALUE(x1));
            break;
        case STRING_CONSTANT:
            sprintf(buf, " message=\"%s\"", getXmlEscapedStr(EXPV_STR(x1)));
            break;
        default:
            abort();
        }
    }

    outx_tagOfStatementNoChild(l, v, buf);
}


/**
 * output FexitStatement/FcycleStatement
 */
static void
outx_EXITCYCLE_statement(int l, expv v)
{
    outx_tagOfStatementWithConstructName(l, v, EXPR_ARG1(v), 0);
}


static const char*
getFormatID(expv v)
{
    static char buf[CHAR_BUF_SIZE];

    if(v && EXPV_CODE(v) == LIST)
        v = EXPR_ARG1(v);

    if(v == NULL)
        strcpy(buf, "*");
    else
        strcpy(buf, getRawString(v));

    return buf;
}


/**
 * output FformatDecl
 */
static void
outx_formatDecl(int l, expv v)
{
    const char *fmt = getXmlEscapedStr(EXPV_STR(EXPV_LEFT(v)));
    outx_tagOfDeclNoChild(l, "%s format=\"%s\"", EXPR_LINE(v), XTAG(v), fmt);
}


/**
 * output print statement
 */
static void
outx_printStatement(int l, expv v)
{
    /* Checkes expr code of arg1.
       arg1 must be
       1) string constant
       2) syntax number (int constant and larger than 0, smaller than 1000000.)
       3) *
       4) sclar integer variable.
     */
    expv format = EXPR_ARG1(v);
    if (format != NULL && EXPV_CODE(format) == LIST) {
        format = EXPR_ARG1(format);
    }

    if (format != NULL)
    switch(EXPV_CODE(format)) {
    case INT_CONSTANT:
    case STRING_CONSTANT:
    case IDENT:
    case F_VAR:
    case F_PARAM:
        break;
    case F_ARRAY_REF:
        error_at_node(v, "cannot use array in format specifier");
        exit(1);
    case F_SUBSTR_REF:
        error_at_node(v, "cannot use sub string in format specifier");
        exit(1);
    default:
        error_at_node(v, "invalid expression '%s' in a format specifier",
              EXPR_CODE_NAME(EXPR_CODE(format)));
        exit(1);
        break;
    }

    outx_tagOfStatement1(l, v,
        " format=\"%s\"", getFormatID(format));
    outx_valueList(l + 1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output read/write/inquire statement
 */
static void
outx_RW_statement(int l, expv v)
{
    const int l1 = l + 1;
    static const char *keys[] = { "unit", "fmt" };

    outx_tagOfStatement(l, v);
    outx_namedValueList(l1, keys, ARRAY_LEN(keys), EXPR_ARG1(v), "*");
    outx_valueList(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output open/close/rewind/endfile/backspace statement
 */
static void
outx_IO_statement(int l, expv v)
{
    static const char *keys[] = { "unit" };

    outx_tagOfStatement(l, v);
    outx_namedValueList(l + 1, keys, ARRAY_LEN(keys), EXPR_ARG1(v), "*");
    outx_expvClose(l, v);
}


/**
 * output flush statement
 */
static void
outx_FLUSH_statement(int l, expv v)
{
    static const char *keys[] = { "unit" };

    outx_tagOfStatement(l, v);
    outx_namedValueList(l + 1, keys, ARRAY_LEN(keys), EXPR_ARG1(v), "*");
    outx_expvClose(l, v);
}

/**
 * output FpointerAssignExpr
 */
static void
outx_pointerAssignStatement(int l, expv v)
{
    list lp;
    expv vPointer = NULL;
    expv vPointee = NULL;
    int nArgs = 0;

    FOR_ITEMS_IN_LIST(lp, v) {
        nArgs++;
    }
    if (nArgs != 2) {
        fatal("%s: Invalid arguments number.", __func__);
        return;
    }
    vPointer = EXPR_ARG1(v);
    vPointee = EXPR_ARG2(v);

    if (EXPV_CODE(vPointer) != F_VAR &&
        EXPV_CODE(vPointer) != ARRAY_REF &&
        EXPV_CODE(vPointer) != F95_MEMBER_REF) {
        fatal("%s: Invalid argument, expected F_VAR or F_ARRAY_REF or F95_MEMBER_REF.", __func__);
    }
    if (EXPV_CODE(vPointee) != F_VAR &&
        EXPV_CODE(vPointee) != ARRAY_REF &&
        EXPV_CODE(vPointee) != F95_MEMBER_REF &&
        EXPV_CODE(vPointee) != FUNCTION_CALL) {
        fatal("%s: Invalid argument, "
              "expected F_VAR or ARRAY_REF or F95_MEMBER_REF of FUNCTION_CALL.", __func__);
    }

    outx_tagOfStatement(l, v);
    outx_expv(l + 1, vPointer);
    outx_expv(l + 1, vPointee);
    outx_expvClose(l, v);
}


/**
 * output FentryDecl
 */
static void
outx_entryDecl(int l, expv v) {
    expv symV = EXPR_ARG1(v);
    EXT_ID ep = EXPV_ENTRY_EXT_ID(symV);

    if (ep == NULL) {
        fatal("%s: can't get EXT_ID for entry '%s'.",
              __func__, SYM_NAME(EXPV_NAME(symV)));
        return;
    }

    outx_tagOfStatement(l, v);
    outx_symbolNameWithFunctionType_EXT(l + 1, ep);
    outx_expvClose(l, v);
}


/**
 * output FcharacterRef
 */
static void
outx_characterRef(int l, expv v)
{
    const int l1 = l + 1;
    expv vrange;
    outx_tagOfExpression(l, v);
    outx_varRef_EXPR(l1, EXPR_ARG1(v));
    vrange = EXPR_ARG2(v);
    outx_indexRange(l1, EXPR_ARG1(vrange), EXPR_ARG2(vrange), NULL);
    outx_expvClose(l, v);
}


/**
 * output FmemberRef
 */
static void
outx_memberRef(int l, expv v)
{
    expv v_left = EXPV_LEFT(v);
    expv v_right = EXPV_RIGHT(v);

    outx_typeAttrOnly_EXPR(l, v, XTAG(v));
    outx_print(" member=\"%s\">\n", SYM_NAME(EXPV_NAME(v_right)));
    outx_varRef_EXPR(l + 1, v_left);
    outx_expvClose(l, v);
}


/**
 * output indexRange or arrayIndex for arrayRef
 */
static void
outx_arraySpec(int l, expv v)
{
    expv iv;
    list lp;

    FOR_ITEMS_IN_LIST(lp, v) {
        iv = LIST_ITEM(lp);
	if(iv == NULL) continue;
        switch(EXPV_CODE(iv)) {
        case F_INDEX_RANGE:
            outx_indexRangeInList(l, iv);
            break;
        default: /* int or int array expression */
            outx_childrenWithTag(l, "arrayIndex", iv);
            break;
        }
    }
}


/**
 * output FarrayRef
 */
static void
outx_arrayRef(int l, expv v)
{
    int l1 = l + 1;

    outx_tagOfExpression(l, v);
    outx_varRef_EXPR(l1, EXPR_ARG1(v));
    outx_arraySpec(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output coarrayRef
 */
static void
outx_coarrayRef(int l, expv v)
{
    int l1 = l + 1;

    outx_tagOfExpression(l, v);
    outx_varRef_EXPR(l1, EXPR_ARG1(v));
    outx_arraySpec(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output alloc
 */
static void
outx_alloc(int l, expv v)
{
    const int l1 = l + 1;
    list lp;
    expr v1;

    outx_tag(l, "alloc");

    switch(EXPV_CODE(v)) {
        case F_VAR:
        case F95_MEMBER_REF:
            outx_expv(l1, v);
            break;

        case ARRAY_REF:
            outx_expv(l1, EXPR_ARG1(v));
            outx_arraySpec(l1, EXPR_ARG2(v));
            break;

        case XMP_COARRAY_REF: {

            v1 = EXPR_ARG1(v);

            switch (EXPR_CODE(v1)){

                case F_VAR:
                case F95_MEMBER_REF:
                    outx_expv(l1, v1);
                    break;
                case ARRAY_REF:
                    outx_expv(l1, EXPR_ARG1(v1));
                    outx_arraySpec(l1, EXPR_ARG2(v1));
                    break;
                default:
                    abort();
            }

            outx_printi(l1, "<coShape>\n");
            FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)) {

                expr cobound = LIST_ITEM(lp);
                /* expr upper = EXPR_ARG2(cobound); */
                /* ARRAY_ASSUME_KIND defaultAssumeKind = ASSUMED_SHAPE; */
                /* if (upper && EXPR_CODE(upper) == F_ASTERISK){ */
                /*   upper = NULL; */
                /*   defaultAssumeKind = ASSUMED_SIZE; */
                /* } */

                outx_indexRange0(l+2, ASSUMED_NONE, ASSUMED_SIZE,
                                 EXPR_ARG1(cobound), EXPR_ARG2(cobound), EXPR_ARG3(cobound));

            }
            outx_close(l1, "coShape");
        } break;

        default:
            abort();
    }

    outx_close(l, "alloc");
}


/**
 * output list of alloc
 */
static void
outx_allocList(int l, expv v)
{
    list lp;
    FOR_ITEMS_IN_LIST(lp, v) {
        outx_alloc(l, LIST_ITEM(lp));
    }
}


/**
 * output FnullifyStatement
 */
static void
outx_nullifyStatement(int l, expv v)
{
    const int l1 = l + 1;

    outx_tagOfStatement(l, v);
    outx_allocList(l1, EXPR_ARG1(v));
    outx_expvClose(l, v);
}


#if 0
static void
print_allocate_keyword(char * buf, expv v, const char * keyword)
{
    if (v) {
        switch (EXPR_CODE(v)){
            case F_VAR:
                sprintf(buf, " %s=\"%s\"", keyword, SYM_NAME(EXPV_NAME(v)));
                break;
            case ARRAY_REF:
                warning("cannot use array ref. in allocate/deallocate keyword specifier");
                buf[0] = '\0';
                break;
            case F95_MEMBER_REF:
                warning("cannot use member ref. in allocate/deallocate keyword specifier");
                buf[0] = '\0';
                break;
            default:
                buf[0] = '\0';
                break;
        }
    }
}
#endif


/**
 * output FallocateStatement/FdeallocateStatement
 */
static void
outx_ALLOCDEALLOC_statement(int l, expv v)
{
    const int l1 = l + 1;
    const int l2 = l + 2;

    expv keywords = EXPR_ARG2(v);

    expv vstat = expr_list_get_n(keywords, 0);
    expv vmold = expr_list_get_n(keywords, 1);
    expv vsource = expr_list_get_n(keywords, 2);
    expv verrmsg = expr_list_get_n(keywords, 3);
    char type_buf[128];
    char stat_buf[128];

    const char *tid = NULL;

    if (EXPV_TYPE(v)) {
        tid = getTypeID(EXPV_TYPE(v));
        sprintf(type_buf, " type=\"%s\"", tid);
    } else {
        type_buf[0] = '\0';
    }

#if 0
    /* Deprecated */
    print_allocate_keyword(stat_buf, vstat, "stat_name");
#endif

    outx_tagOfStatement1(l, v, type_buf, stat_buf);
    outx_allocList(l1, EXPR_ARG1(v));
    if (vstat) {
        outx_printi(l1,"<allocOpt kind=\"stat\">\n");
        outx_expv(l2, vstat);
        outx_close(l1,"allocOpt");
    }
    if (vsource) {
        outx_printi(l1,"<allocOpt kind=\"source\">\n");
        outx_expv(l2, vsource);
        outx_close(l1,"allocOpt");
    }
    if (vmold) {
        outx_printi(l1,"<allocOpt kind=\"mold\">\n");
        outx_expv(l2, vmold);
        outx_close(l1,"allocOpt");
    }
    if (verrmsg) {
        outx_printi(l1,"<allocOpt kind=\"errmsg\">\n");
        outx_expv(l2, verrmsg);
        outx_close(l1,"allocOpt");
    }
    outx_expvClose(l, v);
}


static const char*
getKindParameter(TYPE_DESC tp)
{
    static char buf[256];
    expv v = TYPE_KIND(tp);

    if(IS_DOUBLED_TYPE(tp)) {
        sprintf(buf, "%d", KIND_PARAM_DOUBLE);
    } else if(v && (EXPV_CODE(v) == INT_CONSTANT || EXPV_CODE(v) == IDENT)) {
        strcpy(buf, getRawString(v));
    } else {
        return NULL;
    }

    return buf;
}



/**
 * output FintConstant/FcharacterConstant/FlogicalConstant/FcomplexConstant
 */
static void
outx_constants(int l, expv v)
{
    static char buf[CHAR_BUF_SIZE];
    const int l1 = l + 1;
    const char *tid;
    const char *tag = XTAG(v);
    TYPE_DESC tp = EXPV_TYPE(v);
    const char *kind;

    switch(EXPV_CODE(v)) {
    case INT_CONSTANT:
        if(IS_LOGICAL(EXPV_TYPE(v))) {
            sprintf(buf, ".%s.", EXPV_INT_VALUE(v) ? "TRUE" : "FALSE");
            tag = "FlogicalConstant";
        } else {
            omllint_t n = EXPV_INT_VALUE(v);
            sprintf(buf, OMLL_DFMT, n);
        }
        if(tp == NULL)
            tp = type_INT;
        //tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
        tid = getTypeID(tp);
        goto print_constant;

    case FLOAT_CONSTANT:
        if(EXPV_ORIGINAL_TOKEN(v))
            strcpy(buf, EXPV_ORIGINAL_TOKEN(v));
        else
            sprintf(buf, "%Lf", EXPV_FLOAT_VALUE(v));
        if(tp == NULL)
            tp = type_REAL;
        //tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
        tid = getTypeID(tp);
        goto print_constant;

    case STRING_CONSTANT:
        strcpy(buf, getXmlEscapedStr(EXPV_STR(v)));
        assert(tp);
        tid = getTypeID(tp);
        goto print_constant;

    print_constant:
        outx_printi(l, "<%s type=\"%s\"", tag, tid);
        if(EXPV_CODE(v) != STRING_CONSTANT &&
            (kind = getKindParameter(tp)) != NULL)
            outx_print(" kind=\"%s\"", kind);
        outx_print(">%s</%s>\n", buf, tag);
        break;

    case COMPLEX_CONSTANT:
        assert(tp);
        outx_tagOfExpression(l, v);
        outx_expv(l1, EXPR_ARG1(v));
        outx_expv(l1, EXPR_ARG2(v));
        outx_expvClose(l, v);
        break;

    default:
        abort();
    }
}


/**
 * output pragma statement
 */
static void
outx_pragmaStatement(int l, expv v)
{
    list lp = EXPV_LIST(v);
    outx_tagOfStatement2(l, v);
    outx_puts(getXmlEscapedStr(EXPV_STR(LIST_ITEM(lp))));
    outx_expvClose(0, v);
}

/* outx_expv with <list> ...</list> */
static void
outx_expv_withListTag(int l,expv v)
{
  list lp;
  if(v == NULL) {
      outx_printi(l,"<list/>\n");
      return;
  }
  if(EXPV_CODE(v) == LIST){
    outx_tag(l, "list");
    FOR_ITEMS_IN_LIST(lp, v)
      outx_expv_withListTag(l+1, LIST_ITEM(lp));
    outx_close(l, "list");
  } else
    outx_expv(l,v);
}

/**
 * output OMP pragma statement
 */
static void outx_OMP_dir_string(int l,expv v);
static void outx_OMP_dir_clause_list(int l,expv v);

static void
outx_OMP_pragma(int l, expv v)
{
    const int l1 = l + 1;

    outx_tagOfStatement(l, v);
    outx_OMP_dir_string(l1,EXPR_ARG1(v));
    // outx_expv_withListTag(l1, EXPR_ARG1(v));

    if (EXPV_INT_VALUE(EXPR_ARG1(v)) != OMP_THREADPRIVATE &&
	EXPV_INT_VALUE(EXPR_ARG1(v)) != OMP_FLUSH &&
	EXPV_INT_VALUE(EXPR_ARG1(v)) != OMP_CRITICAL) {
        outx_OMP_dir_clause_list(l1,EXPR_ARG2(v));
        // outx_expv_withListTag(l1, EXPR_ARG2(v));

        /* output body */
        outx_expv_withListTag(l1, EXPR_ARG3(v));
    } else {
        list lp;
        expv lV;

        outx_tag(l1, "list");
        FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)) {
            lV = LIST_ITEM(lp);
            outx_varOrFunc(l1 + 1, lV);
        }
        outx_close(l1, "list");
	if(EXPV_INT_VALUE(EXPR_ARG1(v)) == OMP_CRITICAL)
	   outx_expv_withListTag(l1, EXPR_ARG3(v));
    }

    outx_expvClose(l, v);
}

static void
outx_OMP_dir_string(int l,expv v)
{
  char *s;
  s = NULL;
  if(EXPV_CODE(v) != INT_CONSTANT)
    fatal("outx_OMP_dir_string: not INT_CONSTANT");
  switch(EXPV_INT_VALUE(v)){
  case OMP_PARALLEL: s = "PARALLEL"; break;
  case OMP_FOR: s = "FOR"; break;
  case OMP_SECTIONS:s = "SECTIONS"; break;
  case OMP_SECTION: s = "SECTION"; break;
  case OMP_SINGLE: s = "SINGLE"; break;
  case OMP_MASTER: s = "MASTER"; break;
  case OMP_CRITICAL: s = "CRITICAL"; break;
  case OMP_BARRIER: s = "BARRIER"; break;
  case OMP_ATOMIC: s = "ATOMIC"; break;
  case OMP_FLUSH: s = "FLUSH"; break;
  case OMP_ORDERED: s = "ORDERED"; break;
  case OMP_THREADPRIVATE: s = "THREADPRIVATE"; break;
  case OMP_TASK: s = "TASK";break;
  case OMP_SIMD: s = "SIMD";break;
  case OMP_DECLARE: s = "DECLARE";break;
  default:
    fatal("out_OMP_dir_string: unknown value=%d\n",EXPV_INT_VALUE(v));
  }
  outx_printi(l, "<string>%s</string>\n", s);
}

static void outx_OMP_DATA_DEFAULT_kind(int l,char *s ,expv v)
{
    outx_printi(l+2, "<string>%s</string>\n", s);
    //printf("EXPV_INT_VALUE(%d)\n",EXPV_INT_VALUE(v));
    //    outx_printi(l+3,"<list>\n");

    switch(EXPV_INT_VALUE(v))
       {
       case OMP_DEFAULT_NONE:
       outx_printi(l+4,"<string>DEFAULT_NONE</string>\n");
       break;
       case OMP_DEFAULT_SHARED:
       outx_printi(l+4,"<string>DEFAULT_SHARED</string>\n");
       break;
       case OMP_DEFAULT_PRIVATE:
       outx_printi(l+4,"<string>DEFAULT_PRIVATE</string>\n");
       break;
       }//outx_expv_withListTag(l+2, EXPR_ARG2(vv));
//    outx_printi(l+3,"</list>\n");
    outx_printi(l+1,"</list>\n");

}

static void outx_OMP_sched_kind(int l,char *s,expv v)
{
  expr vv = EXPR_ARG2(v);
  outx_printi(l+2, "<string>%s</string>\n", s);
  //printf("EXPV_INT_VALUE(%d)\n",EXPV_INT_VALUE(EXPR_ARG1(EXPR_ARG2(v))));
//  printf("vv=%d\n",EXPV_INT_VALUE(expr_list_get_n(vv,0)));
  outx_printi(l+3,"<list>\n");

  switch(EXPV_INT_VALUE(EXPR_ARG1(EXPR_ARG2(v))))
    {
    case OMP_SCHED_NONE:
      outx_printi(l+4,"<string>SCHED_NONE</string>\n");
      break;
    case OMP_SCHED_STATIC:
      outx_printi(l+4,"<string>SCHED_STATIC</string>\n");
      break;
     case  OMP_SCHED_DYNAMIC:
       outx_printi(l+4,"<string>SCHED_DYNAMIC</string>\n");
       break;
    case  OMP_SCHED_GUIDED:
       outx_printi(l+4,"<string>SCHED_GUIDED</string>\n");
       break;
     case  OMP_SCHED_RUNTIME:
       outx_printi(l+4,"<string>SCHED_RUNTIME</string>\n");
       break;
     case  OMP_SCHED_AFFINITY:
       outx_printi(l+4,"<string>SCHED_AFFINITY</string>\n");
       break;
    default:
      fatal("OMP Sched error");
    }
	//printf("ARG2=%d\n",EXPV_INT_VALUE(expr_list_get_n(vv,1)));
	if(expr_list_get_n(vv,1)!=NULL)
 	outx_expv(l+4,expr_list_get_n(vv,1));
    outx_printi(l+3,"</list>\n");
  outx_printi(l+1,"</list>\n");
}


static void
outx_OMP_dir_clause_list(int l,expv v)
{
  struct list_node *lp;
  const int l1 = l + 1;
  expv vv,dir;
  char *s = NULL;

  if(EXPV_CODE(v) != LIST)
    fatal("outx_OMP_dir_clause_list: not LIST");
  outx_printi(l,"<list>\n");

  FOR_ITEMS_IN_LIST(lp, v) {
    vv = LIST_ITEM(lp);

    if(EXPV_CODE(vv) != LIST)
      fatal("outx_OMP_dir_clause_list: not LIST2");

    outx_printi(l1,"<list>\n");

    dir = EXPR_ARG1(vv);
    if(EXPV_CODE(dir) != INT_CONSTANT)
      fatal("outx_OMP_dir_clause_list: clause not INT_CONSTANT");
    switch(EXPV_INT_VALUE(dir)){
    case OMP_DATA_DEFAULT:          s = "DATA_DEFAULT"; outx_OMP_DATA_DEFAULT_kind(l,s,EXPR_ARG2(vv)); continue;
    case OMP_DATA_PRIVATE:          s = "DATA_PRIVATE"; break;
    case OMP_DATA_SHARED:           s = "DATA_SHARED"; break;
    case OMP_DATA_FIRSTPRIVATE:     s = "DATA_FIRSTPRIVATE"; break;
    case OMP_DATA_LASTPRIVATE:      s = "DATA_LASTPRIVATE"; break;
    case OMP_DATA_COPYIN:           s = "DATA_COPYIN"; break;
    case OMP_DATA_REDUCTION_PLUS:   s = "DATA_REDUCTION_PLUS"; break;
    case OMP_DATA_REDUCTION_MINUS:  s = "DATA_REDUCTION_MINUS"; break;
    case OMP_DATA_REDUCTION_MUL:    s = "DATA_REDUCTION_MUL"; break;
    case OMP_DATA_REDUCTION_BITAND: s = "DATA_REDUCTION_BITAND"; break;
    case OMP_DATA_REDUCTION_BITOR:  s = "DATA_REDUCTION_BITOR"; break;
    case OMP_DATA_REDUCTION_BITXOR: s = "DATA_REDUCITON_BITXOR"; break;
    case OMP_DATA_REDUCTION_LOGAND: s = "DATA_REDUCITON_LOGAND"; break;
    case OMP_DATA_REDUCTION_LOGOR:  s = "DATA_REDUCTION_LOGOR"; break;
    case OMP_DATA_REDUCTION_MIN:    s = "DATA_REDUCTION_MIN"; break;
    case OMP_DATA_REDUCTION_MAX:    s = "DATA_REDUCTION_MAX"; break;
    case OMP_DATA_REDUCTION_EQV:    s = "DATA_REDUCTION_EQV"; break;
    case OMP_DATA_REDUCTION_NEQV:   s = "DATA_REDUCTION_NEQV"; break;
    case OMP_DATA_COPYPRIVATE:      s = "DATA_COPYPRIVATE"; break;
    case OMP_DIR_ORDERED:           s = "DIR_ORDERED"; break;
    case OMP_DIR_IF:                s = "DIR_IF"; break;
    case OMP_DIR_NUM_THREADS:       s = "DIR_NUM_THREADS"; break;
    case OMP_DIR_COLLAPSE:          s = "COLLAPSE"; break;
    case OMP_DIR_NOWAIT:            s = "DIR_NOWAIT"; break;
    case OMP_DIR_SCHEDULE:          s = "DIR_SCHEDULE";  outx_OMP_sched_kind(l,s,vv);continue;
    case OMP_DATA_DEPEND_IN:        s = "DATA_DEPEND_IN";break;
    case OMP_DATA_DEPEND_OUT:       s = "DATA_DEPEND_OUT";break;
    case OMP_DATA_DEPEND_INOUT:     s = "DATA_DEPEND_INOUT";break;
    case OMP_DIR_UNTIED:            s = "DIR_UNTIED";break;
    case OMP_DIR_MERGEABLE:         s = "DIR_MERGEABLE";break;
    case OMP_DATA_FINAL:            s = "DATA_FINAL";break;
    default:
      fatal("out_OMP_dir_clause: unknown value=%d\n",EXPV_INT_VALUE(v));
    }
    outx_printi(l+2, "<string>%s</string>\n", s);
    outx_expv_withListTag(l+2, EXPR_ARG2(vv));
    outx_printi(l1,"</list>\n");
  }
  outx_printi(l,"</list>\n");
}


/**
 * output XMP pragma statement
 */
static void outx_XMP_dir_string(int l,expv v);
static void outx_XMP_dir_clause_list(int l,expv v);

static void
outx_XMP_pragma(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfStatement(l, v);
    outx_XMP_dir_string(l1,EXPR_ARG1(v));
    if (EXPR_ARG2(v)) outx_XMP_dir_clause_list(l1,EXPR_ARG2(v));

    /* output body */
    if(EXPR_HAS_ARG3(v))
	outx_expv_withListTag(l1, EXPR_ARG3(v));
    outx_expvClose(l, v);
}

static void
outx_XMP_dir_string(int l,expv v)
{
  char *s = NULL;

  if(EXPV_CODE(v) != INT_CONSTANT)
    fatal("outx_XMP_dir_string: not INT_CONSTANT");
  switch(EXPV_INT_VALUE(v)){
  case XMP_NODES: s = "NODES"; break;
  case XMP_TEMPLATE: s = "TEMPLATE"; break;
  case XMP_DISTRIBUTE: s = "DISTRIBUTE"; break;
  case XMP_ALIGN: s = "ALIGN"; break;
  case XMP_SHADOW: s = "SHADOW"; break;
  case XMP_LOCAL_ALIAS: s = "LOCAL_ALIAS"; break;
  case XMP_SAVE_DESC: s = "SAVE_DESC"; break;
  case XMP_TASK: s = "TASK"; break;
  case XMP_TASKS: s = "TASKS"; break;
  case XMP_LOOP: s = "LOOP"; break;
  case XMP_REFLECT: s = "REFLECT"; break;
  case XMP_GMOVE: s = "GMOVE"; break;
  case XMP_ARRAY: s = "ARRAY"; break;
  case XMP_BARRIER: s = "BARRIER"; break;
  case XMP_REDUCTION: s = "REDUCTION"; break;
  case XMP_BCAST: s = "BCAST"; break;
  case XMP_WAIT_ASYNC: s = "WAIT_ASYNC"; break;
  case XMP_COARRAY: s = "COARRAY"; break;
  case XMP_IMAGE: s = "IMAGE"; break;
  case XMP_TEMPLATE_FIX: s = "TEMPLATE_FIX"; break;
  case XMP_MASTER_IO: s = "XMP_MASTER_IO"; break;
  case XMP_GLOBAL_IO: s = "XMP_GLOBAL_IO"; break;

  default:
    fatal("out_XMP_dir_string: unknown value=%d\n",EXPV_INT_VALUE(v));
  }
  outx_printi(l, "<string>%s</string>\n", s);
}

void
outx_XMP_dir_clause_list(int l,expv v)
{
  struct list_node *lp;
  const int l1 = l + 1;
  expv vv;

  if(EXPV_CODE(v) != LIST)
    fatal("outx_XMP_dir_clause_list: not LIST");
  outx_printi(l,"<list>\n");
  FOR_ITEMS_IN_LIST(lp, v) {
      vv = LIST_ITEM(lp);
      if(vv == NULL)
	  outx_printi(l1,"<list/>\n");
      else if(EXPR_CODE(vv) == LIST)
	  outx_XMP_dir_clause_list(l1,vv);
      else
	  outx_expv(l1,vv);
  }
  outx_printi(l,"</list>\n");
}


/**
 * output ACC pragma statement
 */
static void outx_ACC_dir_string(int l,expv v);
static void outx_ACC_dir_clause_list(int l,expv v);
static void outx_ACC_clause_string(int l, expv v);
static void outx_ACC_clause(int l, expv v);
#define outx_listClose(l) outx_close((l), "list")
#define outx_tagOfList(l) outx_printi((l),"<list>\n")
static void
outx_ACC_pragma(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfStatement(l, v);
    outx_ACC_dir_string(l1,EXPR_ARG1(v));

    if (EXPR_ARG2(v)) {
      outx_ACC_dir_clause_list(l1,EXPR_ARG2(v));
    }else{
      outx_printi(l1,"<list></list>\n");
    }

    /* output body */
    if(EXPR_HAS_ARG3(v)){
      outx_expv_withListTag(l1, EXPR_ARG3(v));
    }
    outx_expvClose(l, v);
}

static void
outx_ACC_dir_string(int l,expv v)
{
  char *s = NULL;

  if(EXPV_CODE(v) != INT_CONSTANT) fatal("outx_ACC_dir_string: not INT_CONSTANT");

  switch(EXPV_INT_VALUE(v)){
  case ACC_PARALLEL:		s = "PARALLEL"; break;
  case ACC_KERNELS:		s = "KERNELS"; break;
  case ACC_DATA:		s = "DATA"; break;
  case ACC_LOOP:		s = "LOOP"; break;
  case ACC_PARALLEL_LOOP:	s = "PARALLEL_LOOP"; break;
  case ACC_KERNELS_LOOP:	s = "KERNELS_LOOP"; break;
  case ACC_ATOMIC:		s = "ATOMIC"; break;
  case ACC_WAIT:		s = "WAIT"; break;
  case ACC_CACHE:		s = "CACHE"; break;
  case ACC_ROUTINE:		s = "ROUTINE"; break;
  case ACC_ENTER_DATA:		s = "ENTER_DATA"; break;
  case ACC_EXIT_DATA:		s = "EXIT_DATA"; break;
  case ACC_HOST_DATA:		s = "HOST_DATA"; break;
  case ACC_DECLARE:		s = "DECLARE"; break;
  case ACC_UPDATE_D:		s = "UPDATE"; break;
  case ACC_INIT:		s = "INIT"; break;
  case ACC_SHUTDOWN:		s = "SHUTDOWN"; break;
  case ACC_SET:			s = "SET"; break;

  case ACC_END_PARALLEL:
  case ACC_END_KERNELS:
  case ACC_END_DATA:
  case ACC_END_ATOMIC:
  case ACC_END_HOST_DATA:
  case ACC_END_PARALLEL_LOOP:
  case ACC_END_KERNELS_LOOP:
    fatal("out_ACC_dir_string: illegal end directive=%d\n", EXPV_INT_VALUE(v));
    break;
  default:
    fatal("out_ACC_dir_string: unknown value=%d\n",EXPV_INT_VALUE(v));
  }
  outx_tagText(l, "string", s);
}

static void
outx_ACC_clause_string(int l, expv v)
{
  char *s = NULL;

  if(EXPV_CODE(v) != INT_CONSTANT) fatal("outx_ACC_dir_string: not INT_CONSTANT");

  switch(EXPV_INT_VALUE(v)){
  case ACC_CLAUSE_IF:			s = "IF"; break;
  case ACC_CLAUSE_WAIT:			s = "WAIT"; break;
  case ACC_CLAUSE_ASYNC:		s = "ASYNC"; break;
  case ACC_CLAUSE_DEVICE_TYPE:		s = "DEVICE_TYPE"; break;
  case ACC_CLAUSE_WAIT_ARG:		s = "WAIT"; break;
  case ACC_CLAUSE_CACHE_ARG:		s = "CACHE"; break;
  case ACC_CLAUSE_ROUTINE_ARG:		s = "ROUTINE"; break;

    //data clause
  case ACC_CLAUSE_COPY:			s = "COPY"; break;
  case ACC_CLAUSE_COPYIN:		s = "COPYIN"; break;
  case ACC_CLAUSE_COPYOUT:		s = "COPYOUT"; break;
  case ACC_CLAUSE_CREATE:		s = "CREATE"; break;
  case ACC_CLAUSE_PRESENT:		s = "PRESENT"; break;
  case ACC_CLAUSE_PRESENT_OR_COPY:	s = "PRESENT_OR_COPY"; break;
  case ACC_CLAUSE_PRESENT_OR_COPYIN:	s = "PRESENT_OR_COPYIN"; break;
  case ACC_CLAUSE_PRESENT_OR_COPYOUT:	s = "PRESENT_OR_COPYOUT"; break;
  case ACC_CLAUSE_PRESENT_OR_CREATE:	s = "PRESENT_OR_CREATE"; break;
  case ACC_CLAUSE_DEVICEPTR:		s = "DEVICEPTR"; break;

  case ACC_CLAUSE_NUM_GANGS:		s = "NUM_GANGS"; break;
  case ACC_CLAUSE_NUM_WORKERS:		s = "NUM_WORKERS"; break;
  case ACC_CLAUSE_VECTOR_LENGTH:	s = "VECTOR_LENGTH"; break;

  case ACC_CLAUSE_REDUCTION_PLUS:	s = "REDUCTION_PLUS"; break;
  case ACC_CLAUSE_REDUCTION_MUL:	s = "REDUCTION_MUL"; break;
  case ACC_CLAUSE_REDUCTION_MAX:	s = "REDUCTION_MAX"; break;
  case ACC_CLAUSE_REDUCTION_MIN:	s = "REDUCTION_MIN"; break;
  case ACC_CLAUSE_REDUCTION_BITAND:	s = "REDUCTION_BITAND"; break;
  case ACC_CLAUSE_REDUCTION_BITOR:	s = "REDUCTION_BITOR"; break;
  case ACC_CLAUSE_REDUCTION_BITXOR:	s = "REDUCTION_BITXOR"; break;
  case ACC_CLAUSE_REDUCTION_LOGAND:	s = "REDUCTION_LOGAND"; break;
  case ACC_CLAUSE_REDUCTION_LOGOR:	s = "REDUCTION_LOGOR"; break;
  case ACC_CLAUSE_REDUCTION_EQV:	s = "REDUCTION_EQV"; break;
  case ACC_CLAUSE_REDUCTION_NEQV:	s = "REDUCTION_NEQV"; break;

  case ACC_CLAUSE_PRIVATE:		s = "PRIVATE"; break;
  case ACC_CLAUSE_FIRSTPRIVATE:		s = "FIRSTPRIVATE"; break;
  case ACC_CLAUSE_DEFAULT:		s = "DEFAULT"; break;
  case ACC_CLAUSE_NONE:			s = "NONE"; break;

  case ACC_CLAUSE_COLLAPSE:		s = "COLLAPSE"; break;
  case ACC_CLAUSE_GANG:			s = "GANG"; break;
  case ACC_CLAUSE_WORKER:		s = "WORKER"; break;
  case ACC_CLAUSE_VECTOR:		s = "VECTOR"; break;
  case ACC_CLAUSE_SEQ:			s = "SEQ"; break;
  case ACC_CLAUSE_AUTO:			s = "AUTO"; break;
  case ACC_CLAUSE_TILE:			s = "TILE"; break;
  case ACC_CLAUSE_INDEPENDENT:		s = "INDEPENDENT"; break;

  case ACC_CLAUSE_BIND:			s = "BIND"; break;
  case ACC_CLAUSE_NOHOST:		s = "NOHOST"; break;

  case ACC_CLAUSE_STATIC:		s = "STATIC"; break;
  case ACC_CLAUSE_READ:			s = "READ"; break;
  case ACC_CLAUSE_WRITE:		s = "WRITE"; break;
  case ACC_CLAUSE_UPDATE:		s = "UPDATE"; break;
  case ACC_CLAUSE_CAPTURE:		s = "CAPTURE"; break;
  case ACC_CLAUSE_DELETE:		s = "DELETE"; break;
  case ACC_CLAUSE_FINALIZE:		s = "FINALIZE"; break;

  case ACC_CLAUSE_USE_DEVICE:		s = "USE_DEVICE"; break;

  case ACC_CLAUSE_DEVICE_RESIDENT:	s = "DEVICE_RESIDENT"; break;
  case ACC_CLAUSE_LINK:			s = "LINK"; break;

  case ACC_CLAUSE_HOST:			s = "HOST"; break;
  case ACC_CLAUSE_DEVICE:		s = "DEVICE"; break;
  case ACC_CLAUSE_IF_PRESENT:		s = "IF_PRESENT"; break;

  case ACC_CLAUSE_DEVICE_NUM:		s = "DEVICE_NUM"; break;
  case ACC_CLAUSE_DEFAULT_ASYNC:	s = "DEFAULT_ASYNC"; break;


    //others 
  case ACC_ASTERISK:			s = "ASTERISK"; break;
  case ACC_COMMONBLOCK:			s = "COMMONBLOCK"; break;

  default:
    fatal("out_ACC_clause_string: unknown value=%d\n",EXPV_INT_VALUE(v));
  }
  outx_tagText(l, "string", s);
}

static void
outx_ACC_clause_arg(int l, expr v)
{
  if(v == NULL){
    outx_printi(l, "<list/>\n");
  }else if(EXPV_CODE(v) == ACC_PRAGMA){
    outx_ACC_clause(l, v);
  }else if(EXPV_CODE(v) == LIST){
    //F_ARRAY_REF
    outx_expv(l, v);
  }else{
    outx_expv(l, v);
  }
}

static void
outx_ACC_clause_arg_list(int l, expr v)
{
  struct list_node *lp;
  const int l1 = l + 1;
  expv vv;

  if(EXPV_CODE(v) != LIST) fatal("outx_ACC_clause_arg_list: not LIST");

  outx_tagOfList(l);
  FOR_ITEMS_IN_LIST(lp, v) {
    vv = LIST_ITEM(lp);
    outx_ACC_clause_arg(l1, vv);
  }
  outx_listClose(l);
}

static void
outx_ACC_clause(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfList(l);
    outx_ACC_clause_string(l1,EXPR_ARG1(v));

    if(EXPR_HAS_ARG2(v)){
      expv arg = EXPR_ARG2(v);
      if(EXPV_CODE(arg) != LIST){
	outx_ACC_clause_arg(l1, arg);
      }else{
	outx_ACC_clause_arg_list(l1, arg);
      }
    }
    outx_listClose(l);
}

static void
outx_ACC_dir_clause_list(int l,expv v)
{
  struct list_node *lp;
  const int l1 = l + 1;
  expv vv;

  if(EXPV_CODE(v) != LIST) fatal("outx_ACC_dir_clause_list: not LIST");

  outx_tagOfList(l);
  FOR_ITEMS_IN_LIST(lp, v) {
    vv = LIST_ITEM(lp);
    if(vv == NULL) {
      outx_printi(l1,"<list/>\n");
    }else if(EXPV_CODE(vv) == ACC_PRAGMA){
      outx_ACC_clause(l1, vv);
    }else{
      fatal("outx_ACC_dir_clause_list: unknown list element");
    }
  }
  outx_listClose(l);
}


/**
 * output FassignStatement
 */
static void
outx_assignStatement(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfStatement(l, v);
    outx_expv(l1, EXPR_ARG1(v));
    outx_expv(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}

/**
 * output user defined operator
 */
static void
outx_udefOp(int l, expv v)
{
    char buf[1024];
    const int l1 = l + 1;
    expv id = EXPR_ARG1(v);

    sprintf(buf, "%s name=\"%s\"", XTAG(v), SYM_NAME(EXPR_SYM(id)));

    outx_typeAttrs(l, EXPV_TYPE(v), buf,
        TOPT_TYPEONLY|TOPT_NEXTLINE);

    outx_expv(l1, EXPR_ARG2(v));
    if(EXPR_CODE(v) == F95_USER_DEFINED_BINARY_EXPR)
        outx_expv(l1, EXPR_ARG3(v));

    outx_expvClose(l, v);
}

/**
 * output binary operator
 */
static void
outx_binaryOp(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfExpression(l, v);
    outx_expv(l1, EXPR_ARG1(v));
    outx_expv(l1, EXPR_ARG2(v));
    outx_expvClose(l, v);
}


/**
 * output unary operator
 */
static void
outx_unaryOp(int l, expv v)
{
    outx_tagOfExpression(l, v);
    outx_expv(l + 1, EXPR_ARG1(v));
    outx_expvClose(l, v);
}

/**
 * output rename
 */
static void
outx_useRename(int l, expv local, expv use)
{
    assert(local != NULL);
    assert(use != NULL);

    outx_printi(l, "<rename local_name=\"%s\"", getRawString(local));
    outx_printi(0, " use_name=\"%s\"/>\n", getRawString(use));
}

/**
 * output FuseDecl
 */
static void
outx_useDecl(int l, expv v, int is_intrinsic)
{
    list lp;
    const char *mod_name = SYM_NAME(EXPV_NAME(EXPR_ARG1(v)));

    if(is_intrinsic) {
        outx_tagOfDecl1(l, "%s name=\"%s\" intrinsic=\"true\"", EXPR_LINE(v), XTAG(v), mod_name);
    } else {
        outx_tagOfDecl1(l, "%s name=\"%s\"", EXPR_LINE(v), XTAG(v), mod_name);
    }

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)) {
        expv x = LIST_ITEM(lp);
        outx_useRename(l+1, EXPR_ARG1(x), EXPR_ARG2(x));
    }

    include_module_file(print_fp,EXPV_NAME(EXPR_ARG1(v)));

    outx_expvClose(l, v);
}

/**
 * output rename
 */
static void
outx_useRenamable(int l, expv local, expv use)
{
    assert(use != NULL);

    outx_printi(l, "<renamable");

    if (local != NULL)
        outx_printi(0," local_name=\"%s\"", getRawString(local));

    outx_printi(0, " use_name=\"%s\"/>\n", getRawString(use));
}

/**
 * output FuseOnlyDecl
 */
static void
outx_useOnlyDecl(int l, expv v, int is_intrinsic)
{
    list lp;
    const char *mod_name = SYM_NAME(EXPV_NAME(EXPR_ARG1(v)));

    if(is_intrinsic) {
        outx_tagOfDecl1(l, "%s name=\"%s\" intrinsic=\"true\"", EXPR_LINE(v), XTAG(v), mod_name);    
    } else {
        outx_tagOfDecl1(l, "%s name=\"%s\"", EXPR_LINE(v), XTAG(v), mod_name);
    }

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)) {
        expv x = LIST_ITEM(lp);
        outx_useRenamable(l+1, EXPR_ARG1(x), EXPR_ARG2(x));
    }

    include_module_file(print_fp,EXPV_NAME(EXPR_ARG1(v)));

    outx_expvClose(l, v);
}



static void
outx_syncstat_list(int l, expv v)
{
    list lp;
    expv x;
    FOR_ITEMS_IN_LIST(lp, v) {
        x = LIST_ITEM(lp);
        outx_printi(l, "<syncStat kind=\"%s\">\n",
                    EXPV_KWOPT_NAME(x)
                    );
        outx_varOrFunc(l + 1, x);
        outx_close(l, "syncStat");
    }
}


/**
 * output syncAllStatement
 */
static void
outx_SYNCALL_statement(int l, expv v)
{
    outx_tagOfStatement(l, v);
    outx_syncstat_list(l + 1, v);
    outx_expvClose(l, v);
}


/**
 * output syncImagesStatement
 */
static void
outx_SYNCIMAGES_statement(int l, expv v)
{
    outx_tagOfStatement(l, v);
    if (EXPR_HAS_ARG1(v)) {
        outx_expv(l + 1, EXPR_ARG1(v));
    }
    if (EXPR_HAS_ARG2(v)) {
        outx_syncstat_list(l + 1, EXPR_ARG2(v));
    }
    outx_expvClose(l, v);
}

/**
 * output syncmemoryStatement
 */
static void
outx_SYNCMEMORY_statement(int l, expv v)
{
    outx_tagOfStatement(l, v);
    outx_syncstat_list(l + 1, v);
    outx_expvClose(l, v);
}

/*
 * output criticalStatement
 */
static void
outx_CRITICAL_statement(int l, expv v)
{
    char buf[128];

    if (XMP_coarray_flag) {
        outx_expv(l, EXPR_ARG1(v));

    } else {
        if (EXPR_HAS_ARG2(v) && EXPR_ARG2(v) != NULL) {
            sprintf(buf, " construct_name=\"%s\"",
                    SYM_NAME(EXPR_SYM(EXPR_ARG2(v))));
            outx_tagOfStatement1(l, v, buf);
        } else {
            outx_tagOfStatement(l, v);
        }
        outx_body(l + 1, EXPR_ARG1(v));
        outx_expvClose(l, v);
    }
}


/**
 * output syncmemoryStatement
 */
static void
outx_LOCK_statement(int l, expv v)
{
    outx_tagOfStatement(l, v);
    if (EXPR_HAS_ARG1(v)) {
        outx_expv(l + 1, EXPR_ARG1(v));
    }
    if (EXPR_HAS_ARG2(v)) {
        outx_syncstat_list(l + 1, EXPR_ARG2(v));
    }
    outx_expvClose(l, v);
}


/**
 * output syncmemoryStatement
 */
static void
outx_UNLOCK_statement(int l, expv v)
{
    outx_tagOfStatement(l, v);
    if (EXPR_HAS_ARG1(v)) {
        outx_expv(l + 1, EXPR_ARG1(v));
    }
    if (EXPR_HAS_ARG2(v)) {
        outx_syncstat_list(l + 1, EXPR_ARG2(v));
    }
    outx_expvClose(l, v);
}


/*
 * output blockStatement
 */
static void
outx_BLOCK_statement(int l, expv v)
{
    char buf[128];
    EXT_ID ep;
    BLOCK_ENV block;
    list lp;
    ID id;
    int l1 = l + 1;
    int l2 = l + 2;

    if (EXPR_HAS_ARG2(v) && EXPR_ARG2(v) != NULL) {
        sprintf(buf, " construct_name=\"%s\"",
                SYM_NAME(EXPR_SYM(EXPR_ARG2(v))));
        outx_tagOfStatement1(l, v, buf);
    } else {
        outx_tagOfStatement(l, v);
    }
    block = EXPR_BLOCK(v);

    outx_tag(l1, "symbols");
    FOREACH_ID(id, BLOCK_LOCAL_SYMBOLS(block)) {
        if (id_is_visibleVar_for_symbols(id))
            outx_id(l2, id);
    }
    outx_close(l1, "symbols");

    outx_tag(l1, "declarations");

    /*
     * FuseDecl
     */
    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(v)) {
        expv u = LIST_ITEM(lp);
        switch(EXPV_CODE(u)) {
        case F03_USE_INTRINSIC_STATEMENT:
            outx_useDecl(l2, u, TRUE);
            break;
        case F95_USE_STATEMENT:
            outx_useDecl(l2, u, FALSE);
            break;
        case F03_USE_ONLY_INTRINSIC_STATEMENT:
            outx_useOnlyDecl(l2, u, TRUE);
            break;            
        case F95_USE_ONLY_STATEMENT:
            outx_useOnlyDecl(l2, u, FALSE);
            break;
        default:
            break;
        }
    }

    outx_id_declarations(l2, BLOCK_LOCAL_SYMBOLS(block), FALSE, NULL);

    /*
     * FinterfaceDecl
     */
    FOREACH_EXT_ID(ep, BLOCK_LOCAL_INTERFACES(block)) {
        outx_interfaceDecl(l2, ep);
    }

    outx_close(l1, "declarations");
    outx_body(l1, EXPR_ARG1(v));
    outx_expvClose(l, v);
}

/*
 * output forallStatement
 */
static void
outx_FORALL_statement(int l, expv v)
{
    list lp;
    int l1 = l + 1;
    expv init = EXPR_ARG1(v);
    expv mask = EXPR_ARG2(v);
    expv body = EXPR_ARG3(v);
    const char *tid = NULL;

    outx_vtagLineno(l, XTAG(v), EXPR_LINE(v), NULL);

    if (EXPR_HAS_ARG4(v) && EXPR_ARG4(v) != NULL) {
        outx_print(" construct_name=\"%s\"",
                   SYM_NAME(EXPR_SYM(EXPR_ARG4(v))));
    }
    if (EXPV_TYPE(v)) {
        tid = getTypeID(EXPV_TYPE(v));
        outx_print(" type=\"%s\"", tid);
    }
    outx_print(">\n");

#if 0
    /*
     * NOTE:
     *  Comment out by specification changed.
     *  the BLOCK statement will have symbols for FORALL statement
     *
     *  It may be useful to output <symbols> for FORALL statement
     *  to describe the indices of FORALL statement
     *
     * ex)
     *
     *   FORALL( INTEGER :: I = 1:3 )
     *   ! print I to the <symbols> in <forallStatement>
     *
     */
    if (BLOCK_LOCAL_SYMBOLS(EXPR_BLOCK(v))) {
        ID id;
        BLOCK_ENV block = EXPR_BLOCK(v);
        outx_tag(l1, "symbols");
        FOREACH_ID(id, BLOCK_LOCAL_SYMBOLS(block)) {
            if (id_is_visibleVar_for_symbols(id))
                outx_id(l2, id);
        }
        outx_close(l1, "symbols");
    }
#endif


    FOR_ITEMS_IN_LIST(lp, init) {
        expv name = EXPR_ARG1(LIST_ITEM(lp));
        expv indexRange = EXPR_ARG2(LIST_ITEM(lp));

        outx_varOrFunc(l1, name);
        outx_indexRange(l1,
                        EXPR_ARG1(indexRange),
                        EXPR_ARG2(indexRange),
                        EXPR_ARG3(indexRange));
    }

    if (mask) {
        outx_condition(l1, mask);
    }

    outx_body(l1, body);
    outx_expvClose(l, v);
}

static void
outx_lenspec(int l, expv v)
{
    outx_printi(l, "<len ");
    switch (EXPR_CODE(v)) {
        case LEN_SPEC_ASTERISC:
            outx_print(" is_assumed_size=\"true\">\n");
            break;
        case F08_LEN_SPEC_COLON:
            outx_print(" is_assumed_shape=\"true\">\n");
            break;
        default:
            // never reach
            break;
    }
    outx_close(l, "len");
}

static void
outx_expv(int l, expv v)
{
    enum expr_code code;
    if(v == NULL)
        return;
    code = EXPV_CODE(v);

    switch(code) {
    /*
     * identifiers
     */
    case F_FUNC:            outx_varOrFunc(l, v); break;
    case IDENT:
    case F_VAR:
    case F_PARAM:           outx_varOrFunc(l, v); break;

    /*
     * constants
     */
    case INT_CONSTANT:
    case STRING_CONSTANT:
    case FLOAT_CONSTANT:
    case COMPLEX_CONSTANT:  outx_constants(l, v); break;

    /*
     * declarations
     */
    case F_FORMAT_DECL:     outx_formatDecl(l, v); break;

    /*
     * general statements
     */
    case EXPR_STATEMENT:            outx_exprStatement(l, v); break;
    case F_DO_STATEMENT:            outx_doStatement(l, v); break;
    case F_DOWHILE_STATEMENT:       outx_doWhileStatement(l, v); break;
    case F_SELECTCASE_STATEMENT:    outx_selectStatement(l, v); break;
    case F03_SELECTTYPE_STATEMENT:  outx_selectTypeStatement(l, v); break;
    case IF_STATEMENT:
    case F_WHERE_STATEMENT:         outx_IFWHERE_Statement(l, v); break;
    case F_RETURN_STATEMENT:        outx_returnStatement(l, v); break;
    case F_CONTINUE_STATEMENT:      outx_continueStatement(l, v); break;
    case GOTO_STATEMENT:            outx_gotoStatement(l, v); break;
    case F_COMPGOTO_STATEMENT:      outx_compgotoStatement(l, v); break;
    case STATEMENT_LABEL:           outx_labeledStatement(l, v); break;
    case F_CASELABEL_STATEMENT:     outx_caseLabel(l, v); break;
    case F03_CLASSIS_STATEMENT:     outx_typeGuard(l, v, 1); break;
    case F03_TYPEIS_STATEMENT:      outx_typeGuard(l, v, 0); break;
    case F_STOP_STATEMENT:
    case F_PAUSE_STATEMENT:         outx_STOPPAUSE_statement(l, v); break;
    case F_LET_STATEMENT:           outx_assignStatement(l, v); break;
    case F_PRAGMA_STATEMENT:        outx_pragmaStatement(l, v); break;
    case F95_CYCLE_STATEMENT:
    case F95_EXIT_STATEMENT:        outx_EXITCYCLE_statement(l, v); break;
    case F_ENTRY_STATEMENT:         outx_entryDecl(l, v); break;

    /*
     * IO statements
     */
    case F_PRINT_STATEMENT:         outx_printStatement(l, v); break;
    case F_READ_STATEMENT:
    case F_WRITE_STATEMENT:
    case F_INQUIRE_STATEMENT:       outx_RW_statement(l, v); break;
    case F_READ1_STATEMENT:
    case F_OPEN_STATEMENT:
    case F_CLOSE_STATEMENT:
    case F_BACKSPACE_STATEMENT:
    case F_ENDFILE_STATEMENT:
    case F_REWIND_STATEMENT:        outx_IO_statement(l, v); break;

    case F03_FLUSH_STATEMENT:       outx_FLUSH_statement(l, v); break;

    /*
     * F03 statements
     */
    case F03_WAIT_STATEMENT:        outx_IO_statement(l, v); break;


    /*
     * F90/95 Pointer related.
     */
    case F95_POINTER_SET_STATEMENT: outx_pointerAssignStatement(l, v); break;
    case F95_ALLOCATE_STATEMENT:
    case F95_DEALLOCATE_STATEMENT:  outx_ALLOCDEALLOC_statement(l, v); break;
    case F95_NULLIFY_STATEMENT:     outx_nullifyStatement(l, v); break;

    /*
     * expressions
     */
    case FUNCTION_CALL:     outx_functionCall(l, v); break;
    case F95_MEMBER_REF:    outx_memberRef(l, v); break;
    case ARRAY_REF:         outx_arrayRef(l, v); break;
    case F_SUBSTR_REF:      outx_characterRef(l, v); break;
    case F95_ARRAY_CONSTRUCTOR:     outx_arrayConstructor(l, v); break;
    case F03_TYPED_ARRAY_CONSTRUCTOR:     outx_typedArrayConstructor(l, v); break;
    case F95_STRUCT_CONSTRUCTOR:    outx_structConstructor(l, v); break;

    case XMP_COARRAY_REF:   outx_coarrayRef(l, v); break;
    /*
     * operators
     */
    case PLUS_EXPR:
    case MINUS_EXPR:
    case MUL_EXPR:
    case DIV_EXPR:
    case POWER_EXPR:
    case LOG_EQ_EXPR:
    case LOG_NEQ_EXPR:
    case LOG_GE_EXPR:
    case LOG_GT_EXPR:
    case LOG_LE_EXPR:
    case LOG_LT_EXPR:
    case LOG_AND_EXPR:
    case LOG_OR_EXPR:
    case F_EQV_EXPR:
    case F_NEQV_EXPR:
    case F_CONCAT_EXPR:     outx_binaryOp(l, v); break;
    case LOG_NOT_EXPR:
    case UNARY_MINUS_EXPR:  outx_unaryOp(l, v); break;


    case F95_USER_DEFINED_BINARY_EXPR:
    case F95_USER_DEFINED_UNARY_EXPR: outx_udefOp(l, v); break;

    /*
     * misc.
     */
    case F_IMPLIED_DO:          outx_doLoop(l, v); break;
    case F_INDEX_RANGE:         outx_indexRangeInList(l, v); break;
    case F_SCENE_RANGE_EXPR:    outx_sceneRange(l, v); break;
    case F_MODULE_INTERNAL:
        /*
         * When using other module, F_MODULE_INTERNAL is set as dummy
         * expression insted of real value defined in module.
         * We emit dummy FintConstant for F_Back.
         */
        outx_printi(l, "<!-- dummy value for imported type -->\n");
        outx_intAsConst(l, -INT_MAX);
        break;


    /*
     * type parameter values.
     */
    case LEN_SPEC_ASTERISC:
    case F08_LEN_SPEC_COLON:
        outx_lenspec(l, v);
        break;
    /*
     * elements to skip
     */
    case F_DATA_DECL:
    case F_EQUIV_DECL:
    case F95_TYPEDECL_STATEMENT:
    case FIRST_EXECUTION_POINT:
    case F95_INTERFACE_STATEMENT:
    case F95_USE_STATEMENT:
    case F95_USE_ONLY_STATEMENT:
    case F03_USE_INTRINSIC_STATEMENT:
    case F03_USE_ONLY_INTRINSIC_STATEMENT:
        break;

    /*
     * child elements
     */
    case LIST:
	{
            list lp;
            FOR_ITEMS_IN_LIST(lp, v)
                outx_expv(l, LIST_ITEM(lp));
        }
        break;

    /*
     * invalid or no corresponding tag
     */
    case ERROR_NODE:
    case BASIC_TYPE_NODE:
    case DEFAULT_LABEL:
    case ID_LIST:
    case VAR_DECL:
    case EXT_DECL:
    case F_PROGRAM_STATEMENT:
    case F_BLOCK_STATEMENT:
    case F_SUBROUTINE_STATEMENT:
    case F_FUNCTION_STATEMENT:
    case F_INCLUDE_STATEMENT:
    case F_END_STATEMENT:
    case F_TYPE_DECL:
    case F_COMMON_DECL:
    case F_EXTERNAL_DECL:
    case F_INTRINSIC_DECL:
    case F_IMPLICIT_DECL:
    case F_NAMELIST_DECL:
    case F_SAVE_DECL:
    case F_PARAM_DECL:
    case F_DUP_DECL:
    case F_UNARY_MINUS:
    case F_ENDDO_STATEMENT:
    case F_ELSEWHERE_STATEMENT:
    case F_ENDWHERE_STATEMENT:
    case F_ENDSELECT_STATEMENT:
    case F_IF_STATEMENT:
    case F_ELSEIF_STATEMENT:
    case F_ELSE_STATEMENT:
    case F_ENDIF_STATEMENT:
    case F_ASSIGN_LABEL_STATEMENT:
    case F_GOTO_STATEMENT:
    case F_ASGOTO_STATEMENT:
    case F_ARITHIF_STATEMENT:
    case F_CALL_STATEMENT:
    case F_CRAY_POINTER_DECL:
    case F_SET_EXPR:
    case F_LABEL_REF:
    case F_PLUS_EXPR:
    case F_MINUS_EXPR:
    case F_MUL_EXPR:
    case F_DIV_EXPR:
    case F_UNARY_MINUS_EXPR:
    case F_POWER_EXPR:
    case F_EQ_EXPR:
    case F_GT_EXPR:
    case F_GE_EXPR:
    case F_LT_EXPR:
    case F_LE_EXPR:
    case F_NE_EXPR:
    case F_OR_EXPR:
    case F_AND_EXPR:
    case F_NOT_EXPR:
    case F_ARRAY_REF:
    case F_STARSTAR:
    case F_TRUE_CONSTANT:
    case F_FALSE_CONSTANT:
    case F_TYPE_NODE:
    case F95_CONSTANT_WITH:
    case F95_TRUE_CONSTANT_WITH:
    case F95_FALSE_CONSTANT_WITH:
    case F95_ENDPROGRAM_STATEMENT:
    case F95_ENDSUBROUTINE_STATEMENT:
    case F95_ENDFUNCTION_STATEMENT:
    case F95_MODULE_STATEMENT:
    case F95_ENDMODULE_STATEMENT:
    case F95_ENDINTERFACE_STATEMENT:
    case F95_CONTAINS_STATEMENT:
    case F95_RECURSIVE_SPEC:
    case F95_PURE_SPEC:
    case F95_ELEMENTAL_SPEC:
    case F95_DIMENSION_DECL:
    case F95_ENDTYPEDECL_STATEMENT:
    case F95_PRIVATE_STATEMENT:
    case F03_PROTECTED_STATEMENT:
    case F95_SEQUENCE_STATEMENT:
    case F95_PARAMETER_SPEC:
    case F95_ALLOCATABLE_SPEC:
    case F95_DIMENSION_SPEC:
    case F95_EXTERNAL_SPEC:
    case F95_INTENT_SPEC:
    case F95_INTRINSIC_SPEC:
    case F95_OPTIONAL_SPEC:
    case F95_POINTER_SPEC:
    case F95_SAVE_SPEC:
    case F95_TARGET_SPEC:
    case F95_PUBLIC_SPEC:
    case F95_PRIVATE_SPEC:
    case F03_PROTECTED_SPEC:
    case F03_BIND_SPEC:
    case F03_VALUE_SPEC:
    case F95_IN_EXTENT:
    case F95_OUT_EXTENT:
    case F95_INOUT_EXTENT:
    case F95_KIND_SELECTOR_SPEC:
    case F95_LEN_SELECTOR_SPEC:
    case F95_STAT_SPEC:
    case F95_TRIPLET_EXPR:
    case F95_PUBLIC_STATEMENT:
    case F95_OPTIONAL_STATEMENT:
    case F95_POINTER_STATEMENT:
    case F95_INTENT_STATEMENT:
    case F95_TARGET_STATEMENT:
    case F_ASTERISK:
    case F_EXTFUNC:
    case F_DOUBLE_CONSTANT:
    case F95_ASSIGNOP:
    case F95_DOTOP:
    case F95_POWEOP:
    case F95_MULOP:
    case F95_DIVOP:
    case F95_PLUSOP:
    case F95_MINUSOP:
    case F95_EQOP:
    case F95_NEOP:
    case F95_LTOP:
    case F95_LEOP:
    case F95_GEOP:
    case F95_GTOP:
    case F95_NOTOP:
    case F95_ANDOP:
    case F95_OROP:
    case F95_EQVOP:
    case F95_NEQVOP:
    case F95_CONCATOP:
    case F95_USER_DEFINED:
    case F95_MODULEPROCEDURE_STATEMENT:
    case F95_ARRAY_ALLOCATION:
    case F95_ALLOCATABLE_STATEMENT:
    case F95_GENERIC_SPEC:
    case XMP_CODIMENSION_SPEC:
    case EXPR_CODE_END:
    case F2008_ENDCRITICAL_STATEMENT:

        if(debug_flag)
            expv_output(v, stderr);
        fatal("invalid exprcode : %s", EXPR_CODE_NAME(code));
        abort();

    case OMP_PRAGMA:
      outx_OMP_pragma(l, v);
      break;

    case XMP_PRAGMA:
      outx_XMP_pragma(l, v);
      break;

    case F2008_SYNCALL_STATEMENT:
      outx_SYNCALL_statement(l, v);
      break;

    case F2008_SYNCIMAGES_STATEMENT:
      outx_SYNCIMAGES_statement(l, v);
      break;

    case F2008_SYNCMEMORY_STATEMENT:
      outx_SYNCMEMORY_statement(l, v);
      break;

    case F2008_CRITICAL_STATEMENT:
      outx_CRITICAL_statement(l, v);
      break;

    case F2008_LOCK_STATEMENT:
      outx_LOCK_statement(l, v);
      break;

    case F2008_UNLOCK_STATEMENT:
      outx_UNLOCK_statement(l, v);
      break;

    case ACC_PRAGMA:
      outx_ACC_pragma(l, v);
      break;

    case F2008_BLOCK_STATEMENT:
      outx_BLOCK_statement(l, v);
      break;

    case F_FORALL_STATEMENT:
      outx_FORALL_statement(l, v);
      break;

    default:
        fatal("unkown exprcode : %d", code);
        abort();
    }
}

static void mark_type_desc_in_structure(TYPE_DESC tp);
//static void check_type_desc(TYPE_DESC tp);

static void mark_type_desc(TYPE_DESC tp);

static void
mark_type_desc_skip_tbp(TYPE_DESC tp, int skip_tbp)
{
    if (tp == NULL || TYPE_IS_REFERENCED(tp) == TRUE || IS_MODULE(tp))
        return;

    if (skip_tbp &&  IS_PROCEDURE_TYPE(tp) && TYPE_REF(tp) != NULL) {
        /* procedure variable or type-bound procedure with a PASS argument
         * may cause a circulation reference,
         * so store them to a list and check them later.
         */
        TYPE_LINK(tp) = NULL;
        TYPE_IS_REFERENCED(tp) = TRUE;
        TYPE_LINK_ADD(tp, tbp_list, tbp_list_tail);
        return;
    }

    if (TYPE_BOUND_GENERIC_TYPE_GENERICS(tp) != NULL) {
        /* the type for type-bound generic, skip it */
        return;
    }

    if (TYPE_REF(tp) != NULL) {
        TYPE_DESC sTp = NULL;
        if (IS_ARRAY_TYPE(tp)){
            mark_type_desc(array_element_type(tp));
        }
        sTp = reduce_type(TYPE_REF(tp));
        mark_type_desc(sTp);
        TYPE_REF(tp) = sTp;
    }

    collect_type_desc(TYPE_KIND(tp));
    collect_type_desc(TYPE_LENG(tp));
    collect_type_desc(TYPE_DIM_SIZE(tp));
    collect_type_desc(TYPE_DIM_UPPER(tp));
    collect_type_desc(TYPE_DIM_LOWER(tp));
    collect_type_desc(TYPE_DIM_STEP(tp));

    TYPE_LINK_ADD(tp, type_list, type_list_tail);
    TYPE_IS_REFERENCED(tp) = TRUE;

    if (IS_PROCEDURE_TYPE(tp)) {
        ID ip;
        mark_type_desc(TYPE_REF(tp));
        mark_type_desc(FUNCTION_TYPE_RETURN_TYPE(tp));
        FOREACH_ID(ip, FUNCTION_TYPE_ARGS(tp)) {
            mark_type_desc(ID_TYPE(ip));
        }
    }

    if (IS_STRUCT_TYPE(tp)) {
        mark_type_desc_in_structure(tp);
    }
}

static void
mark_type_desc(TYPE_DESC tp) {
    mark_type_desc_skip_tbp(tp, TRUE);
}

/* static void */
/* check_type_desc(TYPE_DESC tp) */
/* { */
/*   if (tp == NULL || TYPE_IS_REFERENCED(tp) || IS_MODULE(tp)) */
/*     return; */

/*   TYPE_IS_REFERENCED(tp) = 1; */

/*   if (TYPE_REF(tp)){ */

/*     if (IS_ARRAY_TYPE(tp)){ */

/*       if (TYPE_IS_COSHAPE(tp)){ */

/* 	if (TYPE_IS_COSHAPE(TYPE_REF(tp))){ */
/* 	  check_type_desc(TYPE_REF(tp)); */
/* 	} */
/* 	else { */

/* 	  mark_type_desc(TYPE_REF(tp)); */
/* 	} */

/*       } */
/*       else { */
/* 	mark_type_desc(array_element_type(tp)); */
/*       } */

/*     } */
/*     else { */

/*       mark_type_desc(TYPE_REF(tp)); */

/*     } */

/*   } */

/* } */


static void
mark_type_desc_in_structure(TYPE_DESC tp)
{
    ID id;
    TYPE_DESC itp, siTp;

    if (TYPE_PARENT(tp)) {
        mark_type_desc(TYPE_PARENT_TYPE(tp));
    }

    FOREACH_MEMBER(id, tp) {
        itp = ID_TYPE(id);
        siTp = reduce_type(itp);
        mark_type_desc(siTp);
        ID_TYPE(id) = siTp;
        if (!IS_PROCEDURE_TYPE(ID_TYPE(id)) &&  VAR_INIT_VALUE(id) != NULL) {
            collect_type_desc(VAR_INIT_VALUE(id));
        }
        if (IS_STRUCT_TYPE(ID_TYPE(id))) {
            mark_type_desc_in_structure(ID_TYPE(id));
        }
    }
}


static void
collect_type_desc(expv v)
{
    list lp;
    TYPE_DESC sTp;

    if (v == NULL) return;
    sTp = reduce_type(EXPV_TYPE(v));
    mark_type_desc(sTp);
    EXPV_TYPE(v) = sTp;
    if (EXPR_CODE_IS_TERMINAL(EXPV_CODE(v))) return;

    FOR_ITEMS_IN_LIST(lp, v)
        collect_type_desc(LIST_ITEM(lp));
}


void
add_type_ext_id(EXT_ID ep)
{
    //TYPE_EXT_ID te = (TYPE_EXT_ID)malloc(sizeof(struct type_ext_id));
    TYPE_EXT_ID te = XMALLOC(TYPE_EXT_ID, sizeof(struct type_ext_id));
    //bzero(te, sizeof(struct type_ext_id));
    te->ep = ep;
    FUNC_EXT_LINK_ADD(te, type_ext_id_list, type_ext_id_last);
}


static void
mark_type_desc_id(ID id)
{
    TYPE_DESC sTp;

    sTp = reduce_type(ID_TYPE(id));
    mark_type_desc(sTp);
    ID_TYPE(id) = sTp;
    collect_type_desc(ID_ADDR(id));
    switch(ID_CLASS(id)) {
        case CL_PARAM:
            collect_type_desc(VAR_INIT_VALUE(id));
            return;
        case CL_VAR:
            collect_type_desc(VAR_INIT_VALUE(id));
            /* fall through */
        case CL_PROC:
            if (PROC_EXT_ID(id) && EXT_TAG(PROC_EXT_ID(id)) != STG_UNKNOWN &&
                (PROC_CLASS(id) == P_INTRINSIC ||
                 PROC_CLASS(id) == P_EXTERNAL ||
                 PROC_CLASS(id) == P_DEFINEDPROC)) {
                /* symbol declared as intrinsic */
                sTp = reduce_type(EXT_PROC_TYPE(PROC_EXT_ID(id)));
                mark_type_desc(sTp);
                EXT_PROC_TYPE(PROC_EXT_ID(id)) = sTp;
            }
            return;
        case CL_MULTI:
            mark_type_desc_in_id_list(MULTI_ID_LIST(id));
            return;
        default:
            return;
    }
}

static void
mark_type_desc_in_id_list(ID ids)
{
    ID id;
    FOREACH_ID(id, ids) {
        mark_type_desc_id(id);
    }
}

/**
 * remove mark from type in type list.
 */
static void
unmark_type_table()
{
    TYPE_DESC tp;
    for (tp = type_list; tp != NULL; tp = TYPE_LINK(tp)){
        if (tp == NULL || TYPE_IS_REFERENCED(tp) == FALSE || IS_MODULE(tp))
            continue;
        TYPE_IS_REFERENCED(tp) = FALSE;
    }
}

static void
outx_kind(int l, TYPE_DESC tp)
{
    static expv doubledKind = NULL;
    expv vkind;

    if(doubledKind == NULL)
        doubledKind = expv_int_term(INT_CONSTANT, type_INT, KIND_PARAM_DOUBLE);

    if(IS_DOUBLED_TYPE(tp))
        vkind = doubledKind;
    else if(tp && TYPE_KIND(tp))
        vkind = TYPE_KIND(tp);
    else
        return;

    outx_childrenWithTag(l, "kind", vkind);
}


/**
 * output basicType of coarray
 */
static void
outx_coShape(int l, TYPE_DESC tp)
{
  list lp;
  codims_desc *codims = tp->codims;

  outx_printi(l, "<coShape>\n");

  FOR_ITEMS_IN_LIST(lp, codims->cobound_list){
    expr cobound = LIST_ITEM(lp);
    expr upper = EXPR_ARG2(cobound);
    ARRAY_ASSUME_KIND defaultAssumeKind = ASSUMED_SHAPE;

    if (upper && EXPR_CODE(upper) == F_ASTERISK){
      upper = NULL;
      defaultAssumeKind = ASSUMED_SIZE;
    }

    outx_indexRange0(l+1, ASSUMED_NONE, defaultAssumeKind,
                     EXPR_ARG1(cobound), upper, EXPR_ARG3(cobound));
  }

  outx_close(l, "coShape");
}


/**
 * output basicType of character
 */
static void
outx_characterType(int l, TYPE_DESC tp)
{
    const int l1 = l + 1, l2 = l1 + 1;
    int charLen = TYPE_CHAR_LEN(tp);
    expv vcharLen = TYPE_LENG(tp);
    const char *tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
    TYPE_DESC tRef = TYPE_REF(tp);

    outx_typeAttrs(l, tp, "FbasicType", 0);

    if(tRef && checkBasic(tRef)) {
        tp = tRef;
        tRef = NULL;
    }

    if (tRef) {
        if (tp->codims && !(tRef->codims)) {
            outx_print(" ref=\"C" ADDRX_PRINT_FMT "\">\n", Addr2Uint(tRef));
            outx_coShape(l+1, tp);
            outx_close(l, "FbasicType");
        } else {
            outx_print(" ref=\"C" ADDRX_PRINT_FMT "\"/>\n", Addr2Uint(tRef));
        }
    }
    else if (TYPE_KIND(tp) || charLen != 1 || vcharLen != NULL || tp->codims) {
        outx_print(" ref=\"%s\">\n", tid);
        outx_kind(l1, tp);

        if(charLen != 1|| vcharLen != NULL) {
            if (!IS_CHAR_LEN_UNFIXED(tp) && !IS_CHAR_LEN_ALLOCATABLE(tp)) {
                outx_tag(l1, "len");
                if(vcharLen != NULL)
                    outx_expv(l2, vcharLen);
                else
                    outx_intAsConst(l2, TYPE_CHAR_LEN(tp));
            } else if (IS_CHAR_LEN_UNFIXED(tp)) {
                outx_printi(l1, "<len");
                outx_true(TRUE, "is_assumed_size");
                outx_print(">\n");
            } else if (IS_CHAR_LEN_ALLOCATABLE(tp)) {
                outx_printi(l1, "<len");
                outx_true(TRUE, "is_assumed_shape");
                outx_print(">\n");
            }
            outx_close(l1, "len");
        }
        if (tp->codims) outx_coShape(l+1, tp);
        outx_close(l, "FbasicType");
    } else {
        outx_print(" ref=\"%s\"/>\n", tid);
    }
}


/**
 * output basicType except character, array
 */
static void
outx_basicTypeNoCharNoAry(int l, TYPE_DESC tp)
{
    TYPE_DESC rtp = TYPE_REF(tp);
    assert(rtp);
    outx_typeAttrs(l, tp, "FbasicType", 0);
    if (TYPE_TYPE_PARAM_VALUES(tp) || tp->codims){
        outx_print(" ref=\"%s\">\n", getTypeID(rtp));
        if (tp->codims) outx_coShape(l+1, tp);
        if (TYPE_TYPE_PARAM_VALUES(tp)) {
            outx_typeParamValues(l+1, TYPE_TYPE_PARAM_VALUES(tp));
        }
        outx_close(l ,"FbasicType");
    }
    else
        outx_print(" ref=\"%s\"/>\n", getTypeID(rtp));
}


/**
 * output basicType except character, array, without TYPE_REF.
 */
static void
outx_basicTypeNoCharNoAryNoRef(int l, TYPE_DESC tp)
{

    /* TYPE_FUNCTION comes here in the following case (maybe for the reference
       in the argument list). This is only an ad-hoc fix.

       subroutine sub(subsub)
       implicit none
       external subsub
       end
    */
    if (IS_FUNCTION_TYPE(tp)) return;

    outx_typeAttrs(l, tp, "FbasicType", 0);

    /* Output type as generic numeric type when type is not fixed */
    if (TYPE_IS_NOT_FIXED(tp) && TYPE_BASIC_TYPE(tp) == TYPE_UNKNOWN) {
        outx_print(" ref=\"%s\"", "FnumericAll");
    } else {
        /* tp is basic data type */
        outx_print(" ref=\"%s\"", getBasicTypeID(TYPE_BASIC_TYPE(tp)));
    }

    if (TYPE_KIND(tp) || IS_DOUBLED_TYPE(tp) || tp->codims){
        outx_print(">\n");
        outx_kind(l + 1, tp); // !!!
        if (tp->codims) outx_coShape(l+1, tp);
        outx_close(l, "FbasicType");
    } else {
        outx_print("/>\n");
    }
}


/**
 * output basicType of array
 */
static void
outx_arrayType(int l, TYPE_DESC tp)
{
    const int l1 = l + 1;

      outx_typeAttrs(l, tp, "FbasicType", 0);
      outx_print(" ref=\"%s\">\n", getTypeID(array_element_type(tp)));

      outx_indexRangeOfType(l1, tp);

      if (tp->codims) outx_coShape(l1, tp);

      outx_close(l ,"FbasicType");
}


/**
 * output the type of CLASS(*)
 */
static void
outx_unlimitedClass(int l, TYPE_DESC tp)
{
    outx_typeAttrs(l, tp ,"FbasicType", TOPT_CLOSE);
}


/**
 * output functionType of type bound procedure
 */
static void
outx_functionType_procedure(int l, TYPE_DESC tp)
{
    outx_typeAttrs(l, tp, "FbasicType", 0);

    if (FUNCTION_TYPE_HAS_BINDING_ARG(tp)) {
        if (FUNCTION_TYPE_HAS_PASS_ARG(tp)) {
            outx_printi(0, " pass=\"pass\"");
        } else {
            outx_printi(0, " pass=\"nopass\"");
        }
        if (FUNCTION_TYPE_PASS_ARG(tp) != NULL) {
            outx_printi(0, " pass_arg_name=\"%s\"",
                        SYM_NAME(ID_SYM(FUNCTION_TYPE_PASS_ARG(tp))));
        }
    }

    if (TYPE_REF(tp) && !TYPE_IS_IMPLICIT(TYPE_REF(tp))) {
        /*
         * TYPE_IS_IMPLICIT(TYPE_REF(tp)) means that
         * the procedure variable declared like "PROCEDURE(), POINTER :: p".
         * So don't emit this attribute.
         */
        outx_print(" ref=\"%s\"/>\n", getTypeID(TYPE_REF(tp)));
    } else {
        outx_print("/>\n");
    }
}


/**
 * output functionType of type bound procedure
 */
static void
outx_functionType(int l, TYPE_DESC tp)
{
    if (TYPE_IS_PROCEDURE(tp) || TYPE_REF(tp) != NULL) {
        /* type-bound procedure or procedure type */
        outx_functionType_procedure(l, tp);

    } else {
        const int l1 = l + 1, l2 = l1 + 1;
        const char *rtid = NULL;

        const char *tid = getTypeID(tp);

        outx_printi(l,"<FfunctionType type=\"%s\"", tid);

        if (FUNCTION_TYPE_RESULT(tp)) {
            outx_print(" result_name=\"%s\"",
                       SYM_NAME(FUNCTION_TYPE_RESULT(tp)));
        }

        /* outx_typeAttrOnly_functionTypeWithResultVar(l, ep, "FfunctionType"); */

        if (FUNCTION_TYPE_RETURN_TYPE(tp)) {
            rtid = getTypeID(FUNCTION_TYPE_RETURN_TYPE(tp));
        } else {
            rtid = "Fvoid";
        }

        outx_print(" return_type=\"%s\"", rtid);
        outx_true(FUNCTION_TYPE_IS_PROGRAM(tp), "is_program");

        if (FUNCTION_TYPE_IS_VISIBLE_INTRINSIC(tp))
            outx_true(TYPE_IS_INTRINSIC(tp), "is_intrinsic");

        outx_true(TYPE_IS_RECURSIVE(tp), "is_recursive");
        outx_true(TYPE_IS_PURE(tp), "is_pure");
        outx_true(TYPE_IS_ELEMENTAL(tp), "is_elemental");
        outx_true(TYPE_IS_MODULE(tp), "is_module");

        if (is_emitting_for_submodule) {
            /*
             * "is_defined" attribute is only for SUBMODULE
             */
            outx_true(FUNCTION_TYPE_IS_DEFINED(tp), "is_defined");
        }

        if (!TYPE_IS_INTRINSIC(tp) &&
            (TYPE_IS_EXTERNAL(tp) ||
             (XMP_flag &&
              !TYPE_IS_FOR_FUNC_SELF(tp) &&
              !FUNCTION_TYPE_IS_INTERNAL(tp) &&
              !FUNCTION_TYPE_IS_MOUDLE_PROCEDURE(tp)))) {
            outx_true(TRUE, "is_external");
        }

        outx_true(TYPE_IS_PUBLIC(tp), "is_public");
        outx_true(TYPE_IS_PRIVATE(tp), "is_private");
        outx_true(TYPE_IS_PROTECTED(tp), "is_protected");

        if(TYPE_HAS_BIND(tp)){
            outx_print(" bind=\"%s\"", "C");
            if(TYPE_BIND_NAME(tp)){
                outx_print(" bind_name=\"%s\"", EXPR_STR(TYPE_BIND_NAME(tp)));
            }
        }

        if (FUNCTION_TYPE_HAS_EXPLICIT_ARGS(tp)) {
            ID ip;
            outx_print(">\n");
            outx_tag(l1, "params");
            FOREACH_ID(ip, FUNCTION_TYPE_ARGS(tp)) {
                outx_symbolNameWithType_ID(l2, ip);
            }
            outx_close(l1, "params");
            outx_close(l, "FfunctionType");
        } else {
            outx_print("/>\n");
        }
    }
}


/**
 * output FstructType
 */
static void
outx_structType(int l, TYPE_DESC tp)
{
    ID id;
    int l1 = l + 1, l2 = l1 + 1, l3 = l2 + 1, l4 = l3 + 1;
    int has_type_bound_procedure = FALSE;

    outx_typeAttrs(l, tp ,"FstructType", TOPT_NEXTLINE);
    if (TYPE_TYPE_PARAMS(tp)) {
        outx_tag(l1, "typeParams");
        FOREACH_TYPE_PARAMS(id, tp) {
            if (!TYPE_IS_KIND(ID_TYPE(id)) && !TYPE_IS_LEN(ID_TYPE(id))) {
                error("'%s' is neither KIND nor LEN", ID_NAME(id));
                continue;
            }
            outx_printi(l2, "<typeParam ");
            outx_print("type=\"%s\" ", getTypeID(ID_TYPE(id)));
            if (TYPE_IS_KIND(ID_TYPE(id))) {
                outx_print("attr=\"kind\">\n");
            } else if (TYPE_IS_LEN(ID_TYPE(id))) {
                outx_print("attr=\"length\">\n");
            } else {
                fatal("%s: NEVER REACH.", __func__);
            }
            outx_symbolName(l3, ID_SYM(id));
            outx_value(l3, VAR_INIT_VALUE(id));
            outx_close(l2, "typeParam");
        }
        outx_close(l1, "typeParams");
    }

    outx_tag(l1, "symbols");
    FOREACH_MEMBER(id, tp) {
        if (ID_CLASS(id) == CL_TYPE_BOUND_PROC) {
            has_type_bound_procedure = TRUE;
            continue;
        }
        outx_printi(l2, "<id type=\"%s\">\n", getTypeID(ID_TYPE(id)));
        outx_symbolName(l3, ID_SYM(id));
        if (VAR_INIT_VALUE(id) != NULL) {
            outx_value(l3, VAR_INIT_VALUE(id));
        }
        outx_close(l2, "id");
    }
    outx_close(l1,"symbols");

    if (has_type_bound_procedure) {
        outx_tag(l1, "typeBoundProcedures");
        FOREACH_MEMBER(id, tp) {
            if (ID_CLASS(id) != CL_TYPE_BOUND_PROC) {
                continue;
            }
            if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_IS_GENERIC) {
                ID binding;
                int is_defined_io = FALSE;
                outx_printi(l2, "<typeBoundGenericProcedure");
                if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_IS_OPERATOR) {
                    outx_true(TRUE, "is_operator");
                }
                if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_IS_ASSIGNMENT) {
                    outx_true(TRUE, "is_assignment");
                }
                if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_READ ||
                    TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_WRITE) {
                    is_defined_io = TRUE;
                    outx_puts(" is_defined_io=\"");
                    if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_READ) {
                        outx_puts("READ");
                    } else {
                        outx_puts("WRITE");
                    }
                    if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_FORMATTED) {
                        outx_puts("(FORMATTED)");
                    } else {
                        outx_puts("(UNFORMATTED)");
                    }
                    outx_puts("\" ");
                }
                outx_printi(0,">\n");
                if (!is_defined_io) {
                    outx_tagText(l3, "name", SYM_NAME(ID_SYM(id)));
                }
                outx_tag(l3, "binding");
                FOREACH_ID(binding, TBP_BINDING(id)) {
                    outx_tagText(l4, "name", SYM_NAME(ID_SYM(binding)));
                }
                outx_close(l3, "binding");
                outx_close(l2, "typeBoundGenericProcedure");

            } else if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_IS_FINAL) {
                ID final;
                FOREACH_ID(final, TBP_BINDING(id)) {
                    outx_tag(l2, "finalProcedure");
                    outx_tagText(l3, "name", SYM_NAME(ID_SYM(final)));
                    outx_close(l2, "finalProcedure");
                }
            } else {
                outx_printi(l2, "<typeBoundProcedure");
                outx_printi(0, " type=\"%s\"", getTypeID(ID_TYPE(id)));

                if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_PASS)
                    outx_printi(0, " pass=\"pass\"");
                if (TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_NOPASS)
                    outx_printi(0, " pass=\"nopass\"");
                if (TBP_PASS_ARG(id))
                    outx_printi(0, " pass_arg_name=\"%s\"",
                                SYM_NAME(ID_SYM(TBP_PASS_ARG(id))));

                outx_true(TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_DEFERRED,
                          "is_deferred");
                outx_true(TBP_BINDING_ATTRS(id) & TYPE_BOUND_PROCEDURE_NON_OVERRIDABLE,
                          "is_non_overridable");
                outx_true(TYPE_IS_PUBLIC(id), "is_public");
                outx_true(TYPE_IS_PRIVATE(id), "is_private");
                outx_true(TYPE_IS_PROTECTED(id), "is_protected");

                outx_printi(0,">\n");

                outx_tagText(l3, "name", SYM_NAME(ID_SYM(id)));
                outx_tag(l3, "binding");
                outx_tagText(l4, "name", SYM_NAME(ID_SYM(TBP_BINDING(id))));
                outx_close(l3, "binding");

                outx_close(l2, "typeBoundProcedure");
            }
        }
        outx_close(l1, "typeBoundProcedures");
    }

    outx_close(l,"FstructType");
}


/**
 * output types in typeTable
 */
static void
outx_type(int l, TYPE_DESC tp)
{
    TYPE_DESC tRef = TYPE_REF(tp);

    if (IS_VOID(tp)) {
        /* output nothing */
    } else if(IS_CHAR(tp)) {
        if(checkBasic(tp) == FALSE || checkBasic(tRef) == FALSE)
            outx_characterType(l, tp);

    } else if(IS_ARRAY_TYPE(tp)) {
        outx_arrayType(l, tp);

    } else if(IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL) {
        if (TYPE_IS_CLASS(tp)) {
            outx_unlimitedClass(l, tp);
        } else {
            outx_structType(l, tp);
        }

    } else if(IS_PROCEDURE_TYPE(tp)) {
        outx_functionType(l, tp);

    } else if (tRef != NULL) {
        if (has_attribute_except_func_attrs(tp) ||
            TYPE_KIND(tRef) ||
            IS_DOUBLED_TYPE(tRef) ||
            IS_STRUCT_TYPE(tRef)) {
            outx_basicTypeNoCharNoAry(l, tp);
        }

    } else if (tRef == NULL) {
        if(checkBasic(tp) == FALSE)
            outx_basicTypeNoCharNoAryNoRef(l, tp);
    } else {
        fatal("%s: type not covered yet.", __func__);
    }
}


/**
 * distinguish the id is to be output
 */
static int
id_is_visibleVar(ID id)
{
    if (ID_DEFINED_BY(id) != NULL) {
        TYPE_DESC tp = ID_TYPE(id);
        if (tp == NULL) {
            return FALSE;
        }
        if (IS_GENERIC_TYPE(tp)) {
            /* The generic function visibility is differed from
             * function and subroutine.
             */
            return TRUE;
        }
        if (ID_CLASS(id) == CL_PROC &&
            CRT_FUNCEP != NULL &&
            CRT_FUNCEP != PROC_EXT_ID(id)) {
            return FALSE;
        }
        if (TYPE_IS_MODIFIED(tp)) {
            return TRUE;
        }
        if ((is_outputed_module && CRT_FUNCEP == NULL)
            && (TYPE_IS_PUBLIC(tp) || TYPE_IS_PRIVATE(tp))) { // TODO PROTECTED
            return TRUE;
        }
        return FALSE;
    }

    switch(ID_CLASS(id)) {
    case CL_MULTI:
        return FALSE;
        break;
    case CL_VAR:
        if(TYPE_IS_MODIFIED(ID_TYPE(id)))
            return TRUE;
        if(VAR_IS_IMPLIED_DO_DUMMY(id))
            return FALSE;
        if(ID_STORAGE(id) == STG_INDEX) /* Don't declare as a variable */
            return FALSE;
        break;
    case CL_PARAM:
        return TRUE;
    case CL_NAMELIST:
        return TRUE;
    case CL_PROC:
        if(PROC_CLASS(id) == P_DEFINEDPROC) {
            /* this id is of function.
               Checkes if this id is of the current function or not. */
            if (CRT_FUNCEP == PROC_EXT_ID(id)) {
                return TRUE;
            } else if (TYPE_IS_MODIFIED(ID_TYPE(id))) {
                return TRUE;
            } else {
                return FALSE;
            }
        }
        if (EXT_IS_DEFINED_IO(PROC_EXT_ID(id))) {
            return FALSE;
        }
        /* FALL THROUGH */
    default:
        switch(ID_STORAGE(id)) {
        case STG_TAGNAME:
            return TRUE;
        case STG_UNKNOWN:
        case STG_NONE:
            return FALSE;
        case STG_INDEX:
            return FALSE;
        default:
            break;
        }
    }

    return TRUE;
}

/**
 * Check id is visible in <symbols>
 */
static int
id_is_visibleVar_for_symbols(ID id)
{
    if (id == NULL)
        return FALSE;

    if (ID_STORAGE(id) == STG_INDEX)
        return TRUE;

    return (id_is_visibleVar(id) && IS_MODULE(ID_TYPE(id)) == FALSE) ||
            ((ID_STORAGE(id) == STG_ARG ||
              ID_STORAGE(id) == STG_SAVE ||
              (ID_STORAGE(id) == STG_EXT && !EXT_IS_DEFINED_IO(PROC_EXT_ID(id))) ||
              ID_STORAGE(id) == STG_AUTO) && ID_CLASS(id) == CL_PROC);
}


/**
 * output symbols in FfunctionDefinition/FmoduleProcedureDefinition
 */
static void
outx_definition_symbols(int l, EXT_ID ep)
{
    ID id;
    const int l1 = l + 1;

    outx_tag(l, "symbols");

    FOREACH_ID(id, EXT_PROC_ID_LIST(ep)) {
        if (id_is_visibleVar_for_symbols(id))
            outx_id(l1, id);
    }

    /* print common ids */
    FOREACH_ID(id, EXT_PROC_COMMON_ID_LIST(ep)) {
        if (IS_MODULE(ID_TYPE(id)) == FALSE)
            outx_id(l1, id);
    }

    /* print label */
    /* decided not to output labels
    FOREACH_ID(id, EXT_PROC_LABEL_LIST(ep)) {
        outx_id(l1, id);
    }
    */

    outx_close(l, "symbols");
}


/**
 * output FcommonDecl
 */
static void
outx_commonDecl(int l, ID cid)
{
    list lp;
    expv var;
    char buf[256];
    const int l1 = l + 1, l2 = l1 + 1;

    outx_tagOfDecl1(l, xtag(F_COMMON_DECL), ID_LINE(cid));

    if(COM_IS_BLANK_NAME(cid) == FALSE)
        sprintf(buf, " name=\"%s\"", ID_NAME(cid));
    else
        buf[0] = '\0';

    outx_tag(l1, "varList%s", buf);

    FOR_ITEMS_IN_LIST(lp, COM_VARS(cid)) {
        var = LIST_ITEM(lp);
        outx_varRef_EXPR(l2, var);
    }

    outx_close(l1, "varList");
    outx_close(l, xtag(F_COMMON_DECL));
}


/**
 * output FequivalenceDecl
 */
static void
outx_equivalenceDecl(int l, expv v)
{
    const int l1 = l + 1;
    expv v1 = v;

    outx_tagOfDecl1(l, xtag(F_EQUIV_DECL), EXPV_LINE(v));
    outx_varRef_EXPR(l1, EXPR_ARG1(v1));
    outx_varList(l1, EXPR_ARG2(v1), NULL);
    outx_close(l, xtag(F_EQUIV_DECL));
}


static int
qsort_compare_id(const void *v1, const void *v2)
{
    int o1 = ID_ORDER(*(ID*)v1);
    int o2 = ID_ORDER(*(ID*)v2);

    return (o1 == o2) ? 0 : ((o1 < o2) ? -1 : 1);
}


/**
 * sort id by order value
 */
ID*
genSortedIDs(ID ids, int *retnIDs)
{
    ID id, *sortedIDs;
    int i = 0, nIDs = 0;

    if(ids == NULL)
        return NULL;

    FOREACH_ID(id, ids)
        ++nIDs;

    if(nIDs == 0)
        return NULL;

    sortedIDs = (ID*)malloc(nIDs * sizeof(ID));

    FOREACH_ID(id, ids)
        sortedIDs[i++] = id;

    qsort((void*)sortedIDs, nIDs, sizeof(ID), qsort_compare_id);
    *retnIDs = nIDs;

    return sortedIDs;
}


#define IS_NO_PROC_OR_DECLARED_PROC(id) \
    ((ID_CLASS(id) != CL_PROC || \
      (PROC_CLASS(id) == P_EXTERNAL && \
       (TYPE_IS_EXTERNAL(ID_TYPE(id)) || \
        (FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)) != NULL && \
         !TYPE_IS_IMPLICIT(FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(id)))))) || \
      (ID_TYPE(id) && TYPE_IS_EXTERNAL(ID_TYPE(id))) || \
      (ID_TYPE(id) && TYPE_IS_INTRINSIC(ID_TYPE(id))) || \
      PROC_CLASS(id) == P_UNDEFINEDPROC || \
      PROC_CLASS(id) == P_DEFINEDPROC)     \
  && (PROC_EXT_ID(id) == NULL ||           \
      PROC_CLASS(id) == P_UNDEFINEDPROC || \
      PROC_CLASS(id) == P_DEFINEDPROC || \
      (TYPE_IS_PUBLIC(id) || TYPE_IS_PRIVATE(id) || ( \
      EXT_PROC_IS_INTERFACE(PROC_EXT_ID(id)) == FALSE && \
      EXT_PROC_IS_INTERFACE_DEF(PROC_EXT_ID(id)) == FALSE)))    \
  && (ID_TYPE(id) \
      && TYPE_IS_IMPLICIT(id) == FALSE \
      && IS_MODULE(ID_TYPE(id)) == FALSE   \
      && (IS_VOID(ID_TYPE(id)) == FALSE || \
      has_attribute_except_private_public(ID_TYPE(id)))))


static int
is_id_used_in_struct_member(ID id, TYPE_DESC sTp)
{
    /*
     * FIXME:
     *	Actually, it is not checked if id is used in sTp's member.
     *	Instead, just checking line number.
     */
    ID mId;

    if (ID_LINE(id) == NULL) {
        return FALSE;
    }

    FOREACH_MEMBER(mId, sTp) {
        if (ID_LINE(mId) == NULL) {
            continue;
        }
        if (ID_LINE_FILE_ID(id) == ID_LINE_FILE_ID(mId)) {
            if (ID_LINE_NO(id) < ID_LINE_NO(mId)) {
                return TRUE;
            }
        }
    }

    return FALSE;
}


static void
emit_decl(int l, ID id)
{
    if (ID_IS_EMITTED(id) == TRUE) {
        return;
    }
    if (ID_IS_OFMODULE(id) == TRUE && ID_CLASS(id) != CL_PARAM) {
        return;
    }

    if (CRT_FUNCEP != NULL && EXT_PROC_IS_PROCEDUREDECL(CRT_FUNCEP)) {
        /*
         * In the Separate Module Procedure,
         * a function, argments, and a result variable are not decleared
         */
        TYPE_DESC ftp = EXT_PROC_TYPE(CRT_FUNCEP);

        if (ID_CLASS(id) == CL_PROC && PROC_EXT_ID(id) == CRT_FUNCEP) {
            /* id is this procedure */
            return;
        }
        if (ID_SYM(id) == FUNCTION_TYPE_RESULT(ftp)) {
            /* id is this result */
            return;
        }
        if (ID_STORAGE(id) == STG_ARG) {
            /* id is a argument */
            return;
        }
    }

    switch(ID_CLASS(id)) {
    case CL_NAMELIST:
        outx_namelistDecl(l, id);
        break;

    case CL_PARAM:
        if (id_is_visibleVar(id))
            outx_varDecl(l, id);
        break;

    case CL_ENTRY:
        break;

    default:
        switch (ID_STORAGE(id)) {
            case STG_ARG:
            case STG_TYPE_PARAM:
            case STG_SAVE:
            case STG_AUTO:
            case STG_EQUIV:
            case STG_COMEQ:
            case STG_COMMON:
            case STG_INDEX:
                if (id_is_visibleVar(id) &&
                    IS_NO_PROC_OR_DECLARED_PROC(id)) {
                    outx_varDecl(l, id);
                }
                break;

            case STG_TAGNAME:
                break;

            case STG_EXT:
                if (id_is_visibleVar(id) &&
                    IS_NO_PROC_OR_DECLARED_PROC(id)) {
                    outx_varDecl(l, id);
                }
                break;

            case STG_UNKNOWN:
            case STG_NONE:
                break;
        }
        break;
    }
}

static void
outx_id_declarations(int l, ID id_list, int hasResultVar, const char * functionName)
{
    TYPE_DESC tp;
    ID id, *ids;
    int i, nIDs;
    SYMBOL modname = NULL;

    ids = genSortedIDs(id_list, &nIDs);

    if (ids) {
        /*
         * Firstly emit struct base type (TYPE_REF(tp) == NULL).
         * ex) type(x)::p = x(1)
         */
        for (i = 0; i < nIDs; i++) {
            id = ids[i];

            if (ID_CLASS(id) == CL_MODULE) {
                modname = ID_SYM(id);
            }

            if (ID_CLASS(id) != CL_TAGNAME) {
                continue;
            }
            if (ID_IS_EMITTED(id) == TRUE) {
                continue;
            }

            if (ID_IS_OFMODULE(id) == TRUE && ID_MODULE_NAME(id) != modname) {
                continue;
            }

            if (TYPE_IS_MODIFIED(ID_TYPE(id)) == TRUE) {
                continue;
            }

            tp = ID_TYPE(id);
            if (IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL) {
                int j;
                for (j = 0; j < nIDs; j++) {
                    if (i != j) {
                        if (ID_IS_EMITTED(ids[j]) == TRUE) {
                            continue;
                        }
                        if (is_id_used_in_struct_member(ids[j], tp) == TRUE) {
                            emit_decl(l, ids[j]);
                            ID_IS_EMITTED(ids[j]) = TRUE;
                        }
                    }
                }
                outx_structDecl(l, id);
                ID_IS_EMITTED(id) = TRUE;
            }
        }

        /*
         * varDecl except structDecl, namelistDecl
         */
        for (i = 0; i < nIDs; ++i) {
            id = ids[i];

            if (hasResultVar == TRUE && functionName != NULL &&
                strcasecmp(functionName, SYM_NAME(ID_SYM(id))) == 0) {
                continue;
            }

            if (TYPE_IS_MODIFIED(ID_TYPE(id)) == TRUE) {
                continue;
            }

            emit_decl(l, id);
            ID_IS_EMITTED(id) = TRUE;
        }
        free(ids);
    }
}


/**
 * output declarations with pragmas
 */
static void
outx_declarations1(int l, EXT_ID parent_ep, int outputPragmaInBody)
{
    const int l1 = l + 1;
    list lp;
    ID id;
    EXT_ID ep;
    expv v;
    int hasResultVar = (EXT_PROC_RESULTVAR(parent_ep) != NULL) ? TRUE : FALSE;
    const char *myName = SYM_NAME(EXT_SYM(parent_ep));

    outx_tag(l, "declarations");

    /*
     * FuseDecl and FimportDecl
     */
    FOR_ITEMS_IN_LIST(lp, EXT_PROC_BODY(parent_ep)) {
        v = LIST_ITEM(lp);
        switch(EXPV_CODE(v)) {
        case F03_USE_INTRINSIC_STATEMENT:
            outx_useDecl(l1, v, TRUE);
            break;
        case F95_USE_STATEMENT:
            outx_useDecl(l1, v, FALSE);
            break;
        case F95_USE_ONLY_STATEMENT:
            outx_useOnlyDecl(l1, v, FALSE);
            break;
        case F03_USE_ONLY_INTRINSIC_STATEMENT:
            outx_useOnlyDecl(l1, v, TRUE);
            break;            
        case F03_IMPORT_STATEMENT:
            outx_importStatement(l1, v);
            break;
        default:
            break;
        }
    }

    outx_id_declarations(l1, EXT_PROC_ID_LIST(parent_ep), hasResultVar, myName);

    /*
     * FdataDecl / FequivalenceDecl
     */
    FOR_ITEMS_IN_LIST(lp, EXT_PROC_BODY(parent_ep)) {
        v = LIST_ITEM(lp);
        switch(EXPV_CODE(v)) {
        case F_DATA_DECL:
            outx_dataDecl(l1, v);
            break;
        case F_EQUIV_DECL:
            outx_equivalenceDecl(l1, v);
            break;
        default:
            break;
        }
    }

    /*
     * FcommonDecl
     */
    FOREACH_ID(id, EXT_PROC_COMMON_ID_LIST(parent_ep)) {
        outx_commonDecl(l1, id);
    }

    /*
     * FinterfaceDecl
     */
    FOREACH_EXT_ID(ep, EXT_PROC_INTERFACES(parent_ep)) {
        outx_interfaceDecl(l1, ep);
    }

    /*
     * FpragmaStatement
     */
    if(outputPragmaInBody) {
        FOR_ITEMS_IN_LIST(lp, EXT_PROC_BODY(parent_ep)) {
            v = LIST_ITEM(lp);
            switch(EXPV_CODE(v)) {
            case F_PRAGMA_STATEMENT:
                // for FmoduleDefinition-declarations
                outx_pragmaStatement(l1, v);
                break;
	    case XMP_PRAGMA:
		outx_XMP_pragma(l1, v);
		break;
	    case ACC_PRAGMA:
		outx_ACC_pragma(l1, v);
		break;
            default:
                break;
            }
        }
    }

    outx_close(l, "declarations");
}


/**
 * output declarations
 */
static void
outx_declarations(int l, EXT_ID parent_ep)
{
    outx_declarations1(l, parent_ep, FALSE);
}



/**
 * output FmoduleProcedureDecl
 */
static void
outx_moduleProcedureDecl(int l, EXT_ID parent_ep, SYMBOL parentName)
{
    const int l1 = l + 1;
    int hasModProc = FALSE;
    EXT_ID ep;

    FOREACH_EXT_ID(ep, parent_ep) {
        if(EXT_TAG(ep) == STG_EXT &&
            EXT_PROC_IS_MODULE_PROCEDURE(ep)) {
            hasModProc = TRUE;
            break;
        }
    }

    if(hasModProc == FALSE)
        return;

    if (EXT_PROC_IS_MODULE_SPECIFIED(parent_ep)) {
        outx_tagOfDecl1(l, "FmoduleProcedureDecl is_module_specified=\"true\"",
            GET_EXT_LINE(ep));
    } else {
        outx_tagOfDecl1(l, "FmoduleProcedureDecl", GET_EXT_LINE(ep));
    }

    if (is_emitting_xmod() == FALSE) {
        FOREACH_EXT_ID(ep, parent_ep) {
            if (EXT_TAG(ep) == STG_EXT &&
                EXT_PROC_IS_MODULE_PROCEDURE(ep) &&
                !EXT_IS_OFMODULE(ep)) {
                outx_symbolName(l1, EXT_SYM(ep));
            }
        }
    } else {
        if (parentName != NULL) {
            gen_proc_t gp = find_generic_procedure(SYM_NAME(parentName));
            if (gp != NULL) {
                HashTable *tPtr = GEN_PROC_MOD_TABLE(gp);
                if (tPtr != NULL) {
                    HashEntry *hPtr;
                    HashSearch sCtx;
                    mod_proc_t mp;

                    FOREACH_IN_HASH(hPtr, &sCtx, tPtr) {
                        mp = (mod_proc_t)GetHashValue(hPtr);
                        outx_symbolNameWithFunctionType_EXT(l1,
                                                            MOD_PROC_EXT_ID(mp));
                    }
                } else {
                    fatal("invalid generic procedure structure.");
                    /* not reached. */
                    return;
                }
            } else {
                fatal("can't find a generic function '%s'.",
                      SYM_NAME(parentName));
                /* not reached. */
                return;
            }
        } else {
            fatal("module procedure w/o generic name.");
            /* not reached. */
            return;
        }
    }

    outx_close(l, "FmoduleProcedureDecl");
}


/**
 * output FfunctionDecl
 */
static void
outx_functionDecl(int l, EXT_ID ep)
{
    const int l1 = l + 1;
    CRT_FUNCEP_PUSH(ep);
    outx_tagOfDecl1(l, "FfunctionDecl", GET_EXT_LINE(ep));
    outx_symbolNameWithFunctionType_EXT(l1, ep);
    outx_declarations(l1, ep);
    outx_close(l, "FfunctionDecl");
    CRT_FUNCEP_POP;
}


static void
outx_innerDefinitions(int l, EXT_ID extids, SYMBOL parentName, int asDefOrDecl)
{
    EXT_ID ep;

    FOREACH_EXT_ID(ep, extids) {
        if (EXT_TAG(ep) != STG_EXT)
            continue;
        if (EXT_PROC_IS_ENTRY(ep) == TRUE)
            continue;
        if (!EXT_PROC_BODY(ep) && !EXT_PROC_ID_LIST(ep))
            continue;
        if (EXT_PROC_TYPE(ep) != NULL
            && TYPE_BASIC_TYPE(EXT_PROC_TYPE(ep)) == TYPE_MODULE)
            continue;

        if(asDefOrDecl) {
            outx_functionDefinition(l, ep);
        } else {
            if(EXT_PROC_IS_MODULE_PROCEDURE(ep) == FALSE)
                outx_functionDecl(l, ep);
        }
    }

    outx_moduleProcedureDecl(l, extids, parentName);
}


/**
 * output FcontainsStatement
 */
static void
outx_contains(int l, EXT_ID parent)
{
    EXT_ID contains;
    lineno_info *contains_line;

    contains = EXT_PROC_CONT_EXT_SYMS(parent);
    contains_line = EXT_PROC_CONT_EXT_LINE(parent);
    if (contains == NULL) {
        return;
    }
    assert(contains_line != NULL);
    outx_tagOfDecl1(l, "FcontainsStatement", contains_line);
    outx_innerDefinitions(l + 1, contains, NULL, TRUE);
    outx_close(l, "FcontainsStatement");
}


/**
 * output FinterfaceDecl
 */
static void
outx_interfaceDecl(int l, EXT_ID ep)
{
    EXT_ID extids;

    extids = EXT_PROC_INTR_DEF_EXT_IDS(ep);
    if(extids == NULL)
        return;

    if(EXT_IS_OFMODULE(ep) == TRUE)
        return;

    CRT_FUNCEP_PUSH(NULL);
    outx_printi(l, "<FinterfaceDecl");

    switch(EXT_PROC_INTERFACE_CLASS(ep)) {
        case INTF_DECL:
            if(EXT_IS_BLANK_NAME(ep) == FALSE)
                outx_printi(0, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));
            break;
        case INTF_ASSIGNMENT:
            outx_true(TRUE, "is_assignment");
            break;
        case INTF_OPERATOR:
        case INTF_USEROP:
            outx_printi(0, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));
            outx_true(TRUE, "is_operator");
            break;
        case INTF_GENERIC_WRITE_FORMATTED:
            outx_printi(0, " is_defined_io=\"WRITE(FORMATTED)\"");
            break;
        case INTF_GENERIC_WRITE_UNFORMATTED:
            outx_printi(0, " is_defined_io=\"WRITE(UNFORMATTED)\"");
            break;
        case INTF_GENERIC_READ_FORMATTED:
            outx_printi(0, " is_defined_io=\"READ(FORMATTED)\"");
            break;
        case INTF_GENERIC_READ_UNFORMATTED:
            outx_printi(0, " is_defined_io=\"READ(UNFORMATTED)\"");
            break;
        case INTF_ABSTRACT:
            outx_true(TRUE, "is_abstract");
            break;
        default:
            /* never reach. here*/
            break;
    }

    outx_lineno(EXT_LINE(ep));
    outx_printi(0,">\n");
    outx_innerDefinitions(l + 1, extids, EXT_SYM(ep), FALSE);
    outx_close(l, "FinterfaceDecl");
    CRT_FUNCEP_POP;
}


/**
 * output FfunctionDefinition
 */
static void
outx_functionDefinition(int l, EXT_ID ep)
{
    const int l1 = l + 1, l2 = l + 2;

    const char *tag = NULL;

    if (!EXT_PROC_IS_PROCEDUREDECL(ep)) {
        tag = "FfunctionDefinition";
    } else {
        tag = "FmoduleProcedureDefinition";
    }


    CRT_FUNCEP_PUSH(ep);

    outx_tagOfDecl1(l, tag, GET_EXT_LINE(ep));
    outx_symbolNameWithFunctionType_EXT(l1, ep);
    outx_definition_symbols(l1, ep);
    outx_declarations(l1, ep);
    outx_tag(l1, "body");
    outx_expv(l2, EXT_PROC_BODY(ep));
    outx_contains(l2, ep);
    outx_close(l1, "body");
    outx_close(l, tag);

    CRT_FUNCEP_POP;
}


/**
 * output FmoduleDefinition
 */
static void
outx_moduleDefinition(int l, EXT_ID ep)
{
    const int l1 = l + 1;

    is_outputed_module = TRUE;
    CRT_FUNCEP = NULL;

    outx_tagOfDeclNoClose(l, "%s name=\"%s\"", GET_EXT_LINE(ep),
                          "FmoduleDefinition", SYM_NAME(EXT_SYM(ep)));

    if (EXT_MODULE_IS_SUBMODULE(ep)) {
        outx_true(TRUE, "is_sub");
        if (EXT_MODULE_ANCESTOR(ep)) {
            outx_print(" parent_name=\"%s:%s\"",
                       SYM_NAME(EXT_MODULE_ANCESTOR(ep)),
                       SYM_NAME(EXT_MODULE_PARENT(ep)));
        } else {
            outx_print(" parent_name=\"%s\"",
                       SYM_NAME(EXT_MODULE_PARENT(ep)));
        }
    }
    outx_puts(">\n");

    outx_definition_symbols(l1, ep);
    outx_declarations1(l1, ep, TRUE); // output with pragma
    outx_contains(l1, ep);
    outx_close(l, "FmoduleDefinition");

    is_outputed_module = FALSE;
}


/**
 * output FblockDataDefinition
 */
static void
outx_blockDataDefinition(int l, EXT_ID ep)
{
    const int l1 = l + 1;
    char buf[256];

    if(EXT_IS_BLANK_NAME(ep))
        buf[0] = '\0';
    else
        sprintf(buf, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));

    outx_tagOfDecl1(l, "%s%s", GET_EXT_LINE(ep),
                    "FblockDataDefinition", buf);
    outx_definition_symbols(l1, ep);
    outx_declarations(l1, ep);
    outx_close(l, "FblockDataDefinition");
}


static const char*
getTimestamp()
{
    const time_t t = time(NULL);
    struct tm *ltm = localtime(&t);
    strftime(s_timestamp, CEXPR_OPTVAL_CHARLEN, "%F %T", ltm);
    return s_timestamp;
}


/**
 * recursively collect TYPE_DESC from block constructs
 */
static void
collect_types_from_block(BLOCK_ENV block)
{
    BLOCK_ENV bp;
    TYPE_DESC tp;

    if (block == NULL) {
        return;
    }

    FOREACH_BLOCKS(bp, block) {
        mark_type_desc_in_id_list(BLOCK_LOCAL_SYMBOLS(bp));
        collect_types_inner(BLOCK_LOCAL_EXTERNAL_SYMBOLS(bp));
        collect_types_inner(BLOCK_LOCAL_INTERFACES(bp));
        for(tp = BLOCK_LOCAL_STRUCT_DECLS(bp); tp != NULL; tp = TYPE_SLINK(tp)) {
            if(TYPE_IS_DECLARED(tp)) {
                mark_type_desc(tp);
                mark_type_desc_in_structure(tp);
            }
        }
        collect_types_from_block(BLOCK_CHILDREN(bp));
    }
}


/**
 * recursively collect TYPE_DESC to type_list
 */
static void
collect_types1(EXT_ID extid)
{
    EXT_ID ep;
    TYPE_DESC tp, sTp;

    if(extid == NULL)
        return;

    /* collect used types */
    FOREACH_EXT_ID(ep, extid) {
        if((EXT_TAG(ep) != STG_EXT &&
            EXT_TAG(ep) != STG_COMMON) ||
            EXT_IS_DEFINED(ep) == FALSE) {
            /* STG_EXT
             *      is program, subroutine, function, module, interface.
             * STG_COMMON
             *      is block data.
             * EXT_IS_DEFINED
             *      is FALSE when the id is not declared
             *      but called as function or subroutine.
             */
            continue;
        }

        if (EXT_TAG(ep) == STG_EXT &&
            IS_MODULE(EXT_PROC_TYPE(ep)) == FALSE &&
            EXT_PROC_IS_INTERFACE_DEF(ep) == FALSE &&
            (EXT_PROC_IS_INTERFACE(ep) == FALSE ||
            EXT_IS_BLANK_NAME(ep) == FALSE)) {
            add_type_ext_id(ep);
        }

        /* symbols in CONTAINS */
        collect_types1(EXT_PROC_CONT_EXT_SYMS(ep));
        /* INTERFACE symbols */
        collect_types1(EXT_PROC_INTERFACES(ep));
        /* symbols in INTERFACE */
        collect_types1(EXT_PROC_INTR_DEF_EXT_IDS(ep));

        /* symbols in BLOCK */
        collect_types_from_block(EXT_PROC_BLOCKS(ep));

        sTp = reduce_type(EXT_PROC_TYPE(ep));
        mark_type_desc(sTp);
        EXT_PROC_TYPE(ep) = sTp;
        collect_type_desc(EXT_PROC_ARGS(ep));
        mark_type_desc_in_id_list(EXT_PROC_ID_LIST(ep));
        collect_type_desc(EXT_PROC_BODY(ep));

        for(tp = EXT_PROC_STRUCT_DECLS(ep); tp != NULL; tp = TYPE_SLINK(tp)) {
            if(TYPE_IS_DECLARED(tp)) {
                mark_type_desc(tp);
                mark_type_desc_in_structure(tp);
            }
        }
    }
}


/**
 * recursively collect TYPE_DESC to type_list
 */
static void
collect_types(EXT_ID extid)
{
    TYPE_EXT_ID te;
    TYPE_DESC tp, tq;
    TYPE_DESC sTp;

    collect_types1(extid);
    FOREACH_TYPE_EXT_ID(te, type_ext_id_list) {
        TYPE_DESC tp = EXT_PROC_TYPE(te->ep);
        if (tp && EXT_TAG(te->ep) == STG_EXT) {
            sTp = reduce_type(EXT_PROC_TYPE(te->ep));
            mark_type_desc(sTp);
            EXT_PROC_TYPE(te->ep) = sTp;
        }
    }

    /*
     * now mark type-bound procedures
     */
    for (tp = tbp_list; tp != NULL; tp = tq){
        tq = TYPE_LINK(tp);
        TYPE_LINK(tp) = NULL;
        TYPE_IS_REFERENCED(tp) = FALSE;
        mark_type_desc_skip_tbp(tp, FALSE);
    }

}


/**
 * recursively collect TYPE_DESC to type_list
 */
static void
collect_types_inner(EXT_ID extid)
{
    TYPE_EXT_ID te;
    TYPE_DESC sTp;

    collect_types1(extid);
    FOREACH_TYPE_EXT_ID(te, type_ext_id_list) {
        TYPE_DESC tp = EXT_PROC_TYPE(te->ep);
        if (tp && EXT_TAG(te->ep) == STG_EXT) {
            sTp = reduce_type(EXT_PROC_TYPE(te->ep));
            mark_type_desc(sTp);
            EXT_PROC_TYPE(te->ep) = sTp;
        }
    }

}


/**
 * output typeTable
 */
static void
outx_typeTable(int l)
{
    const int l1 = l + 1;
    TYPE_DESC tp;

    outx_tag(l, "typeTable");

    for (tp = type_list; tp != NULL; tp = TYPE_LINK(tp)){
        outx_type(l1, tp);
    }

    outx_close(l, "typeTable");
}


/**
 * output globalSymbols
 */
static void
outx_globalSymbols(int l)
{
    const int l1 = l + 1;
    EXT_ID ep;

    outx_tag(l, "globalSymbols");
    FOREACH_EXT_ID(ep, EXTERNAL_SYMBOLS) {
        if (EXT_IS_DUMMY(ep) || EXT_IS_BLANK_NAME(ep)) {
            continue;
        }
        outx_ext_id(l1, ep);
    }
    outx_close(l, "globalSymbols");
}


/**
 * output globalDeclarations
 */
static void
outx_globalDeclarations(int l)
{
    const int l1 = l + 1;
    EXT_ID ep;

    outx_tag(l, "globalDeclarations");
    FOREACH_EXT_ID(ep, EXTERNAL_SYMBOLS) {
        switch(EXT_TAG(ep)) {
        case STG_COMMON:
            outx_blockDataDefinition(l1, ep);
            break;
        case STG_EXT:
            if ((EXT_PROC_IS_ENTRY(ep) == FALSE) &&
                (EXT_PROC_BODY(ep) || EXT_PROC_ID_LIST(ep))) {

                if(EXT_PROC_TYPE(ep) != NULL &&
                    TYPE_BASIC_TYPE(EXT_PROC_TYPE(ep)) == TYPE_MODULE)
                    outx_moduleDefinition(l1, ep);
                else
                    outx_functionDefinition(l1, ep);
            }
            break;
        default:
            break;
        }
    }
    outx_close(l, "globalDeclarations");
}


/**
 * output XcodeML
 */
void
output_XcodeML_file()
{
    if(flag_module_compile)
        return; // DO NOTHING

    type_list = NULL;

    type_module_proc_list = NULL;
    type_module_proc_last = NULL;
    type_ext_id_list = NULL;
    type_ext_id_last = NULL;
    tbp_list = NULL;
    tbp_list_tail = NULL;

    collect_types(EXTERNAL_SYMBOLS);
    CRT_FUNCEP = NULL;

    print_fp = output_file;
    const int l = 0, l1 = l + 1;

    outx_printi(l,
        "<XcodeProgram source=\"%s\"\n"
        "              language=\"%s\"\n"
        "              time=\"%s\"\n"
        "              compiler-info=\"%s\"\n"
        "              version=\"%s\">\n",
        getXmlEscapedStr((source_file_name) ? source_file_name : "<stdin>"),
        F_TARGET_LANG,
        getTimestamp(),
        F_FRONTEND_NAME, F_FRONTEND_VER);

    outx_typeTable(l1);
    outx_globalSymbols(l1);
    outx_globalDeclarations(l1);

    outx_close(l, "XcodeProgram");
}

/*
 * functions defined below is those related to xmod(modules).
 */

/**
 * output <id> node
 */
static void
outx_id_mod(int l, ID id)
{
    if(ID_STORAGE(id) == STG_EXT && PROC_EXT_ID(id) == NULL) {
        fatal("outx_id: PROC_EXT_ID is NULL: symbol=%s", ID_NAME(id));
    }

    if ((ID_CLASS(id) == CL_PROC || ID_CLASS(id) == CL_ENTRY) &&
        PROC_EXT_ID(id)) {
        outx_typeAttrOnly_functionType(l, ID_TYPE(id), "id");
    } else {
        outx_typeAttrOnly_ID(l, id, "id");
    }

    const char *sclass = get_sclass(id);

    outx_print(" sclass=\"%s\"", sclass);
    outx_print(" original_name=\"%s\"", SYM_NAME(id->use_assoc->original_name));
    outx_print(" declared_in=\"%s\"", SYM_NAME(id->use_assoc->module_name));
    if(ID_IS_AMBIGUOUS(id))
        outx_print(" is_ambiguous=\"true\"");
    outx_print(">\n", SYM_NAME(id->use_assoc->module_name));
    outx_symbolName(l + 1, ID_SYM(id));
    outx_close(l, "id");
}

/**
 * output <identifiers> node
 */
static void
outx_identifiers(int l, ID ids)
{
    const int l1 = l + 1;
    ID id;

    outx_tag(l, "identifiers");

    FOREACH_ID(id, ids) {
        outx_id_mod(l1, id);
    }

    outx_close(l, "identifiers");
}

/**
 * output declaraions for .xmod file
 */
static void
outx_module_declarations(int l, ID ids)
{
    const int l1 = l + 1;
    ID id;

    outx_tag(l, "declarations");

    FOREACH_ID(id, ids) {
      switch(ID_CLASS(id)) {
	/* only value PARAM value is exported from module ??? */
      case CL_PARAM:
        if (id_is_visibleVar(id))
	  outx_varDecl(l1, id);
        break;
      default:
	break;
      }
    }

    outx_close(l, "declarations");
}

/**
 * output <interfaceDecls> node
 */
static void
outx_interfaceDecls(int l, ID ids)
{
    const int l1 = l + 1;
    ID id;
    EXT_ID ep;

    outx_tag(l, "interfaceDecls");

    FOREACH_ID(id, ids) {
        ep = PROC_EXT_ID(id);
        if (ep != NULL) {
            outx_interfaceDecl(l1, ep);
        }
    }

    outx_close(l, "interfaceDecls");
}

/**
 * output <XcalablempFortranModule> node
 */
static void
outx_module(struct module * mod)
{
    const int l = 0, l1 = l + 1, l2 = l1 + 1;
    struct depend_module * mp;
    outx_tag(l, "OmniFortranModule version=\"%s\"", F_MODULE_VER);

    outx_printi(l1, "<name>%s</name>\n", SYM_NAME(mod->name));

    mp = mod->depend.head;

    outx_tag(l1, "depends");

    while (mp != NULL) {
        outx_printi(l2, "<name>%s</name>\n", SYM_NAME(mp->module_name));
        mp = mp->next;
    }

    outx_close(l1, "depends");

    outx_typeTable(l1);

    outx_identifiers(l1, mod->head);

    outx_module_declarations(l1,mod->head);

    outx_interfaceDecls(l1, mod->head);

    /* output pragmas etc in CURRENT_STATEMENTS */
    outx_tag(l1,"aux_info");
    if(CURRENT_STATEMENTS != NULL){
	list lp;
	expv v;
	FOR_ITEMS_IN_LIST(lp,CURRENT_STATEMENTS){
            v = LIST_ITEM(lp);
	    outx_expv(l1+1,v);
	}
    }
    outx_close(l1,"aux_info");

    outx_close(l, "OmniFortranModule");
}

static void
unmark_ids_in_struct(TYPE_DESC tp) {
    if (tp == NULL) {
        return;
    }

    if (IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL) {
        ID member;
        FOREACH_MEMBER(member, tp) {
            ID_IS_EMITTED(member) = FALSE;
        }
        if (TYPE_PARENT(tp)) {
            unmark_ids_in_struct(TYPE_PARENT_TYPE(tp));
        }
    }
}

/**
 * unmark id in proc
 */
void
unmark_ids(EXT_ID ep)
{
    ID id;
    EXT_ID interface, sub_program, external_proc;

    FOREACH_ID(id, EXT_PROC_ID_LIST(ep)) {
        TYPE_DESC tp;
        ID_IS_EMITTED(id) = FALSE;

        tp = ID_TYPE(id);
        if (IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL) {
            unmark_ids_in_struct(tp);
        }
        ID_IS_EMITTED(id) = FALSE;
    }

    // recursive apply
    FOREACH_EXT_ID(interface, EXT_PROC_INTERFACES(ep)) {
        unmark_ids(interface);
    }

    FOREACH_EXT_ID(sub_program, EXT_PROC_CONT_EXT_SYMS(ep)) {
        unmark_ids(sub_program);
    }

    FOREACH_EXT_ID(external_proc, EXT_PROC_INTR_DEF_EXT_IDS(ep)) {
        unmark_ids(external_proc);
    }
}


/**
 * output module to .xmod file
 */
int
output_module_file(struct module * mod, const char * filename)
{
    ID id;
    EXT_ID ep;
    TYPE_EXT_ID te;
    TYPE_DESC sTp;
    TYPE_DESC tp;
    TYPE_DESC tq;
    int oEmitMode;
    expr modTypeList;
    list lp;
    expv v;

    if (flag_module_compile) {
        print_fp = stdout;
    } else {
        if ((print_fp = fopen(filename, "w")) == NULL) {
            fatal("could'nt open module file to write.");
            return FALSE;
        }
    }

    is_emitting_for_submodule = MODULE_IS_FOR_SUBMODULE(mod);

    oEmitMode = is_emitting_xmod();
    set_module_emission_mode(TRUE);

    type_list = NULL;

    type_module_proc_list = NULL;
    type_module_proc_last = NULL;
    type_ext_id_list = NULL;
    type_ext_id_last = NULL;
    tbp_list = NULL;
    tbp_list_tail = NULL;

    /*
     * collect types used in this module
     */
    FOREACH_ID(id, mod->head) {
        mark_type_desc_id(id);

        ep = PROC_EXT_ID(id);
        // if id is external,  ...
        if (ep != NULL) {
            collect_types1(ep);
            FOREACH_TYPE_EXT_ID(te, type_ext_id_list) {
                TYPE_DESC tp = EXT_PROC_TYPE(te->ep);
                if (tp && EXT_TAG(te->ep) == STG_EXT) {
                    sTp = reduce_type(EXT_PROC_TYPE(te->ep));
                    mark_type_desc(sTp);
                    EXT_PROC_TYPE(te->ep) = sTp;
                }
            }
        }
    }

    modTypeList = collect_all_module_procedures_types();
    FOR_ITEMS_IN_LIST(lp, modTypeList) {
        v = LIST_ITEM(lp);
        if (EXPV_INT_VALUE(EXPR_ARG1(v)) == 1) {
            if (EXPR_ARG3(v) != NULL) {
                ep = EXPV_ANY(EXT_ID, EXPR_ARG3(v));
                collect_types1(ep);
                sTp = reduce_type(EXT_PROC_TYPE(ep));
                mark_type_desc(sTp);
                EXT_PROC_TYPE(ep) = sTp;
            }
        }
    }

    /*
     * now mark type-bound procedures
     */
    for (tp = tbp_list; tp != NULL; tp = tq){
        tq = TYPE_LINK(tp);
        TYPE_LINK(tp) = NULL;
        TYPE_IS_REFERENCED(tp) = FALSE;
        mark_type_desc_skip_tbp(tp, FALSE);
    }

    outx_module(mod);

    unmark_type_table(); // unmark types collected
    unmark_ids(UNIT_CTL_CURRENT_EXT_ID(CURRENT_UNIT_CTL));

    set_module_emission_mode(oEmitMode);

    if(!flag_module_compile) {
        fclose(print_fp);
    }

    is_emitting_for_submodule = FALSE;

    return TRUE;
}


/**
 * Fix type of forward-referenced function calls to actual type if possible.
 */
static void
fixup_function_call(expv v) {
    if (EXPR_CODE(v) == FUNCTION_CALL) {
        int n = expr_list_length(v);
        ID fid = (n >= 3 && EXPR_ARG3(v) != NULL) ?
            EXPV_ANY(ID, EXPR_ARG3(v)) : NULL;
        if (fid != NULL) {
            EXT_ID eid = PROC_EXT_ID(fid);
            TYPE_DESC tp = (eid != NULL) ? EXT_PROC_TYPE(eid) : NULL;
            if (tp != NULL) {
                if (EXPV_NEED_TYPE_FIXUP(v) == TRUE) {
                    ID_TYPE(fid) = tp;
                    EXPV_TYPE(EXPR_ARG1(v)) = tp;
                    EXPV_TYPE(v) = FUNCTION_TYPE_RETURN_TYPE(tp);
                }
            } else {
                if (!ID_IS_DUMMY_ARG(fid) &&
                    !(ID_TYPE(fid) != NULL &&
                      IS_PROCEDURE_TYPE(ID_TYPE(fid)))) {
                    error_at_node(v, "undefined function/subroutine: '%s'.",
                                  ID_NAME(fid));
                }
            }
        }
    }
    if (!(EXPR_CODE_IS_TERMINAL(EXPR_CODE(v)))) {
        list lp;
        expv vv;
        FOR_ITEMS_IN_LIST(lp, v) {
            vv = LIST_ITEM(lp);
            if (vv != NULL) {
                fixup_function_call(vv);
            }
        }
    }
}


void
final_fixup() {
    EXT_ID ep;
    expv v;

    FOREACH_EXT_ID(ep, EXTERNAL_SYMBOLS) {
        if (EXT_TAG(ep) == STG_EXT &&
            (v = EXT_PROC_BODY(ep)) != NULL) {
            fixup_function_call(v);
        }
    }
}

