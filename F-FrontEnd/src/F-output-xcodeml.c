/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-output-xcodeml.c
 */

#include "F-front.h"
#include "F-output-xcodeml.h"

#define CHAR_BUF_SIZE 8192

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

extern int      flag_module_compile;

static void     outx_expv(int l, expv v);
static void     outx_functionDefinition(int l, EXT_ID ep);
static void     outx_interfaceDecl(int l, EXT_ID ep);

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
static TYPE_EXT_ID  type_ext_id_list = NULL, type_ext_id_last = NULL;
static FILE         *print_fp;
static char         s_charBuf[CHAR_BUF_SIZE];
static int          is_outputed_module = FALSE;

#define GET_EXT_LINE(ep) \
    (EXT_LINE(ep) ? EXT_LINE(ep) : \
    EXT_PROC_ID_LIST(ep) ? ID_LINE(EXT_PROC_ID_LIST(ep)) : NULL)

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
    case F_CASELABEL_STATEMENT:     return "FcaseLabel";
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
                                
    /*                          
     * misc.                    
     */                         
    case F_IMPLIED_DO:              return "FdoLoop";
    case F_INDEX_RANGE:             return "indexRange";

    /*                          
     * module.                    
     */                         
    case F95_USE_STATEMENT:         return "FuseDecl";
    case F95_USE_ONLY_STATEMENT:    return "FuseOnlyDecl";

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
    case EXPR_CODE_END:

        fatal("invalid exprcode : %s", EXPR_CODE_NAME(code));
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
        TYPE_IS_PARAMETER(tp) ||
        TYPE_IS_ALLOCATABLE(tp) ||
        TYPE_IS_OPTIONAL(tp) ||
        TYPE_IS_POINTER(tp) ||
        TYPE_IS_SAVE(tp) ||
        TYPE_IS_TARGET(tp) ||
        TYPE_IS_PUBLIC(tp) ||
        TYPE_IS_PRIVATE(tp) ||
        TYPE_IS_SEQUENCE(tp) ||
        TYPE_IS_INTERNAL_PRIVATE(tp) ||
        TYPE_IS_INTENT_IN(tp) ||
        TYPE_IS_INTENT_OUT(tp) ||
        TYPE_IS_INTENT_INOUT(tp) ||
        tp->codims;
}


static int
has_attribute(TYPE_DESC tp)
{
    return
        has_attribute_except_func_attrs(tp) ||
        TYPE_IS_EXTERNAL(tp) ||
        TYPE_IS_INTRINSIC(tp) ||
        TYPE_IS_RECURSIVE(tp);
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
    case TYPE_GNUMERIC_ALL: tid = "FnumericAll"; break;
    case TYPE_SUBR:         /* fall through */
    case TYPE_MODULE:       tid = "Fvoid"; break;
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
      IS_SUBR(tp) ||                            \
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
        case TYPE_GNUMERIC_ALL: pfx = 'V'; break;
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

    outx_printi(l,"<%s type=\"%s\"", tag, getTypeID(tp));

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

#if 0
        /*
         * FIXME:
         *	Actually we want this assertions.
         */
        assert(TYPE_IS_RECURSIVE(tp) == FALSE);
        assert(TYPE_IS_EXTERNAL(tp) == FALSE);
        assert(TYPE_IS_INTRINSIC(tp) == FALSE);
#endif

        outx_true(TYPE_IS_PUBLIC(tp),           "is_public");
        outx_true(TYPE_IS_PRIVATE(tp),          "is_private");
        outx_true(TYPE_IS_POINTER(tp),          "is_pointer");
        outx_true(TYPE_IS_TARGET(tp),           "is_target");
        outx_true(TYPE_IS_OPTIONAL(tp),         "is_optional");
        outx_true(TYPE_IS_SAVE(tp),             "is_save");
        outx_true(TYPE_IS_PARAMETER(tp),        "is_parameter");
        outx_true(TYPE_IS_ALLOCATABLE(tp),      "is_allocatable");
        outx_true(TYPE_IS_SEQUENCE(tp),         "is_sequence");
        outx_true(TYPE_IS_INTERNAL_PRIVATE(tp), "is_internal_private");
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
outx_typeAttrOnly_functionType(int l, EXT_ID ep, const char *tag)
{
    char *tid = genFunctionTypeID(ep);
    outx_printi(l,"<%s type=\"%s\"", tag, tid);
}


static void
outx_typeAttrOnly_functionTypeWithResultVar(
    int l, EXT_ID ep, const char *tag)
{
    outx_typeAttrOnly_functionType(l, ep, tag);
    if (EXT_PROC_RESULTVAR(ep) != NULL) {
        expv res = EXT_PROC_RESULTVAR(ep);
        outx_print(" result_name=\"%s\"",
                   SYM_NAME(EXPV_NAME(res)));
    }
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
outx_symbolNameWithFunctionType(int l, EXT_ID ep)
{
    outx_typeAttrOnly_functionType(l, ep, "name");
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
 * output an expr as name with type
 */
static void
outx_expvNameWithType(int l, expv v)
{
    if(EXPV_PROC_EXT_ID(v)) {
        // for high order function
        outx_typeAttrOnly_functionType(l, EXPV_PROC_EXT_ID(v), "name");
    } else {
        outx_typeAttrs(l, EXPV_TYPE(v), "name", TOPT_TYPEONLY);
    }
    outx_print(">%s</name>\n", SYM_NAME(EXPV_NAME(v)));
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
        outx_typeAttrOnly_functionType(l, ep, "id");
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
    if(ID_STORAGE(id) == STG_EXT && PROC_EXT_ID(id) == NULL) {
        fatal("outx_id: PROC_EXT_ID is NULL: symbol=%s", ID_NAME(id));
    }

    if(ID_CLASS(id) == CL_PROC && PROC_EXT_ID(id)) {
        outx_typeAttrOnly_functionType(l, PROC_EXT_ID(id), "id");
    } else {
        outx_typeAttrOnly_ID(l, id, "id");
    }

    const char *sclass = get_sclass(id);

    outx_print(" sclass=\"%s\">\n", sclass);
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

    outx_tagOfDecl(l, "varDecl", id);

    if(PROC_EXT_ID(id)) {
        /* high order func */
        outx_symbolNameWithFunctionType(l1, PROC_EXT_ID(id));
    } else {
        outx_symbolNameWithType_ID(l1, id);
        outx_value(l1, VAR_INIT_VALUE(id));
    }

    outx_close(l, "varDecl");
}


static void
outx_functionCall0(int l, expv v)
{
    const int l1 = l + 1, l2 = l1 + 1;
    list lp;
    expv arg, v2;
    int opt = 0;
    int isIntrinsic = (SYM_TYPE(EXPV_NAME(EXPR_ARG1(v))) == S_INTR);

    if (isIntrinsic)
        opt |= TOPT_INTRINSIC;
    outx_tagOfExpression1(l, v, opt);

    if (isIntrinsic)
        outx_expvName(l1, EXPR_ARG1(v));
    else {
        assert(EXPR_ARG3(v));
        outx_symbolNameWithFunctionType(
            l1, PROC_EXT_ID(EXPV_ANY(ID, EXPR_ARG3(v))));
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
    if(IS_SUBR(EXPV_TYPE(v)) || IS_SUBR(TYPE_REF(EXPV_TYPE(v))))
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
outx_structConstructor(int l, expv v)
{
    const int l1 = l + 1;
    outx_tagOfExpression(l, v);
    outx_expv(l1, EXPR_ARG1(v));
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
        outx_tag(l1, "else");
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
    static char buf[128];

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
        EXPV_CODE(vPointer) != F95_MEMBER_REF) {
        fatal("%s: Invalid argument, expected F_VAR or F95_MEMBER_REF.", __func__);
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
    outx_symbolNameWithFunctionType(l + 1, ep);
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
    case XMP_COARRAY_REF:
      outx_varRef_EXPR(l1, EXPR_ARG1(v));
      outx_printi(l1, "<coShape>\n"); 
      FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)){
	expr cobound = LIST_ITEM(lp);
	outx_indexRange0(l+2, ASSUMED_NONE, ASSUMED_SIZE,
			 EXPR_ARG1(cobound), EXPR_ARG2(cobound), EXPR_ARG3(cobound));

      }
      outx_close(l1, "coShape");
      break;
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
    FOR_ITEMS_IN_LIST(lp, v)
        outx_alloc(l, LIST_ITEM(lp));
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


/**
 * output FallocateStatement/FdeallocateStatement
 */
static void
outx_ALLOCDEALLOC_statement(int l, expv v)
{
    const int l1 = l + 1;
    char buf[128];
    expv vstat = expr_list_get_n(v, 1);

    if(vstat)
      switch (EXPR_CODE(vstat)){
      case F_VAR:
	sprintf(buf, " stat_name=\"%s\"", SYM_NAME(EXPV_NAME(vstat)));
	break;
      case ARRAY_REF:
        //error_at_node(v, "cannot use array ref. in stat specifier");
	warning("cannot use array ref. in stat specifier");
        buf[0] = '\0';
	break;
	//        exit(1);
      case F95_MEMBER_REF:
        //error_at_node(v, "cannot use member ref. in stat specifier");
	warning("cannot use member ref. in stat specifier");
        buf[0] = '\0';
        //exit(1);
	break;
      default:
        buf[0] = '\0';
	break;
      }
    else
        buf[0] = '\0';

    outx_tagOfStatement1(l, v, buf);
    outx_allocList(l1, EXPR_ARG1(v));
    outx_expvClose(l, v);
}


static const char*
getKindParameter(TYPE_DESC tp)
{
    static char buf[256];
    expv v = TYPE_KIND(tp);

    if(IS_DOUBLED_TYPE(tp)) {
        sprintf(buf, "%d", KIND_PARAM_DOUBLE);
    } else if(v) {
        strcpy(buf, getRawString(v));
    } else {
        return NULL;
    }

    return buf;
}


static TYPE_DESC
getLargeIntType()
{
    static TYPE_DESC tp = NULL;
    if(tp) return tp;

    tp = type_basic(TYPE_INT);
    TYPE_KIND(tp) = expv_int_term(INT_CONSTANT, type_INT, 8);

    return tp;
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
            if(n > INT_MAX)
                tp = getLargeIntType();
            sprintf(buf, OMLL_DFMT, n);
        }
        if(tp == NULL)
            tp = type_INT;
        tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
        goto print_constant;

    case FLOAT_CONSTANT:
        if(EXPV_ORIGINAL_TOKEN(v))
            strcpy(buf, EXPV_ORIGINAL_TOKEN(v));
        else
            sprintf(buf, "%Lf", EXPV_FLOAT_VALUE(v));
        if(tp == NULL)
            tp = type_REAL;
        tid = getBasicTypeID(TYPE_BASIC_TYPE(tp));
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
    if(EXPR_ARG3(v) != NULL)
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
outx_useDecl(int l, expv v)
{
    list lp;
    const char *mod_name = SYM_NAME(EXPV_NAME(EXPR_ARG1(v)));

    outx_tagOfDecl1(l, "%s name=\"%s\"", EXPR_LINE(v), XTAG(v), mod_name);

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)) {
        expv x = LIST_ITEM(lp);
        outx_useRename(l+1, EXPR_ARG1(x), EXPR_ARG2(x));
    }

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
outx_useOnlyDecl(int l, expv v)
{
    list lp;
    const char *mod_name = SYM_NAME(EXPV_NAME(EXPR_ARG1(v)));

    outx_tagOfDecl1(l, "%s name=\"%s\"", EXPR_LINE(v), XTAG(v), mod_name);

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(v)) {
        expv x = LIST_ITEM(lp);
        outx_useRenamable(l+1, EXPR_ARG1(x), EXPR_ARG2(x));
    }

    outx_expvClose(l, v);
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
    case IF_STATEMENT:
    case F_WHERE_STATEMENT:         outx_IFWHERE_Statement(l, v); break;
    case F_RETURN_STATEMENT:        outx_returnStatement(l, v); break;
    case F_CONTINUE_STATEMENT:      outx_continueStatement(l, v); break;
    case GOTO_STATEMENT:            outx_gotoStatement(l, v); break;
    case F_COMPGOTO_STATEMENT:      outx_compgotoStatement(l, v); break;
    case STATEMENT_LABEL:           outx_labeledStatement(l, v); break;
    case F_CASELABEL_STATEMENT:     outx_caseLabel(l, v); break;
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
        break;

    /*
     * child elements
     */
    case LIST: {
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
    case EXPR_CODE_END:

        if(debug_flag)
            expv_output(v, stderr);
        fatal("invalid exprcode : %s", EXPR_CODE_NAME(code));
        abort();
    }
}

static void mark_type_desc_in_structure(TYPE_DESC tp);
//static void check_type_desc(TYPE_DESC tp);

static void
mark_type_desc(TYPE_DESC tp)
{
  if (tp == NULL || TYPE_IS_REFERENCED(tp) || IS_MODULE(tp))
    return;

  TYPE_IS_REFERENCED(tp) = 1;

  if (TYPE_REF(tp)){

    if (IS_ARRAY_TYPE(tp)){

      mark_type_desc(array_element_type(tp));

    }
    else {

      mark_type_desc(TYPE_REF(tp));

    }

/*     if (IS_ARRAY_TYPE(tp)) */
/*       mark_type_desc(array_element_type(tp)); */
/*     else */
/*       mark_type_desc(TYPE_REF(tp)); */
  }

  TYPE_LINK_ADD(tp, type_list, type_list_tail);
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
    TYPE_DESC itp;

    FOREACH_MEMBER(id, tp) {
        itp = ID_TYPE(id);
        mark_type_desc(itp);
        if(IS_STRUCT_TYPE(itp))
            mark_type_desc_in_structure(itp);
    }
}


static void
collect_type_desc(expv v)
{
    list lp;

    if(v == NULL) return;
    mark_type_desc(EXPV_TYPE(v));
    if(EXPR_CODE_IS_TERMINAL(EXPV_CODE(v))) return;

    FOR_ITEMS_IN_LIST(lp, v)
        collect_type_desc(LIST_ITEM(lp));
}


void
add_type_ext_id(EXT_ID ep)
{
    TYPE_EXT_ID te = (TYPE_EXT_ID)malloc(sizeof(struct type_ext_id));
    bzero(te, sizeof(struct type_ext_id));
    te->ep = ep;
    FUNC_EXT_LINK_ADD(te, type_ext_id_list, type_ext_id_last);
}


static void
mark_type_desc_in_id_list(ID ids)
{
    ID id;
    FOREACH_ID(id, ids) {
        mark_type_desc(ID_TYPE(id));
        collect_type_desc(ID_ADDR(id));
        switch(ID_CLASS(id)) {
        case CL_PARAM:
            collect_type_desc(VAR_INIT_VALUE(id));
            break;
        case CL_VAR:
            collect_type_desc(VAR_INIT_VALUE(id));
            /* fall through */
        case CL_PROC:
            if(PROC_EXT_ID(id) && EXT_TAG(PROC_EXT_ID(id)) != STG_UNKNOWN &&
                (PROC_CLASS(id) == P_INTRINSIC ||
                 PROC_CLASS(id) == P_EXTERNAL ||
                 PROC_CLASS(id) == P_DEFINEDPROC)) {
                /* symbol declared as intrinsic */
                add_type_ext_id(PROC_EXT_ID(id));
                mark_type_desc(EXT_PROC_TYPE(PROC_EXT_ID(id)));
            }
            break;
        default:
            break;
        }
    }
}


static void
outx_kind(int l, TYPE_DESC tp)
{
    static expv doubeledKind = NULL;
    expv vkind;
    
    if(doubeledKind == NULL)
        doubeledKind = expv_int_term(INT_CONSTANT, type_INT, KIND_PARAM_DOUBLE);

    if(IS_DOUBLED_TYPE(tp))
        vkind = doubeledKind;
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
    outx_indexRange0(l+1, TYPE_ARRAY_ASSUME_KIND(tp), ASSUMED_SIZE,
		     EXPR_ARG1(cobound), EXPR_ARG2(cobound), EXPR_ARG3(cobound));
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
        outx_print(" ref=\"C" ADDRX_PRINT_FMT "\"/>\n", Addr2Uint(tRef));
    }
    else if (TYPE_KIND(tp) || charLen != 1 || vcharLen != NULL || tp->codims){
        outx_print(" ref=\"%s\">\n", tid);
        outx_kind(l1, tp);

        if(charLen != 1|| vcharLen != NULL) {
            outx_tag(l1, "len");
            if(IS_CHAR_LEN_UNFIXED(tp) == FALSE) {
                if(vcharLen != NULL)
                    outx_expv(l2, vcharLen);
                else
                    outx_intAsConst(l2, TYPE_CHAR_LEN(tp));
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
    if (tp->codims){
      outx_print(" ref=\"%s\">\n", getTypeID(rtp));
      outx_coShape(l+1, tp);
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
    /* tp is basic data type */
    outx_print(" ref=\"%s\"", getBasicTypeID(TYPE_BASIC_TYPE(tp)));
    if (TYPE_KIND(tp) || IS_DOUBLED_TYPE(tp) || tp->codims){
        outx_print(">\n");
        outx_kind(l + 1, tp);
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
 * output functionType of external symbol
 */
static void
outx_functionType_EXT(int l, EXT_ID ep)
{
    list lp;
    TYPE_DESC tp;
    const int l1 = l + 1, l2 = l1 + 1;
    const char *rtid;

    if(EXT_PROC_IS_OUTPUT(ep))
        return;

    EXT_PROC_IS_OUTPUT(ep) = TRUE;

    tp = EXT_PROC_TYPE(ep);
    outx_typeAttrOnly_functionTypeWithResultVar(l, ep, "FfunctionType");

    /* external symbol whose type is not explicitly defined
     * is possible to be function or subroutine.
     * so it must not be defined explicit type.
     */
    if(tp) {
        if((TYPE_IS_IMPLICIT(tp) && !TYPE_IS_USED_EXPLICIT(tp)))
            /* If type of function is implict and
               used as neither function nor subroutine,
               type of function is ambiguous.
             */
            rtid = "FnumericAll";
        else {
            if(IS_SUBR(tp))
                rtid = "Fvoid";
            else if(IS_FUNCTION_TYPE(tp))
                rtid = getTypeID(TYPE_REF(tp));
            else
                rtid = getTypeID(tp);
        }
    } else {
        rtid = "Fvoid";
    }

    outx_print(" return_type=\"%s\"", rtid);
    outx_true(EXT_PROC_IS_PROGRAM(ep), "is_program");
    outx_true(EXT_PROC_IS_INTRINSIC(ep), "is_intrinsic");

    if (tp) {
        outx_true(TYPE_IS_RECURSIVE(tp), "is_recursive");
        outx_true(TYPE_IS_EXTERNAL(tp), "is_external");
        outx_true(TYPE_IS_PUBLIC(tp), "is_public");
        outx_true(TYPE_IS_PRIVATE(tp), "is_private");
    }

    if(EXT_PROC_ARGS(ep) == NULL) {
        outx_print("/>\n");
    } else {
        outx_print(">\n");
        outx_tag(l1, "params");
        FOR_ITEMS_IN_LIST(lp, EXT_PROC_ARGS(ep))
            outx_expvNameWithType(l2, EXPR_ARG1(LIST_ITEM(lp)));
        outx_close(l1, "params");
        outx_close(l, "FfunctionType");
    }
}


/**
 * output FstructType
 */
static void
outx_structType(int l, TYPE_DESC tp)
{
    ID id;
    int l1 = l + 1, l2 = l1 + 1, l3 = l2 + 1;

    outx_typeAttrs(l, tp ,"FstructType", TOPT_NEXTLINE);
    outx_tag(l1, "symbols");

    FOREACH_MEMBER(id, tp) {
        outx_printi(l2, "<id type=\"%s\">\n", getTypeID(ID_TYPE(id)));
        outx_symbolName(l3, ID_SYM(id));
        outx_close(l2, "id");
    }

    outx_close(l1,"symbols");
    outx_close(l,"FstructType");
}


/**
 * output types in typeTable
 */
static void
outx_type(int l, TYPE_DESC tp)
{
    TYPE_DESC tRef = TYPE_REF(tp);

    if (IS_SUBR(tp)) {
        /* output nothing */
    } else if(IS_CHAR(tp)) {
        if(checkBasic(tp) == FALSE || checkBasic(tRef) == FALSE)
            outx_characterType(l, tp);
    } else if(IS_ARRAY_TYPE(tp)) {
        outx_arrayType(l, tp);
    } else if(IS_STRUCT_TYPE(tp) && TYPE_REF(tp) == NULL) {
        outx_structType(l, tp);
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
        if (ID_CLASS(id) == CL_PROC &&
            CRT_FUNCEP != NULL &&
            CRT_FUNCEP != PROC_EXT_ID(id)) {
            return FALSE;
        }
        if (TYPE_IS_PUBLIC(tp) || TYPE_IS_PRIVATE(tp)) {
            return TRUE;
        }
        return FALSE;
    }

    switch(ID_CLASS(id)) {
    case CL_VAR:
        if(VAR_IS_IMPLIED_DO_DUMMY(id))
            return FALSE;
#if 0
        if(PROC_CLASS(id) == P_DEFINEDPROC) {
            /* this id is of function.
               Checkes if this id is of the current function or not. */
            if(CRT_FUNCEP == PROC_EXT_ID(id)) {
                return TRUE;
            } else {
                return FALSE;
            }
        }
#endif
        break;
    case CL_PARAM:
        return TRUE;
    case CL_PROC:
        if(PROC_CLASS(id) == P_DEFINEDPROC) {
            /* this id is of function.
               Checkes if this id is of the current function or not. */
            if(CRT_FUNCEP == PROC_EXT_ID(id)) {
                return TRUE;
            } else {
                return FALSE;
            }
        }
    default:
        switch(ID_STORAGE(id)) {
        case STG_TAGNAME:
            return TRUE;
        case STG_UNKNOWN:
        case STG_NONE:
            return FALSE;
        default:
            break;
        }
    }

    return TRUE;
}


/**
 * output symbols in FfunctionDefinition
 */
static void
outx_definition_symbols(int l, EXT_ID ep)
{
    ID id;
    const int l1 = l + 1;

    outx_tag(l, "symbols");

    FOREACH_ID(id, EXT_PROC_ID_LIST(ep)) {
        if(id_is_visibleVar(id) && IS_MODULE(ID_TYPE(id)) == FALSE)
            outx_id(l1, id);
    }

    /* print common ids */
    FOREACH_ID(id, EXT_PROC_COMMON_ID_LIST(ep)) {
        if(IS_MODULE(ID_TYPE(id)) == FALSE)
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
        PROC_CLASS(id) == P_EXTERNAL || \
        TYPE_IS_EXTERNAL(ID_TYPE(id)) || \
        TYPE_IS_INTRINSIC(ID_TYPE(id)) || \
        PROC_CLASS(id) == P_UNDEFINEDPROC || \
        PROC_CLASS(id) == P_DEFINEDPROC) \
    && (PROC_EXT_ID(id) == NULL || \
        PROC_CLASS(id) == P_UNDEFINEDPROC || \
        PROC_CLASS(id) == P_DEFINEDPROC || ( \
        EXT_PROC_IS_INTERFACE(PROC_EXT_ID(id)) == FALSE && \
        EXT_PROC_IS_INTERFACE_DEF(PROC_EXT_ID(id)) == FALSE)) \
    && (IS_MODULE(ID_TYPE(id)) == FALSE \
        && (IS_SUBR(ID_TYPE(id)) == FALSE || \
        has_attribute(ID_TYPE(id)))))


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
    if (ID_IS_OFMODULE(id) == TRUE) {
        return;
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
            case STG_SAVE:
            case STG_AUTO:
            case STG_EQUIV:
            case STG_COMEQ:
            case STG_COMMON:
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


/**
 * output declarations with pragmas
 */
static void
outx_declarations1(int l, EXT_ID parent_ep, int outputPragmaInBody)
{
    const int l1 = l + 1;
    list lp;
    ID id, *ids;
    EXT_ID ep;
    expv v;
    int i, nIDs;
    int hasResultVar = (EXT_PROC_RESULTVAR(parent_ep) != NULL) ? TRUE : FALSE;
    const char *myName = SYM_NAME(EXT_SYM(parent_ep));
    TYPE_DESC tp;

    outx_tag(l, "declarations");

    /*
     * FuseDecl
     */
    FOR_ITEMS_IN_LIST(lp, EXT_PROC_BODY(parent_ep)) {
        v = LIST_ITEM(lp);
        switch(EXPV_CODE(v)) {
        case F95_USE_STATEMENT:
            outx_useDecl(l1, v);
            break;
        case F95_USE_ONLY_STATEMENT:
            outx_useOnlyDecl(l1, v);
            break;
        default:
            break;
        }
    }

    ids = genSortedIDs(EXT_PROC_ID_LIST(parent_ep), &nIDs);

    if (ids) {
        /*
         * Firstly emit struct base type (TYPE_REF(tp) == NULL).
         * ex) type(x)::p = x(1)
         */
        for (i = 0; i < nIDs; i++) {
            id = ids[i];

            if (ID_IS_EMITTED(id) == TRUE) {
                continue;
            }

            if (ID_IS_OFMODULE(id) == TRUE) {
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
                            emit_decl(l1, ids[j]);
                            ID_IS_EMITTED(ids[j]) = TRUE;
                        }
                    }
                }
                outx_structDecl(l1, id);
                ID_IS_EMITTED(id) = TRUE;
            }
        }

        /*
         * varDecl except structDecl, namelistDecl
         */
        for (i = 0; i < nIDs; ++i) {
            id = ids[i];

            if (hasResultVar == TRUE &&
                strcasecmp(myName, SYM_NAME(ID_SYM(id))) == 0) {
                continue;
            }

            emit_decl(l1, id);
            ID_IS_EMITTED(id) = TRUE;
        }
        free(ids);
    }


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
outx_moduleProcedureDecl(int l, EXT_ID parent_ep)
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

    outx_tagOfDecl1(l, "FmoduleProcedureDecl", GET_EXT_LINE(ep));

    FOREACH_EXT_ID(ep, parent_ep) {
        if(EXT_TAG(ep) == STG_EXT &&
            EXT_PROC_IS_MODULE_PROCEDURE(ep)) {
            outx_symbolName(l1, EXT_SYM(ep));
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
    outx_symbolNameWithFunctionType(l1, ep);
    outx_declarations(l1, ep);
    outx_close(l, "FfunctionDecl");
    CRT_FUNCEP_POP;
}


static void
outx_innerDefinitions(int l, EXT_ID extids, int asDefOrDecl)
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

    outx_moduleProcedureDecl(l, extids);
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
    outx_innerDefinitions(l + 1, contains, TRUE);
    outx_close(l, "FcontainsStatement");
}


/**
 * output FinterfaceDecl
 */
static void
outx_interfaceDecl(int l, EXT_ID ep)
{
    EXT_ID extids;
#if 0
    char buf[256];
#endif

    extids = EXT_PROC_INTR_DEF_EXT_IDS(ep);
    if(extids == NULL)
        return;

    if(EXT_IS_OFMODULE(ep) == TRUE)
        return;

#if 0
    if(EXT_IS_BLANK_NAME(ep))
        buf[0] = '\0';
    else
        sprintf(buf, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));

    outx_tagOfDecl1(l, "FinterfaceDecl%s", EXT_LINE(ep), buf);
    outx_innerDefinitions(l + 1, extids, FALSE);
    outx_close(l, "FinterfaceDecl");
#endif
    CRT_FUNCEP_PUSH(NULL);
    outx_printi(l, "<FinterfaceDecl");

    switch(EXT_PROC_INTERFACE_CLASS(ep)) {
        case INTF_DECL:
            if(EXT_IS_BLANK_NAME(ep) == FALSE)
                outx_printi(0, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));
            break;
        case INTF_ASSINGMENT:
            outx_true(TRUE, "is_assignment");
            break;
        case INTF_OPERATOR:
        case INTF_USEROP:
            outx_printi(0, " name=\"%s\"", SYM_NAME(EXT_SYM(ep)));
            outx_true(TRUE, "is_operator");
            break;
        default:
            /* never reach. here*/
            break;
    }

    outx_lineno(EXT_LINE(ep));
    outx_printi(0,">\n");
    outx_innerDefinitions(l + 1, extids, FALSE);
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

    CRT_FUNCEP_PUSH(ep);

    outx_tagOfDecl1(l, "FfunctionDefinition", GET_EXT_LINE(ep));
    outx_symbolNameWithFunctionType(l1, ep);
    if(flag_module_compile) {
        outx_tag(l1, "symbols");
        outx_close(l1, "symbols");
        outx_tag(l1, "declarations");
        outx_close(l1, "declarations");
        outx_tag(l1, "body");
        outx_close(l1, "body");
    } else {
        outx_definition_symbols(l1, ep);
        outx_declarations(l1, ep);
        outx_tag(l1, "body");
        outx_expv(l2, EXT_PROC_BODY(ep));
        outx_contains(l2, ep);
        outx_close(l1, "body");
    }
    outx_close(l, "FfunctionDefinition");

    CRT_FUNCEP_POP;
}


/**
 * output FmoduleDefinition
 */
static void
outx_moduleDefinition(int l, EXT_ID ep)
{
    const int l1 = l + 1;

    if(flag_module_compile && is_outputed_module) {
        fatal("output multiple FmoduleDefinition in module file");
    }

    is_outputed_module = TRUE;
    
    outx_tagOfDecl1(l, "%s name=\"%s\"", GET_EXT_LINE(ep),
                    "FmoduleDefinition", SYM_NAME(EXT_SYM(ep)));
    outx_definition_symbols(l1, ep);
    outx_declarations1(l1, ep, TRUE); // output with pragma
    outx_contains(l1, ep);
    outx_close(l, "FmoduleDefinition");
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
 * recursively collect TYPE_DESC to type_list
 */
static void
collect_types1(EXT_ID extid)
{
    EXT_ID ep;
    TYPE_DESC tp;

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

        mark_type_desc(EXT_PROC_TYPE(ep));
        collect_type_desc(EXT_PROC_ARGS(ep));
        mark_type_desc_in_id_list(EXT_PROC_ID_LIST(ep));
        collect_type_desc(EXT_PROC_BODY(ep));

        for(tp = EXT_PROC_STRUCT_DECLS(ep); tp != NULL; tp = TYPE_SLINK(tp)) {
            if(TYPE_IS_DECLARED(tp) == FALSE) {
                error("component of type '%s' is not declared.", SYM_NAME(ID_SYM(TYPE_TAGNAME(tp))));
                exit(1);
            }
            mark_type_desc(tp);
            mark_type_desc_in_structure(tp);
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
    collect_types1(extid);
    FOREACH_TYPE_EXT_ID(te, type_ext_id_list) {
        TYPE_DESC tp = EXT_PROC_TYPE(te->ep);
        if(tp && EXT_TAG(te->ep) == STG_EXT)
            mark_type_desc(EXT_PROC_TYPE(te->ep));
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
    TYPE_EXT_ID te;

    outx_tag(l, "typeTable");

    for (tp = type_list; tp != NULL; tp = TYPE_LINK(tp)){
      outx_type(l1, tp);
    }

    FOREACH_TYPE_EXT_ID(te, type_ext_id_list) {
        assert(EXT_TAG(te->ep) == STG_EXT);
        outx_functionType_EXT(l1, te->ep);
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
    type_list = NULL;

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

