#include <ctype.h>
#include "xcodeml.h"

static int          column_count = 0;
static char         buffer[MAXBUFFER];
static const char   delim[] = "\n";
static FILE *       outfd;

static void outf_expression(XcodeMLNode * expr);
static void outf_constant(XcodeMLNode * expr);
static void outf_varName(XcodeMLNode * expr);
static void outf_complexConst(XcodeMLNode * expr);
static void outf_unary_expression(XcodeMLNode * expr, char * op);
static void outf_binary_expression(XcodeMLNode * expr, char * op);
static void outf_functionCall(XcodeMLNode * expr);
static void outf_memberRef(XcodeMLNode * expr);
static void outf_characterRef(XcodeMLNode * expr);
static void outf_arrayRef(XcodeMLNode * expr);
static void outf_structConst(XcodeMLNode * expr);
static void outf_arrayConst(XcodeMLNode * expr);
static void outf_indexRange(XcodeMLNode * expr);
static void outf_doLoop(XcodeMLNode * expr);
static void outf_expressionList(XcodeMLNode *node);

struct priv_parm_list * priv_parm_list_head = NULL;
struct priv_parm_list * priv_parm_list_tail = NULL;

void
init_outputf(FILE * fd)
{
    column_count = 0;
    outfd = fd;
    priv_parm_list_head = NULL;
    priv_parm_list_tail = NULL;
    current_symbol_stack = NULL;
}

/**
 * \brief Flushes output buffer.
 */
void
outf_flush()
{
    fprintf(outfd, "%s\n", buffer);
    memset(buffer, '\0', MAXBUFFER);
    column_count = 0;
}

/**
 * \brief Outputs a token.
 *
 * @param token
 */
void
outf_token(const char * token)
{
    int len;

    if (token == NULL)
        return;

    len = strlen(token);

    if (len != 0 && column_count == MAXBUFFER) {
        char lastChar;

        lastChar = buffer[LAST_COLUMN];
        buffer[LAST_COLUMN] = '&';
        outf_flush();
        buffer[0] = '&';
        buffer[1] = lastChar;
        column_count = 2;
    }


    while (len + column_count > MAXBUFFER - 1) {
        strncpy(buffer + column_count, token, MAXBUFFER - column_count - 1);
        token += MAXBUFFER - column_count - 1;

        len = strlen(token);

        buffer[LAST_COLUMN] = '&';
        outf_flush();
        buffer[0] = '&';
        column_count++;
    }

    strncpy(buffer + column_count, token, len);
    column_count += len;
}

/**
 * Outputs a token with comma.
 */
static void
outf_tokenc(char * token, int comma)
{
    if(comma)
        outf_token(",");
    outf_token(token);
}

/**
 * Outputs a token and flushes the buffer.
 */
void
outf_tokenln(const char * token)
{
    outf_token(token);
    outf_flush();
}


/**
 * \brief Outputs a type signature as a primitive type.
 */
static int
outf_primitive(char * type_signature)
{
    if (type_signature == NULL)
        return FALSE;

    if (type_signature == NULL ||
        strcmp(type_signature, "Fvoid") == 0 ||
        strcmp(type_signature, "Fnumeric") == 0 ||
        strcmp(type_signature, "FnumericAll") == 0) return FALSE;

    if (strcmp(type_signature, "Fint")       == 0) outf_token("INTEGER");
    if (strcmp(type_signature, "Fcharacter") == 0) outf_token(" CHARACTER");
    if (strcmp(type_signature, "Freal")      == 0) outf_token("REAL");
    if (strcmp(type_signature, "Fcomplex")   == 0) outf_token(" COMPLEX");
    if (strcmp(type_signature, "Flogical")   == 0) outf_token("LOGICAL");

    return TRUE;
}

/**
 * \brief Outputs type.
 */
static int
outf_basic_type(XcodeMLNode * basic_type, char * tagname)
{
    char * type, * ref;
    XcodeMLNode * len, * kind;
    int outputed = FALSE;

    if(basic_type == NULL)
        return FALSE;

    if(strcmp(XCODEML_NAME(basic_type), "FbasicType") == 0) {
        type = GET_TYPE(basic_type);
        ref  = GET_REF(basic_type);

        if (ref == NULL)
            ref = type;

        if (type_isPrimitive(ref)) {
            outputed = outf_primitive(ref);
        }

        len  = GET_LEN(basic_type);
        kind = GET_KIND(basic_type);

        if (len == NULL && kind == NULL)
            return outputed;

        outf_token("(");
        if (len != NULL) {
            outf_token("LEN=");

            if (GET_CHILD0(len) == NULL) {
                outf_token("*");
            } else {
                outf_expression(GET_CHILD0(len));
            }
            if (kind != NULL) {
                outf_token(",");
            }
        }

        if (kind != NULL) {
            if (strcmp(ref, "Fcharacter") == 0)
                outf_token("KIND=");

            if (GET_CHILD0(kind) == NULL) {
                outf_token("*");
            } else {
                outf_expression(GET_CHILD0(kind));
            }
        }
        outf_token(")");

    } else if((strcmp(XCODEML_NAME(basic_type), "FstructType") == 0)) {
        if (tagname != NULL) {
            outf_token("TYPE(");
            outf_token(tagname);
            outf_token(")");
        }

    } else if((strcmp(XCODEML_NAME(basic_type), "FfunctionType") == 0)) {
        ref  = GET_RETURN(basic_type);
        if (type_isPrimitive(ref)) {
            outputed = outf_primitive(ref);
        }
        return outputed;

    } else {
        return FALSE;
    }

    return TRUE;
}

/**
 * \brief Outputs attributes of the type.
 */
static int
outf_type_attribute(XcodeMLNode * first, XcodeMLNode * last, int outputed)
{
    char * intent;

    if (first == NULL)
        return FALSE;

    if (GET_IS_EXTERNAL(first) == true ||
        GET_IS_EXTERNAL(last) == true) {
        outf_tokenc("EXTERNAL", outputed);
        return TRUE;
    }

    if (GET_IS_INTRINSIC(first) == true ||
        GET_IS_INTRINSIC(last) == true) {
        outf_tokenc("INTRINSIC", outputed);
        return TRUE;
    }

    if (GET_IS_POINTER(first) == true ||
        GET_IS_POINTER(last) == true) {
        outf_token(",POINTER");
        outputed = TRUE;
    }

    if (GET_IS_TARGET(first) == true ||
        GET_IS_TARGET(last) == true) {
        outf_token(",TARGET");
        outputed = TRUE;
    }

    if (GET_IS_OPTIONAL(first) == true ||
        GET_IS_OPTIONAL(last) == true) {
        outf_token(",OPTIONAL");
        outputed = TRUE;
    }

    if (GET_IS_SAVE(first) == true ||
        GET_IS_SAVE(last) == true) {
        outf_token(",SAVE");
        outputed = TRUE;
    }

    if (GET_IS_PARAMETER(first) == true ||
        GET_IS_PARAMETER(last) == true) {
        outf_token(",PARAMETER");
        outputed = TRUE;
    }

    if (GET_IS_ALLOCATABLE(first) == true ||
        GET_IS_ALLOCATABLE(last) == true) {
        outf_token(",ALLOCATABLE");
        outputed = TRUE;
    }

    intent = GET_INTENT(first);
    if (intent == NULL)
        intent = GET_INTENT(last);

    if (intent != NULL) {
        outf_token(",INTENT(");
        outf_token(intent);
        outf_token(")");
        outputed = TRUE;
    }

    return outputed;
}

/**
 * \brief Outputs a shape of the type.
 */
static int
outf_type_shape(XcodeMLNode * shape)
{
    XcodeMLNode * x;
    XcodeMLList * lp;
    char * tag;
    int outputed = FALSE;

    bool isNeedBrace = false;

    FOR_ITEMS_IN_XCODEML_LIST(lp, shape) {
        x = XCODEML_LIST_NODE(lp);

        if (x == NULL || XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        tag = XCODEML_NAME(x);

        if (tag == NULL || strlen(tag) <= 0)
            continue;

        if (strcmp("len", tag) == 0)
            continue;

        if ((strcmp("arrayIndex", tag) == 0) ||
            (strcmp("indexRange", tag) == 0)) {
            if (isNeedBrace == false) {
                outf_token(",DIMENSION(");
                isNeedBrace = true;
            } else {
                outf_token(",");
            }
            outf_expression(x);
            outputed = TRUE;
        }
    }

    if (isNeedBrace == true) {
        outf_token(")");
    }

    return outputed;
}

/**
 * \brief Outputs a XcodeML declaration tag.
 */
int
outf_decl(char * type_signature, char * symbol,
          XcodeMLNode * value, bool convertSymbol, int force)
{
    xentry * first, * last, *func_org = NULL;
    char * ref = NULL;
    int outputed = FALSE;
    int is_private_parameter = FALSE;

    if (type_signature == NULL || symbol == NULL ||
        strcmp(type_signature, "Fnumeric") == 0 ||
        strcmp(type_signature, "FnumericAll") == 0)
        return FALSE;

    if (type_isPrimitive(type_signature)) {
        if(force == FALSE && is_use_symbol(symbol) == FALSE)
            return FALSE;
        outputed = outf_primitive(type_signature);
    } else {
        first = typetable_dehash(type_signature);
        if (GET_CONTENT(first) == NULL)
            return FALSE;

        if (IS_FFUNCTIONTYPE(first)) {
#if 0
            if (GET_IS_PRIVATE(GET_CONTENT(first))) {
            }
#endif
            func_org = first;
            if (GET_CONTENT(first) == NULL)
                return FALSE;
            ref = GET_RETURN(GET_CONTENT(first));
#if 0
            if (strcmp(ref, "Fnumeric") == 0 ||
                strcmp(ref, "FnumericAll") == 0)
                return FALSE;
#endif
            if (!type_isPrimitive(ref)) {
                first = typetable_dehash(ref);
                ref = GET_REF(GET_CONTENT(first));
            }
        } else {
            ref = GET_REF(GET_CONTENT(first));
        }

        last = first;
        while (ref != NULL && type_isPrimitive(ref) == false) {
            last = typetable_dehash(ref);
            if (last == NULL) /* error. */
                break;
            ref = GET_REF(GET_CONTENT(last));
        }

        if (last == NULL)
            last = first;

        int is_private =
            GET_IS_PRIVATE(GET_CONTENT(first)) ||
            GET_IS_PRIVATE(GET_CONTENT(last));
        int is_param =
            GET_IS_PARAMETER(GET_CONTENT(first)) ||
            GET_IS_PARAMETER(GET_CONTENT(last));
        int is_use_sym = is_use_symbol(symbol);

        /* If type is private then a declaration will not be appeared. */
        if (is_private) {
            if(func_org != NULL) { /* type is function. */
                if(is_inner_module) {
                    return FALSE;
                } else {
                    convertSymbol = false;
                }
#if 0
            } else if ((!IS_FSTRUCTTYPE(first) ||
                        (first == last)) &&
                       IS_FSTRUCTTYPE(last)) {
#endif
            } else if ((IS_FSTRUCTTYPE(first) &&
                        GET_IS_PRIVATE(GET_CONTENT(last)) == false)) {
                // do nothing.
            } else if (is_param) {
                is_private_parameter = TRUE;
            } else {
                return FALSE;
            }
        } else if (is_param && is_use_sym == FALSE) {
            is_private_parameter = TRUE;
        }

        if(is_private_parameter) {
            struct priv_parm_list * pp;
            pp = XMALLOC(struct priv_parm_list *, sizeof(struct priv_parm_list));
            PRIV_PARM_SYM(pp) = symbol;
            PRIV_PARM_LINK_ADD(pp);
            symbol = convert_to_non_use_symbol(symbol);
        }

        if(force == FALSE &&
            is_private_parameter == FALSE &&
            is_use_sym == FALSE)
            return FALSE;

        if(func_org == NULL) { /* not for function type. */
            outputed |= outf_basic_type(GET_CONTENT(last), GET_TAGNAME(last));
            if(outputed == FALSE)
                return FALSE;
            outputed |= outf_type_attribute(GET_CONTENT(first), GET_CONTENT(last), outputed);
            outputed |= outf_type_shape(GET_CONTENT(first));
        } else {              /* for function type. */
            outputed |= outf_basic_type(GET_CONTENT(last), GET_TAGNAME(last));
            outputed |= outf_type_attribute(GET_CONTENT(func_org), GET_CONTENT(last), outputed);
            if(outputed && !type_isPrimitive(GET_RETURN(GET_CONTENT(first))))
                outputed |= outf_type_shape(GET_CONTENT(first));
        }
    }

    if(outputed == FALSE)
        return FALSE;

    outf_token(" :: ");

    outf_token(symbol);

    if (value != NULL) {
        outf_token(" = ");
        outf_expression(XCODEML_ARG1(value));
    }

    return TRUE;
}

/**
 * \brief Outputs a XcodeML tag as a expression.
 */
static void
outf_expression(XcodeMLNode * expr)
{
    char * op;
    char * name;

    if (expr == NULL)
        return; /* error */

    name = (XCODEML_NAME(expr));

    if (name == NULL)
        return; /* error */

    if (strcmp(name, "FintConstant") == 0) {
        outf_constant(expr);
    } else if (strcmp(name, "FrealConstant") == 0) {
        outf_constant(expr);
    } else if (strcmp(name, "FcharacterConstant") == 0) {
        outf_constant(expr);
    } else if (strcmp(name, "FlogicalConstant") == 0) {
        outf_constant(expr);
    } else if (strcmp(name, "Var") == 0) {
        outf_varName(expr);
    } else if (strcmp(name, "FcomplexConstant") == 0) {
        outf_complexConst(expr);
    } else if (strcmp(name, "FarrayConstructor") == 0) {
        outf_arrayConst(expr);
    } else if (strcmp(name, "FstructConstructor") == 0) {
        outf_structConst(expr);
    } else if (strcmp(name, "varRef") == 0) {
        outf_expression(GET_CHILD0(expr));
    } else if (strcmp(name, "arrayIndex") == 0) {
        outf_expression(GET_CHILD0(expr));
    } else if (strcmp(name, "functionCall") == 0) {
        outf_functionCall(expr);
    } else if (strcmp(name, "FmemberRef") == 0) {
        outf_memberRef(expr);
    } else if (strcmp(name, "indexRange") == 0) {
        outf_indexRange(expr);
    } else if (strcmp(name, "FdoLoop") == 0) {
        outf_doLoop(expr);
    } else if (strcmp(name, "FarrayRef") == 0) {
        outf_arrayRef(expr);
    } else if (strcmp(name, "FcharacterRef") == 0) {
        outf_characterRef(expr);
    } else if (strcmp(name, "plusExpr") == 0) {
        op = "+";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "minusExpr") == 0) {
        op = "-";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "mulExpr") == 0) {
        op = "*";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "divExpr") == 0) {
        op = "/";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "FpowerExpr") == 0) {
        op = "**";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "FconcatExpr") == 0) {
        op = "//";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logEQExpr") == 0) {
        op = "==";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logNEQExpr") == 0) {
        op = "/=";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logGEExpr") == 0) {
        op = ">=";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logGTExpr") == 0) {
        op = ">";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logLEExpr") == 0) {
        op = "<=";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logLTExpr") == 0) {
        op = "<";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logAndExpr") == 0) {
        op = ".AND.";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logOrExpr") == 0) {
        op = ".OR.";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logEQVExpr") == 0) {
        op = ".EQV.";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "logNEQVExpr") == 0) {
        op = ".NEQV.";
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "unaryMinusExpr") == 0) {
        op ="-";
        outf_unary_expression(expr, op);
    } else if (strcmp(name, "logNotExpr") == 0) {
        op =".NOT.";
        outf_unary_expression(expr, op);
    } else if (strcmp(name, "userBinaryExpr") == 0) {
        op = xcodeml_getAsString(GET_NAME(expr));
        outf_binary_expression(expr, op);
    } else if (strcmp(name, "userUnaryExpr") == 0) {
        op = xcodeml_getAsString(GET_NAME(expr));
        outf_unary_expression(expr, op);
    }
}

/**
 * \brief Outputs a XcodeML tag as a constant expression, exclude a complex constant.
 */
static void
outf_constant(XcodeMLNode * expr)
{
    char * content, * p;
    const char sq[] = "'";
    const char dq[] = "\"";
    XcodeMLNode * kind;

    content = xcodeml_getAsString(expr);

    if (content == NULL)
        return;

    if (strcmp(XCODEML_NAME(expr), "FcharacterConstant") == 0) {
        const char * q = (index(content, *dq) == NULL)?dq:sq;
        outf_token(q);
        outf_token(content);
        outf_token(q);
    } else {
        outf_token(content);
    }

    kind = GET_KIND(expr);

    if (kind != NULL) {
        // gfortran rejects kind with 'd' exponent
        for (p = content; *p != '\0' ; p++) {
            if (*p == 'd' || *p == 'D')
                kind = NULL;
        }

        if(kind != NULL) {
            int i;
            int num_kind = TRUE;
            char * kind_symbol;
            kind_symbol = xcodeml_getAsString(kind);
            outf_token("_");
            for (i = 0; kind_symbol[i] != '\0'; i++) {
                if (!isdigit(kind_symbol[i])) {
                    num_kind = FALSE;
                }
            }
            if (!num_kind && is_use_symbol(kind_symbol) == false) {
                kind_symbol = convert_to_non_use_symbol(kind_symbol);
            }
            outf_token(kind_symbol);
        }
    }
}


/**
 * Output variable name.
 */
static void
outf_varName(XcodeMLNode * expr)
{
    char * content;

    content = xcodeml_getAsString(expr);

    if (content == NULL)
        return;

    if (is_use_symbol(content) == false) {
        content = convert_to_non_use_symbol(content);

    } else {
        struct priv_parm_list * lp;
        PRIV_PARM_LINK_FOR(lp, priv_parm_list_head) {
            if(strcmp(PRIV_PARM_SYM(lp), content) == 0) {
                content = convert_to_non_use_symbol(content);
                break;
            }
        }
    }

    outf_token(content);
}


/**
 * \brief Outputs a XcodeML tag as a complex constant.
 */
static void
outf_complexConst(XcodeMLNode * expr)
{
    outf_token("(");
    outf_expression(GET_CHILD0(expr));
    outf_token(",");
    outf_expression(GET_CHILD1(expr));
    outf_token(")");
}

/**
 * \brief Outputs a XcodeML tag as a binary expression.
 */
static void
outf_binary_expression(XcodeMLNode * expr, char * op)
{
    outf_token("(");
    outf_expression(GET_CHILD0(expr));
    outf_token(")");
    outf_token(op);
    outf_token("(");
    outf_expression(GET_CHILD1(expr));
    outf_token(")");
}

/**
 * \brief Outputs a XcodeML tag as a unary expression.
 */
static void
outf_unary_expression(XcodeMLNode * expr, char * op)
{
    outf_token(op);
    outf_expression(GET_CHILD0(expr));
}

/**
 * \brief Outputs a XcodeML tag as a function call expression.
 */
static void
outf_functionCall(XcodeMLNode * expr)
{
    char * funcName;
    XcodeMLNode * arg;
    funcName = xcodeml_getAsString(GET_NAME(expr));
    arg = GET_ARGUMENTS(expr);

    outf_token(funcName);
    outf_token("(");
    if (arg != NULL)
        outf_expressionList(arg);
    outf_token(")");
}

/**
 * \brief Outputs a XcodeML tag as a member reference expression.
 */
static void
outf_memberRef(XcodeMLNode * expr)
{
    char * member;
    member = GET_MEMBER(expr);

    outf_expression(GET_CHILD0(expr));
    outf_token("%");
    outf_token(member);
}

/**
 * \brief Outputs a XcodeML tag as a character reference expression.
 */
static void
outf_characterRef(XcodeMLNode * expr)
{
    outf_expression(GET_CHILD0(expr));
    outf_token("(");
    outf_expression(GET_CHILD1(expr));
    outf_token(")");
}

/**
 * \brief Outputs a XcodeML tag as a array reference expression.
 */
static void
outf_arrayRef(XcodeMLNode * expr)
{
    int i = 0;
    XcodeMLList * lp;

    FOR_ITEMS_IN_XCODEML_LIST(lp, expr) {
        XcodeMLNode * x;
        x = XCODEML_LIST_NODE(lp);

        if (XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        outf_expression(x);

        if (i == 0) {
            outf_token("(");
        } else {
            if (XCODEML_LIST_NEXT(lp) != NULL &&
                XCODEML_LIST_NODE(XCODEML_LIST_NEXT(lp)) != NULL) {
                outf_token(",");
            }
        }
    }
    outf_token(")");
}

/**
 * \brief Outputs a XcodeML tag as a struct constructer.
 */
static void
outf_structConst(XcodeMLNode * expr)
{
    xentry * xe;
    char * type_sig;

    type_sig = GET_TYPE(expr);
    xe = typetable_dehash(type_sig);

    outf_token(GET_TAGNAME(xe));
    outf_token("(");
    outf_expressionList(expr);
    outf_token(")");
}

/**
 * \brief Outputs a XcodeML tag as a array constructer.
 */
static void
outf_arrayConst(XcodeMLNode * expr)
{
    outf_token("(/");
    outf_expressionList(expr);
    outf_token("/)");
}

/**
 * \brirf Outputs a XcodeML tag as an index range expression.
 */
static void
outf_indexRange(XcodeMLNode * expr)
{
    if (GET_IS_ASHAPE(expr) == true) {
        outf_token(":");
    } else if (GET_IS_ASIZE(expr) == true) {
        outf_token("*");
    } else {
        XcodeMLNode * step;

        outf_expression(GET_CHILD0(GET_LOWER(expr)));
        outf_token(":");
        outf_expression(GET_CHILD0(GET_UPPER(expr)));

        step = GET_CHILD1(GET_STEP(expr));
        if (step != NULL) {
            outf_token(":");
            outf_expression(step);
        }
    }
}

/**
 * \brief Outputs a XcodeML tag as a do loop expression.
 */
static void
outf_doLoop(XcodeMLNode * expr)
{
    char * var;
    XcodeMLList * lp;
    XcodeMLNode * indexRange, * step;

    var = xcodeml_getAsString(GET_VAR(expr));
    indexRange = GET_INDEXRANGE(expr);

    outf_token("(");

    FOR_ITEMS_IN_XCODEML_LIST(lp, expr) {
        XcodeMLNode * x = XCODEML_LIST_NODE(lp);

        if (x == NULL || XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if ((strcmp(XCODEML_NAME(x), "Var") == 0) ||
            (strcmp(XCODEML_NAME(x), "indexRange") == 0))
            continue;

        if (strcmp(XCODEML_NAME(x), "value") == 0) {
            outf_expression(GET_CHILD0(x));
            outf_token(",");
        }
    }

    outf_token(var);
    outf_token("=");

    outf_expression(GET_LOWER(indexRange));
    outf_token(",");
    outf_expression(GET_UPPER(indexRange));

    step = GET_STEP(indexRange);
    if (step != NULL) {
        outf_token(",");
        outf_expression(step);
    }
    outf_token(")");
}


/**
 * \brief Outputs a list of the XcodeML tag sequencialy.
 *
 * @param expr XcodeML tag.
 */
static void
outf_expressionList(XcodeMLNode * expr)
{
    XcodeMLList * lp;

    FOR_ITEMS_IN_XCODEML_LIST(lp, expr) {
        XcodeMLNode * x;
        x = XCODEML_LIST_NODE(lp);

        if (x == expr)
            continue;

        if (XCODEML_TYPE(x) != XcodeML_Element)
            continue;

        if (strcmp(XCODEML_NAME(x), "namedValue") == 0) {
            char * name = xcodeml_getAsString(GET_NAME(x));
            if(name != NULL) {
                outf_token(name);
                outf_token("=");
                x = XCODEML_ARG2(x);
            }
        }

        outf_expression(x);
        if (XCODEML_LIST_NEXT(lp) != NULL &&
            XCODEML_LIST_NODE(XCODEML_LIST_NEXT(lp)) != NULL) {
            outf_token(",");
        }
    }
}

