/**
 * \file F-io-xcodeml.c
 */

#include "F-front.h"

#define IOTYPE_SEQUENTIAL       0
#define IOTYPE_DIRECT           1
#define IOTYPE_UNITONLY         2
#define IOTYPE_INTERNAL         3
#define IOTYPE_NAMELIST         4

static char fmtVBuf[1024];


#define COND_ERR        0
#define COND_END        1

extern int order_sequence;

static char *
StatusLineToFormatVariableName(n)
     int n;
{
    sprintf(fmtVBuf, "Fmt_%06d", n);
    return strdup(fmtVBuf);
}

static char *
FormatVariableNameToStatusLineStr(name)
     char *name;
{
    char *s = strstr(name, "Fmt_");
    
    if (s == NULL) {
        return name;
    } else {
        s = name + 4;
        while (*s == '0') {
            s++;
        }
        return s;
    }
}

static SYMBOL nml = (struct symbol []){{NULL, "nml", 0, 0}};

static expv
compile_io_arguments(expr args)
{
    list lp;
    expr arg;
    expv varg, vargs;

    vargs = list0(LIST);

    if (args == NULL) {
        return vargs;
    }

    FOR_ITEMS_IN_LIST(lp, args) {
        arg = LIST_ITEM(lp);

        if(arg == NULL) {
            varg = NULL;
        } else if(EXPR_CODE(arg) != F_SET_EXPR) {
            /* Don't check type of arguments. */
            varg = expv_reduce(compile_expression(arg), FALSE);
            ID id = find_ident(EXPR_SYM(varg));
            if(id != NULL && ID_CLASS(id) == CL_NAMELIST) {
                varg = list2(F_SET_EXPR, make_enode(IDENT, nml), varg);
            }
        } else {
            expr rkey = EXPR_ARG1(arg);
            expr rval = EXPR_ARG2(arg);
            expr vkey, vval;

            vkey = rkey;

            if(rval && (EXPR_CODE(rval) == IDENT)) {
                vval = rval;

            } else if(rval == NULL) {
                vval = NULL;

            }else {
                vval = expv_reduce(compile_expression(rval), FALSE);

                if(vval == NULL) {
                    return NULL;
                }
            }
            varg = list2(F_SET_EXPR, vkey, vval);
        }

        list_put_last(vargs, varg);
    }

    return vargs;
}


void
compile_FORMAT_decl(st_no, x)
     int st_no;
     expr x;
{
    ID fId;
    SYMBOL sym = NULL;

    sym = find_symbol(StatusLineToFormatVariableName(st_no));
    fId = declare_ident(sym, CL_UNKNOWN);
    if (ID_CLASS(fId) == CL_UNKNOWN) {
        /*
         * means this format is declared before appeared in I/O
         * statement(s).
         */
        fId = declare_ident(sym, CL_FORMAT);
    } else if (ID_CLASS(fId) != CL_FORMAT) {
        fatal("compile_FORMAT_decl: format type label is declared as other type??");
    }

    if (FORMAT_STR(fId) == NULL) {
        switch (EXPR_CODE(EXPR_ARG1(x))) {
            case STRING_CONSTANT: {
                int len = strlen(EXPR_STR(EXPR_ARG1(x)));
                FORMAT_STR(fId) = expv_str_term(STRING_CONSTANT,
                                                type_char(len),
                                                strdup(EXPR_STR(EXPR_ARG1(x))));
                break;
            }
            default: {
                error("invalid format.");
                break;
            }
        }
    }
    output_statement(list1(F_FORMAT_DECL,FORMAT_STR(fId)));
    return;
}


void
FinalizeFormat()
{
    ID id;

    FOREACH_ID(id, LOCAL_SYMBOLS) {
        if (ID_CLASS(id) == CL_FORMAT) {
            if (FORMAT_STR(id) == NULL) {
                error("missing statement number %s (format).",
                      FormatVariableNameToStatusLineStr(SYM_NAME(ID_SYM(id))));
            }
        }
    }
}


void
compile_IO_statement(x)
     expr x;
{
    list lp;
    expv v = NULL, x2;
    expv callArgs;

    if (EXPR_CODE(x) == F_PRINT_STATEMENT) {
        callArgs = list1(LIST, EXPR_ARG1(x));
    } else {
        callArgs = compile_io_arguments(EXPR_ARG1(x));
    }

    expv v2 = list0(LIST);
    FOR_ITEMS_IN_LIST(lp,EXPR_ARG2(x)){
        x2 = LIST_ITEM(lp);
        if (x2 == NULL)
            list_put_last(v2, NULL);
        else
            list_put_last(v2, expv_reduce(compile_expression(x2), FALSE));
    }

    switch (EXPR_CODE(x)) {
        case F_PRINT_STATEMENT: {
            v = expv_cons(F_PRINT_STATEMENT, NULL, callArgs, v2);
            break;
        }
        case F_WRITE_STATEMENT: {
            v = expv_cons(F_WRITE_STATEMENT, NULL, callArgs, v2);
            break;
        }
        case F_READ_STATEMENT: 
        case F_READ1_STATEMENT: {
            v = expv_cons(F_READ_STATEMENT, NULL, callArgs, v2);
            break;
        }
        default: {
            fatal("no IO statement.");
        }
    }
    output_statement(v);
    return;
}


void
compile_OPEN_statement(x)
     expr x;
{
    expr v, callArgs;

    if (EXPV_CODE(EXPR_ARG1(x)) != LIST) {
        fatal("syntax error in OPEN???");
    }

    callArgs = compile_io_arguments(EXPR_ARG1(x));
    v = expv_cons(F_OPEN_STATEMENT, NULL, callArgs, NULL);
    output_statement(v);
    return;
}


void
compile_CLOSE_statement(x)
     expr x;
{
    expr v, callArgs;

    if (EXPV_CODE(EXPR_ARG1(x)) != LIST) {
        fatal("syntax error in CLOSE ???");
    }

    callArgs = compile_io_arguments(EXPR_ARG1(x));
    v = expv_cons(F_CLOSE_STATEMENT, NULL, callArgs, NULL);
    output_statement(v);
    return;
}

#define GEN_NODE(TYPE, VALUE) \
  make_enode((TYPE), ((void *)((_omAddrInt_t)(VALUE))))



/*
 * BACKSPACE
 * ENDFILE
 * REWIND
 */
void
compile_FPOS_statement(expr x)
{
    expr v = NULL, callArgs;

    if (EXPV_CODE(EXPR_ARG1(x)) != LIST) {
        callArgs = list0(LIST);
        list_put_last(callArgs,
                      expv_reduce(compile_expression(EXPR_ARG1(x)), FALSE));
    } else {
        callArgs = compile_io_arguments(EXPR_ARG1(x));
    }

    switch (EXPR_CODE(x)) {
        case F_BACKSPACE_STATEMENT: {
            v = expv_cons(F_BACKSPACE_STATEMENT, NULL, callArgs, NULL);
            break;
        }
        case F_REWIND_STATEMENT: {
            v = expv_cons(F_REWIND_STATEMENT, NULL, callArgs, NULL);
            break;
        }
        case F_ENDFILE_STATEMENT: {
            v = expv_cons(F_ENDFILE_STATEMENT, NULL, callArgs, NULL);
            break;
        }
        default: {
            fatal("unknown file positioning statement.");
        }
    }

    output_statement(v);
    return;
}


void
compile_INQUIRE_statement(x)
     expr x;
{
    list lp;
    expv v, callArgs;
    expv outputList;

    if (EXPV_CODE(EXPR_ARG1(x)) != LIST) {
        fatal("syntax error in INQUIRE???");
    }

    callArgs = compile_io_arguments(EXPR_ARG1(x));

    outputList = list0(LIST);
    FOR_ITEMS_IN_LIST(lp,EXPR_ARG2(x)){
        expv x2 = LIST_ITEM(lp);
        if (x2 == NULL)
            list_put_last(outputList, NULL);
        else
            list_put_last(outputList, expv_reduce(compile_expression(x2), FALSE));
    }

    v = expv_cons(F_INQUIRE_STATEMENT, NULL, callArgs, outputList);
    output_statement(v);
    return;
}


void
compile_NAMELIST_decl(x)
     expr x;
{
    ID nlId;
    list lp, lq;
    TYPE_DESC tp;
    expr nlName;
    expr idList;
    expr nlVX;

    FOR_ITEMS_IN_LIST(lp, x) {
        nlName = EXPR_ARG1(LIST_ITEM(lp));
        idList = EXPR_ARG2(LIST_ITEM(lp));

        nlId = declare_ident(EXPR_SYM(nlName), CL_UNKNOWN);
        tp = new_type_desc();
        TYPE_BASIC_TYPE(tp) = TYPE_NAMELIST;
        declare_id_type(nlId, tp);
        ID_ORDER(nlId) = order_sequence++;

        ID_LINE(nlId) = EXPR_LINE(x); /* set line_no */
        if (ID_CLASS(nlId) == CL_UNKNOWN) {
            /*
             * First.
             */
            if (NL_LIST(nlId) == NULL) {
                NL_LIST(nlId) = list0(LIST);
            }
            declare_ident(EXPR_SYM(nlName), CL_NAMELIST);
        } else {
            if (ID_CLASS(nlId) != CL_NAMELIST) {
                error("'%s' is not a namelist.", SYM_NAME(ID_SYM(nlId)));
                continue;
            }
        }

        FOR_ITEMS_IN_LIST(lq, idList) {
            nlVX = LIST_ITEM(lq);
            if (EXPR_CODE(nlVX) != IDENT) {
                error("invalid type in namelist.");
                continue;
            }
            NL_LIST(nlId) = list_put_last(NL_LIST(nlId), LIST_ITEM(lq));
        }
    }
}
