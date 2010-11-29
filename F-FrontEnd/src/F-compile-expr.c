/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-compile-expr.c
 */

#include "F-front.h"
#include <math.h>

static expv compile_args _ANSI_ARGS_((expr args));
static expv compile_data_args _ANSI_ARGS_((expr args));

static expv compile_implied_do_expression _ANSI_ARGS_((expr x));
static expv compile_dup_decl _ANSI_ARGS_((expv x));
static expv compile_array_constructor _ANSI_ARGS_((expr x));
static char compile_array_ref_dimension _ANSI_ARGS_((expr x, expv dims, expv subs));
static expv compile_member_array_ref  _ANSI_ARGS_((expr x, expv v));

struct replace_item replace_stack[MAX_REPLACE_ITEMS];
struct replace_item *replace_sp = replace_stack;


/*
 * Convert expr terminal node to expv terminal node.
 */
expv
compile_terminal_node(x)
     expr x;
{
    expv ret = NULL;

    if (!(EXPR_CODE_IS_TERMINAL_OR_CONST(EXPR_CODE(x)))) {
        fatal("compile_terminal_node: not a terminal.");
        return NULL;
    }

    switch (EXPR_CODE(x)) {

        case STRING_CONSTANT: {
            int len = strlen(EXPR_STR(x));
            if(len > 132 * 39)
                len = CHAR_LEN_UNFIXED;
            ret = expv_str_term(STRING_CONSTANT, type_char(len),
                                EXPR_STR(x));
            break;
        }

        case F_DOUBLE_CONSTANT:
        case FLOAT_CONSTANT: {
            TYPE_DESC tp = type_basic((EXPR_CODE(x) == F_DOUBLE_CONSTANT) ?
                TYPE_DREAL : TYPE_REAL);
            ret = expv_float_term(FLOAT_CONSTANT, tp, EXPR_FLOAT(x),
                EXPR_ORIGINAL_TOKEN(x));
            break;
        }

        case F_TRUE_CONSTANT: {
            ret = expv_int_term(INT_CONSTANT, type_basic(TYPE_LOGICAL), 1);
            break;
        }

        case F_FALSE_CONSTANT: {
            ret = expv_int_term(INT_CONSTANT, type_basic(TYPE_LOGICAL), 0);
            break;
        }

        case INT_CONSTANT: {
            ret = expv_int_term(INT_CONSTANT, type_basic(TYPE_INT), EXPR_INT(x));
            break;
        }

        case COMPLEX_CONSTANT: {
            expv re = NULL;
            expv im = NULL;
            TYPE_DESC tp = NULL;

            if (expr_is_constant(EXPR_ARG1(x))) {
                re = expr_constant_value(EXPR_ARG1(x));
            } else {
                error("non constant expression (real) is in complex constant.");
                break;
            }

            if (expr_is_constant(EXPR_ARG2(x))) {
                im = expr_constant_value(EXPR_ARG2(x));
            } else {
                error("non constant expression (imag) is in complex constant.");
                break;
            }

            if (re == NULL || im == NULL) {
                fatal("compile_terminal_node: can't create complex constant.");
                break;
            } else if ((!(IS_NUMERIC(EXPV_TYPE(re)))) ||
                       (!(IS_NUMERIC(EXPV_TYPE(im))))) {
                error("non numeric expression(s) is in complex constant.");
                break;
            }

            /* Last check before cons. */
            if (expr_is_constant(re) == FALSE) {
                error("about to create complex constant with non-constant (real).");
                break;
            }
            if (expr_is_constant(im) == FALSE) {
                error("about to create complex constant with non-constant (imag).");
                break;
            }

            tp = max_type(EXPV_TYPE(re), EXPV_TYPE(im));
            if (IS_REAL(tp) == FALSE) {
                /* Use DREAL. */
                tp = type_basic(TYPE_DREAL);
            }

            re = expv_reduce_conv_const(tp, re);
            im = expv_reduce_conv_const(tp, im);
            ret = expv_cons(
                COMPLEX_CONSTANT, 
                type_basic((TYPE_BASIC_TYPE(tp) == TYPE_DREAL) ?
                    TYPE_DCOMPLEX : TYPE_COMPLEX),
                re, im);
            break;
        }

        case IDENT: {
            ret = compile_ident_expression(x);
            break;
        }

        case F_VAR:
        case F_PARAM:
        case F_FUNC:
        case ID_LIST:
        case F_ASTERISK:
            ret = x;
            break;

        default: {
            ret = NULL;
            break;
        }
    }

    return ret;
}

TYPE_DESC
bottom_type(type)
    TYPE_DESC type;
{
    TYPE_DESC tp = type;

    while(IS_ARRAY_TYPE(tp)) {
        tp = TYPE_REF(tp);
    }

    return tp;
}


static TYPE_DESC
force_to_logica_type(expv v)
{
    TYPE_DESC tp = EXPV_TYPE(v);
    TYPE_DESC tp1 = type_basic(TYPE_LOGICAL);
    TYPE_ATTR_FLAGS(tp1) = TYPE_ATTR_FLAGS(tp);
    EXPV_TYPE(v) = tp1;
    return tp1;
}


enum binary_expr_type {
    ARITB,
    RELAB,
    LOGIB,
};

/* evaluate expression */
expv
compile_expression(expr x)
{
    expr x1;
    expv left,right,right2,v;
    expv shape,lshape,rshape;
    TYPE_DESC lt,rt,tp = NULL;
    TYPE_DESC bLType,bRType,bType = NULL;
    ID id = NULL;
    enum expr_code op;
    enum binary_expr_type biop;
    expv v1, v2;
    SYMBOL s;
    int is_userdefined = FALSE;
    char * error_msg = NULL;

    if (x == NULL) {
        return NULL;
    }

    if (EXPR_CODE_IS_TERMINAL_OR_CONST(EXPR_CODE(x))) {
        return compile_terminal_node(x);
    }

    if(EXPR_CODE_SYMBOL(EXPR_CODE(x)) != NULL) {
        if (find_symbol_without_allocate(EXPR_CODE_SYMBOL(EXPR_CODE(x))) != NULL)
        is_userdefined = TRUE;
    }

    switch (EXPR_CODE(x)) {

        case F_ARRAY_REF: {     /* (F_ARRAY_REF name args) */
            x1 = EXPR_ARG1(x);
            if (EXPR_CODE(x1) == F_ARRAY_REF) {
                /* substr of character array */
                return compile_substr_ref(x);

            } else if (EXPR_CODE(x1) == IDENT) {
                s = EXPR_SYM(x1);
                id = find_ident(s);
                if(id == NULL) {
                    id = declare_ident(s,CL_UNKNOWN);
                    if(id == NULL)
                        return NULL;
                }
                v = NULL;
                tp = ID_TYPE(id);
            } else {
                id = NULL;
                v = compile_lhs_expression(x1);
                if (v == NULL) /* error recovery. */
                    return NULL;
                tp = EXPV_TYPE(v);

                if (EXPR_CODE(v) == F95_MEMBER_REF) {
                    return compile_member_array_ref(x,v);
                }
            }

            if (id == NULL || (
                EXPR_ARG2(x) != NULL && 
                EXPR_ARG1(EXPR_ARG2(x)) != NULL &&
                EXPR_CODE(EXPR_ARG1(EXPR_ARG2(x))) == F95_TRIPLET_EXPR)) {

                if (IS_ARRAY_TYPE(tp)) {
                    return compile_array_ref(id, v, EXPR_ARG2(x), FALSE);
                } else if (IS_CHAR(tp)) {
                    return compile_substr_ref(x);
                } else {
                    if(id)
                        error("%s is not array nor character", ID_NAME(id));
                    else
                        error("not array nor character", ID_NAME(id));
                    goto err;
                }
            }

            if (ID_CLASS(id) == CL_PROC ||
                ID_CLASS(id) == CL_ENTRY ||
                ID_CLASS(id) == CL_UNKNOWN) {
                expv vRet = NULL;
                if (ID_CLASS(id) == CL_PROC && IS_SUBR(ID_TYPE(id))) {
                    error("function invocation of subroutine");
                    goto err;
                }
                if (ID_STORAGE(id) == STG_ARG) {
                    vRet = compile_highorder_function_call(id,
                                                           EXPR_ARG2(x),
                                                           FALSE);
                } else {
                    vRet = compile_function_call(id, EXPR_ARG2(x));
                }
                return vRet;
            }
            if (ID_CLASS(id) == CL_TAGNAME) {
                return compile_struct_constructor(id, EXPR_ARG2(x));
            }
            return compile_array_ref(id, NULL, EXPR_ARG2(x), FALSE);
        }

        case XMP_COARRAY_REF: /* (XMP_COARRAY_REF expr args) */

	  return compile_coarray_ref(x);

        /* implied do expression */
        case F_IMPLIED_DO: {     /* (F_IMPLIED_DO loop_spec do_args) */
            return compile_implied_do_expression(x);
        }

        case F_UNARY_MINUS_EXPR: {
            v = compile_expression(EXPR_ARG1(x));
            if(v == NULL)
                return NULL;
            tp = getBaseType(EXPV_TYPE(v));
            if (!IS_NUMERIC(tp) && !IS_GENERIC_TYPE(tp)) {
                error_msg = "nonarithmetic operand of negation";
            }
            if (error_msg != NULL) {
                if(is_userdefined) {
                    tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
                } else {
                    goto err;
                }
            } else {
                tp = EXPV_TYPE(v);
            }
            return expv_cons(UNARY_MINUS_EXPR,tp,v,NULL);
        }

        /* arithmetic expression */
        case F_PLUS_EXPR:   op = PLUS_EXPR;   biop = ARITB; goto binary_op;
        case F_MINUS_EXPR:  op = MINUS_EXPR;  biop = ARITB; goto binary_op;
        case F_MUL_EXPR:    op = MUL_EXPR;    biop = ARITB; goto binary_op;
        case F_DIV_EXPR:    op = DIV_EXPR;    biop = ARITB; goto binary_op;
        /* power operator */
        case F_POWER_EXPR: op = POWER_EXPR;   biop = ARITB; goto binary_op;

        /* relational operator */
        case F_EQ_EXPR:  op = LOG_EQ_EXPR;    biop = RELAB; goto binary_op;
        case F_NE_EXPR:  op = LOG_NEQ_EXPR;   biop = RELAB; goto binary_op;
        case F_GT_EXPR:  op = LOG_GT_EXPR;    biop = RELAB; goto binary_op;
        case F_GE_EXPR:  op = LOG_GE_EXPR;    biop = RELAB; goto binary_op;
        case F_LT_EXPR:  op = LOG_LT_EXPR;    biop = RELAB; goto binary_op;
        case F_LE_EXPR:  op = LOG_LE_EXPR;    biop = RELAB; goto binary_op;

        /* logical operator */
        case F_EQV_EXPR:    op = F_EQV_EXPR;    biop = LOGIB; goto binary_op;
        case F_NEQV_EXPR:   op = F_NEQV_EXPR;   biop = LOGIB; goto binary_op;
        case F_OR_EXPR:     op = LOG_OR_EXPR;   biop = LOGIB; goto binary_op;
        case F_AND_EXPR:    op = LOG_AND_EXPR;  biop = LOGIB; goto binary_op;

        binary_op: {
            left = compile_expression(EXPR_ARG1(x));
            right = compile_expression(EXPR_ARG2(x));
            if (left == NULL || right == NULL) {
                goto err;
            }

            lt = EXPV_TYPE(left);
            rt = EXPV_TYPE(right);
            if(rt == NULL)
                right = compile_expression(EXPR_ARG2(x));

            bLType = bottom_type(lt);
            bRType = bottom_type(rt);

            switch (biop) {
            case ARITB:
                if ((!IS_NUMERIC(bLType) && !IS_GENERIC_TYPE(bLType)) ||
                    (!IS_NUMERIC(bRType) && !IS_GENERIC_TYPE(bRType))) {
                    error_msg = "nonarithmetic operand of arithmetic operator";
                }

                if(error_msg == NULL) {
                    bType = max_type(bLType, bRType);
                } else if(is_userdefined) {
                    tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
                } else {
                    goto err;
                }

                break;

            case RELAB:
                if (IS_CHAR(bLType) || IS_CHAR(bRType) ||
                    IS_LOGICAL(bLType) || IS_LOGICAL(bRType)) {
                    if (TYPE_BASIC_TYPE(bLType) != TYPE_BASIC_TYPE(bRType)) {
                        error_msg = "illegal comparison";
                    }
                } else if (IS_COMPLEX(bLType) || IS_COMPLEX(bRType)) {
                    if (op != LOG_EQ_EXPR && op!= LOG_NEQ_EXPR) {
                        error_msg = "order comparison of complex data";
                    }
                } else if ((!IS_NUMERIC(bLType) && !IS_GENERIC_TYPE(bLType)) ||
                    (!IS_NUMERIC(bRType) && !IS_GENERIC_TYPE(bRType))) {
                    error_msg = "comparison of nonarithmetic data";
                }

                if(error_msg == NULL) {
                    bType = type_basic(TYPE_LOGICAL);
                } else if(is_userdefined) {
                    tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
                } else {
                    goto err;
                }

                break;

            case LOGIB:
                /*
                 * FIXME
                 * If the function defined in CONTAINS at end of
                 * block, its type is implicitly declared at
                 * this context. To avoid error, we forced to
                 * change the type to logical.
                 */
                if(!IS_LOGICAL(bLType) &&
                    EXPV_CODE(left) == FUNCTION_CALL &&
                    TYPE_IS_IMPLICIT(bLType)) {
                    bLType = force_to_logica_type(left);
                    lt = bLType;
                }
                if(!IS_LOGICAL(bRType) &&
                    EXPV_CODE(right) == FUNCTION_CALL &&
                    TYPE_IS_IMPLICIT(bRType)) {
                    bRType = force_to_logica_type(right);
                    rt = bRType;
                }

                if ((!IS_LOGICAL(bLType) && !IS_GENERIC_TYPE(bLType) && !IS_GNUMERIC_ALL(bLType)) ||
                    (!IS_LOGICAL(bRType) && !IS_GENERIC_TYPE(bRType) && !IS_GNUMERIC_ALL(bRType))) {
                    error_msg = "logical expression required";
                }

                if(error_msg == NULL) {
                    bType = type_basic(TYPE_LOGICAL);
                } else if(is_userdefined) {
                    tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
                } else {
                    goto err;
                }

                break;

            default:
                error("internal compiler error");
                goto err;
            }

            if(tp == NULL) {

                if (TYPE_N_DIM(lt) != TYPE_N_DIM(rt) &&
                    (TYPE_N_DIM(lt) != 0 && TYPE_N_DIM(rt) != 0)) {
                    error("operation between different rank array. ");
                    goto err;
                }

                lshape = list0(LIST);
                rshape = list0(LIST);

                generate_shape_expr(lt,lshape);
                generate_shape_expr(rt,rshape);

                shape = max_shape(lshape, rshape, bType == bLType);

                if(shape == NULL) {
                    delete_list(lshape);
                    delete_list(rshape);
                    goto err;
                }

                /* NOTE:
                 * if type and shape is same with the original type,
                 * then type of expression is not created but used with
                 * the origianl type.
                 *
                 * if operator is logical operator,
                 * don't care about type but only shape.
                 */
                if((biop == LOGIB || bType == bLType) && shape == lshape) {
                    tp = lt;
                } else if ((biop == LOGIB || bType == bRType) && shape == rshape) {
                    tp = rt;
                } else {
                    /* NOTE:
                     * if shape is scalar (list0(LIST)), tp = bType.
                     */
                    tp = compile_dimensions(bType, shape);
                    fix_array_dimensions(tp);
                }

                delete_list(lshape);
                delete_list(rshape);
            }

            return expv_cons(op,tp,left,right);
        }

        case F_NOT_EXPR: {
            /* accept logical array. */
            v = compile_logical_expression_with_array(EXPR_ARG1(x));
            if (v == NULL)
                goto err;
            return expv_cons(LOG_NOT_EXPR,EXPV_TYPE(v),v,NULL);
        }

        case F_CONCAT_EXPR: {
            left = compile_expression(EXPR_ARG1(x));
            right = compile_expression(EXPR_ARG2(x));
            if (left == NULL || right == NULL) {
                goto err;
            }
            lt = EXPV_TYPE(left);
            rt = EXPV_TYPE(right);
            if ((!IS_CHAR(lt) && !IS_GNUMERIC(lt) && !IS_GNUMERIC_ALL(lt)) ||
                (!IS_CHAR(rt) && !IS_GNUMERIC(rt) && !IS_GNUMERIC_ALL(rt))) {
                error("concatenation of nonchar data");
                goto err;
            }

            {
                int l1 = TYPE_CHAR_LEN(lt);
                int l2 = TYPE_CHAR_LEN(rt);
                tp = type_char((l1 <= 0 || l2 <=0) ? 0 : l1 + l2);
            }

            return expv_cons(F_CONCAT_EXPR,tp,left,right);
        }

        case F95_USER_DEFINED_BINARY_EXPR:
        case F95_USER_DEFINED_UNARY_EXPR:
        {
            expr id = EXPR_ARG1(x);
            left = compile_expression(EXPR_ARG2(x));
            if (left == NULL) {
                goto err;
            }
            right = compile_expression(EXPR_ARG3(x));
            tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);

            if (right != NULL)
                return expv_user_def_cons(F95_USER_DEFINED_BINARY_EXPR,tp,id,left,right);
            else
                return expv_user_def_cons(F95_USER_DEFINED_UNARY_EXPR,tp,id,left,right);
        }


        case F_LABEL_REF: {
            error("label argument is not supported");
            break;
        }

        case F95_TRIPLET_EXPR: {
            left = compile_expression(EXPR_ARG1(x));
            right = compile_expression(EXPR_ARG2(x));
            right2 = compile_expression(EXPR_ARG3(x));
	    if (right && EXPR_CODE(right) == F_ASTERISK) right = NULL;
            if ((EXPR_ARG1(x) && left == NULL) ||
                //(EXPR_ARG2(x) && right == NULL) ||
                (EXPR_ARG3(x) && right2 == NULL)) {
                goto err;
            }
            return list3(F_INDEX_RANGE, left, right, right2);
        }

        case F95_CONSTANT_WITH:  {
            v1 = compile_expression(EXPR_ARG1(x)); /* constant */
            if(v1 == NULL)
                return NULL;
            assert(EXPV_TYPE(v1));
            v2 = compile_expression(EXPR_ARG2(x)); /* kind */
        calc_kind:
            if (v2 != NULL) {
                v2 = expv_reduce(v2, FALSE);
                if (v2 == NULL) {
                    error("bad expression in constant kind parameter");
                    break;
                }
                if(expr_is_constant_typeof(v2, TYPE_INT) == FALSE) {
                    error("bad expression in constant kind parameter");
                    break;
                }
                if(TYPE_BASIC_TYPE(EXPV_TYPE(v1)) == TYPE_DREAL) {
                    error("kind parameter with 'd' exponent");
                    break;
                }
            }
            /* if kind is differ, make new type desc.
               for example, type desc for l2 and l3 should be differ.
               l2 = .false.
               l3 = .false._1
            */
            TYPE_KIND(EXPV_TYPE(v1)) = v2;
            return v1;
        }

        case F95_TRUE_CONSTANT_WITH:
        case F95_FALSE_CONSTANT_WITH:
            v1 = expv_int_term(INT_CONSTANT, type_basic(TYPE_LOGICAL),
                (EXPR_CODE(x) == F95_TRUE_CONSTANT_WITH));
            v2 = compile_expression(EXPR_ARG1(x));
            goto calc_kind;

        case F_SET_EXPR:
            return compile_set_expr(x);

        case F95_MEMBER_REF: {
            expv v = compile_member_ref(x);
            /* TODO
             * if type of member is array
             * and type of argument is array,
             * then raise error.
             */
            return v;
        }

        case F_DUP_DECL:
            return compile_dup_decl(x);

        case F95_KIND_SELECTOR_SPEC:
            if(expr_is_constant(EXPR_ARG1(x)))
                return compile_int_constant(EXPR_ARG1(x));
            else
                /* allow any expression
                 * ex) kind = selected_int_kind(..)
                 */
                return compile_expression(EXPR_ARG1(x));

        case F95_LEN_SELECTOR_SPEC: {
            expr v;
            if(EXPR_ARG1(x) == NULL)
                return expv_any_term(F_ASTERISK, NULL);
            v = compile_expression(EXPR_ARG1(x));
            if(expr_is_specification(v))
                return v;
            return compile_int_constant(EXPR_ARG1(x));
        }
        case PLUS_EXPR:
        case MINUS_EXPR:
        case DIV_EXPR:
        case MUL_EXPR:
        case POWER_EXPR:
        case ARRAY_REF:
        case UNARY_MINUS_EXPR:
        case FUNCTION_CALL:
            /* already compiled */
            return x;

        case F95_ARRAY_CONSTRUCTOR:
            return compile_array_constructor(x);

        default: {
            fatal("compile_expression: unknown code '%s'",
                  EXPR_CODE_NAME(EXPR_CODE(x)));
        }
    }

    err:
    if(error_msg != NULL)
        error(error_msg);
    return NULL;
}


static expv
is_statement_function_or_replace(ID id)
{
    struct replace_item *rp;

    if(replace_stack == replace_sp)
        return NULL;

    rp = replace_sp;
    /* check if name is on the replace list? */
    while(rp-- > replace_stack) {
        if(rp->id == id) return rp->v;
    }

    return NULL;
}


/* evaluates identifier as expression.
 * translates him to Var expression.
 */
expv
compile_ident_expression(expr x)
{
    ID id;
    SYMBOL sym;
    expv v;

    if(EXPR_CODE(x) != IDENT) {
        goto err;
    }

    sym = EXPR_SYM(x);

    /*
     * FIXME:
     *  declare_ident() makes an id in struct member if in struct
     *  compilation. To avoid this, call find_ident() to check the id
     *  exists or not.
     */
    if ((id = find_ident(sym)) == NULL) {
        id = declare_ident(sym, CL_UNKNOWN);
    }

    /* check if name is on the replace list? */
    if((v = is_statement_function_or_replace(id)) != NULL)
        return v;

    if((id = declare_variable(id)) == NULL)
        goto err;

    EXPV_CODE(x) = F_VAR;
    EXPV_TYPE(x) = ID_TYPE(id);
    EXPV_NAME(x) = ID_SYM(id);

    return x;

err:
    fatal("%s: invalid code", __func__);
    return NULL;
}

int
substr_length(expv x)
{
    int length;
    expv upper,lower;

    if(EXPV_CODE(x) != F_INDEX_RANGE)
        return CHAR_LEN_UNFIXED;

    lower = EXPR_ARG1(x);
    upper = EXPR_ARG2(x);

    if (lower == NULL && upper == NULL) {
        return CHAR_LEN_UNFIXED;
    }

    if (lower == NULL && upper != NULL) {
        if (EXPV_CODE(upper) == INT_CONSTANT) {
            return EXPV_INT_VALUE(upper);
        } else {
            return CHAR_LEN_UNFIXED;
        }
    }

    if(lower == NULL ||
       EXPV_CODE(lower) != INT_CONSTANT)
        return CHAR_LEN_UNFIXED;

    if(upper == NULL ||
       EXPV_CODE(upper) != INT_CONSTANT)
        return CHAR_LEN_UNFIXED;

    length = EXPV_INT_VALUE(upper) - EXPV_INT_VALUE(lower) + 1;

    return length;
}

expv
compile_substr_ref(expr x)
{
    int length;
    expv v1, v2, retv, v, dims;
    TYPE_DESC charType, tp;

    if (EXPR_CODE(x) != F_ARRAY_REF) {
        /* substr is recognized as F_ARRAY_REF by parser */
        return NULL;
    }
    v1 = compile_expression(EXPR_ARG1(x));
    if (v1 == NULL)
        return NULL;
    if(!IS_CHAR(bottom_type(EXPV_TYPE(v1)))){
        error("substring for non character");
        return NULL;
    }
    v = EXPR_ARG2(x);
    if (v == NULL || EXPR_ARG1(v) == NULL) {
        return NULL;
    }
    if (EXPR_CODE(EXPR_ARG1(v)) != F95_TRIPLET_EXPR) {
        fatal("not F95_TRIPLET_EXPR");
        return NULL;
    }
    v2 = compile_expression(EXPR_ARG1(v));
    retv = list2(F_SUBSTR_REF, v1, v2);

    length = substr_length(v2);

    charType = type_char(length);

    /* succeed shape form child. */
    dims = list0(LIST);
    generate_shape_expr(EXPV_TYPE(v1), dims);
    tp = compile_dimensions(charType, dims);
    fix_array_dimensions(tp);

    EXPV_TYPE(retv) = tp;

    return retv;
}

/* evaluate left-hand side expression */
/* x = ident 
 *    | (F_SUBSTR ident substring)
 *    | (F_ARRAY_REF ident fun_arg_list)
 *    | (F_SUBSTR (F_ARRAY_REF ident fun_arg_list) substring)
 *    | (F95_MEMBER_REF expression ident)
 *    | (XMP_COARRAY_REF expression image_selector)
 */
expv
compile_lhs_expression(x)
     expr x;
{
    expr x1;
    expv v;
    TYPE_DESC tp;
    ID id;
    SYMBOL s;

    switch(EXPR_CODE(x)){
    case F_ARRAY_REF: {/* (F_ARRAY_REF ident fun_arg_list) */
        /* ident must be CL_VAR */
        x1 = EXPR_ARG1(x);
        if(EXPR_CODE(x1) == F_ARRAY_REF) {
            /* substr of character array */
            return compile_substr_ref(x);
        }

        if (EXPR_CODE(x1) == IDENT) {
            s = EXPR_SYM(x1);
            id = find_ident(s);
            if(id == NULL) {
                id = declare_ident(s,CL_UNKNOWN);
                if(id == NULL)
                    goto err;
            }
            v = NULL;
            tp = ID_TYPE(id);
        } else {
            id = NULL;
            v = compile_lhs_expression(x1);
            if(v == NULL)
                goto err;

            tp = EXPV_TYPE(v);

            if (EXPR_CODE(v) == F95_MEMBER_REF) {
                return compile_member_array_ref(x,v);
            }
        }

        if (id == NULL || (
            EXPR_ARG2(x) != NULL && EXPR_ARG1(EXPR_ARG2(x)) != NULL
                && EXPR_CODE(EXPR_ARG1(EXPR_ARG2(x))) == F95_TRIPLET_EXPR)) {
            if (IS_ARRAY_TYPE(tp)) {
                return compile_array_ref(id, v, EXPR_ARG2(x), TRUE);
            } else if (IS_CHAR(tp)) {
                return compile_substr_ref(x);
            } else {
                error("%s is not arrray nor character", ID_NAME(id));
                goto err;
            }
        }

        if(!IS_ARRAY_TYPE(tp)){
            error("subscripts on scalar variable, '%s'",ID_NAME(id));
            goto err;
        }
        if((v = compile_array_ref(id, NULL, EXPR_ARG2(x), TRUE)) == NULL)
            goto err;

        return v;
        }

    case IDENT:         /* terminal */
        s = EXPR_SYM(x);
        id = declare_ident(s,CL_UNKNOWN);

        /* check if name is on the replace list? */
        if((v = is_statement_function_or_replace(id)) != NULL)
            return v;
        
        if((id = declare_variable(id)) == NULL)
            goto err;

        return ID_ADDR(id);

    case F95_MEMBER_REF:
        return compile_member_ref(x);

    case XMP_COARRAY_REF:
      return compile_coarray_ref(x);

    default:
        fatal("compile_lhs_expression: unknown code");
        /* error ? */
    }
 err:
    return NULL;
}

int
expv_is_this_func(expv v)
{
    ID id;

    switch(EXPV_CODE(v)) {
    case IDENT:
    case F_VAR:
    case F_FUNC:
        id = find_ident(EXPV_NAME(v));
        if (ID_CLASS(id) == CL_PROC) {
            return (PROC_CLASS(id) == P_THISPROC);
        }
        break;
    default:
        break;
    }
    return FALSE;
}

int
expv_is_lvalue(expv v)
{
    if (v == NULL) return FALSE;
    if (EXPV_IS_RVALUE(v) == TRUE) return FALSE;
    if (EXPR_CODE(v) == ARRAY_REF || EXPR_CODE(v) == F_VAR ||
	EXPR_CODE(v) == F95_MEMBER_REF || EXPR_CODE(v) == XMP_COARRAY_REF)
        return TRUE;
    if (expv_is_this_func(v))
        return TRUE;
    return FALSE;
}

int
expv_is_str_lvalue(expv v)
{
#if 0
    if(!IS_CHAR(EXPV_TYPE(v))) return FALSE;
#endif
    if(!IS_CHAR(bottom_type(EXPV_TYPE(v)))) return FALSE;

    if(EXPR_CODE(v) == F_SUBSTR_REF ||
       EXPR_CODE(v) == F_VAR) 
        return TRUE;
    return FALSE;
}


/* compile into integer constant */
expv
compile_int_constant(expr x)
{
    expv v;

    if((v = compile_expression(x)) == NULL) return NULL;
    if((v = expv_reduce(v, FALSE)) == NULL) return NULL;
    if (expr_is_constant_typeof(v, TYPE_INT)) {
        return v;
    } else {
        error("integer constant is required");
        return NULL;
    }
}


static expv
compile_logical_expression0(expr x, int allowArray)
{
    expv v;
    TYPE_DESC tp;

    if((v = compile_expression(x)) == NULL) return NULL;

    tp = EXPV_TYPE(v);
    if(allowArray && IS_ARRAY_TYPE(tp)) {
        tp = array_element_type(tp);
        if(IS_LOGICAL(tp) == FALSE) {
            error("logical array expression is required");
            return NULL;
        }
    } else if(!IS_LOGICAL(tp) &&
              !IS_GENERIC_TYPE(tp) &&
              !IS_GNUMERIC_ALL(tp)) {
        /*
         * FIXME
         * If the function defined in CONTAINS at end of block,
         * its type is implicitly declared at this context.
         * To avoid error, we forced to change the type to logical.
         */
        if(EXPV_CODE(v) == FUNCTION_CALL &&
            TYPE_IS_IMPLICIT(tp)) {
            (void)force_to_logica_type(v);
        } else {
            error("logical expression is required");
            return NULL;
        }
    }
    return v;
}


expv
compile_logical_expression(expr x)
{
    return compile_logical_expression0(x, FALSE);
}


expv
compile_logical_expression_with_array(expr x)
{
    return compile_logical_expression0(x, TRUE);
}


expv
expv_assignment(v1,v2)
     expv v1,v2;
{
    /* check assignment operator is user defined or not. */
    if(find_symbol_without_allocate(EXPR_CODE_SYMBOL(F95_ASSIGNOP)) != NULL)
        return expv_cons(F_LET_STATEMENT, NULL, v1, v2);

    if(EXPV_IS_RVALUE(v1) == TRUE) {
        error("bad left hand side expression in assignment");
        return NULL;
    }

    TYPE_DESC tp1 = EXPV_TYPE(v1);
    TYPE_DESC tp2 = EXPV_TYPE(v2);

    if (EXPV_CODE(v2) == F95_ARRAY_CONSTRUCTOR) {
        if (!IS_ARRAY_TYPE(tp1)) {
            error("lhs expr is not an array.");
            return NULL;
        }
    }
    if(EXPV_CODE(v2) != FUNCTION_CALL &&
              type_is_compatible_for_assignment(tp1, tp2) == FALSE) {
        error("incompatible type in assignment");
        return NULL;
    }

    return expv_cons(F_LET_STATEMENT, NULL, v1, v2);
}


static int
checkSubscriptIsInt(expv v)
{
    if(v == NULL) return TRUE;
    TYPE_DESC tp = EXPV_TYPE(v);

    if(IS_ARRAY_TYPE(tp) && IS_INT(array_element_type(tp)) == FALSE) {
        error_at_node(v,
            "subscript must be integer or integer array expression");
        return FALSE;
    }

    return TRUE;
}


/**
 * \brief Compiles a dimension expression from args.
 *
 * @return NULL if some errors occurred.
 */
static char
compile_array_ref_dimension(expr args, expv dims, expv subs)
{
    int n = 0;
    list lp;
    char err_flag = FALSE;

    assert(dims != NULL);

    FOR_ITEMS_IN_LIST(lp, args) {
        expr v, d = LIST_ITEM(lp);
        if((v = compile_expression(d)) == NULL){
            err_flag = TRUE;
            continue;
        }

        EXPR_LINE(v) = EXPR_LINE(d);

        switch(EXPR_CODE(v)) {
        case F_INDEX_RANGE:
            /* lower, upper, step */
            if(checkSubscriptIsInt(EXPR_ARG1(v)) == FALSE ||
                checkSubscriptIsInt(EXPR_ARG2(v)) == FALSE ||
                checkSubscriptIsInt(EXPR_ARG3(v)) == FALSE) {
                err_flag = TRUE;

                continue;
            }

            list_put_last(dims, v);
            ++n;

            break;

        default:
            if(checkSubscriptIsInt(v) == FALSE) {
                err_flag = TRUE;
                continue;
            }

            if(IS_ARRAY_TYPE(EXPV_TYPE(v))) {
                /* support vector subscripts.
                 *
                 * ex)
                 * REAL,DIMENSION(1:5)    :: array = (/1.0, 0.0, 2.0, 0.0, 3.0/)
                 * REAL,DIMENSION(1:3)    :: new_array
                 * INTEGER,DIMENSION(1:3) :: index = (/1, 3, 5/)
                 *
                 * new_array = array(index(1:3))
                 *
                 * print *, new_array ! must be prinetd as 1.0 2.0 3.0
                 *
                 * Checkes must be done as follows.
                 *
                 * 1) A vector subscript must be integer type and one dimensional array.
                 * 2) All elements of the vector subscript must be indeces of the array.
                 * 3) Internal files must not be a vector subscripted subarray.
                 * 4) A duplicated vector subscripted subarray must not be
                 *    a left expression of a pointer assignment statement.
                 *
                 * Sorry, but only 1st check is implemented now.
                 */
                TYPE_DESC tq = EXPV_TYPE(v);

                expr shape;
                expr dim = NULL;

                if(TYPE_N_DIM(tq) != 1) {
                    error_at_node(v, "is an array of rank %d", TYPE_N_DIM(tq));
                    return TRUE;
                }
                shape = list0(LIST);
                generate_shape_expr(tq,shape);

                tq = bottom_type(tq);
                if(IS_INT(tq) == FALSE) {
                    error_at_node(v, "is not an integer array.");
                    return TRUE;
                }

                if (is_variable_shape(shape)) {
                    dim = list2(LIST, NULL, NULL);

                } else {
                    int size;
                    expr lb,ub;

                    lb = EXPR_ARG1(EXPR_ARG1(shape));
                    ub = EXPR_ARG2(EXPR_ARG1(shape));
                    size = EXPV_INT_VALUE(ub) - EXPV_INT_VALUE(lb) + 1;
                    dim = list2(LIST, expv_constant_1, expv_int_term(INT_CONSTANT,type_INT,size));
                }

                list_put_last(dims, dim);

                ++n;
            }

            break;
        }

        if(n > MAX_DIM){
            error_at_node(args,"too many subscripts");
            return TRUE;
        }

        if(subs != NULL)
            list_put_last(subs, v);
    }

    return err_flag;
}


expv
compile_array_ref(ID id, expv vary, expr args, int isLeft)
{
    int n;
    list lp;
    expv subs;
    expv dims;
    TYPE_DESC tp;

    assert((id && vary == NULL) || (id == NULL && vary));

    tp = (id ? ID_TYPE(id) : EXPV_TYPE(vary));

    if (id != NULL && (
        PROC_CLASS(id) == P_EXTERNAL ||
        PROC_CLASS(id) == P_DEFINEDPROC ||
        (ID_STORAGE(id) == STG_ARG &&
         !(IS_ARRAY_TYPE(tp)) &&
         isLeft == FALSE))) {
        return compile_highorder_function_call(id, args, FALSE);
    }

    if(!IS_ARRAY_TYPE(tp)) fatal("%s: not ARRAY_TYPE", __func__);
    if(!TYPE_DIM_FIXED(tp)) fix_array_dimensions(tp);

    subs = list0(LIST);
    dims = list0(LIST);

    /* get dims and subs*/
    if(compile_array_ref_dimension(args, dims, subs)) {
        return NULL;
    }

    tp = compile_dimensions(bottom_type(tp), dims);
    fix_array_dimensions(tp);

    if(tp == NULL) return NULL;

    n = 0;
    FOR_ITEMS_IN_LIST(lp, dims) {
        n++;
    }

    if(TYPE_N_DIM(tp) != n){
        if(id) {
            error_at_node(
                args,"wrong number of subscript on '%s'",ID_NAME(id));
        } else {
            error_at_node(
                args,"wrong number of subscript");
        }

        return NULL;
    }

    if(id != NULL) {
        vary = expv_sym_term(F_VAR, ID_TYPE(id), ID_SYM(id));
        ID_ADDR(id) = vary;

        if (TYPE_N_DIM(ID_TYPE(id)) < n) {
            error_at_node(args, "too large dimension, %d.", n);
            return NULL;
        }
    }

    return expv_reduce(expv_cons(ARRAY_REF,
                                 tp, vary, subs), FALSE);
}


// find the coindexed, that is, rightmost id
ID
find_coindexed_id(expr ref){

  SYMBOL s;
  ID id;
  
  switch (EXPR_CODE(ref)){

  case IDENT:

    s = EXPR_SYM(ref);
    id = find_ident(s);
    
    if (!id){
      id = declare_ident(s, CL_UNKNOWN);
    }

    return id;

  case F95_MEMBER_REF: {

/*     expr parent = EXPR_ARG1(ref); */
/*     expr child = EXPR_ARG2(ref); */
/*     TYPE_DESC parent_type; */

/*     assert(EXPR_CODE(child) == IDENT); */

/* /\*     stVTyp = EXPV_TYPE(ref); *\/ */

/* /\*     if (IS_ARRAY_TYPE(stVTyp)){ *\/ */
/* /\*       stVTyp = bottom_type(stVTyp); *\/ */
/* /\*     } *\/ */

/*     if (EXPR_CODE(parent) == IDENT){ */
/*       s = EXPR_SYM(ref); */
/*       id = find_ident(s); */
/*       if (!id){ */
/* 	id = declare_ident(s, CL_UNKNOWN); */
/*       } */
/*       parent_type = ID_TYPE(id); */
/*     } */

/*     return find_struct_member(parent_type, EXPR_SYM(child)); */

    return find_coindexed_id(EXPR_ARG2(ref));
  }

  case F_ARRAY_REF:

    return find_coindexed_id(EXPR_ARG1(ref));

  default:

    error("wrong object coindexed");
    return NULL;
  }
}


// find the coindexed, that is, rightmost id
TYPE_DESC
get_rightmost_id_type(expv ref){

  if (!ref) return NULL;

  switch (EXPV_CODE(ref)){

  case F95_MEMBER_REF: {

    expr parent = EXPV_LEFT(ref);
    expr child = EXPV_RIGHT(ref);
    TYPE_DESC parent_type;
    ID member_id;

    assert(EXPR_CODE(child) == IDENT);

    parent_type = EXPV_TYPE(parent);

    if (IS_ARRAY_TYPE(parent_type)){
      parent_type = bottom_type(parent_type);
    }

    member_id = find_struct_member(parent_type, EXPR_SYM(child));
    return ID_TYPE(member_id);
  }

  case ARRAY_REF:

    return EXPV_TYPE(EXPV_LEFT(ref));

  default:

    return EXPV_TYPE(ref);
  }
}

int is_array(expv obj){

  assert(EXPV_CODE(obj) == ARRAY_REF);

  list lp;
  FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(obj)){
    expr x = LIST_ITEM(lp);
    if (EXPR_CODE(x) == LIST || EXPR_CODE(x) == F_INDEX_RANGE){
      return TRUE;
    }
  }

  return FALSE;
}


int
check_ancestor(expv obj){

  // Note: nested coarrays should have been checked when compiling types.

  assert(obj);

  TYPE_DESC tp;

  if (EXPV_CODE(obj) == F95_MEMBER_REF){

    expv parent = EXPV_LEFT(obj);
    if (!check_ancestor(parent)) return FALSE;

    expv child = EXPV_RIGHT(obj);

    assert(EXPR_CODE(child) == IDENT);

    TYPE_DESC parent_type = EXPV_TYPE(parent);

    if (IS_ARRAY_TYPE(parent_type)){
      parent_type = bottom_type(parent_type);
    }

    ID member_id = find_struct_member(parent_type, EXPR_SYM(child));
    tp = ID_TYPE(member_id);

    if (IS_ARRAY_TYPE(tp) || TYPE_IS_ALLOCATABLE(tp) || TYPE_IS_POINTER(tp)) return FALSE;

  }
  else if (EXPV_CODE(obj) == ARRAY_REF){

    if (is_array(obj)) return FALSE;

    expv var = EXPV_LEFT(obj);

    if (EXPV_CODE(var) == F95_MEMBER_REF){

      expv parent = EXPV_LEFT(var);
      if (!check_ancestor(parent)) return FALSE;

      expv child = EXPV_RIGHT(var);

      assert(EXPR_CODE(child) == IDENT);

      TYPE_DESC parent_type = EXPV_TYPE(parent);

      if (IS_ARRAY_TYPE(parent_type)){
	parent_type = bottom_type(parent_type);
      }

      ID member_id = find_struct_member(parent_type, EXPR_SYM(child));
      tp = ID_TYPE(member_id);

    }
    else {
      tp = EXPV_TYPE(var);
    }

    if (TYPE_IS_ALLOCATABLE(tp) || TYPE_IS_POINTER(tp)) return FALSE;

  }
  else {
    tp = EXPV_TYPE(obj);
    if (IS_ARRAY_TYPE(tp) || TYPE_IS_ALLOCATABLE(tp) || TYPE_IS_POINTER(tp)) return FALSE;
  }

  return TRUE;
}


int is_in_alloc = FALSE;

expv
compile_coarray_ref(expr coarrayRef){

  expr ref = EXPR_ARG1(coarrayRef);
  expr image_selector = EXPR_ARG2(coarrayRef);

  TYPE_DESC tp = NULL;
  list lp;

  //
  // (1) process the object coindexed.
  //

  expv obj = compile_expression(ref);

  if (obj){

    expv obj2 = obj;
    
    if (EXPV_CODE(obj) == ARRAY_REF){
      obj2 = EXPV_LEFT(obj);
    }

    if (EXPV_CODE(obj2) == F95_MEMBER_REF){

      expv parent = EXPV_LEFT(obj2);

      if (!check_ancestor(parent)){
	error_at_node(coarrayRef, "Each ancestor of the coarray component must be a non-allocatable, non-pointer scalar.");
	return NULL;
      }
    }
  }

  tp = get_rightmost_id_type(obj);
  if (!tp) return NULL;
  if (!tp->codims){
    error_at_node(coarrayRef, "Only coarrays can be coindexed.");
    return NULL;
  }

  //
  // (2) process the cosubscripts.
  //

  expv cosubs = list0(LIST);
  expv codims = list0(LIST);

  /* get codims and cosubs*/
  if (compile_array_ref_dimension(image_selector, codims, cosubs)){
    return NULL;
  }

/*   tp = compile_dimensions(bottom_type(tp), codims); */
/*   fix_array_dimensions(tp); */

  int n = 0;
  FOR_ITEMS_IN_LIST(lp, cosubs){

    if (is_in_alloc){

      expr x = LIST_ITEM(lp);
      expr lower = NULL, upper = NULL;

      if (!x) {
	if (!LIST_NEXT(lp))
	  error("only last cobound may be \"*\"");
      }
      else if (EXPR_CODE(x) == LIST){ /* (LIST lower upper NULL) */
	lower = EXPR_ARG1(x);
	upper = EXPR_ARG2(x);
	assert(!EXPR_ARG3(x));
      }
      else if (EXPR_CODE(x) == F_INDEX_RANGE) {
	lower = EXPR_ARG1(x);
	upper = EXPR_ARG2(x);
	assert(!EXPR_ARG3(x));
      }
      else {
	upper = x;
      }

      if (LIST_NEXT(lp)){
	if ((upper && EXPV_CODE(upper) == F_ASTERISK) || !upper)
	  error("Only last upper-cobound can be \"*\".");
      }

      if (!LIST_NEXT(lp)){
	if (!upper){
	  ;
	}
	else if (EXPV_CODE(upper) == F_ASTERISK){
	  upper = NULL;
	}
	else {
	  error("Last upper-cobound must be \"*\".");
	}
      }
    }

    n++;
  }

  if (tp->codims->corank != n){
    error_at_node(image_selector, "wrong number of cosubscript on '%s'");
    return NULL;
  }

/*   if (id){ */
/*     vary = expv_sym_term(F_VAR, ID_TYPE(id), ID_SYM(id)); */
/*     ID_ADDR(id) = vary; */

/*     if (TYPE_N_DIM(ID_TYPE(id)) < n){ */
/*       error_at_node(args, "too large dimension, %d.", n); */
/*       return NULL; */
/*     } */
/*   } */

  return expv_reduce(expv_cons(XMP_COARRAY_REF, EXPV_TYPE(obj), obj, cosubs),
		     FALSE);
}


char*
genFunctionTypeID(EXT_ID ep)
{
    char buf[128];
    sprintf(buf, "F" ADDRX_PRINT_FMT, Addr2Uint(ep));
    return strdup(buf);
}


expv
compile_highorder_function_call(ID id, expr args, int isCall)
{
    if (ID_STORAGE(id) != STG_ARG) {
        fatal("%s: '%s' is not a dummy arg.",
              __func__, SYM_NAME(ID_SYM(id)));
        /* not reached. */
        return NULL;
    } else {
        /*
         * A high order sub program invocation.
         */
        expv ret;
        enum name_class sNC = ID_CLASS(id);
        int sUAF = VAR_IS_USED_AS_FUNCTION(id);

        ID_CLASS(id) = CL_UNKNOWN;
        VAR_IS_USED_AS_FUNCTION(id) = TRUE;
        
        (void)declare_external_id_for_highorder(id, isCall);
        ret = compile_function_call(id, args);

        ID_CLASS(id) = sNC;
        VAR_IS_USED_AS_FUNCTION(id) = sUAF;

        if (isCall == TRUE) {
            EXPV_TYPE(ret) = type_SUBR;
            VAR_IS_USED_AS_FUNCTION(id) = TRUE;
        }

        return ret;
    }
}


expv
compile_function_call(ID f_id, expr args) {
    expv a, v = NULL;
    EXT_ID ep = NULL;
    TYPE_DESC tp = NULL;

    /* declare as function */
    if (declare_function(f_id) == NULL) return NULL;

    switch (PROC_CLASS(f_id)) {
        case P_UNDEFINEDPROC:
            /* f_id is not defined yet. */
	  //tp = ID_TYPE(f_id);
  	    tp = ID_TYPE(f_id) ? ID_TYPE(f_id) : new_type_desc();
            TYPE_SET_USED_EXPLICIT(tp);

            a = compile_args(args);
#if 0
            ep = PROC_EXT_ID(f_id);
            if (ID_DEFINED_BY(f_id) != NULL) {
                ep = PROC_EXT_ID(ID_DEFINED_BY(f_id));
            }
            if (ep == NULL) {
                ep = new_external_id_for_external_decl(ID_SYM(f_id), type_GNUMERIC_ALL);
                PROC_EXT_ID(f_id) = ep;
            }
            EXT_TAG(ep) = STG_EXT;
            EXT_IS_DEFINED(ep) = TRUE;
#endif
            v = list3(FUNCTION_CALL, ID_ADDR(f_id), a,
                expv_any_term(F_EXTFUNC, f_id));
            EXPV_TYPE(v) = type_GNUMERIC_ALL;
            break;

        case P_THISPROC:
            if (!TYPE_IS_RECURSIVE(ID_TYPE(f_id))) {
                error("recursive call in not a recursive function");
            }
            /* FALL THROUGH */
        case P_DEFINEDPROC:
        case P_EXTERNAL:
            if (ID_TYPE(f_id) == NULL) {
                error("attempt to use untyped function,'%s'",ID_NAME(f_id));
                goto err;
            }
	    //            tp = ID_TYPE(f_id);
  	    tp = ID_TYPE(f_id) ? ID_TYPE(f_id) : new_type_desc();
            TYPE_SET_USED_EXPLICIT(tp);
            a = compile_args(args);
            ep = PROC_EXT_ID(f_id);
            if (ID_DEFINED_BY(f_id) != NULL) {
                ep = PROC_EXT_ID(ID_DEFINED_BY(f_id));
                tp = ID_TYPE(ID_DEFINED_BY(f_id));
            }
            if (ep == NULL) {
                ep = new_external_id_for_external_decl(ID_SYM(f_id), ID_TYPE(f_id));
                PROC_EXT_ID(f_id) = ep;
            }

            v = list3(FUNCTION_CALL, ID_ADDR(f_id), a,
                expv_any_term(F_EXTFUNC, f_id));
            EXPV_TYPE(v) = IS_SUBR(tp)?type_SUBR:(IS_GENERIC_TYPE(tp)?type_GNUMERIC_ALL:tp);
            break;

        case P_INTRINSIC:
            v = compile_intrinsic_call(f_id, compile_data_args(args));
            break;

        case P_STFUNCT:
            v = statement_function_call(f_id, compile_args(args));
            break;

        default:
            fatal("%s: unknown proc_class %d", __func__,
                  PROC_CLASS(f_id));
    }
    return(v);
 err:
    return NULL;
}

expv
compile_struct_constructor(ID struct_id, expr args)
{
    ID member;
    list lp;
    expv v, result, l;

    assert(ID_TYPE(struct_id) != NULL);
    l = list0(LIST);
    result = list1(F95_STRUCT_CONSTRUCTOR, l);

    EXPV_TYPE(result) = find_struct_decl(ID_SYM(struct_id));
    assert(EXPV_TYPE(result) != NULL);

    if(args == NULL)
        return result;

    EXPV_LINE(result) = EXPR_LINE(args);
    lp = EXPR_LIST(args);
    FOREACH_MEMBER(member, ID_TYPE(struct_id)) {
        assert(ID_TYPE(member) != NULL);
        if (lp == NULL || LIST_ITEM(lp) == NULL) {
            error("not all member are specified.");
            return NULL;
        }
        v = compile_expression(LIST_ITEM(lp));
        assert(EXPV_TYPE(v) != NULL);
        if (!type_is_compatible_for_assignment(ID_TYPE(member),
                                               EXPV_TYPE(v))) {
            error("type is not applicable in struct constructor");
            return NULL;
        }
        list_put_last(l, v);
        lp = LIST_NEXT(lp);
    }

    if (lp != NULL) {
        error("Too much elements in struct constructor");
        return NULL;
    }
    return result;
}


static expv
compile_args(expr args)
{
    list lp;
    expr a;
    expv v, arglist;
    ID id;

    arglist = list0(LIST);
    if (args == NULL) return arglist;

    FOR_ITEMS_IN_LIST(lp, args) {
        a = LIST_ITEM(lp);
        /* check function address */
        if (EXPR_CODE(a) == IDENT) {
            id = find_ident(EXPR_SYM(a));
            if (id == NULL)
                id = declare_ident(EXPR_SYM(a), CL_UNKNOWN);
            switch (ID_CLASS(id)) {
            case CL_PROC:
            case CL_ENTRY:
                if (PROC_CLASS(id) != P_THISPROC &&
                    (declare_function(id) == NULL)) {
                    continue;
                }
                break;
            case CL_VAR: 
            case CL_UNKNOWN:
                /* check variable name */
                declare_variable(id);
                break;
            case CL_PARAM:
                break;

            default: 
                error("illegal argument");
                continue;
            }
        }
        if ((v = compile_expression(a)) == NULL) continue;
        if ((v = expv_reduce(v, FALSE)) == NULL) continue;

        arglist = list_put_last(arglist, v);
    }

    return arglist;
}


static expv
compile_data_args(expr args)
{
    list lp;
    expr a,v,arglist;

    if(args == NULL) return NULL;
    arglist = list0(LIST);
    FOR_ITEMS_IN_LIST(lp,args){
        a = LIST_ITEM(lp);
        v = compile_expression(a);
        arglist = list_put_last(arglist,v);
    }
    return arglist;
}


static expv
genCastCall(const char *name, TYPE_DESC tp, expv arg, expv kind)
{
    SYMBOL sym;
    expv symV, args;
    TYPE_DESC ftp;
    EXT_ID extid;
    ID id;

    sym = find_symbol(strdup(name));
    symV = expv_sym_term(F_FUNC, NULL, sym);
    id = find_ident(sym);

    if(id == NULL)
        id = declare_ident(sym, CL_PROC);

    ftp = function_type(tp);
    TYPE_SET_INTRINSIC(ftp);

    extid = new_external_id_for_external_decl(sym, ftp);
    EXT_PROC_CLASS(extid) = EP_INTRINSIC;
    ID_TYPE(id) = ftp;
    PROC_EXT_ID(id) = extid;

    if(kind)
        args = list2(LIST, arg, kind);
    else
        args = list1(LIST, arg);

    return expv_cons(FUNCTION_CALL, tp, symV, args);
}


/* stementment function call */
expv
statement_function_call(f_id, arglist)
     ID f_id;
     expv arglist;
{
    list arg_lp,param_lp;
    ID id;
    TYPE_DESC idtp, vtp;
    expv v;
    struct replace_item *old_sp;

    if(PROC_STBODY(f_id) == NULL) return NULL; /* error */

    if(ID_TYPE(f_id) == NULL){
        error("attempt to use untyped statement function");
        return NULL;
    }

    old_sp = replace_sp;        /* save */

    arg_lp = EXPV_LIST(arglist);
    param_lp = EXPV_LIST(PROC_ARGS(f_id));
    /* copy actual arguments into temporaries */
    while(arg_lp != NULL && param_lp != NULL){
        v = LIST_ITEM(arg_lp);
        id = declare_ident(EXPR_SYM(LIST_ITEM(param_lp)),CL_UNKNOWN);
        replace_sp->id = id;
        replace_sp->v = v;
        if(++replace_sp >= &replace_stack[MAX_REPLACE_ITEMS])
            fatal("too nested statement function call");
        
        arg_lp = LIST_NEXT(arg_lp);
        param_lp = LIST_NEXT(param_lp);
    }
    
    if(arg_lp != NULL || param_lp != NULL){
        error("statement function definition and argument list differ");
        goto err;
    }

    if((v = compile_expression(PROC_STBODY(f_id))) == NULL) goto err;

    idtp = ID_TYPE(f_id);
    vtp = EXPV_TYPE(v);

    if (idtp && vtp) {
        /* call type casting intrinsic */
        TYPE_DESC tp1, tp2;
        BASIC_DATA_TYPE b1, b2;
        const char *castName = NULL;
        expv kind = NULL;

        tp1 = bottom_type(idtp);
        tp2 = bottom_type(vtp);
        b1 = TYPE_BASIC_TYPE(tp1);
        b2 = TYPE_BASIC_TYPE(tp2);
        kind = TYPE_KIND(tp1);

        if(b1 != b2 || TYPE_KIND(tp1)) {
            switch(b1) {
            case TYPE_INT:
                castName = "int"; break;
            case TYPE_DREAL:
                kind = expv_int_term(
                    INT_CONSTANT, type_INT, KIND_PARAM_DOUBLE);
                /* fall through */
            case TYPE_REAL:
                castName = "real"; break;
            case TYPE_DCOMPLEX:
                kind = expv_int_term(
                    INT_CONSTANT, type_INT, KIND_PARAM_DOUBLE);
                /* fall through */
            case TYPE_COMPLEX:
                castName = "cmplx"; break;
            default:
                break;
            }
        }

        if(castName) {
            v = genCastCall(castName, tp1, v, kind);
        }
    }

    replace_sp = old_sp;        /* restore */
    return v;

 err:
    replace_sp = old_sp;        /* restore */
    return NULL;
}


/* 
 * power expression
 */
/* runtime library
 * pow_ii: INTEGER*INTGER -> INTEGER
 * pow_ri: REAL*INTGER -> DREAL
 * pow_di: DREAL*INTGER -> DREAL
 * pow_ci: COMPLEX**INTEGER -> COMPLEX
 * pow_dd: DREAL*DREAL -> DREAL
 * pow_hh, pow_zz, pow_zi, pow_qq is not used
 */
expv expv_power_expr(expv left,expv right)
{
    TYPE_DESC lt,rt,tp;

    /* check constant expression */
    left = expv_reduce(left, FALSE);
    right = expv_reduce(right, FALSE);

    lt = EXPV_TYPE(left);
    rt = EXPV_TYPE(right);

    int lisary = IS_ARRAY_TYPE(lt);
    int risary = IS_ARRAY_TYPE(rt);

    if((lisary && !IS_NUMERIC(array_element_type(lt))) ||
        (risary && !IS_NUMERIC(array_element_type(rt))) ||
        (lisary == FALSE && risary == FALSE &&
        (!IS_NUMERIC(lt) || !IS_NUMERIC(rt)))) {

        error("nonarithmetic operand of power operator");
        return NULL;
    }

    if (IS_COMPLEX(lt)) {
        tp = type_basic(TYPE_COMPLEX);
    } else if (IS_REAL(lt) || IS_REAL(rt)) {
        tp = type_basic(TYPE_REAL);
    } else {
        tp = type_basic(TYPE_INT);
    }

    return expv_cons(POWER_EXPR, tp, left, right);
}


/*
 * implied do expression
 */
static expv
compile_implied_do_expression(expr x)
{
    expv do_var, do_init, do_limit, do_incr;
    expr var, init, limit, incr;
    SYMBOL do_var_sym;
    CTL *cp;

    expr loopSpec = EXPR_ARG1(x);

    var = EXPR_ARG1(loopSpec);
    init = EXPR_ARG2(loopSpec);
    limit = EXPR_ARG3(loopSpec);
    incr = EXPR_ARG4(loopSpec);

    if (EXPR_CODE(var) != IDENT) {
        fatal("compile_implied_do_expression: DO var is not IDENT");
    }
    do_var_sym = EXPR_SYM(var);
    
    /* check nested loop with the same variable */
    for (cp = ctls; cp < ctl_top; cp++) {
        if(CTL_TYPE(cp) == CTL_DO && CTL_DO_VAR(cp) == do_var_sym) {
            error("nested loops with variable '%s'", SYM_NAME(do_var_sym));
            break;
        }
    }

    do_var = compile_lhs_expression(var);
    if (!expv_is_lvalue(do_var))
        error("compile_implied_do_expression: bad DO variable");
    do_init = expv_reduce(compile_expression(init), FALSE);
    do_limit = expv_reduce(compile_expression(limit), FALSE);
    if (incr != NULL) do_incr = expv_reduce(compile_expression(incr), FALSE);
    else do_incr = expv_constant_1;
    expv x1 = list4(LIST, do_var, do_init, do_limit, do_incr);

    list lp;
    expv v = EXPR_ARG2(x);
    expv x2 = list0(LIST);
    int nItems = 0;
    TYPE_DESC retTyp = NULL;
    FOR_ITEMS_IN_LIST(lp, v) {
        x2 = list_put_last(x2, compile_expression(LIST_ITEM(lp)));
        nItems++;
    }
    if (nItems > 0) {
        retTyp = EXPV_TYPE(EXPR_ARG1(x2));
    }

    return expv_cons(F_IMPLIED_DO, retTyp, x1, x2);
}


/**
 * (F_DUP_DECL INT_CONSTANT CONSTANT)
 */
static expv
compile_dup_decl(expv x)
{
    expv numV = expr_constant_value(EXPR_ARG1(x));

    if (numV == NULL) {
        error("number of data value not integer constant.");
        return NULL;
    }

    if(IS_INT(EXPV_TYPE(numV)) == FALSE) {
        error("multiplier is not an integer value.");
        return NULL;
    }

#ifdef DATA_C_IMPL
    if (EXPR_CODE(numV) == F_VAR) {
        numV = expv_reduce(numV, TRUE);
    }
#endif /* DATA_C_IMPL */

    expv valV = expr_constant_value(EXPR_ARG2(x));

    if (valV == NULL) {
        error("data value not constant.");
        return NULL;
    }

    if(EXPV_CODE(numV) == INT_CONSTANT && EXPV_INT_VALUE(numV) <= 0) {
        error("illegal initialize value list number.");
        return NULL;
    }

    return expv_cons(F_DUP_DECL, EXPV_TYPE(valV), numV, valV);
}

static expv
compile_array_constructor(expr x)
{
    int nDims = 0;
    list lp;
    expv v, res, l;
    TYPE_DESC tp;
    BASIC_DATA_TYPE elem_type = TYPE_UNKNOWN;

    l = list0(LIST);
    res = list1(F95_ARRAY_CONSTRUCTOR, l);
    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        nDims++;
        v = compile_expression(LIST_ITEM(lp));
        list_put_last(l, v);
        if (elem_type == TYPE_UNKNOWN) {
            elem_type = get_basic_type(EXPV_TYPE(v));
            continue;
        }
        if (get_basic_type(EXPV_TYPE(v)) != elem_type) {
            error("Array constructor elements have different data types.");
            return NULL;
        }
    }

    assert(elem_type != TYPE_UNKNOWN);
    if (elem_type == TYPE_CHAR) {
        tp = type_char(-1);
    } else {
        tp = type_basic(elem_type);
    }

    EXPV_TYPE(res) = compile_dimensions(tp, list1(LIST,
        (list2(LIST, expv_constant_1, expv_int_term(INT_CONSTANT,type_INT,nDims)))));

    return res;
}


expv
compile_member_array_ref(expr x, expv v)
{
    expr indices = EXPR_ARG2(x);
    expv org_expr = EXPR_ARG1(v); // if v is member ref, org_expr is member_id.
    TYPE_DESC tq, tp;
    expv shape = list0(LIST);

    /* compile (org_expr)%id(indices)
     * x = ARRAY_REF(v, indices)
     * v = MEMBER_REF(org_expr, id)
     */
    tq = EXPV_TYPE(org_expr);

#if 0
    if(compile_array_ref_dimension(indices,shape,NULL)) {
        return NULL;
    }
    // Checkes two or more nonzero rank arrey references are not appeared.
    // i.e) a(1:5)%n(1:5) not accepted
    if(((EXPV_CODE(org_expr) == F_ARRAY_REF) &&
        (IS_ARRAY_TYPE(tq) && IS_CHAR(bottom_type(tq)) == FALSE)) &&
       (EXPR_LIST(shape) != NULL)) {
        error("Two or more part references with nonzero rank must not be specified");
        return NULL;
    }
#endif
    if(EXPR_LIST(shape) == NULL) {
        generate_shape_expr(tq, shape);
    }

    tp = EXPV_TYPE(v);
    if (IS_ARRAY_TYPE(tp)) {
        TYPE_DESC new_tp;
        expv new_v = compile_array_ref(NULL, v, indices, TRUE);
        new_tp = EXPV_TYPE(new_v);

        if((TYPE_IS_POINTER(tp) || TYPE_IS_TARGET(tp)) &&
           !(TYPE_IS_POINTER(new_tp) || TYPE_IS_TARGET(new_tp))) {
            TYPE_DESC btp = bottom_type(new_tp);
            if(!EXPR_HAS_ARG1(shape))
                generate_shape_expr(new_tp, shape);
            btp = wrap_type(btp);
            TYPE_ATTR_FLAGS(btp) |= TYPE_IS_POINTER(tp) | TYPE_IS_TARGET(tp);
            new_tp = btp;
        }
        new_tp = compile_dimensions(new_tp, shape);
        fix_array_dimensions(new_tp);
        EXPV_TYPE(new_v) = new_tp;
        return new_v;
    } else if (IS_CHAR(tp)) {
        return compile_substr_ref(x);
    } else {
        error_at_node(v, "Subscripted object is neither array nor character.");
        return NULL;
    }
}
