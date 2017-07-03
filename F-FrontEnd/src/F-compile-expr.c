/**
 * \file F-compile-expr.c
 */

#include "F-front.h"
#include "F-second-pass.h"
#include <math.h>

static expv compile_data_args _ANSI_ARGS_((expr args));

static expv compile_implied_do_expression _ANSI_ARGS_((expr x));
static expv compile_dup_decl _ANSI_ARGS_((expv x));
static expv compile_array_constructor _ANSI_ARGS_((expr x));
static int compile_array_ref_dimension _ANSI_ARGS_((expr x, expv dims, expv subs));
static expv compile_member_array_ref  _ANSI_ARGS_((expr x, expv v));

struct replace_item replace_stack[MAX_REPLACE_ITEMS];
struct replace_item *replace_sp = replace_stack;

static TYPE_DESC getLargeIntType();

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

        case F_QUAD_CONSTANT:
        case F_DOUBLE_CONSTANT:
        case FLOAT_CONSTANT: {
            TYPE_DESC tp = type_basic((EXPR_CODE(x) == F_DOUBLE_CONSTANT) ?
                TYPE_DREAL : TYPE_REAL);
            ret = expv_float_term(FLOAT_CONSTANT, tp, EXPR_FLOAT(x),
                EXPR_ORIGINAL_TOKEN(x));
            if (ret != NULL && EXPR_CODE(x) == F_QUAD_CONSTANT) {
                /*
                 * We won't introduce TYPE_QUAD. Instead, quad-real
                 * constant is treated as a constant with a kind
                 * specifier _16.
                 */
                TYPE_KIND(EXPV_TYPE(ret)) =
                    expv_int_term(INT_CONSTANT, type_INT, 16);
            }
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
            omllint_t n;
            ret = expv_int_term(INT_CONSTANT, type_basic(TYPE_INT), EXPR_INT(x));
            n = EXPV_INT_VALUE(ret);
            if (n > INT_MAX) {
                EXPV_TYPE(ret) = getLargeIntType();
            }
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
force_to_logical_type(expv v)
{
    TYPE_DESC tp = EXPV_TYPE(v);
    TYPE_DESC tp1 = type_basic(TYPE_LOGICAL);
    TYPE_ATTR_FLAGS(tp1) = TYPE_ATTR_FLAGS(tp);
    EXPV_TYPE(v) = tp1;
    return tp1;
}


int
are_dimension_and_shape_conformant_by_type(expr x,
                                           TYPE_DESC lt, TYPE_DESC rt,
                                           expv *shapePtr) {
    int ret = FALSE;
    expv lShape = list0(LIST);
    expv rShape = list0(LIST);

    if (lt == NULL || rt == NULL) {
        fatal("%s: at least a TYPE_DESC is NULL.", __func__);
        /* not reached. */
        return FALSE;
    }

    generate_shape_expr(lt, lShape);
    generate_shape_expr(rt, rShape);

    if (shapePtr != NULL) {
        *shapePtr = NULL;
    }

    /*
     * NOTE:
     *	For type check between dummy args and actual args, the lt must
     *	be the dummy arg type, means, treat the left hand type always as
     *	an assignment destination type.
     */
    if (TYPE_N_DIM(lt) > 0 &&
        TYPE_N_DIM(rt) > 0 &&
        TYPE_N_DIM(lt) == TYPE_N_DIM(rt)) {
        int nDims = TYPE_N_DIM(lt);
        int i;
        expv laSpec;
        expv raSpec;
        expv aSpec;
        int laSz;
        int raSz;
        expv retShape = list0(LIST);

        for (i = 0; i < nDims; i++) {
            aSpec = NULL;
            laSpec = expr_list_get_n(lShape, i);
            raSpec = expr_list_get_n(rShape, i);
            laSz = array_spec_size(laSpec, NULL, NULL);
            raSz = array_spec_size(raSpec, NULL, NULL);
            if (laSz > 0 && raSz > 0) {
                if (laSz == raSz) {
                    /*
                     * Both the array-specs are identical. Use left.
                     */
                    aSpec = laSpec;
                } else {
                    if (x != NULL) {
                        error_at_node(x,
                                      "Subscript #%d array-spec size differs, "
                                      "%d and %d.", i + 1, laSz, raSz);
                    } else {
                        error("Subscript #%d array-spec size differs, "
                              "%d and %d.", i + 1, laSz, raSz);
                    }
                    goto Done;
                }
            } else if (laSz > 0 && raSz < 0) {
                /*
                 * Use left.
                 */
                aSpec = laSpec;
            } else if (laSz < 0 && raSz > 0) {
                /*
                 * Use right.
                 */
                aSpec = raSpec;
            } else {
                /*
                 * FIXME: 
                 *	I'm not so sure about this case.
                 */
                aSpec = combine_array_specs(laSpec, raSpec);
            }

            if (aSpec == NULL) {
                fatal("must not happen but can't determine which "
                      "array-spec to use.");
            }

            list_put_last(retShape, aSpec);
        }

        if (shapePtr != NULL) {
            *shapePtr = retShape;
        }

        ret = TRUE;

    } else if (type_is_assumed_size_array(lt) == TRUE &&
               TYPE_N_DIM(rt) > 0) {
        /*
         * An assumed size array accept all arrays. Only the size
         * matters. So check basic types.
         */
        ret = TRUE;
    } else {
        if (x != NULL) {
            error_at_node(x,
                          "incompatible dimension for the operation, "
                          "%d and %d.",
                          TYPE_N_DIM(lt), TYPE_N_DIM(rt));
        } else {
            error("incompatible dimension for the operation, %d and %d.",
                  TYPE_N_DIM(lt), TYPE_N_DIM(rt));
        }
    }

    Done:
    return ret;
}


static int
are_dimension_and_shape_conformant(expr x, 
                                   expv left, expv right,
                                   expv *shapePtr) {
    TYPE_DESC lt = EXPV_TYPE(left);
    TYPE_DESC rt = EXPV_TYPE(right);

    return are_dimension_and_shape_conformant_by_type(x, lt, rt, shapePtr);
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
    int type_is_not_fixed = FALSE;

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
                if(ID_IS_AMBIGUOUS(id)) {
                    error_at_node(x, "an ambiguous reference to symbol '%s'",
                                  ID_NAME(id));
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
                        error("%s is not an array nor a character", ID_NAME(id));
                    else
                        error("not array nor character", ID_NAME(id));
                    goto err;
                }
            }

            ID_LINE(id) = EXPR_LINE(x); // set line number

            if (ID_CLASS(id) == CL_PROC ||
                ID_CLASS(id) == CL_ENTRY ||
                ID_CLASS(id) == CL_MULTI ||
                ID_CLASS(id) == CL_UNKNOWN) {
                expv vRet = NULL;
                if (ID_CLASS(id) == CL_PROC && IS_SUBR(ID_TYPE(id))) {
                    if (PROC_CLASS(id) == P_EXTERNAL &&
                        PROC_IS_FUNC_SUBR_AMBIGUOUS(id) == TRUE) {
                        error("'%s' is not yet determined as a function or "
                              "a subroutine.", ID_NAME(id));
                    } else {
                        error("'%s' is a subroutine, not a function.",
                              ID_NAME(id));
                    }
                    goto err;
                }

#if 0
		// to be resolved
                if (is_in_module() == FALSE &&
                    CURRENT_STATE == INEXEC &&
                    is_intrinsic_function(id) == FALSE &&
                    ID_TYPE(id) == NULL) {
                    implicit_declaration(id);
                    if (ID_TYPE(id) == NULL) {
                        /*
                         * NOTE:
                         *	We don't have to care about the external'd
                         *	ids, well, I think.
                         */
                        error("'%s' could be a function but the type is "
                              "still unknown.", ID_NAME(id));
                        /*
                         * Choke futher error message by classifying the id.
                         */
                        ID_CLASS(id) = CL_PROC;
                        goto err;
                    }
                }
#endif

                if (ID_IS_DUMMY_ARG(id)) {
                    vRet = compile_highorder_function_call(id,
                                                           EXPR_ARG2(x),
                                                           FALSE);
                } else {
                    vRet = compile_function_call(id, EXPR_ARG2(x));
                }
                return vRet;
            }

            if (ID_CLASS(id) == CL_TAGNAME) {
                return compile_struct_constructor(id, NULL, EXPR_ARG2(x));
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
            if (TYPE_IS_NOT_FIXED(tp)) {
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

/* FEAST change start */
            /* if (TYPE_IS_NOT_FIXED(bLType) || TYPE_IS_NOT_FIXED(bRType)) { */
            /*     type_is_not_fixed = TRUE; */
            /* } */
            if(bLType == NULL || bRType == NULL){
                type_is_not_fixed = TRUE;
            } else if (TYPE_IS_NOT_FIXED(bLType) || TYPE_IS_NOT_FIXED(bRType)) {
                type_is_not_fixed = TRUE;
            }
/* FEAST change end */

            switch (biop) {
            case ARITB:
                if (!type_is_not_fixed) {
                    if ((!IS_NUMERIC(bLType) && !IS_GENERIC_TYPE(bLType)) ||
                        (!IS_NUMERIC(bRType) && !IS_GENERIC_TYPE(bRType))) {
                        error_msg = "nonarithmetic operand of arithmetic operator";
                    }
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
                if (!type_is_not_fixed) {
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
                    bLType = force_to_logical_type(left);
                    lt = bLType;
                }
                if(!IS_LOGICAL(bRType) &&
                    EXPV_CODE(right) == FUNCTION_CALL &&
                    TYPE_IS_IMPLICIT(bRType)) {
                    bRType = force_to_logical_type(right);
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

            if (tp == NULL) {
                int expDim = 0;

                /*
                 * First of all, check if which side is reshaped type.
                 */
/* FEAST add start */
                if(!lt || !rt){
                  type_is_not_fixed = TRUE;
                  goto doEmit;
                }
/* FEAST add end */                
                if (TYPE_IS_RESHAPED(lt) || TYPE_IS_RESHAPED(rt)) {
                    if (bType == NULL) {
                        tp = TYPE_IS_RESHAPED(rt) ? lt : rt;
                        goto doEmit;
                    } else {
                        shape = list0(LIST);
                        generate_shape_expr(TYPE_IS_RESHAPED(rt) ? lt : rt,
                            shape);
                        tp = compile_dimensions(bType, shape);
                        fix_array_dimensions(tp);
                        delete_list(shape);
                        goto doEmit;
                    }
                }

                /*
                 * Check the dimension of the both expressions.
                 */
                if (TYPE_N_DIM(lt) != TYPE_N_DIM(rt) &&
                    (TYPE_N_DIM(lt) != 0 && TYPE_N_DIM(rt) != 0)) {
                    error_at_node(x,
                                  "operation between different rank array.");
                    error_at_node(x, "left dim %d != right dim %d.",
                                  TYPE_N_DIM(lt), TYPE_N_DIM(rt));
                    goto err;
                }

                lshape = list0(LIST);
                rshape = list0(LIST);

                generate_shape_expr(lt, lshape);
                generate_shape_expr(rt, rshape);

                /*
                 * Check if (ARRAY binop scalar) or (scalar binop
                 * ARRAY). In these case, ALWAYS use ARRAY's dimension
                 * and the basic type must be the bType.
                 */
                if (TYPE_N_DIM(lt) == 0 && TYPE_N_DIM(rt) != 0) {
                    expDim = TYPE_N_DIM(rt);
                    shape = rshape;
                } else if (TYPE_N_DIM(lt) != 0 && TYPE_N_DIM(rt) == 0) {
                    expDim = TYPE_N_DIM(lt);
                    shape = lshape;
                }
                if (expDim != 0) {
                    tp = compile_dimensions(bType, shape);
                    fix_array_dimensions(tp);
                    goto doEmit;
                }

                /*
                 * Then check the shape. After the above check, both
                 * the left and the right are/aren't array references
                 * so check the case both are array ref.
                 */
                if (TYPE_N_DIM(lt) > 0 && TYPE_N_DIM(rt) > 0 &&
                    are_dimension_and_shape_conformant(x, left, right, 
                                                       &shape) == TRUE) {
                    tp = compile_dimensions(bType, shape);
                    fix_array_dimensions(tp);
                    goto doEmit;
                }

                shape = max_shape(lshape, rshape, bType == bLType);

                if(shape == NULL) {
                    delete_list(lshape);
                    delete_list(rshape);
		    error("operation between non-conformable arrays. ");
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

            doEmit:
/* FEAST CHANGE start */
            /* if (type_is_not_fixed) */
            if(type_is_not_fixed && tp)
/* FEAST CHANGE end */
                TYPE_SET_NOT_FIXED(bottom_type(tp));
            return expv_cons(op, tp, left, right);
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
        {
            expr id = EXPR_ARG1(x);
            left = compile_expression(EXPR_ARG2(x));
            if (left == NULL) {
                goto err;
            }
            if ((right = compile_expression(EXPR_ARG3(x))) == NULL) {
                goto err;
            }
            tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);

            return expv_user_def_cons(F95_USER_DEFINED_BINARY_EXPR,tp,id,left,right);
        }

        case F95_USER_DEFINED_UNARY_EXPR:
        {
            expr id = EXPR_ARG1(x);
            left = compile_expression(EXPR_ARG2(x));
            if (left == NULL) {
                goto err;
            }
            tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);

            return expv_user_def_cons(F95_USER_DEFINED_UNARY_EXPR,tp,id,left,NULL);
        }


        case F_LABEL_REF: {
            error("label argument is not supported");
            break;
        }

        case F95_TRIPLET_EXPR: {
            expv retV = NULL;
            left = compile_expression(EXPR_ARG1(x));
            right = compile_expression(EXPR_ARG2(x));
            right2 = compile_expression(EXPR_ARG3(x));
	    if (right && EXPR_CODE(right) == F_ASTERISK) right = NULL;
            if ((EXPR_ARG1(x) && left == NULL) ||
                //(EXPR_ARG2(x) && right == NULL) ||
                (EXPR_ARG3(x) && right2 == NULL)) {
                goto err;
            }
            retV = list3(F_INDEX_RANGE, left, right, right2);
            set_index_range_type(retV);
            return retV;
        }

        case F95_CONSTANT_WITH:  {
            v1 = compile_expression(EXPR_ARG1(x)); /* constant */
            if(v1 == NULL)
                return NULL;
            assert(EXPV_TYPE(v1));
            v2 = compile_expression(EXPR_ARG2(x)); /* kind */
        calc_kind:
            if (v2 != NULL) {
                v2 = expv_reduce_kind(v2);
                if (v2 == NULL) {
                    error("bad expression in constant kind parameter");
                    break;
                }
                if(expr_is_constant_typeof(v2, TYPE_INT) == FALSE) {
                    error("bad expression in constant kind parameter");
                    break;
                }
                if (TYPE_BASIC_TYPE(EXPV_TYPE(v1)) == TYPE_DREAL) {
                    error("an invalid constant expression, which has "
                          "a 'd' exponent and an explicit kind.");
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

        case F95_KIND_SELECTOR_SPEC: {
            expv v = NULL;
            v = compile_expression(EXPR_ARG1(x));
            if (v != NULL) v = expv_reduce_kind(v)?:v;
            if(v != NULL && !expv_is_specification(v)){
                error_at_node(EXPR_ARG1(x),
                    "kind must be a specification expression.");
            }
            if (v != NULL) {
                EXPV_KWOPT_NAME(v) = (const char *)strdup("kind");
            }
            return v;
        }

        case F95_LEN_SELECTOR_SPEC: {
            expv v;
            if (EXPR_ARG1(x) == NULL)
                return expv_any_term(F_ASTERISK, NULL);
            if (EXPR_CODE(EXPR_ARG1(x)) == F08_LEN_SPEC_COLON)
                return expv_any_term(F08_LEN_SPEC_COLON, NULL);

            v = compile_expression(EXPR_ARG1(x));
            if((v = expv_reduce(v, FALSE)) == NULL) return NULL;
            /* if type is not fixed yet, do implicit declaration here */
            if((EXPV_CODE(v) == IDENT || EXPV_CODE(v) == F_VAR) &&
               EXPV_TYPE(v) == NULL) {
                id = find_ident(EXPV_NAME(v));
                if (ID_TYPE(id) == NULL) {
                   implicit_declaration(id);
                }
                EXPV_TYPE(v) = ID_TYPE(id);
            }
/* FEAST change start */
            /* if(!expv_is_specification(v)) */
            /*     error_at_node(EXPR_ARG1(x), */
            /*         "character string length must be integer."); */
            if(!expv_is_specification(v)){
              EXPV_TYPE(v) = NULL;
              sp_link_expr((expr)v, SP_ERR_CHAR_LEN, current_line);
            }
/* FEAST change  end  */
            return v;
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

        case F_MODULE_INTERNAL:
            /* we regard a module imported expression as compiled */
            return x;

        case F_STRING_CONST_SUBSTR: {
            expr strConstX = EXPR_ARG1(x);
            expr indexX = EXPR_ARG2(x);
            char *str = NULL;
            int len = 0;
            expr paramNameX = NULL;
            expr typeDeclX = NULL;
            expr accessX = NULL;

            /*
             * Generate parameter declaration.
             */
            if (EXPR_CODE(strConstX) != STRING_CONSTANT) {
                fatal("1st arg is not a string.");
                /* not reached. */
                return NULL;
            }
            str = EXPR_STR(strConstX);
            len = strlen(str);

            paramNameX = make_enode(IDENT,
                                    (void *)gen_temp_symbol("strconst")),
            typeDeclX = list3(F_TYPE_DECL,
                              list2(LIST,
                                    make_enode(F_TYPE_NODE, (void *)TYPE_CHAR),
                                    list2(LIST,
                                          list1(F95_LEN_SELECTOR_SPEC,
                                                make_int_enode(len)),
                                          NULL)),
                              list1(LIST,
                                    list5(LIST,
                                          paramNameX,
                                          NULL,
                                          NULL,
                                          make_enode(STRING_CONSTANT,
                                                     (void *)str),
                                          NULL)),
                              (is_in_module() == FALSE) ?
                              list1(LIST,
                                    list0(F95_PARAMETER_SPEC)) :
                              list2(LIST,
                                    list0(F95_PARAMETER_SPEC),
                                    list0(F95_PRIVATE_SPEC)));
            compile_type_decl(EXPR_ARG1(typeDeclX),
                              NULL,
                              EXPR_ARG2(typeDeclX),
                              EXPR_ARG3(typeDeclX));

            accessX = list2(F_ARRAY_REF,
                            paramNameX,
                            indexX);
            return compile_expression(accessX);
        }

        case F03_STRUCT_CONSTRUCT: {
            x1 = EXPR_ARG1(x);
            assert (EXPR_CODE(x1) == IDENT);
            id = find_ident(EXPR_SYM(x1));
            assert (id);
            assert (ID_CLASS(id) == CL_TAGNAME);
            return compile_struct_constructor(id, EXPR_ARG2(x), EXPR_ARG3(x));
        }

        case F08_LEN_SPEC_COLON:
        case LEN_SPEC_ASTERISC: {
            error_msg = "bad expression";
            goto err;
        }

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


/*
 * evaluate an identifier as an expression and translate it to Var
 * expression.
 */
expv
compile_ident_expression(expr x)
{
    ID id;
    SYMBOL sym;
    expv ret = NULL;
    TYPE_DESC tp;

    if (EXPR_CODE(x) != IDENT) {
        goto done;
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
    if(ID_IS_AMBIGUOUS(id)) {
        error("an ambiguous reference to symbol '%s'", ID_NAME(id));
        return NULL;
    }

    /* check if name is on the replace list? */
    if ((ret = is_statement_function_or_replace(id)) != NULL) {
        goto done;
    }

    if (ID_IS_DUMMY_ARG(id) && ID_TYPE(id) == NULL) {
        /*
         * Don't declare (means not determine the type) this variable
         * at this moment, since the id is a dummy arg and it is not
         * declared yet.
         */
        ret = expv_sym_term(F_VAR,NULL,ID_SYM(id));
        goto done;
    }


    if(ID_CLASS(id) == CL_PARAM){
        if(VAR_INIT_VALUE(id) != NULL) 
            return VAR_INIT_VALUE(id);
    }


    if ((id = declare_variable(id)) != NULL) {
        tp = ID_TYPE(id);
        if (ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_THISPROC) {
            tp = FUNCTION_TYPE_RETURN_TYPE(tp);
        } else if (ID_CLASS(id) == CL_ENTRY) {
            tp = FUNCTION_TYPE_RETURN_TYPE(tp);
        }

        TYPE_ATTR_FLAGS(tp) |= TYPE_ATTR_FLAGS(id);

        if (ID_ADDR(id)) {
            /*
             * Renaming trick:
             *  EXPV_NAME(ID_ADDR(id)) might be replaced to other name in
             *  compile_FORALL_statement().
             */
            ret = expv_sym_term(F_VAR, tp, EXPV_NAME(ID_ADDR(id)));
        } else {
            ret = expv_sym_term(F_VAR, tp, ID_SYM(id));
        }
        goto done;
    }

    done:
#ifdef not
    if (ret == NULL) {
/* FEAST change start */
        /* fatal("%s: invalid code", __func__); */
        ret = expv_sym_term(EXPR_CODE(x),NULL,EXPR_SYM(x));
        sp_link_expr((expr)ret, SP_ERR_FATAL, current_line);
/* FEAST change  end  */
    }
#endif

    return ret;
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
            if(ID_IS_AMBIGUOUS(id)) {
                error("an ambiguous reference to symbol '%s'", ID_NAME(id));
                goto err;
            }
            v = NULL;
            tp = ID_TYPE(id);

            if (ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_THISPROC) {
                tp = FUNCTION_TYPE_RETURN_TYPE(tp);
            }

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
        if(ID_IS_AMBIGUOUS(id)) {
            error("an ambiguous reference to symbol '%s'", ID_NAME(id));
            goto err;
        }

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
    if(!IS_CHAR(bottom_type(EXPV_TYPE(v)))) return FALSE;

    if(EXPR_CODE(v) == F_SUBSTR_REF ||
       EXPR_CODE(v) == F_VAR) 
        return TRUE;
    return FALSE;
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


/* compile into integer constant */
expv
compile_int_constant(expr x)
{
    expv v;

    if((v = compile_expression(x)) == NULL) return NULL;
    if((v = expv_reduce(v, FALSE)) == NULL) return NULL;
    if (expr_is_constant_typeof(v, TYPE_INT)) {
        omllint_t n = EXPV_INT_VALUE(v);
        if (n > INT_MAX) {
            EXPV_TYPE(v) = getLargeIntType();
        }
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
            (TYPE_IS_IMPLICIT(tp) || TYPE_IS_NOT_FIXED(tp))) {
            (void)force_to_logical_type(v);
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
expv_assignment(expv v1, expv v2)
{
    /* check assignment operator is user defined or not. */
    if(find_symbol_without_allocate(EXPR_CODE_SYMBOL(F95_ASSIGNOP)) != NULL)
        return expv_cons(F_LET_STATEMENT, NULL, v1, v2);

    if(EXPV_IS_RVALUE(v1) == TRUE) {
        error("bad left hand side expression in assignment.");
        return NULL;
    }

    TYPE_DESC tp1 = EXPV_TYPE(v1);
    TYPE_DESC tp2 = EXPV_TYPE(v2);

/* FEAST add start */
    if(!tp1 || !tp2){
      return NULL;
    }
/* FEAST add end */

    if (EXPV_CODE(v2) == F95_ARRAY_CONSTRUCTOR ||
        EXPV_CODE(v2) == F03_TYPED_ARRAY_CONSTRUCTOR) {
        if (!IS_ARRAY_TYPE(tp1)) {
            error("lhs expr is not an array.");
            return NULL;
        }
    }
    if (!TYPE_IS_NOT_FIXED(tp1) && !TYPE_IS_NOT_FIXED(tp2) &&
        EXPV_CODE(v2) != FUNCTION_CALL &&
        type_is_compatible_for_assignment(tp1, tp2) == FALSE) {
        error("incompatible type in assignment.");
        return NULL;
    }
    if (IS_PROCEDURE_TYPE(EXPV_TYPE(v1)) &&
        FUNCTION_TYPE_IS_TYPE_BOUND(EXPV_TYPE(v1))) {
            error("lhs expr is type bound procedure.");
            return NULL;
    }
    if (IS_PROCEDURE_TYPE(EXPV_TYPE(v2)) &&
        FUNCTION_TYPE_IS_TYPE_BOUND(EXPV_TYPE(v2))) {
            error("rhs expr is type bound procedure.");
            return NULL;
    }

    if (TYPE_IS_RESHAPED(tp2) == FALSE &&
        EXPR_CODE(v2) != F95_ARRAY_CONSTRUCTOR &&
        EXPR_CODE(v2) != F03_TYPED_ARRAY_CONSTRUCTOR &&
        ((TYPE_N_DIM(tp1) > 0 && TYPE_N_DIM(tp2) > 0) &&
         (are_dimension_and_shape_conformant(NULL, v1, v2,
                                             NULL) == FALSE))) {
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
 * \brief Compiles a dimension (and a shape if needed) from
 * subscript expressions.
 *
 *	@param args    Subscript expressions.
 *	@param subs    If not NULL, compiled subscriptions are put.
 *	@param aSpecs  If not NULL, array-specs coresponding to each
 *		       subsctiption are put. NULLs might be put for
 *		       non-array subscriptions, thus the # of the
 *		       elemsnts in the dims and aSpec is identical and
 *		       its are also identical to the # of elements in
 *		       the args.
 *
 *	@return TRUE if some errors occurred.
 */
static int
compile_array_ref_dimension(expr args, expv subs, expv aSpecs) {
    int n = 0;
    list lp;
    int err_flag = FALSE;
    expv v;
    expv d;
    expv aSpecV;
    TYPE_DESC tp;
    int nDims;

    assert(subs != NULL);

    FOR_ITEMS_IN_LIST(lp, args) {
        v = NULL;
        aSpecV = NULL;
        tp = NULL;
        nDims = 0;
        d = LIST_ITEM(lp);

        if ((v = compile_expression(d)) == NULL){
            err_flag = TRUE;
            continue;
        }

        EXPR_LINE(v) = EXPR_LINE(d);

        if (EXPR_CODE(v) == F_INDEX_RANGE) {
            /* lower, upper, step */
            if (checkSubscriptIsInt(EXPR_ARG1(v)) == FALSE ||
                checkSubscriptIsInt(EXPR_ARG2(v)) == FALSE ||
                checkSubscriptIsInt(EXPR_ARG3(v)) == FALSE) {
                error_at_node(d,
                              "Subscript #%d is not a valid array-spec, "
                              "not consists of integer expressions.", n + 1);
                err_flag = TRUE;
                continue;
            }
            /*
             * F_INDEX_RANGE is very identical as an array-spec.
             */
            aSpecV = v;
        } else if (checkSubscriptIsInt(v) == TRUE) {
            tp = EXPV_TYPE(v);
            if (tp == NULL) {
                warning_at_node(d,
                                "Subscript #%d could consist of (still) "
                                "undefined variable(s).", n + 1);
            } else {
                if ((nDims = TYPE_N_DIM(tp)) > 0) {
                    /*
                     * Could have to be treat as it is having
                     * array-spec.
                     */
                    if (nDims != 1) {
                        error_at_node(d,
                                      "#%d subscription is not a "
                                      "one-dimensional expression.", n + 1);
                        err_flag = TRUE;
                        continue;
                    }
                    aSpecV = list0(LIST);
                    generate_shape_expr(tp, aSpecV);
                    if (expr_list_length(aSpecV) != 1) {
                        fatal("Invalid # of array-specs for one-dim "
                              "array shape.");
                        /* not reached. */
                        err_flag = TRUE;
                        continue;
                    }
                    aSpecV = EXPR_ARG1(aSpecV);
                }
                /*
                 * Otherwise this is just a integer expression
                 * suitable for a subscription and not an array-spec.
                 */
            }
        } else {
            error_at_node(d,
                          "#%d subscrition is not an integer expression.",
                          n + 1);
            continue;
        }

        list_put_last(subs, v);
        if (aSpecs != NULL) {
            list_put_last(aSpecs, aSpecV);
        }
    }

    return err_flag;
}


expv
compile_array_ref(ID id, expv vary, expr args, int isLeft) {
    int nDims;
    int n;
    int i;
    expv aSpecs;
    expv subs;
    expv argASpec;
    expv idASpec;
    expv aSpec;
    expv shape = NULL;
    int aSpecSz;
    int idASpecSz;

    TYPE_DESC tp;
    TYPE_DESC tq;
    int nIdxRanges = 0;
    expv idShape = NULL;

    assert((id && vary == NULL) || (id == NULL && vary));

    tp = (id ? ID_TYPE(id) : EXPV_TYPE(vary));

    if (id != NULL && (
        (tp != NULL && IS_PROCEDURE_TYPE(tp)
         && !IS_ARRAY_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp))) ||
        PROC_CLASS(id) == P_EXTERNAL ||
        PROC_CLASS(id) == P_DEFINEDPROC ||
        (ID_IS_DUMMY_ARG(id) &&
         !(IS_ARRAY_TYPE(tp)) &&
         isLeft == FALSE))) {
        return compile_highorder_function_call(id, args, FALSE);
    }

    if (id != NULL && ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_THISPROC) {
        tp = FUNCTION_TYPE_RETURN_TYPE(tp);
    }

    nDims = TYPE_N_DIM(tp);

    if (!IS_ARRAY_TYPE(tp)){ //fatal("%s: not ARRAY_TYPE", __func__);
      error_at_id(id, "identifier '%s' is not of array type", ID_NAME(id));
      return NULL;
    }
    if (!TYPE_DIM_FIXED(tp)) fix_array_dimensions(tp);

    /*
     * Firstly, check the # of subsctipts.
     */
    n = expr_list_length(args);
    if (n != nDims) {
        if (id != NULL) {
            error_at_node(args,
                          "wrong # of subscripts on '%s', "
                          "got %d, should be %d.",
                          ID_NAME(id), n, TYPE_N_DIM(tp));
        } else {
            error_at_node(args, "wrong number of subscript.");
        }
        return NULL;
    }

    aSpecs = list0(LIST);
    subs = list0(LIST);

    /*
     * Get subscripts and array-specs.
     */
    if (compile_array_ref_dimension(args, subs, aSpecs) == TRUE) {
        return NULL;
    }

    /*
     * Check if the subscript expressions are compiled successfully.
     */
    n = expr_list_length(subs);
    if (n != nDims) {
        if (id != NULL) {
            error_at_node(args, "wrong number of subscript on '%s'",
                          ID_NAME(id));
        } else {
            error_at_node(args, "wrong number of subscript");
        }
        return NULL;
    }

    idShape = list0(LIST);
    generate_shape_expr(tp, idShape);

    shape = list0(LIST);

    /*
     * Then fix the shape of this expression. To do this, we use two
     * array-specs source; 1) the aSpecs, generated from the
     * subscripts. 2) the idShape, generated from the variable
     * definition.
     */
    for (i = 0; i < nDims; i++) {
        argASpec = expr_list_get_n(aSpecs, i);
        if (argASpec != NULL) {
            idASpec = expr_list_get_n(idShape, i);

            expv lower = expr_list_get_n(argASpec, 0);
            //if (lower == NULL) expr_list_set_n(argASpec, 0, expr_list_get_n(idASpec, 0), FALSE);
	    if (lower == NULL){
	      expv lower0 = expr_list_get_n(idASpec, 0);
	      if (!lower0 || EXPR_CODE_IS_CONSTANT(lower0) || TYPE_IS_PARAMETER(EXPV_TYPE(lower0))){
		expr_list_set_n(argASpec, 0, lower0, FALSE);
	      }
	    }

	    expv upper = expr_list_get_n(argASpec, 1);
            //if (upper == NULL) expr_list_set_n(argASpec, 1, expr_list_get_n(idASpec, 1), FALSE);
	    if (upper == NULL){
	      expv upper0 = expr_list_get_n(idASpec, 1);
	      if (!upper0 || EXPR_CODE_IS_CONSTANT(upper0) || TYPE_IS_PARAMETER(EXPV_TYPE(upper0))){
		expr_list_set_n(argASpec, 1, upper0, FALSE);
	      }
	    }

            /*
             * Now we have two array-spec. Determine which one to be used.
             */
            aSpec = NULL;
            aSpecSz = array_spec_size(argASpec, idASpec, &aSpec);
            if (aSpec == NULL) {
                fatal("an array-spec to use can't be determined.");
                /* not reached. */
                continue;
            }
            /*
             * size check.
             */
            idASpecSz = array_spec_size(idASpec, NULL, NULL);
            if ((aSpecSz > 0 && idASpecSz > 0) &&
                (aSpecSz > idASpecSz)) {
                if (id != NULL) {
                    error_at_node(args, 
                                  "The size of sunscript #%d is %d, exceeds "
                                  "the size of '%s', %d.",
                                  i + 1, aSpecSz, ID_NAME(id), idASpecSz);
                } else {
                    error_at_node(args, 
                                  "The size of sunscript #%d is %d, exceeds "
                                  "%d.",
                                  i + 1, aSpecSz, idASpecSz);
                }
                continue;
            }

            list_put_last(shape, aSpec);
            nIdxRanges++;
        }
    }

    if (nIdxRanges == 0) {
        shape = NULL;
    }

    if (nIdxRanges > 0) {
        tq = compile_dimensions(bottom_type(tp), shape);
        if (tq == NULL) {
            return NULL;
        }
        fix_array_dimensions(tq);
    } else {
        /*
         * Otherwise the type should be basic type of the array.
         */
        tq = bottom_type(tp);
    }

    /*
     * copy coShape from original type
     */
    if (TYPE_IS_COINDEXED(tp)) {
        TYPE_CODIMENSION(tq) = TYPE_CODIMENSION(tp);
    }

    /*
     * copy type attributes from original type
     */
    if (id != NULL) {
        if (TYPE_IS_POINTER(id)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_POINTER(id);
        }
        if (TYPE_IS_TARGET(id)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_TARGET(id);
        }
        if (TYPE_IS_VOLATILE(id)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_VOLATILE(id);
        }
        if (TYPE_IS_ASYNCHRONOUS(id)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_ASYNCHRONOUS(id);
        }
    }
    while (tp != NULL) {
        if (TYPE_IS_POINTER(tp)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_POINTER(tp);
        }
        if (TYPE_IS_TARGET(tp)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_TARGET(tp);
        }
        if (TYPE_IS_VOLATILE(tp)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_VOLATILE(tp);
        }
        if (TYPE_IS_ASYNCHRONOUS(tp)) {
            TYPE_ATTR_FLAGS(tq) |= TYPE_IS_ASYNCHRONOUS(tp);
        }
        tp = TYPE_REF(tp);
    }

    if (id != NULL) {
        vary = expv_sym_term(F_VAR, ID_TYPE(id), ID_SYM(id));
        ID_ADDR(id) = vary;

        tp = ID_TYPE(id);
        if (id != NULL && ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_THISPROC) {
            tp = FUNCTION_TYPE_RETURN_TYPE(tp);
        }

        if (TYPE_N_DIM(tp) < n) {
            error_at_node(args, "too large dimension, %d.", n);
            return NULL;
        }
    }

    return expv_reduce(expv_cons(ARRAY_REF,
                                 tq, vary, subs), FALSE);
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
    error_at_node(coarrayRef, "The variable is not declared as a coarray.");
    return NULL;
  }

  //
  // (2) process the cosubscripts.
  //

  expv cosubs = list0(LIST);
  //expv codims = list0(LIST);

  /* get codims and cosubs*/
  //if (compile_array_ref_dimension(image_selector, codims, cosubs)){
  if (compile_array_ref_dimension(image_selector, cosubs, NULL)){
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
/* 	else if (EXPV_CODE(upper) == F_ASTERISK){ */
/* 	  upper = NULL; */
/* 	} */
	else {
	  error("Last upper-cobound must be \"*\".");
	}
      }

      if (!lower && upper && EXPV_CODE(upper) != F_ASTERISK)
	EXPR_ARG1(x) = expv_constant_1;
    }

    n++;
  }

  if (tp->codims->corank != n){
    error_at_node(image_selector, "wrong number of cosubscript.");
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

expv
compile_highorder_function_call(ID id, expr args, int isCall)
{
    if (!(ID_IS_DUMMY_ARG(id)) &&
        !(ID_TYPE(id) && IS_PROCEDURE_TYPE(ID_TYPE(id)))) {
        fatal("%s: '%s' is not a dummy arg.",
              __func__, SYM_NAME(ID_SYM(id)));
        /* not reached. */
        return NULL;
    } else {
        /*
         * A high order sub program invocation.
         */
        expv ret;
        ret = compile_function_call(id, args);

        if (isCall == TRUE) {
            EXPV_TYPE(ret) = type_SUBR;
            VAR_IS_USED_AS_FUNCTION(id) = TRUE;
        }

        return ret;
    }
}


static TYPE_DESC
choose_module_procedure_by_args(EXT_ID mod_procedures, expv args)
{
    EXT_ID ep;
    FOREACH_EXT_ID(ep, mod_procedures) {
        if (function_type_is_appliable(EXT_PROC_TYPE(ep), args)) {
            return EXT_PROC_TYPE(ep);
        }
    }
    return NULL;
}

expv
compile_function_call(ID f_id, expr args) {
    return compile_function_call_check_intrinsic_arg_type(f_id, args, FALSE);

}

expv
compile_function_call_check_intrinsic_arg_type(ID f_id, expr args, int ignoreTypeMismatch) {
    expv a, v = NULL;
    EXT_ID ep = NULL;
    TYPE_DESC tp = NULL;
    ID tagname = NULL;

    if (declare_function(f_id) == NULL) return NULL;

    if (ID_CLASS(f_id) == CL_VAR && IS_PROCEDURE_TYPE(ID_TYPE(f_id))) {
        tp = get_bottom_ref_type(ID_TYPE(f_id));
        a = compile_args(args);
        v = list3(FUNCTION_CALL,
                  expv_sym_term(F_VAR, ID_TYPE(f_id), ID_SYM(f_id)),
                  a,
                  expv_any_term(F_EXTFUNC, f_id));

        EXPV_TYPE(v) = !tp ? type_GNUMERIC_ALL :
                IS_GENERIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp)) ?
                type_GNUMERIC_ALL :
                FUNCTION_TYPE_RETURN_TYPE(tp) ;
        goto line_info;
    }

    if (ID_CLASS(f_id) == CL_MULTI) {
        tagname = multi_find_class(f_id, CL_TAGNAME);
        f_id = multi_find_class(f_id, CL_PROC);
    }

    switch (PROC_CLASS(f_id)) {
        case P_UNDEFINEDPROC:
            /* f_id is not defined yet. */

            if (ID_TYPE(f_id) != NULL) {
                if (!IS_PROCEDURE_TYPE(ID_TYPE(f_id))) {
                    if (TYPE_IS_SAVE(ID_TYPE(f_id))) {
                        TYPE_UNSET_SAVE(ID_TYPE(f_id));
                    }
                    ID_TYPE(f_id) = function_type(ID_TYPE(f_id));
                    EXPV_TYPE(ID_ADDR(f_id)) = FUNCTION_TYPE_RETURN_TYPE(ID_TYPE(f_id));
                }
                tp = ID_TYPE(f_id);
            } else {
                /* f_id is function, but it's return type is unknown */
                tp = function_type(new_type_desc());
                TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp)) = TYPE_GNUMERIC;
            }

            TYPE_SET_USED_EXPLICIT(tp);
/* FEAST add start */
            if (TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp)) == TYPE_UNKNOWN){
                /* ID_TYPE(f_id) = NULL; */
                sp_link_id(f_id, SP_ERR_UNDEF_TYPE_FUNC, current_line);
            }
/* FEAST add  end  */

            a = compile_args(args);

            v = list3(FUNCTION_CALL, ID_ADDR(f_id), a,
                      expv_any_term(F_EXTFUNC, f_id));

            if (IS_GENERIC_TYPE(tp)) {
                EXPV_TYPE(v) = type_GNUMERIC_ALL;
            } else {
                EXPV_TYPE(v) = FUNCTION_TYPE_RETURN_TYPE(tp);
                /*
                 * EXPV_TYPE(v) should be replaced in finalization phase as:
                 * EXT_PROC_TYPE(PROC_EXT_ID(EXPV_ANY(ID, EXPR_ARG3(v))))
                 */
                EXPV_NEED_TYPE_FIXUP(v) = TRUE;
            }

            break;

        case P_THISPROC:
            if (!TYPE_IS_RECURSIVE(ID_TYPE(f_id))) {
                error("recursive call in not a recursive function");
            } else if (IS_FUNCTION_TYPE(ID_TYPE(f_id)) &&
                       FUNCTION_TYPE_RESULT(ID_TYPE(f_id)) == NULL) {
                error("Use a RESULT variable for recursion");
            }
            /* FALL THROUGH */
        case P_DEFINEDPROC:
        case P_EXTERNAL: {
            EXT_ID modProcs = NULL;
            TYPE_DESC modProcType = NULL;

            if (ID_TYPE(f_id) == NULL) {
                error("attempt to use untyped function,'%s'",
                      ID_NAME(f_id));
                goto err;
            }
            tp = ID_TYPE(f_id);

            if (!IS_PROCEDURE_TYPE(tp)) {
                tp = function_type(tp);
                ID_TYPE(f_id) = tp;
                EXPV_TYPE(ID_ADDR(f_id)) = ID_TYPE(f_id);
            }

            if (TYPE_IS_ABSTRACT(ID_TYPE(f_id))) {
                error("'%s' is abstract", ID_NAME(f_id));
                goto err;
            }

            TYPE_SET_USED_EXPLICIT(tp);
            if (FUNCTION_TYPE_RETURN_TYPE(tp) != NULL &&
                TYPE_BASIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp)) == TYPE_UNKNOWN) {
                TYPE_SET_NOT_FIXED(FUNCTION_TYPE_RETURN_TYPE(tp));
            }
            a = compile_args(args);

            if (ID_DEFINED_BY(f_id) != NULL) {
                ep = PROC_EXT_ID(ID_DEFINED_BY(f_id));
            } else {
                ep = PROC_EXT_ID(f_id);
            }
            if (ep != NULL && EXT_PROC_CLASS(ep) == EP_INTERFACE &&
                (modProcs = EXT_PROC_INTR_DEF_EXT_IDS(ep)) != NULL) {
                modProcType = choose_module_procedure_by_args(modProcs, a);
                if (modProcType != NULL) {
                    tp = modProcType;

                } else if (tagname != NULL) {
                    return compile_struct_constructor(tagname, NULL, args);

                } else {

                    warning_at_id(f_id, "can't determine a function to "
                                    "be actually called for a generic "
                                    "interface function call of '%s', "
                                    "this is a current limitation.",
                                    SYM_NAME(EXT_SYM(ep)));
                }
            } else if (ep == NULL) {
                ep = new_external_id_for_external_decl(ID_SYM(f_id),
                                                       ID_TYPE(f_id));
                PROC_EXT_ID(f_id) = ep;
            }

            v = list3(FUNCTION_CALL, ID_ADDR(f_id), a,
                      expv_any_term(F_EXTFUNC, f_id));

            EXPV_TYPE(v) = IS_GENERIC_TYPE(FUNCTION_TYPE_RETURN_TYPE(tp)) ?
                    type_GNUMERIC_ALL :
                    FUNCTION_TYPE_RETURN_TYPE(tp);

            break;
        }

        case P_INTRINSIC:
            v = compile_intrinsic_call0(f_id, compile_data_args(args), ignoreTypeMismatch);
            break;

        case P_STFUNCT:
            v = statement_function_call(f_id, compile_args(args));
            break;

        default:
            fatal("%s: unknown proc_class %d", __func__,
                  PROC_CLASS(f_id));
    }

line_info:
    if (v != NULL) {
        if (args != NULL) {
            EXPR_LINE(v) = EXPR_LINE(args);
        } else {
            EXPR_LINE(v) = current_line;
        }
    }
    return v;

err:
    return NULL;
}

static int
type_param_values_required0(TYPE_DESC struct_tp, ID * head, ID * tail)
{
    ID ip;

    if (TYPE_PARENT(struct_tp) &&
        type_param_values_required0(TYPE_PARENT_TYPE(struct_tp), head, tail)) {
        return TRUE;
    }

    FOREACH_ID(ip, TYPE_TYPE_PARAMS(struct_tp)) {
        if (!VAR_INIT_VALUE(ip)) {
            return TRUE;
        }
    }

    return FALSE;
}


int
type_param_values_required(TYPE_DESC tp)
{
    ID head = NULL, tail = NULL;
    return type_param_values_required0(tp, &head, &tail);
}



static void
get_type_params0(TYPE_DESC struct_tp, ID * head, ID * tail)
{
    ID id, ip;

    if (TYPE_PARENT(struct_tp))
        get_type_params0(TYPE_PARENT_TYPE(struct_tp), head, tail);

    FOREACH_ID(ip, TYPE_TYPE_PARAMS(struct_tp)) {
        id = XMALLOC(ID,sizeof(*id));
        *id = *ip;
        ID_LINK_ADD(id, *head, *tail);
    }
}

ID
get_type_params(TYPE_DESC struct_tp)
{
    ID head = NULL, tail = NULL;

    get_type_params0(struct_tp, &head, &tail);

    return head;
}


// Expects to use for the dummy derived type
// (genereted by declare_struct_type_wo_component)
int
compile_type_param_values_dummy(TYPE_DESC struct_tp, expr type_param_args) {
    list lp;
    expv type_param_values = list0(LIST);
    FOR_ITEMS_IN_LIST(lp, type_param_args) {
        expv v = compile_expression(LIST_ITEM(lp));
        if (v == NULL) {
            return FALSE;
        }
        list_put_last(type_param_values, v);
    }
    TYPE_TYPE_PARAM_VALUES(struct_tp) = type_param_values;
    return TRUE;
}


/**
 * Compile type parameter values for the parameterized derived-type
 *
 * Check `type_param_args` as the type parameter values for the parameter derived-type,
 * and compile them into `type_param_values`.
 * `used` will store type parameter identifiers and its values even if they don't exist in `type_param_values`
 * (and will be passed to type_apply_type_parameter())
 */
int
compile_type_param_values(TYPE_DESC struct_tp, expr type_param_args, expv type_param_values, ID * used)
{
    int has_keyword = FALSE;
    list lp;
    ID ip;
    ID used_last = NULL;
    ID match = NULL; // ID specified by a type parameter argument
    ID cur;
    ID rest_type_params;
    SYMBOL sym;
    enum expr_code e_code;
    expv v;
    *used = NULL;

    /* collect type parameters recursively */
    rest_type_params = get_type_params(struct_tp);
    cur = rest_type_params;

    FOR_ITEMS_IN_LIST(lp, type_param_args) {
        expv arg = LIST_ITEM(lp);

        if (EXPR_CODE(arg) == F_SET_EXPR) {
            /* A type parameter value has a KEYWORD */
            sym = EXPR_SYM(EXPR_ARG1(arg));

            if (has_keyword == FALSE) {
                rest_type_params = cur;
                has_keyword = TRUE;
            }

            /* check keyword is not duplicate */
            if (find_ident_head(sym, *used) != NULL) {
                error("type parameter '%s' is already specified", SYM_NAME(sym));
                return FALSE;
            }

            if ((match = find_ident_head(sym, rest_type_params)) == NULL) {
                error("'%s' is not type value keyword", SYM_NAME(sym));
                return FALSE;
            }

            e_code = EXPR_CODE(EXPR_ARG2(arg));
        } else {
            sym = NULL;

            if (has_keyword == TRUE) {
                /* KEYWORD connot be omitted after an argument which has a KEYWORD */
                error("KEYWORD connot be ommited after the type parameter value with a keyword");
                return FALSE;
            }

            if (cur == NULL) {
                /* There no more type parameters */
                error("unexpected type value");
                return FALSE;
            }

            match = cur;
            e_code = EXPR_CODE(arg);
        }

        switch (e_code) {
            case LEN_SPEC_ASTERISC:
            case F08_LEN_SPEC_COLON:
                if (!TYPE_IS_LEN(ID_TYPE(match))) {
                    error("length spec for no-length parameter");
                    return FALSE;
                }
                v = make_enode(e_code, NULL);
                if (sym) {
                    EXPV_KWOPT_NAME(v) = (const char *)strdup(SYM_NAME(sym));
                }
                break;
            case F95_TRIPLET_EXPR:
                // NOTE: sorry but the current parser cannot differs the length
                // spec and triplet with no values
                if (!TYPE_IS_LEN(ID_TYPE(match))) {
                    error("length spec for no-length parameter");
                    return FALSE;
                }
                if (sym) {
                    arg = EXPR_ARG2(arg);
                }
                if (EXPR_ARG1(arg) != NULL || EXPR_ARG2(arg) != NULL) {
                    error("Invalid length specifier");
                    return FALSE;
                }
                v = make_enode(F08_LEN_SPEC_COLON, NULL);
                if (sym) {
                    EXPV_KWOPT_NAME(v) = (const char *)strdup(SYM_NAME(sym));
                }
                break;
            default:
                /* A type parameter valeus should be a constatnt integer */
                if (!expr_is_constant_typeof(EXPR_CODE(arg) == F_SET_EXPR ?
                                             EXPR_ARG2(arg) : arg, TYPE_INT)) {
                    error("type parameter value should be "
                          "a constant integer expression");
                    return FALSE;
                }
                v = compile_expression(arg);
                break;
        }

        if (!has_keyword) {
            cur = ID_NEXT(cur);
        }

        v = expv_reduce(v, TRUE);
        id_link_remove(&rest_type_params, match);
        VAR_INIT_VALUE(match) = v;
        ID_LINK_ADD(match, *used, used_last);
        list_put_last(type_param_values, v);
    }
    /* Check if not-initialized type parameters don't exist */
    FOREACH_ID(ip, rest_type_params) {
        if (!VAR_INIT_VALUE(ip)) {
            error("type parameter %s is not initialized", ID_NAME(ip));
            return FALSE;
        }
    }
    if (used_last != NULL) {
        // The rest type parameters are used with its initial values
        ID_NEXT(used_last) = rest_type_params;
    } else {
        // All type parameters are used with its initial values
        *used = rest_type_params;
    }
    return TRUE;
}


static void
get_struct_members0(TYPE_DESC struct_tp, ID * head, ID * tail)
{
    ID id, ip;

    if (TYPE_PARENT(struct_tp))
        get_struct_members0(TYPE_PARENT_TYPE(struct_tp), head, tail);

    FOREACH_ID(ip, TYPE_MEMBER_LIST(struct_tp)) {
        id = XMALLOC(ID,sizeof(*id));
        *id = *ip;
        ID_LINK_ADD(id, *head, *tail);
    }
}

static ID
get_struct_members(TYPE_DESC struct_tp)
{
    ID head = NULL, tail = NULL;

    get_struct_members0(struct_tp, &head, &tail);

    return head;
}


static expv
compile_struct_constructor_with_components(const ID struct_id,
                                           const TYPE_DESC stp,
                                           const expr args)
{
    int has_keyword = FALSE;
    list lp;
    ID ip, cur, members, used = NULL, used_last = NULL;
    ID match = NULL;
    SYMBOL sym;
    expv v;
    expv result, components;
    TYPE_DESC tp;
    components = list0(LIST);
    result = list2(F95_STRUCT_CONSTRUCTOR, NULL, components);

    // Check PRIVATE components
    // (PRIVATE works if the derived type is use-associated)
    int is_use_associated = ID_USEASSOC_INFO(struct_id) != NULL;

    members = get_struct_members(stp?:ID_TYPE(struct_id));
    cur = members;

    FOR_ITEMS_IN_LIST(lp, args) {
        expr arg = LIST_ITEM(lp);

        if (EXPV_CODE(arg) == F_SET_EXPR) {
            sym = EXPR_SYM(EXPR_ARG1(arg));

            if (has_keyword == FALSE) {
                members = cur;
                has_keyword = TRUE;
            }

            // check keyword is duplicate
            if (find_ident_head(sym, used) != NULL) {
                error("member'%s' is already specified", SYM_NAME(sym));
                return NULL;
            }

            if ((match = find_ident_head(sym, members)) == NULL) {
                error("'%s' is not member", SYM_NAME(sym));
                return NULL;
            }
        } else {
            sym = NULL;

            if (has_keyword == TRUE) {
                // KEYWORD connot be ommit after KEYWORD-ed arg
                error("KEYWORD connot be ommited after the component with a keyword");
                return NULL;
            }

            if (cur == NULL) {
                error("unexpected member");
                return NULL;
            }

            match = cur;
        }

        if (is_use_associated && ID_TYPE(match) != NULL &&
            ((TYPE_IS_INTERNAL_PRIVATE(match) ||
              TYPE_IS_INTERNAL_PRIVATE(ID_TYPE(match))) &&
             !(TYPE_IS_PUBLIC(match) ||
               TYPE_IS_PUBLIC(ID_TYPE(match))))) {
            error("accessing a private component");
            return NULL;
        }

        v = compile_expression(arg);
        assert(EXPV_TYPE(v) != NULL);
        if (!type_is_compatible_for_assignment(ID_TYPE(match),
                                               EXPV_TYPE(v))) {
            error("type is not applicable in struct constructor");
            return NULL;
        }

        if (!has_keyword) {
            cur = ID_NEXT(cur);
        }
        id_link_remove(&members, match);

        ID_LINK_ADD(match, used, used_last);
        list_put_last(components, v);
    }

    /*
     * check all members are initialized
     */
    FOREACH_ID(ip, members) {
        if (ID_CLASS(ip) != CL_TYPE_BOUND_PROC && (
                !VAR_INIT_VALUE(ip) &&
                !TYPE_IS_ALLOCATABLE(ID_TYPE(ip)))) {
            error("member %s is not initialized", ID_NAME(ip));
        }
    }

    if (TYPE_REF(stp)) {
        tp = stp;
    } else {
        tp = wrap_type(stp);
    }

    EXPV_TYPE(result) = tp;

    return result;
}


expv
compile_struct_constructor(ID struct_id, expr type_param_args, expr args)
{
    expv result, component;
    TYPE_DESC base_stp;
    TYPE_DESC tp;

    assert(ID_TYPE(struct_id) != NULL);

    component = list0(LIST);
    result = list2(F95_STRUCT_CONSTRUCTOR, NULL, component);

    base_stp = find_struct_decl(ID_SYM(struct_id));
    assert(EXPV_TYPE(result) != NULL);
    if (TYPE_IS_ABSTRACT(base_stp)) {
        error("abstract type in an derived-type constructor");
    }

    if (type_param_args) {
        tp = type_apply_type_parameter(base_stp, type_param_args);
        EXPR_ARG1(result) = TYPE_TYPE_PARAM_VALUES(tp);
    } else if (type_param_values_required(base_stp)) {
        error("struct type '%s' requires type parameter values",
              SYM_NAME(ID_SYM(struct_id)));
        return NULL;
    } else {
        tp = base_stp;
    }

    if (args) {
        EXPV_LINE(result) = EXPR_LINE(args);
        return compile_struct_constructor_with_components(struct_id, tp, args);
    }

    if (tp == base_stp) {
        tp = wrap_type(tp);
    }

    EXPV_TYPE(result) = tp;
    return result;
}


expv
compile_args(expr args)
{
    list lp;
    expr a;
    expv v, arglist;
    ID id;
    int is_declared = FALSE;

    arglist = list0(LIST);
    if (args == NULL) return arglist;

    FOR_ITEMS_IN_LIST(lp, args) {
        a = LIST_ITEM(lp);
        /* check function address */
        if (EXPR_CODE(a) == IDENT) {
            id = find_ident(EXPR_SYM(a));
            if (id == NULL)
                id = declare_ident(EXPR_SYM(a), CL_UNKNOWN);
            if (ID_IS_AMBIGUOUS(id)) {
                error("an ambiguous reference to symbol '%s'", ID_NAME(id));
                continue;
            }
            if (type_is_nopolymorphic_abstract(ID_TYPE(id))) {
                error("an abstract interface '%s' in the actual argument", ID_NAME(id));
            }

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
                is_declared = ID_IS_DECLARED(id);
                /* check variable name */
                declare_variable(id);
                ID_IS_DECLARED(id) = is_declared;
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

    if(id == NULL) {
        id = declare_ident(sym, CL_UNKNOWN);
        declare_function(id);
    }

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
    expv do_var, do_init, do_limit, do_incr, retv;
    expr var, init, limit, incr;
    SYMBOL do_var_sym;
    CTL cp;

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
    FOR_CTLS(cp) {
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

    retv = expv_cons(F_IMPLIED_DO, retTyp, x1, x2);
    EXPR_LINE(retv) = EXPR_LINE(x);
    return retv;
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
    int nElems = 0;
    list lp;
    expv v, res, l;
    TYPE_DESC tp = NULL;
    TYPE_DESC base_type = NULL;
    BASIC_DATA_TYPE elem_type = TYPE_UNKNOWN;

    l = list0(LIST);
    if ((base_type = compile_type(EXPR_ARG2(x), /*allow_predecl=*/FALSE)) != NULL) {
        if (type_is_nopolymorphic_abstract(base_type)) {
            error("abstract type in an array constructor");
        }
        elem_type = get_basic_type(base_type);
        res = list1(F03_TYPED_ARRAY_CONSTRUCTOR, l);
    } else {
        res = list1(F95_ARRAY_CONSTRUCTOR, l);
    }


    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(x)) {
        nElems++;
        v = compile_expression(LIST_ITEM(lp));
        list_put_last(l, v);
        tp = EXPV_TYPE(v);
        if (elem_type == TYPE_UNKNOWN) {
            elem_type = get_basic_type(tp);
            continue;
        }
        if (get_basic_type(tp) != elem_type) {
            error("Array constructor elements have different data types.");
            return NULL;
        }

        if (base_type) {
            if (!type_is_soft_compatible(base_type, tp)) {
                error("Unexpected element type");
                return NULL;
            }
        }
    }

    assert(elem_type != TYPE_UNKNOWN);
    if (base_type) {
        tp = base_type;
    } else if (elem_type == TYPE_CHAR) {
        tp = type_char(-1);
    } else if (elem_type != TYPE_STRUCT) {
        tp = type_basic(elem_type);
        /*
         * If elem_type == TYPE_STRUCT, we just use the tp.
         */
    }

    EXPV_TYPE(res) =
        compile_dimensions(tp,
                           list1(LIST,
                                 (list2(LIST,
                                        expv_constant_1,
                                        expv_int_term(INT_CONSTANT,
                                                      type_INT, nElems)))));

    return res;
}


expv
compile_type_bound_procedure_call(expv memberRef, expr args) {
    expv v;
    expv a;

    TYPE_DESC ftp;
    TYPE_DESC stp;
    TYPE_DESC ret_type = type_GNUMERIC_ALL;

    a = compile_args(args);

    ftp = EXPV_TYPE(memberRef);
    stp = EXPV_TYPE(EXPR_ARG1(memberRef));
    if (TYPE_BOUND_GENERIC_TYPE_GENERICS(ftp)) {
        // for type-bound GENERIC
        ID bind;
        ID bindto;
        FOREACH_ID(bind, TYPE_BOUND_GENERIC_TYPE_GENERICS(ftp)) {
            bindto = find_struct_member_allow_private(stp, ID_SYM(bind), TRUE);
            if (TYPE_REF(ID_TYPE(bindto)) &&
                function_type_is_appliable(TYPE_REF(ID_TYPE(bindto)), a)) {
                ftp = TYPE_REF(ID_TYPE(bindto));
                /* EXPV_TYPE(memberRef) = ftp; */
            }
        }

        if (ftp) {
            ret_type = FUNCTION_TYPE_RETURN_TYPE(ftp);
        }
        else {
            if (debug_flag)
                fprintf(debug_fp, "There is no appliable type-bound procedure");
        }
        /* type-bound generic procedure type does not exist in XcodeML */
        EXPV_TYPE(memberRef) = NULL;
    } else {
        // for type-bound PROCEDURE
        if (ftp != NULL) {
#if 0 // to be solved
            if (function_type_is_appliable(ftp, a)) {
                error("argument type mismatch");
            }
#endif
            if (TYPE_REF(ftp)) {
                ret_type = FUNCTION_TYPE_RETURN_TYPE(TYPE_REF(ftp));
            } else {
                /*
                 * type-bound procedure is not bound yet,
                 * so set a dummy type.
                 */
                ret_type = FUNCTION_TYPE_RETURN_TYPE(ftp);
            }
        }
    }

    v = list2(FUNCTION_CALL, memberRef, a);
    EXPV_TYPE(v) = ret_type;
    return v;
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

#if 0 // to be solved
    if(compile_array_ref_dimension(indices,shape,NULL)) {
        return NULL;
    }
    // Checks two or more nonzero rank array references are not appeared.
    // i.e) a(1:5)%n(1:5) not accepted
    if(((EXPV_CODE(org_expr) == F_ARRAY_REF) &&
        (IS_ARRAY_TYPE(tq) && IS_CHAR(bottom_type(tq)) == FALSE)) &&
       (EXPR_LIST(shape) != NULL)) {
        error("Two or more part references with nonzero rank must not be specified");
        return NULL;
    }
#endif

    if (IS_PROCEDURE_TYPE(EXPV_TYPE(v))) {
        /*
         * type bound procedure coall
         */
        return compile_type_bound_procedure_call(v, indices);
    }

    if(EXPR_LIST(shape) == NULL) {
        generate_shape_expr(tq, shape);
    }

    tp = EXPV_TYPE(v);

    /*
     * copy coShape from original type
     */
    if (TYPE_IS_COINDEXED(tq)) {
        TYPE_CODIMENSION(tp) = TYPE_CODIMENSION(tq);
    }

    if (IS_ARRAY_TYPE(tp)) {
        TYPE_DESC new_tp;
        expv new_v = compile_array_ref(NULL, v, indices, TRUE);
        new_tp = EXPV_TYPE(new_v);

        if ((TYPE_IS_POINTER(tp) || TYPE_IS_TARGET(tp) ||
             TYPE_IS_VOLATILE(tp) || TYPE_IS_ASYNCHRONOUS(tp)) &&
           !(TYPE_IS_POINTER(new_tp) || TYPE_IS_TARGET(new_tp)
             || TYPE_IS_VOLATILE(new_tp) || TYPE_IS_ASYNCHRONOUS(new_tp))) {
            TYPE_DESC btp = bottom_type(new_tp);
            if(!EXPR_HAS_ARG1(shape))
                generate_shape_expr(new_tp, shape);
            btp = wrap_type(btp);
            TYPE_ATTR_FLAGS(btp) |=
                    TYPE_IS_POINTER(tp) | TYPE_IS_TARGET(tp) |
                    TYPE_IS_VOLATILE(tp) | TYPE_IS_ASYNCHRONOUS(tp);
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
