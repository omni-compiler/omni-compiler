/**
 * \file F-opt-expv.c
 */

#include "F-front.h"
#include <math.h>
#include <float.h>

#define EXPV_IS_INT_ZERO(v) \
  (EXPV_CODE(v) == INT_CONSTANT && EXPV_INT_VALUE(v) == 0)
#define EXPV_IS_INT_ONE(v)  \
  (EXPV_CODE(v) == INT_CONSTANT && EXPV_INT_VALUE(v) == 1)


expv
expv_numeric_const_reduce(left, right, code, v)
     expv left;
     expv right;
     enum expr_code code;
     expv v;
{
    BASIC_DATA_TYPE bTyp;
    TYPE_DESC tp = NULL;

    expv nL = NULL;
    expv nR = NULL;

    if (v != NULL) {
        tp = EXPV_TYPE(v);
    }
    if (tp != NULL) {
        bTyp = TYPE_BASIC_TYPE(tp);
    } else {
        if (right != NULL) {
            tp = max_type(EXPV_TYPE(left), EXPV_TYPE(right));
        } else {
            tp = EXPV_TYPE(left);
        }
        bTyp = TYPE_BASIC_TYPE(tp);
    }

    if (IS_NUMERIC_CONST_V(left)) {
        if(expr_has_param(left) || expr_has_type_param(left))
            goto NonReducedReturn;
        nL = expv_reduce_conv_const(tp, left);
    }
    if (right != NULL) {
        if (IS_NUMERIC_CONST_V(right)) {
            if(expr_has_param(right) || expr_has_type_param(right))
                goto NonReducedReturn;
            nR = expv_reduce_conv_const(tp, right);
        }
    }

    switch (bTyp) {
        case TYPE_INT: {
            omllint_t i;
            
            switch (code) {
                case MUL_EXPR: {
                    i = EXPV_INT_VALUE(nL) * EXPV_INT_VALUE(nR);
                    break;
                }
                case DIV_EXPR: {
                    if (EXPV_INT_VALUE(nR) == 0) {
                        error_at_node(v, "divide by zero");
                        goto NonReducedReturn;
                    }
                    i = EXPV_INT_VALUE(nL) / EXPV_INT_VALUE(nR);
                    break;
                }
                case PLUS_EXPR: {
                    i = EXPV_INT_VALUE(nL) + EXPV_INT_VALUE(nR);
                    break;
                }
                case MINUS_EXPR: {
                    i = EXPV_INT_VALUE(nL) - EXPV_INT_VALUE(nR);
                    break;
                }
                case UNARY_MINUS_EXPR: {
                    i = -EXPV_INT_VALUE(nL);
                    break;
                }
                case LOG_EQ_EXPR: {
                    i = (EXPV_INT_VALUE(nL) == EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_NEQ_EXPR: {
                    i = (EXPV_INT_VALUE(nL) != EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_GE_EXPR: {
                    i = (EXPV_INT_VALUE(nL) >= EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_GT_EXPR: {
                    i = (EXPV_INT_VALUE(nL) > EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_LE_EXPR: {
                    i = (EXPV_INT_VALUE(nL) <= EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_LT_EXPR: {
                    i = (EXPV_INT_VALUE(nL) < EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_AND_EXPR: {
                    i = (EXPV_INT_VALUE(nL) && EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_OR_EXPR: {
                    i = (EXPV_INT_VALUE(nL) || EXPV_INT_VALUE(nR));
                    break;
                }
                case LOG_NOT_EXPR: {
                    i = (!(EXPV_INT_VALUE(nL)));
                    break;
                }
                default: {
                    goto NonReducedReturn;
                }
            }
            return expv_int_term(INT_CONSTANT, tp, i);
        }

        default: {
            goto NonReducedReturn;
        }
    }

    NonReducedReturn:
    if (code == EXPV_CODE(v) && tp == EXPV_TYPE(v) &&
        left == EXPV_LEFT(v) && right == EXPV_RIGHT(v)) {
        return v;
    } else {
        return expv_cons(code, tp, left, right);
    }
}


/* 
 * optimize expression value
 */
expv
expv_reduce(expv v, int doParamReduce)
{
    enum expr_code code, lcode, rcode;
    TYPE_DESC tp;
    expv left,right,arg;
    list lp;
    int bytes;
    omllint_t n;
    omldouble_t f;
    
    if (v == NULL) return(v);   /* error recovery */

    code = EXPV_CODE(v);
    if (doParamReduce == TRUE && code == F_VAR) {
        /* Could be a parameter. */
        ID vId = find_ident(EXPV_NAME(v));
        if (vId != NULL && 
            (ID_CLASS(vId) == CL_PARAM ||
             TYPE_IS_PARAMETER(vId))) {
            /* Yes it is a parameter. */
            if (VAR_INIT_VALUE(vId) != NULL) {
                v = expv_reduce(VAR_INIT_VALUE(vId), TRUE);
                code = EXPV_CODE(v);
            }
        }
    }
    tp = EXPV_TYPE(v);

    /* check for terminal */
    if (EXPR_CODE_IS_TERMINAL_OR_CONST(code)) {
        return v;
    }

    if(doParamReduce && code == FUNCTION_CALL &&
        strcmp(SYM_NAME(EXPV_NAME(EXPR_ARG1(v))), "selected_int_kind") == 0) {
        arg = expv_reduce(
            expr_list_get_n(expr_list_get_n(v, 1), 0), TRUE);
        if(arg && EXPR_CODE(arg) == INT_CONSTANT) {
            n = EXPV_INT_VALUE(arg);
            bytes = 0;
            if(0 <= n && n <= 9)
                bytes = 4; // 32bit int
            else if(10 <= n && n <= 18)
                bytes = 8; // 64bit int
            else if(19 <= n && n <= 38)
                bytes = 16; // 128bit int
            if(bytes > 0)
                return expv_int_term(
                    INT_CONSTANT, type_basic(TYPE_INT), bytes);
        }
    }
    
    if(doParamReduce && code == FUNCTION_CALL &&
        strcmp(SYM_NAME(EXPV_NAME(EXPR_ARG1(v))), "kind") == 0) {
        arg = expv_reduce(
            expr_list_get_n(expr_list_get_n(v, 1), 0), TRUE);
        if(arg) {
            bytes = 0;
            TYPE_DESC at = EXPV_TYPE(arg);
            expv etk = TYPE_KIND(at);
            if(etk && EXPR_CODE(etk) == INT_CONSTANT)
                bytes = (int)EXPV_INT_VALUE(etk);
            else if(IS_DOUBLED_TYPE(at))
                bytes = 8;

            if(bytes == 0) {
                if(EXPR_CODE(arg) == INT_CONSTANT) {
                    n = EXPV_INT_VALUE(arg);
                    if(n < (int32_t)0x80000000 || (int32_t)0x7FFFFFFF < n)
                        bytes = 8; // 64bit int
                    else
                        bytes = 4; // 32bit int
                } else if(EXPR_CODE(arg) == FLOAT_CONSTANT) {
                    f = EXPV_FLOAT_VALUE(arg);
                    if(f < DBL_MIN || DBL_MAX < f)
                        bytes = SIZEOF_LONG_DOUBLE; // long double precision
                    if(f < FLT_MIN || FLT_MAX < f)
                        bytes = 8; // double precision
                    else if(DBL_MIN <= f && f <= DBL_MAX)
                        bytes = 4; // single precision
                }
            }
            if(bytes > 0)
                return expv_int_term(
                    INT_CONSTANT, type_basic(TYPE_INT), bytes);
        }
    }
    
    if(EXPR_CODE_IS_LIST(code)){
        /* expand list. */
        FOR_ITEMS_IN_LIST(lp,v)
            LIST_ITEM(lp) = expv_reduce(LIST_ITEM(lp), doParamReduce);
        return v;
    }

    if(code == F95_STRUCT_CONSTRUCTOR) {
        return v;
    }


    /* internal node */
    left = expv_reduce(EXPV_LEFT(v), doParamReduce);
    lcode = EXPV_CODE(left);
    right = EXPV_RIGHT(v);
    rcode = ERROR_NODE; /* if right is null, never use it. */
    if(right != NULL) {
        right = expv_reduce(right, doParamReduce);
        rcode = EXPV_CODE(right);
    }

    if (lcode == FUNCTION_CALL ||
        rcode == FUNCTION_CALL ||
        lcode == F95_MEMBER_REF ||
        rcode == F95_MEMBER_REF ||
        lcode == F_ARRAY_REF ||
        rcode == F_ARRAY_REF ||
        lcode == ARRAY_REF ||
        rcode == ARRAY_REF) {
        return v;
    }



    /* constant folding */
    switch(code){
    /*
     * constant folding with arithmetic operators are applied,
     * only when two types has no explicit kind. Since if they have
     * explicit kind, resulting expv may have different kind to
     * original expression.
     */

    case MUL_EXPR:
        if(TYPE_HAVE_KIND(EXPV_TYPE(left)) ||
           TYPE_HAVE_KIND(EXPV_TYPE(right)))
            break;
        if(EXPV_IS_INT_ZERO(left) || EXPV_IS_INT_ZERO(right))
            return(expv_constant_0); /* x*0 = 0 */
        if(EXPV_IS_INT_ONE(left))
            return(right);        /* x*1 = x */
        if(EXPV_IS_INT_ONE(right))
            return(left);         /* 1*x = x */

        if(lcode == INT_CONSTANT &&
           rcode == INT_CONSTANT){
            return(expv_int_term(INT_CONSTANT,EXPV_TYPE(v),
                                 EXPV_INT_VALUE(left)*EXPV_INT_VALUE(right)));
        }

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case DIV_EXPR:
        if(TYPE_HAVE_KIND(EXPV_TYPE(left)) ||
           TYPE_HAVE_KIND(EXPV_TYPE(right)))
            break;
        if(EXPV_IS_INT_ZERO(left)) return(expv_constant_0); /* 0/x = 0 */
        if(EXPV_IS_INT_ZERO(right)) /* x/0 = error */
          {
              error_at_node((expr)v, "divide by zero");
              return(v);
          }
        if(EXPV_IS_INT_ONE(right))
            return(left);         /* x/1 = x */

        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,EXPV_TYPE(v),
                                 EXPV_INT_VALUE(left)/EXPV_INT_VALUE(right)));

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case PLUS_EXPR:
        if(TYPE_HAVE_KIND(EXPV_TYPE(left)) ||
           TYPE_HAVE_KIND(EXPV_TYPE(right)))
            break;
        if(EXPV_IS_INT_ZERO(left))
            return(right);        /* 0 + x = x */
        if(EXPV_IS_INT_ZERO(right))
            return(left);         /* x + 0 = x */

        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,EXPV_TYPE(v),
                               EXPV_INT_VALUE(left)+EXPV_INT_VALUE(right)));

        /* deleted reducing float constant */

        /*  (PLUS (PLUS x c1) c2) => (PLUS x c1+c2) */
        if(rcode == INT_CONSTANT  &&
           lcode == PLUS_EXPR &&
           EXPV_CODE((expv)EXPV_RIGHT(left)) == INT_CONSTANT)
            return(expv_cons(code,tp,EXPV_LEFT(left),
                           expv_int_term(INT_CONSTANT,EXPV_TYPE(right),
                                     EXPV_INT_VALUE(right)+
                                     EXPV_INT_VALUE((expv)EXPV_RIGHT(left)))));
        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case MINUS_EXPR:
        if(TYPE_HAVE_KIND(EXPV_TYPE(left)) ||
           TYPE_HAVE_KIND(EXPV_TYPE(right)))
            break;
        if(EXPV_IS_INT_ZERO(left))
          {
              /* 0 - x -> unary minus */
              code = UNARY_MINUS_EXPR;
              left = right;
              rcode = lcode;
              right = NULL;
              break;
          }
        if(EXPV_IS_INT_ZERO(right))
            return(left);               /* x - 0 = x */

        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,EXPV_TYPE(v),
                               EXPV_INT_VALUE(left)-EXPV_INT_VALUE(right)));

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case POWER_EXPR:
        if(TYPE_HAVE_KIND(EXPV_TYPE(left)) ||
           TYPE_HAVE_KIND(EXPV_TYPE(right)))
            break;
        if(EXPV_IS_INT_ZERO(left))
            return(expv_constant_0); /* 0**x = 0 */
        if(EXPV_IS_INT_ZERO(right))
            return(expv_constant_1); /* x**0 = 1 */
        if(EXPV_IS_INT_ONE(left))
            return(expv_constant_1); /* 1**x = 1 */
        if(EXPV_IS_INT_ONE(right))
            return(left);            /* x**1 = x */

        if(lcode == INT_CONSTANT &&
           rcode == INT_CONSTANT) {
            return(expv_int_term(INT_CONSTANT,EXPV_TYPE(v),
                power_ii(EXPV_INT_VALUE(left), EXPV_INT_VALUE(right))));
        }

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case UNARY_MINUS_EXPR:
        if(lcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,EXPV_TYPE(v),
                               -EXPV_INT_VALUE(left)));

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left)) {
            return expv_numeric_const_reduce(left, (expv)NULL, code, v);
        }
        break;

    case LOG_EQ_EXPR:
        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,tp,
                               EXPV_INT_VALUE(left) == EXPV_INT_VALUE(right)));

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_NEQ_EXPR:
        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,tp,
                               EXPV_INT_VALUE(left) != EXPV_INT_VALUE(right)));

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_GE_EXPR:
        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT){
            return(expv_int_term(INT_CONSTANT,tp,
                                 EXPV_INT_VALUE(left)>=EXPV_INT_VALUE(right)));
        }

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_GT_EXPR:
        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT) {
            return(expv_int_term(INT_CONSTANT,tp,
                                 EXPV_INT_VALUE(left)>EXPV_INT_VALUE(right)));
        }

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_LE_EXPR:
        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT){
            return(expv_int_term(INT_CONSTANT,tp,
                                 EXPV_INT_VALUE(left)<=EXPV_INT_VALUE(right)));
        }

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_LT_EXPR:
        if(lcode == INT_CONSTANT && 
           rcode == INT_CONSTANT)
          {
              return(expv_int_term(INT_CONSTANT,tp,
                                   EXPV_INT_VALUE(left)<EXPV_INT_VALUE(right)));
          }

        /* deleted reducing float constant */

        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_AND_EXPR:
        if(lcode == INT_CONSTANT &&
           rcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,tp,
                               EXPV_INT_VALUE(left) && EXPV_INT_VALUE(right)));
        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_OR_EXPR:
        if(lcode == INT_CONSTANT &&
           rcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,tp,
                               EXPV_INT_VALUE(left) || EXPV_INT_VALUE(right)));
        if (IS_NUMERIC_CONST_V(left) &&
            IS_NUMERIC_CONST_V(right)) {
            return expv_numeric_const_reduce(left, right, code, v);
        }
        break;

    case LOG_NOT_EXPR:
        if(lcode == INT_CONSTANT)
            return(expv_int_term(INT_CONSTANT,tp,!EXPV_INT_VALUE(left)));
        if (IS_NUMERIC_CONST_V(left)) {
            return expv_numeric_const_reduce(left, (expv)NULL, code, v);
        }
        break;

    case FUNCTION_CALL:
        break;
    default: {}
    }
    if(code == EXPV_CODE(v) && tp == EXPV_TYPE(v) &&
       left == EXPV_LEFT(v) && right == EXPV_RIGHT(v))
        return(v);                /* no change */
    else                        /* re-construct */
        return(expv_cons(code,tp,left,right));
}


/* 
 * convert numeric value v to type 'tp' 
 */
expv expv_reduce_conv_const(TYPE_DESC tp, expv v)
{
    BASIC_DATA_TYPE vbt;

    if (tp == NULL) {
        return NULL;
    }

    if (EXPV_TYPE(v) == NULL) {
        return NULL;
    }

    vbt = TYPE_BASIC_TYPE(EXPV_TYPE(v));
    if (TYPE_BASIC_TYPE(tp) == vbt) {
        return v;
    }

    switch (TYPE_BASIC_TYPE(tp)) {

        case TYPE_REAL:
        case TYPE_DREAL: {
            omldouble_t d;
            const char *token = NULL;
            char *token1 = NULL;
            switch (vbt) {
                case TYPE_INT: {
                    assert(EXPV_CODE(v) == INT_CONSTANT);
                    d = EXPV_INT_VALUE(v);
                    token1 = (char*)malloc(64);
                    sprintf(token1, OMLL_DFMT, EXPV_INT_VALUE(v));
                    token = token1;
                    break;
                }
                case TYPE_REAL:
                case TYPE_DREAL: {
                    assert(EXPV_CODE(v) == FLOAT_CONSTANT);
                    d = EXPV_FLOAT_VALUE(v);
                    token = EXPV_ORIGINAL_TOKEN(v);
                    break;
                }
                case TYPE_COMPLEX:
                case TYPE_DCOMPLEX: {
                    assert(EXPV_CODE(v) == COMPLEX_CONSTANT);
                    expv re = EXPV_COMPLEX_REAL(v);
                    d = EXPV_FLOAT_VALUE(re);
                    token = EXPV_ORIGINAL_TOKEN(re);
                    break;
                }

                default: {
                    fatal("expv_reduce_conv_const: not a numeric constant.");
                    return NULL;
                }
            }

            return expv_float_term(FLOAT_CONSTANT, tp, d, token);
        }

        case TYPE_CHAR:
        case TYPE_INT: {
            omllint_t i;
            switch (vbt) {
                case TYPE_INT: {
                    assert(EXPV_CODE(v) == INT_CONSTANT);
                    i = EXPV_INT_VALUE(v);
                    break;
                }
                case TYPE_REAL:
                case TYPE_DREAL: {
                    assert(EXPV_CODE(v) == FLOAT_CONSTANT);
                    i = (omllint_t)EXPV_FLOAT_VALUE(v);
                    break;
                }
                case TYPE_COMPLEX:
                case TYPE_DCOMPLEX: {
                    assert(EXPV_CODE(v) == COMPLEX_CONSTANT);
                    expv re = EXPV_COMPLEX_REAL(v);
                    i = (omllint_t)EXPV_FLOAT_VALUE(re);
                    break;
                }
                default: {
                    fatal("expv_reduce_conv_const: not a numeric constant.");
                    return NULL;
                }
            }
            if (TYPE_BASIC_TYPE(tp) == TYPE_CHAR) {
                unsigned char c = (unsigned char)i;
                return expv_int_term(INT_CONSTANT, tp, c);
            } else {
                return expv_int_term(INT_CONSTANT, tp, i);
            }
        }

        case TYPE_COMPLEX:
        case TYPE_DCOMPLEX: {
            omldouble_t d;
            const char *token = NULL;
            TYPE_DESC btp = type_basic((vbt == TYPE_REAL || vbt == TYPE_COMPLEX) ?
                TYPE_REAL : TYPE_DREAL);
            switch (vbt) {
                case TYPE_INT: {
                    assert(EXPV_CODE(v) == INT_CONSTANT);
                    d = (omldouble_t)EXPV_INT_VALUE(v);
                    break;
                }
                case TYPE_DREAL:
                case TYPE_REAL: {
                    assert(EXPV_CODE(v) == FLOAT_CONSTANT);
                    d = EXPV_FLOAT_VALUE(v);
                    token = EXPV_ORIGINAL_TOKEN(v);
                    break;
                }
                case TYPE_COMPLEX:
                case TYPE_DCOMPLEX: {
                    return v;
                }
                default: {
                    fatal("expv_reduce_conv_const: not a numeric constant.");
                    return NULL;
                }
            }
            return list2(COMPLEX_CONSTANT,
                expv_float_term(FLOAT_CONSTANT, btp, d, token), expv_float_0);
        }

        case TYPE_GNUMERIC:
        case TYPE_GNUMERIC_ALL: {
            /* Children are constant, but reducible.
             * Children may have different kind variable.
             */
            return v;
            break;
        }

        default: {
            fatal("expv_reduce_conv_const: bad arithmetic type");
        }
    }
    return NULL;
}


omllint_t
power_ii(omllint_t x, omllint_t n)
{
    omllint_t pow;
    unsigned long u;

    if (n <= 0) {
        if (n == 0 || x == 1)
            return 1;
        if (x != -1)
            return x == 0 ? 1/x : 0;
        n = -n;
    }
    u = n;
    for(pow = 1; ; )
    {
        if(u & 01)
            pow *= x;
        if(u >>= 1)
            x *= x;
        else
            break;
    }
    return(pow);
}


expv
expv_complex_const_reduce(v, tp)
     expv v;
     TYPE_DESC tp;
{
    expv vI, vR;

    vR = expv_reduce_conv_const(tp, EXPR_ARG1(v));
    if(vR == NULL)
        return NULL;
    vI = expv_reduce_conv_const(tp, EXPR_ARG2(v));
    if(vI == NULL)
        return NULL;

    return expv_cons(COMPLEX_CONSTANT, tp, vR, vI);
}


int
expv_is_constant_typeof(expv x, BASIC_DATA_TYPE bt)
{
    return expr_is_constant_typeof(x, bt);
}


int
expv_is_constant(expv x)
{
    return expv_is_constant_typeof(x, TYPE_UNKNOWN);
}


