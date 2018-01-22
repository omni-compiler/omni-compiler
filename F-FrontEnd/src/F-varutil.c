/**
 * \file F-varutil.c
 */

#include "F-front.h"
#include "F-intrinsics-types.h"
#include "module-manager.h"

static expv             getTerminalExpr _ANSI_ARGS_((expr x, expv l));
static TYPE_DESC        getConstExprType _ANSI_ARGS_((expr x));


int
type_is_assumed_size_array(TYPE_DESC tp) {
    if (TYPE_N_DIM(tp) > 0 && TYPE_ARRAY_ASSUME_KIND(tp) == ASSUMED_SIZE) {
        return TRUE;
    } else if (TYPE_REF(tp) != NULL) {
        return type_is_assumed_size_array(TYPE_REF(tp));
    } else {
        return FALSE;
    }
}


BASIC_DATA_TYPE
get_basic_type(TYPE_DESC tp) {
    BASIC_DATA_TYPE ret = TYPE_UNKNOWN;

    if (tp != NULL) {
        ret = TYPE_BASIC_TYPE(tp);
        if (ret == TYPE_UNKNOWN ||
            ret == TYPE_ARRAY) {
            if (TYPE_REF(tp) != NULL) {
                ret = get_basic_type(TYPE_REF(tp));
            }
        }
    }

    return ret;
}


TYPE_DESC
get_bottom_ref_type(TYPE_DESC tp)
{
    assert(tp);
    if(TYPE_REF(tp) == NULL)
        return tp;
    return get_bottom_ref_type(TYPE_REF(tp));
}


static int
expr_is_param_typeof(expr x, BASIC_DATA_TYPE bt)
{
    if (EXPR_CODE(x) == IDENT || EXPR_CODE(x) == F_VAR) {
        ID id = find_ident(EXPR_SYM(x));
        if (id == NULL) {
            /* must not be error to compile ENTRY's parameter
             * like following.
             *
             * SUBROUTINE S(A)
             * IMPLICITI NONE
             * INTEGER A
             * INTEGER B(N)
             * ...
             * ENTRY E(A, B, N)
             * ...
             * END
             */
            warning_at_node(x, "'%s' is implicitly declared and used, "
                            "should be declared explicitly as a parameter.",
                            SYM_NAME(EXPR_SYM(x)));
            return FALSE;
        }
        if (ID_TYPE(id) != NULL) {
            if (ID_CLASS(id) == CL_PARAM &&
                TYPE_IS_PARAMETER(ID_TYPE(id)) &&
                (bt == TYPE_UNKNOWN || bt == TYPE_BASIC_TYPE(ID_TYPE(id)))) {
                return TRUE;
            }
            if (ID_STORAGE(id) == STG_ARG &&
                (bt == TYPE_UNKNOWN || bt == TYPE_BASIC_TYPE(ID_TYPE(id)))) {
                return TRUE;
            }
        } else {
            if (bt == TYPE_UNKNOWN &&
                (ID_CLASS(id) == CL_PARAM ||
                 TYPE_IS_PARAMETER(id) ||
                 ID_STORAGE(id) == STG_ARG)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}


static int
expr_is_type_param_typeof(expr x, BASIC_DATA_TYPE bt)
{
    if (EXPR_CODE(x) == IDENT || EXPR_CODE(x) == F_VAR) {
        ID id = find_ident(EXPR_SYM(x));
        if (id == NULL) {
            return FALSE;
        }
        if (ID_TYPE(id) != NULL) {
            if (ID_CLASS(id) == CL_TYPE_PARAM &&
                (bt == TYPE_UNKNOWN || bt == TYPE_BASIC_TYPE(ID_TYPE(id)))) {
                return TRUE;
            }
        }
    }
    return FALSE;
}


int
expr_is_param(x)
     expr x;
{
    return expr_is_param_typeof(x, TYPE_UNKNOWN);
}


int
expr_has_param(x)
     expr x;
{
    if(x == NULL)
        return FALSE;

    if(expr_is_param(x))
        return TRUE;

    if(EXPR_CODE_IS_TERMINAL_OR_CONST(EXPR_CODE(x)) == FALSE) {
        list lp;
        FOR_ITEMS_IN_LIST(lp, x) {
            if(lp && expr_has_param(LIST_ITEM(lp)))
                return TRUE;
        }
    }

    return FALSE;
}

int
expr_is_type_param(x)
     expr x;
{
    return expr_is_type_param_typeof(x, TYPE_UNKNOWN);
}


int
expr_has_type_param(x)
     expr x;
{
    if(x == NULL)
        return FALSE;

    if(expr_is_type_param(x))
        return TRUE;

    if(EXPR_CODE_IS_TERMINAL_OR_CONST(EXPR_CODE(x)) == FALSE) {
        list lp;
        FOR_ITEMS_IN_LIST(lp, x) {
            if(lp && expr_has_type_param(LIST_ITEM(lp)))
                return TRUE;
        }
    }

    return FALSE;
}

int
expr_list_is_constant_expr(const expr x)
{
    list lp;
    FOR_ITEMS_IN_LIST(lp, x) {
        if (LIST_ITEM(lp) == NULL)
            continue;
        if (!expr_is_constant(LIST_ITEM(lp)))
            return FALSE;
    }
    return TRUE;
}

int
intrinsic_call_is_constant_expression(const ID id, const expr arg)
{
    intrinsic_entry *ep = NULL;

    if (id == NULL ||
        ID_SYM(id) == NULL)
        return FALSE;

    if (SYM_TYPE(ID_SYM(id)) != S_INTR) {
        if (TYPE_IS_INTRINSIC(id) ||
            (ID_TYPE(id) && TYPE_IS_INTRINSIC(ID_TYPE(id)))) {
            /*
             * `id` has a INTRINSIC attribute but is not listed in F-intrinsic-tables.c.
             * So appcept this as a elemental function.
             */
            return expr_list_is_constant_expr(arg);
        } else {
            return FALSE;
        }
    }

    ep = &(intrinsic_table[SYM_VAL(ID_SYM(id))]);

    switch (INTR_CLASS(ep)) {
        case INTRINSIC_CLASS_ELEMENTAL_FUN:
            return expr_list_is_constant_expr(arg);
            break;
        case INTRINSIC_CLASS_TRANS:
            if (INTR_OP(ep) == INTR_NUM_IMAGES ||
                INTR_OP(ep) == INTR_THIS_IMAGE) {
                /* TODO(shingo-s):
                 *  Add INTR_COMMAND_ARGUMENT_COUNT
                 */
                return FALSE;
            } else if (INTR_OP(ep) == INTR_NULL) {
                /* TODO(shingo-s):
                 *  Check if an arugment does not have assumed size type parameters
                 */
                return expr_list_is_constant_expr(arg);
            } else {
                return expr_list_is_constant_expr(arg);
            }
            break;
        default:
            break;
    }

    return FALSE;
}


int
expr_is_constant(x)
     expr x;
{
    return expr_is_constant_typeof(x, TYPE_UNKNOWN);
}

int
expr_is_constant_typeof(x, bt)
     expr x;
     BASIC_DATA_TYPE bt;
{
    switch (EXPR_CODE(x)) {
    /* terminal */
    case COMPLEX_CONSTANT:
        return (bt == TYPE_UNKNOWN || bt == TYPE_COMPLEX || bt == TYPE_DCOMPLEX);
    case F_TRUE_CONSTANT:
    case F_FALSE_CONSTANT:
        return (bt == TYPE_UNKNOWN || bt == TYPE_LOGICAL);
    case STRING_CONSTANT:
        return (bt == TYPE_UNKNOWN || bt == TYPE_CHAR);
    case INT_CONSTANT:
        return (bt == TYPE_UNKNOWN || bt == TYPE_INT);
    case F_DOUBLE_CONSTANT:
        return (bt == TYPE_UNKNOWN || bt == TYPE_DREAL);
    case FLOAT_CONSTANT:
        return (bt == TYPE_UNKNOWN || bt == TYPE_REAL || bt == TYPE_DREAL);
    case F95_CONSTANT_WITH:
        return expr_is_constant_typeof(EXPR_ARG1(x), bt);
    case IDENT:
    case F_VAR:
        return expr_is_param_typeof(x, bt) || expr_is_type_param_typeof(x, bt);

    case F_UNARY_MINUS_EXPR:
    case UNARY_MINUS_EXPR:
        if (EXPR_ARG1(x) == NULL) {
            fatal("internal compiler error.");
        }
        return expr_is_constant_typeof(EXPR_ARG1(x), bt);

    case F_PLUS_EXPR:
    case F_MINUS_EXPR:
    case F_MUL_EXPR:
    case F_DIV_EXPR:
    case F_POWER_EXPR:
    case PLUS_EXPR:
    case MINUS_EXPR:
    case MUL_EXPR:
    case DIV_EXPR:
    case POWER_EXPR:
    case F_CONCAT_EXPR:
        if (EXPR_ARG1(x) == NULL) {
            fatal("internal compiler error.");
        }
        if (EXPR_ARG2(x) == NULL) {
            fatal("internal compiler error.");
        }
        if (expr_is_constant_typeof(EXPR_ARG1(x), bt) == FALSE) {
            return FALSE;
        }
        if (expr_is_constant_typeof(EXPR_ARG2(x), bt) == FALSE) {
            return FALSE;
        }
        return TRUE;

    case ARRAY_REF: /* acutually, ARRAY_REF is for expv */
    case F_ARRAY_REF: {
        list lp;
        expr x1 = EXPR_ARG1(x);
        expv v;
        TYPE_DESC tp;
        ID id;
        if (EXPR_CODE(x1) == IDENT || EXPR_CODE(x1) == F_VAR) {
            /*
             * expr may be array ref or character ref or intrinsic call
             */

            if (((id = find_ident(EXPR_SYM(x1))) == NULL && SYM_TYPE(EXPR_SYM(x1)) == S_INTR) ||
                (ID_CLASS(id) == CL_PROC && PROC_CLASS(id) == P_INTRINSIC)) {
                if (id == NULL) {
                    id = declare_ident(EXPR_SYM(x1), CL_UNKNOWN);
                    ID_CLASS(id) = CL_PROC;
                    PROC_CLASS(id) = P_INTRINSIC;
                    TYPE_SET_INTRINSIC(id);
                    ID_STORAGE(id) = STG_NONE;
                    ID_IS_DECLARED(id) = TRUE;
                }

                if (!intrinsic_call_is_constant_expression(id, EXPR_ARG2(x))) {
                    return FALSE;
                }

                if ((v = compile_function_call_check_intrinsic_arg_type(id, EXPR_ARG2(x), TRUE)) == NULL) {
                    return FALSE;
                }
                if (bt != TYPE_UNKNOWN) {
                    tp = EXPV_TYPE(v);
                    if (getBasicType(tp) != bt) {
                        return FALSE;
                    }
                }
            } else if (id != NULL) {
                if (bt != TYPE_UNKNOWN) {
                    if (ID_TYPE(id) != NULL) {
                        if ((ID_CLASS(id) != CL_PARAM ||
                             (!TYPE_IS_PARAMETER(ID_TYPE(id)))) &&
                            (bt != get_basic_type(ID_TYPE(id)))) {
                            return FALSE;
                        }
                    } else {
                        return FALSE;
                    }
                } else {
                    if (ID_CLASS(id) != CL_PARAM &&
                        !TYPE_IS_PARAMETER(id) &&
                        !(ID_TYPE(id) && TYPE_IS_PARAMETER(ID_TYPE(id)))) {
                        return FALSE;
                    }
                }
            } else {
                return FALSE;
            }
        }
        /*
         * Check array indices or instinsic function arguments are constant values
         */
        FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(x)) {
            if (EXPR_CODE(x1) == IDENT && LIST_ITEM(lp) == x1) {
                continue;
            }
            if (!expr_is_constant(LIST_ITEM(lp))) {
                return FALSE;
            }
        }
        return TRUE;
    } break;

    case FUNCTION_CALL: {

        char *name = NULL;
        SYMBOL s = EXPV_NAME(EXPR_ARG1(x));
        ID fId = find_ident(s);
        expv v;

        if(fId != NULL && PROC_CLASS(fId) == P_INTRINSIC){
            name = SYM_NAME(ID_SYM(fId));
        } else if(SYM_TYPE(s) == S_INTR){
            fId = declare_ident(s, CL_PROC);
            PROC_CLASS(fId) = P_INTRINSIC;
            declare_function(fId);
            name = SYM_NAME(s);
        }

        if (name == NULL) return FALSE; // error

        if (!intrinsic_call_is_constant_expression(fId, EXPR_ARG2(x))) {
            return FALSE;
        }

        if ((v = compile_function_call_check_intrinsic_arg_type(fId, EXPR_ARG2(x), TRUE)) == NULL) {
            return FALSE;
        }
        if (bt != TYPE_UNKNOWN) {
            TYPE_DESC tp = EXPV_TYPE(v);
            if (getBasicType(tp) != bt) {
                return FALSE;
            }
        }

        return TRUE;
    } break;

    default:
        break;
    }
    return FALSE;
}


static expv
getTerminalExpr(x, l)
     expr x;
     expv l;
{
    if (l == NULL) {
        l = list0(LIST);
    }
    if (EXPR_CODE_IS_TERMINAL(EXPR_CODE(x))) {
        list_put_last(l, compile_expression(x));
    } else {
        list lp;
        FOR_ITEMS_IN_LIST(lp, x) {
            getTerminalExpr(LIST_ITEM(lp), l);
        }
    }
    return l;
}


static TYPE_DESC
getConstExprType(x)
     expr x;
{
    expv l = getTerminalExpr(x, NULL);
    list lp;
    TYPE_DESC ret = BASIC_TYPE_DESC(TYPE_INT); /* minimum numeric type. */
    expv v;

    FOR_ITEMS_IN_LIST(lp, l) {
        v = LIST_ITEM(lp);
        if (!(IS_NUMERIC_CONST_V(v))) {
            return EXPV_TYPE(v);
        }
        ret = max_type(ret, EXPV_TYPE(v));
    }
    return ret;
}


expv
expr_constant_value(expr x)
{
    expv ret = NULL;

    if (EXPR_CODE_IS_TERMINAL_OR_CONST(EXPR_CODE(x))) {
        return compile_terminal_node(x);
    }

    switch (EXPR_CODE(x)) {
        case F95_CONSTANT_WITH:
        case F_POWER_EXPR:
        case POWER_EXPR:
        case F_UNARY_MINUS_EXPR:
        case UNARY_MINUS_EXPR:
        case F_PLUS_EXPR:
        case PLUS_EXPR:
        case F_MINUS_EXPR:
        case MINUS_EXPR:
        case F_MUL_EXPR:
        case MUL_EXPR:
        case F_DIV_EXPR:
        case DIV_EXPR: {

            if (expr_is_constant(x)) {
                TYPE_DESC tp = getConstExprType(x);

                if (tp == NULL) {
                    fatal("can't determine type of constant expression.");
                    break;
                }
                if (!(IS_NUMERIC(tp))) {
                    expv new = expv_reduce(compile_expression(x), FALSE);
                    if (new != NULL) {
                        if (expr_is_constant(new)) {
                            ret = new;
                        }
                    }
                } else {
                    switch (TYPE_BASIC_TYPE(tp)) {
                        case TYPE_COMPLEX:
                        case TYPE_DCOMPLEX: {
                            ret = expv_complex_const_reduce(x, tp);
                            break;
                        }

                        default: {
                            expv new = expv_reduce(compile_expression(x), FALSE);
                            if (expr_is_constant(new) == TRUE) {
                                ret = new;
                            }
                            break;
                        }
                    }
                }
            }
            break;
        }

        default: {
            break;
        }
    }

    if (ret == NULL) {
        return NULL;
    }
    return ret;
}


expv
expr_label_value(x)
     expr x;
{
    expv v = expr_constant_value(x);
    if (v == NULL)
        return NULL;
    return (EXPV_CODE(v) == INT_CONSTANT) ? v : NULL;
}


int
expr_is_variable(expr x, int force, ID *idPtr)
{
    ID id = NULL;
    expr varName;
    int ret = FALSE;

    switch (EXPR_CODE(x)) {
        case IDENT: {
            varName = x;
            break;
        }
        case F_SUBSTR_REF:
        case F_ARRAY_REF: {
            varName = EXPR_ARG1(x);
            break;
        }
        default: {
            goto Done;
        }
    }

    id = declare_ident(EXPR_SYM(varName), CL_UNKNOWN);
    if (force == TRUE) {
        if (ID_CLASS(id) == CL_UNKNOWN) {
            ID_CLASS(id) = CL_VAR;
        }
    }
    if (ID_CLASS(id) == CL_VAR) {
        ret = TRUE;
    }

    Done:
    if (idPtr != NULL) {
        *idPtr = id;
    }
    return ret;
}


int
expr_is_array(x, force, idPtr)
     expr x;
     int force;
     ID *idPtr;
{
    ID id = NULL;
    int ret = FALSE;

    if (expr_is_variable(x, force, &id) == TRUE) {
        if (ID_TYPE(id) != NULL) {
            if (IS_ARRAY_TYPE(ID_TYPE(id))) {
                ret = TRUE;
            }
        } else {
            if (VAR_IS_UNCOMPILED_ARRAY(id) == TRUE) {
                ret = TRUE;
            }
        }
    }

    if (idPtr != NULL) {
        *idPtr = id;
    }
    
    return ret;
}

static void
getArrayDimSpec(tp, new)
     TYPE_DESC tp;
     expv new;
{
    if (IS_ARRAY_TYPE(tp) && TYPE_REF(tp) != NULL) {
        expv sV;
        fix_array_dimensions(tp);
        getArrayDimSpec(TYPE_REF(tp), new);
        sV = list3(LIST,
                   TYPE_DIM_SIZE(tp),
                   TYPE_DIM_LOWER(tp),
                   TYPE_DIM_UPPER(tp));
        list_put_last(new, sV);
    }
}


expv
id_array_dimension_list(id)
     ID id;
{
    expv ret = NULL;

    if (IS_ARRAY_TYPE(ID_TYPE(id))) {
        ret = list0(LIST);
        getArrayDimSpec(ID_TYPE(id), ret);
    }
    return ret;
}


expv
id_array_spec_list(id)
     ID id;
{
    expv ret = NULL;

    if (IS_ARRAY_TYPE(ID_TYPE(id))) {
        ret = list2(LIST,
                    expv_int_term(INT_CONSTANT, type_INT, TYPE_N_DIM(ID_TYPE(id))),
                    id_array_dimension_list(id));
    }
    return ret;
}


expv
expr_array_spec_list(x, idPtr)
     expr x;
     ID *idPtr;
{
    ID id = NULL;
    expv ret = NULL;

    if (expr_is_array(x, TRUE, &id) == FALSE) {
        if (id != NULL) {
            VAR_ARRAY_INFO(id) = NULL;
        }
        goto Done;
    }
    if (VAR_ARRAY_INFO(id) == NULL) {
        ret = id_array_spec_list(id);
        VAR_ARRAY_INFO(id) = ret;
    } else {
        ret = VAR_ARRAY_INFO(id);
    }

    Done:
    if (idPtr != NULL) {
        *idPtr = id;
    }
    return ret;
}


int
compute_element_offset(aSpec, idxV)
     expv aSpec;        /* VAR_ARRAY_INFO(id)
                           or
                           expr_array_spec_list(expr, ID *) */
     expv idxV;         /* (LIST (INT_CONSTANT xx) (INT_CONSTANT xx) ...) */
{
    int mul = 1;
    int off = 0;
    int i;
    int n = EXPV_INT_VALUE(EXPR_ARG1(aSpec));
    expv v = EXPR_ARG2(aSpec);
    expv cDimV;
    int cIdx;
    int idxDim = 0;
    list lp;

    FOR_ITEMS_IN_LIST(lp, idxV) {
        idxDim++;
    }
    if (n != idxDim) {
        error("invalid dimension, array = %d, index = %d.\n",
              n, idxDim);
        return -1;
    }

    for (i = 0; i < n; i++) {
        cDimV = expr_list_get_n(v, i);
        if (cDimV == NULL) {
            error("can't get array offset to initialize.");
            return -1;
        }
        cIdx = EXPV_INT_VALUE(expr_list_get_n(idxV, i)) - EXPV_INT_VALUE(EXPR_ARG2(cDimV));
        off += mul * cIdx;
        mul *= EXPV_INT_VALUE(EXPR_ARG1(cDimV));
    }
    return off;
}


expv
expr_array_index(x)
     expr x;
{
    expr idx;
    list lp;
    expr y;
    expv ret;
    expv tmp;

    if (expr_is_array(x, FALSE, NULL) == FALSE) {
        return NULL;
    }
    idx = EXPR_ARG2(x);
    ret = list0(LIST);

    FOR_ITEMS_IN_LIST(lp, idx) {
        y = LIST_ITEM(lp);

        switch (EXPR_CODE(y)) {
            case INT_CONSTANT: {
                tmp = expv_int_term(INT_CONSTANT, type_INT, EXPR_INT(y));
                break;
            }
            default: {
                tmp = expr_constant_value(y);
                if (tmp == NULL || IS_INT(EXPV_TYPE(tmp)) == FALSE) {
                    error("array index is not a integer constant.");
                    return NULL;
                }
                break;
            }
        }
        ret = list_put_last(ret, tmp);
    }

    return ret;
}

int
type_is_inqurable(TYPE_DESC tp)
{
    /*
     * TODO: implement check code according to JIS X 3001-1 (ISO/IEC 1539-1)
     *       7.1.6 Specification expression (7)-(b)
     */
    if(tp == NULL)
        return FALSE;
    return TRUE;
}

#define EXPV_IS_INTRINSIC_CALL(x) \
    (EXPV_CODE(x) == FUNCTION_CALL && SYM_TYPE(EXPV_NAME(EXPR_ARG1(x))) == S_INTR)
#define EXPV_IS_OBJECT_DESIGNATOR(x) \
    (EXPV_CODE(x) == F_VAR || \
     EXPV_CODE(x) == F_ARRAY_REF || \
     EXPV_CODE(x) == ARRAY_REF || \
     EXPV_CODE(x) == F95_MEMBER_REF)

expv
base_object(expv x)
{
    expv ret = NULL;

    if (x != NULL) {
        switch (EXPV_CODE(x)) {
        case F_VAR:
            ret = x;
            break;
        case F_ARRAY_REF:
        case ARRAY_REF:
        case F95_MEMBER_REF:
            ret = base_object(EXPV_LEFT(x));
            break;
        default:
            break;
        }
    }

    return ret;
}

/**
 * Checks if expv is function argument and don't have attribute optional or intent(out).
 */
int
expv_is_restricted_argument(expv x)
{
    if(EXPV_CODE(x) == F_VAR) {
        TYPE_DESC tp;
        ID id = find_ident(EXPV_NAME(x));
        /* check id is argument. */
        if(id == NULL || !(ID_IS_DUMMY_ARG(id)))
            return FALSE;
        /* check type of id. */
        if((tp = ID_TYPE(id)) == NULL)
            return FALSE;
        if(TYPE_IS_SCALAR(tp))
            return TRUE;
        return FALSE;
    }

    return FALSE;
}

/**
 * Checks if expv is an intrinsic inquiry function call
 *
 * NOTE:
 *
 *  The inquiry function is a function returns value which depends on only the
 *  type of arguments (not on the value of arguments)
 */
static int
expv_is_intrinsic_inquiry(expv x)
{
    if (EXPV_IS_INTRINSIC_CALL(x)) {
        intrinsic_entry *ep = NULL;
        ep = &(intrinsic_table[SYM_VAL(EXPV_NAME(EXPR_ARG1(x)))]);
        return INTR_CLASS(ep) == INTRINSIC_CLASS_INQUIRY;
    } else {
        return FALSE;
    }
}


typedef struct {
    const char * module_name;
    const char * function_name;
} inquiry_intrinsic_module_entry;


const inquiry_intrinsic_module_entry inquiry_in_intrinsic_module_table[] = {
    {"iso_c_binding",       "c_sizeof"},
    {"iso_fortran_env",     "compiler_version"},
    {NULL,                  "compiler_options"},
    {"ieee_arithmetic",     "ieee_support_datatype"},
    {NULL,                  "ieee_support_denormal"},
    {NULL,                  "ieee_support_divide"},
    {NULL,                  "ieee_support_inf"},
    {NULL,                  "ieee_support_io"},
    {NULL,                  "ieee_support_nan"},
    {NULL,                  "ieee_support_rounding"},
    {NULL,                  "ieee_support_sqrt"},
    {NULL,                  "ieee_support_standard"},
    {NULL,                  "ieee_support_underflow_control"},
    {"ieee_exceptions",     "ieee_support_flag"},
    {NULL,                  "ieee_support_halting"},
    {NULL,                  NULL},
};


/**
 * Checks if expv is an inquiry function of the intrinsic module
 */
static int
expv_is_inquiry_in_intrinsic_module(const expv x)
{
    assert(x != NULL);

    SYMBOL sym = EXPV_NAME(EXPR_ARG1(x));
    ID id = find_ident(sym);

    if (id != NULL &&
        ID_USEASSOC_INFO(id) != NULL &&
        MODULE_IS_INTRINSIC(ID_MODULE(id)) == TRUE) {
        const inquiry_intrinsic_module_entry * entry = NULL;
        const char * module_name = NULL;

        for (entry = inquiry_in_intrinsic_module_table;
             entry->module_name != NULL && entry->function_name != NULL;
             entry++) {
            if (entry->module_name != NULL) {
                module_name = entry->module_name;
            }

            if (strcmp(module_name, SYM_NAME(ID_MODULE_NAME(id))) == 0 &&
                strcmp(entry->function_name,
                       SYM_NAME(ID_ORIGINAL_NAME(id)?:ID_SYM(id))) == 0) {
                return TRUE;
            }
        }
    }

    return FALSE;
}


int
expv_is_specification_function_call(expv x)
{
    ID id;
    TYPE_DESC tp;

    /* specification function is a function */
    if (EXPV_CODE(x) != FUNCTION_CALL)
        return FALSE;

    /* not an intrinsic function */
    if (EXPV_IS_INTRINSIC_CALL(x))
        return FALSE;

    id = find_ident(EXPV_NAME(EXPR_ARG1(x)));
    if (id == NULL)
        return FALSE;

    /* not a statement function */
    if (PROC_CLASS(id) == P_STFUNCT)
        return FALSE;

    tp = ID_TYPE(id);
    if (tp == NULL)
        return FALSE;

    /* must be pure */
    if ((!PROC_IS_PURE(id) && !TYPE_IS_PURE(tp)) ||
        (TYPE_IS_ELEMENTAL(tp) && !TYPE_IS_IMPURE(tp)))
        return FALSE;

    /* NOT IMPLEMENTED: not an internal function. but we can't check this */

    /* does not have dummy procedure as argument */
    list lp;
    FOR_ITEMS_IN_LIST(lp, PROC_ARGS(id)) {
        ID arg = find_ident(EXPV_NAME(LIST_ITEM(lp)));
        EXT_ID eid = arg != NULL ? PROC_EXT_ID(arg) : NULL;
        if (eid != NULL && EXT_IS_DUMMY(eid))
            return FALSE;
        if (arg != NULL && ID_TYPE(arg) && IS_PROCEDURE_TYPE(ID_TYPE(arg)))
            return FALSE;

    }

    return TRUE;
}

#define EXPV_IS_INTRINSIC_OP(x) \
    ((EXPV_CODE(x) == PLUS_EXPR) || \
    (EXPV_CODE(x) == MINUS_EXPR) || \
    (EXPV_CODE(x) == MUL_EXPR) || \
    (EXPV_CODE(x) == DIV_EXPR) || \
    (EXPV_CODE(x) == POWER_EXPR) || \
    (EXPV_CODE(x) == LOG_EQ_EXPR) || \
    (EXPV_CODE(x) == LOG_NEQ_EXPR) || \
    (EXPV_CODE(x) == LOG_GT_EXPR) || \
    (EXPV_CODE(x) == LOG_GE_EXPR) || \
    (EXPV_CODE(x) == LOG_LT_EXPR) || \
    (EXPV_CODE(x) == LOG_LE_EXPR) || \
    (EXPV_CODE(x) == F_EQV_EXPR) || \
    (EXPV_CODE(x) == F_NEQV_EXPR) || \
    (EXPV_CODE(x) == LOG_OR_EXPR) || \
    (EXPV_CODE(x) == LOG_AND_EXPR))

int expv_list_is_restricted(expv x);

/**
 * Checks if expv is a restricted expression.
 */
int
expv_is_restricted(expv x)
{
    if(x == NULL)
        return FALSE;

    /* x is intrinsic operation, all child must be restricted. */
    if(EXPV_IS_INTRINSIC_OP(x)) {
        list lp;
        FOR_ITEMS_IN_LIST(lp, x) {
            expv xx = LIST_ITEM(lp);
            if(expv_is_restricted(xx) == FALSE)
                return FALSE;
        }
        return TRUE;
    }

    /* x is constant or part of it. */
    if(expr_is_constant(x))
        return TRUE;

    if(EXPV_IS_OBJECT_DESIGNATOR(x)) {
        expv base = base_object(x);
        if (base == NULL)
            return FALSE;

        /* x is a part of constant */
        if(expr_is_constant(base)) {
            return TRUE;
        }

        /* x is argument without optional or intent(out) or part of it. */
        if(expv_is_restricted_argument(base)) {
            return TRUE;
        }

        /* x is a variable in common block or part of it. */
        if(EXPV_CODE(base) == IDENT || EXPV_CODE(base) == F_VAR) {
            ID id = find_ident(EXPV_NAME(base));
            if (id != NULL) {
                if (ID_STORAGE(id) == STG_COMMON)
                    return TRUE;

                if (ID_STORAGE(id) == STG_ARG &&
                    !(TYPE_IS_OPTIONAL(id) ||
                      TYPE_IS_INTENT_OUT(id) ||
                      TYPE_IS_INTENT_INOUT(id)) &&
                    !(ID_TYPE(id) != NULL &&
                      (TYPE_IS_OPTIONAL(ID_TYPE(id)) ||
                       TYPE_IS_INTENT_OUT(ID_TYPE(id)) ||
                       TYPE_IS_INTENT_INOUT(ID_TYPE(id)))))
                    return TRUE;
            }
        }

        /* x is a variable which is be accecible by host or use association */
        if(EXPV_CODE(base) == IDENT || EXPV_CODE(base) == F_VAR) {
            ID id = find_ident(EXPV_NAME(base));
            if (id != NULL && ID_IS_OFMODULE(id))
                return TRUE;
            id = find_ident_outer_scope(EXPV_NAME(base));
            if (id != NULL)
                return TRUE;
        }
    }

    /* x is array constructor with all elements is restricted expression. */
    if(EXPV_CODE(x) == F95_ARRAY_CONSTRUCTOR ||
       EXPV_CODE(x) == F03_TYPED_ARRAY_CONSTRUCTOR) {
        if (expv_list_is_restricted(EXPV_LEFT(x)))
            return TRUE;
    }

    /* x is implied do with expression except variable is restricted expression. */
    if(EXPV_CODE(x) == F_IMPLIED_DO) {
        list lp;
        FOR_ITEMS_IN_LIST(lp,EXPV_LEFT(x)) {
            expv xx = LIST_ITEM(lp);
            if(EXPV_CODE(xx) == F_VAR)
                continue;
            if(!expv_is_restricted(xx))
                return FALSE;
        }
    }

    /* x is struct constructor with all elements is restricted expression. */
    if(EXPV_CODE(x) == F95_STRUCT_CONSTRUCTOR) {
        if (expv_list_is_restricted(EXPV_RIGHT(x)))
            return TRUE;
    }


    /* x is a specification inquiry */
    if(expv_is_intrinsic_inquiry(x)) {
        /*check parameter*/
        list lp;
        expv v = EXPR_ARG2(x);
        FOR_ITEMS_IN_LIST(lp,v) {
            expv xx = LIST_ITEM(lp);
            if(expv_is_restricted(xx))
                continue;
            if(type_is_inqurable(EXPV_TYPE(xx)))
                continue;
        }
        return TRUE;
    }

    /* x is an other intrinsic function call. */
    if(EXPV_IS_INTRINSIC_CALL(x)) {
       if (expv_list_is_restricted(EXPR_ARG2(x)))
           return TRUE;
    }

    /* x is an inquiry function of the intrinsic module */
    if(expv_is_inquiry_in_intrinsic_module(x)) {
       if (expv_list_is_restricted(EXPR_ARG2(x)))
           return TRUE;
    }


    /* x is a specification function call. */
    if(expv_is_specification_function_call(x)) {
       if (expv_list_is_restricted(EXPR_ARG2(x)))
           return TRUE;
    }

    return FALSE;
}

/**
 * Checks if list of expv is a restricted expression.
 */
int
expv_list_is_restricted(expv x)
{
    list lp;

    if (x == NULL || EXPR_CODE(x) != LIST)
        return FALSE;

    FOR_ITEMS_IN_LIST(lp,x) {
        expv xx = LIST_ITEM(lp);
        if(!expv_is_restricted(xx))
            return FALSE;
    }

    return TRUE;
}

/**
 * Checks if expv is a specification expression.
 */
int
expv_is_specification(expv x)
{
    if (EXPV_TYPE(x) != NULL &&
        TYPE_BASIC_TYPE(EXPV_TYPE(x)) == TYPE_INT &&
        TYPE_N_DIM(EXPV_TYPE(x)) == 0 &&
        (expv_is_restricted(x)))
        return TRUE;
    else
        return FALSE;
}
