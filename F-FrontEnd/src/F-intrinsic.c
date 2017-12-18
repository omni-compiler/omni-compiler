/**
 * \file F-intrinsic.c
 */

#include "F-front.h"
#include "F-intrinsics-types.h"

#define isValidType(tp)         \
    (tp != NULL && get_basic_type(tp) != TYPE_UNKNOWN)
#define isValidTypedExpv(v)     (v != NULL && isValidType(EXPV_TYPE(v)))

int langSpecSet = LANGSPEC_DEFAULT_SET;


static int              compare_intrinsic_arg_type(expv arg,
                                                   TYPE_DESC tp,
                                                   INTR_DATA_TYPE iType);
static void             generate_reverse_dimension_expr(TYPE_DESC tp,
                                                        expr dimSpec);
static TYPE_DESC        get_intrinsic_return_type(intrinsic_entry *ep,
                                                  expv args,
                                                  expv kindV);
static BASIC_DATA_TYPE  intr_type_to_basic_type(INTR_DATA_TYPE iType);

static INTR_DATA_TYPE COARRAY_TO_BASIC_MAP[] = {
    /* INTR_TYPE_COARRAY_ANY            -> */ INTR_TYPE_ANY,
    /* INTR_TYPE_COARRAY_INT            -> */ INTR_TYPE_INT,
    /* INTR_TYPE_COARRAY_REAL           -> */ INTR_TYPE_REAL,
    /* INTR_TYPE_COARRAY_LOGICAL        -> */ INTR_TYPE_LOGICAL,
};

#define CONVERT_COARRAY_TO_BASIC(x) \
    (COARRAY_TO_BASIC_MAP[(x) - INTR_TYPE_COARRAY_ANY])


void
initialize_intrinsic() {
    int i;
    SYMBOL sp;
    intrinsic_entry *ep;

    for (i = 0, ep = &intrinsic_table[0];
        INTR_OP((ep = &intrinsic_table[i])) != INTR_END; i++){
        if ((ep->langSpec & langSpecSet) == 0) {
            continue;
        }
        if (!(isValidString(INTR_NAME(ep)))) {
            continue;
        }
        if (INTR_HAS_KIND_ARG(ep)) {
            if (((INTR_OP(ep) != INTR_MINLOC) && 
                 (INTR_OP(ep) != INTR_MAXLOC)) &&
                INTR_RETURN_TYPE_SAME_AS(ep) != -1) {
                fatal("%: Invalid intrinsic initialization.", __func__);
            }
        }
        sp = find_symbol((char *)INTR_NAME(ep));
        SYM_TYPE(sp) = S_INTR;
        SYM_VAL(sp) = i;
    }
}


int
is_intrinsic_function(ID id) {
    return (SYM_TYPE(ID_SYM(id)) == S_INTR) ? TRUE : FALSE;
}


expv
compile_intrinsic_call(ID id, expv args) {
    return compile_intrinsic_call0(id, args, FALSE);
}

expv
compile_intrinsic_call0(ID id, expv args, int ignoreTypeMismatch) {
    intrinsic_entry *ep = NULL;
    int found = 0;
    int nArgs = 0;
    int nIntrArgs = 0;
    int i;
    expv ret = NULL;
    expv a = NULL;
    TYPE_DESC tp = NULL, ftp;
    list lp;
    INTR_OPS iOps = INTR_END;
    const char *iName = NULL;
    expv kindV = NULL;
    int typeNotMatch = 0;
    int isVarArgs = 0;
    EXT_ID extid;

    if (SYM_TYPE(ID_SYM(id)) != S_INTR) {
        if (args == NULL) {
            args = list0(LIST);
        }

        tp = ID_TYPE(id);

        if (tp == NULL) {
            warning_at_node(args,
                          "unknown type of '%s' declared as intrinsic",
                          SYM_NAME(ID_SYM(id)));
            ID_TYPE(id) = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
            TYPE_ATTR_FLAGS(ID_TYPE(id)) = TYPE_ATTR_FLAGS(id);
            tp = ID_TYPE(id);
        }

        expv symV = expv_sym_term(F_FUNC, NULL, ID_SYM(id));

        if (IS_PROCEDURE_TYPE(tp)) {
            ftp = tp;
            tp = FUNCTION_TYPE_RETURN_TYPE(ftp);

        } else {
            ftp = intrinsic_function_type(tp);

            extid = new_external_id_for_external_decl(ID_SYM(id), ftp);
            ID_TYPE(id) = ftp;
            PROC_EXT_ID(id) = extid;
            if (TYPE_IS_EXTERNAL(tp)){
                ID_STORAGE(id) = STG_EXT;
            }
            else {
                EXT_PROC_CLASS(extid) = EP_INTRINSIC;
            }
        }

        EXPV_TYPE(symV) = ftp;
        return expv_cons(FUNCTION_CALL, tp, symV, args);
    }

    ep = &(intrinsic_table[SYM_VAL(ID_SYM(id))]);
    iOps = INTR_OP(ep);
    iName = ID_NAME(id);

    /* Count a number of argument, first. */
    nArgs = 0;
    if (args == NULL) {
        args = list0(LIST);
    }
    FOR_ITEMS_IN_LIST(lp, args) {
        nArgs++;
    }

    /* Search an intrinsic by checking argument types. */
    found = 0;
    for (;
         ((INTR_OP(ep) == iOps) &&
          ((strcasecmp(iName, INTR_NAME(ep)) == 0) ||
           !(isValidString(INTR_NAME(ep)))));
         ep++) {

        kindV = NULL;
        typeNotMatch = 0;
        isVarArgs = 0;

        /* Check a number of arguments. */
        if (INTR_N_ARGS(ep) < 0 ||
            INTR_N_ARGS(ep) == nArgs) {
            /* varriable args or no kind arg. */
            if (INTR_N_ARGS(ep) < 0) {
                isVarArgs = 1;
            }
            nIntrArgs = nArgs;
        } else if (INTR_HAS_KIND_ARG(ep) &&
                   ((INTR_N_ARGS(ep) + 1) == nArgs)) {
            /* could be intrinsic call with kind arg. */

            expv lastV = expr_list_get_n(args, nArgs - 1);
            if (lastV == NULL) {
                return NULL;    /* error recovery */
            }
            if (EXPV_KW_IS_KIND(lastV)) {
                goto gotKind;
            }
            tp = EXPV_TYPE(lastV);
            if (!(isValidType(tp))) {
                return NULL;    /* error recovery */
            }
            if (TYPE_BASIC_TYPE(tp) != TYPE_INT) {
                /* kind arg must be integer type. */
                continue;
            }

            gotKind:
            nIntrArgs = INTR_N_ARGS(ep);
            kindV = lastV;
        } else {
            continue;
        }

        /* The number of arguments matchs. Then check types. */
        for (i = 0; i < nIntrArgs; i++) {
            a = expr_list_get_n(args, i);
            if (a == NULL) {
                return NULL;    /* error recovery */
            }
            tp = EXPV_TYPE(a);
            if (!(isValidType(tp))) {
                //return NULL;    /* error recovery */
                continue;
            }
            if (compare_intrinsic_arg_type(a, tp,
                                           ((isVarArgs == 0) ?
                                            INTR_ARG_TYPE(ep)[i] :
                                            INTR_ARG_TYPE(ep)[0])) != 0) {
                /* Type mismatch. */
                typeNotMatch = 1;
                break;
            }
        }
        if (typeNotMatch == 1) {
            continue;
        } else {
            found = 1;
            break;
        }
    }

    if (found == 1) {
        /* Yes we found an intrinsic to use. */
        SYMBOL sp = NULL;
        expv symV = NULL;

        /* Then we have to determine return type. */
        if (INTR_RETURN_TYPE(ep) != INTR_TYPE_NONE) {
            tp = get_intrinsic_return_type(ep, args, kindV);
            if (!(isValidType(tp))) {
                //fatal("%s: can't determine return type.", __func__);
                //return NULL;
                tp = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
            }
        } else {
            tp = type_VOID;
        }

        /* Finally find symbol for the intrinsic and make it expv. */
        sp = find_symbol((char *)iName);
        if (sp == NULL) {
            fatal("%s: symbol '%s' is not created??",
                  __func__,
                  INTR_NAME(ep));
            /* not reached */
            return NULL;
        }
        symV = expv_sym_term(F_FUNC, NULL, sp);
        if (symV == NULL) {
            fatal("%s: symbol expv creation failure.", __func__);
            /* not reached */
            return NULL;
        }

        if (IS_VOID(tp)) {
            ftp = intrinsic_subroutine_type();
        } else {
            ftp = intrinsic_function_type(tp);
        }

        /* set external id for functionType's type ID.
         * dont call declare_external_id() */
        extid = new_external_id_for_external_decl(ID_SYM(id), ftp);
        ID_TYPE(id) = ftp;
        PROC_EXT_ID(id) = extid;
        if(TYPE_IS_EXTERNAL(ftp)){
           ID_STORAGE(id) = STG_EXT;
        }else{
           EXT_PROC_CLASS(extid) = EP_INTRINSIC;
        }
        ret = expv_cons(FUNCTION_CALL, tp, symV, args);
    }

    if (ret == NULL && !ignoreTypeMismatch) {
        error_at_node((expr)args,
                      "argument(s) mismatch for an intrinsic '%s()'.",
                      iName);
    }
    return ret;
}



/*
 * Returns like strcmp().
 */
static int
compare_intrinsic_arg_type(expv arg,
    TYPE_DESC tp, INTR_DATA_TYPE iType) {

    BASIC_DATA_TYPE bType;
    int ret = 1;
    int isArray = 0;
    int isCoarray = 0;

    if(IS_GNUMERIC_ALL(tp))
        return 0;

    if (TYPE_IS_COINDEXED(tp)) {
        isCoarray = 1;
    }

    if (IS_ARRAY_TYPE(tp)) {
        while (IS_ARRAY_TYPE(tp)) {
            tp = TYPE_REF(tp);
        }
        isArray = 1;
    }

    bType = TYPE_BASIC_TYPE(tp);

    if (isArray == 1) {
        switch (iType) {
            case INTR_TYPE_ANY_ARRAY: {
                ret = 0;
                break;
            }
            case INTR_TYPE_INT_ARRAY: {
                if (bType == TYPE_INT ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_REAL_ARRAY: {
                if (bType == TYPE_REAL ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_DREAL_ARRAY: {
                if (type_is_possible_dreal(tp) ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_ALL_REAL_ARRAY: {
                if (bType == TYPE_REAL ||
                    bType == TYPE_DREAL ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_ALL_COMPLEX_ARRAY:
            case INTR_TYPE_COMPLEX_ARRAY: {
                if (bType == TYPE_COMPLEX ||
                    bType == TYPE_DCOMPLEX ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_CHAR_ARRAY: {
                if (bType == TYPE_CHAR) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_LOGICAL_ARRAY: {
                if (bType == TYPE_LOGICAL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_NUMERICS_ARRAY: {
                if (bType == TYPE_INT ||
                    bType == TYPE_REAL ||
                    bType == TYPE_DREAL ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_ALL_NUMERICS_ARRAY: {
                if (bType == TYPE_INT ||
                    bType == TYPE_REAL ||
                    bType == TYPE_DREAL ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_COMPLEX ||
                    bType == TYPE_DCOMPLEX ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }

            case INTR_TYPE_ANY_ARRAY_ALLOCATABLE: {
                if (TYPE_IS_ALLOCATABLE(tp)) {
                    ret = 0;
                }
                break;
            }

            default: {
                goto DoCompareBasic;
            }
        }

    } else {
        DoCompareBasic:
        switch (iType) {
            case INTR_TYPE_ANY: {
                ret = 0;
                break;
            }
            case INTR_TYPE_INT: {
                if (bType == TYPE_INT ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_REAL: {
                if (bType == TYPE_REAL ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_DREAL: {
                if (type_is_possible_dreal(tp) ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_ALL_REAL: {
                if (bType == TYPE_REAL ||
                    bType == TYPE_DREAL ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_ALL_COMPLEX:
            case INTR_TYPE_COMPLEX: {
                if (bType == TYPE_COMPLEX ||
                    bType == TYPE_DCOMPLEX ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_DCOMPLEX: {
                if (bType == TYPE_DCOMPLEX ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_CHAR: {
                if (bType == TYPE_CHAR) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_LOGICAL: {
                if (bType == TYPE_LOGICAL) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_NUMERICS: {
                if (bType == TYPE_INT ||
                    bType == TYPE_REAL ||
                    bType == TYPE_DREAL ||
                    bType == TYPE_GNUMERIC) {
                    ret = 0;
                }
                break;
            }
            case INTR_TYPE_ALL_NUMERICS: {
                if (bType == TYPE_INT ||
                    bType == TYPE_REAL ||
                    bType == TYPE_DREAL ||
                    bType == TYPE_GNUMERIC ||
                    bType == TYPE_COMPLEX ||
                    bType == TYPE_DCOMPLEX ||
                    bType == TYPE_GNUMERIC_ALL) {
                    ret = 0;
                }
                break;
            }

            case INTR_TYPE_POINTER:
            case INTR_TYPE_TARGET:
            case INTR_TYPE_ANY_OPTIONAL: {
                ID id;
                TYPE_DESC argtp = NULL;
                switch(EXPV_CODE(arg)) {
                    case(F_VAR): {
                        id = find_ident(EXPV_NAME(arg));
                        if(id == NULL)
                            break;
                        argtp = ID_TYPE(id);
                    } break;
                    case(ARRAY_REF):
                    case(F95_MEMBER_REF): {
                        if(iType != INTR_TYPE_ANY_OPTIONAL)
                            argtp = EXPV_TYPE(arg);
                    } break;
                    default: {
                        break;
                    }
                }
                if(argtp == NULL)
                    break;
                if(iType == INTR_TYPE_POINTER) {
                    if(TYPE_IS_POINTER(argtp) == FALSE)
                        break;
                } else if(iType == INTR_TYPE_TARGET) {
                    if(TYPE_IS_TARGET(argtp) == FALSE)
                        break;
                } else {
                    if(TYPE_IS_OPTIONAL(argtp) == FALSE)
                        break;
                }
                ret = 0;
                break;
            }


            case INTR_TYPE_COARRAY_ANY:
            case INTR_TYPE_COARRAY_INT:
            case INTR_TYPE_COARRAY_REAL:
            case INTR_TYPE_COARRAY_LOGICAL:
            {
                if (!isCoarray)
                    break;
                return compare_intrinsic_arg_type(
                    arg, tp, CONVERT_COARRAY_TO_BASIC(iType));
            }

            default: {
                break;
            }
        }
    }
    return ret;
}

void
generate_shape_expr(TYPE_DESC tp, expr dimSpec) {
    expv dimElm;

    if ((TYPE_REF(tp) == NULL) || !IS_ARRAY_TYPE(tp))
        return;

    dimElm = list3(F_INDEX_RANGE,
                   TYPE_DIM_LOWER(tp), TYPE_DIM_UPPER(tp), TYPE_DIM_STEP(tp));
    set_index_range_type(dimElm);
    generate_shape_expr(TYPE_REF(tp), dimSpec);

    if(TYPE_N_DIM(tp) != 0)
        list_put_last(dimSpec, dimElm);
}


static void
generate_assumed_shape_expr(expr dimSpec, int dim) {
    expv dimElm;

    if (dim == 1)
        return;

    dimElm = list3(F_INDEX_RANGE, NULL, NULL, NULL);
    set_index_range_type(dimElm);

    generate_assumed_shape_expr(dimSpec, dim - 1);

    list_put_last(dimSpec, dimElm);
}


static void
generate_contracted_shape_expr(TYPE_DESC tp, expr dimSpec, int dim) {
    expv dimElm;

    if ((TYPE_REF(tp) == NULL) || !IS_ARRAY_TYPE(tp))
        return;

    dimElm = list3(F_INDEX_RANGE,
                   TYPE_DIM_LOWER(tp), TYPE_DIM_UPPER(tp), TYPE_DIM_STEP(tp));
    set_index_range_type(dimElm);

    if (dim == 0) {
        generate_shape_expr(TYPE_REF(tp), dimSpec);
    }  else {
        generate_contracted_shape_expr(TYPE_REF(tp), dimSpec, dim - 1);
        list_put_last(dimSpec, dimElm);
    }
}


static void
generate_expand_shape_expr(TYPE_DESC tp, expr dimSpec, expr extDim, int dim) {
    expv dimElm;

    //    if ((TYPE_REF(tp) == NULL) || !IS_ARRAY_TYPE(tp))
    //        return;

    //    dimElm = list3(LIST, TYPE_DIM_LOWER(tp), TYPE_DIM_UPPER(tp), TYPE_DIM_STEP(tp));

    if (dim == 0) {
        generate_shape_expr(tp, dimSpec);
        list_put_last(dimSpec, extDim);
    }  else {
        //generate_contracted_shape_expr(TYPE_REF(tp), dimSpec, dim - 1);
        dimElm = list3(F_INDEX_RANGE, TYPE_DIM_LOWER(tp),
                       TYPE_DIM_UPPER(tp), TYPE_DIM_STEP(tp));
        set_index_range_type(dimElm);
        generate_expand_shape_expr(TYPE_REF(tp), dimSpec, extDim, dim - 1);
        list_put_last(dimSpec, dimElm);
    }
}


static void
generate_reverse_dimension_expr(TYPE_DESC tp, expr dimSpec) {

    if (TYPE_REF(tp) != NULL && IS_ARRAY_TYPE(tp)) {

        expr lower = NULL;
        expr upper = NULL;
        expr step = NULL;
        expr dims = NULL;
        int n;

        if (TYPE_DIM_UPPER(tp) != NULL) {
            n = (int)EXPV_INT_VALUE(TYPE_DIM_UPPER(tp));
            upper = make_int_enode(n);
        }
        if (TYPE_DIM_LOWER(tp) != NULL) {
            n = (int)EXPV_INT_VALUE(TYPE_DIM_LOWER(tp));
            lower = make_int_enode(n);
        }
        if (TYPE_DIM_STEP(tp) != NULL) {
            n = (int)EXPV_INT_VALUE(TYPE_DIM_STEP(tp));
            step = make_int_enode(n);
        }

        dims = list3(F_INDEX_RANGE, lower, upper, step);
        set_index_range_type(dims);
        list_put_last(dimSpec, dims);
        generate_reverse_dimension_expr(TYPE_REF(tp), dimSpec);
    }
}


static BASIC_DATA_TYPE
intr_type_to_basic_type(INTR_DATA_TYPE iType) {
    BASIC_DATA_TYPE ret = TYPE_UNKNOWN;

    switch (iType) {
        case INTR_TYPE_NONE: {
            break;
        }

        case INTR_TYPE_INT_DYNAMIC_ARRAY:
        case INTR_TYPE_INT_ARRAY:
        case INTR_TYPE_INT: {
            ret = TYPE_INT;
            break;
        }

        case INTR_TYPE_ALL_REAL_DYNAMIC_ARRAY:
        case INTR_TYPE_REAL_DYNAMIC_ARRAY:
        case INTR_TYPE_ALL_REAL_ARRAY:
        case INTR_TYPE_REAL_ARRAY:
        case INTR_TYPE_ALL_REAL:
        case INTR_TYPE_REAL: {
            ret = TYPE_REAL;
            break;
        }

        case INTR_TYPE_DREAL_DYNAMIC_ARRAY:
        case INTR_TYPE_DREAL_ARRAY:
        case INTR_TYPE_DREAL: {
            ret = TYPE_DREAL;
            break;
        }

        case INTR_TYPE_ALL_COMPLEX_DYNAMIC_ARRAY:
        case INTR_TYPE_COMPLEX_DYNAMIC_ARRAY:
        case INTR_TYPE_ALL_COMPLEX_ARRAY:
        case INTR_TYPE_COMPLEX_ARRAY:
        case INTR_TYPE_ALL_COMPLEX:
        case INTR_TYPE_COMPLEX: {
            ret = TYPE_COMPLEX;
            break;
        }

        case INTR_TYPE_DCOMPLEX_DYNAMIC_ARRAY:
        case INTR_TYPE_DCOMPLEX_ARRAY:
        case INTR_TYPE_DCOMPLEX: {
            ret = TYPE_DCOMPLEX;
            break;
        }

        case INTR_TYPE_CHAR_DYNAMIC_ARRAY:
        case INTR_TYPE_CHAR_ARRAY:
        case INTR_TYPE_CHAR: {
            ret = TYPE_CHAR;
            break;
        }

        case INTR_TYPE_LOGICAL_DYNAMIC_ARRAY:
        case INTR_TYPE_LOGICAL_ARRAY:
        case INTR_TYPE_LOGICAL: {
            ret = TYPE_LOGICAL;
            break;
        }

        case INTR_TYPE_ANY_DYNAMIC_ARRAY:
        case INTR_TYPE_ANY_ARRAY:
        case INTR_TYPE_ANY: {
            /*
             * FIXME: The super type is needed.
             */
            ret = TYPE_UNKNOWN;
            break;
        }

        case INTR_TYPE_NUMERICS_DYNAMIC_ARRAY:
        case INTR_TYPE_NUMERICS_ARRAY:
        case INTR_TYPE_NUMERICS: {
            ret = TYPE_GNUMERIC;
            break;
        }

        case INTR_TYPE_ALL_NUMERICS_DYNAMIC_ARRAY:
        case INTR_TYPE_ALL_NUMERICS_ARRAY:
        case INTR_TYPE_ALL_NUMERICS: {
            ret = TYPE_GNUMERIC_ALL;
            break;
        }

        case INTR_TYPE_POINTER: {
            /*
             * FIXME:
             */
            ret = TYPE_UNKNOWN;
            break;
        }

        case INTR_TYPE_TARGET: {
            /*
             * FIXME:
             */
            ret = TYPE_UNKNOWN;
            break;
        }

        case INTR_TYPE_ANY_ARRAY_ALLOCATABLE: {
            ret = TYPE_UNKNOWN;
            break;
        }

        default: {
            fatal("%s: Unknown INTR_TYPE.", __func__);
            break;
        }
    }

    return ret;
}


static TYPE_DESC
intr_convert_to_dimension_ifneeded(intrinsic_entry *ep,
                                   expv args, TYPE_DESC ret_tp)
{
    TYPE_DESC tp0;

    if(INTR_RETURN_TYPE_SAME_AS(ep) == -6)
        return ret_tp;

    tp0 = EXPV_TYPE(EXPR_ARG1(args));

    if(INTR_IS_ARG_TYPE0_ARRAY(ep) == FALSE &&
        IS_ARRAY_TYPE(tp0)) {
        ret_tp = copy_dimension(tp0, get_bottom_ref_type(ret_tp));
    }

    return ret_tp;
}


static TYPE_DESC
get_intrinsic_return_type(intrinsic_entry *ep, expv args, expv kindV) {
    BASIC_DATA_TYPE bType = TYPE_UNKNOWN;
    TYPE_DESC bTypeDsc = NULL;
    TYPE_DESC ret = NULL;
    expv a = NULL;

    if (INTR_RETURN_TYPE(ep) == INTR_TYPE_NONE) {
        return NULL;
    }

    if (INTR_RETURN_TYPE_SAME_AS(ep) >= 0) {
        /* return type is in args. */
        a = expr_list_get_n(args, INTR_RETURN_TYPE_SAME_AS(ep));
        if (!(isValidTypedExpv(a))) {
            return NULL;
        }
        ret = EXPV_TYPE(a);
    } else {
        switch (INTR_RETURN_TYPE_SAME_AS(ep)) {

            case -1 /* if not dynamic return type,
                        argument is scalar/array and
                        return type is scalar/array */ :
            case -6 /* if not dynamic return type,
                        argument is scalar/array, return
                        return type is scalar */ : {

                if (!(INTR_IS_RETURN_TYPE_DYNAMIC(ep)) &&
                    (INTR_RETURN_TYPE(ep) != INTR_TYPE_ALL_NUMERICS &&
                     INTR_RETURN_TYPE(ep) != INTR_TYPE_NUMERICS)) {
                    bType = intr_type_to_basic_type(INTR_RETURN_TYPE(ep));
                    if (bType == TYPE_UNKNOWN) {
                        fatal("invalid intrinsic return type (case -1/-6).");
                        /* not reached. */
                        return NULL;
                    } else {
                        if (kindV == NULL) {
                            ret = (bType != TYPE_CHAR) ? type_basic(bType) :
                                type_char(1);
                        } else {
                            /*
                             * Don't use BASIC_TYPE_DESC(bType) very
                             * here, since we need to set a kind to
                             * the TYPE_DESC.
                             */
                            ret = type_basic(bType);
                            TYPE_KIND(ret) = kindV;
                        }
                    }
                    ret = intr_convert_to_dimension_ifneeded(
                        ep, args, ret);
                } else {
                    expv shape = list0(LIST);
                    TYPE_DESC tp;

                    switch (INTR_OP(ep)) {

                    case INTR_ALL:
                    case INTR_ANY:
                    case INTR_MAXVAL:
                    case INTR_MINVAL:
                    case INTR_PRODUCT:
                    case INTR_SUM:
                    case INTR_COUNT:
                    {
                        /* intrinsic arguments */
                        expv array, dim;

                        array = expr_list_get_n(args, 0);
                        if (!(isValidTypedExpv(array))) {
                            return NULL;
                        }
                        tp = EXPV_TYPE(array);

                        dim = expr_list_get_n(args, 1);
                        if (!(isValidTypedExpv(dim))) {
                            return NULL;
                        }

                        /* set basic type of array type */
                        switch (INTR_OP(ep)) {
                        case INTR_ALL:
                        case INTR_ANY:
                            bType = TYPE_LOGICAL;
                            break;
                        case INTR_COUNT:
                            bType = TYPE_INT;
                            break;
                        default:
                            bType = get_basic_type(tp);
                            break;
                        }

                        if (kindV == NULL) {
                            bTypeDsc = BASIC_TYPE_DESC(bType);
                        } else {
                            bTypeDsc = type_basic(bType);
                            TYPE_KIND(bTypeDsc) = kindV;
                        }

                        dim = expv_reduce(dim, FALSE);

                        if(EXPV_CODE(dim) == INT_CONSTANT) {
                            int nDim;
                            nDim  = (int)EXPV_INT_VALUE(dim);

                            if(nDim > TYPE_N_DIM(tp) || nDim <= 0) {
                                error("value DIM of intrinsic %s "
                                      "out of range.", INTR_NAME(ep));
                                return NULL;
                            }

                            generate_contracted_shape_expr(
                                tp, shape, TYPE_N_DIM(tp) - nDim);
                        } else {
                            generate_assumed_shape_expr(
                                shape, TYPE_N_DIM(tp) - 1);
                        }
                    }
                    break;

                    case INTR_SPREAD:
                    {
                        /* intrinsic arguments */
                        expv array, dim, ncopies;

                        array = expr_list_get_n(args, 0);
                        if (!(isValidTypedExpv(array))) {
                            return NULL;
                        }
                        dim = expr_list_get_n(args, 1);
                        if (!(isValidTypedExpv(dim))) {
                            return NULL;
                        }
                        ncopies = expr_list_get_n(args, 2);
                        if (!(isValidTypedExpv(ncopies))) {
                            return NULL;
                        }

                        tp = EXPV_TYPE(array);
                        bType = get_basic_type(tp);
                        if (kindV == NULL) {
                            bTypeDsc = BASIC_TYPE_DESC(bType);
                        } else {
                            bTypeDsc = type_basic(bType);
                            TYPE_KIND(bTypeDsc) = kindV;
                        }

                        dim = expv_reduce(dim, FALSE);

                        if(EXPR_CODE(dim) == INT_CONSTANT) {
                            int nDim;
                            nDim  = (int)EXPV_INT_VALUE(dim);

                            if(nDim > (TYPE_N_DIM(tp) + 1) || nDim <= 0) {
                                error("value DIM of intrinsic %s "
                                      "out of range.", INTR_NAME(ep));
                                return NULL;
                            }

                            generate_expand_shape_expr(
                                tp, shape, ncopies, TYPE_N_DIM(tp) + 1 - nDim);
                        } else {
                            generate_assumed_shape_expr(
                                shape, TYPE_N_DIM(tp) - 1);
                        }
                    }
                    break;

                    case INTR_RESHAPE:
                    {
                        /* intrinsic arguments */
                        expv source, arg_shape;

                        source = expr_list_get_n(args, 0);
                        if (!(isValidTypedExpv(source))) {
                            return NULL;
                        }
                        arg_shape = expr_list_get_n(args, 1);
                        if (!(isValidTypedExpv(arg_shape))) {
                            return NULL;
                        }

                        tp = EXPV_TYPE(source);
                        bType = get_basic_type(tp);
                        if (kindV == NULL) {
                            bTypeDsc = BASIC_TYPE_DESC(bType);
                        } else {
                            bTypeDsc = type_basic(bType);
                            TYPE_KIND(bTypeDsc) = kindV;
                        }

                        tp = EXPV_TYPE(arg_shape);
                        if (TYPE_N_DIM(tp) != 1) {
                            error("SHAPE argument of intrinsic "
                                  "RESHAPE is not vector.");
                            return NULL;
                        }

                        /*
                         * We can't determine # of the elements in
                         * this array that represents dimension of the
                         * return type, which is identical to the
                         * reshaped array. In order to express this,
                         * we introduce a special TYPE_DESC, which is
                         * having a flag to specify that the type is
                         * generated by the reshape() intrinsic.
                         */

                        /*
                         * dummy one dimensional assumed array.
                         */
                        generate_assumed_shape_expr(shape, 2);
                        ret = compile_dimensions(bTypeDsc, shape);
                        fix_array_dimensions(ret);
                        TYPE_IS_RESHAPED(ret) = TRUE;

                        return ret;
                    }
                    break;

                    case INTR_MATMUL:
                    {
                        expv m1 = expr_list_get_n(args, 0);
                        expv m2 = expr_list_get_n(args, 1);
                        TYPE_DESC t1 = EXPV_TYPE(m1);
                        TYPE_DESC t2 = EXPV_TYPE(m2);
                        expv s1 = list0(LIST);
                        expv s2 = list0(LIST);

                        /*
                         * FIXME:
                         *  Should we use
                         *  get_binary_numeric_intrinsic_operation_type()
                         *  instead of max_type()? I think so but
                         *  not sure at this moment.
                         */
                        bType = get_basic_type(max_type(t1, t2));

                        if (kindV == NULL) {
                            bTypeDsc = BASIC_TYPE_DESC(bType);
                        } else {
                            bTypeDsc = type_basic(bType);
                            TYPE_KIND(bTypeDsc) = kindV;
                        }

                        generate_shape_expr(t1, s1);
                        generate_shape_expr(t2, s2);

                        if (TYPE_N_DIM(t1) == 2 &&
                            TYPE_N_DIM(t2) == 2) {
                            /*
                             * (n, m) * (m, k) => (n, k).
                             */
                            shape = list2(LIST,
                                          EXPR_ARG1(s1), EXPR_ARG2(s2));
                        } else if (TYPE_N_DIM(t1) == 2 &&
                                   TYPE_N_DIM(t2) == 1) {
                            /*
                             * (n, m) * (m) => (n).
                             */
                            shape = list1(LIST, EXPR_ARG1(s1));
                        } else if (TYPE_N_DIM(t1) == 1 &&
                                   TYPE_N_DIM(t2) == 2) {
                            /*
                             * (m) * (m, k) => (k).
                             */
                            shape = list1(LIST, EXPR_ARG2(s2));
                        } else {
                            error("an invalid dimension combination for "
                                  "matmul(), %d and %d.",
                                  TYPE_N_DIM(t1), TYPE_N_DIM(t2));
                            return NULL;
                        }

                        ret = compile_dimensions(bTypeDsc, shape);
                        fix_array_dimensions(ret);

                        return ret;
                    }
                    break;

                    case INTR_DOT_PRODUCT:
                    {
                        expv m1 = expr_list_get_n(args, 0);
                        expv m2 = expr_list_get_n(args, 1);
                        TYPE_DESC t1 = EXPV_TYPE(m1);
                        TYPE_DESC t2 = EXPV_TYPE(m2);

                        if (TYPE_N_DIM(t1) == 1 &&
                            TYPE_N_DIM(t2) == 1) {
                            TYPE_DESC tp =
                                get_binary_numeric_intrinsic_operation_type(
                                    t1, t2);
                            return array_element_type(tp);
                        } else {
                            error("argument(s) is not a one-dimensional "
                                  "array.");
                            return NULL;
                        }
                    }
                    break;

                    case INTR_PACK:
                    {

                        if (INTR_N_ARGS(ep) == 3){
                            expv v = expr_list_get_n(args, 2);
                            return EXPV_TYPE(v);
                        }
                        else {
                            a = expr_list_get_n(args, 0);
                            if (!(isValidTypedExpv(a))) {
                                return NULL;
                            }

                            bType = get_basic_type(EXPV_TYPE(a));
                            bTypeDsc = BASIC_TYPE_DESC(bType);
                            expr dims = list1(LIST, NULL);
                            ret = compile_dimensions(bTypeDsc, dims);
                            fix_array_dimensions(ret);
                            return ret;
                        }
                    }
                    break;

                    case INTR_UNPACK:
                    {
                        a = expr_list_get_n(args, 0);
                        if (!(isValidTypedExpv(a))) {
                            return NULL;
                        }
                        bType = get_basic_type(EXPV_TYPE(a));
                        bTypeDsc = BASIC_TYPE_DESC(bType);

                        a = expr_list_get_n(args, 1);
                        if (!(isValidTypedExpv(a))) {
                            return NULL;
                        }
                        TYPE_DESC tp = EXPV_TYPE(a);
                        ret = copy_dimension(tp, bTypeDsc);
                        fix_array_dimensions(ret);
                        return ret;
                    }
                    break;

                    case INTR_THIS_IMAGE:
                    case INTR_UCOBOUND:
                    case INTR_LCOBOUND:
                    {
                        /* `THIS_IMAGE(COARRAY)` returns an 1-rank array.
                           Its length is euquals to the corank of COARRAY */
                        int corank;
                        expv dims;
                        a = expr_list_get_n(args, 0);
                        if (!(isValidTypedExpv(a))) {
                            return NULL;
                        }

                        corank = TYPE_CODIMENSION(EXPV_TYPE(a))->corank;

                        dims = list1(LIST,
                                     list2(LIST,
                                           make_int_enode(1),
                                           make_int_enode(corank)));

                        ret = compile_dimensions(type_INT, dims);
                        fix_array_dimensions(ret);
                        return ret;
                    }
                    break;

                    case INTR_MAXLOC:
                    case INTR_MINLOC:
                    {
                        /*
                         * `MAXLOC/MINLOC(ARRAY, DIM)` returns an 1-rank array.
                         * Its shape is [d_1, ..., d_dim-1, d_dim+1, ..., d_n]
                         */

                        expr dim;
                        int array_has_dim = FALSE;
                        int i;
                        TYPE_DESC tp = NULL;
                        TYPE_DESC first = NULL;
                        TYPE_DESC prev = NULL;

                        ret = type_basic(TYPE_INT);

                        TYPE_KIND(ret) = expr_list_get_n(args, 3);

                        a = expr_list_get_n(args, 0);
                        tp = EXPV_TYPE(a);
                        if (TYPE_N_DIM(tp) < 1) {
                            goto return_assumed_shape;
                        }

                        dim = expr_list_get_n(args, 1);
                        dim = expv_reduce(dim, FALSE);

                        if (dim == NULL || EXPV_CODE(dim) != INT_CONSTANT) {
                            goto return_assumed_shape;
                        }

                        for (i = TYPE_N_DIM(tp); IS_ARRAY_TYPE(tp); i--, tp = TYPE_REF(tp)) {
                            TYPE_DESC tp0;

                            if (i == EXPR_INT(dim)) {
                                array_has_dim = TRUE;
                                continue;
                            }

                            tp0 = new_type_desc();
                            TYPE_BASIC_TYPE(tp0)        = TYPE_ARRAY;
                            TYPE_ARRAY_ASSUME_KIND(tp0) = TYPE_ARRAY_ASSUME_KIND(tp);
                            TYPE_DIM_SIZE(tp0)          = TYPE_DIM_SIZE(tp);
                            TYPE_DIM_LOWER(tp0)         = TYPE_DIM_LOWER(tp);
                            TYPE_DIM_UPPER(tp0)         = TYPE_DIM_UPPER(tp);
                            TYPE_DIM_STEP(tp0)          = TYPE_DIM_STEP(tp);

                            if (prev != NULL) {
                                TYPE_REF(prev) = tp0;
                            } else {
                                first = tp0;
                            }
                            prev = tp0;
                        }

                        if (!array_has_dim) {
                            error("not valid dimension index");
                            return NULL;
                        }

                        if (prev != NULL) {
                            TYPE_REF(prev) = ret;
                            fix_array_dimensions(first);
                            return first;
                        }

                  return_assumed_shape:
                        /*
                         * dummy N-1 dimensional assumed array or scala.
                         */

                        if (TYPE_N_DIM(tp) > 1) {
                            generate_assumed_shape_expr(shape, TYPE_N_DIM(tp));
                            ret = compile_dimensions(ret, shape);
                            fix_array_dimensions(ret);
                        }


                        return ret;
                    }
                    break;


                    default:
                    {
                        /* not  reached ! */
                        ret = BASIC_TYPE_DESC(TYPE_GNUMERIC_ALL);
                    }

                    }

                    ret = compile_dimensions(bTypeDsc, shape);
                    fix_array_dimensions(ret);
                }

                break;
            }

            case -2: {
                /*
                 * Returns BASIC_TYPE of the first arg.
                 */
                a = expr_list_get_n(args, 0);
                if (!(isValidTypedExpv(a))) {
                    return NULL;
                }
                bType = get_basic_type(EXPV_TYPE(a));
                if (kindV == NULL) {
                    ret = BASIC_TYPE_DESC(bType);
                } else {
                    ret = type_basic(bType);
                    TYPE_KIND(ret) = kindV;
                }
                break;
            }

            case -3: {
                /*
                 * Returns single dimension array of integer having
                 * elemnets that equals to the first arg's dimension.
                 */
                /*
                 * FIXME:
                 *	No need to check kindV?? I believe we don't, though.
                 */
                bTypeDsc = BASIC_TYPE_DESC(TYPE_INT);
                TYPE_DESC tp = NULL;
                expr dims = NULL;
                int nDims = 0;
                a = expr_list_get_n(args, 0);
                if (!(isValidTypedExpv(a))) {
                    return NULL;
                }

                bTypeDsc = BASIC_TYPE_DESC(TYPE_INT);
                tp = EXPV_TYPE(a);
                nDims = TYPE_N_DIM(tp);
                dims = list1(LIST, make_int_enode(nDims));
                ret = compile_dimensions(bTypeDsc, dims);

                if (INTR_OP(ep) == INTR_MAXLOC ||
                    INTR_OP(ep) == INTR_MINLOC) {
                    TYPE_KIND(ret) = expr_list_get_n(args, 2);
                }

                fix_array_dimensions(ret);

                break;
            }

            case -4:{
                /*
                 * Returns transpose of the first arg (matrix).
                 */
                TYPE_DESC tp = NULL;
                expr dims = list0(LIST);

                a = expr_list_get_n(args, 0);
                if (!(isValidTypedExpv(a))) {
                    return NULL;
                }
                tp = EXPV_TYPE(a);
                bType = get_basic_type(tp);
                if (kindV == NULL) {
                    bTypeDsc = BASIC_TYPE_DESC(bType);
                } else {
                    bTypeDsc = type_basic(bType);
                    TYPE_KIND(bTypeDsc) = kindV;
                }

                if (TYPE_N_DIM(tp) != 2) {
                    error("Dimension is not two.");
                    return NULL;
                }

                generate_reverse_dimension_expr(tp, dims);
                ret = compile_dimensions(bTypeDsc, dims);
                fix_array_dimensions(ret);

                break;
            }

            case -5: {
                /*
                 * -5 : BASIC_TYPE of return type is 'returnType' and
                 * kind of return type is same as first arg.
                 */
                int nDims = 0;
                TYPE_DESC tp = NULL;

                a = expr_list_get_n(args, 0);
                if (!(isValidTypedExpv(a))) {
                    return NULL;
                }

                tp = EXPV_TYPE(a);

                switch (INTR_OP(ep)) {
                    case INTR_AIMAG: case INTR_DIMAG: {
                        bType = get_basic_type(tp);
                        if (bType != TYPE_COMPLEX &&
                            bType != TYPE_DCOMPLEX) {
                            error("argument is not a complex type.");
                            return NULL;
                        }
                        bType = (bType == TYPE_COMPLEX) ?
                            TYPE_REAL : TYPE_DREAL;
                        break;
                    }
                    default: {
                        bType = intr_type_to_basic_type(INTR_RETURN_TYPE(ep));
                        break;
                    }
                }

                if (bType == TYPE_UNKNOWN) {
                    fatal("invalid intrinsic return type (case -5).");
                    /* not reached. */
                    return NULL;
                }
                bTypeDsc = type_basic(bType);
                TYPE_KIND(bTypeDsc) = TYPE_KIND(tp);

                if ((nDims = TYPE_N_DIM(tp)) > 0) {
                    ret = copy_dimension(tp, bTypeDsc);
                    fix_array_dimensions(ret);
                } else {
                    ret = bTypeDsc;
                }

                break;
            }

            case -7: {
                TYPE_DESC lhsTp = new_type_desc();
                TYPE_BASIC_TYPE(lhsTp) = TYPE_LHS;
                TYPE_ATTR_FLAGS(lhsTp) |= TYPE_ATTR_TARGET;
                ret = lhsTp;
                break;
            }

            case -8: {
                bType = intr_type_to_basic_type(INTR_RETURN_TYPE(ep));
                if (bType == TYPE_UNKNOWN) {
                    fatal("invalid intrinsic return type (case -8).");
                    return NULL;
                } else {
                    ret = type_basic(bType);
                }
                TYPE_SET_EXTERNAL(ret);
                break;
            }

            case -9: {
                bType = intr_type_to_basic_type(INTR_RETURN_TYPE(ep));
                ret = type_basic(bType);
                break;
            }

            default: {
                fatal("%s: Unknown return type specification.", __func__);
                break;
            }
        }
    }

    return ret;
}
