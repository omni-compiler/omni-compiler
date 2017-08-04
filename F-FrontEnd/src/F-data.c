/**
 * \file F-data.c
 */

#include "F-front.h"

#ifdef DATA_C_IMPL

static char *varInitTypeStr[] = {
    "never",
    "whole array",
    "substring",
    "array elements",
    "equivalence",
    NULL
};

static expv
serializeInitialValue(expr x, expv new)
{
    expv valV;

    if (new == NULL)
        new = list0(LIST);

    switch (EXPR_CODE(x)) {
    case LIST: {
        list lp;
        expr xL;
        
        FOR_ITEMS_IN_LIST(lp, x) {
            xL = LIST_ITEM(lp);
            new = serializeInitialValue(xL, new);
            if(new == NULL)
                return NULL;
        }
        break;
    }

    case F_DUP_DECL: {
        expv vdup = compile_expression(x);
        if(vdup == NULL)
            return NULL;
        expv numV = EXPR_ARG1(vdup);
        valV = EXPR_ARG2(vdup);
        int i, num;

        num = EXPV_INT_VALUE(numV);
        assert(num > 0);

        for(i = 0; i < num; i++)
            new = list_put_last(new, valV);
        break;
    }

    default:
        valV = expr_constant_value(x);
        if(valV == NULL) {
            error("data value not constant.");
            return NULL;
        }
        new = list_put_last(new, valV);
        break;
    }

    return new;
}


static expv
findArrayRef(expv spec, expv new)
{
    list lp;
    expv v;

    if (new == NULL) {
        new = list0(LIST);
    }

    FOR_ITEMS_IN_LIST(lp, spec) {
        v = LIST_ITEM(lp);
        if (v == NULL) {
            continue;
        }

        switch (EXPR_CODE(v)) {
            
            case F_ARRAY_REF: {
                new = list_put_last(new, v);
                break;
            }

            default: {
                if (EXPR_CODE_IS_TERMINAL(EXPR_CODE(v)) != TRUE) {
                    new = findArrayRef(v, new);
                }
                break;
            }
        }
    }
    
    return new;
}


static expv
getVariableSpec(expv v, ID *idPtr)
{
    ID id;
    expv ret = NULL;
    expv iList = NULL;
    expv aList = NULL;

    if (idPtr != NULL) {
        *idPtr = NULL;
    }

    if (expr_is_variable(v, TRUE, &id) == FALSE) {
        if (id != NULL) {
            if (ID_CLASS(id) == CL_PROC) {
                if (PROC_CLASS(id) == P_THISPROC) {
                    error("can't give data to '%s'.", ID_NAME(id));
                    return NULL;
                }
            }
            error("'%s' is not a variable.", ID_NAME(id));
        } else {
            error("not a variable.");
        }
        return NULL;
    }

    if (idPtr != NULL) {
        *idPtr = id;
    }

    if (EXPR_CODE(v) == F_ARRAY_REF && IS_CHAR(ID_TYPE(id))) {
        return list1(F_SUBSTR_REF, EXPR_ARG1(v));
    }

    ret = expr_array_spec_list(v, &id);

    if(IS_CHAR(ID_TYPE(id))) {
        return NULL;
    }

    if (VAR_INIT_LIST(id) == NULL) {
        if (IS_ARRAY_TYPE(ID_TYPE(id))) {
            list lp;
            int numElem = 1;
            int i;

            if (ret == NULL) {
                fatal("'%s' is array but can't determine array spec??", ID_NAME(id));
            }
            FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(ret)) {
                numElem *= EXPV_INT_VALUE(EXPR_ARG1(LIST_ITEM(lp)));
            }

            iList = list1(LIST, expv_int_term(INT_CONSTANT, type_INT, numElem));
            aList = list0(LIST);
            for (i = 0; i < numElem; i++) {
                list_put_last(aList, NULL);
            }
            list_put_last(iList, aList);
        } else if(IS_CHAR(ID_TYPE(id)) == FALSE) {
            iList = list1(LIST, expv_int_term(INT_CONSTANT, type_INT, 1));
            aList = list0(LIST);
            list_put_last(aList, NULL);
            list_put_last(iList, aList);
        }
        VAR_INIT_LIST(id) = iList;
    } else {
        ret = VAR_ARRAY_INFO(id);
    }
    
    return ret;
}


static expv
genImpliedDo(expv loopSpec, int dim, int lvl, expv refSpec)
{
    expv new = NULL;

    if (lvl < dim) {
        expv cV;
        new = list0(F_IMPLIED_DO);
        new = list_put_last(new, expr_list_get_n(loopSpec, lvl));
        cV = genImpliedDo(loopSpec, dim, lvl + 1, refSpec);
        if (cV != NULL) {
            new = list_put_last(new, list1(LIST, cV));
        }
    } else {
        new = refSpec;
    }
    return new;
}

static expv
serializeVariable(expv v, expv new)
{
    ID id;

    if (new == NULL) {
        new = list0(LIST);
    }
    switch (EXPR_CODE(v)) {
        case F_IMPLIED_DO: {
            list lp;
            expr x;
            expv refList = findArrayRef(v, (expv)NULL);
            
            FOR_ITEMS_IN_LIST(lp, refList) {
                x = LIST_ITEM(lp);

                getVariableSpec(x, &id);
                if (id == NULL) {
                    return NULL;
                }
                if (VAR_INIT_TYPE(id) != VAR_INIT_NEVER &&
                    VAR_INIT_TYPE(id) != VAR_INIT_PARTIAL) {
                    error("\"%s\" is already initialized as %s.",
                          SYM_NAME(ID_SYM(id)),
                          varInitTypeStr[VAR_INIT_TYPE(id)]);
                    return NULL;
                }
                VAR_INIT_TYPE(id) = VAR_INIT_PARTIAL;
            }

            new = ExpandImpliedDoInDATA(v, new);
            break;
        }

        case LIST: {
            list lp;
            expr x;

            FOR_ITEMS_IN_LIST(lp, v) {
                x = LIST_ITEM(lp);
                new = serializeVariable(x, new);
                if (new == NULL) {
                    return NULL;
                }
            }
            break;
        }

        case IDENT: {
            expv asV = getVariableSpec(v, &id);
            if (id == NULL) {
                return NULL;
            }
            if (VAR_INIT_TYPE(id) != VAR_INIT_NEVER) {
                error("\"%s\" is already initialized as %s.",
                      SYM_NAME(ID_SYM(id)),
                      varInitTypeStr[VAR_INIT_TYPE(id)]);
                return NULL;
            }
            VAR_INIT_TYPE(id) = VAR_INIT_WHOLE;

            if (asV != NULL) {
                /*
                 * Initialize whole array.
                 */
                int nDim = (int)EXPV_INT_VALUE(EXPR_ARG1(asV));
                int i;
                expv ll;
                SYMBOL sp;
                char varName[64];
                expv var;
                expv vL;
                expv refSpec;
                expv loopSpec = list0(LIST);
                expv doSpec;
                ID dumId;

                for (i = 0; i < nDim; i++) {
                    ll = expr_list_get_n(EXPR_ARG2(asV), nDim - i -1);
                    if (ll == NULL) {
                        error("can't initialize %s.", SYM_NAME(ID_SYM(id)));
                        return NULL;
                    }
                    sprintf(varName, "__ImpDoIdx_%c", 'i' + nDim - i -1);
                    sp = find_symbol(varName);
                    dumId = declare_ident(sp, CL_VAR);
                    declare_id_type(dumId, type_INT);
                    declare_variable(dumId);
                    VAR_IS_IMPLIED_DO_DUMMY(dumId) = TRUE;
                    var = list0(LIST);
                    var = list_put_last(var, make_enode(IDENT, (void *)sp));
                    var = list_put_last(var, EXPR_ARG2(ll));
                    var = list_put_last(var, EXPR_ARG3(ll));
                    var = list_put_last(var, NULL);
                    loopSpec = list_put_last(loopSpec, var);
                }
                vL = list0(LIST);

                for (i = 0; i < nDim; i++) {
                    ll = expr_list_get_n(loopSpec, nDim - i - 1);
                    vL = list_put_last(vL, EXPR_ARG1(ll));
                }
                refSpec = list2(F_ARRAY_REF, v, vL);

                doSpec = genImpliedDo(loopSpec, nDim, 0, refSpec);
                new = ExpandImpliedDoInDATA(doSpec, new);
            } else {
                new = list_put_last(new, v);
            }
            break;
        }

        case F_ARRAY_REF: {
            expv idxV = NULL;
            expv newV;
            expv asV = getVariableSpec(v, &id);
            if (id == NULL) {
                return NULL;
            }
            if (asV == NULL) {
                if (VAR_IS_UNCOMPILED_ARRAY(id) == FALSE) {
                    error("'%s' is not an array.", ID_NAME(id));
                    return NULL;
                } else {
                    goto DoPut;
                } 
            }
            if(EXPV_CODE(asV) == F_SUBSTR_REF) {
                new = list_put_last(new, asV);
                VAR_INIT_TYPE(id) = VAR_INIT_SUBSTR;
                break;
            }

            if (VAR_INIT_TYPE(id) != VAR_INIT_NEVER &&
                VAR_INIT_TYPE(id) != VAR_INIT_PARTIAL) {
                error("\"%s\" is already initialized as %s.",
                      SYM_NAME(ID_SYM(id)),
                      varInitTypeStr[VAR_INIT_TYPE(id)]);
                return NULL;
            }
            VAR_INIT_TYPE(id) = VAR_INIT_PARTIAL;

            DoPut:
            idxV = expr_array_index(v);
            if (idxV == NULL) {
                return NULL;
            }
            newV = list2(F_ARRAY_REF, EXPR_ARG1(v), idxV);
            new = list_put_last(new, newV);
            break;
        }

        case F_SUBSTR_REF: {
            expv vTmp = compile_expression(v);
            ID id;
            expv newV;

            if (vTmp == NULL ||
                !IS_CHAR(EXPV_TYPE(vTmp))) {
                fatal("sub string not char???");
            }

            getVariableSpec(EXPR_ARG1(v), &id);
            if (id == NULL) {
                return NULL;
            }
            if (VAR_INIT_TYPE(id) != VAR_INIT_NEVER &&
                VAR_INIT_TYPE(id) != VAR_INIT_SUBSTR) {
                error("\"%s\" is already initialized as %s.",
                      SYM_NAME(ID_SYM(id)),
                      varInitTypeStr[VAR_INIT_TYPE(id)]);
                return NULL;
            }
            VAR_INIT_TYPE(id) = VAR_INIT_SUBSTR;
            newV = expv_sym_term(IDENT, ID_TYPE(id), EXPR_SYM(EXPR_ARG1(v)));
            new = list_put_last(new, newV);
            break;
        }

        default: {
            error("invalid type in DATA statement.");
            return NULL;
        }
    }

    return new;
}

static char idxStrBuf[4096];

static char *
idxToStr(expv v)
{
    list lp;
    char buf[sizeof(idxStrBuf)];
    int len;
    memset(idxStrBuf, 0, sizeof(idxStrBuf));

    FOR_ITEMS_IN_LIST(lp, v) {
        sprintf(buf, "%lld,", EXPV_INT_VALUE(LIST_ITEM(lp)));
        strcat(idxStrBuf, buf);
    }
    len = strlen(idxStrBuf);
    idxStrBuf[len - 1] = '\0';

    return idxStrBuf;
}


/*
 * vrV : variable
 * vlV : initial value
 */
static int
setInitialValue(expv vrV, expv vlV)
{
    ID id;
    expv aSpec = getVariableSpec(vrV, &id);
    expv tryAssign = NULL;
    BASIC_DATA_TYPE rHTyp;
    BASIC_DATA_TYPE lHTyp;
    expv rHV = NULL;
    expv lHV = NULL;

    if (id == NULL) {
        return FALSE;
    }
    if (ID_CLASS(id) == CL_PROC) {
        if (PROC_CLASS(id) == P_THISPROC) {
            error("can't give data to '%s'.", ID_NAME(id));
            return FALSE;
        }
    }

    if (ID_TYPE(id) == NULL) {
        if (ID_CLASS(id) == CL_VAR &&
            VAR_IS_UNCOMPILED(id) == TRUE) {
            /*
             * We can't determine the type yet, so at here, at this moment,
             * assume that having an ID is the proof of validness.
             */
            return TRUE;
        }
    }

    if (ID_STORAGE(id) != STG_SAVE &&
        ID_STORAGE(id) != STG_COMMON &&
        ID_STORAGE(id) != STG_COMEQ) {
        ID_STORAGE(id) = STG_SAVE;
        TYPE_SET_SAVE(id);
        ID_IS_DECLARED(id) = FALSE;
    }

    if (VAR_INIT_TYPE(id) == VAR_INIT_SUBSTR) {
        if (IS_CHAR(EXPV_TYPE(vlV)) == FALSE) {
            error("initializer type is not a character.");
            return FALSE;
        }
        return TRUE;
    }

    rHV = compile_expression(vlV);
    if (rHV == NULL) {
        return FALSE;
    }

    lHV = compile_expression(vrV);
    if (lHV == NULL) {
        return FALSE;
    }
    rHTyp = getBasicType(EXPV_TYPE(rHV));
    lHTyp = getBasicType(EXPV_TYPE(lHV));

    tryAssign = expv_assignment(lHV, vlV);
    if (tryAssign == NULL) {
        return FALSE;
    }

    if (aSpec != NULL) {
        int off = compute_element_offset(aSpec, EXPR_ARG2(vrV));
        if (off < 0) {
            return FALSE;
        }
        if (off >= EXPV_INT_VALUE(EXPR_ARG1(VAR_INIT_LIST(id)))) {
            error("element index range error, %s(%s) -> %d >= %d.",
                  ID_NAME(id), idxToStr(EXPR_ARG2(vrV)),
                  off, EXPV_INT_VALUE(EXPR_ARG1(VAR_INIT_LIST(id))));
            return FALSE;
        }
        if (expr_list_set_n(EXPR_ARG2(VAR_INIT_LIST(id)), off, vlV, FALSE) != TRUE) {
            error("%s(%s) is already initialized.",
                  ID_NAME(id), idxToStr(EXPR_ARG2(vrV)));
            return FALSE;
        }
    } else {
        expv initList = VAR_INIT_LIST(id);
        expv val2 = initList ? EXPR_ARG2(initList) : NULL;

        if (val2) {
            if(LIST_ITEM(EXPV_LIST(val2))) {
                error("%s is already initialized.", ID_NAME(id));
                return FALSE;
            }
            LIST_ITEM(EXPV_LIST(val2)) = vlV;
        }
    }

    return TRUE;
}


static int
isValidDataDecl(expr x)
{
    int valNum = 0;
    int varNum = 0;
    int num = 0;
    int i;
    list lp;
    expv vrV, vlV;
    list vrLp, vlLp;

    expv varList = serializeVariable(EXPR_ARG1(x), (expv)NULL);
    expv valList = serializeInitialValue(EXPR_ARG2(x), (expv)NULL);

    if (varList == NULL || valList == NULL)
        return FALSE;

    FOR_ITEMS_IN_LIST(lp, varList) {
        varNum++;
    }
    FOR_ITEMS_IN_LIST(lp, valList) {
        valNum++;
    }

    num = (valNum > varNum) ? varNum : valNum;

    if (varNum != valNum) {
        warning("variable number (%d) is differ to initializer number (%d).",
                varNum, valNum);
    }

    for(i = 0, vrLp = EXPR_LIST(varList), vlLp = EXPR_LIST(valList);
         i < num;
         i++, vrLp = LIST_NEXT(vrLp), vlLp = LIST_NEXT(vlLp)) {
        vrV = LIST_ITEM(vrLp);
        vlV = LIST_ITEM(vlLp);
        if (setInitialValue(vrV, vlV) == FALSE) {
            return FALSE;
        }
    }

    return TRUE;
}


#endif /* DATA_C_IMPL */


/* static void */
/* fixIdTypesInDataDecl(expr vList) */
/* { */
/*     list lp; */
/*     expr x; */
/*     expr iX; */
/*     ID id; */

/*     FOR_ITEMS_IN_LIST(lp, vList) { */
/*         iX = NULL; */
/*         x = LIST_ITEM(lp); */
/*         switch (EXPR_CODE(x)) { */
/*             case IDENT: { */
/*                 iX = x; */
/*                 break; */
/*             } */
/*             case F_ARRAY_REF: { */
/*                 iX = EXPR_ARG1(x); */
/*                 break; */
/*             } */
/*             default: { */
/*                 break; */
/*             } */
/*         } */
/*         if (iX == NULL || EXPR_CODE(iX) != IDENT) { */
/*             continue; */
/*         } */
/*         id = find_ident(EXPR_SYM(iX)); */
/*         fix_type(id); */
/*     } */
/* } */

static void
fixIdTypesInDataDecl(expr x)
{
    expr iX = NULL;
    ID id;

    switch (EXPR_CODE(x)) {
    case IDENT: {
      iX = x;
      break;
    }
    case F_ARRAY_REF: {
      iX = EXPR_ARG1(x);
      break;
    }
    default: {
      break;
    }
    }

    if (iX == NULL || EXPR_CODE(iX) != IDENT) {
      return;
    }

    id = find_ident(EXPR_SYM(iX));
    if (!id) id = declare_ident(EXPR_SYM(iX), CL_VAR);
    fix_type(id);
}


static int
compile_DATA_decl_or_statement0(expv varAndVal, int is_declaration)
{
    expv v, vVars, vVals, vLp;
    list lp;

    vVars = list0(LIST); 
    vVals = list0(LIST); 

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG1(varAndVal)) {
        vLp = LIST_ITEM(lp);

        /*
         * Elimicate statement like:
         *
         *	data a() / ... /
         */
        if (EXPR_CODE(vLp) == F_ARRAY_REF &&
            EXPR_ARG2(vLp) == NULL) {
            error_at_node(vLp, "Invalid array reference.");
            continue;
        }

        v = compile_expression(vLp);
        if (v == NULL) {
            return FALSE;
        }
        list_put_last(vVars, v);
    }

    FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(varAndVal)) {
        vLp = LIST_ITEM(lp);
        v = compile_expression(vLp);
        if (v == NULL) {
            return FALSE;
        }
        list_put_last(vVals, v);
    }

    if(is_declaration){
        v = list2(F_DATA_DECL, vVars, vVals);
    } else {
        v = list2(F_DATA_STATEMENT, vVars, vVals);
    }
    EXPR_LINE(v) = EXPR_LINE(varAndVal);

    output_statement(v);

    return TRUE;
}


void
compile_DATA_decl_or_statement(expr x, int is_declaration)
{
    list lp;
    list lp1;
    expr lx;
    expv varAndVal;

    /*
     * x: (LIST (LIST (IDENT+) LIST (VALUES)))
     * x => (LIST ((LIST m n) (LIST 5 6)) ((LIST p) (LIST 7)))
     */

    FOR_ITEMS_IN_LIST(lp, x) {
        varAndVal = LIST_ITEM(lp);
	/* varAndVal => ((LIST m n) (LIST 5 6)) */

	FOR_ITEMS_IN_LIST(lp1, EXPR_ARG1(varAndVal)) {
            lx = LIST_ITEM(lp1);
            /* lx => m */
            fixIdTypesInDataDecl(lx);
        }

#ifdef DATA_C_IMPL
        if (isValidDataDecl(varAndVal) == FALSE) {
            error_at_node(x, "invalid data statement");
            return;
        }
#endif /* DATA_C_IMPL */

        if (compile_DATA_decl_or_statement0(varAndVal, is_declaration) == FALSE) 
        {
            return;
        }
    }
}
