/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-datasub.c
 */

#include "F-front.h"


typedef struct {
    char        *varName;
    omllint_t   val;
    int         inited;
} variableEntry;

static variableEntry *varTbl = NULL;
static int nVarTbl = 0;


static int
varComp(const void *v1, const void *v2)
{
    return strcmp(((variableEntry *)v1)->varName,
                  ((variableEntry *)v2)->varName);
}


static void
InitializeVariableTable()
{
    if (nVarTbl > 0 && varTbl != NULL) {
        int i;
        for (i = 0; i < nVarTbl; i++) {
            free(varTbl[i].varName);
        }
        free(varTbl);
    }
    varTbl = NULL;
    nVarTbl = 0;
}


static void
AddVariable(char *var)
{
    if (varTbl == NULL) {
        varTbl = (variableEntry *)malloc(sizeof(variableEntry) * 1);
        varTbl[0].varName = strdup(var);
        varTbl[0].val = 0;
        varTbl[0].inited = FALSE;
        nVarTbl = 1;
    } else {
        variableEntry *t;

        if (nVarTbl < 2) {
            t = NULL;
        } else {
            variableEntry key;
            key.varName = var;
            t = (variableEntry *)bsearch((void *)&key,
                                         (void *)varTbl, 
                                         nVarTbl, sizeof(variableEntry),
                                         varComp);
        }
        if (t == NULL) {
            varTbl = (variableEntry *)realloc(varTbl,
                                              sizeof(variableEntry) * (nVarTbl + 1));
            varTbl[nVarTbl].varName = strdup(var);
            varTbl[nVarTbl].val = 0;
            varTbl[nVarTbl].inited = FALSE;
            nVarTbl++;
            qsort((void *)varTbl, nVarTbl, sizeof(variableEntry), varComp);
        }
    }
}


static void
SetVariableValue(char *var, omllint_t val)
{
    variableEntry key;
    variableEntry *t;

    key.varName = var;
    t = (variableEntry *)bsearch((void *)&key,
                                 (void *)varTbl,
                                 nVarTbl, sizeof(variableEntry),
                                 varComp);
    
    if (t != NULL) {
        t->val = val;
        t->inited = TRUE;
    } else {
        fatal("'%s' is not in variable table.", var);
    }
}


static omllint_t
GetVariableValue(char *var)
{
    variableEntry key;
    variableEntry *t;

    key.varName = var;
    t = (variableEntry *)bsearch((void *)&key,
                                 (void *)varTbl,
                                 nVarTbl, sizeof(variableEntry),
                                 varComp);
    
    if (t != NULL) {
        if (t->inited == TRUE) {
            return t->val;
        } else {
            fatal("'%s' is not initialized.", var);
        }
    } else {
        fatal("'%s' is not in variable table.", var);
    }
    return 0;
}


static expv
findIdent(expv spec, expv new)
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
            
            case IDENT: {
                list lq;
                expv vv;
                int found = FALSE;
                FOR_ITEMS_IN_LIST(lq, new) {
                    vv = LIST_ITEM(lq);
                    if (EXPR_SYM(vv) == EXPR_SYM(v)) {
                        found = TRUE;
                        break;
                    }
                }
                if (found == FALSE) {
                    new = list_put_last(new, v);
                }
                break;
            }
            
            default: {
                if (EXPR_CODE_IS_TERMINAL(EXPR_CODE(v)) != TRUE) {
                    new = findIdent(v, new);
                }
                break;
            }
        }
    }
    
    return new;
}


omllint_t
getExprValue(expv v)
{
    omllint_t ret = 0;

    switch (EXPR_CODE(v)) {

        case IDENT: {
            ret = GetVariableValue(SYM_NAME(EXPR_SYM(v)));
            break;
        }

        case INT_CONSTANT: {
            ret = EXPV_INT_VALUE(v);
            break;
        }

        case F_UNARY_MINUS_EXPR:
        case UNARY_MINUS_EXPR: {
            ret = -getExprValue(v);
            break;
        }

        case F_PLUS_EXPR:
        case PLUS_EXPR: {
            ret = getExprValue(EXPR_ARG1(v)) + getExprValue(EXPR_ARG2(v));
            break;
        }

        case F_MINUS_EXPR:
        case MINUS_EXPR: {
            ret = getExprValue(EXPR_ARG1(v)) - getExprValue(EXPR_ARG2(v));
            break;
        }
        
        case F_MUL_EXPR:
        case MUL_EXPR: {
            ret = getExprValue(EXPR_ARG1(v)) * getExprValue(EXPR_ARG2(v));
            break;
        }

        case F_DIV_EXPR:
        case DIV_EXPR: {
            ret = getExprValue(EXPR_ARG1(v)) / getExprValue(EXPR_ARG2(v));
            break;
        }

        case F_POWER_EXPR:
        case POWER_EXPR: {
            ret = power_ii(getExprValue(EXPR_ARG1(v)),
                           getExprValue(EXPR_ARG2(v)));
            break;
        }

        default: {
            error("only integer expression is allowed in implied DO in DATA statement.");
            return 0;
        }
    }

    return ret;
}


static int
InterpretImpliedDo(expv doSpec, expv new)
{
    expv loopVar;
    char *varName;
    
    int thisLoop;

    int loopInit;
    expv loopInitV;
    int loopEnd;
    expv loopEndV;
    int loopIncr;
    expv loopIncrV;

    expv v;
    list lp;
    list lq;

    if (EXPR_CODE(doSpec) != F_IMPLIED_DO) {
        return FALSE;
    }

    if (new == NULL) {
        new = list0(LIST);
    }

    v = EXPR_ARG1(doSpec);
    loopVar = EXPR_ARG1(v);
    varName = SYM_NAME(EXPV_NAME(loopVar));

    loopInitV = EXPR_ARG2(v);
    loopInit = getExprValue(loopInitV);
    loopEndV = EXPR_ARG3(v);
    loopEnd = getExprValue(loopEndV);
    loopIncrV = EXPR_ARG4(v);
    if (loopIncrV == NULL) {
        loopIncr = 1;
    } else {
        loopIncr = getExprValue(loopIncrV);
    }

    thisLoop = loopInit;
    SetVariableValue(varName, (int)loopInit);

    for (;thisLoop <= loopEnd;
         thisLoop += loopIncr, SetVariableValue(varName, thisLoop)) {
    
        FOR_ITEMS_IN_LIST(lp, EXPR_ARG2(doSpec)) {
            v = LIST_ITEM(lp);

            switch (EXPR_CODE(v)) {
            case F_IMPLIED_DO: {
                if (InterpretImpliedDo(v, new) == FALSE) {
                    return FALSE;
                }
                break;
            }

            case F_ARRAY_REF: {
                expv aRefV;
                SYMBOL sym;
                expv idxV = list0(LIST);
                FOR_ITEMS_IN_LIST(lq, EXPR_ARG2(v)) {
                    expv x = LIST_ITEM(lq);
                    omllint_t idx = getExprValue(x);
                    idxV = list_put_last(idxV,
                        expv_int_term(INT_CONSTANT, type_INT, idx));
                }
                sym = find_symbol(SYM_NAME(EXPR_SYM(EXPR_ARG1(v))));
                aRefV = list2(F_ARRAY_REF, make_enode(IDENT, sym), idxV);
                new = list_put_last(new, aRefV);
                break;
            }

            default:
                error("invalid expression in implied DO in DATA statement.");
                return FALSE;
            }
        }
    }

    return TRUE;
}


expv
ExpandImpliedDoInDATA(expv spec, expv new)
{
    list lp;
    expv v;
    expv idents = findIdent(spec, (expv)NULL);

    if (new == NULL) {
        new = list0(LIST);
    }

    InitializeVariableTable();
    FOR_ITEMS_IN_LIST(lp, idents) {
        v = LIST_ITEM(lp);
        AddVariable(SYM_NAME(EXPR_SYM(v)));
    }

    if (InterpretImpliedDo(spec, new) == FALSE) {
        return NULL;
    }

    return new;
}

