/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-mem.c
 */

#include "F-front.h"

/* construct expression */
expv
expv_cons(code,tp,left,right)
     enum expr_code code;
     TYPE_DESC tp;
     expv left,right;
{
    expv v;
    struct list_node *l,*r;

    v = XMALLOC(expv,sizeof(*v));
    l = XMALLOC(struct list_node *,sizeof(struct list_node));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_LIST(v) = l;
    l->l_item = left;
    if(right){
        r = XMALLOC(struct list_node *,sizeof(struct list_node));
        r->l_item = right;
        l->l_next = r;
    }
    return(v);
}

expv
expv_user_def_cons(code, tp, id, left, right)
     enum expr_code code;
     TYPE_DESC tp;
     expv id,left,right;
{
    expv v;
    struct list_node *i,*l,*r;

    v = XMALLOC(expv,sizeof(*v));
    i = XMALLOC(struct list_node *,sizeof(struct list_node));
    l = XMALLOC(struct list_node *,sizeof(struct list_node));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_LIST(v) = i;
    i->l_item = id;
    i->l_next = l;
    l->l_item = left;
    if(right){
        r = XMALLOC(struct list_node *,sizeof(struct list_node));
        r->l_item = right;
        l->l_next = r;
    }
    return(v);
}


/* FOR:
 *      IDENT 
 *      STRING_CONSTANT
 *      LABEL_CONSTANT
 */     
expv
expv_sym_term(code,tp,name)
     enum expr_code code;
     TYPE_DESC tp;
     SYMBOL name;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_NAME(v) = name;
    return(v);
}

expv
expv_str_term(code,tp,str)
     enum expr_code code;
     TYPE_DESC tp;
     char *str;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_STR(v) = strdup(str);
    return(v);
}


expv
expv_int_term(code, tp, i)
     enum expr_code code;
     TYPE_DESC tp;
     omllint_t i;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_INT_VALUE(v) = i;
    return(v);
}


expv
expv_any_term(code,p)
     enum expr_code code;
     void *p;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = NULL;
    EXPV_GEN(v) = p;
    return(v);
}

expv
expv_float_term(code,tp,d,token)
     enum expr_code code;
     TYPE_DESC tp;
     omldouble_t d;
     const char *token;
{
    expv v;

    v = XMALLOC(expv,sizeof(*v));
    EXPV_CODE(v) = code;
    EXPV_TYPE(v) = tp;
    EXPV_FLOAT_VALUE(v) = d;
    EXPV_ORIGINAL_TOKEN(v) = token;
    return(v);
}

expv
expv_retype(tp,v)
     TYPE_DESC tp;
     expv v;
{
    expv vv;
    vv = XMALLOC(expv,sizeof(*v));
    bcopy(v,vv,sizeof(*v));
    EXPV_TYPE(vv) = tp;
    return(vv);
}


static void
listToArray(lp)
     list lp;
{
    if (LIST_ARRAY(lp) == NULL) {
        list lq;
        int i = 0;
        int n = 0;
        for (lq = lp; lq != NULL; lq = lq->l_next) {
            n++;
        }
        LIST_N_ITEMS(lp) = n;
        if (LIST_N_ITEMS(lp) > 0) {
            LIST_ARRAY(lp) = (struct list_node **)malloc(sizeof(struct list_node *) * LIST_N_ITEMS(lp));
            for (lq = lp; i < LIST_N_ITEMS(lp); i++, lq = lq->l_next) {
                lp->l_array[i] = lq;
            }
        }
    }
}

expr
expr_list_get_n(x, n)
     expr x;
     int n;
{
    list lp;
    int i;
    for (i = 0, lp = EXPR_LIST(x); (i < n && lp != NULL); i++, lp = LIST_NEXT(lp)) {};
    if (lp == NULL) {
        return NULL;
    }
    return LIST_ITEM(lp);
}

int
expr_list_set_n(x, n, val, doOverride)
     expr x;
     int n;
     expr val;
     int doOverride;
{
    list lp = EXPR_LIST(x);
    
    if (LIST_ARRAY(lp) == NULL) {
        listToArray(lp);
    }
    if (n >= LIST_N_ITEMS(lp)) {
        return FALSE;
    }
    lp = LIST_ARRAY(lp)[n];
    if (doOverride == FALSE) {
        if (LIST_ITEM(lp) != NULL) {
            return FALSE;
        }
    }
    LIST_ITEM(lp) = val;

    return TRUE;
}


int
expr_list_length(expr x) {
    int ret = 0;

    if (EXPR_CODE_IS_LIST(EXPR_CODE(x))) {
        list lp;

        FOR_ITEMS_IN_LIST(lp, x) {
            ret++;
        }
    }

    return ret;
}
