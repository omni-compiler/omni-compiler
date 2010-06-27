/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file C-expr-mem.c
 */

#include "F-front.h"

static struct list_node *cons_list _ANSI_ARGS_((expr x, struct list_node *l));

char *
xmalloc(size)
     int size;
{
    char *p;
    if((p = (char *)malloc(size)) == NULL)
      fatal("no memory");
    bzero(p,size);
    return(p);
}

/* lineno */
lineno_info *new_line_info(int fid,int ln)
{
    lineno_info *l;
    l = XMALLOC(lineno_info *, sizeof(*l));
    l->file_id = fid;
    l->ln_no = ln;
    return l;
}

#define SYMBOL_HASH_SIZE        0x400
#define SYMBOL_HASH_MASK        (SYMBOL_HASH_SIZE - 1)
static SYMBOL symbol_hash_table[SYMBOL_HASH_SIZE];

SYMBOL
find_symbol(const char *name) {
    SYMBOL sp;
    int hcode;
    const char *cp;

    if (name == NULL || *name == '\0') {
        fatal("%s: About to find symbol NULL or null string.",
              __func__);
        /* not reached */
        return NULL;
    }

    /* hash code, bad ?? */
    hcode = 0;
    for (cp = name; *cp != '\0'; cp++) {
        hcode = (hcode << 1) + *cp;
    }
    hcode &= SYMBOL_HASH_MASK;
    
    for (sp = symbol_hash_table[hcode]; sp != NULL; sp = sp->s_next) {
        if (strcmp(name, sp->s_name) == 0) {
            return sp;
        }
    }

    /* not found, then allocate symbol */
    sp = XMALLOC(SYMBOL, sizeof(*sp));
    bzero(sp, sizeof(*sp));
    sp->s_name = strdup(name);

    /* link it */
    sp->s_next = symbol_hash_table[hcode];
    symbol_hash_table[hcode] = sp;
    return(sp);
}

SYMBOL
find_symbol_without_allocate(const char *name) {
    SYMBOL sp;
    int hcode;
    const char *cp;

    if (name == NULL || *name == '\0') {
        fatal("%s: About to find symbol NULL or null string.",
              __func__);
        /* not reached */
        return NULL;
    }

    /* hash code, bad ?? */
    hcode = 0;
    for (cp = name; *cp != '\0'; cp++) {
        hcode = (hcode << 1) + *cp;
    }
    hcode &= SYMBOL_HASH_MASK;
    
    for (sp = symbol_hash_table[hcode]; sp != NULL; sp = sp->s_next) {
        if (strcmp(name, sp->s_name) == 0) {
            return sp;
        }
    }
    return NULL;
}

#ifndef HAVE_STRDUP
char *
strdup(s)
     char *s;
{
    char *p;
    int len = strlen(s);
    
    p = XMALLOC(char *, len + 1);
    memcpy(p, s, len);
    p[len] = '\0';
    return p;
}
#endif /* !HAVE_STRDUP */

expr
make_enode(code,v)
     enum expr_code code;
     void *v;
{
    expr ep;
    
    ep = XMALLOC(expr,sizeof(*ep));
    ep->e_code = code;
    ep->e_line = current_line;
    ep->v.e_gen = v;
    return(ep);
}

expr make_float_enode(code,d,token)
     enum expr_code code;
     omldouble_t d;
     const char *token;
{
    expr ep;
    
    ep = XMALLOC(expr,sizeof(*ep));
    ep->e_code = code;
    ep->e_line = current_line;
    ep->v.e_lfval = d;
    ep->e_original_token = token;
    return(ep);
}

expr make_int_enode(i)
     omllint_t i;
{
    expr ep;
    
    ep = XMALLOC(expr,sizeof(*ep));
    ep->e_code = INT_CONSTANT;
    ep->e_line = current_line;
    ep->v.e_llval = i;
    return(ep);
}

struct list_node *cons_list(x,l)
     expr x;
     struct list_node *l;
{
    struct list_node *lp;
    
    lp = XMALLOC(struct list_node *,sizeof(struct list_node));
    lp->l_next = l;
    lp->l_item = x;
    lp->l_last = NULL;
    lp->l_array = NULL;
    lp->l_nItems = 1;
    return(lp);
}

expr list0(code)
     enum expr_code code;
{
    return(make_enode(code,NULL));
}

expr list1(code,x1)
     enum expr_code code;
     expr x1;
{
    return(make_enode(code,(void *)cons_list(x1,NULL)));
}

expr list2(code,x1,x2)
     enum expr_code code;
     expr x1,x2;
{
    return(make_enode(code,(void *)cons_list(x1,cons_list(x2,NULL))));
}

expr list3(code,x1,x2,x3)
     enum expr_code code;
     expr x1,x2,x3;
{
    return(make_enode(code,(void *)cons_list(x1,cons_list(x2,cons_list(x3,NULL)))));
}

expr list4(code,x1,x2,x3,x4)
     enum expr_code code;
     expr x1,x2,x3,x4;
{
    return(make_enode(code,(void *)cons_list(x1,cons_list(x2,cons_list(x3,cons_list(x4,NULL))))));
}

expr list5(code,x1,x2,x3,x4,x5)
     enum expr_code code;
     expr x1,x2,x3,x4,x5;
{
    return(make_enode(code,(void *)cons_list(x1,cons_list(x2,cons_list(x3,cons_\
list(x4,cons_list(x5,NULL)))))));
}

expr list_cons(v,w)
     expv v,w;
{
    EXPR_LIST(w) = cons_list(v,EXPR_LIST(w));
    return(w);
}

expr list_put_last(lx,x)
     expr lx;
     expr x;
{
    struct list_node *lp;

    if (lx == NULL) return(lx); /* error recovery in C-parser.y */
    lp = lx->v.e_lp;
    if (lp == NULL) {
      lx->v.e_lp = cons_list(x,NULL);
    } else {
        if (LIST_LAST(lp) != NULL) {
            lp = LIST_LAST(lp);
        } else {
            for (; lp->l_next != NULL; lp = lp->l_next) /* */;
        }
        lp->l_next = cons_list(x,NULL);
        LIST_LAST(lx->v.e_lp) = lp->l_next;
        LIST_N_ITEMS(lx->v.e_lp) += 1;
    }
    return(lx);
}

expr
list_delete_item(lx, x)
     expr lx;
     expr x;
{
    list lp;
    list oLp;
    list first;

    first = oLp = lp = EXPR_LIST(lx);
    FOR_ITEMS_IN_LIST(lp, lx) {
        if (LIST_ITEM(lp) == x) {
            break;
        }
        oLp = lp;
    }
    if (lp != NULL) {
        if (lp == first) {
            struct list_node *l = (struct list_node *)malloc(sizeof(struct list_node));
            memcpy(l, LIST_NEXT(lp), sizeof(struct list_node));
            LIST_NEXT(lp) = NULL;
            EXPR_LIST(lx) = l;
        } else {
            LIST_NEXT(oLp) = LIST_NEXT(lp);
            LIST_NEXT(lp) = NULL;
        }
        lp = lx->v.e_lp;
        LIST_N_ITEMS(lp) -= 1;
        if (LIST_N_ITEMS(lp) < 0) {
            LIST_N_ITEMS(lp) = 0;
        }
        if (LIST_ARRAY(lp) != NULL) {
            free(LIST_ARRAY(lp));
            LIST_ARRAY(lp) = NULL;
        }
    }
    return lx;
}


void
delete_list(expr lx)
{
    list lp;
    list prev_list;

    lp = lx->v.e_lp;

    while(lp != NULL) {
        prev_list = lp;
        lp = LIST_NEXT(lp);
        free(prev_list);
    }

    free(lx);
}

