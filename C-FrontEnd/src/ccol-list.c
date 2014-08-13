/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file ccol-list.c
 * implementation of CCOL_SList, CCOL_DList.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ccol-list.h"

/**
 * \brief
 * handler for memory allocation failure in ccol_MallocNoInit, ccol_Malloc.
 */
static void (*s_outOfMemoryHandler)(void) = NULL;

/* -----------------------------------------------------------
 * Common
 */

#ifndef MTRACE
/**
 * \brief
 * call malloc.
 */
void*
ccol_MallocNoInit(unsigned int sz)
{
    void *p = malloc(sz);
    if(p == NULL) {
        if(s_outOfMemoryHandler != NULL)
            s_outOfMemoryHandler();
        return NULL;
    }

#ifdef CCOL_DEBUG_MEM
    printf("@ccol:0x%08x@\n", (unsigned int)p);
#endif

    return p;
}

/**
 * \brief
 * call malloc and intiialize.
 */
void*
ccol_Malloc(unsigned int sz)
{
    void *p = ccol_MallocNoInit(sz);
    if(p == NULL)
        return NULL;
    memset(p, 0, sz);
    return p;
}
#endif

/**
 * \brief
 * set handler for memory allocation failure occuring in
 * ccol_MallocNoInit, ccol_Malloc.
 */
void
ccol_outOfMemoryHandler(void (*handler)(void))
{
    s_outOfMemoryHandler = handler;
}


/**
 * \brief
 * Duplicates a string.
 */
char *ccol_strdup(const char *s, unsigned int maxlen)
{
    if(s == NULL || maxlen == 0)
        return NULL;

    int len = strlen(s);
    if(len > maxlen)
        len = maxlen;

    char *d = ccol_MallocNoInit(len + 1);
    if(d == NULL)
        return NULL; /* handled by s_outOfMemoryHandler */

    memcpy(d, s, len);
    d[len] = 0;
    return d;
}


/**
 * \brief
 * Returns if 's' starts with 'token'.
 */
int ccol_strstarts(const char *s, const char *token)
{
    int len = strlen(token);
    return strncmp(s, token, len);
}


/* -----------------------------------------------------------
 * CCOL_SList
 */

/**
 * \brief
 * Returns next 'n'th node from head of 'l'.
 * if n is 0, returns head.
 */

CCOL_SListNode*
ccol_SListAt(CCOL_SList *l, unsigned int n)
{
    assert(l != NULL);

    return ccol_SListNextN(l->sl_head, n);
}

/**
 * \brief
 * Returns next 'n'th node of 'x'.
 *
 * if n is 0, returns 'x'.
 * 'x' is allowed to be NULL when 'n' is 0.
 */

CCOL_SListNode*
ccol_SListNextN(CCOL_SListNode *x, unsigned int n)
{
    int i;
    CCOL_SListNode *next;

    assert(n >= 0);

    if(n == 0)
        return x;

    assert(x != NULL);
    next = x;

    for(i = 0; i < n; ++i) {
        next = next->sln_next;
        if(next == NULL)
            return NULL;
    }

    return next;
}

/**
 * \brief
 * Add a node of which data is 'd' to the head of 'l' and returns 'l'.
 *
 * 'd' is allowed to be NULL.
 */

CCOL_SList*
ccol_SListCons(CCOL_SList *l, CCOL_Data d)
{
    CCOL_SListNode *x;
    assert(l != NULL);

    x = CCOL_SL_ALLOC_NODE;
    x->sln_next = l->sl_head;
    x->sln_data = d;
    l->sl_head = x;
    ++l->sl_size;

    return l;
}

/**
 * \brief
 * Joins 'l' and 'm', returns head of the joined list.
 *
 * The nodes in 'm' move into 'l', and the length of 'm' will be 0.
 * 'm' are allowed to be NULL.
 */

CCOL_SList*
ccol_SListJoin(CCOL_SList *m, CCOL_SList *l)
{
    CCOL_SListNode *tail;

    assert(l != NULL);

    if(m == NULL || m->sl_head == NULL)
        return l;

    if(l->sl_head == NULL) {
        memcpy(l, m, sizeof(struct CCOL_SList_));
    } else {
        tail = l->sl_head;
        while(tail->sln_next != NULL)
            tail = tail->sln_next;

        tail->sln_next = m->sl_head;
        l->sl_size += m->sl_size;
    }

    memset(m, 0, sizeof(struct CCOL_SList_));

    return l;
}

/**
 * \brief
 * Inserts a node of which data is 'd' after 'prev'.
 *
 * 'd' is allowed to be NULL.
 */

CCOL_SListNode*
ccol_SListInsertNext(CCOL_SList *l, CCOL_Data d, CCOL_SListNode *prev)
{
    CCOL_SListNode *x;
    assert(prev != NULL);

    x = CCOL_SL_ALLOC_NODE;
    x->sln_data = d;
    x->sln_next = prev->sln_next;
    prev->sln_next = x;

    ++l->sl_size;

    return x;
}

/**
 * \brief
 * Removes 'x' in 'l'.
 *
 * 'x' is allowed to be NULL.
 * If succeeds, returns data of 'x' or otherwise NULL.
 * But if the data of 'x' is NULL, the distinction between successful
 * and failure does not adhere.
 */

CCOL_Data
ccol_SListRemove(CCOL_SList *l, CCOL_SListNode *x)
{
    CCOL_SListNode *prev;
    CCOL_Data d;

    assert(l != NULL);

    if(x == NULL || l->sl_head == NULL)
        return NULL;

    prev = l->sl_head;
    while(prev->sln_next != x)
        prev = prev->sln_next;
    if(prev == NULL)
        return NULL;

    prev->sln_next = x->sln_next;
    x->sln_next = NULL;
    d = x->sln_data;
    --l->sl_size;
    ccol_Free(x);

    return d;
}

/**
 * \brief
 * Removes the head of 'l'.
 *
 * If succeeds, returns the data of the head or otherwise NULL.
 * But if the data of the head is NULL, the distinction between successful
 * and failure does not adhere.
 */

CCOL_Data
ccol_SListRemoveHead(CCOL_SList *l)
{
    CCOL_SListNode *x;
    CCOL_Data d;

    assert(l != NULL);

    x = l->sl_head;
    if(x == NULL)
        return NULL;

    l->sl_head = x->sln_next;
    x->sln_next = NULL;
    d = x->sln_data;
    --l->sl_size;
    ccol_Free(x);

    return d;
}

/**
 * \brief
 * Reverses the order of nodes in 'l'.
 */
CCOL_SList*
ccol_SListReverse(CCOL_SList *l)
{
    CCOL_SListNode *x, *n, *p;

    assert(l != NULL);

    if(l->sl_size <= 1)
        return l;

    for(x = l->sl_head, n = x->sln_next, p = NULL; x != NULL;
        p = x, x = n, n = (n == NULL ? NULL : n->sln_next)) {

        x->sln_next = p;
    }

    l->sl_head = p;

    return l;
}

/**
 * \brief
 * Removes all nodes in 'l'.
 */

void
ccol_SListClear(CCOL_SList *l)
{
    CCOL_SListNode *p, *n;

    assert(l != NULL);

    if(l->sl_head != NULL) {
        for(p = l->sl_head, n = p->sln_next; p != NULL;
            p = n, n = (p == NULL ? NULL : p->sln_next))
            ccol_Free(p);
    }
    
    memset(l, 0, sizeof(struct CCOL_SList_));
}

/* -----------------------------------------------------------
 * CCOL_DList
 */

/**
 * \brief
 * Returns next 'n'th node from head of 'l'.
 *
 * if n is 0, returns head.
 */

CCOL_DListNode*
ccol_DListAt(CCOL_DList *l, unsigned int n)
{
    assert(l != NULL);

    return ccol_DListNextN(l->dl_head, n);
}

/**
 * \brief
 *
 * Returns previous 'n'th node from tail of 'l'.
 * if n is 0, returns tail.
 */

CCOL_DListNode*
ccol_DListTailAt(CCOL_DList *l, unsigned int n)
{
    assert(l != NULL);

    return ccol_DListPrevN(l->dl_tail, n);
}

/**
 * \brief
 * Returns previous 'n'th node of 'x'.
 *
 * if n is 0, returns 'x'.
 * 'x' is allowed to be NULL when 'n' is 0.
 */

CCOL_DListNode*
ccol_DListPrevN(CCOL_DListNode *x, unsigned int n)
{
    int i;
    CCOL_DListNode *prev;

    assert(n >= 0);

    if(n == 0)
        return x;

    assert(x != NULL);
    prev = x;

    for(i = 0; i < n; ++i) {
        prev = prev->dln_prev;
        if(prev == NULL)
            return NULL;
    }

    return prev;
}

/**
 * \brief
 * Returns next 'n'th node of 'x'.
 *
 * if n is 0, returns 'x'.
 * 'x' is allowed to be NULL when 'n' is 0.
 */

CCOL_DListNode*
ccol_DListNextN(CCOL_DListNode *x, unsigned int n)
{
    int i;
    CCOL_DListNode *next;

    assert(n >= 0);

    if(n == 0)
        return x;

    assert(x != NULL);
    next = x;
    for(i = 0; i < n; ++i) {
        next = next->dln_next;
        if(next == NULL)
	  return NULL;
    }

    return next;
}

/**
 * \brief
 * Add a node of which data is 'd' to the tail of 'l' and returns 'l'.
 *
 * 'd' is allowed to be NULL.
 */

CCOL_DList*
ccol_DListAdd(CCOL_DList *l, CCOL_Data d)
{
    CCOL_DListNode *x;

    assert(l != NULL);

    x = CCOL_DL_ALLOC_NODE;
    x->dln_data = d;

    if(l->dl_tail != NULL) {
        l->dl_tail->dln_next = x;
        x->dln_prev = l->dl_tail;
    }

    l->dl_tail = x;
    x->dln_next = NULL;

    if(l->dl_head == NULL)
        l->dl_head = x;

    ++l->dl_size;

    return l;
}

/**
 * \brief
 * Add a node of which data is 'd' to the head of 'l' and returns 'l'.
 *
 * 'd' is allowed to be NULL.
 */

CCOL_DList*
ccol_DListCons(CCOL_DList *l, CCOL_Data d)
{
    CCOL_DListNode *x;

    assert(l != NULL);

    x = CCOL_DL_ALLOC_NODE;
    x->dln_data = d;

    if(l->dl_head != NULL) {
        l->dl_head->dln_prev = x;
        x->dln_next = l->dl_head;
    }

    l->dl_head = x;
    x->dln_prev = NULL;

    if(l->dl_tail == NULL)
        l->dl_tail = x;

    ++l->dl_size;

    return l;
}

/**
 * \brief
 * Joins 'l' and 'm', returns head of the joined list.
 *
 * The nodes in 'm' move into 'l', and the length of 'm' will be 0.
 * 'm' is allowed to be NULL.
 */

CCOL_DList*
ccol_DListJoin(CCOL_DList *l, CCOL_DList *m)
{
    assert(l != NULL);
    
    if(m == NULL || m->dl_head == NULL)
        return l;

    if(l->dl_head == NULL) {
        memcpy(l, m, sizeof(struct CCOL_DList_));
    } else {
        l->dl_tail->dln_next = m->dl_head;
        m->dl_head->dln_prev = l->dl_tail;
        l->dl_tail = m->dl_tail;

        l->dl_size += m->dl_size;
    }

    memset(m, 0, sizeof(struct CCOL_DList_));

    return l;
}

/**
 * \brief
 * Inserts a node of which data is 'd' before 'next'.
 *
 * 'd' is allowed to be NULL.
 */

CCOL_DListNode*
ccol_DListInsertPrev(CCOL_DList *l, CCOL_Data d, CCOL_DListNode *next)
{
    CCOL_DListNode *x;

    assert(next != NULL);
    assert(l != NULL);

    x = CCOL_DL_ALLOC_NODE;
    x->dln_data = d;

    x->dln_prev = next->dln_prev;
    x->dln_next = next;
    next->dln_prev = x;

    if(x->dln_prev != NULL)
        x->dln_prev->dln_next = x;

    if(l->dl_head == next)
        l->dl_head = x;

    ++l->dl_size;

    return x;
}

/**
 * \brief
 * Inserts a node of which data is 'd' after 'prev'.
 *
 * 'd' is allowed to be NULL.
 */

CCOL_DListNode*
ccol_DListInsertNext(CCOL_DList *l, CCOL_Data d, CCOL_DListNode *prev)
{
    CCOL_DListNode *x;

    assert(prev != NULL);
    assert(l != NULL);

    x = CCOL_DL_ALLOC_NODE;
    x->dln_data = d;

    x->dln_next = prev->dln_next;
    x->dln_prev = prev;
    prev->dln_next = x;

    if(x->dln_next != NULL)
        x->dln_next->dln_prev = x;

    if(l->dl_tail == prev)
        l->dl_tail = x;

    ++l->dl_size;

    return x;
}

/**
 * \brief
 * Removes 'x' in 'l'.
 *
 * If succeeds, returns the data of 'x' or otherwise NULL.
 * But if the data of 'x' is NULL, the distinction between successful
 * and failure does not adhere.
 */

CCOL_Data
ccol_DListRemove(CCOL_DList *l, CCOL_DListNode *x)
{
    CCOL_DListNode *p;
    CCOL_Data d;

    assert(l != NULL);

    if(x == NULL || l->dl_head == NULL)
        return NULL;

    for(p = l->dl_head; p != NULL; p = p->dln_next)
        if(p == x) break;
    
    if(p == NULL)
        return 0;

    if(x->dln_prev != NULL)
        x->dln_prev->dln_next = x->dln_next;
    if(x->dln_next != NULL)
        x->dln_next->dln_prev = x->dln_prev;
    if(l->dl_head == x)
        l->dl_head = x->dln_next;
    if(l->dl_tail == x)
        l->dl_tail = x->dln_prev;

    x->dln_prev = NULL;
    x->dln_next = NULL;
    d = x->dln_data;
    --l->dl_size;
    ccol_Free(x);

    return d;
}

/**
 * \brief
 * Removes the head of 'l'.
 *
 * If succeeds, returns the data of the head or otherwise NULL.
 * But if the data of the head is NULL, the distinction between successful
 * and failure does not adhere.
 */

CCOL_Data
ccol_DListRemoveHead(CCOL_DList *l)
{
    CCOL_DListNode *x;
    CCOL_Data d;

    if(l == NULL || l->dl_head == NULL)
        return NULL;
    
    x = l->dl_head;
    l->dl_head = x->dln_next;
    if(x->dln_next != NULL)
        x->dln_next->dln_prev = NULL;
    x->dln_next = NULL;

    if(x == l->dl_tail)
        l->dl_tail = NULL;

    d = x->dln_data;
    --l->dl_size;
    ccol_Free(x);

    return d;
}

/**
 * \brief
 * Removes the head of 'l'.
 *
 * If succeeds, returns the data of the tail or otherwise NULL.
 * But if the data of the tail is NULL, the distinction between successful
 * and failure does not adhere.
 */

CCOL_Data
ccol_DListRemoveTail(CCOL_DList *l)
{
    CCOL_DListNode *x;
    CCOL_Data d;

    if(l == NULL || l->dl_tail == NULL)
        return NULL;
    
    x = l->dl_tail;
    l->dl_tail = x->dln_prev;
    if(x->dln_prev != NULL)
        x->dln_prev->dln_next = NULL;
    x->dln_prev = NULL;

    if(x == l->dl_head)
        l->dl_head = NULL;
    
    d = x->dln_data;
    --l->dl_size;
    ccol_Free(x);

    return d;
}

/**
 * \brief
 *
 * Removes all nodes in 'l'.
 */

void
ccol_DListClear(CCOL_DList *l)
{
    CCOL_DListNode *p, *n;

    assert(l != NULL);

    if(l->dl_head != NULL) {
        for(p = l->dl_head, n = p->dln_next; p != NULL;
            p = n, n = (p == NULL ? NULL : p->dln_next))
            ccol_Free(p);
    }
    
    memset(l, 0, sizeof(struct CCOL_DList_));
}

