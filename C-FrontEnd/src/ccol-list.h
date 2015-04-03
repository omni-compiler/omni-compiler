/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file ccol-list.h
 */

#ifndef _CCOL_LIST_H_
#define _CCOL_LIST_H_

#include "ccol-cmn.h"

/**
 * CCOL_SList : singlely-linked list
 */

/**
 * \brief
 * node of CCOL_SList
 */
typedef struct CCOL_SListNode_ {
    /** next node */
    struct CCOL_SListNode_*     sln_next;
    /** node data */
    CCOL_Data                   sln_data;
} CCOL_SListNode;

/**
 * \brief
 * singlely-linked list
 */
typedef struct CCOL_SList_ {
    /** head node */
    struct CCOL_SListNode_*     sl_head;
    /** number of nodes */
    unsigned int                sl_size;
} CCOL_SList;

extern CCOL_SListNode*  ccol_SListAt(CCOL_SList *l, unsigned int n);
extern CCOL_SListNode*  ccol_SListNextN(CCOL_SListNode *x, unsigned int n);
extern CCOL_SList*      ccol_SListCons(CCOL_SList *l, CCOL_Data d);
extern CCOL_SList*      ccol_SListJoin(CCOL_SList *l, CCOL_SList *m);
extern CCOL_SListNode*  ccol_SListInsertNext(CCOL_SList *l, CCOL_Data d, CCOL_SListNode *prev);
extern CCOL_Data        ccol_SListRemove(CCOL_SList *l, CCOL_SListNode *x);
extern CCOL_Data        ccol_SListRemoveHead(CCOL_SList *l);
extern CCOL_SList*      ccol_SListReverse(CCOL_SList *l);
extern void             ccol_SListClear(CCOL_SList *l);

#define CCOL_SL_ALLOC_NODE              ((CCOL_SListNode*)ccol_Malloc(sizeof(CCOL_SListNode)))
#define CCOL_SL_DATA(x)                 ((x) != NULL ? (x)->sln_data : NULL)
#define CCOL_SL_SET_DATA(x, d)          ((x)->sln_data = (d))
#define CCOL_SL_HEAD(l)                 ((l)->sl_head)
#define CCOL_SL_AT(l, n)                ccol_SListAt((n), (l))
#define CCOL_SL_NEXT(x)                 ((x)->sln_next)
#define CCOL_SL_NEXTN(x, n)             (ccol_SListNextN((x), (n)))
#define CCOL_SL_CONS(l, d)              ccol_SListCons((l), (d))
#define CCOL_SL_JOIN(l, m)              ccol_SListJoin((l), (m))
#define CCOL_SL_INSERT_NEXT(l, d, prev) ccol_SListInsertNext((l), (d), (prev))
#define CCOL_SL_REMOVE(l, x)            ccol_SListRemove((l), (x))
#define CCOL_SL_REMOVE_HEAD(l)          ccol_SListRemoveHead(l)
#define CCOL_SL_REVERSE(l)              ccol_SListReverse(l)
#define CCOL_SL_CLEAR(l)                ccol_SListClear(l)
#define CCOL_SL_SIZE(l)                 ((l)->sl_size)

#define CCOL_SL_FOREACH(ite, l) \
    if((l)->sl_head != (void*)0)\
        for(ite = (l)->sl_head; ite != (void*)0; ite = CCOL_SL_NEXT(ite))

#define CCOL_SL_FOREACH_SAFE(ite, iten, head) \
    if((head) != (void*)0)\
        for(ite = (head), iten = CCOL_SL_NEXT(head); ite != (void*)0;\
            ite = iten, iten = (ite == (void*)0 ? (void*)0 : CCOL_SL_NEXT(ite)))

/**
 * CCOL_DList : doublely-linked list
 */

/**
 * \brief
 * node of CCOL_DListNode
 */
typedef struct CCOL_DListNode_ {
    /** previous node */
    struct CCOL_DListNode_ *dln_prev;
    /** next node */
    struct CCOL_DListNode_ *dln_next;
    /** node data */
    CCOL_Data dln_data;
} CCOL_DListNode;

/**
 * \brief
 * doublely-linked list
 */
typedef struct CCOL_DList_ {
    /** head node */
    struct CCOL_DListNode_ *dl_head;
    /** tail node */
    struct CCOL_DListNode_ *dl_tail;
    /** number of nodes */
    int dl_size;
} CCOL_DList;

extern CCOL_DListNode*  ccol_DListAt(CCOL_DList *l, unsigned int n);
extern CCOL_DListNode*  ccol_DListTailAt(CCOL_DList *l, unsigned int n);
extern CCOL_DListNode*  ccol_DListPrevN(CCOL_DListNode *x, unsigned int n);
extern CCOL_DListNode*  ccol_DListNextN(CCOL_DListNode *x, unsigned int n);
extern CCOL_DList*      ccol_DListCons(CCOL_DList *l, CCOL_Data d);
extern CCOL_DList*      ccol_DListAdd(CCOL_DList *l, CCOL_Data d);
extern CCOL_DList*      ccol_DListJoin(CCOL_DList *l, CCOL_DList *m);
extern CCOL_DListNode*  ccol_DListInsertPrev(CCOL_DList *l, CCOL_Data d, CCOL_DListNode *next);
extern CCOL_DListNode*  ccol_DListInsertNext(CCOL_DList *l, CCOL_Data d, CCOL_DListNode *prev);
extern CCOL_Data        ccol_DListRemove(CCOL_DList *l, CCOL_DListNode *x);
extern CCOL_Data        ccol_DListRemoveHead(CCOL_DList *l);
extern CCOL_Data        ccol_DListRemoveTail(CCOL_DList *l);
extern void             ccol_DListClear(CCOL_DList *l);

#define CCOL_DL_ALLOC_NODE              ((CCOL_DListNode*)ccol_Malloc(sizeof(CCOL_DListNode)))
#define CCOL_DL_DATA(x)                 ((x) == NULL ? NULL : (x)->dln_data)
#define CCOL_DL_SET_DATA(x, d)          ((x)->dln_data = (d))
#define CCOL_DL_HEAD(l)                 ((l)->dl_head)
#define CCOL_DL_TAIL(l)                 ((l)->dl_tail)
#define CCOL_DL_SIZE(l)                 ((l)->dl_size)
#define CCOL_DL_PREV(x)                 ((x) == NULL ? NULL : (x)->dln_prev)
#define CCOL_DL_NEXT(x)                 ((x) == NULL ? NULL : (x)->dln_next)
#define CCOL_DL_AT(l, n)                ccol_DListAt((l), (n))
#define CCOL_DL_TAIL_AT(l, n)           ccol_DListTailAt((l), (n))
#define CCOL_DL_PREVN(x, n)             ccol_DListPrevN((x), (n))
#define CCOL_DL_NEXTN(x, n)             ccol_DListNextN((x), (n))
#define CCOL_DL_CONS(l, x)              ccol_DListCons((l), (x))
#define CCOL_DL_ADD(l, x)               ccol_DListAdd((l), (x))
#define CCOL_DL_JOIN(l, m)              ccol_DListJoin((l), (m))
#define CCOL_DL_INSERT_PREV(l, x, next) ccol_DListInsertPrev((l), (x), (next))
#define CCOL_DL_INSERT_NEXT(l, x, prev) ccol_DListInsertNext((l), (x), (prev))
#define CCOL_DL_REMOVE(l, x)            ccol_DListRemove((l), (x))
#define CCOL_DL_REMOVE_HEAD(l)          ccol_DListRemoveHead(l)
#define CCOL_DL_REMOVE_TAIL(l)          ccol_DListRemoveTail(l)
#define CCOL_DL_CLEAR(l)                ccol_DListClear(l)

#define CCOL_DL_FOREACH(ite, l) \
  for(ite = CCOL_DL_HEAD(l); ite != (void*)0; ite = CCOL_DL_NEXT(ite))

#define CCOL_DL_FOREACH_SAFE(ite, iten, l) \
  for(ite = CCOL_DL_HEAD(l), iten = CCOL_DL_NEXT(ite); ite != (void*)0;	\
      ite = iten, iten = CCOL_DL_NEXT(ite))

#define CCOL_DL_FOREACH_FROM(ite, l) \
  for(; ite != (void*)0; ite = CCOL_DL_NEXT(ite))

#define CCOL_DL_FOREACH_REVERSE(ite, l) \
  for(ite = CCOL_DL_TAIL(l); ite != (void*)0; ite = CCOL_DL_PREV(ite))

#define CCOL_DL_FOREACH_REVERSE_SAFE(ite, iten, l) \
  for(ite = CCOL_DL_TAIL(l), iten = CCOL_DL_PREV(ite); ite != (void*)0;	\
      ite = iten, iten = CCOL_DL_PREV(ite))

#define CCOL_DL_FOREACH_REVERSE_FROM(ite, l) \
  for(; ite != (void*)0; ite = CCOL_DL_PREV(ite))

#endif /* _CCOL_LIST_H_ */

