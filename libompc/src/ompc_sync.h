/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_sync.h
 */

#include <abt.h>

#define MAX_SPIN_COUNT 0
#define OMPC_SIGNAL(statement, condvar, mutex) \
    ABT_mutex_lock(mutex); \
    statement; \
    ABT_cond_signal(condvar); \
    ABT_mutex_unlock(mutex)
#define OMPC_BROADCAST(statement, condvar, mutex) \
    ABT_mutex_lock(mutex); \
    statement; \
    ABT_cond_broadcast(condvar); \
    ABT_mutex_unlock(mutex)
#define OMPC_WAIT_UNTIL(condexpr, condvar, mutex) \
    for (int c = 0; c <= MAX_SPIN_COUNT; c++) { \
        if (condexpr) { \
            break; \
        } \
        if (c == MAX_SPIN_COUNT) { \
            ABT_mutex_lock(mutex); \
            while (!(condexpr)) { \
                ABT_cond_wait(condvar, mutex); \
            } \
            ABT_mutex_unlock(mutex); \
        } \
    }

struct ompc_tree_barrier_node
{
    int num_children;
    int count;
    _Bool volatile sense;
    ABT_mutex mutex;
    ABT_cond cond;
} __attribute__((aligned(CACHE_LINE_SIZE)));

struct ompc_tree_barrier
{
    int depth;
    struct ompc_tree_barrier_node nodes[MAX_PROC - 1];
};

struct ompc_event
{
    _Bool flag;
    ABT_mutex mutex;
    ABT_cond cond;
};

struct ompc_thread;

void ompc_tree_barrier_init(struct ompc_tree_barrier *barrier,
                            int num_threads);
void ompc_tree_barrier_finalize(struct ompc_tree_barrier *barrier);
void ompc_tree_barrier_wait(struct ompc_tree_barrier *barrier,
                            struct ompc_thread *thread);

void ompc_event_init(struct ompc_event *event);
void ompc_event_finalize(struct ompc_event *event);
void ompc_event_set(struct ompc_event *event);
void ompc_event_reset(struct ompc_event *event);
void ompc_event_wait(struct ompc_event *event);
