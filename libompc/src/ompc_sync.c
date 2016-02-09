/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_sync.c
 */

#include "ompclib.h"

void ompc_tree_barrier_init(struct ompc_tree_barrier *barrier,
                            int num_threads)
{
    barrier->depth = 0;
    for (int n = num_threads; n > 2; n = (n + 1) / 2) {
        barrier->depth++;
    }

    for (int d = 0; d <= barrier->depth; d++) {
        int start_idx = (1 << d) - 1;
        int leaf_incr = 2 << (barrier->depth - d);
        int leaf_count = 0;
        for (int i = 0; i < (1 << d); i++) {
            if (leaf_count >= num_threads) break;
            struct ompc_tree_barrier_node *node = &barrier->nodes[start_idx + i];
            int num_children = num_threads > leaf_count + leaf_incr / 2 ? 2 : 1;
            node->num_children = node->count = num_children;
            node->sense = 0;
            if (num_children == 2) {
                ABT_mutex_create(&node->mutex);
                ABT_cond_create(&node->cond);
            }
            leaf_count += leaf_incr;
        }
    }
}

void ompc_tree_barrier_finalize(struct ompc_tree_barrier *barrier)
{
    for (int d = 0; d <= barrier->depth; d++) {
        int start_idx = (1 << d) - 1;
        for (int i = 0; i < (1 << d); i++) {
            struct ompc_tree_barrier_node *node = &barrier->nodes[start_idx + i];
            if (node->num_children == 2) {
                ABT_mutex_free(&node->mutex);
                ABT_cond_free(&node->cond);
            }
        }
    }
}

void ompc_tree_barrier_wait(struct ompc_tree_barrier *barrier,
                            struct ompc_thread *thread)
{
    int node_idx = (1 << barrier->depth) + (thread->num >> 1) - 1;
    int stack_count = 0;
    _Bool sense = !thread->barrier_sense;
    thread->barrier_sense = sense;

    while (1) {
        struct ompc_tree_barrier_node *node = &barrier->nodes[node_idx];
        if (__sync_fetch_and_sub(&node->count, 1) != 1) {
            OMPC_WAIT_UNTIL(node->sense == sense, node->cond, node->mutex);
            break;
        }
        thread->node_stack[stack_count++] = node;
        if (node_idx == 0) break;
        node_idx = (node_idx - 1) >> 1;
    }

    while (stack_count > 0) {
        struct ompc_tree_barrier_node *node = thread->node_stack[--stack_count];
        node->count = node->num_children;
        if (node->num_children == 2) {
            OMPC_SIGNAL(node->sense = sense, node->cond, node->mutex);
        } else {
            node->sense = sense;
        }
    }
}

void ompc_event_init(struct ompc_event *event)
{
    event->flag = 0;
    ABT_mutex_create(&event->mutex);
    ABT_cond_create(&event->cond);
}

void ompc_event_finalize(struct ompc_event *event)
{
    ABT_cond_free(&event->cond);
    ABT_mutex_free(&event->mutex);
}

void ompc_event_set(struct ompc_event *event)
{
    OMPC_BROADCAST(event->flag = 1, event->cond, event->mutex);
}

void ompc_event_reset(struct ompc_event *event)
{
    event->flag = 0;
}

void ompc_event_wait(struct ompc_event *event)
{
    OMPC_WAIT_UNTIL(event->flag == 1, event->cond, event->mutex);
}
