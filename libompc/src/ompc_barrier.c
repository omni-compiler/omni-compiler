/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_barrier.c
 */

#include "ompclib.h"

// returns whether mynode has any children
static _Bool init_node(struct ompc_tree_barrier_desc *desc,
                       struct ompc_tree_barrier_node *parent,
                       int height)
{
    struct ompc_tree_barrier_node *mynode = &desc->nodes[desc->node_count++];
    mynode->sense = 0;
    mynode->parent = parent;
    
    int num_children;
    if (height == 0) {
        if (desc->leaf_count < desc->num_leaves) {
            desc->leaves[desc->leaf_count++] = mynode;
            num_children = desc->leaf_count * 2 > desc->num_threads ? 1 : 2;
        } else {
            num_children = 0;
        }
    } else {
        if (!init_node(desc, mynode, height - 1)) {
            num_children = 0;
        } else if (!init_node(desc, mynode, height - 1)) {
            num_children = 1;
        } else {
            num_children = 2;
        }
    }
    mynode->num_children = mynode->count = num_children;
    
    if (num_children > 0) {
        ABT_mutex_create(&mynode->mutex);
        ABT_cond_create(&mynode->cond);
        return 1;
    }
    
    return 0;
}

void ompc_init_tree_barrier(struct ompc_tree_barrier_desc *desc,
                            int num_threads)
{
    desc->num_threads = num_threads;
    desc->num_leaves = (num_threads + 1) / 2;
    int height = 0;
    for (int n = num_threads; n > 1; n /= 2) {
        height++;
    }
    init_node(desc, NULL, height);
}

void ompc_finalize_tree_barrier(struct ompc_tree_barrier_desc *desc)
{
    for (int i = 0; i < desc->node_count; i++) {
        if (desc->nodes[i].num_children > 0) {
            ABT_cond_free(&desc->nodes[i].cond);
            ABT_mutex_free(&desc->nodes[i].mutex);
        }
    }
}

static void wait_on_node(struct ompc_tree_barrier_node *node,
                         _Bool sense)
{
    if (__sync_fetch_and_sub(&node->count, 1) == 1) {
        if (node->parent != NULL) {
            wait_on_node(node->parent, sense);
        }
        node->count = node->num_children;
        node->sense = sense;
    } else {
        OMPC_WAIT_UNTIL(node->sense == sense, node->cond, node->mutex);
    }
}

void ompc_tree_barrier(struct ompc_thread *thread,
                       struct ompc_tree_barrier_desc *desc,
                       int thread_num)
{
    thread->barrier_sense = !thread->barrier_sense;
    wait_on_node(desc->leaves[thread_num / 2], thread->barrier_sense);
}
