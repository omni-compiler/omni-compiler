/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_task_scheduler.c
 */

#include <abt.h>
#include "ompclib.h"

__thread struct ompc_thread *current_thread;

static int task_sched_init(ABT_sched self, ABT_sched_config config)
{
    return 0;
}

static void task_sched_run(ABT_sched self)
{
    struct ompc_thread *thread;
    ABT_sched_get_data(self, &thread);
    struct ompc_thread *last_thread = current_thread;
    current_thread = thread;

    // 0: private pool, 1: shared pool
    ABT_pool pools[2];
    ABT_sched_get_pools(self, 2, 0, pools);
    int work_count = 0;

    while (1) {
        // search private pool first, then shared pool
        int i;
        for (i = 0; i < 2; i++) {
            size_t size;
            ABT_pool_get_size(pools[i], &size);
            if (size > 0) {
                ABT_unit unit;
                ABT_pool_pop(pools[i], &unit);
                if (unit != ABT_UNIT_NULL) {
                    ABT_xstream_run_unit(unit, pools[i]);
                }
                break;
            }
        }

        // check events frequently, or when there's no unit to run
        if (++work_count >= 50 || i == 2) {
            int stop;
            ABT_sched_has_to_stop(self, &stop);
            if (stop) break;
            work_count = 0;
            ABT_xstream_check_events(self);
        }

        if (i == 2) {
            current_thread = last_thread;
            ABT_thread_yield();
            current_thread = thread;
        }
    }

    current_thread = last_thread;
    ompc_event_set(&thread->sched_finished);
}

static int task_sched_free(ABT_sched self)
{
    return 0;
}

ABT_sched_def const task_sched_def = {
    .type = ABT_SCHED_TYPE_ULT,
    .init = task_sched_init,
    .run = task_sched_run,
    .free = task_sched_free,
};
