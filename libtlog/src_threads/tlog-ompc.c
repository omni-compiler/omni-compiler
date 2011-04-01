/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include "exc_platform.h"
#include "tlog.h"

void tlog_parallel_IN(int id);
void tlog_parallel_OUT(int id);
void tlog_barrier_IN(int id);
void tlog_barrier_OUT(int id);
void tlog_loop_init_EVENT(int id);
void tlog_loop_next_EVENT(int id);
void tlog_section_EVENT(int id);
void tlog_single_EVENT(int id);
void tlog_critial_IN(int id);
void tlog_critial_OUT(int id);

void tlog_parallel_IN(int id)
{
    tlog_log(id,TLOG_PARALLEL_IN);
}

void tlog_parallel_OUT(int id)
{
    tlog_log(id,TLOG_PARALLEL_OUT);
}

void tlog_barrier_IN(int id)
{
    tlog_log(id,TLOG_BARRIER_IN);
}

void tlog_barrier_OUT(int id)
{
    tlog_log(id,TLOG_BARRIER_OUT);
}

void tlog_loop_init_EVENT(int id)
{
    tlog_log(id,TLOG_LOOP_INIT_EVENT);
}

void tlog_loop_next_EVENT(int id)
{
    tlog_log(id,TLOG_LOOP_NEXT_EVENT);
}

void tlog_section_EVENT(int id)
{
    tlog_log(id,TLOG_SECTION_EVENT);
}

void tlog_single_EVENT(int id)
{
    tlog_log(id,TLOG_SIGNLE_EVENT);
}

void tlog_critial_IN(int id)
{
    tlog_log(id,TLOG_CRITICAL_IN);
}

void tlog_critial_OUT(int id)
{
    tlog_log(id,TLOG_CRITICAL_OUT);
}

