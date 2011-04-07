#include "tlog_mpi.h"

void _tlog_block_swap_bytes(TLOG_DATA *dp)
{
    union {
	char c[8];
	short int s;
	long int l;
	double d;
    } x;
    char t;
    TLOG_DATA *end_dp;

    end_dp = (TLOG_DATA *)(((char *)dp) + TLOG_BLOCK_SIZE);
    for(; dp < end_dp; dp++){
	x.s = dp->proc_id;
	t = x.c[0]; x.c[0] = x.c[1]; x.c[1] = t;
	dp->proc_id = x.s;
	x.l = dp->arg2;
	t = x.c[0]; x.c[0] = x.c[3]; x.c[3] = t;
	t = x.c[1]; x.c[1] = x.c[2]; x.c[2] = t;
	dp->arg2 = x.l;
	x.d = dp->time_stamp;
	t = x.c[0]; x.c[0] = x.c[7]; x.c[7] = t;
	t = x.c[1]; x.c[1] = x.c[6]; x.c[6] = t;
	t = x.c[2]; x.c[2] = x.c[5]; x.c[5] = t;
	t = x.c[3]; x.c[3] = x.c[4]; x.c[4] = t;
	dp->time_stamp = x.d;
    }
}
