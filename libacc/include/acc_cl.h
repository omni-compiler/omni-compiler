#ifndef _ACC_CL_HEADER
#define _ACC_CL_HEADER

#pragma OPENCL EXTENSION cl_khr_fp64: enable

//#include "acc_cl_reduction.h"

#define _ACC_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
#define _ACC_M_MAX(a_, b_) ((a_) > (b_) ? (a_) : (b_))
#define _ACC_M_MIN(a_, b_) ((a_) > (b_) ? (b_) : (a_))

#define _ACC_calc_niter(niter, init, cond, step)			\
    do{*(niter) = ((cond) - (init) + ((step) > 0? -1 : 1)) / (step) + 1;}while(0)


#define _ACC_init_iter_block_x(gang_init, gang_cond, gang_step, totaliter)\
    do{ \
    int gang_size = _ACC_M_CEILi((totaliter), get_num_groups(0));	\
    *(gang_init) = (gang_size) * get_group_id(0);\
    *(gang_cond) = _ACC_M_MIN(*(gang_init) + (gang_size), (totaliter));	\
    *(gang_step) = 1;\
    }while(0)

#define _ACC_calc_idx(id, idx, lower, upper, stride)\
    *(idx) = (lower) + (stride) * (id)

#define _ACC_init_iter_thread_x(iter, cond, step, totaliter)\
    do{\
	*(iter) = get_local_id(0);		\
	*(cond) = (totaliter);			\
	*(step) = get_local_size(0);		\
    }while(0)

#define _ACC_sync_threads() barrier(CLK_GLOBAL_MEM_FENCE)

#define _ACC_init_iter_block_thread_x(bt_init, bt_cond, bt_step, totalIter)\
  do{\
  int gang_size = _ACC_M_CEILi((totalIter), get_num_groups(0));\
  *(bt_init) = (gang_size) * get_group_id(0) + get_local_id(0);\
  *(bt_cond) = _ACC_M_MIN(gang_size * (get_group_id(0) + 1), (totalIter));\
  *(bt_step) = get_local_size(0);\
  }while(0)

#endif
