#ifndef _ACC_PEZY_HEADER
#define _ACC_PEZY_HEADER

#include "acc_pezy_util.hpp"
#include "acc_pezy_reduction.hpp"

//Macro
#define _ACC_M_FLOORi(a_, b_) ((a_) / (b_))
#define _ACC_M_COUNT_TRIPLETi(l_, u_, s_) ( ((u_) >= (l_))? _ACC_M_FLOORi((u_) - (l_), s_) + 1 : 0)
#define _ACC_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
#define _ACC_M_MAX(a_, b_) ((a_) > (b_) ? (a_) : (b_))
#define _ACC_M_MIN(a_, b_) ((a_) > (b_) ? (b_) : (a_))

template<typename T, typename T0>
static inline 
void _ACC_init_iter_block_thread_x(T *bt_idx, T *bt_cond, T *bt_step, T0 totalIter)
{
  T gang_size = _ACC_M_CEILi(totalIter, /*gridDim.x*/get_maxpid());
  *bt_idx  = get_pid() * get_maxtid() + get_tid(); //gang_size * /*blockIdx.x*/get_pid() + /*threadIdx.x*/get_tid();
  *bt_cond = totalIter; //_ACC_M_MIN(gang_size * (/*blockIdx.x*/get_pid() + 1), totalIter);
  *bt_step = get_maxpid() * get_maxtid(); ///*blockDim.x*/get_maxtid();
}

template<typename T, typename T0>
static inline
void _ACC_init_iter_block_x(T *gang_iter, T *gang_cond, T *gang_step, T0 totaliter){
  T0 gang_size = _ACC_M_CEILi(totaliter, /*gridDim.x*/get_maxpid());
  *gang_iter = get_pid(); //gang_size * /*blockIdx.x*/get_pid();
  *gang_cond = totaliter; //_ACC_M_MIN(*gang_iter + gang_size, totaliter);
  *gang_step = get_maxpid(); //1;
}

template<typename T, typename T0>
static inline
void _ACC_init_iter_thread_x(T *iter, T *cond, T *step, T0 totaliter){
  *iter = /*threadIdx.x*/get_tid();
  *cond = totaliter;
  *step = /*blockDim.x*/get_maxtid();
}

template<typename T, typename T0, typename T1, typename T2>
static inline
void _ACC_calc_niter(T *niter, T0 init, T1 cond, T2 step)
{
  *niter = _ACC_M_COUNT_TRIPLETi(init, cond - 1, step);
}

template<typename T, typename T0, typename T1, typename T2, typename T3>
static inline
void _ACC_calc_idx(T id, T0 *idx, T1 lower, T2 upper, T3 stride)
{
  *idx = lower + stride * id;
}

static inline
void _ACC_sync_threads()
{
  wait_pe();
}

static inline
void _ACC_flush()
{
  flush();
}

static inline
void _ACC_sync(int const n)
{
  switch(n){
  case 0:
    wait_pe();
    break;
  case 1:
    wait_village();
    break;
  case 2:
    wait_city();
    break;
  case 3:
    wait_prefecture();
    break;
  }
}

static inline
void _ACC_sync_gangs()
{
  //sync();
  flush();
}

static inline
void _ACC_sync_all()
{
  sync();
}

static inline
void _ACC_flush(int const n)
{
  switch(n){
  case 1:
    flush_L1();
    break;
  case 2:
    flush_L2();
    break;
  case 3:
    flush_L3();
    break;
  }
}

static inline
void _ACC_flush_all()
{
  _ACC_flush();
}

static inline
void _ACC_yield()
{
  chgthread();
}

#define _ACC_block_x_id (get_pid())
#define _ACC_thread_x_id (get_tid())

static inline
void _ACC_stack_init(unsigned long *pos)
{
  *pos = 0;
}

static inline
void *_ACC_stack_push(void *base, unsigned long *pos, size_t size)
{
  void *ret = (void*)((char*)base + *pos);
  *pos += size;
  return ret;
}

template<typename T, typename T0, typename T1>
static inline
int _ACC_calc_vidx(T *idx, T0 niter, T1 total_idx){
  *idx = total_idx % niter;
  return total_idx / niter;
}

#endif
