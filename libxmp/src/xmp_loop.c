/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <math.h>
#include "xmp_internal.h"

// normalize ser_init, ser_cond, ser_step -------------------------------------------------------------------------------------------
#define _XMP_SM_GTOL_BLOCK(_i, _m, _w) \
(((_i) - (_m)) % (_w))

#define _XMP_SM_GTOL_CYCLIC(_i, _m, _P) \
(((_i) - (_m)) / (_P))

#define _XMP_SM_GTOL_BLOCK_CYCLIC(_b, _i, _m, _P) \
(((((_i) - (_m)) / (((_P) * (_b)))) * (_b)) + (((_i) - (_m)) % (_b)))


#define _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter) \
{ \
  if (ser_step == 0) _XMP_fatal("loop step is 0"); \
  if (ser_step == 1) ser_cond--; \
  else { \
    if (ser_step > 0) ser_cond -= ((ser_cond - ser_init) % ser_step); \
    else { \
      reverse_iter = _XMP_N_INT_TRUE; \
\
      ser_step = -ser_step; \
      ser_cond++; \
      ser_cond += ((ser_init - ser_cond) % ser_step); \
      int swap_temp = ser_init; \
      ser_init = ser_cond; \
      ser_cond = swap_temp; \
    } \
  } \
}

#define _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter) \
{ \
  if (reverse_iter) { \
    int temp = *par_init; \
    *par_init = *par_cond; \
    *par_cond = temp - 1; \
    *par_step = -(*par_step); \
  } else { \
    (*par_cond)++; \
  } \
}

#define _XMP_SM_SCHED_LOOP_TEMPLATE_WIDTH_1(ser_init, ser_cond, ser_step, par_init, par_cond, \
                                            template_lower, template_upper, template_stride) \
{ \
  if (ser_step != 1) { \
    _XMP_fatal("loop step is not 1, -1: unsupported case"); \
  } \
\
  int x, x_max = (int)floor(((double)(template_upper - template_lower)) / ((double)template_stride)); \
\
  /* calc par_init */ \
  x = (int)ceil(((double)(ser_init - template_lower)) / (double)(template_stride)); \
  if (x < 0) *par_init = template_lower; \
  else if (x > x_max) goto no_iter; \
  else *par_init = (x * template_stride) + template_lower; \
\
  /* calc par_cond */ \
  x = (int)floor(((double)(ser_cond - template_lower)) / ((double)template_stride)); \
  if (x < 0) goto no_iter; \
  else if (x > x_max) *par_cond = template_upper; \
  else *par_cond = (x * template_stride) + template_lower; \
}

#define _XMP_SM_SCHED_LOOP_TEMPLATE_WIDTH_N(ser_init, ser_cond, ser_step, par_init, par_cond, \
                                            template_lower, template_upper, template_stride, \
                                            width, template_ser_lower, template_ser_upper) \
{ \
  int si = ((ser_init - template_ser_lower) / width) + template_ser_lower; \
  int sc = ((ser_cond - template_ser_lower) / width) + template_ser_lower; \
  int tl = ((template_lower - template_ser_lower) / width) + template_ser_lower; \
  int tu = ((template_upper - template_ser_lower) / width) + template_ser_lower; \
  int ts = template_stride / width; \
\
  /* FIXME HOW IMPLEMENT??? */ \
  _XMP_SM_SCHED_LOOP_TEMPLATE_WIDTH_1(si, sc, 1, par_init, par_cond, tl, tu, ts) \
\
  int par_init_temp = ((*par_init - template_ser_lower) * width) + template_ser_lower; \
  if (par_init_temp < ser_init) *par_init = ser_init; \
  else *par_init = par_init_temp; \
\
  int par_cond_temp = ((*par_cond - template_ser_lower) * width) + template_ser_lower + width - 1; \
  if (par_cond_temp > template_ser_upper) par_cond_temp = template_ser_upper; \
  if (par_cond_temp > ser_cond) *par_cond = ser_cond; \
  else *par_cond = par_cond_temp; \
}

// schedule by template -------------------------------------------------------------------------------------------------------------
// duplicate distribution -----------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_DUPLICATION(int ser_init, int ser_cond, int ser_step,
                                          int *par_init, int *par_cond, int *par_step,
                                          _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed); // FIXME too strict?

  if (!template->is_owner) {
    goto no_iter;
  }

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  _XMP_SM_SCHED_LOOP_TEMPLATE_WIDTH_1(ser_init, ser_cond, ser_step, par_init, par_cond,
                                      template_lower, template_upper, template_stride)
  *par_step = ser_step;

  // finalize iter
  _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  return;

no_iter:
  *par_init = 0;
  *par_cond = 0;
  *par_step = 1;
}

// block distribution ---------------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_BLOCK(int ser_init, int ser_cond, int ser_step,
                                    int *par_init, int *par_cond, int *par_step,
                                    _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    goto no_iter;
  }

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;
  int width = template_chunk->par_chunk_width;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  _XMP_SM_SCHED_LOOP_TEMPLATE_WIDTH_1(ser_init, ser_cond, ser_step, par_init, par_cond,
                                      template_lower, template_upper, template_stride)
  *par_init = _XMP_SM_GTOL_BLOCK(*par_init, template_lower, width);
  *par_cond = _XMP_SM_GTOL_BLOCK(*par_cond, template_lower, width);
  *par_step = ser_step;

  // finalize iter
  _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  return;

no_iter:
  *par_init = 0;
  *par_cond = 0;
  *par_step = 1;
}

// cyclic distribution ---------------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                     int *par_init, int *par_cond, int *par_step,
                                     _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    goto no_iter;
  }

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int nodes_size = (template_chunk->onto_nodes_info)->size;
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  _XMP_SM_SCHED_LOOP_TEMPLATE_WIDTH_1(ser_init, ser_cond, ser_step, par_init, par_cond,
                                      template_lower, template_upper, template_stride)
  *par_init = _XMP_SM_GTOL_CYCLIC(*par_init, template_lower, nodes_size);
  *par_cond = _XMP_SM_GTOL_CYCLIC(*par_cond, template_lower, nodes_size);
  *par_step = 1;

  // finalize iter
  _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  return;

no_iter:
  *par_init = 0;
  *par_cond = 0;
  *par_step = 1;
}

// block-cyclic distribution ---------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_BLOCK_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                           int *par_init, int *par_cond, int *par_step,
                                           _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    goto no_iter;
  }

  _XMP_template_info_t *template_info = &(template->info[template_index]);
  int template_ser_lower = template_info->ser_lower;
  int template_ser_upper = template_info->ser_upper;

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int nodes_size = (template_chunk->onto_nodes_info)->size;
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;
  int width = template_chunk->par_width;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  _XMP_SM_SCHED_LOOP_TEMPLATE_WIDTH_N(ser_init, ser_cond, ser_step, par_init, par_cond,
                                      template_lower, template_upper, template_stride,
                                      width, template_ser_lower, template_ser_upper)
  *par_init = _XMP_SM_GTOL_BLOCK_CYCLIC(width, *par_init, template_lower, nodes_size);
  *par_cond = _XMP_SM_GTOL_BLOCK_CYCLIC(width, *par_cond, template_lower, nodes_size);
  *par_step = 1;

  // finalize iter
  _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  return;

no_iter:
  *par_init = 0;
  *par_cond = 0;
  *par_step = 1;
}

// schedule by nodes ----------------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_nodes(int ser_init, int ser_cond, int ser_step,
                           int *par_init, int *par_cond, int *par_step,
                           _XMP_nodes_t *nodes, int nodes_index) {
  if (!nodes->is_member) {
    goto no_iter;
  }

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  int rank1O = ((nodes->info[nodes_index].rank) + 1);
  if ((rank1O < ser_init) || (rank1O > ser_cond)) {
    goto no_iter;
  }

  if (((rank1O - ser_init) % ser_step) == 0) {
    *par_init = rank1O;
    *par_cond = rank1O + 1;
    *par_step = ser_step;
    return;
  }

no_iter:
  *par_init = 0;
  *par_cond = 0;
  *par_step = 1;
}
