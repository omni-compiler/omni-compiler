/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

// normalize ser_init, ser_cond, ser_step -------------------------------------------------------------------------------------------
#define _XMP_M_GTOL_BLOCK(_i, _w) \
((_i) % (_w))

#define _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step) \
{ \
  if (ser_step == 0) _XMP_fatal("loop step is 0"); \
  if (ser_step == 1) ser_cond--; \
  else { \
    if (ser_step > 0) ser_cond -= ((ser_cond - ser_init) % ser_step); \
    else { \
      ser_step = -ser_step; \
      ser_cond++; \
      ser_cond += ((ser_init - ser_cond) % ser_step); \
      int swap_temp = ser_init; \
      ser_init = ser_cond; \
      ser_cond = swap_temp; \
    } \
  } \
}

// schedule by template -------------------------------------------------------------------------------------------------------------
// block distribution ---------------------------------------------------------------------------------------------------------------
#define _XMP_SM_NORM_TEMPLATE_BLOCK(ser_init, ser_step, template_lower, template_upper) \
{ \
  if (ser_step != 1) { \
    int dst_mod = ser_init % ser_step; \
    if (dst_mod < 0) dst_mod = (dst_mod + ser_step) % ser_step; \
    /* normalize template lower */ \
    int lower_mod = template_lower % ser_step; \
    if (lower_mod < 0) lower_mod = (lower_mod + ser_step) % ser_step; \
    if (lower_mod != dst_mod) { \
      if (lower_mod < dst_mod)   template_lower += (dst_mod - lower_mod); \
      else template_lower += (ser_step - lower_mod + dst_mod); \
    } \
    if (template_lower > template_upper) goto no_iter; \
  } \
}

#define _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK(ser_init, ser_cond, ser_step, par_init, par_cond, par_step, \
                                          template_lower, template_upper) \
{ \
  /* calc par_init */ \
  if (ser_init < template_lower) *par_init = template_lower; \
  else if (template_upper < ser_init) goto no_iter; \
  else  *par_init = ser_init; \
  /* calc par_cond */ \
  if (ser_cond < template_lower) goto no_iter; \
  else if (template_upper < ser_cond) *par_cond = template_upper; \
  else *par_cond = ser_cond; \
  /* calc par_step */ \
  *par_step = ser_step; \
}

void _XMP_sched_loop_template_BLOCK(int ser_init, int ser_cond, int ser_step,
                                    int *par_init, int *par_cond, int *par_step,
                                    _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed); // FIXME too strict?

  if (!template->is_owner) {
    goto no_iter;
  }

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;

  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step)
  _XMP_SM_NORM_TEMPLATE_BLOCK(ser_init, ser_step, template_lower, template_upper)
  _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK(ser_init, ser_cond, ser_step, par_init, par_cond, par_step,
                                    template_lower, template_upper)

  // GtoL
  int width = template_chunk->par_chunk_width;
  *par_init = _XMP_M_GTOL_BLOCK(*par_init, width);
  *par_cond = _XMP_M_GTOL_BLOCK(*par_cond, width) + 1; // for (i = par_init; i < par_cond; i += par_step) . . .
  // par_step: no translation

  return;

no_iter:
  *par_init = 0;
  *par_cond = 0;
  *par_step = 1;
}

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

  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step)
  _XMP_SM_NORM_TEMPLATE_BLOCK(ser_init, ser_step, template_lower, template_upper)
  _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK(ser_init, ser_cond, ser_step, par_init, par_cond, par_step,
                                    template_lower, template_upper)

  // no GtoL is needed
  *par_cond = *par_cond + 1; // for (i = par_init; i < par_cond; i += par_step) . . .

  return;

no_iter:
  *par_init = 0;
  *par_cond = 0;
  *par_step = 1;
}

// cyclic distribution ---------------------------------------------------------------------------------------------------------------
#define _XMP_SM_CALC_PAR_INIT_CYCLIC_C1 \
{ \
  int par_init_temp = ser_init; \
  int dst_mod = template_lower % nodes_size; \
  if (dst_mod < 0) dst_mod = (dst_mod + nodes_size) % nodes_size; \
  int init_mod = par_init_temp % nodes_size; \
  if (init_mod < 0) init_mod = (init_mod + nodes_size) % nodes_size; \
  if (init_mod != dst_mod) { \
    if (init_mod < dst_mod) par_init_temp += (dst_mod - init_mod); \
    else par_init_temp += (nodes_size - init_mod + dst_mod); \
  } \
  if (ser_cond < par_init_temp) goto no_iter; \
  *par_init = par_init_temp; \
}

void _XMP_sched_loop_template_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                     int *par_init, int *par_cond, int *par_step,
                                     _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    goto no_iter;
  }

  int nodes_size = template->chunk[template_index].onto_nodes_info->size;
  int template_lower = template->chunk[template_index].par_lower;

  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step)

  if (ser_step == 1) {
    _XMP_SM_CALC_PAR_INIT_CYCLIC_C1
    *par_cond = ser_cond + 1; /* restore normalized value */
    *par_step = nodes_size;
    return;
  } else {
    _XMP_fatal("not implemented condition: (loop step is not 1 && cyclic distribution)");
  }

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

  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step)

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
