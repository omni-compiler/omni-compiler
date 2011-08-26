/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

// normalize ser_init, ser_cond, ser_step -------------------------------------------------------------------------------------------
#define _XMP_SM_NORM_SCHED_PARAMS_S(_type, ser_init, ser_cond, ser_step) \
if (ser_step == 0) _XMP_fatal("loop step is 0"); \
if (ser_step == 1) ser_cond--; \
else { \
  if (ser_step > 0) ser_cond -= ((ser_cond - ser_init) % ser_step); \
  else { \
    ser_step = -ser_step; \
    ser_cond++; \
    ser_cond += ((ser_init - ser_cond) % ser_step); \
    _type swap_temp = ser_init; \
    ser_init = ser_cond; \
    ser_cond = swap_temp; \
  } \
}

// schedule by template -------------------------------------------------------------------------------------------------------------
// block distribution ---------------------------------------------------------------------------------------------------------------
#define _XMP_SM_GET_TEMPLATE_INFO_BLOCK(_type, template, template_lower, template_upper) \
{ \
  _XMP_ASSERT(template->is_distributed); \
  if (!template->is_owner) goto no_iter; \
  template_lower = (_type)template->chunk[template_index].par_lower; \
  template_upper = (_type)template->chunk[template_index].par_upper; \
}

#define _XMP_SM_NORM_TEMPLATE_BLOCK_S(_type, ser_init, ser_step, template_lower, template_upper) \
{ \
  if (ser_step != 1) { \
    _type dst_mod = ser_init % ser_step; \
    if (dst_mod < 0) dst_mod = (dst_mod + ser_step) % ser_step; \
    /* normalize template lower */ \
    _type lower_mod = template_lower % ser_step; \
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
  else if (template_upper < ser_cond) *par_cond = template_upper + 1; /* exprcode is LT(<) */ \
  else *par_cond = ser_cond + 1; /* exprcode is LT(<) */ \
  /* calc par_step */ \
  *par_step = ser_step; \
  return; \
}

#define _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const _XMP_template_t *const template, const int template_index) { \
  _type template_lower, template_upper; \
\
  _XMP_SM_GET_TEMPLATE_INFO_BLOCK(_type, template, template_lower, template_upper) \
  _XMP_SM_NORM_SCHED_PARAMS_S(_type, ser_init, ser_cond, ser_step) \
  _XMP_SM_NORM_TEMPLATE_BLOCK_S(_type, ser_init, ser_step, template_lower, template_upper) \
  _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK(ser_init, ser_cond, ser_step, par_init, par_cond, par_step, \
                                           template_lower, template_upper) \
\
no_iter: \
  *par_init = 0; \
  *par_cond = 0; \
  *par_step = 1; \
  return; \
}

void _XMP_sched_loop_template_BLOCK_INT                _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(int)

// cyclic distribution ---------------------------------------------------------------------------------------------------------------
#define _XMP_SM_GET_TEMPLATE_INFO_CYCLIC(_type) \
_XMP_ASSERT(template->is_distributed); \
if (!template->is_owner) goto no_iter; \
_type nodes_size = (_type)template->chunk[template_index].onto_nodes_info->size; \
_type template_lower = (_type)template->chunk[template_index].par_lower;

#define _XMP_SM_CALC_PAR_INIT_CYCLIC_S1_S(_type) \
{ \
  _type par_init_temp = ser_init; \
  _type dst_mod = template_lower % nodes_size; \
  if (dst_mod < 0) dst_mod = (dst_mod + nodes_size) % nodes_size; \
  _type init_mod = par_init_temp % nodes_size; \
  if (init_mod < 0) init_mod = (init_mod + nodes_size) % nodes_size; \
  if (init_mod != dst_mod) { \
    if (init_mod < dst_mod) par_init_temp += (dst_mod - init_mod); \
    else par_init_temp += (nodes_size - init_mod + dst_mod); \
  } \
  if (ser_cond < par_init_temp) goto no_iter; \
  *par_init = par_init_temp; \
}

#define _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const _XMP_template_t *const template, const int template_index) { \
  _XMP_SM_GET_TEMPLATE_INFO_CYCLIC(_type) \
  _XMP_SM_NORM_SCHED_PARAMS_S(_type, ser_init, ser_cond, ser_step) \
  if (ser_step == 1) { \
    /* calc par_init */ \
    _XMP_SM_CALC_PAR_INIT_CYCLIC_S1_S(_type) \
    /* calc par_cond */ \
    *par_cond = ser_cond + 1; /* restore normalized value */ \
    /* calc par_step */ \
    *par_step = nodes_size; \
  } \
  else _XMP_fatal("not implemented condition: (loop step is not 1 && cyclic distribution)"); \
  return; \
no_iter: \
  *par_init = 0; \
  *par_cond = 0; \
  *par_step = 1; \
  return; \
}

void _XMP_sched_loop_template_CYCLIC_INT                _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(int)

// schedule by nodes ----------------------------------------------------------------------------------------------------------------
#define _XMP_SM_SCHED_LOOP_NODES(_type, ser_init, ser_cond, ser_step, par_init, par_cond, par_step, \
                                        nodes, nodes_index) \
{ \
  _type rank1O = (_type)((nodes->info[nodes_index].rank) + 1); \
  if (rank1O < ser_init) goto no_iter; \
  if (rank1O > ser_cond) goto no_iter; \
  if (((rank1O - ser_init) % ser_step) == 0) { \
    *par_init = rank1O; \
    *par_cond = rank1O + 1; \
    *par_step = ser_step; \
    return; \
  } \
  else goto no_iter; \
}

#define _XMP_SM_SCHED_LOOP_NODES_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const _XMP_nodes_t *const nodes, const int nodes_index) { \
  if (!nodes->is_member) goto no_iter; \
  _XMP_SM_NORM_SCHED_PARAMS_S(_type, ser_init, ser_cond, ser_step) \
  _XMP_SM_SCHED_LOOP_NODES(_type, ser_init, ser_cond, ser_step, par_init, par_cond, par_step, \
                                  nodes, nodes_index) \
no_iter: \
  *par_init = 0; \
  *par_cond = 0; \
  *par_step = 1; \
  return; \
}

void _XMP_sched_loop_nodes_INT                _XMP_SM_SCHED_LOOP_NODES_S(int)
