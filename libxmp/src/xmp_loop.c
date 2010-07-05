#include "xmp_internal.h"

// normalize ser_init, ser_cond, ser_step -------------------------------------------------------------------------------------------
#define _XCALABLEMP_SM_NORM_SCHED_PARAMS_S(_type) \
if (ser_step == 0) _XCALABLEMP_fatal("loop step is 0"); \
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

#define _XCALABLEMP_SM_NORM_SCHED_PARAMS_U \
if (ser_step == 0) _XCALABLEMP_fatal("loop step is 0"); \
if (ser_step == 1) ser_cond--; \
else ser_cond -= ((ser_cond - ser_init) % ser_step);

// schedule by template -------------------------------------------------------------------------------------------------------------
// block distribution ---------------------------------------------------------------------------------------------------------------
#define _XCALABLEMP_SM_GET_TEMPLATE_INFO_BLOCK(_type) \
if (template->chunk == NULL) _XCALABLEMP_fatal("template in loop directive is not distributed"); \
_type template_lower = (_type)template->chunk[template_index].par_lower; \
_type template_upper = (_type)template->chunk[template_index].par_upper;

#define _XCALABLEMP_SM_NORM_TEMPLATE_BLOCK(_type) \
if (ser_step != 1) { \
  _type dst_mod = ser_init % ser_step; \
  /* normalize template lower */ \
  _type lower_mod = template_lower % ser_step; \
  if (lower_mod != dst_mod) { \
    if (lower_mod < dst_mod)   template_lower += (dst_mod - lower_mod); \
    else template_lower += (ser_step - lower_mod + dst_mod); \
  } \
  if (template_lower > template_upper) goto no_iter; \
  /* normalize template upper FIXME needed??? */ \
  _type upper_mod = template_upper % ser_step; \
  if (upper_mod != dst_mod) { \
    if (dst_mod < upper_mod)   template_upper -= (upper_mod - dst_mod); \
    else template_upper -= (ser_step - dst_mod + upper_mod); \
  } \
}

#define _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK \
  /* calc par_init */ \
  if (ser_init < template_lower) *par_init = template_lower; \
  else if (template_upper < ser_init) goto no_iter; \
  else  *par_init = ser_init; \
  /* calc par_cond */ \
  if (ser_cond < template_lower) goto no_iter; \
  else if (template_upper < ser_cond) *par_cond = template_upper + 1; /* exprcode is LT(<) */ \
  else *par_cond = ser_cond + 1; /* exprcode is LT(<) */ \
  return; \
no_iter: \
  *par_init = 0; \
  *par_cond = 0; \
  return;

#define _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, \
 const _XCALABLEMP_template_t *const template, const int template_index) { \
  _XCALABLEMP_SM_GET_TEMPLATE_INFO_BLOCK(_type) \
  _XCALABLEMP_SM_NORM_SCHED_PARAMS_S(_type) \
  _XCALABLEMP_SM_NORM_TEMPLATE_BLOCK(_type) \
  _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK \
}

#define _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(_type) \
(const _type ser_init, _type ser_cond, const _type ser_step, \
 _type *const par_init, _type *const par_cond, \
 const _XCALABLEMP_template_t *const template, const int template_index) { \
  _XCALABLEMP_SM_GET_TEMPLATE_INFO_BLOCK(_type) \
  _XCALABLEMP_SM_NORM_SCHED_PARAMS_U \
  _XCALABLEMP_SM_NORM_TEMPLATE_BLOCK(_type) \
  _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK \
}

void _XCALABLEMP_sched_loop_template_BLOCK_CHAR               _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(char)
void _XCALABLEMP_sched_loop_template_BLOCK_UNSIGNED_CHAR      _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned char)
void _XCALABLEMP_sched_loop_template_BLOCK_SHORT              _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(short)
void _XCALABLEMP_sched_loop_template_BLOCK_UNSIGNED_SHORT     _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned short)
void _XCALABLEMP_sched_loop_template_BLOCK_INT                _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(int)
void _XCALABLEMP_sched_loop_template_BLOCK_UNSIGNED_INT       _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned int)
void _XCALABLEMP_sched_loop_template_BLOCK_LONG               _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(long)
void _XCALABLEMP_sched_loop_template_BLOCK_UNSIGNED_LONG      _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned long)
void _XCALABLEMP_sched_loop_template_BLOCK_LONG_LONG          _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(long long)
void _XCALABLEMP_sched_loop_template_BLOCK_UNSIGNED_LONG_LONG _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned long long)

// cyclic distribution ---------------------------------------------------------------------------------------------------------------
#define _XCALABLEMP_SM_GET_TEMPLATE_INFO_CYCLIC(_type) \
if (template->chunk == NULL) _XCALABLEMP_fatal("template in loop directive is not distributed"); \
_type nodes_size = (_type)template->chunk[template_index].onto_nodes_info->size; \
_type template_lower = (_type)template->chunk[template_index].par_lower;

#define _XCALABLEMP_SM_CALC_PAR_INIT_CYCLIC_S1(_type) \
{ \
  _type par_init_temp = ser_init; \
  _type dst_mod = template_lower % nodes_size; \
  _type init_mod = par_init_temp % nodes_size; \
  if (init_mod != dst_mod) { \
    if (init_mod < dst_mod) par_init_temp += (dst_mod - init_mod); \
    else par_init_temp += (nodes_size - init_mod + dst_mod); \
  } \
  if (ser_cond < par_init_temp) goto no_iter; \
  *par_init = par_init_temp; \
}

#define _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC(_type) \
  if (ser_step == 1) { \
    _XCALABLEMP_SM_CALC_PAR_INIT_CYCLIC_S1(_type) \
    /* calc par_step */ \
    *par_step = nodes_size; \
  } \
  else _XCALABLEMP_fatal("not implemented condition: (loop step is not 1 && cyclic distribution)"); \
  return; \
no_iter: \
  *par_init = ser_cond + 1; /* exprcode is LT(<) */ \
  return;

#define _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_step, \
 const _XCALABLEMP_template_t *const template, const int template_index) { \
  _XCALABLEMP_SM_GET_TEMPLATE_INFO_CYCLIC(_type) \
  _XCALABLEMP_SM_NORM_SCHED_PARAMS_S(_type) \
  _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC(_type) \
}

#define _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(_type) \
(const _type ser_init, _type ser_cond, const _type ser_step, \
 _type *const par_init, _type *const par_step, \
 const _XCALABLEMP_template_t *const template, const int template_index) { \
  _XCALABLEMP_SM_GET_TEMPLATE_INFO_CYCLIC(_type) \
  _XCALABLEMP_SM_NORM_SCHED_PARAMS_U \
  _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC(_type) \
}

void _XCALABLEMP_sched_loop_template_CYCLIC_CHAR               _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(char)
void _XCALABLEMP_sched_loop_template_CYCLIC_UNSIGNED_CHAR      _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned char)
void _XCALABLEMP_sched_loop_template_CYCLIC_SHORT              _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(short)
void _XCALABLEMP_sched_loop_template_CYCLIC_UNSIGNED_SHORT     _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned short)
void _XCALABLEMP_sched_loop_template_CYCLIC_INT                _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(int)
void _XCALABLEMP_sched_loop_template_CYCLIC_UNSIGNED_INT       _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned int)
void _XCALABLEMP_sched_loop_template_CYCLIC_LONG               _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(long)
void _XCALABLEMP_sched_loop_template_CYCLIC_UNSIGNED_LONG      _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned long)
void _XCALABLEMP_sched_loop_template_CYCLIC_LONG_LONG          _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(long long)
void _XCALABLEMP_sched_loop_template_CYCLIC_UNSIGNED_LONG_LONG _XCALABLEMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned long long)

// schedule by nodes ----------------------------------------------------------------------------------------------------------------
#define _XCALABLEMP_SM_SCHED_LOOP_NODES(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, \
 const _XCALABLEMP_nodes_t *const nodes, const int nodes_index) { \
  _XCALABLEMP_SM_NORM_SCHED_PARAMS_S(_type) \
  _type rank1O = (_type)((nodes->info[nodes_index].rank) + 1); \
  if (rank1O < ser_init) goto no_iter; \
  if (rank1O > ser_cond) goto no_iter; \
  if (((rank1O - ser_init) % ser_step) == 0) { \
    *par_init = rank1O; \
    *par_cond = rank1O + 1; \
    return; \
  } \
  else goto no_iter; \
no_iter: \
  *par_init = 0; \
  *par_cond = 0; \
  return; \
}

void _XCALABLEMP_sched_loop_nodes_CHAR               _XCALABLEMP_SM_SCHED_LOOP_NODES(char)
void _XCALABLEMP_sched_loop_nodes_UNSIGNED_CHAR      _XCALABLEMP_SM_SCHED_LOOP_NODES(unsigned char)
void _XCALABLEMP_sched_loop_nodes_SHORT              _XCALABLEMP_SM_SCHED_LOOP_NODES(short)
void _XCALABLEMP_sched_loop_nodes_UNSIGNED_SHORT     _XCALABLEMP_SM_SCHED_LOOP_NODES(unsigned short)
void _XCALABLEMP_sched_loop_nodes_INT                _XCALABLEMP_SM_SCHED_LOOP_NODES(int)
void _XCALABLEMP_sched_loop_nodes_UNSIGNED_INT       _XCALABLEMP_SM_SCHED_LOOP_NODES(unsigned int)
void _XCALABLEMP_sched_loop_nodes_LONG               _XCALABLEMP_SM_SCHED_LOOP_NODES(long)
void _XCALABLEMP_sched_loop_nodes_UNSIGNED_LONG      _XCALABLEMP_SM_SCHED_LOOP_NODES(unsigned long)
void _XCALABLEMP_sched_loop_nodes_LONG_LONG          _XCALABLEMP_SM_SCHED_LOOP_NODES(long long)
void _XCALABLEMP_sched_loop_nodes_UNSIGNED_LONG_LONG _XCALABLEMP_SM_SCHED_LOOP_NODES(unsigned long long)
