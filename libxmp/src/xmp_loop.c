#include "xmp_internal.h"
#include "xmp_math_function.h"

// normalize ser_init, ser_cond, ser_step -------------------------------------------------------------------------------------------
#define _XMP_SM_GTOL_BLOCK(_i, _m, _w) \
(((_i) - (_m)) % (_w))

#define _XMP_SM_GTOL_CYCLIC(_i, _m, _P) \
(((_i) - (_m)) / (_P))

#define _XMP_SM_GTOL_BLOCK_CYCLIC(_b, _i, _m, _P) \
(((((_i) - (_m)) / (((_P) * (_b)))) * (_b)) + (((_i) - (_m)) % (_b)))

#define _XMP_SM_GTOL_GBLOCK(_i, _g) \
((_i) - (_g))

#define _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter) \
{ \
  if (ser_step == 0) _XMP_fatal("loop step is 0"); \
  if (ser_step == 1) ser_cond--; \
  else { \
    if (ser_step > 0){ \
      ser_cond--; \
      ser_cond -= ((ser_cond - ser_init) % ser_step);	\
    } \
    else { \
      if(reverse_iter != _XMP_N_INT_TRUE) /* This branch hides warning in _XMP_sched_loop_nodes() */	\
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

int _XMP_sched_loop_template_width_1(int ser_init, int ser_cond, int ser_step,
                                     int *par_init, int *par_cond, int *par_step,
                                     int template_lower, int template_upper, int template_stride) {
  int x, x_max = _XMP_floori((template_upper - template_lower), template_stride);
  if (ser_step == 1) {
    // calc par_init
    x = _XMP_ceili((ser_init - template_lower), template_stride);
    if (x < 0) {
      *par_init = template_lower;
    } else if (x > x_max) {
      return _XMP_N_INT_FALSE;
    } else {
      *par_init = (x * template_stride) + template_lower;
    }

    // calc par_cond
    x = _XMP_floori((ser_cond - template_lower), template_stride);
    if (x < 0) {
      return _XMP_N_INT_FALSE;
    } else if (x > x_max) {
      *par_cond = template_upper;
    } else {
      *par_cond = (x * template_stride) + template_lower;
    }

    // calc par_step
    *par_step = template_stride;
  } else {
    if ((template_upper < ser_init) || (ser_cond < template_lower)) {
      return _XMP_N_INT_FALSE;
    }

    // calc par_init
    for (int i = template_lower; i <= template_upper; i += template_stride) {
      if (i < ser_init) {
        continue;
      } else if (((i - ser_init) % ser_step) == 0) {
        *par_init = i;
        goto calc_par_cond;
      }
    }
    return _XMP_N_INT_FALSE;

calc_par_cond:
    // calc par_cond
    for (int i = template_upper; i >= template_lower; i -= template_stride) {
      if (i > ser_cond) {
        continue;
      } else if (((i - ser_init) % ser_step) == 0) {
        *par_cond = i;
        goto calc_par_step;
      }
    }
    return _XMP_N_INT_FALSE;

calc_par_step:
    // calc par_step
    *par_step = _XMP_lcm(ser_step, template_stride);
  }

  return _XMP_N_INT_TRUE;
}

int _XMP_sched_loop_template_width_N(int ser_init, int ser_cond, int ser_step,
                                     int *par_init, int *par_cond, int *par_step,
                                     int template_lower, int template_upper, int template_stride,
                                     int width, int template_ser_lower, int template_ser_upper) {
  int si = ((ser_init - template_ser_lower) / width) + template_ser_lower;
  int sc = ((ser_cond - template_ser_lower) / width) + template_ser_lower;
  int tl = ((template_lower - template_ser_lower) / width) + template_ser_lower;
  int tu = ((template_upper - template_ser_lower) / width) + template_ser_lower;
  int ts = template_stride / width;

  /* FIXME HOW IMPLEMENT??? */
  if (ser_step != 1) {
    _XMP_fatal("loop step is not 1, -1: unsupported case");
  }

  if (_XMP_sched_loop_template_width_1(si, sc, 1, par_init, par_cond, par_step, tl, tu, ts)) {
    // init par_init
    int par_init_temp = ((*par_init - template_ser_lower) * width) + template_ser_lower;
    if (par_init_temp < ser_init) {
      *par_init = ser_init;
    } else {
      *par_init = par_init_temp;
    }

    // init par_cond
    int par_cond_temp = ((*par_cond - template_ser_lower) * width) + template_ser_lower + width - 1;
    if (par_cond_temp > template_ser_upper) {
      par_cond_temp = template_ser_upper;
    }

    if (par_cond_temp > ser_cond) {
      *par_cond = ser_cond;
    } else {
      *par_cond = par_cond_temp;
    }

    // init par_step
    // FIXME how implement???

    return _XMP_N_INT_TRUE;
  } else {
    return _XMP_N_INT_FALSE;
  }
}

// schedule by template -------------------------------------------------------------------------------------------------------------
// duplicate distribution -----------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_DUPLICATION(int ser_init, int ser_cond, int ser_step,
                                          int *par_init, int *par_cond, int *par_step,
                                          _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed); // FIXME too strict?

  if (!template->is_owner) {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
    return;
  }

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  if (_XMP_sched_loop_template_width_1(ser_init, ser_cond, ser_step, par_init, par_cond, par_step,
                                       template_lower, template_upper, template_stride)) {
    // finalize iter
    _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  } else {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
  }
}

// block distribution ---------------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_BLOCK(int ser_init, int ser_cond, int ser_step,
                                    int *par_init, int *par_cond, int *par_step,
                                    _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
    return;
  }

  _XMP_template_info_t *template_info = &(template->info[template_index]);
  int template_ser_lower = template_info->ser_lower;

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;
  int chunk_width = template_chunk->par_chunk_width;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  if (_XMP_sched_loop_template_width_1(ser_init, ser_cond, ser_step, par_init, par_cond, par_step,
                                       template_lower, template_upper, template_stride)) {
    *par_init = _XMP_SM_GTOL_BLOCK(*par_init, template_ser_lower, chunk_width);
    *par_cond = _XMP_SM_GTOL_BLOCK(*par_cond, template_ser_lower, chunk_width);
    *par_step = ser_step;

    // finalize iter
    _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  } else {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
  }
}

// cyclic distribution ---------------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                     int *par_init, int *par_cond, int *par_step,
                                     _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
    return;
  }

  _XMP_template_info_t *template_info = &(template->info[template_index]);
  int template_ser_lower = template_info->ser_lower;

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int nodes_size = (template_chunk->onto_nodes_info)->size;
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  if (_XMP_sched_loop_template_width_1(ser_init, ser_cond, ser_step, par_init, par_cond, par_step,
                                       template_lower, template_upper, template_stride)) {
    *par_init = _XMP_SM_GTOL_CYCLIC(*par_init, template_ser_lower, nodes_size);
    *par_cond = _XMP_SM_GTOL_CYCLIC(*par_cond, template_ser_lower, nodes_size);
    *par_step = 1;

    // finalize iter
    _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  } else {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
  }
}

// block-cyclic distribution ---------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_BLOCK_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                           int *par_init, int *par_cond, int *par_step,
                                           _XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
    return;
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
  if (_XMP_sched_loop_template_width_N(ser_init, ser_cond, ser_step, par_init, par_cond, par_step,
                                       template_lower, template_upper, template_stride,
                                       width, template_ser_lower, template_ser_upper)) {
    *par_init = _XMP_SM_GTOL_BLOCK_CYCLIC(width, *par_init, template_ser_lower, nodes_size);
    *par_cond = _XMP_SM_GTOL_BLOCK_CYCLIC(width, *par_cond, template_ser_lower, nodes_size);
    *par_step = 1;

    // finalize iter
    _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  } else {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
  }
}

// gblock distribution ---------------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_template_GBLOCK(int ser_init, int ser_cond, int ser_step,
				     int *par_init, int *par_cond, int *par_step,
				     _XMP_template_t *template, int template_index) {

  _XMP_ASSERT(template->is_distributed);

  if (!template->is_owner) {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
    return;
  }

  _XMP_template_chunk_t *template_chunk = &(template->chunk[template_index]);
  int rank = template_chunk->onto_nodes_info->rank;
  int template_lower = template_chunk->par_lower;
  int template_upper = template_chunk->par_upper;
  int template_stride = template_chunk->par_stride;
  long long *mapping_array = template_chunk->mapping_array;

  int reverse_iter = _XMP_N_INT_FALSE;
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  // calc par_init, par_cond, par_step
  if (_XMP_sched_loop_template_width_1(ser_init, ser_cond, ser_step, par_init, par_cond, par_step,
                                       template_lower, template_upper, template_stride)) {

    *par_init = _XMP_SM_GTOL_GBLOCK(*par_init, mapping_array[rank]);
    *par_cond = _XMP_SM_GTOL_GBLOCK(*par_cond, mapping_array[rank]);
    *par_step = ser_step;

    // finalize iter
    _XMP_SM_FINALIZE_ITER(par_init, par_cond, par_step, reverse_iter);
  } else {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
  }
}

// schedule by nodes ----------------------------------------------------------------------------------------------------------------
void _XMP_sched_loop_nodes(int ser_init, int ser_cond, int ser_step,
                           int *par_init, int *par_cond, int *par_step,
                           _XMP_nodes_t *nodes, int nodes_index) {
  if (!nodes->is_member){
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
    return;
  }

  int reverse_iter = _XMP_N_INT_TRUE;  // reverse_iter is not used in this function
  _XMP_SM_NORM_SCHED_PARAMS(ser_init, ser_cond, ser_step, reverse_iter)

  int rank1O = ((nodes->info[nodes_index].rank) + 1);
  if ((rank1O < ser_init) || (rank1O > ser_cond)) {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
    return;
  }

  if (((rank1O - ser_init) % ser_step) == 0) {
    *par_init = rank1O;
    *par_cond = rank1O + 1;
    *par_step = ser_step;
  } else {
    *par_init = 0;
    *par_cond = 0;
    *par_step = 1;
  }
}
