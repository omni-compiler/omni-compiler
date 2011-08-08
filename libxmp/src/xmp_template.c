/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include "mpi.h"
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

static void _XMP_calc_template_size(_XMP_template_t *t) {
  int dim;
  if (t->is_fixed) {
    dim = t->dim;
  }
  else {
    dim = t->dim - 1;
  }

  for (int i = 0; i < dim; i++) {
    int ser_lower = t->info[i].ser_lower;
    int ser_upper = t->info[i].ser_upper;

    if (ser_lower > ser_upper) {
      _XMP_fatal("the lower bound of template should be less than or equal to the upper bound");
    }

    t->info[i].ser_size = _XMP_M_COUNTi(ser_lower, ser_upper);
  }
}

static void _XMP_validate_template_ref(long long *lower, long long *upper, long long *stride,
                                       long long lb, long long ub) {
  // setup temporary variables
  long long l, u, s = *stride;
  if (s > 0) {
    l = *lower;
    u = *upper;
  }
  else if (s < 0) {
    l = *upper;
    u = *lower;
  }
  else {
    _XMP_fatal("the stride of <template-ref> is 0");
    l = 0; u = 0; // XXX dummy
  }

  // check boundary
  if (lb > l) {
    _XMP_fatal("<template-ref> is out of bounds, <ref-lower> is less than the template lower bound");
  }

  if (l > u) {
    _XMP_fatal("<template-ref> is out of bounds, <ref-upper> is less than <ref-lower>");
  }

  if (u > ub) {
    _XMP_fatal("<template-ref> is out of bounds, <ref-upper> is greater than the template upper bound");
  }

  // validate values
  if (s > 0) {
    u = u - ((u - l) % s);
    *upper = u;
  }
  else {
    s = -s;
    l = l + ((u - l) % s);
    *lower = l;
    *upper = u;
    *stride = s;
  }
}

static _Bool _XMP_check_template_ref_inclusion(long long ref_lower, long long ref_upper, long long ref_stride,
                                               _XMP_template_t *t, int index) {
  _XMP_template_info_t *info = &(t->info[index]);
  _XMP_template_chunk_t *chunk = &(t->chunk[index]);

  _XMP_validate_template_ref(&ref_lower, &ref_upper, &ref_stride, info->ser_lower, info->ser_upper);

  switch (chunk->dist_manner) {
    case _XMP_N_DIST_DUPLICATION:
      return true;
    case _XMP_N_DIST_BLOCK:
      {
        long long template_lower = chunk->par_lower;
        long long template_upper = chunk->par_upper;

        if (ref_stride != 1) {
          int ref_stride_mod = _XMP_modi_ll_i(ref_lower, ref_stride);
          /* normalize template lower */
          int template_lower_mod = _XMP_modi_ll_i(template_lower, ref_stride);
          if (template_lower_mod != ref_stride_mod) {
            if (template_lower_mod < ref_stride_mod) {
              template_lower += (ref_stride_mod - template_lower_mod);
            }
            else {
              template_lower += (ref_stride - template_lower_mod + ref_stride_mod);
            }
          }

          if (template_lower > template_upper) {
            return false;
          }

          /* normalize template upper */
          int template_upper_mod = _XMP_modi_ll_i(template_upper, ref_stride);
          if (template_upper_mod != ref_stride_mod) {
            if (ref_stride_mod < template_upper_mod) {
              template_upper -= (template_upper_mod - ref_stride_mod);
            }
            else {
              template_upper -= (ref_stride - ref_stride_mod + template_upper_mod);
            }
          }
        }

        if (ref_upper < template_lower) {
          return false;
        }

        if (template_upper < ref_lower) {
          return false;
        }

        return true;
      }
    case _XMP_N_DIST_CYCLIC:
      {
        if (ref_stride == 1) {
          int nodes_size = (chunk->onto_nodes_info)->size;
          int par_lower_mod = _XMP_modi_ll_i(chunk->par_lower, nodes_size);
          int ref_lower_mod = _XMP_modi_ll_i(ref_lower, nodes_size);
          if (par_lower_mod != ref_lower_mod) {
            if (ref_lower_mod < par_lower_mod) ref_lower += (par_lower_mod - ref_lower_mod);
            else ref_lower += (nodes_size - ref_lower_mod + par_lower_mod);
          }

          if (ref_upper < ref_lower) {
            return false;
          }
          else {
            return true;
          }
        }
        else {
          _XMP_fatal("not implemented condition: (stride is not 1 && cyclic distribution)");
          return false; // XXX dummy
        }
      }
    default:
      _XMP_fatal("unknown distribution manner");
      return false; // XXX dummy
  }
}

static void _XMP_set_task_desc(_XMP_task_desc_t *desc, int execute, _XMP_nodes_t *n,
                               int dim, long long *lower, long long *upper, long long *stride) {
  desc->inherit_nodes_id = _XMP_get_execution_nodes()->nodes_id;
  desc->execute = execute;
  desc->nodes = n;
  desc->dim = dim;
  for (int i = 0; i < dim; i++) {
    desc->lower[i] = lower[i];
    desc->upper[i] = upper[i];
    desc->stride[i] = stride[i];
  }
}

static int _XMP_compare_task_exec_cond(_XMP_task_desc_t *task_desc, long long *lower, long long *upper, long long *stride) {
  int dim = task_desc->dim;

  if ((_XMP_get_execution_nodes()->nodes_id) != (task_desc->inherit_nodes_id)) {
    return _XMP_N_INT_FALSE;
  }

  for (int i = 0; i < dim; i++) {
    if (((int)(task_desc->lower[i]) != lower[i]) || (int)((task_desc->upper[i]) != upper[i]) ||
        ((int)(task_desc->stride[i]) != stride[i])) {
      return _XMP_N_INT_FALSE;
    }
  }

  return _XMP_N_INT_TRUE;
}

void _XMP_init_template_FIXED(_XMP_template_t **template, int dim, ...) {
  // alloc descriptor
  _XMP_template_t *t = _XMP_alloc(sizeof(_XMP_template_t) +
                                                sizeof(_XMP_template_info_t) * (dim - 1));

  // calc members
  t->is_fixed = true;
  t->is_distributed = true;
  t->is_owner = false;
  t->dim = dim;

  t->onto_nodes = NULL;
  t->chunk = NULL;

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    t->info[i].ser_lower = va_arg(args, long long);
    t->info[i].ser_upper = va_arg(args, long long);
  }
  va_end(args);

  _XMP_calc_template_size(t);

  *template = t;
}

void _XMP_init_template_UNFIXED(_XMP_template_t **template, int dim, ...) {
  // alloc descriptor
  _XMP_template_t *t = _XMP_alloc(sizeof(_XMP_template_t) +
                                                sizeof(_XMP_template_info_t) * (dim - 1));

  // calc members
  t->is_fixed = false;
  t->is_distributed = false;
  t->is_owner = false;

  t->dim = dim;

  t->onto_nodes = NULL;
  t->chunk = NULL;

  va_list args;
  va_start(args, dim);
  for(int i = 0; i < dim - 1; i++) {
    t->info[i].ser_lower = va_arg(args, long long);
    t->info[i].ser_upper = va_arg(args, long long);
  }
  va_end(args);

  _XMP_calc_template_size(t);

  *template = t;
}

void _XMP_init_template_chunk(_XMP_template_t *template, _XMP_nodes_t *nodes) {
  template->is_distributed = true;
  template->is_owner = nodes->is_member;

  template->onto_nodes = nodes;
  template->chunk = _XMP_alloc(sizeof(_XMP_template_chunk_t) * (template->dim));
}

void _XMP_finalize_template(_XMP_template_t *template) {
  if (template->is_distributed) {
    _XMP_free(template->chunk);
  }

  _XMP_free(template);
}

void _XMP_dist_template_DUPLICATION(_XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_template_info_t *ti = &(template->info[template_index]);

  chunk->par_lower = ti->ser_lower;
  chunk->par_upper = ti->ser_upper;

  chunk->par_stride = 1;
  chunk->par_chunk_width = ti->ser_size;
  chunk->dist_manner = _XMP_N_DIST_DUPLICATION;
  chunk->is_regular_chunk = true;

  chunk->onto_nodes_index = _XMP_N_NO_ONTO_NODES;
  chunk->onto_nodes_info = NULL;
}

void _XMP_dist_template_BLOCK(_XMP_template_t *template, int template_index, int nodes_index) {
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_nodes_t *nodes = template->onto_nodes;

  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_nodes_info_t *ni = &(nodes->info[nodes_index]);

  long long nodes_size = (long long)ni->size;

  // calc parallel members
  unsigned long long chunk_width = _XMP_M_CEILi(ti->ser_size, nodes_size);

  if (nodes->is_member) {
    long long nodes_rank = (long long)ni->rank;
    int owner_nodes_size = _XMP_M_CEILi(ti->ser_size, chunk_width);

    chunk->par_lower = nodes_rank * chunk_width + ti->ser_lower;
    if (nodes_rank == (owner_nodes_size - 1)) {
      chunk->par_upper = ti->ser_upper;
    }
    else if (nodes_rank >= owner_nodes_size) {
      template->is_owner = false;
    }
    else {
      chunk->par_upper = chunk->par_lower + chunk_width - 1;
    }
  }

  chunk->par_stride = 1;
  chunk->par_chunk_width = chunk_width;
  chunk->dist_manner = _XMP_N_DIST_BLOCK;
  if ((ti->ser_size % nodes_size) == 0) {
    chunk->is_regular_chunk = true;
  }
  else {
    chunk->is_regular_chunk = false;
  }

  chunk->onto_nodes_index = nodes_index;
  chunk->onto_nodes_info = ni;
}

void _XMP_dist_template_CYCLIC(_XMP_template_t *template, int template_index, int nodes_index) {
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_nodes_t *nodes = template->onto_nodes;

  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_nodes_info_t *ni = &(nodes->info[nodes_index]);

  long long nodes_size = (long long)ni->size;

  // calc parallel members
  if (nodes->is_member) {
    long long nodes_rank = (long long)ni->rank;

    if (ti->ser_size < nodes_size) {
      if (nodes_rank < ti->ser_size) {
        long long par_index = ti->ser_lower + nodes_rank;

        chunk->par_lower = par_index;
        chunk->par_upper = par_index;
      }
      else {
        template->is_owner = false;
      }
    }
    else {
      unsigned long long div = ti->ser_size / nodes_size;
      unsigned long long mod = ti->ser_size % nodes_size;
      unsigned long long par_size = 0;
      if(mod == 0) {
        par_size = div;
      }
      else {
        if(nodes_rank >= mod) {
          par_size = div;
        }
        else {
          par_size = div + 1;
        }
      }

      chunk->par_lower = ti->ser_lower + nodes_rank;
      chunk->par_upper = chunk->par_lower + nodes_size * (par_size - 1);
    }
  }

  chunk->par_stride = nodes_size;
  chunk->par_chunk_width = _XMP_M_CEILi(ti->ser_size, nodes_size);
  chunk->dist_manner = _XMP_N_DIST_CYCLIC;
  if ((ti->ser_size % nodes_size) == 0) {
    chunk->is_regular_chunk = true;
  }
  else {
    chunk->is_regular_chunk = false;
  }

  chunk->onto_nodes_index = nodes_index;
  chunk->onto_nodes_info = ni;
}

int _XMP_exec_task_TEMPLATE_PART(_XMP_task_desc_t **task_desc, int get_upper, _XMP_template_t *ref_template, ...) {
  int shrink[_XMP_N_MAX_DIM];
  long long lower[_XMP_N_MAX_DIM], upper[_XMP_N_MAX_DIM], stride[_XMP_N_MAX_DIM];
  va_list args;
  va_start(args, ref_template);
  int ref_dim = ref_template->dim;
  for (int i = 0; i < ref_dim; i++) {
    _XMP_template_info_t *info = &(ref_template->info[i]);

    shrink[i] = va_arg(args, int);
    if (shrink[i] == 1) {
      lower[i] = info->ser_lower;
      upper[i] = info->ser_upper;
      stride[i] = 1;
    } else {
      lower[i] = va_arg(args, long long);
      if ((i == (ref_dim - 1)) && (get_upper == 1)) {
        upper[i] = info->ser_upper;
      } else {
        upper[i] = va_arg(args, long long);
      }
      stride[i] = va_arg(args, long long);
    }
  }
  va_end(args);

  _XMP_task_desc_t *desc = NULL;
  if (*task_desc == NULL) {
    desc = (_XMP_task_desc_t *)_XMP_alloc(sizeof(_XMP_task_desc_t));
    *task_desc = desc;
  } else {
    desc = *task_desc;
    if (_XMP_compare_task_exec_cond(desc, lower, upper, stride)) {
      if (desc->execute) {
        _XMP_push_nodes(desc->nodes);
        return _XMP_N_INT_TRUE;
      } else {
        return _XMP_N_INT_FALSE;
      }
    } else {
      if (desc->nodes != NULL) {
        _XMP_finalize_nodes(desc->nodes);
      }
    }
  }

  _XMP_ASSERT(ref_template->is_fixed);
  _XMP_ASSERT(ref_template->is_distributed);

  _XMP_nodes_t *onto_nodes = ref_template->onto_nodes;
  if (!onto_nodes->is_member) {
     _XMP_set_task_desc(desc, _XMP_N_INT_FALSE, NULL, ref_dim, lower, upper, stride);
     return _XMP_N_INT_FALSE;
  }

  _Bool is_member = true;
  int color = 1;
  if (!ref_template->is_owner) {
    is_member = false;
  } else {
    int acc_nodes_size = 1;
    long long ref_lower, ref_upper, ref_stride;
    for (int i = 0; i < ref_dim; i++) {
      _XMP_template_chunk_t *chunk = &(ref_template->chunk[i]);

      int size, rank;
      if (chunk->dist_manner == _XMP_N_DIST_DUPLICATION) {
        size = 1;
        rank = 0;
      } else {
        _XMP_nodes_info_t *onto_nodes_info = chunk->onto_nodes_info;
        size = onto_nodes_info->size;
        rank = onto_nodes_info->rank;
      }

      if (shrink[i] == 1) {
        color += (acc_nodes_size * rank);
      } else {
        ref_lower = lower[i];
        ref_upper = upper[i];
        ref_stride = stride[i];

        is_member = is_member && _XMP_check_template_ref_inclusion(ref_lower, ref_upper, ref_stride, ref_template, i);
      }

      acc_nodes_size *= size;
    }
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  if (!is_member) {
    color = 0;
  }

  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, _XMP_world_rank, comm);

  if (is_member) {
    _XMP_nodes_t *n = _XMP_create_nodes_by_comm(comm);
    _XMP_set_task_desc(desc, _XMP_N_INT_TRUE, n, ref_dim, lower, upper, stride);
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  } else {
    _XMP_set_task_desc(desc, _XMP_N_INT_FALSE, NULL, ref_dim, lower, upper, stride);
    _XMP_finalize_comm(comm);
    return _XMP_N_INT_FALSE;
  }
}
