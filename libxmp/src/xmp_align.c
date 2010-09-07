#include <stdarg.h>
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_macro.h"

void _XCALABLEMP_init_array_desc(_XCALABLEMP_array_t **array, _XCALABLEMP_template_t *template, int dim, ...) {
  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (template->chunk == NULL) {
    *array = NULL;
    return;
  }

  _XCALABLEMP_array_t *a = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_array_t) + sizeof(_XCALABLEMP_array_info_t) * (dim - 1));

  a->dim = dim;
  a->align_template = template;

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int lower = 0;
    int size = va_arg(args, int);
    if (size <= 0) _XCALABLEMP_fatal("array size is less or equal to zero");

    int upper = size - 1;

    _XCALABLEMP_array_info_t *ai = &(a->info[i]);

    ai->ser_lower = lower;
    ai->ser_upper = upper;
    ai->ser_size = size;

    ai->par_lower = ai->ser_lower;
    ai->par_upper = ai->ser_upper;
    ai->par_stride = 1;
    ai->par_size = ai->ser_size;
    
    ai->align_subscript = 0;

    ai->shadow_type = _XCALABLEMP_N_SHADOW_NONE;
    ai->shadow_size_lo  = 0;
    ai->shadow_size_hi  = 0;

    ai->align_template_dim = -1;
    ai->align_template_info = NULL;
    ai->align_template_chunk = NULL;
  }
  va_end(args);

  *array = a;
}

void _XCALABLEMP_finalize_array_desc(_XCALABLEMP_array_t **array) {
  _XCALABLEMP_free(*array);
  *array = NULL;
}

void _XCALABLEMP_align_array_DUPLICATION(_XCALABLEMP_array_t **array, int array_index,
                                         _XCALABLEMP_template_t *template, int template_index,
                                         long long align_subscript) {
  if (*array == NULL) return;

  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (template->chunk == NULL) return;

  _XCALABLEMP_array_info_t *ai = &((*array)->info[array_index]);
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);

  ai->align_template_dim = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = &(template->chunk[template_index]);

  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) ||
       (align_upper > ti->ser_upper)))
    _XCALABLEMP_fatal("aligned array is out of template bound");

  // par_* are not modified here
  ai->align_subscript = align_subscript;
}

void _XCALABLEMP_align_array_BLOCK(_XCALABLEMP_array_t **array, int array_index,
                                   _XCALABLEMP_template_t *template, int template_index,
                                   long long align_subscript, int *temp0) {
  if (*array == NULL) return;

  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (template->chunk == NULL) return;

  _XCALABLEMP_array_info_t *ai = &((*array)->info[array_index]);
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);
  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[template_index]);

  ai->align_template_dim = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = chunk;

  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) ||
      (align_upper > ti->ser_upper)))
    _XCALABLEMP_fatal("aligned array is out of template bound");

  long long template_lower = chunk->par_lower;
  long long template_upper = chunk->par_upper;

  // set par_lower
  if (align_lower < template_lower)
    ai->par_lower = template_lower - align_subscript;
  else if (template_upper < align_lower) {
    _XCALABLEMP_finalize_array_desc(array);
    return;
  }
  else
    ai->par_lower = ai->ser_lower;

  // set par_upper
  if (align_upper < template_lower) {
    _XCALABLEMP_finalize_array_desc(array);
    return;
  }
  else if (template_upper < align_upper)
    ai->par_upper = template_upper - align_subscript;
  else
    ai->par_upper = ai->ser_upper;

  // set par_stride, par_size
  ai->par_stride = 1;
  ai->par_size   = _XCALABLEMP_M_COUNT_TRIPLETi(ai->par_lower,
                                                ai->par_upper, 1);

  ai->align_subscript = align_subscript;

  *temp0 = ai->par_lower;
}

void _XCALABLEMP_align_array_CYCLIC(_XCALABLEMP_array_t **array, int array_index,
                                    _XCALABLEMP_template_t *template, int template_index,
                                    long long align_subscript, int *temp0) {
  if (*array == NULL) return;

  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (template->chunk == NULL) return;

  _XCALABLEMP_array_info_t *ai = &((*array)->info[array_index]);
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);
  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[template_index]);

  ai->align_template_dim = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = chunk;

  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;

  if (((align_lower < ti->ser_lower) ||
       (align_upper > ti->ser_upper)))
    _XCALABLEMP_fatal("aligned array is out of template bound");

  int cycle = chunk->onto_nodes_info->size;
  int mod = (chunk->par_lower - align_subscript) % cycle;
  if (mod < 0) mod += cycle;

  int dist = (ai->ser_upper - mod) / cycle;

  ai->par_lower  = mod;
  ai->par_upper  = mod + (dist * cycle);
  ai->par_size   = dist + 1;

  ai->align_subscript = align_subscript;

  *temp0 = cycle;
}

void _XCALABLEMP_alloc_array(void **array_addr, _XCALABLEMP_array_t *array_desc, int datatype_size, ...) {
  if (array_desc == NULL) {
    *array_addr = NULL;
    return;
  }

  unsigned long long total_elmts = 1;
  int dim = array_desc->dim;
  va_list args;
  va_start(args, datatype_size);
  for (int i = dim - 1; i >= 0; i--) {
    unsigned long long *acc = va_arg(args, unsigned long long *);
    *acc = total_elmts;

    _XCALABLEMP_array_info_t *ai = &(array_desc->info[i]);

    unsigned long long elmts = ai->par_size;
    switch (ai->shadow_type) {
      case _XCALABLEMP_N_SHADOW_NONE:
        total_elmts *= elmts;
        break;
      case _XCALABLEMP_N_SHADOW_NORMAL:
        // FIXME support other distribute manners
        total_elmts *= (elmts + ai->shadow_size_lo + ai->shadow_size_hi);
        break;
      case _XCALABLEMP_N_SHADOW_FULL:
        total_elmts *= ai->ser_size;
        break;
      default:
        _XCALABLEMP_fatal("unknown shadow type");
    }
  }
  va_end(args);

  *array_addr = _XCALABLEMP_alloc(total_elmts * datatype_size);
}

void _XCALABLEMP_init_array_addr(void **array_addr, void *param_addr,
                                 _XCALABLEMP_array_t *array_desc, ...) {
  if (array_desc == NULL) {
    *array_addr = NULL;
    return;
  }

  unsigned long long total_elmts = 1;
  int dim = array_desc->dim;
  va_list args;
  va_start(args, array_desc);
  for (int i = dim - 1; i >= 0; i--) {
    unsigned long long *acc = va_arg(args, unsigned long long *);
    *acc = total_elmts;

    _XCALABLEMP_array_info_t *ai = &(array_desc->info[i]);

    unsigned long long elmts = ai->par_size;
    switch (ai->shadow_type) {
      case _XCALABLEMP_N_SHADOW_NONE:
        total_elmts *= elmts;
        break;
      case _XCALABLEMP_N_SHADOW_NORMAL:
        // FIXME support other distribute manners
        total_elmts *= (elmts + ai->shadow_size_lo + ai->shadow_size_hi);
        break;
      case _XCALABLEMP_N_SHADOW_FULL:
        total_elmts *= ai->ser_size;
        break;
      default:
        _XCALABLEMP_fatal("unknown shadow type");
    }
  }
  va_end(args);

  *array_addr = param_addr;
}
