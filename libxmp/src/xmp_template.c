#include <stdarg.h>
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_macro.h"

static void _XCALABLEMP_calc_template_size(_XCALABLEMP_template_t *t, int dim);
static void _XCALABLEMP_validate_template_ref(long long *lower, long long *upper, long long *stride,
                                              long long lb, long long ub);

static void _XCALABLEMP_calc_template_size(_XCALABLEMP_template_t *t, int dim) {
  if (t == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  for (int i = 0; i < dim; i++) {
    int ser_lower = t->info[i].ser_lower;
    int ser_upper = t->info[i].ser_upper;

    if (ser_lower > ser_upper)
      _XCALABLEMP_fatal("the lower bound of template should be less than or equal to the upper bound");

    t->info[i].ser_size = _XCALABLEMP_M_COUNTi(ser_lower, ser_upper);
  }
}

static void _XCALABLEMP_validate_template_ref(long long *lower, long long *upper, long long *stride,
                                              long long lb, long long ub) {
  // XXX node number is 1-origin in this function

  // setup temporary variables
  long long l, u;
  unsigned long long s = *(stride);
  if (s > 0) {
    l = *lower;
    u = *upper;
  }
  else if (s < 0) {
    l = *upper;
    u = *lower;
  }
  else _XCALABLEMP_fatal("the stride of <template-ref> is 0");

  // check boundary
  if (lb > l) _XCALABLEMP_fatal("<template-ref> is out of bounds, <ref-lower> is less than the template lower bound");
  if (l > u) _XCALABLEMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is less than <ref-lower>");
  if (u > ub) _XCALABLEMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is greater than the template upper bound");

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

void _XCALABLEMP_init_template_FIXED(_XCALABLEMP_template_t **template, int dim, ...) {
  // alloc descriptor
  _XCALABLEMP_template_t *t = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_template_t) +
                                                sizeof(_XCALABLEMP_template_info_t) * (dim - 1));

  // calc members
  t->is_fixed = true;
  t->dim = dim;
  t->chunk = NULL;

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    t->info[i].ser_lower = va_arg(args, long long);
    t->info[i].ser_upper = va_arg(args, long long);
  }
  va_end(args);

  _XCALABLEMP_calc_template_size(t, dim);

  *template = t;
}

void _XCALABLEMP_init_template_UNFIXED(_XCALABLEMP_template_t **template, int dim, ...) {
  // alloc descriptor
  _XCALABLEMP_template_t *t = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_template_t) +
                                                sizeof(_XCALABLEMP_template_info_t) * (dim - 1));

  // calc members
  t->is_fixed = false;
  t->dim = dim;
  t->chunk = NULL;

  va_list args;
  va_start(args, dim);
  for(int i = 0; i < dim - 1; i++) {
    t->info[i].ser_lower = va_arg(args, long long);
    t->info[i].ser_upper = va_arg(args, long long);
  }
  va_end(args);

  _XCALABLEMP_calc_template_size(t, dim - 1);

  *template = t;
}

void _XCALABLEMP_init_template_chunk(_XCALABLEMP_template_t *template, _XCALABLEMP_nodes_t *nodes) {
  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (nodes != NULL)
    template->chunk = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_template_chunk_t) * (template->dim));
}

void _XCALABLEMP_dist_template_DUPLICATION(_XCALABLEMP_template_t *template, int template_index,
                                           _XCALABLEMP_nodes_t *nodes,       int nodes_index) {
  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (template->chunk == NULL) return;

  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);

  chunk->par_lower = ti->ser_lower;
  chunk->par_upper = ti->ser_upper;
  chunk->par_stride = 1;

  chunk->par_size = ti->ser_size;
  chunk->par_chunk_width = ti->ser_size;

  chunk->dist_manner = _XCALABLEMP_N_DIST_DUPLICATION;
  chunk->onto_nodes_info = NULL;
}

void _XCALABLEMP_dist_template_BLOCK(_XCALABLEMP_template_t *template, int template_index,
                                     _XCALABLEMP_nodes_t *nodes,       int nodes_index) {
  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (nodes == NULL)
    _XCALABLEMP_fatal("null nodes descriptor detected");

  if (template->chunk == NULL) return;

  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);
  _XCALABLEMP_nodes_info_t *ni = &(nodes->info[nodes_index]);

  long long nodes_rank = (long long)ni->rank;
  long long nodes_size = (long long)ni->size;

  // check template size
  if (ti->ser_size < nodes_size) _XCALABLEMP_fatal("template is too small to distribute");

  // calc parallel members
  unsigned long long chunk_width = _XCALABLEMP_M_CEILi(ti->ser_size, nodes_size);

  chunk->par_lower = nodes_rank * chunk_width + ti->ser_lower;
  if (nodes_rank == (nodes_size - 1)) chunk->par_upper = ti->ser_upper;
  else chunk->par_upper = chunk->par_lower + chunk_width - 1;
  chunk->par_stride = 1;

  chunk->par_size = _XCALABLEMP_M_COUNTi(chunk->par_lower, chunk->par_upper);
  chunk->par_chunk_width = chunk_width;

  chunk->dist_manner = _XCALABLEMP_N_DIST_BLOCK;
  chunk->onto_nodes_info = ni;
}

void _XCALABLEMP_dist_template_CYCLIC(_XCALABLEMP_template_t *template, int template_index,
                                      _XCALABLEMP_nodes_t *nodes,       int nodes_index) {
  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (nodes == NULL)
    _XCALABLEMP_fatal("null nodes descriptor detected");

  if (template->chunk == NULL) return;

  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);
  _XCALABLEMP_nodes_info_t *ni = &(nodes->info[nodes_index]);

  long long nodes_rank = (long long)ni->rank;
  long long nodes_size = (long long)ni->size;

  // check template size
  if (ti->ser_size < nodes_size) _XCALABLEMP_fatal("template is too small to distribute");

  // calc parallel members
  unsigned long long div = ti->ser_size / nodes_size;
  unsigned long long mod = ti->ser_size % nodes_size;
  unsigned long long par_size = 0;
  if(mod == 0) par_size = div;
  else {
    if(nodes_rank >= mod) par_size = div;
    else par_size = div + 1;
  }

  chunk->par_lower = ti->ser_lower + nodes_rank;
  chunk->par_upper = chunk->par_lower + nodes_size * (par_size - 1);
  chunk->par_stride = nodes_size;

  chunk->par_size = par_size;
  chunk->par_chunk_width = _XCALABLEMP_M_CEILi(ti->ser_size, nodes_size);

  chunk->dist_manner = _XCALABLEMP_N_DIST_CYCLIC;
  chunk->onto_nodes_info = ni;
}

void _XCALABLEMP_finalize_template(_XCALABLEMP_template_t *template) {
  if (template != NULL) {
    _XCALABLEMP_free(template->chunk);
    _XCALABLEMP_free(template);
  }
}
