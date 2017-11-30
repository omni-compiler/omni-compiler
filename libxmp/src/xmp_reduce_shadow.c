#include "xmp_internal.h"

void _XMP_reflect_pcopy_sched_dim(_XMP_array_t *adesc, int target_dim, int lwidth, int uwidth, int is_periodic,
				  int is_reduce_shadow);

void _XMP_reflect_pack_dim(_XMP_array_t *a, int i, int *lwidth, int *uwidth, int *is_periodic, int is_reduce_shadow);
void _XMP_sum_vector(int type, char * restrict dst, char * restrict src,
		     int count, int blocklength, long stride);

int _XMP_get_owner_pos(_XMP_array_t *a, int dim, int index);

static void _XMP_reduce_shadow_wait_and_sum(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic);
static void _XMP_reduce_shadow_sum(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic);

static int _xmp_set_reduce_shadow_flag = 0;
static int _xmp_lwidth[_XMP_N_MAX_DIM] = {0};
static int _xmp_uwidth[_XMP_N_MAX_DIM] = {0};
static int _xmp_is_periodic[_XMP_N_MAX_DIM] = {0};


void _XMP_set_reduce_shadow__(_XMP_array_t *a, int dim, int lwidth, int uwidth,
			      int is_periodic)
{
  _xmp_set_reduce_shadow_flag = 1;
  _xmp_lwidth[dim] = lwidth;
  _xmp_uwidth[dim] = uwidth;
  _xmp_is_periodic[dim] = is_periodic;
}


void _XMP_reduce_shadow__(_XMP_array_t *a, int is_diagonal)
// is_diagnonal is not used now.
{

  _XMP_RETURN_IF_SINGLE;
  if (!a->is_allocated){
    _xmp_set_reduce_shadow_flag = 0;
    return;
  }

  if (!_xmp_set_reduce_shadow_flag){
    for (int i = 0; i < a->dim; i++){
      _XMP_array_info_t *ai = &(a->info[i]);
      _xmp_lwidth[i] = ai->shadow_size_lo;
      _xmp_uwidth[i] = ai->shadow_size_hi;
      _xmp_is_periodic[i] = 0;
    }
  }

  for (int i = 0; i < a->dim; i++){

    _XMP_array_info_t *ai = &(a->info[i]);

    if (ai->shadow_type == _XMP_N_SHADOW_NONE){
      continue;
    }
    else if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){

      _XMP_reflect_sched_t *shadow_sched = ai->reflect_sched;

      if (_xmp_lwidth[i] || _xmp_uwidth[i]){

	_XMP_ASSERT(shadow_sched);

	if (!shadow_sched->reduce_is_initialized ||
	    _xmp_lwidth[i] != shadow_sched->lo_width ||
	    _xmp_uwidth[i] != shadow_sched->hi_width ||
	    _xmp_is_periodic[i] != shadow_sched->is_periodic){

	  _XMP_reflect_pcopy_sched_dim(a, i, _xmp_lwidth[i], _xmp_uwidth[i], _xmp_is_periodic[i], 1);

	  shadow_sched->reduce_is_initialized = 1;
	  shadow_sched->lo_width = _xmp_lwidth[i];
	  shadow_sched->hi_width = _xmp_uwidth[i];
	  shadow_sched->is_periodic = _xmp_is_periodic[i];
	}
	
	_XMP_reflect_pack_dim(a, i, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic, 1);

	MPI_Startall(4, shadow_sched->req_reduce);

      }

    }
    else { /* _XMP_N_SHADOW_FULL */
      // not supported yet
      //_XMP_reduce_shadow_shadow_FULL(a->array_addr_p, a, i);
    }
    
  }

  _XMP_reduce_shadow_wait_and_sum(a, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic);

  _xmp_set_reduce_shadow_flag = 0;
  for (int i = 0; i < a->dim; i++){
    _xmp_lwidth[i] = 0;
    _xmp_uwidth[i] = 0;
    _xmp_is_periodic[i] = 0;
  }

}


/* void _XMP_reduce_shadow_async__(_XMP_array_t *a, int is_diagonal, int async_id){ */

/*   _XMP_RETURN_IF_SINGLE; */
/*   if (!a->is_allocated){ */
/*     _xmp_set_reduce_shadow_flag = 0; */
/*     return; */
/*   } */

/*   if (!_xmp_set_reduce_shadow_flag){ */
/*     for (int i = 0; i < a->dim; i++){ */
/*       _XMP_array_info_t *ai = &(a->info[i]); */
/*       if (ai->shadow_type == _XMP_N_SHADOW_FULL){ */
/* 	_XMP_fatal("asynchronous reduce_shadow for full shadow not supported."); */
/*       } */
/*       _xmp_lwidth[i] = ai->shadow_size_lo; */
/*       _xmp_uwidth[i] = ai->shadow_size_hi; */
/*       _xmp_is_periodic[i] = 0; */
/*     } */
/*   } */

/*   int reduce_shadow_ndims = 0; */
/*   for (int i = 0; i < a->dim; i++){ */
/*     if (_xmp_lwidth[i] || _xmp_uwidth[i]){ */
/*       reduce_shadow_ndims++; */
/*     } */
/*   } */

/*   if (reduce_shadow_ndims == 0){ */
/*     return; */
/*   } */

/*   // */
/*   // ... */
/*   // */
  
/*   _xmp_set_reduce_shadow_flag = 0; */
/*   for (int i = 0; i < a->dim; i++){ */
/*     _xmp_lwidth[i] = 0; */
/*     _xmp_uwidth[i] = 0; */
/*     _xmp_is_periodic[i] = 0; */
/*   } */

/* } */


static void _XMP_reduce_shadow_wait_and_sum(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic)
{
  for (int i = 0; i < a->dim; i++){

    if (!lwidth[i] && !uwidth[i]) continue;

    _XMP_array_info_t *ai = &(a->info[i]);

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      _XMP_reflect_sched_t *shadow_sched = ai->reflect_sched;
      MPI_Waitall(4, shadow_sched->req_reduce, MPI_STATUSES_IGNORE);
    }
    else if (ai->shadow_type == _XMP_N_SHADOW_FULL){
      ;
    }

  }

  _XMP_reduce_shadow_sum(a, lwidth, uwidth, is_periodic);

}


static void _XMP_reduce_shadow_sum(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic)
{
  int type_size = a->type_size;

  for (int i = 0; i < a->dim; i++){

    _XMP_array_info_t *ai = &(a->info[i]);
    _XMP_reflect_sched_t *shadow_sched = ai->reflect_sched;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){

      // 0-origin
      int my_pos = a->align_template->chunk[i].onto_nodes_info->rank;
      int lb_pos = _XMP_get_owner_pos(a, i, ai->ser_lower);
      int ub_pos = _XMP_get_owner_pos(a, i, ai->ser_upper);

      // for lower reduce_shadow
      if (lwidth[i] && (is_periodic[i] || my_pos != ub_pos)){
	_XMP_sum_vector(a->type,
			(char *)shadow_sched->lo_send_array,
			(char *)shadow_sched->lo_send_buf,
			shadow_sched->count, lwidth[i] * shadow_sched->blocklength / type_size,
			shadow_sched->stride / type_size);
      }

      // for upper reduce_shadow
      if (uwidth[i] && (is_periodic[i] || my_pos != lb_pos)){
	_XMP_sum_vector(a->type,
			(char *)shadow_sched->hi_send_array,
			(char *)shadow_sched->hi_send_buf,
			shadow_sched->count, uwidth[i] * shadow_sched->blocklength / type_size,
			shadow_sched->stride / type_size);
      }

    }

  }

}
