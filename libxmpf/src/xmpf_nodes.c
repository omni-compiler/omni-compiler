#include "xmpf_internal.h"

static int _xmpf_nodes_n_dim;
static int _xmpf_nodes_dim_size[_XMP_N_MAX_DIM];
static int _xmpf_nodes_shrink[_XMP_N_MAX_DIM];
static int _xmpf_nodes_ref_lower[_XMP_N_MAX_DIM];
static int _xmpf_nodes_ref_upper[_XMP_N_MAX_DIM];
static int _xmpf_nodes_ref_stride[_XMP_N_MAX_DIM];

/* allocate nodes */
void xmpf_nodes_alloc__(_XMP_nodes_t **n_desc, int *n_dim)
{
  //printf("[%d] xmpf_nodes_alloc...\n",_XMP_world_rank);
  /* do nothing for n_desc */
  _xmpf_nodes_n_dim = *n_dim;
}


/* deallocate nodes */
void xmpf_nodes_dealloc__(_XMP_nodes_t **n_desc)
{
  _XMP_finalize_nodes(*n_desc);
}


void xmpf_nodes_dim_size__(_XMP_nodes_t **n_desc, int *i_dim, int *size)
{
  /* do nothing for n_desc */
  if(*i_dim >= _XMP_N_MAX_DIM) 
    _XMP_fatal("nodes dimesion should not greater than _XMP_N_MAX_DIM");
  _xmpf_nodes_dim_size[*i_dim] = *size;
}


void xmpf_nodes_dim_triplet__(_XMP_nodes_t *n_desc, int *i_dim, int *shrink, 
			      int *ref_lower, int *ref_upper, int *ref_stride)
{
  /* do nothing for n_desc */
  if(*i_dim >= _XMP_N_MAX_DIM) 
    _XMP_fatal("nodes dimesion should not greater than _XMP_N_MAX_DIM");
  _xmpf_nodes_shrink[*i_dim] = *shrink;
  _xmpf_nodes_ref_lower[*i_dim] = *ref_lower;
  _xmpf_nodes_ref_upper[*i_dim] = *ref_upper;
  _xmpf_nodes_ref_stride[*i_dim] = *ref_stride;
}


void xmpf_nodes_init_global__(_XMP_nodes_t **n_desc)
{
  int is_static = (_xmpf_nodes_dim_size[_xmpf_nodes_n_dim - 1] != -1);
  
  *n_desc = _XMP_init_nodes_struct_GLOBAL(_xmpf_nodes_n_dim,
					  _xmpf_nodes_dim_size,is_static);
}


void xmpf_nodes_init_exec__(_XMP_nodes_t **n_desc)
{
  int is_static = (_xmpf_nodes_dim_size[_xmpf_nodes_n_dim - 1] != -1);

  *n_desc = _XMP_init_nodes_struct_EXEC(_xmpf_nodes_n_dim,
					_xmpf_nodes_dim_size,is_static);
}


void xmpf_nodes_init_number__(_XMP_nodes_t **n_desc, int *ref_lower, 
			      int *ref_upper, int *ref_stride)
{
  int is_static = (_xmpf_nodes_dim_size[_xmpf_nodes_n_dim - 1] != -1);

  *n_desc = _XMP_init_nodes_struct_NODES_NUMBER(_xmpf_nodes_n_dim,
						*ref_lower,*ref_upper,*ref_stride,
						_xmpf_nodes_dim_size,is_static);
}


void xmpf_nodes_init_named__(_XMP_nodes_t **n_desc, _XMP_nodes_t **ref_node)
{
  int is_static = (_xmpf_nodes_dim_size[_xmpf_nodes_n_dim - 1] != -1);

  *n_desc = _XMP_init_nodes_struct_NODES_NAMED(_xmpf_nodes_n_dim,
					       *ref_node,
					       _xmpf_nodes_shrink,
					       _xmpf_nodes_ref_lower,
					       _xmpf_nodes_ref_upper,
					       _xmpf_nodes_ref_stride,
					       _xmpf_nodes_dim_size,is_static);
}


void xmpf_nodes_free__(_XMP_nodes_t **n_desc)
{
  _XMP_finalize_nodes(*n_desc);
}
