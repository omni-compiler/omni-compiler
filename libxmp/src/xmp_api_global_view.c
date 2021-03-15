#include <stdarg.h>
#include "xmp_api.h"
#include "xmp_internal.h"

extern void _XMP_setup_reduce_type(MPI_Datatype *mpi_datatype, size_t *datatype_size, int datatype);

/* allocate global node set */
xmp_desc_t xmp_global_nodes(int n_dims, int dim_size[], int is_static)
{
  _XMP_nodes_t *np;
  np = _XMP_init_nodes_struct_GLOBAL(n_dims,dim_size,is_static);
  return (xmp_desc_t)np;
}

void xmp_global_nodes_(xmp_desc_t *d, int *n_dims, int *dim_size, int *is_static)
{
  _XMP_nodes_t *np;
  np = _XMP_init_nodes_struct_GLOBAL(*n_dims,dim_size,*is_static);
  *d = (xmp_desc_t)np;
}


#ifdef not
/* ask whether */
int xmp_is_on_nodes(xmp_nodes_t *np, xmp_range_t *rp)
{
  _XMP_fatal("xmp_is_on_nodes: not implemented yet");
}
#endif

/* #pragma xmp template t[N2][N1] */
/*  _XMP_init_template_FIXED(&(_XMP_DESC_t), 2, (long long)(0), (long long)(63), (long long)(0), (long long)(63)); */
/*  _XMP_init_template_chunk(_XMP_DESC_t, _XMP_DESC_p); */
// create new template
xmp_desc_t xmpc_new_template(xmp_desc_t n, int n_dims, long long dim1, ...)
{
  int i;
  _XMP_template_t *t = _XMP_create_template_desc(n_dims, TRUE);

  t->info[0].ser_lower = 0;
  t->info[0].ser_upper = dim1 - 1;
  // calc info
  va_list args;
  va_start(args, dim1);
  for(i = 1;i < n_dims; i++){
    t->info[i].ser_lower = 0;
    t->info[i].ser_upper = va_arg(args, long long) - 1;
  }
  va_end(args);

  _XMP_calc_template_size(t);
  _XMP_init_template_chunk(t,(_XMP_nodes_t *) n); 
  return (xmp_desc_t)t;
}

void xmp_new_template_(xmp_desc_t *d, xmp_desc_t *n_p, int *n_dims_p,
		       long long *dim_lb, long long *dim_ub)
{
  int i;
  int n_dims = *n_dims_p;
  _XMP_template_t *t = _XMP_create_template_desc(n_dims, TRUE);

  for(i = 0;i < n_dims; i++){
    t->info[i].ser_lower = dim_lb[i];
    t->info[i].ser_upper = dim_ub[i];
  }

  _XMP_calc_template_size(t);
  _XMP_init_template_chunk(t,(_XMP_nodes_t *) *n_p);
  *d = (xmp_desc_t)t;
}

/* #pragma xmp distribute t[block][block] onto p */
/* _XMP_dist_template_BLOCK(_XMP_DESC_t, 0, 0); */
/*  _XMP_dist_template_BLOCK(_XMP_DESC_t, 1, 1); */

int xmp_dist_template_BLOCK(xmp_desc_t t, int template_dim_idx, int node_dim_idx)
{
  _XMP_dist_template_BLOCK((_XMP_template_t *)t, template_dim_idx, node_dim_idx);
  return XMP_SUCCESS;
}

void xmp_dist_template_block_(xmp_desc_t *t, int *template_dim_idx, int *node_dim_idx, int *status)
{
  if(*template_dim_idx <= 0 || *node_dim_idx <= 0){
    *status = XMP_ERROR;
    return;
  }
  _XMP_dist_template_BLOCK((_XMP_template_t *)*t, *template_dim_idx-1, *node_dim_idx-1);
  *status = XMP_SUCCESS;
}

extern void _XMP_dist_template_CYCLIC_WIDTH(_XMP_template_t *template, int template_index, int nodes_index,
                                            unsigned long long width);

int xmp_dist_template_CYCLIC(xmp_desc_t t, int template_index, int nodes_index)
{
  _XMP_dist_template_CYCLIC_WIDTH((_XMP_template_t *) t, template_index, nodes_index, 1);
  return XMP_SUCCESS;
}

void xmp_dist_template_cyclic_(xmp_desc_t *t, int *template_index, int *nodes_index, int *status)
{
  if(*template_index <= 0 || *nodes_index <= 0){
    *status = XMP_ERROR;
    return;
  }
  _XMP_dist_template_CYCLIC_WIDTH((_XMP_template_t *) *t, *template_index-1, *nodes_index-1, 1);
  *status = XMP_SUCCESS;
}

int xmp_dist_template_BLOCK_CYCLIC(xmp_desc_t t, int template_index, int nodes_index, unsigned long long width)
{
  _XMP_dist_template_CYCLIC_WIDTH((_XMP_template_t *) t,template_index, nodes_index, width);
  return XMP_SUCCESS;
}

void xmp_dist_template_block_cyclic_(xmp_desc_t *t, int *template_index, int *nodes_index,
				     unsigned long long *width, int *status)
{
  if(*template_index <= 0 || *nodes_index <= 0){
    *status = XMP_ERROR;
    return;
  }
  _XMP_dist_template_CYCLIC_WIDTH((_XMP_template_t *) *t,*template_index-1, *nodes_index-1, *width);
  *status = XMP_SUCCESS;
}

int xmp_dist_template_GBLOCK(xmp_desc_t t, int template_index, int nodes_index,
			       int *mapping_array, int *temp0)
{
  _XMP_dist_template_GBLOCK((_XMP_template_t *) t, template_index, nodes_index, mapping_array, temp0);
  return XMP_SUCCESS;
}

void xmp_dist_template_gblock_(xmp_desc_t *t, int *template_index, int *nodes_index,
			       int *mapping_array, int *temp0, int *status)
{
  if(*template_index <= 0 || *nodes_index <= 0){
    *status = XMP_ERROR;
    return;
  }
  _XMP_dist_template_GBLOCK((_XMP_template_t *) *t, *template_index-1, *nodes_index-1, mapping_array, temp0);
  *status = XMP_SUCCESS;
}

#ifdef not
int xmp_is_on_template(xmp_desc_t t, xmp_dimension_t *dp)
{
  _XMP_fatal("xmp_is_on_template: not implemented yet");
}
#endif

/* #define N1 64 */
/* #define N2 64 */
/* double u[N2][N1], uu[N2][N1]; */
/* #pragma xmp align u[j][i] with t[j][i] */
/* 
  _XMP_init_array_desc(&(_XMP_DESC_u), _XMP_DESC_t, 2, 514, sizeof(double), (int)(0x000000040ll), (int)(0x000000040ll));
*/
extern void _XMP_init_array_desc_n(_XMP_array_t **array, _XMP_template_t *template, int dim,
				   int type, size_t type_size, int dim_size[]);

xmp_desc_t xmpc_new_array(xmp_desc_t t, xmp_datatype_t type, int n_dims, int dim_size1,/* int dim_size2,*/ ... )
{
  _XMP_array_t *array;
  size_t type_size;
  int dim_size[_XMP_N_MAX_DIM];
  va_list args;
  MPI_Datatype mpi_datatype;
  int i;
  
  _XMP_setup_reduce_type(&mpi_datatype, &type_size, (int) type);

  _XMP_ASSERT(n_dims <= _XMP_N_MAX_DIM);

  /* fortran order */
  dim_size[n_dims-1] = dim_size1;
  va_start(args, dim_size1);
  for (i= n_dims-2;i >= 0;i--){
    int size = va_arg(args, int);
    _XMP_ASSERT(size > 0);
    dim_size[i] = size;
  }
  va_end(args);

  // printf("xmpc_new_array #dim=%d\n",n_dims);
  /* for(i = 0; i < n_dims;i++) */
  /*   printf("xmpc_new_array dim=%d, dim_size=%d\n",i,dim_size[i]); */

  _XMP_init_array_desc_n(&array, (_XMP_template_t *)t, n_dims, type, type_size, dim_size);

  return (xmp_desc_t)array;
}

/*
  _XMP_align_array_BLOCK(_XMP_DESC_u, 0, 1, 0, &(_XMP_GTOL_temp0_u_0));
  _XMP_align_array_BLOCK(_XMP_DESC_u, 1, 0, 0, &(_XMP_GTOL_temp0_u_1));
*/
int xmp_align_array(xmp_desc_t a, int array_dim_idx, int template_dim_idx, long long offset)
{
  _XMP_array_t *array = (_XMP_array_t *)a;
  _XMP_template_t *template = array->align_template;
  _XMP_template_chunk_t *tc;
  int tmp; // dummy 

  /* convert */
  template_dim_idx = template->dim - 1 - template_dim_idx;
  tc = &(template->chunk[template_dim_idx]);

  /* BLOCK/CYCLIC/BLOCK_CLYCLE */
  switch (tc->dist_manner) {
  case _XMP_N_DIST_BLOCK:
    _XMP_align_array_BLOCK(array, array_dim_idx, template_dim_idx, offset, &tmp);
    break;
  case _XMP_N_DIST_CYCLIC:
    _XMP_align_array_CYCLIC(array, array_dim_idx, template_dim_idx, offset, &tmp);
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    _XMP_align_array_BLOCK_CYCLIC(array, array_dim_idx, template_dim_idx, offset, &tmp);
    break;
  case _XMP_N_DIST_GBLOCK:
    _XMP_align_array_GBLOCK(array, array_dim_idx, template_dim_idx, offset, &tmp);
    break;
  default:
    _XMP_fatal("xmp_align_array: bad distribution");
  }
  return XMP_SUCCESS;
}


/* REPLIACRTED, NOT_AGLINED */

/*  _XMP_init_shadow(_XMP_DESC_uu, (int)(401), (int)(1), (int)(1), (int)(401), (int)(1), (int)(1)); */
/*  _XMP_init_shadow(_XMP_DESC_uu, (int)(401), (int)(1), (int)(1), (int)(401), (int)(1), (int)(1)); */
extern void _XMP_init_shadow_dim(_XMP_array_t *array, int i, int type, int lo, int hi);

int xmp_set_shadow(xmp_desc_t a, int dim_idx, int shdw_size_lo, int shdw_size_hi)
{
  _XMP_init_shadow_dim((_XMP_array_t *)a, dim_idx, _XMP_N_SHADOW_NORMAL, shdw_size_lo, shdw_size_hi);
  return XMP_SUCCESS;
}

void xmp_set_shadow_(xmp_desc_t *a, int *dim_idx, int *shdw_size_lo, int *shdw_size_hi,
		     int *status)
{
  *status = xmp_set_shadow(*a, *dim_idx-1, *shdw_size_lo, *shdw_size_hi);
}

int xmp_set_full_shadow(xmp_desc_t a, int dim_idx)
{
  _XMP_init_shadow_dim((_XMP_array_t *)a, dim_idx, _XMP_N_SHADOW_FULL, 0, 0);
  return XMP_SUCCESS;
}

void xmp_set_full_shadow_(xmp_desc_t *a, int *dim_idx, int *status)
{
  *status = xmp_set_full_shadow(*a,*dim_idx-1);
}

/*
  _XMP_init_array_comm(_XMP_DESC_u, 0, 0);
  _XMP_init_array_nodes(_XMP_DESC_u);
  _XMP_alloc_array((void * * )(&(_XMP_ADDR_u)), _XMP_DESC_u, 1, (unsigned long long * )(&(_XMP_GTOL_acc_u_1)), (unsigned long long * )(&(_XMP_GTOL_acc_u_0)));
*/
extern void _XMP_alloc_array2(void **array_addr, _XMP_array_t *array_desc, int is_coarray,
			      unsigned long long *acc[]);

int xmp_allocate_array(xmp_desc_t a, void **addr)
{
  int args[_XMP_N_MAX_DIM];
  unsigned long long *acc[_XMP_N_MAX_DIM];
  _XMP_array_t *array = (_XMP_array_t *)a;
  
  for(int i = 0; i < _XMP_N_MAX_DIM; i++){
    args[i] = 0;
    acc[i] = NULL;
  }

  _XMP_init_array_comm2(array,args);
  _XMP_init_array_nodes(array);
  _XMP_alloc_array2(addr,array,TRUE,acc);
  return XMP_SUCCESS;
}


#ifdef not
int xmp_context_on_nodes_enter(xmp_node_t *np, xmp_range_t *rp)
{
}

int xmp_context_on_template_enter(xmp_template_t *tp, xmp_range_t *rp)
{
  
}

int xmp_context_exit(void)
{
  
}
#endif

/*
 *
 */
extern void xmpc_loop_sched(int ser_init, int ser_cond, int ser_step,
			    int *par_init, int *par_cond, int *par_step,
			    _XMP_template_t *t_desc, int t_idx,
			    int expand_type, int lwidth, int uwidth, int unbound_flag);

int xmpc_loop_schedule(int ser_init, int ser_cond, int ser_step,
		       xmp_desc_t t, int t_idx,
		       int *par_init, int *par_cond, int *par_step)
{
  xmpc_loop_sched(ser_init, ser_cond, ser_step,
		  par_init, par_cond, par_step,
		  (_XMP_template_t *) t, t_idx,
		  _XMP_LOOP_NONE, 0, 0, 0);
  return XMP_SUCCESS;
}

void xmp_loop_schedule_(int *ser_start, int *ser_end, int *ser_step,
			xmp_desc_t *t, int *t_idx,
			int *par_start, int *par_end, int *par_step,
			int *status)
{
  xmpc_loop_sched(*ser_start, *ser_end+1, *ser_step,
		  par_start, par_end, par_step,
		  (_XMP_template_t *) *t, *t_idx-1,
		  _XMP_LOOP_NONE, 0, 0, 0);
  *par_end -= 1;
  *status = XMP_SUCCESS;
}

int xmp_array_reflect(xmp_desc_t a)
{
  _XMP_reflect__((_XMP_array_t *)a);
  return XMP_SUCCESS;
}

void xmp_array_reflect_(xmp_desc_t *a, int *status)
{
  _XMP_reflect__((_XMP_array_t *)*a);
  *status = XMP_SUCCESS;
}

#ifdef not
int xmp_array_gmove (xmp_array_t *lhs_ap, xmp_range_t *lhs_rp,
		     xmp_array_t *rhs_ap, xmp_range_t *rhs_rp, xmp_gmove_kind_t kind)
{
  XMP_fatal("xmp_gmove: not implemeted yet");
}
#endif

int xmp_reduction_scalar(xmp_reduction_kind_t kind, xmp_datatype_t type, void *loc)
{
  _XMP_reduce_CLAUSE(loc, 1, (int)type, (int)kind);
  return XMP_SUCCESS;
}

void xmp_reduction_scalar_(int *kind, int *type, void **loc,int *status)
{
  _XMP_reduce_CLAUSE(*loc, 1, (int)*type, (int)*kind);
  *status = XMP_SUCCESS;
}

int xmp_bcast_scalar(xmp_datatype_t type, void *loc)
{
  MPI_Datatype mpi_datatype;
  size_t datatype_size;
  _XMP_setup_reduce_type(&mpi_datatype, &datatype_size, (int) type);
  // _XMP_M_BCAST_EXEC_OMITTED(loc,1, datatype_size);
  _XMP_bcast_NODES_ENTIRE_OMITTED(_XMP_get_execution_nodes(), loc, 1, datatype_size);
  return XMP_SUCCESS;
}

#ifdef not
int xmp_bcast_from_node(xmp_datatype_t type, void *loc, xmp_nodes_t *np, xmp_range_t *rp)
{
}

int xmp_bcast_from_template(xmp_datatype_t type, void *loc, xmp_template_t *tp, xmp_range_t *rp)
{
  XMP_fatal("xmp_bast_from_template: not implemeted yet");
}
#endif

int xmp_template_ltog(xmp_desc_t desc, int dim, int local_idx, long long int *global_idx)
{
  if(_XMP_L2G(local_idx,global_idx,(_XMP_template_t *)desc, dim))
    return XMP_SUCCESS;
  else return XMP_ERROR;
}

void xmp_template_ltog_(xmp_desc_t *desc, int *dim, int *local_idx, long long int *global_idx,
			int *status)
{
  if(_XMP_L2G(*local_idx,global_idx,(_XMP_template_t *)*desc, *dim-1))
    *status = XMP_SUCCESS;
  else *status = XMP_ERROR;
}

void xmp_template_gtol_(xmp_desc_t *desc, int *dim, long long int *global_idx, int *local_idx,
			int *status)
{
  if(_XMP_G2L(*global_idx,local_idx,(_XMP_template_t *)*desc, *dim-1))
    *status = XMP_SUCCESS;
  else *status = XMP_ERROR;
}
