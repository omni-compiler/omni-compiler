#include <stdlib.h>
#include "xmp_internal.h"

extern void (*_XMP_pack_comm_set)(void *sendbuf, int sendbuf_size,
				  _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);
extern void (*_XMP_unpack_comm_set)(void *recvbuf, int recvbuf_size,
				    _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);

static void _XMPC_pack_comm_set(void *sendbuf, int sendbuf_size,
				_XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);
static void _XMPC_unpack_comm_set(void *recvbuf, int recvbuf_size,
				  _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);


#define XMP_DBG 0
#define DBG_RANK 0


extern _XMP_nodes_t *gmv_nodes;
extern int n_gmv_nodes;


void
xmpc_gmv_g_alloc(_XMP_gmv_desc_t **gmv_desc, _XMP_array_t *ap)
{
  _XMP_gmv_desc_t *gp;
  int n = ap->dim;

  gp = (_XMP_gmv_desc_t *)_XMP_alloc(sizeof(_XMP_gmv_desc_t));

  gp->kind = (int *)_XMP_alloc(sizeof(int) * n);
  gp->lb = (int *)_XMP_alloc(sizeof(int) * n);
  gp->ub = (int *)_XMP_alloc(sizeof(int) * n);
  gp->st = (int *)_XMP_alloc(sizeof(int) * n);
  
  if (!gp || !gp->kind || !gp->lb || !gp->st)
    _XMP_fatal("gmv_g_alloc: cannot alloc memory");

  gp->is_global = true;
  gp->ndims = n;
  gp->a_desc = ap;

  gp->local_data = NULL;
  gp->a_lb = NULL;
  gp->a_ub = NULL;

  *gmv_desc = gp;
}


void
xmpc_gmv_g_dim_info(_XMP_gmv_desc_t *gp, int i,
		      int kind, int lb, int len, int st)
{
  gp->kind[i] = kind;

  switch (kind){
  case XMP_N_GMOVE_ALL:
    gp->lb[i] = gp->a_desc->info[i].ser_lower;
    gp->ub[i] = gp->a_desc->info[i].ser_upper;
    gp->st[i] = 1;
    break;
  case XMP_N_GMOVE_INDEX:
  case XMP_N_GMOVE_RANGE:
    gp->lb[i] = lb;
    gp->ub[i] = lb + st * (len - 1);
    gp->st[i] = st;
    break;
  default:
    _XMP_fatal("wrong gmove kind");
  }

}


void
xmpc_gmv_l_alloc(_XMP_gmv_desc_t **gmv_desc, void *local_data, int n)
{
  _XMP_gmv_desc_t *gp;

  gp = (_XMP_gmv_desc_t *)_XMP_alloc(sizeof(_XMP_gmv_desc_t));

  gp->kind = (int *)_XMP_alloc(sizeof(int) * n);
  gp->lb = (int *)_XMP_alloc(sizeof(int) * n);
  gp->ub = (int *)_XMP_alloc(sizeof(int) * n);
  gp->st = (int *)_XMP_alloc(sizeof(int) * n);
  gp->a_lb = (int *)_XMP_alloc(sizeof(int) * n);
  gp->a_ub = (int *)_XMP_alloc(sizeof(int) * n);

  gp->is_global = false;
  gp->ndims = n;
  gp->a_desc = NULL;

  gp->local_data = local_data;

  *gmv_desc = gp;
}


void
xmpc_gmv_l_dim_info(_XMP_gmv_desc_t *gp, int i, int a_lb, int a_len,
		    int kind, int lb, int len, int st)
{
  gp->a_lb[i] = a_lb; // always 0
  gp->a_ub[i] = a_len - 1;

  gp->kind[i] = kind;

  switch (kind){
  case XMP_N_GMOVE_ALL:
    gp->lb[i] = a_lb;
    gp->ub[i] = a_len - 1;
    gp->st[i] = 1;
    break;
  case XMP_N_GMOVE_INDEX:
  case XMP_N_GMOVE_RANGE:
    gp->lb[i] = lb;
    gp->ub[i] = lb + st * (len - 1);
    gp->st[i] = st;
    break;
  default:
    _XMP_fatal("wrong gmove kind");
  }

}


void
xmpc_gmv_dealloc(_XMP_gmv_desc_t *gp){

  _XMP_free(gp->kind);
  _XMP_free(gp->lb);
  _XMP_free(gp->ub);
  _XMP_free(gp->st);

  _XMP_free(gp->a_lb);
  _XMP_free(gp->a_ub);

  _XMP_free(gp);

}


static void _XMPC_larray_alloc(_XMP_array_t **a, _XMP_gmv_desc_t *gmv_desc, int type, _XMP_template_t *t){

  int ndims = gmv_desc->ndims;

  unsigned long long dim_acc[ndims];
  dim_acc[ndims-1] = 1;
  for (int i = ndims-2; i >= 0; i--){
    dim_acc[i] = dim_acc[i+1] * (gmv_desc->a_ub[i] + 1);
  }

  _XMP_init_array_desc_NOT_ALIGNED(a, t, ndims, type, _XMP_get_datatype_size(type),
				   dim_acc, gmv_desc->local_data);

  gmv_desc->a_desc = *a;
  (*a)->total_elmts = -1; // temporal descriptor
}

#ifdef _XMPT
extern void _XMPT_set_gmove_subsc(xmpt_subscript_t subsc, _XMP_gmv_desc_t *gmv_desc);
#endif

void
xmpc_gmv_do(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
	    int mode)
{

  _XMP_pack_comm_set = _XMPC_pack_comm_set;
  _XMP_unpack_comm_set = _XMPC_unpack_comm_set;

#ifdef _XMPT
  xmpt_tool_data_t data = NULL;
  struct _xmpt_subscript_t lhs_subsc, rhs_subsc;
  _XMPT_set_gmove_subsc(&lhs_subsc, gmv_desc_leftp);
  _XMPT_set_gmove_subsc(&rhs_subsc, gmv_desc_rightp);
  xmpt_gmove_kind_t kind = mode - _XMP_N_GMOVE_NORMAL;
  xmpt_async_id_t async_id = 0;
#ifdef _XMP_MPI3
  if (xmp_is_async()){
    _XMP_async_comm_t *async = _XMP_get_current_async();
    async_id = async->async_id;
  }
#endif
#endif
  
  if (gmv_desc_leftp->is_global && gmv_desc_rightp->is_global){

#ifdef _XMPT
    if (xmpt_enabled){
      if (!xmp_is_async() && xmpt_callback[xmpt_event_gmove_begin])
	(*(xmpt_event_gmove_begin_t)xmpt_callback[xmpt_event_gmove_begin])
	  (gmv_desc_leftp->a_desc, &lhs_subsc,
	   gmv_desc_rightp->a_desc, &rhs_subsc,
	   kind, &data);
      else if (xmp_is_async() && xmpt_callback[xmpt_event_gmove_begin_async])
	(*(xmpt_event_gmove_begin_async_t)xmpt_callback[xmpt_event_gmove_begin_async])
	  (gmv_desc_leftp->a_desc, &lhs_subsc,
	   gmv_desc_rightp->a_desc, &rhs_subsc,
	   kind, async_id, &data);
    }
#endif

    _XMP_gmove_garray_garray(gmv_desc_leftp, gmv_desc_rightp, mode);

  }
  else if (gmv_desc_leftp->is_global && !gmv_desc_rightp->is_global){
    if (gmv_desc_rightp->ndims == 0){
      // needs xmpt
      _XMP_gmove_garray_scalar(gmv_desc_leftp, gmv_desc_rightp->local_data, mode);
    }
    else {
      _XMP_array_t *a = NULL;
      _XMPC_larray_alloc(&a, gmv_desc_rightp,
			 gmv_desc_leftp->a_desc->type, gmv_desc_leftp->a_desc->align_template);

#ifdef _XMPT
      if (xmpt_enabled){
	if (!xmp_is_async() && xmpt_callback[xmpt_event_gmove_begin])
	  (*(xmpt_event_gmove_begin_t)xmpt_callback[xmpt_event_gmove_begin])
	    (gmv_desc_leftp->a_desc, &lhs_subsc,
	     gmv_desc_rightp->a_desc, &rhs_subsc,
	     kind, &data);
	else if (xmp_is_async() && xmpt_callback[xmpt_event_gmove_begin_async])
	  (*(xmpt_event_gmove_begin_async_t)xmpt_callback[xmpt_event_gmove_begin_async])
	    (gmv_desc_leftp->a_desc, &lhs_subsc,
	     gmv_desc_rightp->a_desc, &rhs_subsc,
	     kind, async_id, &data);
      }
#endif

      _XMP_gmove_garray_larray(gmv_desc_leftp, gmv_desc_rightp, mode);
      _XMP_finalize_array_desc(a);
    }
  }
  else if (!gmv_desc_leftp->is_global && gmv_desc_rightp->is_global){
    if (gmv_desc_leftp->ndims == 0){
      // needs xmpt
      _XMP_gmove_scalar_garray(gmv_desc_leftp->local_data, gmv_desc_rightp, mode);
    }
    else {

      _XMP_ASSERT(gmv_desc_rightp->a_desc);

      // create a temporal descriptor for the "non-distributed" LHS array (to be possible used
      // in _XMP_gmove_1to1)
      _XMP_array_t *a = NULL;
      _XMPC_larray_alloc(&a, gmv_desc_leftp,
			 gmv_desc_rightp->a_desc->type, gmv_desc_rightp->a_desc->align_template);

#ifdef _XMPT
      if (xmpt_enabled){
	if (!xmp_is_async() && xmpt_callback[xmpt_event_gmove_begin])
	  (*(xmpt_event_gmove_begin_t)xmpt_callback[xmpt_event_gmove_begin])
	    (gmv_desc_leftp->a_desc, &lhs_subsc,
	     gmv_desc_rightp->a_desc, &rhs_subsc,
	     kind, &data);
	else if (xmp_is_async() && xmpt_callback[xmpt_event_gmove_begin_async])
	  (*(xmpt_event_gmove_begin_async_t)xmpt_callback[xmpt_event_gmove_begin_async])
	    (gmv_desc_leftp->a_desc, &lhs_subsc,
	     gmv_desc_rightp->a_desc, &rhs_subsc,
	     kind, async_id, &data);
      }
#endif

      _XMP_gmove_larray_garray(gmv_desc_leftp, gmv_desc_rightp, mode);
      _XMP_finalize_array_desc(a);

    }
  }
  else {
    _XMP_fatal("gmv_do: both sides are local.");
  }

#ifdef _XMPT
  if (xmpt_enabled && xmpt_callback[xmpt_event_gmove_end])
    (*(xmpt_event_end_t)xmpt_callback[xmpt_event_gmove_end])(
     &data);
#endif

}


static void
_XMPC_pack_comm_set(void *sendbuf, int sendbuf_size,
		    _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  int ndims = a->dim;

  char *buf = (char *)sendbuf;
  char *src = (char *)a->array_addr_p;

  for (int dst_node = 0; dst_node < n_gmv_nodes; dst_node++){

    _XMP_comm_set_t *c[ndims];

    int i[_XMP_N_MAX_DIM];

    switch (ndims){

    case 1:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      break;

    case 2:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
    	i[1] = c[1]->l;
    	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      break;

    case 3:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
    	i[2] = c[2]->l;
    	int size = (c[2]->u - c[2]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      break;

    case 4:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
    	i[3] = c[3]->l;
    	int size = (c[3]->u - c[3]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      break;

    case 5:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
    	i[4] = c[4]->l;
    	int size = (c[4]->u - c[4]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      }}
      break;

    case 6:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[dst_node][5]; c[5]; c[5] = c[5]->next){
    	i[5] = c[5]->l;
    	int size = (c[5]->u - c[5]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      break;

    case 7:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[dst_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[6] = comm_set[dst_node][6]; c[6]; c[6] = c[6]->next){
    	i[6] = c[6]->l;
    	int size = (c[6]->u - c[6]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      }}
      break;

    default:
      _XMP_fatal("wrong array dimension");
    }

  }

#if XMP_DBG
  int myrank = gmv_nodes->comm_rank;

  if (myrank == 0){
    printf("\n");
    printf("Send buffer -------------------------------------\n");
  }

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    if (myrank == gmv_rank){
      printf("\n");
      printf("[%d]\n", myrank);
      for (int i = 0; i < sendbuf_size; i++){
  	printf("%.0f ", ((double *)sendbuf)[i]);
      }
      printf("\n");
    }
    fflush(stdout);
    xmp_barrier();
  }
#endif

}


static void
_XMPC_unpack_comm_set(void *recvbuf, int recvbuf_size,
		      _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  //int myrank = gmv_nodes->comm_rank;

  int ndims = a->dim;

  char *buf = (char *)recvbuf;
  char *dst = (char *)a->array_addr_p;

#if XMP_DBG
  int myrank = gmv_nodes->comm_rank;

  if (myrank == 0){
    printf("\n");
    printf("Recv buffer -------------------------------------\n");
  }

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    if (myrank == gmv_rank){
      printf("\n");
      printf("[%d]\n", myrank);
      for (int i = 0; i < recvbuf_size; i++){
  	printf("%.0f ", ((double *)recvbuf)[i]);
      }
      printf("\n");
    }
    fflush(stdout);
    xmp_barrier();
  }
#endif

  for (int src_node = 0; src_node < n_gmv_nodes; src_node++){

    _XMP_comm_set_t *c[ndims];

    int i[_XMP_N_MAX_DIM];

    switch (ndims){

    case 1:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      break;

    case 2:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	i[1] = c[1]->l;
	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	//xmp_dbg_printf("(%d, %d) offset = %03d, size = %d\n", i[0], i[1], o, size);
	buf += size;
      }
      }}
      break;

    case 3:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	i[2] = c[2]->l;
	int size = (c[2]->u - c[2]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      break;

    case 4:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	i[3] = c[3]->l;
	int size = (c[3]->u - c[3]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      break;

    case 5:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	i[4] = c[4]->l;
	int size = (c[4]->u - c[4]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      }}
      break;

    case 6:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[src_node][5]; c[5]; c[5] = c[5]->next){
	i[5] = c[5]->l;
	int size = (c[5]->u - c[5]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      break;

    case 7:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[src_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[6] = comm_set[src_node][6]; c[6]; c[6] = c[6]->next){
	i[6] = c[6]->l;
	int size = (c[6]->u - c[6]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      }}
      break;

    default:
      _XMP_fatal("wrong array dimension");
    }

  }

}
