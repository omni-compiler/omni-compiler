#include "xmpf_internal.h"

extern void (*_XMP_pack_comm_set)(void *sendbuf, int sendbuf_size,
				  _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);
extern void (*_XMP_unpack_comm_set)(void *recvbuf, int recvbuf_size,
				    _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);

static void _XMPF_pack_comm_set(void *sendbuf, int sendbuf_size,
				_XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);
static void _XMPF_unpack_comm_set(void *recvbuf, int recvbuf_size,
				  _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);

#define XMP_DBG 0
#define DBG_RANK 0

extern _XMP_nodes_t *gmv_nodes;
extern int n_gmv_nodes;


/* gmove sequence:
 * For global array,
 *  CALL xmp_gmv_g_alloc_(desc,XMP_DESC_a)
 *  CALL xmp_gmv_g_info(desc,#i_dim,kind,lb,ub,stride)
 * For local array
 *  CALL xmp_gmv_l_alloc_(desc,array,a_dim)
 *  CALL xmp_gmv_l_info(desc,#i_dim,a_lb,a_ub,kind,lb,ub,stride)
 *
 * kind = 2 -> ub, up, stride
 *        1 -> index
 *        0 -> all (:)
 * And, followed by:
 *  CALL xmp_gmv_do(left,right,collective(0)/in(1)/out(2))
 * Note: data type must be describe one of global side
 */

/* private final static int GMOVE_ALL   = 0; */
/* private final static int GMOVE_INDEX = 1; */
/* private final static int GMOVE_RANGE = 2; */
  
void
xmpf_gmv_g_alloc__(_XMP_gmv_desc_t **gmv_desc, _XMP_array_t **a_desc)
{
  _XMP_gmv_desc_t *gp;
  _XMP_array_t *ap = *a_desc;
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
xmpf_gmv_g_dim_info__(_XMP_gmv_desc_t **gmv_desc , int *i_dim,
		      int *kind, int *lb, int *ub, int *st)
{
  _XMP_gmv_desc_t *gp = *gmv_desc;
  int i = *i_dim;
  gp->kind[i] = *kind;

  switch (*kind){
  case XMP_N_GMOVE_ALL:
    gp->lb[i] = gp->a_desc->info[i].ser_lower;
    gp->ub[i] = gp->a_desc->info[i].ser_upper;
    gp->st[i] = 1;
    break;
  case XMP_N_GMOVE_INDEX:
  case XMP_N_GMOVE_RANGE:
    gp->lb[i] = *lb;
    gp->ub[i] = *ub;
    gp->st[i] = *st;
    break;
  default:
    _XMP_fatal("wrong gmove kind");
  }

}


void
xmpf_gmv_l_alloc__(_XMP_gmv_desc_t **gmv_desc , void *local_data, int *ndims)
{
    _XMP_gmv_desc_t *gp;
    int n = *ndims;

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
xmpf_gmv_l_dim_info__(_XMP_gmv_desc_t **gmv_desc , int *i_dim, int *a_lb, int *a_ub,
		      int *kind, int *lb, int *ub, int *st)
{
  _XMP_gmv_desc_t *gp = *gmv_desc;
  int i = *i_dim;

  gp->a_lb[i] = *a_lb;
  gp->a_ub[i] = *a_ub;

  gp->kind[i] = *kind;
  gp->lb[i] = *lb;
  gp->ub[i] = *ub;
  gp->st[i] = *st;
}


void
xmpf_gmv_dealloc__(_XMP_gmv_desc_t **gmv_desc){

  _XMP_gmv_desc_t *gp = *gmv_desc;

  _XMP_free(gp->kind);
  _XMP_free(gp->lb);
  _XMP_free(gp->ub);
  _XMP_free(gp->st);

  _XMP_free(gp->a_lb);
  _XMP_free(gp->a_ub);

  _XMP_free(gp);

}


static void _XMPF_larray_alloc(_XMP_array_t **a, _XMP_gmv_desc_t *gmv_desc, int type, _XMP_template_t *t){
  xmpf_array_alloc__(a, &gmv_desc->ndims, &type, &t);
  for (int i = 0; i < gmv_desc->ndims; i++){
    int t_idx = -1; int off = 0;
    xmpf_align_info__(a, &i, gmv_desc->a_lb + i, gmv_desc->a_ub + i, &t_idx, &off);
  }
  int dummy = 0;
  xmpf_array_set_local_array__(a, gmv_desc->local_data, &dummy);
  gmv_desc->a_desc = *a;
  (*a)->total_elmts = -1; // temporal descriptor
}


void
xmpf_gmv_do__(_XMP_gmv_desc_t **gmv_desc_left, _XMP_gmv_desc_t **gmv_desc_right,
	   int *mode)
{
  _XMP_gmv_desc_t *gmv_desc_leftp = *gmv_desc_left;
  _XMP_gmv_desc_t *gmv_desc_rightp = *gmv_desc_right;

  _XMP_pack_comm_set = _XMPF_pack_comm_set;
  _XMP_unpack_comm_set = _XMPF_unpack_comm_set;

  if (gmv_desc_leftp->is_global && gmv_desc_rightp->is_global){
    _XMP_gmove_garray_garray(gmv_desc_leftp, gmv_desc_rightp, *mode);
  }
  else if (gmv_desc_leftp->is_global && !gmv_desc_rightp->is_global){
    if (gmv_desc_rightp->ndims == 0){
      _XMP_gmove_garray_scalar(gmv_desc_leftp, gmv_desc_rightp->local_data, *mode);
    }
    else {
      _XMP_array_t *a = NULL;
      _XMPF_larray_alloc(&a, gmv_desc_rightp,
			 gmv_desc_leftp->a_desc->type, gmv_desc_leftp->a_desc->align_template);
      _XMP_gmove_garray_larray(gmv_desc_leftp, gmv_desc_rightp, *mode);
      xmpf_array_dealloc__(&a);
    }
  }
  else if (!gmv_desc_leftp->is_global && gmv_desc_rightp->is_global){
    if (gmv_desc_leftp->ndims == 0){
      _XMP_gmove_scalar_garray(gmv_desc_leftp->local_data, gmv_desc_rightp, *mode);
    }
    else {

      _XMP_ASSERT(gmv_desc_rightp->a_desc);

      // create a temporal descriptor for the "non-distributed" LHS array (to be possible used
      // in _XMP_gmove_1to1)
      _XMP_array_t *a = NULL;
      _XMPF_larray_alloc(&a, gmv_desc_leftp,
			 gmv_desc_rightp->a_desc->type, gmv_desc_rightp->a_desc->align_template);
      _XMP_gmove_larray_garray(gmv_desc_leftp, gmv_desc_rightp, *mode);
      xmpf_array_dealloc__(&a);

    }
  }
  else {
    _XMP_fatal("gmv_do: both sides are local.");
  }

}


void
_XMPF_pack_comm_set(void *sendbuf, int sendbuf_size,
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
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      break;

    case 3:
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      break;

    case 4:
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      break;

    case 5:
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      }}
      break;

    case 6:
      for (c[5] = comm_set[dst_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
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
      for (c[6] = comm_set[dst_node][6]; c[6]; c[6] = c[6]->next){
	for (i[6] = c[6]->l; i[6] <= c[6]->u; i[6]++){
      for (c[5] = comm_set[dst_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
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
  	printf("%d ", ((int *)sendbuf)[i]);
      }
      printf("\n");
    }
    fflush(stdout);
    xmp_barrier();
  }
#endif

}


void
_XMPF_unpack_comm_set(void *recvbuf, int recvbuf_size,
		      _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  //int myrank = gmv_nodes->comm_rank;

  int ndims = a->dim;

  char *buf = (char *)recvbuf;
  char *dst = (char *)a->array_addr_p;

#if XMP_DBG
  int myrank = gmv_nodes->comm_rank;

    fflush(stdout);
    xmp_barrier();

  if (myrank == 0){
    printf("\n");
    printf("Recv buffer -------------------------------------\n");
  }

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    if (myrank == gmv_rank){
      printf("\n");
      printf("[%d]\n", myrank);
      for (int i = 0; i < recvbuf_size; i++){
  	printf("%d ", ((int *)recvbuf)[i]);
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
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      break;

    case 3:
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      break;

    case 4:
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      break;

    case 5:
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      }}
      break;

    case 6:
      for (c[5] = comm_set[src_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
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
      for (c[6] = comm_set[src_node][6]; c[6]; c[6] = c[6]->next){
	for (i[6] = c[6]->l; i[6] <= c[6]->u; i[6]++){
      for (c[5] = comm_set[src_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
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
