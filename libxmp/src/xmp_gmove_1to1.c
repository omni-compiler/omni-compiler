/* #ifndef MPI_PORTABLE_PLATFORM_H */
/* #define MPI_PORTABLE_PLATFORM_H */
/* #endif  */

/* #include <stdarg.h> */
/* #include <stdlib.h> */
/* #include <string.h> */
/* #include "xmp.h" */
/* #include "mpi.h" */
/* #include "xmp_internal.h" */
/* #include "xmp_math_function.h" */

/* #define _XMP_SM_GTOL_BLOCK(_i, _m, _w) \ */
/* (((_i) - (_m)) % (_w)) */

/* #define _XMP_SM_GTOL_CYCLIC(_i, _m, _P) \ */
/* (((_i) - (_m)) / (_P)) */

/* #define _XMP_SM_GTOL_BLOCK_CYCLIC(_b, _i, _m, _P) \ */
/* (((((_i) - (_m)) / (((_P) * (_b)))) * (_b)) + (((_i) - (_m)) % (_b))) */

#include <stdlib.h>
#include <stdio.h>

#define _XMP_alloc malloc

/* #define MIN(a,b)  ( (a)<(b) ? (a) : (b) ) */

/* int _XMP_gcd(int a, int b) { */
/*   int r = a % b; */
/*   if (r == 0) { */
/*     return b; */
/*   } else { */
/*     return _XMP_gcd(b, r); */
/*   } */
/* } */

/* int _XMP_lcm(int a, int b) { */
/*   return (a * b) / _XMP_gcd(a, b); */
/* } */


typedef struct _rsd {
  int l;
  int u;
  int s;
} _rsd_t;


typedef struct _bsd {
  int l;
  int u;
  int b;
  int c;
} _bsd_t;


typedef struct _csd {
  int *l;
  int *u;
  int b;
  int s;
} _csd_t;


typedef struct _comm_set {
  int l;
  int u;
  struct _comm_set *next;
} _comm_set_t;


static void print_rsd(_rsd_t *rsd){

  if (!rsd){
    printf("()\n");
    return;
  }

  if (rsd->s){
    printf("(%d : %d : %d)\n", rsd->l, rsd->u, rsd->s);
  }
  else {
    printf("()\n");
  }
}


static void print_bsd(_bsd_t *bsd){
  if (!bsd){
    printf("()\n");
  }
  else {
    printf("(%d : %d : %d : %d)\n", bsd->l, bsd->u, bsd->b, bsd->c);
  }
}


static void print_csd(_csd_t *csd){

  if (!csd || csd->b == 0){
    printf("()\n");
    return;
  }

  printf("(");

  printf("(%d", csd->l[0]);
  for (int i = 1; i < csd->b; i++){
    printf(", %d", csd->l[i]);
  }
  printf(")");

  printf(" : ");

  printf("(%d", csd->u[0]);
  for (int i = 1; i < csd->b; i++){
    printf(", %d", csd->u[i]);
  }
  printf(")");

  printf(" : ");

  printf("%d)\n", csd->s);

}


static void print_comm_set(_comm_set_t *comm_set0){

  if (!comm_set0){
    printf("()\n");
    return;
  }

  _comm_set_t *comm_set = comm_set0;

  printf("(%d : %d)", comm_set->l, comm_set->u);

  while ((comm_set = comm_set->next)){
    printf(", (%d : %d)", comm_set->l, comm_set->u);
  }

  printf("\n");

}


static void free_comm_set(_comm_set_t *comm_set){

  while (comm_set){
    _comm_set_t *next = comm_set->next;
    _XMP_free(comm_set);
    comm_set = next;
  }

}

static _rsd_t *intersection_rsds(_rsd_t *_rsd1, _rsd_t *_rsd2){

  _rsd_t *rsd1, *rsd2;

  if (!_rsd1 || !_rsd2) return NULL;

  if (_rsd1->l <= _rsd2->l){
    rsd1 = _rsd1;
    rsd2 = _rsd2;
  }
  else {
    rsd1 = _rsd2;
    rsd2 = _rsd1;
  }

  if (rsd2->l <= rsd1->u){
    for (int i = rsd2->l; i <= rsd2->u; i += rsd2->s){
      if ((i - rsd1->l) % rsd1->s == 0){
	_rsd_t *rsd0 = (_rsd_t *)_XMP_alloc(sizeof(_rsd_t));
	rsd0->l = i;
	rsd0->s = _XMP_lcm(rsd1->s, rsd2->s);
	int t = (MIN(rsd1->u, rsd2->u) - rsd0->l) / rsd0->s;
	rsd0->u = rsd0->l + rsd0->s * t;
	return rsd0;
      }
    }
  }

  return NULL;

}


static _csd_t *alloc_csd(int b){
  _csd_t *csd = (_csd_t *)_XMP_alloc(sizeof(_csd_t));
  csd->l = (int *)_XMP_alloc(sizeof(int) * b);
  csd->u = (int *)_XMP_alloc(sizeof(int) * b);
  return csd;
}


static void free_csd(_csd_t *csd){
  if (csd){
    _XMP_free(csd->l);
    _XMP_free(csd->u);
    _XMP_free(csd);
  }
}


static _csd_t *intersection_csds(_csd_t *csd1, _csd_t *csd2){

  if (!csd1 || !csd2) return NULL;

  _csd_t *csd0 = alloc_csd(MAX(csd1->b, csd2->b));

  int k = 0;

  csd0->b = 0;

  for (int i = 0; i < csd1->b; i++){
    for (int j = 0; j < csd2->b; j++){

      _rsd_t *tmp = NULL;
      _rsd_t rsd1 = { csd1->l[i], csd1->u[i], csd1->s };
      _rsd_t rsd2 = { csd2->l[j], csd2->u[j], csd2->s };

      tmp = intersection_rsds(&rsd1, &rsd2);

      if (tmp){

	csd0->l[k] = tmp->l;
	csd0->u[k] = tmp->u;

	// bubble sort
	int p = k;
	while (csd0->l[p-1] > tmp->l && p > 0){
	  int s;
	  s = csd0->l[p-1]; csd0->l[p-1] = csd0->l[p]; csd0->l[p] = s;
	  s = csd0->u[p-1]; csd0->u[p-1] = csd0->u[p]; csd0->u[p] = s;
	  p--;
	}

	csd0->b++;
	csd0->s = tmp->s;
	k++;

	_XMP_free(tmp);
      }

    }
  }

  return csd0;

}


static _csd_t *rsd2csd(_rsd_t *rsd){
  if (!rsd) return NULL;
  _csd_t *csd = alloc_csd(1);
  csd->l[0] = rsd->l;
  csd->u[0] = rsd->u;
  csd->b = 1;
  csd->s = rsd->s;
  return csd;
}


static _csd_t *bsd2csd(_bsd_t *bsd){

  if (!bsd) return NULL;

  _csd_t *csd = alloc_csd(bsd->b);
  csd->b = bsd->b;

  for (int i = 0; i < bsd->b; i++){
    csd->l[i] = bsd->l + i;
    int t = (bsd->u - csd->l[i]) / bsd->c;
    csd->u[i] = csd->l[i] + bsd->c * t;
  }

  csd->s = bsd->c;

  return csd;

}


static _comm_set_t *csd2comm_set(_csd_t *csd){

  if (!csd || csd->b == 0) return NULL;

  _comm_set_t *comm_set0 = (_comm_set_t *)_XMP_alloc(sizeof(_comm_set_t));

  _comm_set_t *comm_set = comm_set0;
  comm_set->l = csd->l[0];
  comm_set->u = csd->l[0];
  comm_set->next = NULL;

  for (int j = 0; csd->l[0] + j <= csd->u[0]; j+= csd->s){

    for (int i = 0; i < csd->b; i++){

      int l = csd->l[i] + j;

      if (l > csd->u[i]) continue;

      if (l == comm_set->u + 1){
	comm_set->u = l;
      }
      else if (l <= comm_set->u){
	continue;
      }
      else {
	comm_set->next = (_comm_set_t *)malloc(sizeof(_comm_set_t));
	comm_set = comm_set->next;
	comm_set->l = l;
	comm_set->u = l;
	comm_set->next = NULL;
      }
    }

  }

  return comm_set0;

}


_Bool _XMP_calc_coord_on_target_nodes2(_XMP_nodes_t *n, int *ncoord, 
				       _XMP_nodes_t *target_n, int *target_ncoord){

  if (n == target_n){
    //printf("%d, %d\n", n->dim, target_n->dim);
    memcpy(target_ncoord, ncoord, sizeof(int) * n->dim);
    return true;
  }
  else if (n->attr == XMP_ENTIRE_NODES && target_n->attr == XMP_ENTIRE_NODES){
    int rank = _XMP_calc_linear_rank(n, ncoord);
    _XMP_calc_rank_array(target_n, target_ncoord, rank);
    return true;
  }

  _XMP_nodes_t *target_p = target_n->inherit_nodes;
  if (target_p){

    int target_pcoord[_XMP_N_MAX_DIM];

    if (_XMP_calc_coord_on_target_nodes2(n, ncoord, target_p, target_pcoord)){

      int target_prank = _XMP_calc_linear_rank(target_p, target_pcoord);
      /* printf("dim = %d, m0 = %d, m1 = %d\n", */
      /* 	     target_n->dim, target_n->info[0].multiplier, target_n->info[1].multiplier); */
      _XMP_calc_rank_array(target_n, target_ncoord, target_prank);

      /* _XMP_nodes_inherit_info_t *inherit_info = target_n->inherit_info; */

      /* int j = 0; */
      /* for (int i = 0; i < target_p->dim; i++) { */
      /* 	if (inherit_info[i].shrink) { */
      /* 	  ; */
      /* 	} */
      /* 	else { */
      /* 	  target_ncoord[j] = (target_pcoord[j] - inherit_info[i].lower) / inherit_info[i].stride; */
      /* 	  j++; */
      /* 	} */
      /* } */

      return true;
    }

  }

  return false;

}
    

_Bool _XMP_calc_coord_on_target_nodes(_XMP_nodes_t *n, int *ncoord, 
				      _XMP_nodes_t *target_n, int *target_ncoord){

  //printf("ncoord[0] = %d, ncoord[1] = %d\n", ncoord[0], ncoord[1]);

  if (_XMP_calc_coord_on_target_nodes2(n, ncoord, target_n, target_ncoord)){
    return true;
  }

  _XMP_nodes_t *p = n->inherit_nodes;
  if (p){

    int pcoord[_XMP_N_MAX_DIM];

    int rank = _XMP_calc_linear_rank(n, ncoord);
    _XMP_calc_rank_array(p, pcoord, rank);
     
    /* _XMP_nodes_inherit_info_t *inherit_info = n->inherit_info; */

    /* int j = 0; */
    /* for (int i = 0; i < p->dim; i++) { */
    /*   if (inherit_info[i].shrink) { */
    /* 	pcoord[i] = p->info[i].rank; */
    /*   } */
    /*   else { */
    /* 	pcoord[i] = inherit_info[i].stride * ncoord[j] + inherit_info[i].lower; */
    /* 	j++; */
    /*   } */
    /* } */

    if (_XMP_calc_coord_on_target_nodes(p, pcoord, target_n, target_ncoord)){
      //printf("%d, %d\n", p->dim, target_n->dim);
      return true;
    }

  }

  return false;

}


_csd_t *get_owner_csd(_XMP_array_t *a, int adim, int ncoord[]){

  _XMP_array_info_t *ainfo = &(a->info[adim]);

  int tdim = ainfo->align_template_index;
  _XMP_template_info_t *tinfo = &(a->align_template->info[tdim]);
  _XMP_template_chunk_t *tchunk = &(a->align_template->chunk[tdim]);

  int ndim = tchunk->onto_nodes_index;

  int nidx = ncoord[ndim];

  _bsd_t bsd = { 0, 0, 0, 0};

  switch (ainfo->align_manner){

  case _XMP_N_ALIGN_DUPLICATION:
  case _XMP_N_ALIGN_NOT_ALIGNED:
    bsd.l = ainfo->ser_lower;
    bsd.u = ainfo->ser_upper;
    bsd.b = 1;
    bsd.c = 1;
    break;
	
  case _XMP_N_ALIGN_BLOCK:
    /* printf("[%d] ser_lower = %d, ser_upper = %d, par_lower = %d, par_upper = %d, subscript = %d\n", */
    /* 	   nidx, ainfo->ser_lower, ainfo->ser_upper, ainfo->par_lower, ainfo->par_upper, */
    /* 	   ainfo->align_subscript); */
    bsd.l = tinfo->ser_lower + tchunk->par_chunk_width * nidx - ainfo->align_subscript;
    bsd.u = MIN(tinfo->ser_lower + tchunk->par_chunk_width * (nidx + 1) - 1 - ainfo->align_subscript,
		tinfo->ser_upper - ainfo->align_subscript);
    bsd.b = 1;
    bsd.c = 1;
    break;

  case _XMP_N_ALIGN_CYCLIC:
  case _XMP_N_ALIGN_BLOCK_CYCLIC:
    bsd.l = tinfo->ser_lower + (nidx * tchunk->par_width) - ainfo->align_subscript;
    bsd.u = tinfo->ser_upper;
    bsd.b = tchunk->par_width;
    bsd.c = tchunk->par_stride;
    break;

  case _XMP_N_ALIGN_GBLOCK:
    bsd.l = tchunk->mapping_array[nidx] - ainfo->align_subscript;
    bsd.u = tchunk->mapping_array[nidx + 1] - 1 - ainfo->align_subscript;
    bsd.b = 1;
    bsd.c = 1;
    break;
      
  default:
    _XMP_fatal("_XMP_gmove_1to1: unknown distribution format");

  }

  return bsd2csd(&bsd);

}


void reduce_csd(_csd_t *csd[_XMP_N_MAX_DIM], int ndims){

  for (int i = 0; i < ndims; i++){
    if (!csd[i] || csd[i]->b == 0){
      for (int j = 0; j < ndims; j++){
	free_csd(csd[j]);
	csd[j] = NULL;
      }
      return;
    }
  }      

}

void get_owner_ref_csd(_XMP_array_t *adesc, int *lb, int *ub, int *st,
		       _csd_t *owner_ref_csd[][_XMP_N_MAX_DIM]){

  int n_adims = adesc->dim;

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  int myrank = exec_nodes->comm_rank;
  int n_exec_nodes = exec_nodes->comm_size;

  //
  // referenced region
  //

  _rsd_t rsd_ref[n_adims];
  _csd_t *csd_ref[n_adims];
  for (int i = 0; i < n_adims; i++){
    if (st[i] != 0){
      rsd_ref[i].l = lb[i];
      rsd_ref[i].u = ub[i];
      rsd_ref[i].s = st[i];
    }
    else {
      rsd_ref[i].l = lb[i];
      rsd_ref[i].u = lb[i];
      rsd_ref[i].s = 1;
    }
    csd_ref[i] = rsd2csd(&rsd_ref[i]);
  }

  //
  // owner region
  //

  _csd_t *owner_csd[n_exec_nodes][n_adims];

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){

    int exec_ncoord[_XMP_N_MAX_DIM];
    _XMP_calc_rank_array(exec_nodes, exec_ncoord, exec_rank);

    int target_ncoord[_XMP_N_MAX_DIM];

    if (_XMP_calc_coord_on_target_nodes(exec_nodes, exec_ncoord,
    					adesc->align_template->onto_nodes, target_ncoord)){
      for (int adim = 0; adim < n_adims; adim++){
  	owner_csd[exec_rank][adim] = get_owner_csd(adesc, adim, target_ncoord);
      }

    }
    else {
      _XMP_fatal("_XMP_gmove_1to1: array not allocated on the executing node array");
    }

  }

  //
  // intersection of reference and owner region
  //

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    for (int adim = 0; adim < n_adims; adim++){
      owner_ref_csd[exec_rank][adim] = intersection_csds(owner_csd[exec_rank][adim], csd_ref[adim]);
    }
  }

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    reduce_csd(owner_ref_csd[exec_rank], n_adims);
  }

  if (myrank == 0){
    for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){

      printf("\n");
      printf("[%d]\n", exec_rank);

      printf("owner\n");
      for (int adim = 0; adim < n_adims; adim++){
  	printf("  %d: ", adim); print_csd(owner_csd[exec_rank][adim]);
      }

      /* printf("owner_ref\n"); */
      /* for (int adim = 0; adim < n_adims; adim++){ */
      /* 	printf("  %d: ", adim); print_csd(owner_ref_csd[exec_rank][adim]); */
      /* } */

    }
  }

  for (int adim = 0; adim < n_adims; adim++){
    free_csd(csd_ref[adim]);
  }

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    for (int adim = 0; adim < n_adims; adim++){
      free_csd(owner_csd[exec_rank][adim]);
    }
  }

}


size_t get_commbuf_size(_comm_set_t *comm_set[][_XMP_N_MAX_DIM], int ndims, int counts[]){

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  //int myrank = exec_nodes->comm_rank;
  int n_exec_nodes = exec_nodes->comm_size;

  size_t buf_size = 0;

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    size_t size = 1;
    for (int i = 0; i < ndims; i++){
      int dim_size = 0;
      _comm_set_t *c = comm_set[exec_rank][i];
      while (c){
	dim_size += (c->u - c->l + 1);
	c = c->next;
      }
      size *= dim_size;
    }
    counts[exec_rank] = size;
    //xmp_dbg_printf("buf_size[%d] = %d\n", exec_rank, size);
    buf_size += size;
  }

  return buf_size;

}


unsigned long long _XMP_gtol_calc_offset(_XMP_array_t *a, int g_idx[]){

  int l_idx[a->dim];
  xmp_array_gtol(a, g_idx, l_idx);

  unsigned long long offset = 0;

  for (int i = 0; i < a->dim; i++){
    offset += (l_idx[i] * a->info[i].dim_acc * a->type_size);
  }

  return offset;

}


void xmpc_pack_comm_set(void *sendbuf, _XMP_array_t *a, _comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  //int myrank = exec_nodes->comm_rank;
  int n_exec_nodes = exec_nodes->comm_size;

  int ndims = a->dim;
  //_XMP_array_info_t *ainfo = &(a->info[adim]);

  /* int tdim = ainfo->align_template_index; */
  /* _XMP_template_info_t *tinfo = &(a->align_template->info[tdim]); */
  /* _XMP_template_chunk_t *tchunk = &(a>align_template->chunk[tdim]); */

  /* int ndim = tchunk->onto_nodes_index; */

  char *buf = (char *)sendbuf;
  char *src = (char *)a->array_addr_p;

  for (int dst_node = 0; dst_node < n_exec_nodes; dst_node++){

    _comm_set_t *c[ndims];

    for (int i = 0; i < ndims; i++){
      c[i] = comm_set[dst_node][i];
      //print_comm_set(c[i]);
    }

    int i[_XMP_N_MAX_DIM];

    switch (ndims){

    case 1:
      while (c[0]){
	i[0] = c[0]->l;
	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
	c[0] = c[0]->next;
      }
      break;

    case 2:
      while (c[0]){ for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      while (c[1]){
	i[1] = c[1]->l;
	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
	//printf("c[1]->l = %d, c[1]-u = %d\n", c[1]->l, c[1]->u);
	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
	buf += size;
	c[1] = c[1]->next;
      }
      } c[0] = c[0]->next; }
      break;

    case 3:
      while (c[0]){ for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      while (c[1]){ for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      while (c[2]){
	i[2] = c[2]->l;
	int size = (c[2]->u - c[2]->l + 1) * a->type_size;
	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
	buf += size;
	c[2] = c[2]->next;
      }
      } c[1] = c[1]->next; }
      } c[0] = c[0]->next; }
      break;

    default:
      _XMP_fatal("wrong array dimension");
    }

  }

}


void xmpc_unpack_comm_set(void *recvbuf, _XMP_array_t *a, _comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  //int myrank = exec_nodes->comm_rank;
  int n_exec_nodes = exec_nodes->comm_size;

  int ndims = a->dim;
  //_XMP_array_info_t *ainfo = &(a->info[adim]);

  /* int tdim = ainfo->align_template_index; */
  /* _XMP_template_info_t *tinfo = &(a->align_template->info[tdim]); */
  /* _XMP_template_chunk_t *tchunk = &(a>align_template->chunk[tdim]); */

  /* int ndim = tchunk->onto_nodes_index; */

  char *buf = (char *)recvbuf;
  char *dst = (char *)a->array_addr_p;

  for (int dst_node = 0; dst_node < n_exec_nodes; dst_node++){

    _comm_set_t *c[ndims];

    for (int i = 0; i < ndims; i++){
      c[i] = comm_set[dst_node][i];
      //print_comm_set(c[i]);
    }

    int i[_XMP_N_MAX_DIM];

    switch (ndims){

    case 1:
      while (c[0]){
	i[0] = c[0]->l;
	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	c[0] = c[0]->next;
      }
      break;

    case 2:
      while (c[0]){ for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      while (c[1]){
	i[1] = c[1]->l;
	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
	c[1] = c[1]->next;
      }
      } c[0] = c[0]->next; }
      break;

    case 3:
      while (c[0]){ for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      while (c[1]){ for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      while (c[2]){
	i[2] = c[2]->l;
	int size = (c[2]->u - c[2]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
	c[2] = c[2]->next;
      }
      } c[1] = c[1]->next; }
      } c[0] = c[0]->next; }
      break;

    default:
      _XMP_fatal("wrong array dimension");
    }

  }

}


void XMP_gmove_1to1(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp){

  _XMP_array_t *lhs_array = gmv_desc_leftp->a_desc;
  int *lhs_lb = gmv_desc_leftp->lb;
  int *lhs_ub = gmv_desc_leftp->ub;
  int *lhs_st = gmv_desc_leftp->st;
  int n_lhs_dims = lhs_array->dim;

  _XMP_array_t *rhs_array = gmv_desc_rightp->a_desc;
  int *rhs_lb = gmv_desc_rightp->lb;
  int *rhs_ub = gmv_desc_rightp->ub;
  int *rhs_st = gmv_desc_rightp->st;
  int n_rhs_dims = rhs_array->dim;

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  int myrank = exec_nodes->comm_rank;
  int n_exec_nodes = exec_nodes->comm_size;
  MPI_Comm *exec_comm = exec_nodes->comm;

  //
  // LHS
  //

  if (myrank == 0){
    printf("\n");
    printf("LHS -------------------------------------\n");
  }

  // get referenced and owned section

  _csd_t *lhs_owner_ref_csd[n_exec_nodes][_XMP_N_MAX_DIM];
  get_owner_ref_csd(lhs_array, lhs_lb, lhs_ub, lhs_st, lhs_owner_ref_csd);

  //
  // RHS
  //

  if (myrank == 0){
    printf("\n");
    printf("RHS -------------------------------------\n");
  }

  // get referenced and owned section

  _csd_t *rhs_owner_ref_csd[n_exec_nodes][_XMP_N_MAX_DIM];
  get_owner_ref_csd(rhs_array, rhs_lb, rhs_ub, rhs_st, rhs_owner_ref_csd);

  //
  // get send list
  //

  if (myrank == 0){
    printf("\n");
    printf("Send List -------------------------------------\n");
  }

  _csd_t *send_csd[n_exec_nodes][_XMP_N_MAX_DIM];

  for (int dst_node = 0; dst_node < n_exec_nodes; dst_node++){

    _csd_t *r;

    int lhs_adim = 0;
    for (int rhs_adim = 0; rhs_adim < n_rhs_dims; rhs_adim++){

      if (rhs_st[rhs_adim] == 0){
  	r = alloc_csd(1);
  	r->l[0] = rhs_lb[rhs_adim];
  	r->u[0] = rhs_lb[rhs_adim];
  	r->s = 1;
      }
      else {
      	while (lhs_st[lhs_adim] == 0 && lhs_adim < n_lhs_dims) lhs_adim++;
      	if (lhs_adim == n_lhs_dims) _XMP_fatal("_XMP_gmove_1to1: lhs and rhs not conformable");
      	_csd_t *l = lhs_owner_ref_csd[dst_node][lhs_adim];
	if (l){
	  r = alloc_csd(l->b);
	  for (int i = 0; i < l->b; i++){
	    r->l[i] = l->l[i] + rhs_lb[rhs_adim] - lhs_lb[lhs_adim];
	    r->u[i] = l->u[i] + rhs_lb[rhs_adim] - lhs_lb[lhs_adim];
	  }
	  r->s = l->s * rhs_st[rhs_adim] / lhs_st[lhs_adim];
	}
	else {
	  r = NULL;
	}
      	lhs_adim++;
      }

      send_csd[dst_node][rhs_adim] = intersection_csds(r, rhs_owner_ref_csd[myrank][rhs_adim]);

      if (r) free_csd(r);

    }

    reduce_csd(send_csd[dst_node], n_rhs_dims);

  }

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    if (myrank == exec_rank){
      for (int dst_rank = 0; dst_rank < n_exec_nodes; dst_rank++){

	printf("\n");
	printf("[%d] -> [%d]\n", myrank, dst_rank);

	for (int adim = 0; adim < n_rhs_dims; adim++){
	  printf("  %d: ", adim); print_csd(send_csd[dst_rank][adim]);
	}
      }
    }
    fflush(stdout);
    xmp_barrier();
  }

  //
  // get recv list
  //

  if (myrank == 0){
    printf("\n");
    printf("Recv List -------------------------------------\n");
  }

  _csd_t *recv_csd[n_exec_nodes][_XMP_N_MAX_DIM];

  for (int src_node = 0; src_node < n_exec_nodes; src_node++){

    _csd_t *l;

    int rhs_adim = 0;
    for (int lhs_adim = 0; lhs_adim < n_lhs_dims; lhs_adim++){

      if (lhs_st[lhs_adim] == 0){
  	l = alloc_csd(1);
  	l->l[0] = lhs_lb[lhs_adim];
  	l->u[0] = lhs_lb[lhs_adim];
  	l->s = 1;
      }
      else {
      	while (rhs_st[rhs_adim] == 0 && rhs_adim < n_rhs_dims) rhs_adim++;
      	if (rhs_adim == n_rhs_dims) _XMP_fatal("_XMP_gmove_1to1: lhs and rhs not conformable");
      	_csd_t *r = rhs_owner_ref_csd[src_node][rhs_adim];
	if (r){
	  l = alloc_csd(r->b);
	  for (int i = 0; i < r->b; i++){
	    l->l[i] = r->l[i] + lhs_lb[lhs_adim] - rhs_lb[rhs_adim];
	    l->u[i] = r->u[i] + lhs_lb[lhs_adim] - rhs_lb[rhs_adim];
	  }
	  l->s = r->s * lhs_st[lhs_adim] / rhs_st[rhs_adim];
	}
	else {
	  l = NULL;
	}
      	rhs_adim++;
      }

      recv_csd[src_node][lhs_adim] = intersection_csds(l, lhs_owner_ref_csd[myrank][lhs_adim]);

      if (l) free_csd(l);

    }

    reduce_csd(recv_csd[src_node], n_lhs_dims);

  }

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    if (myrank == exec_rank){
      for (int src_rank = 0; src_rank < n_exec_nodes; src_rank++){

	printf("\n");
	printf("[%d] <- [%d]\n", myrank, src_rank);

	for (int adim = 0; adim < n_lhs_dims; adim++){
	  printf("  %d: ", adim); print_csd(recv_csd[src_rank][adim]);
	}
      }
    }
    fflush(stdout);
    xmp_barrier();
  }

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    for (int adim = 0; adim < n_lhs_dims; adim++){
      free_csd(lhs_owner_ref_csd[exec_rank][adim]);
    }
  }

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    for (int adim = 0; adim < n_rhs_dims; adim++){
      free_csd(rhs_owner_ref_csd[exec_rank][adim]);
    }
  }

  //
  // Get communication sets
  //

  _comm_set_t *send_comm_set[n_exec_nodes][_XMP_N_MAX_DIM];

  for (int dst_node = 0; dst_node < n_exec_nodes; dst_node++){
    for (int adim = 0; adim < n_rhs_dims; adim++){
      send_comm_set[dst_node][adim] = csd2comm_set(send_csd[dst_node][adim]);
    }
  }

  for (int dst_node = 0; dst_node < n_exec_nodes; dst_node++){
    for (int adim = 0; adim < n_rhs_dims; adim++){
      free_csd(send_csd[dst_node][adim]);
    }
  }

  _comm_set_t *recv_comm_set[n_exec_nodes][_XMP_N_MAX_DIM];

  for (int src_node = 0; src_node < n_exec_nodes; src_node++){
    for (int adim = 0; adim < n_lhs_dims; adim++){
      recv_comm_set[src_node][adim] = csd2comm_set(recv_csd[src_node][adim]);
    }
  }

  for (int src_node = 0; src_node < n_exec_nodes; src_node++){
    for (int adim = 0; adim < n_lhs_dims; adim++){
      free_csd(recv_csd[src_node][adim]);
    }
  }

  //
  // Allocate buffers
  //

  // send buffer

  int sendcounts[n_exec_nodes];
  size_t sendbuf_size = get_commbuf_size(send_comm_set, n_rhs_dims, sendcounts);
  void *sendbuf = _XMP_alloc(sendbuf_size * rhs_array->type_size);

  int sdispls[n_exec_nodes];
  sdispls[0] = 0;
  for (int i = 1; i < n_exec_nodes; i++){
    sdispls[i] = sdispls[i-1] + sendcounts[i-1];
  }

  // recv buffer

  int recvcounts[n_exec_nodes];
  size_t recvbuf_size = get_commbuf_size(recv_comm_set, n_lhs_dims, recvcounts);
  void *recvbuf = _XMP_alloc(recvbuf_size * lhs_array->type_size);

  int rdispls[n_exec_nodes];
  rdispls[0] = 0;
  for (int i = 1; i < n_exec_nodes; i++){
    rdispls[i] = rdispls[i-1] + recvcounts[i-1];
  }

  //
  // Packing
  //

  xmpc_pack_comm_set(sendbuf, rhs_array, send_comm_set);

  //
  // communication
  //

  MPI_Alltoallv(sendbuf, sendcounts, sdispls, rhs_array->mpi_type,
  		recvbuf, recvcounts, rdispls, lhs_array->mpi_type,
  		*exec_comm);

  //
  // Unpack
  //

  xmpc_unpack_comm_set(recvbuf, lhs_array, send_comm_set);

  //
  // Deallocate temporarls
  //

  _XMP_free(sendbuf);
  _XMP_free(recvbuf);

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    for (int adim = 0; adim < n_rhs_dims; adim++){
      free_comm_set(send_comm_set[exec_rank][adim]);
    }
  }

  for (int exec_rank = 0; exec_rank < n_exec_nodes; exec_rank++){
    for (int adim = 0; adim < n_lhs_dims; adim++){
      free_comm_set(recv_comm_set[exec_rank][adim]);
    }
  }

}


void _XMP_gmove_SENDRECV_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array,
                               int type, size_t type_size, ...) {

  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim];
  unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    if (dst_s[i] != 0) dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  int src_dim = src_array->dim;;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim];
  unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_u[i] = src_l[i] + (size - 1) * src_s[i];
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    if (src_s[i] != 0) src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  } else {
    //gmove_total_elmts = dst_total_elmts;
  }

  int dummy[7] = { 2, 2, 2, 2, 2, 2, 2 }; /* temporarily assuming maximum 7-dimensional */

  gmv_desc_leftp.is_global = true;       gmv_desc_rightp.is_global = true;
  gmv_desc_leftp.ndims = dst_array->dim; gmv_desc_rightp.ndims = src_array->dim;

  gmv_desc_leftp.a_desc = dst_array;     gmv_desc_rightp.a_desc = src_array;

  gmv_desc_leftp.local_data = NULL;      gmv_desc_rightp.local_data = NULL;
  gmv_desc_leftp.a_lb = NULL;            gmv_desc_rightp.a_lb = NULL;
  gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

  gmv_desc_leftp.kind = dummy;           gmv_desc_rightp.kind = dummy; // always triplet
  gmv_desc_leftp.lb = dst_l;             gmv_desc_rightp.lb = src_l;
  gmv_desc_leftp.ub = dst_u;             gmv_desc_rightp.ub = src_u;
  gmv_desc_leftp.st = dst_s;             gmv_desc_rightp.st = src_s;

  XMP_gmove_1to1(&gmv_desc_leftp, &gmv_desc_rightp);
}
