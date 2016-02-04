#include <string.h>
#include <stdlib.h>
#include "xmp_internal.h"
#include "xmp_math_function.h"

void print_rsd(_XMP_rsd_t *rsd){

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


void print_bsd(_XMP_bsd_t *bsd){
  if (!bsd){
    printf("()\n");
  }
  else {
    printf("(%d : %d : %d : %d)\n", bsd->l, bsd->u, bsd->b, bsd->c);
  }
}


void print_csd(_XMP_csd_t *csd){

  if (!csd || csd->n == 0){
    printf("()\n");
    return;
  }

  printf("(");

  printf("(%d", csd->l[0]);
  for (int i = 1; i < csd->n; i++){
    printf(", %d", csd->l[i]);
  }
  printf(")");

  printf(" : ");

  printf("(%d", csd->u[0]);
  for (int i = 1; i < csd->n; i++){
    printf(", %d", csd->u[i]);
  }
  printf(")");

  printf(" : ");

  printf("%d)\n", csd->s);

}


void print_comm_set(_XMP_comm_set_t *comm_set0){

  if (!comm_set0){
    printf("()\n");
    return;
  }

  _XMP_comm_set_t *comm_set = comm_set0;

  printf("(%d : %d)", comm_set->l, comm_set->u);

  while ((comm_set = comm_set->next)){
    printf(", (%d : %d)", comm_set->l, comm_set->u);
  }

  printf("\n");

}


_XMP_rsd_t *intersection_rsds(_XMP_rsd_t *_rsd1, _XMP_rsd_t *_rsd2){

  _XMP_rsd_t *rsd1, *rsd2;

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
	_XMP_rsd_t *rsd0 = (_XMP_rsd_t *)_XMP_alloc(sizeof(_XMP_rsd_t));
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


_XMP_csd_t *intersection_csds(_XMP_csd_t *csd1, _XMP_csd_t *csd2){

  if (!csd1 || !csd2) return NULL;

  _XMP_csd_t *csd0 = alloc_csd(MAX(csd1->n, csd2->n));

  int k = 0;

  csd0->n = 0;

  for (int i = 0; i < csd1->n; i++){
    for (int j = 0; j < csd2->n; j++){

      _XMP_rsd_t *tmp = NULL;
      _XMP_rsd_t rsd1 = { csd1->l[i], csd1->u[i], csd1->s };
      _XMP_rsd_t rsd2 = { csd2->l[j], csd2->u[j], csd2->s };

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

	csd0->n++;
	csd0->s = tmp->s;
	k++;

	_XMP_free(tmp);
      }

    }
  }

  return csd0;

}


_XMP_csd_t *alloc_csd(int n){
  _XMP_csd_t *csd = (_XMP_csd_t *)_XMP_alloc(sizeof(_XMP_csd_t));
  csd->l = (int *)_XMP_alloc(sizeof(int) * n);
  csd->u = (int *)_XMP_alloc(sizeof(int) * n);
  csd->n = n;
  return csd;
}


void free_csd(_XMP_csd_t *csd){
  if (csd){
    _XMP_free(csd->l);
    _XMP_free(csd->u);
    _XMP_free(csd);
  }
}


_XMP_csd_t *copy_csd(_XMP_csd_t *csd){
  _XMP_csd_t *new_csd = alloc_csd(csd->n);
  for (int i = 0; i < csd->n; i++){
    new_csd->l[i] = csd->l[i];
    new_csd->u[i] = csd->u[i];
  }
  new_csd->s = csd->s;
  return new_csd;
}


int get_csd_size(_XMP_csd_t *csd){
  int size = 0;
  for (int i = 0; i < csd->n; i++){
    size += _XMP_M_COUNT_TRIPLETi(csd->l[i], csd->u[i], csd->s);
  }
  return size;
}


void free_comm_set(_XMP_comm_set_t *comm_set){

  while (comm_set){
    _XMP_comm_set_t *next = comm_set->next;
    _XMP_free(comm_set);
    comm_set = next;
  }

}


_XMP_csd_t *rsd2csd(_XMP_rsd_t *rsd){
  if (!rsd) return NULL;
  _XMP_csd_t *csd = alloc_csd(1);
  csd->l[0] = rsd->l;
  csd->u[0] = rsd->u;
  csd->n = 1;
  csd->s = rsd->s;
  return csd;
}


_XMP_csd_t *bsd2csd(_XMP_bsd_t *bsd){

  if (!bsd) return NULL;

  _XMP_csd_t *csd = alloc_csd(bsd->b);
  csd->n = bsd->b;

  for (int i = 0; i < bsd->b; i++){
    csd->l[i] = bsd->l + i;
    int t = (bsd->u - csd->l[i]) / bsd->c;
    csd->u[i] = csd->l[i] + bsd->c * t;
  }

  csd->s = bsd->c;

  return csd;

}


_XMP_comm_set_t *csd2comm_set(_XMP_csd_t *csd){

  if (!csd || csd->n == 0) return NULL;

  _XMP_comm_set_t *comm_set0 = (_XMP_comm_set_t *)_XMP_alloc(sizeof(_XMP_comm_set_t));

  _XMP_comm_set_t *comm_set = comm_set0;
  comm_set->l = csd->l[0];
  comm_set->u = csd->l[0];
  comm_set->next = NULL;

  for (int j = 0; csd->l[0] + j <= csd->u[0]; j+= csd->s){

    for (int i = 0; i < csd->n; i++){

      int l = csd->l[i] + j;

      if (l > csd->u[i]) continue;

      if (l == comm_set->u + 1){
	comm_set->u = l;
      }
      else if (l <= comm_set->u){
	continue;
      }
      else {
	comm_set->next = (_XMP_comm_set_t *)malloc(sizeof(_XMP_comm_set_t));
	comm_set = comm_set->next;
	comm_set->l = l;
	comm_set->u = l;
	comm_set->next = NULL;
      }
    }

  }

  return comm_set0;

}


void reduce_csd(_XMP_csd_t *csd[_XMP_N_MAX_DIM], int ndims){

  for (int i = 0; i < ndims; i++){
    if (!csd[i] || csd[i]->n == 0){
      for (int j = 0; j < ndims; j++){
	free_csd(csd[j]);
	csd[j] = NULL;
      }
      return;
    }
  }      

}
