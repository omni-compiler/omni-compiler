#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "xmp_internal.h"

static void sort_up(void *a, void *b, int n);
static void comp_pivots(void *a, int n, void *pivot);
static void make_histogram(void *a, int n, int *size, int *start, void *pivot);
static void dist_keys(void *a, int *size, int *start,
		      void **buf, int *bufSize, int *bufStart);
static void kway_inplace_merge_sort(void *a, int *start, int k);


#define PROTOTYPE_COMPARE(_type) \
  static int compare_##_type##s(const void *a, const void *b);

#define PROTOTYPE_PIVOT(_type) \
  static void get_rotate_pivot_for_##_type(void *p, const void *a, const int an, \
					     const void *b, const int bn);

PROTOTYPE_COMPARE(int)
PROTOTYPE_COMPARE(float)

PROTOTYPE_PIVOT(int)
PROTOTYPE_PIVOT(float)


MPI_Comm *comm;
int me;
int nprocs;

int (*compare_func)(const void *a, const void *b);
void (*get_rotate_pivot)(void *p, const void *a, const int an, const void *b, const int bn);
size_t datasize;

//#define dbg_printf printf

void dbg_printf(char *fmt, ...){
  ;
}

#define THRESHOLD 100

static void set_funcs(_XMP_array_t *a_desc);

void xmp_sort_up(_XMP_array_t *a_desc, _XMP_array_t *b_desc){

  void *a_adr = a_desc->array_addr_p;
  int a_lshadow = a_desc->info[0].shadow_size_lo;

  void *b_adr = b_desc->array_addr_p;
  int b_lshadow = b_desc->info[0].shadow_size_lo;

  int size = a_desc->info[0].par_size;

  comm = a_desc->array_nodes->comm;
  me = a_desc->array_nodes->comm_rank;
  nprocs = a_desc->array_nodes->comm_size;

  set_funcs(a_desc);
  datasize = a_desc->type_size;

  sort_up((char *)a_adr + a_lshadow * datasize,
	  (char *)b_adr + b_lshadow * datasize, size);
}


static void set_funcs(_XMP_array_t *a_desc){

  switch (a_desc->type){

  case _XMP_N_TYPE_BOOL:
    break;

  case _XMP_N_TYPE_CHAR:
  case _XMP_N_TYPE_UNSIGNED_CHAR:
    break;

  case _XMP_N_TYPE_SHORT:
  case _XMP_N_TYPE_UNSIGNED_SHORT:
    break;

  case _XMP_N_TYPE_INT:
  case _XMP_N_TYPE_UNSIGNED_INT:
    compare_func = compare_ints;
    get_rotate_pivot = get_rotate_pivot_for_int;
    break;

  case _XMP_N_TYPE_LONG:
  case _XMP_N_TYPE_UNSIGNED_LONG:
    break;

  case _XMP_N_TYPE_LONGLONG:
  case _XMP_N_TYPE_UNSIGNED_LONGLONG:
    break;

  case _XMP_N_TYPE_FLOAT:
  case _XMP_N_TYPE_FLOAT_IMAGINARY:
    compare_func = compare_floats;
    get_rotate_pivot = get_rotate_pivot_for_float;
    break;

  case _XMP_N_TYPE_DOUBLE:
  case _XMP_N_TYPE_DOUBLE_IMAGINARY:
    break;

  case _XMP_N_TYPE_LONG_DOUBLE:
  case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
    break;

  case _XMP_N_TYPE_FLOAT_COMPLEX:
    break;

  case _XMP_N_TYPE_DOUBLE_COMPLEX:
    break;

  case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
    break;

  case _XMP_N_TYPE_NONBASIC:
  default:
    break;
  }

}


static void sort_up(void *a, void *b, int n){

  void *pivot = malloc(datasize * nprocs);

  int size[nprocs];
  int start[nprocs];

  void *buf;
  int bufSize[nprocs];
  int bufStart[nprocs + 1];

  // local sort
  //dbg_printf("(%d) local sort\n", me);
  qsort(a, n, datasize, compare_func);

  // compute pivots
  //dbg_printf("(%d) comp_pivots\n", me);
  comp_pivots(a, n, pivot);

  // make histogram
  //dbg_printf("(%d) make_histogram\n", me);
  make_histogram(a, n, size, start, pivot);

  // distribute keys
  //dbg_printf("(%d) dist_keys\n", me);
  dist_keys(a, size, start, &buf, bufSize, bufStart);

  // merge_keys
  //dbg_printf("(%d) merge_sort\n", me);
  kway_inplace_merge_sort(buf, bufStart, nprocs);

  //dbg_printf("(%d) finished\n", me);
  MPI_Barrier(*comm);

  /* if (me == 2){ */
  /*   dbg_printf("(%d) b = ( ", me); */
  /* for (int i = 0; i < bufStart[nprocs]; i++){ */
  /*   dbg_printf("%.1f ", ((float *)buf)[i]); */
  /* } */
  /* dbg_printf(")\n"); */
  /* } */

  for (int i = 1; i < bufStart[nprocs]; i++){
    if (compare_func(buf + datasize * (i-1), buf + datasize * i) > 0){
      printf("(%d) verification failed. %.1f > %.1f\n", me, ((float *)buf)[i-1], ((float *)buf)[i]);
    }
  }

  MPI_Barrier(*comm);

  if (me == 0)
    printf("done\n");

  free(pivot);
  free(buf);

}


static void comp_pivots(void *a, int n, void *pivot){

  if (n < nprocs){
    fprintf(stderr, "a is too small\n");
    exit(1);
  }

  void *tmp = calloc(nprocs * nprocs, datasize);
  //int tmp[nprocs * nprocs];

  /* for (int i = 0; i < nprocs * nprocs; i++){ */
  /*   tmp[i] = 0; */
  /* } */

  // Sampling
  for (int i = 0; i < nprocs; i++){
    //tmp[me * nprocs + i] = a[(i * n) / nprocs];
    memcpy(tmp + datasize * (me * nprocs + i), a + datasize * ((i * n) / nprocs),
	   datasize);
  }

  MPI_Allgather(MPI_IN_PLACE, nprocs /* dummy */, MPI_INT /* dummy */,
		tmp, nprocs, MPI_INT, *comm);

  qsort(tmp, nprocs * nprocs, datasize, compare_func);

  // get final pivot
  for (int i = 0; i < nprocs - 1; i++){
    //pivot[i] = tmp[(i + 1) * nprocs - 1];
    memcpy(pivot + datasize * i, tmp + datasize * ((i + 1) * nprocs - 1), datasize);
  }				

  //pivot[nprocs - 1] = -1; /* dummy */

}


static void make_histogram(void *a, int n, int *size, int *start, void *pivot){

  int j = 0;
  for (int i = 0; i < nprocs - 1; i++){
    start[i] = j;
    size[i] = 0;
    while (j < n &&
	   compare_func((char *)a + j * datasize, (char *)pivot + i * datasize) <= 0){
      size[i]++;
      j++;
    }
  }

  start[nprocs - 1] = j;
  size[nprocs - 1] = n - j;

}


static void dist_keys(void *a, int *size, int *start,
		      void **buf, int *bufSize, int *bufStart){

  MPI_Alltoall(size, 1, MPI_INT,
	       bufSize, 1, MPI_INT,
	       *comm);

  bufStart[0] = 0;
  for (int i = 1; i <= nprocs; i++){
    bufStart[i] = bufStart[i - 1] + bufSize[i - 1];
  }

  *buf = malloc(datasize * bufStart[nprocs]);

  MPI_Alltoallv(a, size, start, MPI_INT,
		*buf, bufSize, bufStart, MPI_INT,
		*comm);
}


static void my_bsearch(void *p, void *a, int n, int *m){

  if (n < THRESHOLD){
    int i = 0;
    for (i = 0; i < n; i++){
      if (compare_func(a + i * datasize, p) > 0) break; // if a[i] > p
    }
    *m = i;
    return;
  }

  int mid = n / 2;

  if (compare_func(a + mid * datasize, p) < 0){ // if a[mid] < p
    my_bsearch(p, a + mid * datasize, n - mid, m);
    *m = mid + (*m);
  }
  else if (compare_func(a + mid * datasize, p) > 0){ // if a[mid] > p
    my_bsearch(p, a, mid, m);
  }
  else { // if a[mid] == p
    int i = mid;
    while (compare_func(a + i * datasize, p) == 0) i++; // if a[mid] == p
    *m = i;
  }
  
}


static void reverse(void *a, int l, int r){
  int tmp;
  while (l < r){
    memcpy(&tmp, (char *)a + r * datasize, datasize);
    memcpy((char *)a + r * datasize, (char *)a + l * datasize, datasize);
    memcpy((char *)a + l * datasize, &tmp, datasize);
    /* memcpy(&tmp, &a[r], datasize); */
    /* memcpy(&a[r], &a[l], datasize); */
    /* memcpy(&a[r], &tmp, datasize); */
    /* int tmp = a[r]; */
    /* a[r] = a[l]; */
    /* a[l] = tmp; */
    l++; r--;
  }
}


static void rotate(void *a, int l, int m, int r){
  reverse(a, l, m-1);
  reverse(a, m, r);
  reverse(a, l, r);
}


static void two_way_inplace_merge_sort(void *a, int n, int m){

  if (me == 2){
    dbg_printf("(%d) two_way_inplace_merge_sort\n", me);
    dbg_printf("(%d)   n = %d, m = %d\n", me, n, m);
    dbg_printf("(%d)   a = ( ", me);
    for (int i = 0; i < n; i++){
      dbg_printf("%.1f ", ((float *)a)[i]);
    }
    dbg_printf(")\n");
  }

  if (m <= 0 || m > n) return;

  if (n < THRESHOLD){
    qsort(a, n, datasize, compare_func);
    return;
  }

  int an = m;

  void *b = (char *)a + m * datasize;
  int bn = n - m;

  float p = 0;
  //long long p = 0;

  get_rotate_pivot(&p, a, an, b, bn);

  int mid_a, mid_b;

  my_bsearch(&p, a, an, &mid_a);
  my_bsearch(&p, b, bn, &mid_b);

  if (me == 2){
    dbg_printf("(%d) p = %.1f, mid_a = %d, mid_b = %d\n", me, (float)p, mid_a, mid_b);

    dbg_printf("(%d) before ( ", me);
    for (int i = 0; i < n; i++){
      dbg_printf("%.1f ", ((float *)a)[i]);
    }
    dbg_printf(")\n");
  }
  rotate(a, mid_a, m, mid_b + m - 1);
  if (me == 2){
    dbg_printf("(%d) after ( ", me);
    for (int i = 0; i < n; i++){
      dbg_printf("%.1f ", ((float *)a)[i]);
    }
    dbg_printf(")\n");
  }

  int new_an = mid_a + mid_b;
  int new_mid_a = mid_a;
  int new_bn = n - new_an;
  int new_mid_b = m - mid_a;;

  if (mid_a > 0 && mid_a < an)
    two_way_inplace_merge_sort(a, new_an, new_mid_a);
  if (mid_b > 0 && mid_b < bn)
    two_way_inplace_merge_sort((char *)a + new_an * datasize, new_bn, new_mid_b);

}


static void kway_inplace_merge_sort(void *a, int *start, int k){

  int an = start[k] - start[0];

  /* if (me == 2){ */
  /*   dbg_printf("(%d) A = ( ", me); */
  /*   for (int i = 0; i < an; i++){ */
  /*     dbg_printf("%.1f ", ((float *)a)[i]); */
  /*   } */
  /*   dbg_printf("), k = %d, an = %d\n", k, an); */
  /* } */

  if (an < THRESHOLD){
    qsort(a, an, datasize, compare_func);
    return;
  }

  if (k == 1){
    return;
  }
  else if (k == 2){
    two_way_inplace_merge_sort(a, an,  start[1] - start[0]);
    return;
  }
  else {
    int m = k / 2;
    kway_inplace_merge_sort(a, start, m);
    kway_inplace_merge_sort((char *)a + start[m] * datasize, start + m, k - m);
  if (me == 2){
    dbg_printf("(%d)  d = ( ", me);
    for (int i = 0; i < an; i++){
      dbg_printf("%.1f ", ((float *)a)[i]);
    }
    dbg_printf(")\n");
    dbg_printf("\nBEFORE 2way\n\n");
  }
    two_way_inplace_merge_sort(a, an, start[m] - start[0]);
  if (me == 2){
    dbg_printf("(%d)  c = ( ", me);
    for (int i = 0; i < an; i++){
      dbg_printf("%.1f ", ((float *)a)[i]);
    }
    dbg_printf(")\n");
  }
    return;
  }

}


//
//
//

#define COMPARE(_type) \
static int compare_##_type##s(const void *a, const void *b){ \
  if (*(_type *)a < *(_type *)b){ \
    return -1; \
  } \
  else if(*(_type *)a == *(_type *)b){ \
    return 0; \
  } \
  return 1; \
}

COMPARE(int)
COMPARE(float)

#define PIVOT(_type) \
static void get_rotate_pivot_for_##_type(void *p, const void *a, const int an, \
				                  const void *b, const int bn){ \
  _type am, bm; \
  if (an > 0){ \
    if (an % 2 == 0) \
      am = (((_type *)a)[an / 2] + ((_type *)a)[an / 2 - 1]) / 2; \
    else \
      am = ((_type *)a)[an / 2]; \
  } \
  if (bn > 0){ \
    if (bn % 2 == 0) \
      bm = (((_type *)b)[bn / 2] + ((_type *)b)[bn / 2 - 1]) / 2; \
    else \
      bm = ((_type *)b)[bn / 2]; \
  } \
  _type tmp; \
  if (an > 0 && bn > 0) tmp = (am + bm) / 2; \
  else if (an > 0 && bn == 0) tmp = am; \
  else if (an == 0 && bn > 0) tmp = bm; \
  memcpy(p, &tmp, datasize); \
}

PIVOT(int)
PIVOT(float)

/* static void get_rotate_pivot_for_int(void *p, const void *a, const int an, */
/* 				              const void *b, const int bn){ */

/*   int am, bm; */

/*   if (an > 0){ */
/*     if (an % 2 == 0) */
/*       am = (((int *)a)[an / 2] + ((int *)a)[an / 2 - 1]) / 2; */
/*     else */
/*       am = ((int *)a)[an / 2]; */
/*   } */

/*   if (bn > 0){ */
/*     if (bn % 2 == 0) */
/*       bm = (((int *)b)[bn / 2] + ((int *)b)[bn / 2 - 1]) / 2; */
/*     else */
/*       bm = ((int *)b)[bn / 2]; */
/*   } */

/*   int tmp; */

/*   // weighted mean is better? */
/*   if (an > 0 && bn > 0) tmp = (am + bm) / 2; */
/*   else if (an > 0 && bn == 0) tmp = am; */
/*   else if (an == 0 && bn > 0) tmp = bm; */

/*   memcpy(p, &tmp, datasize); */

/* } */
