#include <stdlib.h>
#include <string.h>
#include "xmp_internal.h"

static void set_funcs(_XMP_array_t *a_desc, int is_up);
static void do_sort(void *a, int n, _XMP_array_t *b_desc);
static void comp_pivots(void *a, int n, void *pivot);
static void make_histogram(void *a, int n, int *size, int *start, void *pivot);
static void dist_keys(void *a, int *size, int *start,
		      void **buf, int *bufSize, int *bufStart);
static void kway_inplace_merge_sort(void *a, int *start, int k);
static void do_gmove(void *buf, int *bufStart, _XMP_array_t *b_desc);


static int compare_up_SHORT            (const void *a, const void *b);
static int compare_up_UNSIGNED_SHORT   (const void *a, const void *b);
static int compare_up_INT              (const void *a, const void *b);
static int compare_up_UNSIGNED_INT     (const void *a, const void *b);
static int compare_up_LONG             (const void *a, const void *b);
static int compare_up_UNSIGNED_LONG    (const void *a, const void *b);
static int compare_up_LONGLONG         (const void *a, const void *b);
static int compare_up_UNSIGNED_LONGLONG(const void *a, const void *b);
static int compare_up_FLOAT            (const void *a, const void *b);
static int compare_up_DOUBLE           (const void *a, const void *b);
static int compare_up_LONG_DOUBLE      (const void *a, const void *b);

static int compare_down_SHORT            (const void *a, const void *b);
static int compare_down_UNSIGNED_SHORT   (const void *a, const void *b);
static int compare_down_INT              (const void *a, const void *b);
static int compare_down_UNSIGNED_INT     (const void *a, const void *b);
static int compare_down_LONG             (const void *a, const void *b);
static int compare_down_UNSIGNED_LONG    (const void *a, const void *b);
static int compare_down_LONGLONG         (const void *a, const void *b);
static int compare_down_UNSIGNED_LONGLONG(const void *a, const void *b);
static int compare_down_FLOAT            (const void *a, const void *b);
static int compare_down_DOUBLE           (const void *a, const void *b);
static int compare_down_LONG_DOUBLE      (const void *a, const void *b);

static void get_rotate_pivot_for_SHORT            (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_UNSIGNED_SHORT   (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_INT              (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_UNSIGNED_INT     (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_LONG             (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_UNSIGNED_LONG    (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_LONGLONG         (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_UNSIGNED_LONGLONG(void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_FLOAT            (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_DOUBLE           (void *p, const void *a, const int an,
						   const void *b, const int bn);
static void get_rotate_pivot_for_LONG_DOUBLE      (void *p, const void *a, const int an,
						   const void *b, const int bn);

int dbg_flag = 0;

MPI_Comm *comm;
int me;
int nprocs;

int (*compare_func)(const void *a, const void *b);
void (*get_rotate_pivot)(void *p, const void *a, const int an, const void *b, const int bn);
size_t datasize;
MPI_Datatype mpi_type;

//#define dbg_printf printf

void dbg_printf(char *fmt, ...){
  ;
}

#define THRESHOLD_BSEARCH 8
#define THRESHOLD_2WAY 32
#define THRESHOLD_KWAY THRESHOLD_2WAY*2


void xmp_sort_up(_XMP_array_t *a_desc, _XMP_array_t *b_desc){
  _XMP_sort(a_desc, b_desc, 1);
}


void xmp_sort_down(_XMP_array_t *a_desc, _XMP_array_t *b_desc){
  _XMP_sort(a_desc, b_desc, 0);
}


void _XMP_sort(_XMP_array_t *a_desc, _XMP_array_t *b_desc, int is_up){

  void *a_adr = a_desc->array_addr_p;
  int a_lshadow = a_desc->info[0].shadow_size_lo;

  int a_size = a_desc->info[0].ser_size;
  int a_ndims = a_desc->dim;
  int a_type = a_desc->type;

  int b_size = b_desc->info[0].ser_size;
  int b_ndims = b_desc->dim;
  int b_type = b_desc->type;

  if (a_size != b_size) _XMP_fatal("xmp_sort: different size for array arguments");
  if (a_ndims != 1 || b_ndims != 1) _XMP_fatal("xmp_sort: array arguments must be one-dimensional");
  if (a_type != b_type) _XMP_fatal("xmp_sort: different type for array arguments");

  int lsize = a_desc->info[0].par_size;

  comm = a_desc->array_nodes->comm;
  me = a_desc->array_nodes->comm_rank;
  nprocs = a_desc->array_nodes->comm_size;

  set_funcs(a_desc, is_up);
  datasize = a_desc->type_size;

  do_sort((char *)a_adr + a_lshadow * datasize, lsize, b_desc);

}


static void set_funcs(_XMP_array_t *a_desc, int is_up){

  switch (a_desc->type){

  case _XMP_N_TYPE_SHORT:
    compare_func = is_up ? compare_up_SHORT : compare_down_SHORT;
    get_rotate_pivot = get_rotate_pivot_for_SHORT;
    mpi_type = MPI_SHORT;
    break;

  case _XMP_N_TYPE_UNSIGNED_SHORT:
    compare_func = is_up ? compare_up_UNSIGNED_SHORT : compare_down_UNSIGNED_SHORT;
    get_rotate_pivot = get_rotate_pivot_for_UNSIGNED_SHORT;
    mpi_type = MPI_UNSIGNED_SHORT;
    break;

  case _XMP_N_TYPE_INT:
    compare_func = is_up ? compare_up_INT : compare_down_INT;
    get_rotate_pivot = get_rotate_pivot_for_INT;
    mpi_type = MPI_INT;
    break;

  case _XMP_N_TYPE_UNSIGNED_INT:
    compare_func = is_up ? compare_up_UNSIGNED_INT : compare_down_UNSIGNED_INT;
    get_rotate_pivot = get_rotate_pivot_for_UNSIGNED_INT;
    mpi_type = MPI_UNSIGNED;
    break;

  case _XMP_N_TYPE_LONG:
    compare_func = is_up ? compare_up_LONG : compare_down_LONG;
    get_rotate_pivot = get_rotate_pivot_for_LONG;
    mpi_type = MPI_LONG;
    break;

  case _XMP_N_TYPE_UNSIGNED_LONG:
    compare_func = is_up ? compare_up_UNSIGNED_LONG : compare_down_UNSIGNED_LONG;
    get_rotate_pivot = get_rotate_pivot_for_UNSIGNED_LONG;
    mpi_type = MPI_UNSIGNED_LONG;
    break;

  case _XMP_N_TYPE_LONGLONG:
    compare_func = is_up ? compare_up_LONGLONG : compare_down_LONGLONG;
    get_rotate_pivot = get_rotate_pivot_for_LONGLONG;
    mpi_type = MPI_LONG_LONG;
    break;

  case _XMP_N_TYPE_UNSIGNED_LONGLONG:
    compare_func = is_up ? compare_up_UNSIGNED_LONGLONG : compare_down_UNSIGNED_LONGLONG;
    get_rotate_pivot = get_rotate_pivot_for_UNSIGNED_LONGLONG;
    mpi_type = MPI_UNSIGNED_LONG_LONG;
    break;

  case _XMP_N_TYPE_FLOAT:
  case _XMP_N_TYPE_FLOAT_IMAGINARY:
    compare_func = is_up ? compare_up_FLOAT : compare_down_FLOAT;
    get_rotate_pivot = get_rotate_pivot_for_FLOAT;
    mpi_type = MPI_FLOAT;
    break;

  case _XMP_N_TYPE_DOUBLE:
  case _XMP_N_TYPE_DOUBLE_IMAGINARY:
    compare_func = is_up ? compare_up_DOUBLE : compare_down_DOUBLE;
    get_rotate_pivot = get_rotate_pivot_for_DOUBLE;
    mpi_type = MPI_DOUBLE;
    break;

  case _XMP_N_TYPE_LONG_DOUBLE:
  case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
    compare_func = is_up ? compare_up_LONG_DOUBLE : compare_down_LONG_DOUBLE;
    get_rotate_pivot = get_rotate_pivot_for_LONG_DOUBLE;
    mpi_type = MPI_LONG_DOUBLE;
    break;

  case _XMP_N_TYPE_BOOL:
  case _XMP_N_TYPE_CHAR:
  case _XMP_N_TYPE_UNSIGNED_CHAR:
  case _XMP_N_TYPE_FLOAT_COMPLEX:
  case _XMP_N_TYPE_DOUBLE_COMPLEX:
  case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
  case _XMP_N_TYPE_NONBASIC:
  default:
    _XMP_fatal("xmp_sort: array arguments must be of a numerical type");
    break;
  }

}


static void do_sort(void *a, int n, _XMP_array_t *b_desc){

  void *pivot = malloc(datasize * nprocs);

  int size[nprocs];
  int start[nprocs];

  void *buf;
  int bufSize[nprocs];
  int bufStart[nprocs + 1];

  // local sort
  dbg_printf("(%d) local sort\n", me);
  qsort(a, n, datasize, compare_func);

  /* dbg_printf("(%d) a = ( ", me); */
  /* for (int i = 0; i < n; i++){ */
  /*   dbg_printf("%.1f ", ((double *)a)[i]); */
  /* } */
  /* dbg_printf(")\n"); */

  // compute pivots
  dbg_printf("(%d) comp_pivots\n", me);
  comp_pivots(a, n, pivot);

  /* printf("(%d) pivot = ( ", me); */
  /* for (int i = 0; i < nprocs; i++){ */
  /*   printf("%.1f ", ((double *)pivot)[i]); */
  /* } */
  /* printf(")\n"); */

  // make histogram
  dbg_printf("(%d) make_histogram\n", me);
  make_histogram(a, n, size, start, pivot);

  // distribute keys
  dbg_printf("(%d) dist_keys\n", me);
  dist_keys(a, size, start, &buf, bufSize, bufStart);

  /* printf("(%d) buf = ( ", me); */
  /* for (int i = 0; i < bufStart[nprocs]; i++){ */
  /*   printf("%d ", ((int *)buf)[i]); */
  /* } */
  /* printf(")\n"); */

  // merge_keys
  dbg_printf("(%d) merge_sort\n", me);
  kway_inplace_merge_sort(buf, bufStart, nprocs);

  //MPI_Barrier(*comm);

  // gmove
  dbg_printf("(%d) gmove\n", me);
  do_gmove(buf, bufStart, b_desc);

  free(pivot);
  free(buf);

}


static void comp_pivots(void *a, int n, void *pivot){

  /* if (n < nprocs){ */
  /*   fprintf(stderr, "a is too small\n"); */
  /*   exit(1); */
  /* } */

  void *tmp = calloc(nprocs * nprocs, datasize);
  //int tmp[nprocs * nprocs];

  /* for (int i = 0; i < nprocs * nprocs; i++){ */
  /*   tmp[i] = 0; */
  /* } */

  // Sampling
  for (int i = 0; i < nprocs; i++){
    //tmp[me * nprocs + i] = a[(i * n) / nprocs];
    memcpy((char *)tmp + datasize * (me * nprocs + i),
	   (char *)a + datasize * (((i + 1) * n) / nprocs),
	   datasize);
  }

  /* dbg_printf("(%d) pivot = ( ", me); */
  /* for (int i = 0; i < nprocs * nprocs; i++){ */
  /*   dbg_printf("%.1f ", ((double *)tmp)[i]); */
  /* } */
  /* dbg_printf(")\n"); */

  MPI_Allgather(MPI_IN_PLACE, nprocs /* dummy */, mpi_type /* dummy */,
		tmp, nprocs, mpi_type, *comm);

  qsort(tmp, nprocs * nprocs, datasize, compare_func);

  // get final pivot
  for (int i = 0; i < nprocs - 1; i++){
    //pivot[i] = tmp[(i + 1) * nprocs - 1];
    memcpy((char *)pivot + datasize * i, (char *)tmp + datasize * ((i + 1) * nprocs - 1), datasize);
  }				

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

  /* printf("(%d) bufSize = ( ", me); */
  /* for (int i = 0; i < nprocs; i++){ */
  /*   printf("%d ", bufSize[i]); */
  /* } */
  /* printf(")\n"); */

  bufStart[0] = 0;
  for (int i = 1; i <= nprocs; i++){
    bufStart[i] = bufStart[i - 1] + bufSize[i - 1];
  }

  *buf = malloc(datasize * bufStart[nprocs]);

  MPI_Alltoallv(a, size, start, mpi_type,
		*buf, bufSize, bufStart, mpi_type,
		*comm);
}


// n : length of a
// m : index of the first element larger than p
static void my_bsearch(void *p, void *a, int n, int *m){

  if (compare_func(a, p) > 0){ // if p < a[0]
    /* if (me == 1){ */
    /*   dbg_printf("(%d) a = %d, p = %d\n", me, *(int *)a, *(int *)p); */
    /* } */
    *m = 0;
    return;
  }
  else if (compare_func((char *)a + (n - 1) * datasize, p) < 0){ // if a[n-1] < p
    *m = n;
    return;
  }

  if (n < THRESHOLD_BSEARCH){
    int i;
    for (i = 0; i < n; i++){
      if (compare_func((char *)a + i * datasize, p) > 0) break; // if a[i] > p
    }
    *m = i;
    return;
  }

  int mid = n / 2;

  if (compare_func((char *)a + mid * datasize, p) < 0){ // if a[mid] < p
    my_bsearch(p, (char *)a + (mid + 1) * datasize, n - mid - 1, m);
    *m = mid + 1 + (*m);
  }
  else if (compare_func((char *)a + mid * datasize, p) > 0){ // if a[mid] > p
    my_bsearch(p, a, mid + 1, m);
  }
  else { // if a[mid] == p
    int i;
    for (i = mid; i < n; i++){
      if (compare_func((char *)a + i * datasize, p) != 0) break; // if a[i] == p
    }
    *m = i;
  }
  
}


static void reverse(void *a, int l, int r){
  long long tmp; // longest
  while (l < r){
    memcpy(&tmp, (char *)a + r * datasize, datasize);
    memcpy((char *)a + r * datasize, (char *)a + l * datasize, datasize);
    memcpy((char *)a + l * datasize, &tmp, datasize);
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


// n : length of a
// m : LENGTH of the first half of a (the base index of the second half)
static void two_way_inplace_merge_sort(void *a, int n, int m){

  if (me == 2 && dbg_flag){
    dbg_printf("(%d) two_way_inplace_merge_sort\n", me);
    dbg_printf("(%d)   n = %d, m = %d\n", me, n, m);
    dbg_printf("(%d)   a = ( ", me);
    for (int i = 0; i < n; i++){
      dbg_printf("%d ", ((int *)a)[i]);
    }
    dbg_printf(")\n");
  }

  if (m <= 0 || m >= n) return;

  if (n < THRESHOLD_2WAY){
    qsort(a, n, datasize, compare_func);
    return;
  }

  int an = m;

  void *b = (char *)a + m * datasize;
  int bn = n - m;

  long long p = 0; // longest

  get_rotate_pivot(&p, a, an, b, bn);

  int mid_a, mid_b; // index of the first element larger than p

  my_bsearch(&p, a, an, &mid_a);
  my_bsearch(&p, b, bn, &mid_b);

  if (mid_a == an && mid_b == bn){
    // pivot couldn't be selected properly; use quick sort.
    qsort(a, n, datasize, compare_func);
    return;
  }

  rotate(a, mid_a, m, mid_b + m - 1);

  int new_an = mid_a + mid_b;
  int new_mid_a = mid_a;
  int new_bn = n - new_an;
  int new_mid_b = m - mid_a;;

  if (mid_a > 0 && mid_b > 0)
    two_way_inplace_merge_sort(a, new_an, new_mid_a);

  if (mid_a < an && mid_b < bn)
    two_way_inplace_merge_sort((char *)a + new_an * datasize, new_bn, new_mid_b);

}


// start : start[i] is the base index of the i'th sequence
static void kway_inplace_merge_sort(void *a, int *start, int k){

  /* if (me == 2){ */
  /*   dbg_printf("(%d) kway_inplace_merge_sort\n", me); */
  /*   dbg_printf("(%d)   an = %d, k = %d\n", me, start[k] - start[0], k); */
  /*   dbg_printf("(%d)  d = ( ", me); */
  /*   for (int i = 0; i < start[k] - start[0]; i++){ */
  /*     dbg_printf("%d ", ((int *)a)[i]); */
  /*   } */
  /*   dbg_printf(")\n"); */
  /* } */

  if (k == 1) return;

  int an = start[k] - start[0];

  /* if (me == 4){ */
  /*   dbg_printf("(%d) start = ( ", me); */
  /*   for (int i = 0; i <= k; i++){ */
  /*     dbg_printf("%d ", start[i]); */
  /*   } */
  /*   dbg_printf(")\n"); */
  /*   dbg_printf("(%d) A = ( ", me); */
  /*   for (int i = 0; i < an; i++){ */
  /*     dbg_printf("%d ", ((int *)a)[i]); */
  /*   } */
  /*   dbg_printf("), k = %d, an = %d\n", k, an); */
  /* } */

  if (an < THRESHOLD_KWAY){
    qsort(a, an, datasize, compare_func);
    return;
  }

  if (k == 2){
    two_way_inplace_merge_sort(a, an,  start[1] - start[0]);
    /* if (me == 2){ */
    /*   dbg_printf("(%d)  c = ( ", me); */
    /*   for (int i = 0; i < an; i++){ */
    /* 	dbg_printf("%d ", ((int *)a)[i]); */
    /*   } */
    /*   dbg_printf(")\n"); */
    /* } */
    return;
  }
  else {
    int m = k / 2;
    kway_inplace_merge_sort(a, start, m);
    kway_inplace_merge_sort((char *)a + (start[m] - start[0]) * datasize, start + m, k - m);

    two_way_inplace_merge_sort(a, an, start[m] - start[0]);

    return;
  }

}

//
//
//

static void do_gmove(void *buf, int *bufStart, _XMP_array_t *b_desc){

  _XMP_nodes_t *n_desc = b_desc->array_nodes;
  _XMP_template_t *t_desc;
  _XMP_array_t *buf_desc;

  int n = b_desc->info[0].ser_size;
  int m[nprocs];

  int type = b_desc->type;

  int dummy0, dummy1;

  MPI_Allgather(&bufStart[nprocs], 1, MPI_INT,
		m, 1, MPI_INT, *comm);

  _XMP_init_template_FIXED(&t_desc, 1, (long long)0, (long long)(n - 1));
  _XMP_init_template_chunk(t_desc, n_desc);
  _XMP_dist_template_GBLOCK(t_desc, 0, 0, m, &dummy0);

  _XMP_init_array_desc(&buf_desc, t_desc, 1, type, datasize, n);
  _XMP_align_array_GBLOCK(buf_desc, 0, 0, 0, &dummy1);
  _XMP_init_array_comm(buf_desc, 0);
  _XMP_init_array_nodes(buf_desc);

  buf_desc->info[0].dim_acc = 1;
  _XMP_calc_array_dim_elmts(buf_desc, 0);
  buf_desc->total_elmts = buf_desc->info[0].alloc_size;
  buf_desc->array_addr_p = buf;

  _XMP_gmove_SENDRECV_ARRAY(b_desc, buf_desc, type, datasize,
  			    b_desc->info[0].ser_lower, n, 1, (unsigned long long)1,
  			    0, n, 1, (unsigned long long)1);

  _XMP_finalize_array_desc(buf_desc);
  _XMP_finalize_template(t_desc);

}


//
//
//

#define COMPARE_UP(_type) \
(const void *a, const void *b){ \
  if (*(_type *)a < *(_type *)b){ \
    return -1; \
  } \
  else if(*(_type *)a == *(_type *)b){ \
    return 0; \
  } \
  return 1; \
}

static int compare_up_SHORT               COMPARE_UP(short)
static int compare_up_UNSIGNED_SHORT      COMPARE_UP(unsigned short)
static int compare_up_INT                 COMPARE_UP(int)
static int compare_up_UNSIGNED_INT        COMPARE_UP(unsigned int)
static int compare_up_LONG                COMPARE_UP(long)
static int compare_up_UNSIGNED_LONG       COMPARE_UP(unsigned long)
static int compare_up_LONGLONG            COMPARE_UP(long long)
static int compare_up_UNSIGNED_LONGLONG   COMPARE_UP(unsigned long long)
static int compare_up_FLOAT               COMPARE_UP(float)
static int compare_up_DOUBLE              COMPARE_UP(double)
static int compare_up_LONG_DOUBLE         COMPARE_UP(long double)


#define COMPARE_DOWN(_type) \
(const void *a, const void *b){ \
  if (*(_type *)a > *(_type *)b){ \
    return -1; \
  } \
  else if(*(_type *)a == *(_type *)b){ \
    return 0; \
  } \
  return 1; \
}

static int compare_down_SHORT               COMPARE_DOWN(short)
static int compare_down_UNSIGNED_SHORT      COMPARE_DOWN(unsigned short)
static int compare_down_INT                 COMPARE_DOWN(int)
static int compare_down_UNSIGNED_INT        COMPARE_DOWN(unsigned int)
static int compare_down_LONG                COMPARE_DOWN(long)
static int compare_down_UNSIGNED_LONG       COMPARE_DOWN(unsigned long)
static int compare_down_LONGLONG            COMPARE_DOWN(long long)
static int compare_down_UNSIGNED_LONGLONG   COMPARE_DOWN(unsigned long long)
static int compare_down_FLOAT               COMPARE_DOWN(float)
static int compare_down_DOUBLE              COMPARE_DOWN(double)
static int compare_down_LONG_DOUBLE         COMPARE_DOWN(long double)


#define PIVOT(_type) \
(void *p, const void *a, const int an, \
          const void *b, const int bn){ \
  _type am = 0, bm = 0; \
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

static void get_rotate_pivot_for_SHORT               PIVOT(short)
static void get_rotate_pivot_for_UNSIGNED_SHORT      PIVOT(unsigned short)
static void get_rotate_pivot_for_INT                 PIVOT(int)
static void get_rotate_pivot_for_UNSIGNED_INT        PIVOT(unsigned int)
static void get_rotate_pivot_for_LONG                PIVOT(long)
static void get_rotate_pivot_for_UNSIGNED_LONG       PIVOT(unsigned long)
static void get_rotate_pivot_for_LONGLONG            PIVOT(long long)
static void get_rotate_pivot_for_UNSIGNED_LONGLONG   PIVOT(unsigned long long)
static void get_rotate_pivot_for_FLOAT               PIVOT(float)
static void get_rotate_pivot_for_DOUBLE              PIVOT(double)
static void get_rotate_pivot_for_LONG_DOUBLE         PIVOT(long double)
