/* header file for xmp API */
/* should be marged to xmp.h */
#include <stddef.h> 
#include "xmp.h"
#include "xmp_constant.h"

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

#define XMP_SUCCESS 0

#define XMP_ERR_ARG         12      /* Invalid argument */
#define XMP_ERR_DIMS        11      /* Invalid dimension argument */

typedef enum xmp_datatype {
  XMP_TYPE_BOOL=       	_XMP_N_TYPE_BOOL,	
  XMP_TYPE_CHAR=	_XMP_N_TYPE_CHAR,
  XMP_TYPE_UNSIGNED_CHAR=  _XMP_N_TYPE_UNSIGNED_CHAR,
  XMP_TYPE_SHORT= 		_XMP_N_TYPE_SHORT,
  XMP_TYPE_UNSIGNED_SHORT= _XMP_N_TYPE_UNSIGNED_SHORT,
  XMP_TYPE_INT= 		_XMP_N_TYPE_INT,
  XMP_TYPE_UNSIGNED_INT= 	_XMP_N_TYPE_UNSIGNED_INT,
  XMP_TYPE_LONG= 		_XMP_N_TYPE_LONG,
  XMP_TYPE_UNSIGNED_LONG= 	_XMP_N_TYPE_UNSIGNED_LONG,
  XMP_TYPE_LONGLONG= 	_XMP_N_TYPE_LONGLONG,
  XMP_TYPE_UNSIGNED_LONGLONG= 	_XMP_N_TYPE_UNSIGNED_LONGLONG,
  XMP_TYPE_FLOAT= 		_XMP_N_TYPE_FLOAT,
  XMP_TYPE_DOUBLE= 	_XMP_N_TYPE_DOUBLE,
  XMP_TYPE_LONG_DOUBLE= 	_XMP_N_TYPE_LONG_DOUBLE,
  XMP_TYPE_NONE=0
} xmp_datatype_t;


typedef enum xmp_reduction_kind {
  XMP_REDUCE_SUM= _XMP_N_REDUCE_SUM,
  XMP_REDUCE_PROD= _XMP_N_REDUCE_PROD,
  XMP_REDUCE_BAND= _XMP_N_REDUCE_BAND,
  XMP_REDUCE_LAND= _XMP_N_REDUCE_LAND,
  XMP_REDUCE_BOR= _XMP_N_REDUCE_BOR,
  XMP_REDUCE_LOR= _XMP_N_REDUCE_LOR,
  XMP_REDUCE_BXOR=_XMP_N_REDUCE_BXOR,
  XMP_REDUCE_LXOR= _XMP_N_REDUCE_LXOR,
  XMP_REDUCE_MAX= _XMP_N_REDUCE_MAX,
  XMP_REDUCE_MIN= _XMP_N_REDUCE_MIN,
  XMP_REDUCE_FIRSTMAX= _XMP_N_REDUCE_FIRSTMAX,
  XMP_REDUCE_FIRSTMIN= _XMP_N_REDUCE_FIRSTMIN,
  XMP_REDUCE_LASTMAX= _XMP_N_REDUCE_LASTMAX,
  XMP_REDUCE_LASTMIN= _XMP_N_REDUCE_LASTMIN,
  XMP_REDUCE_EQV= _XMP_N_REDUCE_EQV,
  XMP_REDUCE_NEQV= _XMP_N_REDUCE_NEQV,
  XMP_REDUCE_MINUS= _XMP_N_REDUCE_MINUS,
  XMP_REDUCE_MAXLOC= _XMP_N_REDUCE_MAXLOC,
  XMP_REDUCE_MINLOC= _XMP_N_REDUCE_MINLOC,
  XMP_REDUCE_NONE=0
} xmp_reduction_kind_t;

// triplet for coarray C
struct _xmp_asection_triplet {
  long start;
  long length;  
  long stride;
};

typedef struct _xmp_array_section_t {
  int n_dims; /* # of dimensions */
  struct _xmp_asection_triplet dim_info[1];
} xmp_array_section_t;

/* struct _xmp_dimension_info_t */
/* { */
/*   int lb; // lower bound */
/*   int ub; // upper bound */
/* }; */

/* // dimension addtribute */
/* typedef struct _xmp_dimension_t  */
/* { */
/*   int n_dims; */
/*   struct _xmp_dimension_info_t dim_info[1]; */
/* } xmp_dimension_t; */
  
typedef struct _xmp_local_array_t {
  void *addr;
  int n_dims;
  long *dim_size;
  long *dim_f_offset;
} xmp_local_array_t;

/* allcoate array section structure */
xmp_array_section_t *xmp_new_array_section(int n_dims);
int xmp_array_section_set_info(xmp_array_section_t *ap, int dim_idx,
				long start, long length);
int xmp_array_section_set_triplet(xmp_array_section_t *rp, int dim_idx,
				   long start, long length, int stride);
void xmp_free_array_section(xmp_array_section_t *ap);

xmp_local_array_t *xmp_new_local_array(size_t elmt_size, int n_dims, long dim_size[], void *loc);
void xmp_free_local_array(xmp_local_array_t *ap);

xmp_desc_t xmp_new_coarray(int elmt_size, int ndims, long dim_size[],
			   int img_ndims, int img_dim_size[], void **loc);
int xmp_coarray_put(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
		   xmp_desc_t local_desc, xmp_array_section_t *local_asp);
int xmp_coarray_get(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
		   xmp_desc_t local_desc, xmp_array_section_t *local_asp);
int xmp_coarray_put_local(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
			  xmp_local_array_t *local_ap, xmp_array_section_t *local_asp);
int xmp_coarray_get_local(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
			  xmp_local_array_t *local_ap, xmp_array_section_t *local_asp);
int xmp_coarray_put_scalar( int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp,
			   void *addr);
int xmp_coarray_get_scalar(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
			   void *addr);
int xmp_coarray_deallocate(xmp_desc_t desc);


/** 
    global view
*/
xmp_desc_t xmp_global_nodes(int n_dims, int dim_size[], int is_static);

xmp_desc_t xmpc_new_template(xmp_desc_t n, int n_dims, long long dim1, ...);
int xmp_dist_template_BLOCK(xmp_desc_t t, int template_dim_idx, int node_dim_idx);
int xmp_dist_template_CYCLIC(xmp_desc_t t, int template_index, int nodes_index);
int xmp_dist_template_BLOCK_CYCLIC(xmp_desc_t t, int template_index, int nodes_index, unsigned long long width);
void xmp_dist_template_GBLOCK(xmp_desc_t t, int template_index, int nodes_index,
			       int *mapping_array, int *temp0);

xmp_desc_t xmpc_new_array(xmp_desc_t t, xmp_datatype_t type, int n_dims, int dim_size1,/* int dim_size2,*/ ... );
int xmp_align_array(xmp_desc_t a, int array_dim_idx, int template_dim_idx, long long offset);
int xmp_set_shadow(xmp_desc_t a, int dim_idx, int shdw_size_lo, int shdw_size_hi);
int xmp_set_full_shadow(xmp_desc_t a, int dim_idx);













