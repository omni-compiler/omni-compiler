/* header file for xmp API */
/* should be marged to xmp.h */
#include <stddef.h> 
#include "xmp.h"

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

#define XMP_SUCCESS 0

#define XMP_ERR_ARG         12      /* Invalid argument */
#define XMP_ERR_DIMS        11      /* Invalid dimension argument */

#ifdef not
/* Communication argument parameters */
#define MPI_ERR_BUFFER      xo 1      /* Invalid buffer pointer */
#define MPI_ERR_COUNT        2      /* Invalid count argument */
#define MPI_ERR_TYPE         3      /* Invalid datatype argument */
#define MPI_ERR_TAG          4      /* Invalid tag argument */
#define MPI_ERR_COMM         5      /* Invalid communicator */
#define MPI_ERR_RANK         6      /* Invalid rank */
#define MPI_ERR_ROOT         7      /* Invalid root */

/* MPI Objects (other than COMM) */
#define MPI_ERR_GROUP        8      /* Invalid group */
#define MPI_ERR_OP           9      /* Invalid operation */
#define MPI_ERR_REQUEST     19      /* Invalid mpi_request handle */

/* Special topology argument parameters */
#define MPI_ERR_TOPOLOGY    10      /* Invalid topology */

/* All other arguments.  This is a class with many kinds */
#define MPI_ERR_ARG         12      /* Invalid argument */

/* Other errors that are not simply an invalid argument */
#define MPI_ERR_OTHER       15      /* Other error; use Error_string */

#define MPI_ERR_UNKNOWN     13      /* Unknown error */

/* Multiple completion has three special error classes */
#define MPI_ERR_IN_STATUS           17      /* Look in status for error value */
#define MPI_ERR_PENDING             18      /* Pending request */

/* New MPI-2 Error classes */
#define MPI_ERR_ACCESS      20      /* */
#define MPI_ERR_AMODE       21      /* */
#define MPI_ERR_BAD_FILE    22      /* */
#define MPI_ERR_CONVERSION  23      /* */
#define MPI_ERR_DUP_DATAREP 24      /* */
#define MPI_ERR_FILE_EXISTS 25      /* */
#define MPI_ERR_FILE_IN_USE 26      /* */
#define MPI_ERR_FILE        27      /* */
#define MPI_ERR_IO          32      /* */
#define MPI_ERR_NO_SPACE    36      /* */
#define MPI_ERR_NO_SUCH_FILE 37     /* */
#define MPI_ERR_READ_ONLY   40      /* */
#define MPI_ERR_UNSUPPORTED_DATAREP   43  /* */

/* MPI_ERR_INFO is NOT defined in the MPI-2 standard.  I believe that
   this is an oversight */
#define MPI_ERR_INFO        28      /* */
#define MPI_ERR_INFO_KEY    29      /* */
#define MPI_ERR_INFO_VALUE  30      /* */
#define MPI_ERR_INFO_NOKEY  31      /* */

#define MPI_ERR_NAME        33      /* */
#define MPI_ERR_NO_MEM      34      /* Alloc_mem could not allocate memory */
#define MPI_ERR_NOT_SAME    35      /* */
#define MPI_ERR_PORT        38      /* */
#define MPI_ERR_QUOTA       39      /* */
#define MPI_ERR_SERVICE     41      /* */
#define MPI_ERR_SPAWN       42      /* */
#define MPI_ERR_UNSUPPORTED_OPERATION 44 /* */
#define MPI_ERR_WIN         45      /* */

#define MPI_ERR_BASE        46      /* */
#define MPI_ERR_LOCKTYPE    47      /* */
#define MPI_ERR_KEYVAL      48      /* Erroneous attribute key */
#define MPI_ERR_RMA_CONFLICT 49     /* */
#define MPI_ERR_RMA_SYNC    50      /* */ 
#define MPI_ERR_SIZE        51      /* */
#define MPI_ERR_DISP        52      /* */
#define MPI_ERR_ASSERT      53      /* */

#define MPI_ERR_RMA_RANGE  55       /* */
#define MPI_ERR_RMA_ATTACH 56       /* */
#define MPI_ERR_RMA_SHARED 57       /* */
#define MPI_ERR_RMA_FLAVOR 58       /* */
#endif

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








