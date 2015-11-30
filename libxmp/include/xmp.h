#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#ifndef _XMP_USERAPI
#define _XMP_USERAPI

#define XMP_FAILURE					-2000
#define XMP_UNDEFINED					-2001

#define XMP_ENTIRE_NODES				2000
#define XMP_EXECUTING_NODES				2001
#define XMP_PRIMARY_NODES				2002
#define XMP_EQUIVALENCE_NODES				2003

#define XMP_NOT_DISTRIBUTED				2100
#define XMP_BLOCK					2101
#define XMP_CYCLIC					2102
#define XMP_GBLOCK					2103

#include <stddef.h>
#include <mpi.h>

// Typedef
typedef void* xmp_desc_t;
#ifdef _XMP_GASNET
#include "xmp_lock.h"
#endif

// ----- libxmp
extern MPI_Comm	xmp_get_mpi_comm(void);
extern void	xmp_init_mpi(int *argc, char ***argv);
extern void	xmp_finalize_mpi(void);
extern void	xmp_init(int *argc, char ***argv);
extern void	xmp_finalize(void);
extern int	xmp_num_nodes(void);
extern int	xmp_node_num(void);
extern void	xmp_barrier(void);
extern int	xmp_all_num_nodes(void);
extern int	xmp_all_node_num(void);
extern double	xmp_wtime(void);
extern double	xmp_wtick(void);
extern int      xmp_array_ndims(xmp_desc_t d, int *ndims);
extern int      xmp_array_lbound(xmp_desc_t d, int dim, int *lbound);
extern int      xmp_array_ubound(xmp_desc_t d, int dim, int *ubound);
extern size_t   xmp_array_type_size(xmp_desc_t d);
extern int      xmp_array_gsize(xmp_desc_t d, int dim);
extern int      xmp_array_lsize(xmp_desc_t d, int dim, int *lsize);
extern int      xmp_array_gcllbound(xmp_desc_t d, int dim);
extern int      xmp_array_gclubound(xmp_desc_t d, int dim);
extern int      xmp_array_lcllbound(xmp_desc_t d, int dim);
extern int      xmp_array_lclubound(xmp_desc_t d, int dim);
extern int      xmp_array_gcglbound(xmp_desc_t d, int dim);
extern int      xmp_array_gcgubound(xmp_desc_t d, int dim);
extern int      xmp_array_laddr(xmp_desc_t d, void **laddr);
extern int      xmp_array_lshadow(xmp_desc_t d, int dim, int *lshadow);
extern int      xmp_array_ushadow(xmp_desc_t d, int dim, int *ushadow);
extern int      xmp_array_owner(xmp_desc_t d, int ndims, int index[], int dim);
extern int      xmp_array_lead_dim(xmp_desc_t d, int size[]);
extern int      xmp_array_gtol(xmp_desc_t d, int g_idx[], int lidx[]);
extern int      xmp_align_axis(xmp_desc_t d, int dim, int *axis);
extern int      xmp_align_offset(xmp_desc_t d, int dim, int *offset);
extern int      xmp_align_format(xmp_desc_t d, int dim);
extern int      xmp_align_size(xmp_desc_t d, int dim);
extern int      xmp_align_replicated(xmp_desc_t d, int dim, int *replicated);
extern int      xmp_align_template(xmp_desc_t d, xmp_desc_t *dt);
extern int      xmp_template_fixed(xmp_desc_t d, int *fixed);
extern int      xmp_template_ndims(xmp_desc_t d, int *ndims);
extern int      xmp_template_lbound(xmp_desc_t d, int dim, int *lbound);
extern int      xmp_template_ubound(xmp_desc_t d, int dim, int *ubound);
extern int      xmp_template_gsize(xmp_desc_t d, int dim);
extern int      xmp_template_lsize(xmp_desc_t d, int dim);
extern int      xmp_dist_format(xmp_desc_t d, int dim, int *format);
extern int      xmp_dist_blocksize(xmp_desc_t d, int dim, int *blocksize);
extern int      xmp_dist_stride(xmp_desc_t d, int dim);
extern int      xmp_dist_nodes(xmp_desc_t d, xmp_desc_t *dn);
extern int      xmp_dist_axis(xmp_desc_t d, int dim, int *axis);
extern int      xmp_dist_gblockmap(xmp_desc_t d, int dim, int *map);
extern int      xmp_nodes_ndims(xmp_desc_t d, int *ndims);
extern int      xmp_nodes_index(xmp_desc_t d, int dim, int *index);
extern int      xmp_nodes_size(xmp_desc_t d, int dim, int *size);
extern int      xmp_nodes_rank(xmp_desc_t d, int *rank);
extern int      xmp_nodes_comm(xmp_desc_t d, void **comm);
extern int      xmp_nodes_equiv(xmp_desc_t d, xmp_desc_t *dn, int lb[], int ub[], int st[]);
extern void     xmp_sched_template_index(int* local_start_index, int* local_end_index,
					 const int global_start_index, const int global_end_index, const int step,
					 const xmp_desc_t template, const int template_dim);
extern void     xmp_sync_memory(const int* status);
extern void     xmp_sync_all(const int* status);
extern void     xmp_sync_image(const int image, int* status);
extern void     xmp_sync_images(const int num, int* image_set, int* status);
extern void     xmp_sync_images_all(int* status);
extern void     xmp_sort_up(xmp_desc_t a_desc, xmp_desc_t b_desc);
extern void     xmp_sort_down(xmp_desc_t a_desc, xmp_desc_t b_desc);
extern void    *xmp_malloc(xmp_desc_t d, ...);
extern void     xmp_free(xmp_desc_t d);
extern void     xmp_exit(int status);

// ----- libxmp_gpu
#ifdef _XMP_ENABLE_GPU
extern int	xmp_get_gpu_count(void);
#endif

#endif // _XMP_USERAPI

