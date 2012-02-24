/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_USERAPI
#define _XMP_USERAPI

// #include "mpi.h"

typedef void *xmp_desc_t;

// ----- libxmp
extern void	xmp_MPI_comm(void **comm);
extern int	xmp_num_nodes(void);
extern int	xmp_node_num(void);
extern void	xmp_barrier(void);
extern int	xmp_all_num_nodes(void);
extern int	xmp_all_node_num(void);
extern double	xmp_wtime(void);
extern double	xmp_wtick(void);
extern void     xmp_array_ndim(xmp_desc_t d, int *ndim);
extern void     xmp_array_gsize(xmp_desc_t d, int size[]);
extern void     xmp_array_lsize(xmp_desc_t d, int size[]);
extern void     xmp_array_laddr(xmp_desc_t d, void **laddr);
extern void     xmp_array_shadow(xmp_desc_t d, int ushadow[], int lshadow[]);
extern void     xmp_array_first_idx_node_index(xmp_desc_t d, int idx[]);
extern void     xmp_array_lead_dim(xmp_desc_t d, int *lead_dim);
extern void     xmp_align_axis(xmp_desc_t d, int axis[]);
extern void     xmp_align_offset(xmp_desc_t d, int offset[]);
extern xmp_desc_t xmp_align_template(xmp_desc_t d);
extern _Bool    xmp_template_fixed(xmp_desc_t d);
extern void     xmp_template_ndim(xmp_desc_t d, int *ndim);
extern void     xmp_template_gsize(xmp_desc_t d, int size[]);
extern void     xmp_template_lsize(xmp_desc_t d, int size[]);
extern void     xmp_dist_format(xmp_desc_t d, int dist_format[]);
extern void     xmp_dist_size(xmp_desc_t d, int size[]);
extern xmp_desc_t xmp_dist_nodes(xmp_desc_t d);
extern void     xmp_nodes_ndim(xmp_desc_t d, int *ndim);
extern void     xmp_nodes_index(xmp_desc_t d, int idx[]);
extern void     xmp_nodes_size(xmp_desc_t d, int size[]);

// ----- libxmp_gpu
#ifdef _XMP_ENABLE_GPU
extern int	xmp_get_gpu_count(void);
#endif

#endif // _XMP_USERAPI
