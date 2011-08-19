/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_USERAPI
#define _XMP_USERAPI

// #include "mpi.h"

// ----- libxmp
extern void	xmp_MPI_comm(void **comm);
extern int	xmp_num_nodes(void);
extern int	xmp_node_num(void);
extern void	xmp_barrier(void);
extern int	xmp_all_num_nodes(void);
extern int	xmp_all_node_num(void);
extern double	xmp_wtime(void);

// ----- libxmp_gpu
#ifdef _XMP_ENABLE_GPU
extern int	xmp_get_gpu_count(void);
#endif

#endif // _XMP_USERAPI
