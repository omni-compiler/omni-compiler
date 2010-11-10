/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XCALABLEMP_USERAPI
#define _XCALABLEMP_USERAPI

// #include "mpi.h"

extern void	xmp_get_comm(void **comm);
extern int	xmp_get_size(void);
extern int	xmp_get_rank(void);
extern void	xmp_barrier(void);
extern int	xmp_get_world_size(void);
extern int	xmp_get_world_rank(void);
extern double	xmp_wtime(void);

#endif // _XCALABLEMP_USERAPI
