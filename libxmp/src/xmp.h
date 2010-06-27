#ifndef _XCALABLEMP_USERAPI
#define _XCALABLEMP_USERAPI

// #include "mpi.h"

extern void	xmp_get_comm(void **comm);
extern int	xmp_get_size(void);
extern int	xmp_get_rank(void);
extern void	xmp_barrier(void);

#endif // _XCALABLEMP_USERAPI
