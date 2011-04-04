/** flag to indicate whether event generation is turned on or off */
#include "epik_user.h"
extern int epk_mpi_nogen;

/** profiling start **/
#define _XMP_M_EPIK_USER_START           EPIK_USER_START
/** profiling stop **/
#define _XMP_M_EPIK_USER_END             EPIK_USER_END

/** turn off event generation */
#define _XMP_M_EPIK_GEN_OFF()            epk_mpi_nogen = 1
/** turn on event generation */
#define _XMP_M_EPIK_GEN_ON()             epk_mpi_nogen = 0
