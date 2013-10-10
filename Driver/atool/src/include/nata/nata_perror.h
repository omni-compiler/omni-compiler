/* 
 * $Id: nata_perror.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#include <nata/nata_logger.h>

#ifdef perror
#undef perror
#endif /* perror */
#define perror(str)	nata_MsgError("%s: %s\n", str, strerror(errno))
