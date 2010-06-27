/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#if 0
static char rcsid[] = "$Id$";
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "confdefs.h"


int
main(argc, argv)
     int argc;
     char *argv[];
{
    char *s = "unknown";
    if (SIZEOF_VOID_P == SIZEOF_UNSIGNED_SHORT) {
	s = "short";
    } else if (SIZEOF_VOID_P == SIZEOF_UNSIGNED_INT) {
	s = "unsigned int";
#ifdef HAS_LONGLONG
    } else if (SIZEOF_VOID_P == SIZEOF_UNSIGNED_LONG_LONG) {
	s = "unsigned long long";
#endif /* HAS_LONGLONG */
    } else if (SIZEOF_VOID_P == SIZEOF_UNSIGNED_LONG) {
	s = "unsigned long";
    }

    printf("%s\n", s);
    return 0;
}
