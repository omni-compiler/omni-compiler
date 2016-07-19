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
#include "gmp.h"


int
main(argc, argv)
     int argc;
     char *argv[];
{
    printf("%d\n", (int)sizeof(mp_limb_t));
    return 0;
}
