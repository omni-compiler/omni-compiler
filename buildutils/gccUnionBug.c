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

union large {
    char lBuf[LARGE_BUF];
};

int
main(argc, argv)
     int argc;
     char *argv[];
{
    fprintf(stdout, "0x%08x\n", sizeof(union large));
    return 0;
}

