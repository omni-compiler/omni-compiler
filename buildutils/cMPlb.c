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
    mpf_t f;

    if (argc < 2) {
	printf("FAIL\n");
	return 1;
    }

    mpf_set_default_prec(atoi(argv[1]));
    mpf_init(f);

    printf("%d\n", (((__mpf_struct *)(&f))->_mp_prec + 1));
    return 0;
}
