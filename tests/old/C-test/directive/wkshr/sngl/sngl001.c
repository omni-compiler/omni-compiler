static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* single 001:
 * 単純に single directive を指定したケース
 */

#include <omp.h>
#include "omni.h"


void
incr (int *ptr)
{
  #pragma omp single
  *ptr += 1;
}


main ()
{
  int	thds, buf[4];

  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  memset (buf, 0, sizeof (buf));
  #pragma omp parallel
  {
    #pragma omp single
    buf[0] += 1;

    #pragma omp single
    {
      buf[1] += 1;
    }

    incr (&buf[2]);
  }

  incr(&buf[3]);

  if (buf[0] != 1  ||  buf[1] != 1 || buf[2] != 1 || buf[3] != 1) {
    errors += 1;
  }

  if (errors == 0) {
    printf ("single 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("single 001 : FAILED\n");
    return 1;
  }
}
