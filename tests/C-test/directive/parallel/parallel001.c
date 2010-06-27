static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 001:
 * parallel directive test
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;


void
check_parallel (int v)
{
  if (omp_in_parallel () != v) {
    #pragma omp critical
    {
      ERROR (errors);
    }
  }
}



main ()
{
  int true = 1;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  check_parallel (0);		      /* not parallel */

  #pragma omp parallel
  check_parallel (1);		      /* here is parallel */

  check_parallel (0);		      /* not parallel */

  #pragma omp parallel
  {				      /* this block is parallel */
    check_parallel(1);
  }

  check_parallel (0);		      /* not parallel */

  #pragma omp parallel 
  if (true) {			      /* this if-block is parallel */
    check_parallel (1);
  }


  if (errors == 0) {
    printf ("parallel 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 001 : FAILED\n");
    return 1;
  }
}
