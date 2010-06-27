static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 003:
 * "parallel if(false)" test
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;


int
sameas (int v)
{
  return v;
}


void
check_parallel ()
{
  if (omp_in_parallel () != 0) {
    #pragma omp critical
    {
      ERROR (errors);
    }
  }
  if (omp_get_num_threads () != 1) {
    #pragma omp critical
    {
      ERROR (errors);
    }
  }
}



main ()
{
  double dfail = 0.0;
  int    fail = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  check_parallel ();		      /* not parallel */

  #pragma omp parallel if (0)
  check_parallel ();		      /* here is not parallel */

  check_parallel ();		      /* not parallel */

  #pragma omp parallel if (dfail)
  {				      /* this block is not parallel */
    check_parallel();
  }

  check_parallel ();		      /* not parallel */

  #pragma omp parallel if (fail == 1)
  if (!fail) {			      /* this if-block is not parallel */
    check_parallel ();
  }

  check_parallel ();		      /* not parallel */

  #pragma omp parallel if (sameas(fail))
  check_parallel ();		      /* here is not parallel */


  if (errors == 0) {
    printf ("parallel 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 003 : FAILED\n");
    return 1;
  }
}
