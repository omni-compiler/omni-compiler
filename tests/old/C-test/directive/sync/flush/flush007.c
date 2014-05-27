static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 007:
 * check implicit flush at end of parallel region
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	a, b;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);
  omp_set_num_threads (2);


  #pragma omp parallel
  {
    a = 0;
    b = 0;
    #pragma omp barrier
    if (omp_get_thread_num () == 0) {
      a = 1;
    } else {
      b = 1;
    }
  }
  if (a != 1  ||  b != 1) {
    errors ++;
  }

  #pragma omp parallel
  {
    if (omp_get_thread_num () == 0) {
      a = 0;
    } else {
      b = 0;
    }
  }
  if (a != 0  ||  b != 0) {
    errors ++;
  }


  if (errors == 0) {
    printf ("flush 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 007 : FAILED\n");
    return 1;
  }
}
