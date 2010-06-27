static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 005:
 * check implicit barrier at end of parallel region
 */

#include <omp.h>
#include "omni.h"


main ()
{
  int	thds;

  int	errors = 0;
  int	v = 0;
  int	finish = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel
  {
    int id = omp_get_thread_num ();

    if (id == 0) {
      finish = 1;
      #pragma omp flush
    } else {
      while (finish == 0) {
	#pragma omp flush
      }
      waittime (1);
      v = 1;
    }
  } /* implicit barrier exist, here */

  if (v == 0) {
    errors = 1;
  }
  

  if (errors == 0) {
    printf ("parallel 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 005 : FAILED\n");
    return 1;
  }
}
