static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_get_num_threads : 005
 * nested parallel region が disable の場合の
 * omp_get_num_threads の動作を確認
 */

#include <omp.h>
#include "omni.h"


int
main ()
{
  int	thds;

  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }
  omp_set_dynamic (0);
  omp_set_nested (0);


  #pragma omp parallel
  {
    if (omp_get_num_threads() != thds) {
      #pragma omp critical
      errors += 1;
    }

    #pragma omp parallel
    {
      if (omp_get_num_threads () != 1) {
	#pragma omp critical
	errors += 1;
      }
    }
  }


  if (errors == 0) {
    printf ("omp_get_num_threads 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_get_num_threads 005 : FAILED\n");
    return 1;
  }
}
