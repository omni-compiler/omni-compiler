static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_get_num_threads : 006
 * nested parallel region が enable の場合の
 * omp_get_num_threads の動作を確認
 */

#include <omp.h>
#include "omni.h"


int
main ()
{
  int	thds;
  int	errors = 0;


  /* initialize */
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }
  omp_set_dynamic (0);
  omp_set_nested (1);
  if (omp_get_nested() == 0) {
    printf ("nested parallelism is not supported.\n");
    goto END;
  }

  omp_set_num_threads (1);

  #pragma omp parallel
  {
    if (omp_get_num_threads () != 1) {
      #pragma omp critical
      errors += 1;
    }

    omp_set_num_threads (thds);
    #pragma omp parallel
    {
      int n = omp_get_num_threads ();

      if (n == 1) {
	printf ("nested parallel is serialized\n");
      } else if (n != thds) {
	#pragma omp critical
	errors += 1;
      }
    }
  }


 END:
  if (errors == 0) {
    printf ("omp_get_num_threads 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_get_num_threads 006 : FAILED\n");
    return 1;
  }
}
