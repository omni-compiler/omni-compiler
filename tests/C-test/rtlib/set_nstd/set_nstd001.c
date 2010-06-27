static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_set_nested : 001
 * omp_set_nestedで、nested parallel region を
 * disableにした時の動作を確認
 */

#include <omp.h>
#include "omni.h"


int
main ()
{
  int	thds, i;

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
    #pragma omp parallel
    {
      if (omp_get_num_threads () != 1) {
	#pragma omp critical
	errors += 1;
      }
      if (omp_get_thread_num () != 0) {
	#pragma omp critical
	errors += 1;
      }
    }
  }


#if defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__)
  /* Omni on SCASH do not support omp_set_num_threads.
   * and, some test 
   */
  printf ("skip some tests. because, Omni on SCASH/SHMEM do not support omp_set_num_threads, yet.\n");
#else
  for (i=1; i<=thds; i++) {
    omp_set_num_threads (i);

    #pragma omp parallel 
    {
      #pragma omp parallel
      {
	if (omp_get_num_threads () != 1) {
	  #pragma omp critical
	  errors += 1;
	}
	if (omp_get_thread_num () != 0) {
          #pragma omp critical
	  errors += 1;
	}
      }
    }
  }
#endif


  if (errors == 0) {
    printf ("omp_set_nested 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_set_nested 001 : FAILED\n");
    return 1;
  }
}

