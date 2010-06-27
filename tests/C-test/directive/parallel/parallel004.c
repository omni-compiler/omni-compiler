static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 004:
 * parallel section 中に thread 数が変わらないことを確認。
 */

#include <omp.h>
#include "omni.h"


#define ITER	(thds*1000)

main ()
{
  int	errors = 0;
  int	thds, i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel
  {
    if (thds != omp_get_num_threads ()) {
      #pragma omp critical
      errors += 1;
    }

    #pragma omp for
    for (i = 0;  i<ITER;  i++) {
      if (thds != omp_get_num_threads ()) {
	#pragma omp critical
	errors += 1;
      }
    }

    omp_set_num_threads (thds-1);

    if (thds != omp_get_num_threads ()) {
      #pragma omp critical
      errors += 1;
    }

    #pragma omp for
    for (i = 0;  i<1;  i++) {
      if (thds != omp_get_num_threads ()) {
	#pragma omp critical
	errors += 1;
      }
    }
  }

#if defined(__OMNI_SCASH__) || (__OMNI_SHMEM__)
  printf ("skip some tests. because Omni on SCASH/SHMEM is not support omp_set_num_threads.\n");
#else
  omp_set_num_threads (thds-1);
  #pragma omp parallel
  {
    if (thds-1 != omp_get_num_threads ()) {
      #pragma omp critical
      errors += 1;
    }

    #pragma omp for
    for (i = 0;  i<ITER;  i++) {
      if (thds-1 != omp_get_num_threads ()) {
	#pragma omp critical
	errors += 1;
      }
    }

    omp_set_num_threads (thds);

    if (thds-1 != omp_get_num_threads ()) {
      #pragma omp critical
      errors += 1;
    }

    #pragma omp for
    for (i = 0;  i<1;  i++) {
      if (thds-1 != omp_get_num_threads ()) {
	#pragma omp critical
	errors += 1;
      }
    }
  }
#endif


  if (errors == 0) {
    printf ("parallel 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 004 : FAILED\n");
    return 1;
  }
}
