static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_get_num_procs : 003
 * parallel for 内で omp_get_num_procs の動作を確認
 */

#include <stdlib.h>
#include <omp.h>
#include "omni.h"


#define LOOPNUM	(100*thds)

main ()
{
  int	thds, p, i, j;

  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  p = omp_get_num_procs ();

  #pragma omp parallel for
  for (j=0; j<LOOPNUM; j++) {
    if (omp_get_num_procs() != p) {
      #pragma omp critical
      errors += 1;
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

    #pragma omp parallel for
    for (j=0; j<LOOPNUM; j++) {
      if (omp_get_num_procs() != p) {
	#pragma omp critical
	errors += 1;
      }
    }
  }
#endif

  if (errors == 0) {
    printf ("omp_get_num_procs 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_get_num_procs 003 : FAILED\n");
    return 1;
  }
}
