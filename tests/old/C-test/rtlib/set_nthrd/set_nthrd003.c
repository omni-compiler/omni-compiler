static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_set_num_threads : 003
 * omp_set_num_threads でスレッド数を変更し,
 * omp parallel sections で動作確認
 */

#include <omp.h>
#include "omni.h"


int
main ()
{
  int	*buf, thds, i, j, n;

  int	errors = 0;


#if defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__)
  printf ("Omni on SCASH/SHMEM do not support omp_set_num_threads, yet.\n");
  printf ("test skipped.\n");
  exit (0);
#endif
  /* initialize */
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit(0);
  } else if (4 < thds) {
    thds = 4;
  }
  buf = (int *) malloc ((thds + 1) * sizeof (int));
  if (buf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  omp_set_dynamic (0);


  for (i=1; i<=thds; i++) {

    for (j=0; j<=thds; j++) {
      buf[j] = -1;
    }

    omp_set_num_threads (i);
    #pragma omp parallel sections 
    {
      #pragma omp section
      {
	int id = omp_get_thread_num ();
	buf[id] = 1;
	waittime (1);
	if (omp_get_num_threads() != i) {
	  #pragma omp critical
	  errors += 1;
	}
      }
      #pragma omp section
      {
	int id = omp_get_thread_num ();
	buf[id] = 1;
	waittime (1);
	if (omp_get_num_threads() != i) {
	  #pragma omp critical
	  errors += 1;
	}
      }
      #pragma omp section
      {
	int id = omp_get_thread_num ();
	buf[id] = 1;
	waittime (1);
	if (omp_get_num_threads() != i) {
	  #pragma omp critical
	  errors += 1;
	}
      }
      #pragma omp section
      {
	int id = omp_get_thread_num ();
	buf[id] = 1;
	waittime (1);
	if (omp_get_num_threads() != i) {
	  #pragma omp critical
	  errors += 1;
	}
      }
    }

    /* check */
    n = 0;
    for (j=0; j<=thds; j++) {
      if (buf[j] != -1) {
	n ++;
      }
    }
    if (n != i) {
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("omp_set_num_threads 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_set_num_threads 003 : FAILED\n");
    return 1;
  }
}
