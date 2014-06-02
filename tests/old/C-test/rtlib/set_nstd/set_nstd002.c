static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_set_nested : 002
 * omp_set_nestedで、nested parallel region を
 * enableにした時の動作を確認
 */

#include <string.h>
#include <omp.h>
#include "omni.h"


int
main ()
{
  int	thds, *buf;

  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }
  buf = (int *) malloc (sizeof(int) * (thds + 1));
  if (buf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }

  omp_set_dynamic (0);
  omp_set_nested (1);
  if (omp_get_nested () == 0) {
    printf ("nested parallelism is not implement.\n");
    goto END;
  }


  omp_set_num_threads (1);

  #pragma omp parallel 
  {
    int	i, j;

    if (omp_get_num_threads () != 1) {
      #pragma omp critical
      errors += 1;
    }
    if (omp_get_thread_num () != 0) {
      errors += 1;
    }

    for (i=1; i<=thds; i++) {

      memset (buf, 0, sizeof(int) * (thds+1));
      omp_set_num_threads (i);

      #pragma omp parallel
      {
	int	id = omp_get_thread_num ();

	if (omp_get_num_threads () != i) {
	  #pragma omp critical
	  errors += 1;
	}
	buf[id] += 1;
      }

      for (j=0; j<i; j++) {
	if (buf[j] != 1) {
	  #pragma omp critical
	  errors += 1;
	}	
      }
      for (j=i; j<=thds; j++) {
	if (buf[j] != 0) {
	  #pragma omp critical
	  errors += 1;
	}	
      }
    }
  }


 END:
  if (errors == 0) {
    printf ("omp_set_nested 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_set_nested 002 : FAILED\n");
    return 1;
  }
}

