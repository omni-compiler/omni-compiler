static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_get_thread_num : 006
 * serialize された nested parallel region での
 * omp_get_thread_num の動作を確認
 */

#include <string.h>
#include <omp.h>
#include "omni.h"


int
main ()
{
  int		thds, *buf, i, j, t;

  int		errors = 0;


  /* initialize */
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");	
    exit (0);
  }
  buf = (int *) malloc (sizeof(int) * (thds + 1));

  omp_set_dynamic (0);
  omp_set_nested (1);
  if (omp_get_nested() == 0) {
    printf ("nested parallelism is not suppport.\n");
    goto END;
  }


  for (i=1; i<=thds; i++) {

    memset (buf, 0, sizeof(int)*(thds+1));
    omp_set_num_threads (1);

    #pragma omp parallel
    {
      omp_set_num_threads (i);

      #pragma omp parallel
      {
	int	id = omp_get_thread_num ();
	buf[id] += 1;
      }
    }

    for (t=0, j=0; j<i; j++) {
      if (buf[j] != 0) {
	t ++;
      }
    }
    if (i != 1  &&  t == 1  &&  buf[0] == 1) {
      printf ("nested parallel region is serialized\n");
    } else {
      for (j=0; j<i; j++) {
	if (buf[j] != 1) {
	  errors += 1;
	}
      }
    }
    for (j=i; j<=thds; j++) {
      if (buf[j] != 0) {
	errors += 1;
      }
    }
  }


 END:
  if (errors == 0) {
    printf ("omp_get_thread_num 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_get_thread_num 006 : FAILED\n");
    return 1;
  }
}
