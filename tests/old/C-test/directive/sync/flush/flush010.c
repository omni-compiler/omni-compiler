static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 010:
 * single終了時の暗黙のflushを確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;
int	flag = 0;


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
    int id = omp_get_thread_num ();

    #pragma omp single
    {
      waittime (1);
      flag = 1;
    }

    if (flag == 0) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("flush 010 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 010 : FAILED\n");
    return 1;
  }
}
