static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 012:
 * nowait付きsections終了時の暗黙のflushを確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;
int	flag = 0;
int	chk = 0;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  errors = 1;
  #pragma omp parallel 
  {
    int id = omp_get_thread_num ();

    #pragma omp sections nowait
    {
      #pragma omp section
      {
      }

      #pragma omp section
      {
	waittime(1);
	flag = 1;
      }
    }

    while (flag != 1) {
      if (id == omp_get_thread_num ()) {
	errors = 0;
      }
      #pragma omp flush
    }
    #pragma omp flush
  }


  if (errors == 0) {
    printf ("flush 012 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 012 : FAILED\n");
    return 1;
  }
}
