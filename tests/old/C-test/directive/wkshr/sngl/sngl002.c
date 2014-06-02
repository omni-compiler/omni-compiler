static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* single 002:
 * single directive の終了時のバリアの動作確認
 */

#include <omp.h>
#include "omni.h"


int	thds, id, flag;

int	errors = 0;


void
func_single ()
{
  #pragma omp barrier

  #pragma omp single
  {
    id = omp_get_thread_num ();
    #pragma omp flush
    waittime (1);
    flag = 1;
  }

  if (id != omp_get_thread_num ()) {
    if (flag == 0) {
      #pragma omp critical
      errors += 1;
    }
  }
}


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  id = -1;
  flag = 0;
  #pragma omp parallel
  {
    #pragma omp barrier

    #pragma omp single
    {
      id = omp_get_thread_num ();
      #pragma omp flush
      waittime (1);
      flag = 1;
    }
    if (id != omp_get_thread_num ()) {
      if (flag == 0) {
	#pragma omp critical
	errors += 1;
      }
    }

  }


  id = -1;
  flag = 0;
  #pragma omp parallel
  {
    func_single ();
  }


  if (errors == 0) {
    printf ("single 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("single 002 : FAILED\n");
    return 1;
  }
}
