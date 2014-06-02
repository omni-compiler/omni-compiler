static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 005:
 * check implicit flush at end of critical section
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	a, b;


void
clear ()
{
  a = 0;
  b = 0;
}


void
func_flush ()
{
  a = 0;
  #pragma omp barrier

  if (omp_get_thread_num () == 0) {
    waittime (1);
    #pragma omp critical
    a = 1;
  } else {
    while (a == 0) {
      #pragma omp critical
      {
      }
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


  clear ();
  #pragma omp parallel
  {
    a = 0;
    #pragma omp barrier

    if (omp_get_thread_num () == 0) {
      waittime (1);
      #pragma omp critical
      a = 1;
    } else {
      while (a == 0) {
	#pragma omp critical
	{
	}
      }
    }
  }


  clear ();
  #pragma omp parallel
  func_flush ();


  if (errors == 0) {
    printf ("flush 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 005 : FAILED\n");
    return 1;
  }
}
