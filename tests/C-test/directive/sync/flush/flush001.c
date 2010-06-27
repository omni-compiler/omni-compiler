static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 001:
 * flush の動作を確認
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
  int	id = omp_get_thread_num ();

  switch (id) {
  case 0:
    waittime (1);
    a = 1;
    #pragma omp flush
    while (a == 1) {
      #pragma omp flush
    }
    break;

  case 1:
    while (a == 0) {
      #pragma omp flush
    }
    a = 0;
    #pragma omp flush
    break;

  default:
    break;
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
    int	id = omp_get_thread_num ();

    switch (id) {
    case 0:
      waittime (1);
      a = 1;
      #pragma omp flush
      while (a == 1) {
        #pragma omp flush
      }
      break;

    case 1:
      while (a == 0) {
        #pragma omp flush
      }
      a = 0;
      #pragma omp flush
      break;

    default:
      break;
    }
  }
  clear ();

  #pragma omp parallel
  func_flush ();
  clear ();


  if (errors == 0) {
    printf ("flush 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 001 : FAILED\n");
    return 1;
  }
}
