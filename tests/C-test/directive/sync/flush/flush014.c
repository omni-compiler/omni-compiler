static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 014:
 * ポインタに対する flush の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	*a, *b;
int	zero = 0;
int	one = 1;


void
clear ()
{
  a = &zero;
}


void
func_flush ()
{
  int	id = omp_get_thread_num ();

  switch (id) {
  case 0:
    waittime (1);
    a = &one;
    #pragma omp flush
    while (*a == 1) {
      #pragma omp flush
    }
    break;

  case 1:
    while (*a == 0) {
      #pragma omp flush
    }
    a = &zero;
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
      a = &one;
      #pragma omp flush
      while (*a == 1) {
        #pragma omp flush
      }
      break;

    case 1:
      while (*a == 0) {
        #pragma omp flush
      }
      a = &zero;
      #pragma omp flush
      break;

    default:
      break;
    }
  }


  clear ();
  #pragma omp parallel
  func_flush ();


  if (errors == 0) {
    printf ("flush 014 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 014 : FAILED\n");
    return 1;
  }
}
