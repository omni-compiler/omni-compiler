static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 003:
 * 変数名が指定された場合の flush の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	a, b, c;


void
clear ()
{
  a = 0;
  b = 0;
  c = 0;
}


void
func_flush ()
{
  int	id = omp_get_thread_num ();

  switch (id) {
  case 0:
    waittime (1);
    a = 1;
    b = 2;
    c = 3;
    #pragma omp flush (a,b)
    #pragma omp flush (c)
    while (a != 0 || b != 0 || c != 0) {
      #pragma omp flush (a)
      #pragma omp flush (b, c)
    }
    break;

  case 1:
    while (a == 0 || b == 0 || c == 0) {
      #pragma omp flush (a)
      #pragma omp flush (b)
      #pragma omp flush (c)
    }
    a = 0;
    b = 0;
    c = 0;
    #pragma omp flush (a,b,c)
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
      b = 2;
      c = 3;
      #pragma omp flush (a,b)
      #pragma omp flush (c)
      while (a != 0 || b != 0 || c != 0) {
        #pragma omp flush (a)
        #pragma omp flush (b, c)
      }
      break;

    case 1:
      while (a == 0 || b == 0 || c == 0) {
        #pragma omp flush (a)
        #pragma omp flush (b)
        #pragma omp flush (c)
      }
      a = 0;
      b = 0;
      c = 0;
      #pragma omp flush (a,b,c)
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
    printf ("flush 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 003 : FAILED\n");
    return 1;
  }
}
