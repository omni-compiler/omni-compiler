static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 006:
 * orderd終了時の暗黙のflushを確認
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
  int	i;

  #pragma omp for schedule(static,1) ordered 
  for (i=0;  i<thds;  i++) {
    #pragma omp ordered
    {
      a = i;
    }
    if (i != thds - 1) {
      while (a != thds - 1) {
       #pragma omp flush
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
    int	i;

    #pragma omp for ordered schedule(static,1)
    for (i=0;  i<2;  i++) {
      #pragma omp ordered
      {
	waittime (1);
	a += 1;
      }
      while (a != 2  &&  i == 0) {
	#pragma omp flush
      }
    }
  }


  clear ();
  #pragma omp parallel
  func_flush ();


  if (errors == 0) {
    printf ("flush 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 006 : FAILED\n");
    return 1;
  }
}
