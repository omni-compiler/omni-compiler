static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 009:
 * sections終了時の暗黙のflushを確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;



main ()
{
  int	a, b;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel 
  {
    #pragma omp sections 
    {
      #pragma omp section
      {
	b = 0;
	barrier (2);
	#pragma omp flush (a,b)

	a = 1;
      }

      #pragma omp section
      {
	a = 0;
	barrier (2);
	#pragma omp flush (a,b)

	b = 1;
      }
    }
    if (a != 1  ||  b != 1) {
      errors ++;
    }
  }


  if (errors == 0) {
    printf ("flush 009 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 009 : FAILED\n");
    return 1;
  }
}
