static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* shared 019 :
 * parallel sections directive に shared を設定できる事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	shrd;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  shrd = 0;
  #pragma omp parallel sections shared(shrd)
  {
    #pragma omp section
    {
      #pragma omp critical
      shrd += 1;
      barrier (2);
      if (shrd != 2) {
	#pragma omp critical
	errors += 1;
      }
    }
    #pragma omp section
    {
      #pragma omp critical
      shrd += 1;
      barrier (2);
      if (shrd != 2) {
	#pragma omp critical
	errors += 1;
      }
    }
  }


  if (errors == 0) {
    printf ("shared 019 : SUCCESS\n");
    return 0;
  } else {
    printf ("shared 019 : FAILED\n");
    return 1;
  }
}
