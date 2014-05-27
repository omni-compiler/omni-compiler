static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* lastprivate 021 :
 * sections directiveにlastprivate directiveを設定した場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	prvt;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel sections lastprivate (prvt)
  {
    #pragma omp section
    {
      prvt = 1;
      barrier (2);
      if (prvt != 1) {
        #pragma omp critical
	errors += 1;
      }
      waittime (1);
      prvt = 1;
    }
    #pragma omp section
    {
      prvt = 2;
      barrier (2);
      if (prvt != 2) {
        #pragma omp critical
	errors += 1;
      }
      prvt = 2;
    }
  }

  if (prvt != 2) {
    #pragma omp critical
    errors += 1;
  }


  if (errors == 0) {
    printf ("lastprivate 021 : SUCCESS\n");
    return 0;
  } else {
    printf ("lastprivate 021 : FAILED\n");
    return 1;
  }
}
