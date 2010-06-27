static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 023 :
 * single に private を設定できることを確認。
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


  prvt = 0;
  #pragma omp parallel
  {
    #pragma omp single private (prvt)
    {
      prvt = 1;
      #pragma omp flush
      if (prvt != 1) {
	errors += 1;
      }
    }
  }


  if (errors == 0) {
    printf ("private 023 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 023 : FAILED\n");
    return 1;
  }
}
