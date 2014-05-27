static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_in_parallel : 007
 * parallel if が成り立たない場合の、omp_in_parallel の動作を確認
 */
#include <omp.h>
#include "omni.h"


main ()
{
  int	thds;

  int	errors = 0;
  int	false = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel if(false)
  {
    if (omp_in_parallel () != 0) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("omp_in_parallel 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_in_parallel 007 : FAILED\n");
    return 1;
  }
}
