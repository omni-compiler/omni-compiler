static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* shared 017 :
 * parallel directive に shared を設定できる事を確認
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
  #pragma omp parallel shared(shrd)
  {
    #pragma omp critical
    shrd += 1;

    #pragma omp barrier

    if (shrd != thds) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("shared 017 : SUCCESS\n");
    return 0;
  } else {
    printf ("shared 017 : FAILED\n");
    return 1;
  }
}
