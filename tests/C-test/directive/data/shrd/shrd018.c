static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* shared 018 :
 * parallel for directive に shared を設定できる事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	shrd;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  shrd = 0;
  #pragma omp parallel for shared(shrd)
  for (i=0; i<thds; i++) {
    #pragma omp critical
    shrd += 1;
  }

  if (shrd != thds) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("shared 018 : SUCCESS\n");
    return 0;
  } else {
    printf ("shared 018 : FAILED\n");
    return 1;
  }
}
