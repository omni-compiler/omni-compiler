static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* copyin 018 :
 * defualt と copyin が同時に宣言された場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	org;
int	prvt;
#pragma omp threadprivate(prvt)


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = org = 1;
  #pragma omp parallel default(shared) copyin (prvt)
  {
    if (prvt != org) {
      #pragma omp critical
      errors += 1;
    }
  }

  prvt = org = 2;
  #pragma omp parallel default(none) shared(org,errors) copyin (prvt)
  {
    if (prvt != org) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("copyin 018 : SUCCESS\n");
    return 0;
  } else {
    printf ("copyin 018 : FAILED\n");
    return 1;
  }
}
