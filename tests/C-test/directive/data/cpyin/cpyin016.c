static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* copyin 016 :
 * parallel for に対して copyin 宣言をした場合の動作確認
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
  int	i;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = org = 1;

  #pragma omp parallel for copyin (prvt)
  for (i=0; i<thds; i++) {
    if (prvt != org) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("copyin 016 : SUCCESS\n");
    return 0;
  } else {
    printf ("copyin 016 : FAILED\n");
    return 1;
  }
}
