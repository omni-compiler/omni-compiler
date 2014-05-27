static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* copyin 002 :
 * char 変数に対して copyin 宣言をした場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	prvt1;
int	prvt2;
int	prvt3;
#pragma omp threadprivate(prvt1,prvt2,prvt3)


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt1 = -1;
  prvt2 = 0;
  prvt3 = 1;
  #pragma omp parallel copyin (prvt1) copyin (prvt2) copyin (prvt3)
  {
    if (prvt1 != -1 || prvt2 != 0 || prvt3 != 1) {
      #pragma omp critical
      errors += 1;
    }
  }

  prvt1 = -2;
  prvt2 = 1;
  prvt3 = 2;
  #pragma omp parallel copyin (prvt1,prvt2) copyin (prvt3)
  {
    if (prvt1 != -2 || prvt2 != 1 || prvt3 != 2) {
      #pragma omp critical
      errors += 1;
    }
  }

  prvt1 = -3;
  prvt2 = 2;
  prvt3 = 3;
  #pragma omp parallel copyin (prvt1,prvt2,prvt3)
  {
    if (prvt1 != -3 || prvt2 != 2 || prvt3 != 3) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("copyin 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("copyin 002 : FAILED\n");
    return 1;
  }
}
