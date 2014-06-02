static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of critical 001:
 * critical に識別子が2つ指定された場合の動作を確認
 */

#include <omp.h>


int errors = 0;


main ()
{
  #pragma omp parallel
  {
    #pragma omp critical (lock1) (lock2)
    {
      errors = 1;
    }
  }

  printf ("err_critical 002 : FAILED, can not compile this program.\n");
  return 1;
}
