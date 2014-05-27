static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of for 004:
 * if に対して、for を指定した場合
 */

#include <omp.h>


main ()
{
  int	true = 1;

  #pragma omp parallel
  {
    #pragma omp for
    if (true) {
      true = 0;
    }
  }

  printf ("err_for 004 : FAILED, can not compile this program.\n");
  return 1;
}
