static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of for 001:
 * 基本ブロックに対して、for を指定した場合
 */

#include <omp.h>


int	errors = 0;


main ()
{

  #pragma omp parallel
  {
    #pragma omp for
    {
      errors = 1;
    }
  }

  printf ("err_for 001 : FAILED, can not compile this program.\n");
  return 1;
}
