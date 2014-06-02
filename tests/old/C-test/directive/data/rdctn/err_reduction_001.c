static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of reduction 001 :
 * 異なる operation の reduction に同じ変数名が宣言された場合。
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	rdct;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel reduction(+:rdct) reduction(-:rdct)
  {
  }



  printf ("err_reduction 001 : FAILED, can not compile this program.\n");
  return 1;
}
