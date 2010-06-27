static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of barrier 001:
 * barrier directive が block に含まれていない場合のエラーを確認
 */

#include <omp.h>


int	thds;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel
  {
    if (1)
      #pragma omp barrier
  }


  printf ("err_barrier 001 : FAILED, can not compile this program.\n");
  return 1;
}
