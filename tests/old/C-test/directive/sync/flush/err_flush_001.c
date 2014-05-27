static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of flush 001:
 * 複数指定された場合のエラーを確認
 */

#include <omp.h>

int	errors = 0;
int	thds;

int	a,b;

main ()
{


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp flush (a) (b)

  printf ("err_flush 001 : FAILED, can not compile this program.\n");
  return 1;
}
