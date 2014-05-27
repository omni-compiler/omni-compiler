static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of flush 002:
 * smallest statement が存在しない場合のエラーを確認
 */

#include <omp.h>

int	errors = 0;
int	thds;

int	a;

main ()
{


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  {
    if (a)
      #pragma omp flush (a)
  }

  printf ("err_flush 001 : FAILED, can not compile this program.\n");
  return 1;
}
