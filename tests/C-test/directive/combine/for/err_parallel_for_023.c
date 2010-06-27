static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of parallel for 023:
 * for の構造に入口が2つあるケース。
 */

#include <omp.h>

main ()
{
  int	lp, thds;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  goto JUMP_IN;

  #pragma omp parallel for
  for (lp=0; lp<thds; lp++) {
    JUMP_IN:
  }

  printf ("err_parallel_for 023 : FAILED, can not compile this program.\n");
  return 1;
}
