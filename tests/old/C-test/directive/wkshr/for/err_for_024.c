static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of for 024:
 * for の構造に出口が2つあるケース。
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

  #pragma omp parallel
  {
    #pragma omp for
    for (lp=0; lp<thds; lp++) {
      if (omp_get_thread_num () == 0)
	break;
    }
  }

  printf ("err_for 024 : FAILED, can not compile this program.\n");
  return 1;
}
