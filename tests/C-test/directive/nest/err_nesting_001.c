static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of netsting 003 :
 * for の中に sections がある場合の動作を確認。
 */

#include <omp.h>


int	errors = 0;
int	thds;
int	i;

main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  #pragma omp parallel reduction(+:i)
  {
    int	j;

    #pragma omp for
    for (j=0; j<thds; j++) {
      #pragma omp sections
      {
	#pragma omp section
	i ++;
	#pragma omp section
	i += 2;
      }
    }
  }

  printf ("err_nesting 003 : FAILED, can not compile this program.\n");
  return 1;
}
