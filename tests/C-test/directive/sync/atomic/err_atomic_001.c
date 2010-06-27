static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of atomic 001:
 * atomic に = が使えないことを確認
 */

#include <omp.h>


int	thds;

int	v = 0;


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
    int	i;

    #pragma omp for
    for (i=0; i<thds; i++) {
      #pragma omp atomic
      v = i;
    }
  }

  printf ("err_atomic 001 : FAILED, can not compile this program.\n");
  return 1;
}
