static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of netsting 023 :
 * ordered の中に barrier が存在する場合
 */

#include <omp.h>


int	errors = 0;
int	thds;
int	sum;


void 
func ()
{
  #pragma omp ordered
  {
    #pragma omp barrier
  }
}


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

    #pragma omp for ordered
    for (i=0; i<1; i++) {
      func();
    }
  }


  printf ("err_nesting 023 : FAILED, can not compile this program.\n");
  return 1;
}
