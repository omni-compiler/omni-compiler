static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of shared 007 :
 * ordered に shared を設定できない事を確認
 */

#include <omp.h>


int	errors = 0;
int	thds;

int	shrd;


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
    for (i=0; i<thds; i++) {
      #pragma omp ordered shared (shrd)
      {
	shrd += 1;
      }
    }
  }


  printf ("err_shared 007 : FAILED, can not compile this program.\n");
  return 1;
}
