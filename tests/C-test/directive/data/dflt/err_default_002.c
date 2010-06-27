static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of default 002 :
 * for に対して default が指定された時の動作確認
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

    #pragma omp for default(shared)
    for (i=0; i<thds; i++) {
      shrd += 1;
    }
  }


  printf ("err_default 002 : FAILED, can not compile this program.\n");
  return 1;
}
