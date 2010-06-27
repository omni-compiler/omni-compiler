static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of threadprivate 005 :
 * threadprivate された変数が lastprivate に使えないことを確認
 */

#include <omp.h>


int	errors = 0;
int	thds;


int	x;
#pragma omp threadprivate(x)


main ()
{
  int	i;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel for lastprivate(x)
  for (i=0; i<thds; i++) {
    int	id = omp_get_thread_num ();
    x = id;
  }


  printf ("err_threadprivate 005 : FAILED, can not compile this program.\n");
  return 1;
}
