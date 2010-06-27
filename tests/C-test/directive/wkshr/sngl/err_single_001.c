static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of single 001:
 * nowait が複数回指定された場合
 */

#include <omp.h>


main ()
{
  int	thds;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  #pragma omp parallel
  {
    #pragma omp single nowait nowait
    {
    }
  }

  printf ("err_single 001 : FAILED, can not compile this program.\n");
  return 1;
}
