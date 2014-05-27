static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of sections 004:
 * nowaitを複数回指定した場合のエラーを確認
 */

#include <omp.h>


main ()
{
  int	thds;
  int	buf[3];


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  memset (buf, 0, sizeof (buf));
  #pragma omp parallel
  {
    #pragma omp sections nowait nowait
    {
      #pragma omp section
      buf[0] += 1;
      buf[1] += 2;

      #pragma omp section
      buf[2] += 3;
    }
  }

  printf ("err_sections 004 : FAILED, can not compile this program.\n");
  return 1;
}
