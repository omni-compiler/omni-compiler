static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of firstprivate 012 :
 * single に lastprivate は設定できない。
 */

#include <omp.h>


int	errors = 0;
int	thds;


int	prvt;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = 0;
  #pragma omp parallel
  {
    #pragma omp single lastprivate (prvt)
    {
      prvt = 1;
    }
  }
  if(prvt != 1) {
    errors += 1;
  }


  printf ("err_lastprivate 012: FAILED, can not compile this program.\n");
  return 1;
}
