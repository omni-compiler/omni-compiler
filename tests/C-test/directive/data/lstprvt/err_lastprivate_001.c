static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of lastprivate 001 :
 * const型の変数をlastprivate した場合
 */

#include <omp.h>


int	errors = 0;
int	thds;


const int	prvt;


main ()
{
  int	i;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel for lastprivate(prvt)
  for (i=0;  i<thds;  i++) {
  }


  printf ("err_lastprivate 001 : FAILED, can not compile this program.\n");
  return 1;
}
