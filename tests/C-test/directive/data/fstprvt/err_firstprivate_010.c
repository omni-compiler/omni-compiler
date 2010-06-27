static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of firstprivate 010 :
 * ordered には、firstprivate を設定出来ない事を確認
 */

#include <omp.h>


int	errors = 0;
int	thds;


int	prvt;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel for ordered
  for (i=0;  i<thds;  i++) {
    #pragma omp ordered firstprivate(prvt)
    {
    }
  }


  printf ("err_firstprivate 010 : FAILED, can not compile this program.\n");
  return 1;
}
