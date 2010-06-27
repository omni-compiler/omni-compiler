static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of netsting 013 :
 * critical の中にforがある場合
 */

#include <omp.h>


int	errors = 0;
int	thds;
int	i;


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
    #pragma omp critical
    {
      int i;

      #pragma omp for
      for (i=0; i<thds; i++) {
	i += 1;
      }
    }
  }


  printf ("err_nesting 013 : FAILED, can not compile this program.\n");
  return 1;
}
