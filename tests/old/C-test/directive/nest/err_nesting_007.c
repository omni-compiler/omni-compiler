static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of netsting 007 :
 * sections の中に single がある場合の動作を確認。
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

  #pragma omp parallel reduction(+:i)
  {
    #pragma omp sections
    {
      #pragma omp section
      {
        #pragma omp single
	{
	  i += 1;
	}
      }
    }
  }


  printf ("err_nesting 007 : FAILED, can not compile this program.\n");
  return 1;
}
