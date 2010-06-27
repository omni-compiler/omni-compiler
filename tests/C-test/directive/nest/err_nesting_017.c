static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of netsting 017 :
 * ordered の中に sections がある場合
 */

#include <omp.h>


int	errors = 0;
int	thds;
int	i;


void
func ()
{
  #pragma omp ordered
  {
    #pragma omp sections
    {
      #pragma omp section
      {
	i += 1;
      }
    }
  }
}

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
    int k;

    #pragma omp for ordered
    for (k=0; k<thds; k++){
      func ();
    }
  }


  printf ("err_nesting 017 : FAILED, can not compile this program.\n");
  return 1;
}
