static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* netsting 001 :
 * check nested parallel region at nested parallel is disabled.
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


void
func_nesting ()
{
  #pragma omp parallel
  {
    if (omp_get_thread_num () != 0 ||
	omp_get_num_threads () != 1) {
      #pragma omp critical 
      errors += 1;
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
  omp_set_nested (0);

  #pragma omp parallel
  {
    #pragma omp parallel
    {
      if (omp_get_thread_num () != 0 ||
	  omp_get_num_threads () != 1) {
        #pragma omp critical 
	errors += 1;
      }
    }
  }

  #pragma omp parallel
  func_nesting ();


  if (errors == 0) {
    printf ("nesting 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("nesting 001 : FAILED\n");
    return 1;
  }
}
