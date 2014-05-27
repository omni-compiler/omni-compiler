static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 006:
 * nested parallel section の動作テスト
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;


void
func ()
{
  #pragma omp parallel
  {
    if (omp_get_num_threads () != 1) {
      #pragma omp critical
      errors += 1;
    }
    if (omp_get_thread_num () != 0) {
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
  if (thds == 2) {
    printf ("should be run this program on multi threads(CPU >= 4).\n");
    exit (0);
  }
  omp_set_nested (0);
  omp_set_dynamic (0);


  omp_set_num_threads (2);
  #pragma omp parallel
  {
    #pragma omp parallel
    {
      if (omp_get_num_threads () != 1) {
	#pragma omp critical
	errors += 1;
      }
      if (omp_get_thread_num () != 0) {
        #pragma omp critical
	errors += 1;
      }
    }

    func ();
  }

  if (errors == 0) {
    printf ("parallel 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 006 : FAILED\n");
    return 1;
  }
}
