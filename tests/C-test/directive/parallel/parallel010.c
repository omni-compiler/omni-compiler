static char rcsid[] = "";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 010:
 *   inside nested parallel region is serialized by "if" cluse.
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;

int	true = 1;
int	false = 0;


void
check_parallel (int thds)
{
  if(thds == 0) {
    if (omp_in_parallel () != 0) {
      #pragma omp critical
      {
	ERROR (errors);
      }
    }
    if (omp_get_num_threads() != 1) {
      #pragma omp critical
      {
	ERROR (errors);
      }
    }

  } else {
    if (omp_in_parallel () == 0) {
      #pragma omp critical
      {
	ERROR (errors);
      }
    }
    if (omp_get_num_threads() != thds) {
      #pragma omp critical
      {
	ERROR (errors);
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
  omp_set_nested (0);


  #pragma omp parallel
  {
    #pragma omp parallel if (false)
    {
      /* this nested parallel is serialized, by if(false) cluse */
      check_parallel (1);
    }
  }


  if (errors == 0) {
    printf ("parallel 010 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 010 : FAILED\n");
    return 1;
  }
}
