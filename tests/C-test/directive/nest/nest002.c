static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* netsting 002 :
 * check nested parallel region at nested parallel is enabled.
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	sum;


void
func_nesting ()
{
  #pragma omp parallel
  {
    int	add;

    if (omp_get_num_threads () == 1) {
      add = 2;
      printf ("nested parallel is serialized.\n");
    } else {
      add = 1;
    }

    #pragma omp critical
    {
      sum += add;
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
  omp_set_num_threads (2);
  omp_set_nested (1);
  if (omp_get_nested () == 0) {
    printf ("test skipped.\n");
    exit(0);
  }

  sum = 0;
  #pragma omp parallel
  {
    #pragma omp parallel
    {
      int	add;

      if (omp_get_num_threads () == 1) {
	add = 2;
	printf ("nested parallel is serialized.\n");
      } else {
	add = 1;
      }

      #pragma omp critical
      {
	sum += add;
      }
    }
  }
  if (sum != 2*2) {	
    errors += 1;
  }


  sum = 0;
  #pragma omp parallel
  func_nesting ();
  if (sum != 2*2) {	
    errors += 1;
  }


  if (errors == 0) {
    printf ("nesting 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("nesting 002 : FAILED\n");
    return 1;
  }
}
