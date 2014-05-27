static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* netsting 004 :
 * nested parallel region を通せば、for,sections,singleが、
 * ネストできる事を確認。
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	sum = 0;


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
    int	i;

    #pragma omp for
    for (i=0; i<thds; i++) {
      #pragma omp parallel 
      {
        #pragma omp sections
	{
          #pragma omp section
	  {
	    #pragma omp parallel
	    {
	      #pragma omp single
	      {
		#pragma omp critical	
		{
		  sum += 1;
		}
	      }
	    }
	  }
	}
      }
    }
  }
  if (sum != thds) {
    ERROR (errors);
  }

  if (errors == 0) {
    printf ("nesting 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("nesting 004 : FAILED\n");
    return 1;
  }
}
