static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 022 :
 * parallel sctions に private を設定できることを確認。
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	prvt;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel sections private (prvt)
  {
    #pragma omp section
    {
      int	id = omp_get_thread_num ();

      prvt = id;
      barrier (2);
      if (prvt != id) {
	#pragma omp critical
	errors += 1;
      }
    }
    #pragma omp section
    {
      int	id = omp_get_thread_num ();

      prvt = id;
      barrier (2);
      if (prvt != id) {
	#pragma omp critical
	errors += 1;
      }
    }
  }

  if (errors == 0) {
    printf ("private 022 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 022 : FAILED\n");
    return 1;
  }
}
