static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* single 004:
 * data attribute が指定された場合
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;

int	thds;
int	prvt, fprvt;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  fprvt = -1;
  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    #pragma omp barrier

    prvt = id;
    #pragma omp single private(prvt)
    {
      prvt = id;
      waittime(1);
      if (prvt != id) {
	errors += 1;
      }
    }
    prvt = id;
      
    #pragma omp single firstprivate(fprvt)
    {
      if (fprvt != -1) {
	errors += 1;
      }
      fprvt = id;
      waittime(1);
      if (fprvt != id) {
	errors += 1;
      }
    }
    fprvt = id;

  }


  if (errors == 0) {
    printf ("single 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("single 004 : FAILED\n");
    return 1;
  }
}
