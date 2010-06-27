static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* copyin 001 :
 * copyin の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	org;
int	prvt;
#pragma omp threadprivate(prvt)


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = org = -1;
  #pragma omp parallel copyin (prvt)
  {
    if (prvt != org) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }
  }

  prvt = org = 0;
  #pragma omp parallel copyin (prvt)
  {
    if (prvt != org) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }
  }

  prvt = org = 1;
  #pragma omp parallel copyin (prvt)
  {
    if (prvt != org) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }
  }


  if (errors == 0) {
    printf ("copyin 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("copyin 001 : FAILED\n");
    return 1;
  }
}
