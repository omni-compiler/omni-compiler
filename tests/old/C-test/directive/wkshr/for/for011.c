static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* for structure 011:
 * data attribute が指定された場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;

int	prvt, fprvt, lprvt;
int	rdct, ordd;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = fprvt = -1;
  rdct = ordd = 0;
  #pragma omp parallel
  {
    #pragma omp for private(prvt) firstprivate(fprvt) lastprivate(lprvt) \
		    reduction(+:rdct) ordered schedule(static,1)
    for (i=0; i<thds; i++) {
      int id = omp_get_thread_num ();

      if (omp_in_parallel() == 0) {
        #pragma omp critical
	errors += 1;
      }
      if (omp_get_num_threads() != thds) {
        #pragma omp critical
	errors += 1;
      }
      if (omp_get_thread_num() >= thds) {
        #pragma omp critical
	errors += 1;
      }

      if (fprvt != -1) {
        #pragma omp critical
	errors += 1;
      }

      prvt = id;
      fprvt = id;
      lprvt = i;
      rdct += 1;

      barrier (thds);

      if (prvt != id) {
        #pragma omp critical
	errors += 1;
      }
      if (fprvt != id) {
        #pragma omp critical
	errors += 1;
      }
      if (lprvt != i) {
        #pragma omp critical
	errors += 1;
      }

      #pragma omp ordered
      {
	if (i != ordd) {
	  errors += 1;
	}
	ordd += 1;
      }
    }
  }

  if (lprvt != thds-1) {
    errors += 1;
  }

  if (rdct != thds) {
    errors += 1;
  }

  if (errors == 0) {
    printf ("for 011 : SUCCESS\n");
    return 0;
  } else {
    printf ("for 011 : FAILED\n");
    return 1;
  }
}
