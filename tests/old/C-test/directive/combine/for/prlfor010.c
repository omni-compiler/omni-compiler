static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel for structure 010:
 * data attribute が指定された場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;

int	shrd;
int	prvt, fprvt, lprvt, tprvt;
int	rdct, ordd;

#pragma omp threadprivate (tprvt)


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = fprvt = tprvt = shrd = -1;
  rdct = ordd = 0;
  #pragma omp parallel for private(prvt) firstprivate(fprvt) lastprivate(lprvt) \
	                   reduction(+:rdct) ordered schedule(static,1) \
		           default(none) shared(thds,shrd,errors,ordd) copyin(tprvt)
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

    if (tprvt != -1) {
      #pragma omp critical
      errors += 1;
    }

    if (shrd != -1) {
      #pragma omp critical
      errors += 1;
    }

    barrier (thds);

    prvt = id;
    fprvt = id;
    lprvt = i;
    rdct += 1;
    tprvt = id;
    shrd = id;

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
    if (tprvt != id) {
      #pragma omp critical
      errors += 1;
    }
    if (shrd != i) {
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

  if (lprvt != thds-1) {
    errors += 1;
  }

  if (rdct != thds) {
    errors += 1;
  }

  if (errors == thds-1) {
    printf ("parallel for 010 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel for 010 : FAILED\n");
    return 1;
  }
}
