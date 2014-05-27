static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 007:
 * data attribute が指定された場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;

int	tprvt;
#pragma omp threadprivate (tprvt)

int	prvt, fprvt;
int	shrd, shrd2;
int	rdct;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  tprvt = fprvt = thds;
  rdct = 0;
  shrd2 = 0;
#if defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__) 
  #pragma omp parallel default(none) shared(errors,thds,shrd,shrd2) private(prvt) \
    firstprivate(fprvt) copyin(tprvt) reduction(+:rdct)
#else
  #pragma omp parallel default(none) shared(errors,thds,shrd,shrd2) private(prvt) \
    firstprivate(fprvt) copyin(tprvt) reduction(+:rdct) shared(stderr)
#endif
  {
    int	id = omp_get_thread_num ();

    if (omp_in_parallel() == 0) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }
    if (omp_get_num_threads() != thds) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }
    if (omp_get_thread_num() >= thds) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }

    if (tprvt != thds) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }
    if (fprvt != thds) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }

    prvt = id;
    #pragma omp barrier
    if (prvt != id) {
      #pragma omp critical
      {
	ERROR(errors);
      }
    }

    shrd = id;
    #pragma omp barrier
    if (shrd != id) {
      /* generate thds - 1 errors, because shrd is shared variable */
      #pragma omp critical
      shrd2 += 1;
    }

    rdct += 1;
  }

  if (shrd2 != thds - 1) {
    ERROR(errors);
  }

  if (rdct != thds) {
    ERROR(errors);
  }


  if (errors == 0) {
    printf ("parallel 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 007 : FAILED\n");
    return 1;
  }
}
