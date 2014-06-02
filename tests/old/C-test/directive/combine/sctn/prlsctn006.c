static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel sections 006:
 * data attribute を設定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	thds, ids[2];
int	errors = 0;

int	shrd;
int	prvt, fprvt, lprvt, tprvt;
int	rdct;

#pragma omp threadprivate (tprvt)


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  fprvt = tprvt = shrd = -1;
  rdct = 0;

  #pragma omp parallel sections private(prvt) firstprivate(fprvt) \
				lastprivate(lprvt) reduction(+:rdct) \
				default(none) shared(shrd,errors,thds,ids) copyin(tprvt)
  {
    #pragma omp section
    {
      int id = omp_get_thread_num ();

      if (omp_get_num_threads() != thds) {
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
      barrier (2);

      ids[0] = id;
      prvt = id;
      shrd = id;
      fprvt = id;
      lprvt = id;
      rdct += 1;

      barrier (2);
     
      if (shrd != id) {
	#pragma omp critical
	errors += 1;
      }
      if (prvt != id) {
	#pragma omp critical
	errors += 1;
      }
      if (fprvt != id) {
	#pragma omp critical
	errors += 1;
      }
      if (lprvt != id) {
	#pragma omp critical
	errors += 1;
      }
    }

    #pragma omp section
    {
      int id = omp_get_thread_num ();

      if (omp_get_num_threads() != thds) {
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

      barrier (2);

      ids[1] = id;
      prvt = id;
      shrd = id;
      fprvt = id;
      lprvt = id;
      rdct += 1;

      barrier (2);
     
      if (shrd != id) {
	#pragma omp critical
	errors += 1;
      }
      if (prvt != id) {
	#pragma omp critical
	errors += 1;
      }
      if (fprvt != id) {
	#pragma omp critical
	errors += 1;
      }
      if (lprvt != id) {
	#pragma omp critical
	errors += 1;
      }

      shrd = lprvt;
    }
  }

  if (rdct != 2) {
    errors += 1;
  }
  if (lprvt != shrd) {
    errors += 1;
  }
  if (ids[0] == ids[1]) {
    errors += 1;
  }


  if (errors == 1) {
    printf ("parallel sections 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel sections 006 : FAILED\n");
    return 1;
  }
}
