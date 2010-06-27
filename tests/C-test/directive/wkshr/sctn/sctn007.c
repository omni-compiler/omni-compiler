static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* sections 007:
 * data attribute を設定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;

int	shrd;
int	prvt, fprvt, lprvt, rdct;



main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  fprvt = -1;
  rdct = 0;

  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    #pragma omp sections private(prvt) firstprivate(fprvt) \
	                 lastprivate(lprvt) reduction(+:rdct)
    {

      #pragma omp section
      {
	if (fprvt != -1) {
	  #pragma omp critical
	  errors += 1;
	}

	prvt = id;
	rdct += 1;
	lprvt = id;

	barrier (2);

	if (prvt != id) {
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
	if (fprvt != -1) {
	  #pragma omp critical
	  errors += 1;
	}

	prvt = id;
	rdct += 1;
	lprvt = id;
	shrd = id;

	barrier (2);

	if (prvt != id) {
	  #pragma omp critical
	  errors += 1;
	}
	if (lprvt != id) {
	  #pragma omp critical
	  errors += 1;
	}
      }
    }
  }

  if (rdct != 2) {
    errors += 1;
  }
  if (lprvt != shrd) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("sections 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("sections 007 : FAILED\n");
    return 1;
  }
}
