static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* lastprivate 004 :
 * 複数の変数に対してlastprivateを宣言した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	prvt1, prvt2, prvt3;


void
func (int t)
{
  int	i;


  #pragma omp for schedule(static,1) lastprivate (prvt1,prvt2,prvt3)
  for (i=0; i<thds; i++) {
    prvt1 = i;
    prvt2 = i;
    prvt3 = i;
    barrier (t);
    if (prvt1 != i) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt2 != i) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt3 != i) {
      #pragma omp critical
      errors += 1;
    }
    if (i==0) {
      waittime (1);
    }
    prvt1 = i;
    prvt2 = i;
    prvt3 = i;
  }

  if (prvt1 != thds - 1) {
    #pragma omp critical
    errors += 1;
  }
  if (prvt2 != thds - 1) {
    #pragma omp critical
    errors += 1;
  }
  if (prvt3 != thds - 1) {
    #pragma omp critical
    errors += 1;
  }
}


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel
  {
    #pragma omp for schedule(static,1) lastprivate (prvt1,prvt2) lastprivate (prvt3)
    for (i=0; i<thds; i++) {
      prvt1 = i;
      prvt2 = i;
      prvt3 = i;
      barrier (thds);
      if (prvt1 != i) {
	#pragma omp critical
	errors += 1;
      }
      if (prvt2 != i) {
	#pragma omp critical
	errors += 1;
      }
      if (prvt3 != i) {
	#pragma omp critical
	errors += 1;
      }
      if (i==0) {
	waittime (1);
      }
      prvt1 = i;
      prvt2 = i;
      prvt3 = i;
    }

    if (prvt1 != thds - 1) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt2 != thds - 1) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt3 != thds - 1) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel
  func (thds);


  func (1);


  if (errors == 0) {
    printf ("lastprivate 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("lastprivate 004 : FAILED\n");
    return 1;
  }
}
