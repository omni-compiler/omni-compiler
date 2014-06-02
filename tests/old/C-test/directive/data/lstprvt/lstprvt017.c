static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* lastprivate 017 :
 * 配列変数にlastprivateを宣言した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define ARRAYSIZ	2

int	errors = 0;
int	thds;

int	prvt[ARRAYSIZ];


void
func (int t)
{
  int	i, j;


  #pragma omp for schedule(static,1) lastprivate (prvt)
  for (i=0; i<thds; i++) {
    for (j=0; j<ARRAYSIZ; j++) {
      prvt[j] = i+j;
    }
    barrier (t);
    for (j=0; j<ARRAYSIZ; j++) {
      if (prvt[j] != i+j) {
        #pragma omp critical
	errors += 1;
      }
    }
    if (sizeof(prvt) != sizeof(int)*ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
    if (i==0) {
      waittime (1);
    }
    for (j=0; j<ARRAYSIZ; j++) {
      prvt[j] = i+j;
    }
  }

  for (j=0; j<ARRAYSIZ; j++) {
    if (prvt[j] != (thds-1)+j) {
      #pragma omp critical
      errors += 1;
    }
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
    int	j;

    #pragma omp for schedule(static,1) lastprivate (prvt)
    for (i=0; i<thds; i++) {
      for (j=0; j<ARRAYSIZ; j++) {
	prvt[j] = i+j;
      }
      barrier (thds);
      for (j=0; j<ARRAYSIZ; j++) {
	if (prvt[j] != i+j) {
          #pragma omp critical
	  errors += 1;
	}
      }
      if (sizeof(prvt) != sizeof(int)*ARRAYSIZ) {
        #pragma omp critical
	errors += 1;
      }
      if (i==0) {
	waittime (1);
      }
      for (j=0; j<ARRAYSIZ; j++) {
	prvt[j] = i+j;
      }
    }

    for (j=0; j<ARRAYSIZ; j++) {
      if (prvt[j] != (thds-1)+j) {
        #pragma omp critical
	errors += 1;
      }
    }
  }


  #pragma omp parallel
  func (thds);


  func (1);


  if (errors == 0) {
    printf ("lastprivate 017 : SUCCESS\n");
    return 0;
  } else {
    printf ("lastprivate 017 : FAILED\n");
    return 1;
  }
}
