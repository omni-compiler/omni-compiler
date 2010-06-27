static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* copyin 014 :
 * 配列変数に対して copyin 宣言をした場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	ARRAYSIZ	1024

int	errors = 0;
int	thds;


int	org[ARRAYSIZ];
int	prvt[ARRAYSIZ];
#pragma omp threadprivate(prvt)


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  for (i=0;  i<ARRAYSIZ;  i++) {
    prvt[i] = org[i] = i - 1;
  }
  #pragma omp parallel copyin (prvt) private(i)
  {
    for (i=0; i<ARRAYSIZ; i++) {
      if (prvt[i] != org[i]) {	
        #pragma omp critical
	errors += 1;
      }
    }
    if (sizeof(prvt) != sizeof(int) * ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
  }


  for (i=0;  i<ARRAYSIZ;  i++) {
    prvt[i] = org[i] = i;
  }
  #pragma omp parallel copyin (prvt) private(i)
  {
    for (i=0; i<ARRAYSIZ; i++) {
      if (prvt[i] != org[i]) {	
        #pragma omp critical
	errors += 1;
      }
    }
    if (sizeof(prvt) != sizeof(int) * ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
  }


  for (i=0;  i<ARRAYSIZ;  i++) {
    prvt[i] = org[i] = i + 1;
  }
  #pragma omp parallel copyin (prvt) private(i)
  {
    for (i=0; i<ARRAYSIZ; i++) {
      if (prvt[i] != org[i]) {	
        #pragma omp critical
	errors += 1;
      }
    }
    if (sizeof(prvt) != sizeof(int) * ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("copyin 014 : SUCCESS\n");
    return 0;
  } else {
    printf ("copyin 014 : FAILED\n");
    return 1;
  }
}
