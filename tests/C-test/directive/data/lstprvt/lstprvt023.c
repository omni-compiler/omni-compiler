static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* lastprivate 023 :
 * lastprivate 宣言した変数をfor loop のカウンタに使用。
 */
#include <omp.h>
#include "omni.h"


#define	LOOPNUM	103

int	errors = 0;
int	thds;

int	prvt;


main ()
{
  int	sum;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  sum = 0;
  #pragma omp parallel 
  {
    #pragma omp for lastprivate(prvt) 
    for (prvt=0; prvt<LOOPNUM; prvt ++) {
      #pragma omp critical
      sum += 1;
    }
  }
  if (sum != LOOPNUM) {
    errors += 1;
  }
  if (prvt != LOOPNUM) {
    errors += 1;
  }


  sum = 0;
  #pragma omp parallel 
  {
    #pragma omp for schedule(static) lastprivate(prvt)
    for (prvt=0; prvt<LOOPNUM; prvt ++) {
      #pragma omp critical
      sum += 1;
    }
  }
  if (sum != LOOPNUM) {
    errors += 1;
  }
  if (prvt != LOOPNUM) {
    errors += 1;
  }


  sum = 0;
  #pragma omp parallel 
  {
    #pragma omp for schedule(dynamic) lastprivate(prvt)
    for (prvt=0; prvt<LOOPNUM; prvt ++) {
      #pragma omp critical
      sum += 1;
    }
  }
  if (sum != LOOPNUM) {
    errors += 1;
  }
  if (prvt != LOOPNUM) {
    errors += 1;
  }


  sum = 0;
  #pragma omp parallel 
  {
    #pragma omp for schedule(guided) lastprivate(prvt)
    for (prvt=0; prvt<LOOPNUM; prvt ++) {
      #pragma omp critical
      sum += 1;
    }
  }
  if (sum != LOOPNUM) {
    errors += 1;
  }
  if (prvt != LOOPNUM) {
    errors += 1;
  }


  sum = 0;
  #pragma omp parallel 
  {
    #pragma omp for schedule(runtime) lastprivate(prvt)
    for (prvt=0; prvt<LOOPNUM; prvt ++) {
      #pragma omp critical
      sum += 1;
    }
  }
  if (sum != LOOPNUM) {
    errors += 1;
  }
  if (prvt != LOOPNUM) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("lastprivate 023 : SUCCESS\n");
    return 0;
  } else {
    printf ("lastprivate 023 : FAILED\n");
    return 1;
  }
}
