static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 012 :
 * threadprivateされた変数が for directiveが指定された
 * for ループのループカウンタに指定できる事を確認
 */

#include <omp.h>
#include "omni.h"


#define	STRIDE		100
#define LOOPNUM		(thds * STRIDE)

int	errors = 0;
int	thds, *buf, sum;

int	x;
#pragma omp threadprivate (x)


void
clear ()
{
  int	i;


  sum = 0;
  for (i=0; i<LOOPNUM; i++) {
    buf[x] = -1;
  }
}


void
check ()
{
  int	i;


  if (sum != LOOPNUM) {
    errors += 1;
  }
  for (i=0; i<LOOPNUM; i++) {
    if (buf[i] != i % thds) {
      errors += 1;
    }
  }
}


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  buf = (int*) malloc (sizeof(int) * LOOPNUM);
  if (buf == NULL) {
    printf ("can not allocate memory\n");
    exit (1);
  }
  omp_set_dynamic (0);


  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule(static, 1)
    for (x=0; x<LOOPNUM; x++) {
      buf[x] = omp_get_thread_num ();
      #pragma omp critical
      sum += 1;
    }
  }
  check ();


  if (errors == 0) {
    printf ("threadprivate 012 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 012 : FAILED\n");
    return 1;
  }
}
