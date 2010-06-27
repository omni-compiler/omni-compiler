static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 011 :
 * threadprivateされた変数がscheduleに使用できる事を確認
 */

#include <omp.h>
#include "omni.h"


#define	STRIDE		100
#define LOOPNUM		(thds * STRIDE)

int	errors = 0;
int	thds, *buf;

int	x;
#pragma omp threadprivate (x)


void
check ()
{
  int	i, j;

  for (i=0; i<LOOPNUM; i+=STRIDE) {
    for (j=0; j<STRIDE; j++) {
      if (buf[i+j] != (i/STRIDE)%thds) {
	errors += 1;
      }
    }
  }
}


void
func ()
{
  int i;

  #pragma omp for schedule(static, x)
  for (i=0; i<LOOPNUM; i++) {
    buf[i] = omp_get_thread_num ();
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
  buf = (int*) malloc (sizeof(int) * LOOPNUM);
  if (buf == NULL) {
    printf ("can not allocate memory\n");
    exit (1);
  }
  omp_set_dynamic (0);

  x = STRIDE;
  #pragma omp parallel copyin(x)
  {
    #pragma omp for schedule(static, x)
    for (i=0; i<LOOPNUM; i++) {
      buf[i] = omp_get_thread_num ();
    }
  }
  check ();


  #pragma omp parallel
  func ();
  check ();


  if (errors == 0) {
    printf ("threadprivate 011 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 011 : FAILED\n");
    return 1;
  }
}
