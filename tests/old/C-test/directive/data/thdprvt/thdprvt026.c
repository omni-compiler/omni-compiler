static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 026 : 
 * threadprivate 変数をループカウンタに使用した場合
 */

#include <omp.h>
#include "omni.h"


#define LOOPNUM		(100*thds)

int	errors = 0;
int	thds;
int	*buff;


int	prvt;
#pragma omp threadprivate(prvt)


void
check(int s)
{
  int   i,j, id;

  for (i=0; i<LOOPNUM; i+=s) {
    id = (i/s) % thds;

    for (j=0; j<s; j++) {
      if ((i+j) < LOOPNUM) {
        if (buff[i+j] != id) {
          #pragma omp critical
          errors += 1;
        }
      }
    }
  }
}


main ()
{
  int	ln;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  buff = (int *)malloc(sizeof(int) * LOOPNUM);
  if (buff == NULL) {
    printf ("can not allocate memory.\n");
  }
  omp_set_dynamic (0);
  ln = LOOPNUM;


  #pragma omp parallel
  {
    #pragma omp for schedule(static,ln/thds)
    for (prvt=0; prvt<LOOPNUM; prvt++) {
      buff[prvt] = omp_get_thread_num ();
    }
  }
  check (LOOPNUM/thds);

  #pragma omp parallel for schedule(static,ln/thds)
  for (prvt=0; prvt<LOOPNUM; prvt++) {
    buff[prvt] = omp_get_thread_num ();
  }
  check (LOOPNUM/thds);


  if (errors == 0) {
    printf ("threadprivate 026 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 026 : FAILED\n");
    return 1;
  }
}
