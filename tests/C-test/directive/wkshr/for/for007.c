static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* for 007:
 * continue が通る事を確認
 */
#include <omp.h>
#include "omni.h"


#define	MAX_STRIDE	10
#define LOOPNUM		(MAX_STRIDE*thds)

int	thds;
int	*buf;


void
clear ()
{
  int lp;
  
  for (lp=0; lp<=LOOPNUM; lp++) {
    buf[lp] = -1;
  }
}


int
check_result (int s)
{
  int	lp, lp2, id;

  int	err = 0;


  for (lp=0; lp<LOOPNUM; ) {
    id = buf[lp];
    if (id<0 || thds<=id) {
      err += 1;
    }
    for (lp2=0; lp2<(LOOPNUM-lp); lp2++) {
      if (buf[lp+lp2] != id) {
	if (lp2+1<s) {
	  err += 1;
	}
      }
    }
    lp += lp2;
  }
  if (buf[LOOPNUM] != -1) {
    err += 1;
  }

  return err;
}


main ()
{
  int	lp;

  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  buf = (int *) malloc (sizeof (int) * (LOOPNUM + 1));
  if (buf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  omp_set_dynamic (0);

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (runtime)
    for (lp=0; lp<LOOPNUM; lp++) {
      buf[lp] = omp_get_thread_num ();
      if (omp_get_thread_num () == 0)
	continue;
    }
  }
  errors += check_result (1);

  #pragma omp parallel
  {
    #pragma omp for schedule (static)
    for (lp=0; lp<LOOPNUM; lp++) {
      buf[lp] = omp_get_thread_num ();
      if (omp_get_thread_num () == 0)
	continue;
    }
  }
  errors += check_result (1);

  #pragma omp parallel
  {
    #pragma omp for schedule (dynamic)
    for (lp=0; lp<LOOPNUM; lp++) {
      buf[lp] = omp_get_thread_num ();
      if (omp_get_thread_num () == 0)
	continue;
    }
  }
  errors += check_result (1);

  #pragma omp parallel
  {
    #pragma omp for schedule (guided)
    for (lp=0; lp<LOOPNUM; lp++) {
      buf[lp] = omp_get_thread_num ();
      if (omp_get_thread_num () == 0)
	continue;
    }
  }
  errors += check_result (1);

  if (errors == 0) {
    printf ("for 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("for 007 : FAILED\n");
    return 1;
  }
}
