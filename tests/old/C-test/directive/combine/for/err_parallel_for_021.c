static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of parallel for 021:
 * schedule が複数回指定された場合にエラーになることを確認
 */

#include <omp.h>

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
  int	lp, lp2;

  int	err = 0;


  for (lp=0; lp*s<LOOPNUM; lp++) {
    for (lp2=0; lp2<s; lp2++) {
      if (LOOPNUM<=lp*s+lp2) {
	goto LOOPEND;
      }
      if (buf[lp*s+lp2] != lp%thds) {
	err += 1;
      }
    }
  }
 LOOPEND:
  if (buf[LOOPNUM] != -1) {
    err += 1;
  }

  return err;
}


main ()
{
  int	lp;

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
  #pragma omp parallel for schedule (static) schedule (dynamic)
  for (lp=0; lp<LOOPNUM; lp++) {
    buf[lp] = omp_get_thread_num ();
  }

  printf ("err_parallel_for 021 : FAILED, can not compile this program.\n");
  return 1;
}
