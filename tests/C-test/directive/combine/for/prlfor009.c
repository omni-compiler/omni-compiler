static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel for 009:
 * forの動作を確認
 */

#include <omp.h>
#include "omni.h"


#define LOOPNUM	(thds * 1000)


int	thds;
int	use_thd = 0;
int	*buf, *idbuf;


void clear ()
{
  int lp;
  
  for (lp=0; lp<=LOOPNUM; lp++) {
    buf[lp] = -1;
  }
  for (lp=0; lp<thds; lp++) {
    idbuf[lp] = -1;
  }

}


int
check_result ()
{
  int	lp, thd, cnt;

  int	err = 0;


  for (lp = 0; lp<LOOPNUM; lp++) {
    thd = buf[lp];
    if (thds <= thd) {
      err += 1;
    } else {
      idbuf[thd] = 1;
    }
  }
  if (buf[LOOPNUM] != -1) {
    err += 1;
  }

  cnt = 0;
  for (lp=0; lp<thds; lp++) {
    if (idbuf[lp] == 1) {
      cnt ++;
    }
  }
  if (use_thd < cnt) {
    use_thd = cnt;
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
  idbuf = (int *) malloc (sizeof (int) * thds);
  if (idbuf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  omp_set_dynamic (0);


  clear ();
  #pragma omp parallel for
  for (lp=0; lp<LOOPNUM; lp++) {
    buf[lp] = omp_get_thread_num ();
  }
  errors += check_result ();


  if (errors == 0) {
    printf ("parallel for 009 : SUCCESS\n");
    if (use_thd == 1) {
      printf ("but, for loop executed by 1 thread.\n");
    }
    return 0;
  } else {
    printf ("parallel for 009 : FAILED\n");
    return 1;
  }
}
