static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel for 012:
 * parallel for if が成立しない場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	*buf;

int	errors = 0;


int sameas(int v)
{
  return v;
}


void clear ()
{
  int lp;
  
  for (lp=0; lp<=thds; lp++) {
    buf[lp] = -1;
  }
}


check_parallel (int v)
{
  if (omp_in_parallel () != v) {
    #pragma omp critical
    errors += 1;
  }
  if (v) {
    if (omp_get_num_threads () != thds) {
      #pragma omp critical
      errors += 1;
    }
  } else {
    if (omp_get_num_threads () != 1) {
      #pragma omp critical
      errors += 1;
    }
  }
}


int
check_result ()
{
  int	lp;

  int	err = 0;


  for (lp = 0; lp<thds; lp++) {
    if (buf[lp] != 0) {
      err += 1;
    }
  }
  if (buf[thds] != -1) {
    err += 1;
  }

  return err;
}


main ()
{
  int	lp;

  int	 false = 0;
  double dfalse = 0.0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  buf = (int *) malloc (sizeof (int) * (thds + 1));
  if (buf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  omp_set_dynamic (0);


  clear ();
  #pragma omp parallel for schedule(static,1) if (0)
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (0);
  }
  errors += check_result ();

  clear ();
  #pragma omp parallel for schedule(static,1) if (dfalse)
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (0);
  }
  errors += check_result ();

  clear ();
  #pragma omp parallel for schedule(static,1) if (false == 1)
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (0);
  }
  errors += check_result ();

  clear ();
  #pragma omp parallel for schedule(static,1) if (sameas(false))
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (0);
  }
  errors += check_result ();


  if (errors == 0) {
    printf ("parallel for 012 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel for 012 : FAILED\n");
    return 1;
  }
}
