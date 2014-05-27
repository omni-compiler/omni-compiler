static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel for 011:
 * parallel for if が成立する場合の動作確認
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
check_result (int v)
{
  int	lp;

  int	err = 0;


  for (lp = 0; lp<thds; lp++) {
    if (buf[lp] != lp) {
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

  int	 true = 3;
  double dtrue = 4.0;


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
  #pragma omp parallel for schedule(static,1) if (1)
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (1);
  }
  errors += check_result (thds);

  clear ();
  #pragma omp parallel for schedule(static,1) if (dtrue)
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (1);
  }
  errors += check_result (thds);

  clear ();
  #pragma omp parallel for schedule(static,1) if (true == 3)
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (1);
  }
  errors += check_result (thds);

  clear ();
  #pragma omp parallel for schedule(static,1) if (sameas(4))
  for (lp=0; lp<thds; lp++) {
    buf[lp] = omp_get_thread_num ();
    check_parallel (1);
  }
  errors += check_result (thds);


  if (errors == 0) {
    printf ("parallel for 011 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel for 011 : FAILED\n");
    return 1;
  }
}
