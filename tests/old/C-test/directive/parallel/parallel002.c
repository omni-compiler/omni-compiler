static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel structure 002:
 * parallel if が成立する場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;


int sameas(int v)
{
  return v;
}


void
check_parallel (int v)
{
  int	tn;

  if (omp_in_parallel () != v) {
    #pragma omp critical
    errors += 1;
  }

  if (v == 0) {
    tn = 1;
  } else {
    tn = thds;
  }
  if (omp_get_num_threads () != tn) {
    #pragma omp critical
    errors += 1;
  }
}


main ()
{
  double dtrue = 2.0;
  int	 true = 3;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  check_parallel (0);		      /* not parallel */

  #pragma omp parallel if (1)
  check_parallel (1);		      /* here is parallel */

  check_parallel (0);		      /* not parallel */

  #pragma omp parallel if (dtrue)
  {				      /* this block is parallel */
    check_parallel(1);
  }

  check_parallel (0);		      /* not parallel */

  #pragma omp parallel if (true == 3)
  if (true) {			      /* this if-block is parallel */
    check_parallel (1);
  }

  check_parallel (0);		      /* not parallel */

  #pragma omp parallel if (sameas(4))
  if (true) {			      /* this if-block is parallel */
    check_parallel (1);
  }

  if (errors == 0) {
    printf ("parallel 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel 002 : FAILED\n");
    return 1;
  }
}
