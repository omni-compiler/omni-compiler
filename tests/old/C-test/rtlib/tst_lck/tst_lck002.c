static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_test_lock 002:
 * スレッド間で、omp_test_lock関数の動作を確認します。
 */

#include <omp.h>
#include "omni.h"


main ()
{
  omp_lock_t	lck;
  int		thds;
  int		s,f;

  int		errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }

  omp_init_lock(&lck);
  s = f = 0;

  #pragma omp parallel
  {
    int	t;

    t = omp_test_lock (&lck);
    if (t != 0) {
      /* lock successful */
      #pragma omp critical
      s += 1;
    } else {
      /* lock fail */
      #pragma omp critical
      f += 1;
    }
  }

  if (s != 1) {
    errors += 1;
  }
  if (f != thds - 1) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("omp_test_lock 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_test_lock 002 : FAILED\n");
    return 1;
  }
}
