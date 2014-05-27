static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_test_lock 001:
 * omp_test_lock関数の動作を確認します。
 */

#include <omp.h>
#include "omni.h"


main ()
{
  omp_lock_t	lck;
  int		t;

  int	errors = 0;


  omp_init_lock (&lck);
  t = omp_test_lock (&lck);
  if (t == 0) {
    errors += 1;
  }
  t = omp_test_lock (&lck);
  if (t != 0) {
    errors += 1;
  }
  omp_unset_lock (&lck);
  omp_destroy_lock (&lck);

  omp_init_lock (&lck);
  omp_set_lock (&lck);
  t = omp_test_lock (&lck);
  if (t != 0) {
    errors += 1;
  }
  omp_unset_lock (&lck);
  omp_destroy_lock (&lck);


  if (errors == 0) {
    printf ("omp_test_lock 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_test_lock 001 : FAILED\n");
    return 1;
  }
}
