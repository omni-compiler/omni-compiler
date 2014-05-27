static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_uset_lock 001:
 * lockされたlock変数に対して、omp_unset_nest_lockが実行できる事を確認
 */

#include <omp.h>
#include "omni.h"


main ()
{
  omp_nest_lock_t	lck;

  int			errors = 0;


  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_unset_nest_lock (&lck);
  omp_unset_nest_lock (&lck);
  omp_destroy_nest_lock (&lck);

  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_unset_nest_lock (&lck);

  omp_set_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_unset_nest_lock (&lck);
  omp_unset_nest_lock (&lck);
  


  if (errors == 0) {
    printf ("omp_unset_nest_lock 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_unset_nest_lock 001 : FAILED\n");
    return 1;
  }
}
