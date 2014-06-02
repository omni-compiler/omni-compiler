static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_set_lock 001:
 * omp_init_lock関数で初期化されたlock変数に対して、
 * lockが実行できる事を確認。
 */

#include <omp.h>
#include "omni.h"


main ()
{
  omp_lock_t	lck;

  int	errors = 0;


  omp_init_lock (&lck);
  omp_set_lock (&lck);
  omp_unset_lock (&lck);
  omp_destroy_lock (&lck);

  omp_init_lock (&lck);
  omp_set_lock (&lck);
  omp_unset_lock (&lck);
  omp_destroy_lock (&lck);

  omp_init_lock (&lck);
  omp_set_lock (&lck);
  omp_unset_lock (&lck);
  omp_destroy_lock (&lck);

#if defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__)
  printf ("skip some tests. because, Omni on SCASH/SHMEM do not support destroy lock variable that is locked anyone.\n");
#else
  omp_init_lock (&lck);
  omp_set_lock (&lck);
  omp_destroy_lock (&lck);

  omp_init_lock (&lck);	
  omp_set_lock (&lck);

  omp_init_lock (&lck);
  omp_set_lock (&lck);

  omp_unset_lock (&lck);
  omp_set_lock (&lck);
#endif


  if (errors == 0) {
    printf ("omp_set_lock 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_set_lock 001 : FAILED\n");
    return 1;
  }
}
