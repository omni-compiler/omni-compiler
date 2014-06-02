static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_set_nest_lock 001:
 * omp_set_nest_lockの動作確認
 */

#include <omp.h>
#include "omni.h"


main ()
{
  omp_nest_lock_t	lck;

  int			errors = 0;

  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_unset_nest_lock (&lck);
  omp_destroy_nest_lock (&lck);

  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_unset_nest_lock (&lck);
  omp_destroy_nest_lock (&lck);

  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_unset_nest_lock (&lck);
  omp_destroy_nest_lock (&lck);

#ifdef __OMNI_SCASH__
  printf ("skip some tests. because, Omni on SCASH do not support destroy lock variable that is locked anyone.\n");
#else
  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_destroy_nest_lock (&lck);

  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_destroy_nest_lock (&lck);

  omp_init_nest_lock (&lck);	
  omp_set_nest_lock (&lck);

  omp_init_nest_lock (&lck);
  omp_set_nest_lock (&lck);
  omp_set_nest_lock (&lck);
#endif


  if (errors == 0) {
    printf ("omp_set_nest_lock 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_set_nest_lock 001 : FAILED\n");
    return 1;
  }
}
