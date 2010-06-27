static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_destroy_lock 001:
 * omp_destroy_lock の動作確認
 */

#include <omp.h>
#include "omni.h"


main ()
{
  omp_lock_t	lck, lck2, lck3;
  int		i;

  int	errors = 0;


  for (i=0; i<2; i++) {
    omp_init_lock (&lck);
    omp_destroy_lock (&lck);

    omp_init_lock (&lck2);
    omp_set_lock (&lck2);
    omp_unset_lock (&lck2);
    omp_destroy_lock (&lck2);

#ifdef __OMNI_SCASH__
    /*
     * Omni on SCASH is not support this case, yet.
     */
    printf ("test skip ! : Omni on SCASH is not support destroy lock variable that is locked anyone.\n");
#else
    omp_init_lock (&lck3);
    omp_set_lock (&lck3);
    omp_destroy_lock (&lck3);
#endif
  }


  if (errors == 0) {
    printf ("omp_destroy_lock 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_destroy_lock 001 : FAILED\n");
    return 1;
  }
}
