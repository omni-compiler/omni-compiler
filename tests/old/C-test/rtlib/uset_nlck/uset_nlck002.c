static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_set_nest_lock 002:
 * 既にlockされている変数に対してomp_set_nest_lockを実行すると、
 * omp_unset_nest_lockが実行されるまで、待たされる事を確認する。
 * その状態で、自分自信はnest lockが通る事を確認する。
 */

#include <omp.h>
#include "omni.h"


main ()
{
  omp_nest_lock_t	lck;
  int			thds;
  volatile int		i;

  int		errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }

  omp_init_nest_lock(&lck);
  i = 0;

  #pragma omp parallel
  {
    int	tmp;

    #pragma omp barrier

    omp_set_nest_lock (&lck);

    #pragma omp flush (i)	/* SCASH need flush, here */
    i = i + 1;
    tmp  = i;
    waittime(1);

    #pragma omp flush (i)	/* SCASH need flush, here */
    if (i != tmp) {	
      #pragma omp critical
      errors += 1;
    }
    omp_set_nest_lock (&lck);
    omp_unset_nest_lock (&lck);
    waittime(1);

    #pragma omp flush (i)	/* SCASH need flush, here */
    if (i != tmp) {	
      #pragma omp critical
      errors += 1;
    }

    omp_unset_nest_lock (&lck);
  }

  if (i != thds) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("omp_unset_nest_lock 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_unset_nest_lock 002 : FAILED\n");
    return 1;
  }
}
