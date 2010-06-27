static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 013:
 * nowait付きsingle終了時の暗黙のflushを確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;



main ()
{
  int	b;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);
  omp_set_num_threads (2);


  b = 0;

  #pragma omp parallel 
  {
    #pragma omp single nowait
    {
      waittime (1);
      b = 1;
    }

    while (b != 1) {
      #pragma omp flush
    }

#ifdef __OMNI_SCASH__
    /* Omni on SCASH need this flush directive */
    #pragma omp flush
#endif
  }


  if (errors == 0) {
    printf ("flush 013 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 013 : FAILED\n");
    return 1;
  }
}
