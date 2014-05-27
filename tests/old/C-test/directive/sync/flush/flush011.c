static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 011:
 * nowait付きfor終了時の暗黙のflushを確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	a, b;


main ()
{
  int i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);
  omp_set_num_threads (2);


  #pragma omp parallel 
  {
    #pragma omp for schedule(static) nowait
    for (i=0; i<2; i++){
      if (i == 0) {
	b = 0;
      } else {
	waittime(1);
	b = 1;
      }
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
    printf ("flush 011 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 011 : FAILED\n");
    return 1;
  }
}
