static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* flush 008:
 * for終了時の暗黙のflushを確認
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
    #pragma omp for schedule(static,1)
    for (i=0; i<2; i++){
      if (i == 0) {
	b = 0;
	waittime(1);
	a = 1;
      } else {
	a = 0;
	waittime(1);
	b = 1;
      }
    }

    if (a != 1  ||  b != 1) {
      errors ++;
    }
  }


  if (errors == 0) {
    printf ("flush 008 : SUCCESS\n");
    return 0;
  } else {
    printf ("flush 008 : FAILED\n");
    return 1;
  }
}
