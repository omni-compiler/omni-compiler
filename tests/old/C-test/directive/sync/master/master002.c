static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* master 001:
 * master directive の前後に barrier 同期が存在しないことを確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


main ()
{
  int	tflag = 0, lflag = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    #pragma omp barrier

    if (id == 0) {
      waittime (1);
      tflag = 1;
    }
    #pragma omp master
    {
      lflag = 1;
    }

    #pragma omp flush
    if (id != 0) {
      if(tflag != 0) {
	#pragma omp critical
	errors += 1;
      }
      if(lflag != 0) {
	#pragma omp critical
	errors += 1;
      }
    }
  }


  if (errors == 0) {
    printf ("master for 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("master for 002 : FAILED\n");
    return 1;
  }
}
