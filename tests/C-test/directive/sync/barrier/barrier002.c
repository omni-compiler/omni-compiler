static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* barrier 002:
 * test barrier at nested parallel region.
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;
int	flag;


void
func_barrier (int id)
{
  if (id == 0) {
    #pragma omp barrier
    flag = 1;
    #pragma omp flush
  } else {
    while (flag != 1) {
    #pragma omp flush
      waittime (1);
    }
  }
}


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);
  omp_set_nested (0);


  flag = 0;
  #pragma omp parallel
  {
    int id = omp_get_thread_num ();

    #pragma omp parallel
    {
      /* this nested parallel is serialized. */
      if (id == 0) {
	#pragma omp barrier
	flag = 1;
	#pragma omp flush
      } else {
	while (flag != 1) {
	  #pragma omp flush
	  waittime (1);
	}
      }
    }
  }

  flag = 0;
  #pragma omp parallel
  {
    int id = omp_get_thread_num ();

    #pragma omp parallel
    {
      /* this nested parallel is serialized. */
      func_barrier (id);
    }
  }


  if (errors == 0) {
    printf ("barrier 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("barrier 002 : FAILED\n");
    return 1;
  }
}
