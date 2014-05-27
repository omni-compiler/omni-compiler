static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 003 :
 * nested parallel region内のマスタースレッド同士で、
 * threadprivate がスレッド毎に独立している事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	i;
#pragma omp threadprivate (i)


void
func (int id, int n)
{
  i = id;
  barrier (n);

  if (i != id) {
    #pragma omp critical
    errors += 1;
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
  omp_set_nested(0);


  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    #pragma omp parallel
    {
      i = id;
      barrier (thds);

      if (i != id) {
	#pragma omp critical
	errors += 1;
      }
    }
  }

  #pragma omp parallel 
  {
    int	id = omp_get_thread_num ();

    #pragma omp parallel
    {
      func (id, thds);
    }
  }

  func(0,1);


  if (errors == 0) {
    printf ("threadprivate 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 003 : FAILED\n");
    return 1;
  }
}
