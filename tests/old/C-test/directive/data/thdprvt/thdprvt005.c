static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 005 :
 * master section 内で、threadprivate変数を使用した場合、
 * master thread と同じ物が使用される事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	i;
#pragma omp threadprivate (i)


void
func ()
{
  int	id = omp_get_thread_num ();

  i = id;
  #pragma omp barrier

  #pragma omp master
  {
    if (i != id) {
      errors += 1;
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

  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    i = id;
    #pragma omp barrier

    #pragma omp master
    if (i != id) {
      errors += 1;
    }
  }

  #pragma omp parallel 
  func ();

  func ();


  if (errors == 0) {
    printf ("threadprivate 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 005 : FAILED\n");
    return 1;
  }
}
