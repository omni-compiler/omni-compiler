static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 018 :
 * threadprivateにfloat型変数を指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

float	i;
#pragma omp threadprivate (i)


void
func ()
{
  int	id = omp_get_thread_num ();

  i = id;
  #pragma omp barrier

  if (i != id) {
    #pragma omp critical
    errors += 1;
  }
  if (sizeof (i) != sizeof (float)) {
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

  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    i = id;
    #pragma omp barrier

    if (i != id) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof (i) != sizeof (float)) {
      #pragma omp critical
      errors += 1;
    }
  }

  #pragma omp parallel 
  func ();

  func ();


  if (errors == 0) {
    printf ("threadprivate 018 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 018 : FAILED\n");
    return 1;
  }
}
