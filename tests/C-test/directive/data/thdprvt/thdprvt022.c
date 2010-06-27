static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 022 :
 * threadprivateにenumを指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

enum x {
  ZERO = 0,
  ONE,
  TWO,
  THREE
};

enum x 	i;
#pragma omp threadprivate (i)


void
func ()
{
  int	id = omp_get_thread_num ();

  i = (enum x)id;
  #pragma omp barrier

  if (i != (enum x)id) {
    #pragma omp critical
    errors += 1;
  }
  if (sizeof (i) != sizeof (enum x)) {
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

    i = (enum x)id;
    #pragma omp barrier

    if (i != (enum x)id) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof (i) != sizeof (enum x)) {
      #pragma omp critical
      errors += 1;
    }
  }

  #pragma omp parallel 
  func ();

  func ();


  if (errors == 0) {
    printf ("threadprivate 022 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 022 : FAILED\n");
    return 1;
  }
}
