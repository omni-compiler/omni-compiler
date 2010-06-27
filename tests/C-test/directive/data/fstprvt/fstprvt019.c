static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 019 :
 * parallel に firstprivate を設定できることを確認。
 */

#include <omp.h>
#include "omni.h"


#define MAGICNO	100


int	errors = 0;
int	thds;

int	prvt = MAGICNO;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel firstprivate(prvt)
  {
    int	id = omp_get_thread_num ();

    if (prvt != MAGICNO) {
      #pragma omp critical
      errors += 1;
    }

    prvt = id;

    #pragma omp barrier
    if (prvt != id) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("firstprivate 019 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 019 : FAILED\n");
    return 1;
  }
}
