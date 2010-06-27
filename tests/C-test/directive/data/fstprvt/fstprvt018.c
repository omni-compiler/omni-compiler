static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 018 :
 * const型の変数をfirstprivateした場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define MAGICNO	100

int	errors = 0;
int	thds;


const int	prvt = MAGICNO;


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
    if (prvt != MAGICNO) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("firstprivate 018 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 018 : FAILED\n");
    return 1;
  }
}
