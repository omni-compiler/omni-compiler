static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* shared 022 :
 * const型変数に対して、sharedを指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	MAGICNO	100


int	errors = 0;
int	thds;


const int	shrd = MAGICNO;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel shared (shrd)
  {
    if (shrd != MAGICNO) {
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("shared 022 : SUCCESS\n");
    return 0;
  } else {
    printf ("shared 022 : FAILED\n");
    return 1;
  }
}
