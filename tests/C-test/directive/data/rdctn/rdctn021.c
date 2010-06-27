static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 021 :
 * parallel region に reduction が宣言された場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	rdct;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  #pragma omp parallel reduction(+:rdct)
  {
    rdct += 1;
  }

  if (rdct != thds) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("reduction 021 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 021 : FAILED\n");
    return 1;
  }
}
