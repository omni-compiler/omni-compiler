static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 025 :
 * firstprivateとlastprivateが同じ変数に対して宣言された時の動作確認
 */

#include <omp.h>
#include "omni.h"


#define MAGICNO		100

int	errors = 0;
int	thds;

int	prvt;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = MAGICNO;
  #pragma omp parallel for firstprivate (prvt) lastprivate (prvt)
  for (i=0; i<thds; i++) {
    prvt += i;
  }
  if (prvt != thds - 1 + MAGICNO) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("firstprivate 025 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 025 : FAILED\n");
    return 1;
  }
}
