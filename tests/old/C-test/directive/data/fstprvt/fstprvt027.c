static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 027 :
 * firstprivate 宣言した変数をfor loop のカウンタに使用。
 */

#include <omp.h>
#include "omni.h"


#define MAGICNO	100

int	errors = 0;
int	thds;

int	prvt;


main ()
{
  int	sum = 0;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel 
  {

    #pragma omp for firstprivate(prvt) 
    for (prvt=0; prvt<thds; prvt ++) {
      #pragma omp critical
      sum += 1;
    }
  }

  if (sum != thds) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("firstprivate 027 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 027 : FAILED\n");
    return 1;
  }
}
