static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* default 006 :
 * default を prallel region に設定できる事を確認
 */

#include <omp.h>
#include "omni.h"


#define	MAGICNO	100


int	errors = 0;
int	thds;

int	shrd;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  shrd = 0;
  #pragma omp parallel default (shared)
  {
    #pragma omp critical
    shrd += 1;
  }
  if (shrd != thds) {
    errors += 1;
  }


  shrd = 0;
  #pragma omp parallel default (none) shared(shrd)
  {
    #pragma omp critical
    shrd += 1;
  }
  if (shrd != thds) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("default 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("default 006 : FAILED\n");
    return 1;
  }
}
