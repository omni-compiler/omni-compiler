static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* default 007 :
 * default を prallel for に設定した時の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	MAGICNO	100


int	errors = 0;
int	thds;

int	shrd;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  shrd = 0;
  #pragma omp parallel for default (shared) schedule(static)
  for (i=0; i<thds; i++) {
    #pragma omp critical
    shrd += 1;
  }
  if (shrd != thds) {
    errors += 1;
  }


  shrd = 0;
  #pragma omp parallel for default (none) shared(shrd,thds) schedule(static)
  for (i=0; i<thds; i++) {
    #pragma omp critical
    shrd += 1;
  }
  if (shrd != thds) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("default 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("default 007 : FAILED\n");
    return 1;
  }
}
