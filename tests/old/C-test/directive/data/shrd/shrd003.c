static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* shared 003 :
 * global変数に対するshared 宣言の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	shrd;


void
func1 (int *shrd)
{
  #pragma omp critical
  *shrd += 1;
  #pragma omp barrier

  if (*shrd != thds) {
    #pragma omp critical
    errors += 1;
  }
}


void
func2 ()
{
  #pragma omp critical
  shrd += 1;
  #pragma omp barrier

  if (shrd != thds) {
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


  shrd = 0;
  #pragma omp parallel shared(shrd)
  {
    #pragma omp critical
    shrd += 1;

    #pragma omp barrier

    if (shrd != thds) {
      #pragma omp critical
      errors += 1;
    }
  }


  shrd = 0;
  #pragma omp parallel shared(shrd)
  func1 (&shrd);


  shrd = 0;
  #pragma omp parallel shared(shrd)
  func2 ();


  if (errors == 0) {
    printf ("shared 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("shared 003 : FAILED\n");
    return 1;
  }
}
