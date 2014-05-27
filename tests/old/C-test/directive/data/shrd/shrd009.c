static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* shared 009 :
 * int型変数に対して、sharedを指示した場合の動作を確認
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
  if (sizeof(*shrd) != sizeof(int)) {
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
  if (sizeof(shrd) != sizeof(int)) {
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
    if (sizeof(shrd) != sizeof(int)) {
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
    printf ("shared 009 : SUCCESS\n");
    return 0;
  } else {
    printf ("shared 009 : FAILED\n");
    return 1;
  }
}
