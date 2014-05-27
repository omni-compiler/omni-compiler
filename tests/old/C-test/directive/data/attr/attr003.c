static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* attribute 003 :
 * parallel region 内のstatic変数は shared になる事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


func ()
{
  static int	shrd = 0;

  #pragma omp critical
  {
    shrd += 1;
  }

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


  #pragma omp parallel
  {
    static int	shrd = 0;

    #pragma omp critical
    {
      shrd += 1;
    }

    #pragma omp barrier
    if (shrd != thds) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel
  func ();


  if (errors == 0) {
    printf ("attribute 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("attribute 003 : FAILED\n");
    return 1;
  }
}
