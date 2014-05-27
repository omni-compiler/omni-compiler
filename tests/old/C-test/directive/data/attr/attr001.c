static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* attribute 001 :
 * parallel region 外の変数は defualt で shared になる事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	shrd0;


func (int *shrd1)
{
  #pragma omp critical
  {
    shrd0  += 1;
    *shrd1 += 1;
  }
}


main ()
{
  int	shrd1;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  shrd0 = 0;
  shrd1 = 0;

  #pragma omp parallel
  {
    #pragma omp critical
    {
      shrd0 += 1;
      shrd1 += 1;
    }
  }

  if (shrd0 != thds  ||  shrd1 != thds) {
    errors ++;
  }


  shrd0 = 0;
  shrd1 = 0;

  #pragma omp parallel
  func (&shrd1);

  if (shrd0 != thds  ||  shrd1 != thds) {
    errors ++;
  }

  if (errors == 0) {
    printf ("attribute 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("attribute 001 : FAILED\n");
    return 1;
  }
}
