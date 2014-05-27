static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 010 :
 * threadprivateされた変数がparallel ifに使用できる事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	i;
#pragma omp threadprivate (i)


main ()
{
  int	sum;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  i = 2;
  sum = 0;
  #pragma omp parallel if(i)
  {
    #pragma omp critical
    sum += 1;
  }
  if (sum != thds) {
    errors += 1;
  }

  i = 1;
  sum = 0;
  #pragma omp parallel if(i)
  {
    #pragma omp critical
    sum += 1;
  }
  if (sum != thds) {
    errors += 1;
  }

  i = 0;
  sum = 0;
  #pragma omp parallel if(i)
  {
    #pragma omp critical
    sum += 1;
  }
  if (sum != 1) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("threadprivate 010 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 010 : FAILED\n");
    return 1;
  }
}
