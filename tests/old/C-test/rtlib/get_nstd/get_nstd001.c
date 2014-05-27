static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_get_nested : 001
 * omp_get_nested で、nested parallelism が default disable になっているか確認
 */

#include <omp.h>
#include "omni.h"


int
main ()
{
  int	n;

  int	errors = 0;


  n = omp_get_nested ();
  if (n != 0) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("omp_get_nested 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_get_nested 001 : FAILED\n");
    return 1;
  }
}
