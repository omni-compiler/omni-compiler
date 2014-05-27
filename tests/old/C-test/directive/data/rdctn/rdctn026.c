static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 026 :
 * parallel if で参照している変数を、reduction 宣言した場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	rdct;


void
check_parallel (int n)
{
  if (n == 1) {
    if (omp_in_parallel() != 0) {
      #pragma omp critical
      errors += 1;
    }
    if (omp_get_num_threads() != 1) {
      #pragma omp critical
      errors += 1;
    }

  } else {
    if (omp_in_parallel() == 0) {
      #pragma omp critical
      errors += 1;
    }
    if (omp_get_num_threads() != n) {
      #pragma omp critical
      errors += 1;
    }
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


  rdct = 0;
  #pragma omp parallel if(rdct) reduction(+:rdct)
  {
    check_parallel (1);
  }

  rdct = 1;
  #pragma omp parallel if(rdct) reduction(+:rdct)
  {
    check_parallel (thds);
  }

  rdct = 2;
  #pragma omp parallel if(rdct) reduction(+:rdct)
  {
    check_parallel (thds);
  }


  if (errors == 0) {
    printf ("reduction 026 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 026 : FAILED\n");
    return 1;
  }
}
