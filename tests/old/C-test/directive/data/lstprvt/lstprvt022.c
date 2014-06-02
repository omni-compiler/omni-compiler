static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* lastprivate 022 :
 * parallel if で参照している変数を、lastprivate 宣言した場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	prvt;


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
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = 0;
  #pragma omp parallel for if(prvt) lastprivate(prvt)
  for (i=0; i<thds; i++) {
    prvt = i;
    check_parallel (1);
  }
  if (prvt != thds-1) {
    errors += 1;
  }

  prvt = 1;
  #pragma omp parallel for if(prvt) lastprivate(prvt)
  for (i=0; i<thds; i++) {
    prvt = i;
    check_parallel (thds);
  }
  if (prvt != thds-1) {
    errors += 1;
  }

  prvt = 2;
  #pragma omp parallel for if(prvt) lastprivate(prvt)
  for (i=0; i<thds; i++) {
    prvt = i;
    check_parallel (thds);
  }
  if (prvt != thds-1) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("lastprivate 022 : SUCCESS\n");
    return 0;
  } else {
    printf ("lastprivate 022 : FAILED\n");
    return 1;
  }
}
