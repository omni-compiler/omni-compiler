static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* copyin 010 :
 * 構造体に対して copyin 宣言をした場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


struct x {
  int		i;
  double	d;
};

struct x	org;
struct x	prvt;
#pragma omp threadprivate(prvt)


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt.i = org.i = -1;
  prvt.d = org.d = 1;
  #pragma omp parallel copyin (prvt)
  {
    if (prvt.i != org.i || prvt.d != org.d || sizeof(prvt) != sizeof(struct x)) {
      #pragma omp critical
      errors += 1;
    }
  }


  prvt.i = org.i = 0;
  prvt.d = org.d = 3;
  #pragma omp parallel copyin (prvt)
  {
    if (prvt.i != org.i || prvt.d != org.d || sizeof(prvt) != sizeof(struct x)) {
      #pragma omp critical
      errors += 1;
    }
  }


  prvt.i = org.i = -2;
  prvt.d = org.d = 2;
  #pragma omp parallel copyin (prvt)
  {
    if (prvt.i != org.i || prvt.d != org.d || sizeof(prvt) != sizeof(struct x)) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("copyin 010 : SUCCESS\n");
    return 0;
  } else {
    printf ("copyin 010 : FAILED\n");
    return 1;
  }
}
