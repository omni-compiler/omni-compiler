static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of copyin 009 :
 * private変数に対して copyin 宣言をした場合の動作確認
 */

#include <omp.h>


int	errors = 0;
int	thds;


int	org;
int	prvt;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt = org = 1;

  #pragma omp parallel private(prvt) copyin (prvt)
  {
    if (prvt != org) {
      errors += 1;
    }
  }


  printf ("err_copyin 009 : FAILED, can not compile this program.\n");
  return 1;
}
