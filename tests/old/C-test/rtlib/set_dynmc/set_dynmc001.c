static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_set_dynamic : 001
 * omp_set_dynamicで、dynamic scheduling を 
 * enable/disable した時の動作を確認
 * dynamic scheduling は実装依存なので、スケジューリング自体は確認しない。
 * 確認は、APIのみ。
 */

#include <omp.h>
#include "omni.h"


int
main ()
{
  int	errors = 0;


  omp_set_dynamic (1);
  if(omp_get_dynamic () == 0) {
    printf ("dynamic_threads is not implement.\n");
    goto END;
  }

  omp_set_dynamic (0);
  if(omp_get_dynamic () != 0) {
    errors += 1;
  }

  omp_set_dynamic (1);
  if(omp_get_dynamic () == 0) {
    errors += 1;
  }


 END:
  if (errors == 0) {
    printf ("omp_set_dynamic 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_set_dynamic 001 : FAILED\n");
    return 1;
  }
}
