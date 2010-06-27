static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 004 :
 * extern 変数に対する threadprivate のテスト
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

extern int	ext_i;
extern double	ext_d;
#pragma omp threadprivate (ext_i, ext_d)


void
func ()
{
  int	id = omp_get_thread_num ();

  ext_i = id;
  ext_d = id;
  #pragma omp barrier

  if (ext_i != id) {
    #pragma omp critical
    errors += 1;
  }

  if (ext_d != id) {
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
    int	id = omp_get_thread_num ();

    ext_i = id;
    ext_d = id;
    #pragma omp barrier

    if (ext_i != id) {
      #pragma omp critical
      errors += 1;
    }

    if (ext_d != id) {
      #pragma omp critical
      errors += 1;
    }
  }

  #pragma omp parallel 
  func ();

  func ();


  if (errors == 0) {
    printf ("threadprivate 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 004 : FAILED\n");
    return 1;
  }
}
