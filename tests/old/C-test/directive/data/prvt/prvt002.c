static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 002 :
 * ローカル変数に対する private 宣言の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


void
func1 (int *prvt, int *prvt2)
{
  int	id = omp_get_thread_num ();

  *prvt = id;
  *prvt2 = id;
  #pragma omp barrier

  if (*prvt != id) {
    #pragma omp critical
    errors += 1;
  }
  if (*prvt2 != id) {
    #pragma omp critical
    errors += 1;
  }
}


main ()
{
  int		prvt;
  static int	prvt2;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel private(prvt,prvt2)
  {
    int	id = omp_get_thread_num ();

    prvt = id;
    prvt2 = id;
    #pragma omp barrier

    if (prvt != id) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt2 != id) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel private(prvt,prvt2)
  func1 (&prvt,&prvt2);


  if (errors == 0) {
    printf ("private 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 002 : FAILED\n");
    return 1;
  }
}
