static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 003 :
 * global変数に対する private 宣言の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;
int	prvt;


void
func1 (int *prvt)
{
  int	id = omp_get_thread_num ();

  *prvt = id;
  #pragma omp barrier

  if (*prvt != id) {
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


  #pragma omp parallel private(prvt)
  {
    int	id = omp_get_thread_num ();

    prvt = id;
    #pragma omp barrier

    if (prvt != id) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel private(prvt)
  func1 (&prvt);


  if (errors == 0) {
    printf ("private 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 003 : FAILED\n");
    return 1;
  }
}
