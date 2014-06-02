static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* master 001:
 * master directive の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;


void
func_master ()
{
  #pragma omp master
  if (omp_get_thread_num () != 0) {
    #pragma omp critical
    errors += 1;
  }
}


main ()
{
  int thds, i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel
  {
    #pragma omp master
    {
      if (omp_get_thread_num () != 0) {
        #pragma omp critical
	errors += 1;
      }
    }

    #pragma omp master
    if (omp_get_thread_num () != 0) {
      #pragma omp critical
      errors += 1;
    }

    #pragma omp master
    for (i=0; i<thds;  i++) {
      if (omp_get_thread_num () != 0) {
        #pragma omp critical
	errors += 1;
      }
    }

    #pragma omp master
    ;

    func_master ();
  }

  func_master ();


  if (errors == 0) {
    printf ("master for 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("master for 001 : FAILED\n");
    return 1;
  }
}
