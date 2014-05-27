static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* barrier 001:
 * barrier の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;
int	flag;


void
func_barrier ()
{
  int id = omp_get_thread_num ();

  if (id == 0) {
    waittime (1);
    flag = 1;
  }

  #pragma omp barrier

  if (flag == 0) {
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


  flag = 0;
  #pragma omp parallel
  {
    int id = omp_get_thread_num ();

    if (id == 0) {
      waittime (1);
      flag = 1;
    }

    #pragma omp barrier

    if (flag == 0) {
      #pragma omp critical
      errors += 1;
    }
  }

  flag = 0;
  #pragma omp parallel
  {
    func_barrier ();
  }

  flag = 0;
  func_barrier ();


  if (errors == 0) {
    printf ("barrier 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("barrier 001 : FAILED\n");
    return 1;
  }
}
