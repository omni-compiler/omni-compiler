static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* single 003:
 * nowait を指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	thds, flag;

int	errors = 0;


void
func_single ()
{
  #pragma omp barrier

  #pragma omp single nowait
  {
    int thds = omp_get_num_threads ();
    do {
      #pragma omp flush (flag)
    } while (flag != thds-1);
  }

  #pragma omp critical
  {
    flag += 1;
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
    #pragma omp barrier

    #pragma omp single nowait
    {
      int thds = omp_get_num_threads ();
      do {
        #pragma omp flush (flag)
      } while (flag != thds-1);
    }

    #pragma omp critical
    {
      flag += 1;
    }
  }


  flag = 0;
  #pragma omp parallel
  {
    func_single ();
  }


  if (errors == 0) {
    printf ("single 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("single 003 : FAILED\n");
    return 1;
  }
}
