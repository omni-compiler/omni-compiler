static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* attribute 002 :
 * parallel region 内の変数は defualt で private になる事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


func (int *prvt)
{
  *prvt = omp_get_thread_num ();
  #pragma omp barrier

  if (*prvt != omp_get_thread_num ()) {
    #pragma omp critical
    errors ++;
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
    int	prvt = 0;

    prvt = omp_get_thread_num ();
    #pragma omp barrier

    if (prvt != omp_get_thread_num ()) {
      #pragma omp critical
      errors ++;
    }
  }

  #pragma omp parallel
  {
    int prvt;

    func (&prvt);
  }


  if (errors == 0) {
    printf ("attribute 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("attribute 002 : FAILED\n");
    return 1;
  }
}
