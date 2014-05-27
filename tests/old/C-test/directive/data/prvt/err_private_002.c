static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of private 002 :
 * private 宣言した変数を、再度、private 宣言することは出来ない
 */

#include <omp.h>


int	errors = 0;
int	thds;


int	prvt;


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
    int	i;

    #pragma omp for private(prvt)
    for (i=0; i<thds; i++) {
      prvt = omp_get_thread_num ();
    }
  }

  printf ("err_private 002 : FAILED, can not compile this program.\n");
  return 1;
}
