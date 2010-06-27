static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of private 003 :
 * scope外の変数を private 宣言した場合
 */

#include <omp.h>


int	errors = 0;
int	thds;

main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel private (prvt)
  {
    static int prvt;
    int	id = omp_get_thread_num ();

    prvt = id;
    #pragma omp barrier
    if (prvt != id) {
      errors += 1;
    }
  }

  printf ("err_private 003 : FAILED, can not compile this program.\n");
  return 1;
}
