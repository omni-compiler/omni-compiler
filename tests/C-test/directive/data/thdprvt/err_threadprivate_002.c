static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of threadprivate 002 :
 * ポインタ変数の実体に threadprivate を指定した場合の動作を確認
 */

#include <omp.h>


int	errors = 0;
int	thds;


main ()
{
  int		i;
  static int	*x;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  x = &i;
  #pragma omp threadprivate(*x)


  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    *x = id;
    #pragma omp barrier

    if (*x != id) {
      errors += 1;
    }
  }


  printf ("err_threadprivate 002 : FAILED, can not compile this program.\n");
  return 1;
}
