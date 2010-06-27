static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of lastprivate 009 :
 * critical には lastprivate が宣言できない事を確認
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


  #pragma omp parallel
  {
    #pragma omp critical lastprivate (prvt)
    {
      prvt = omp_get_thread_num ();
    }
  }


  printf ("err_lastprivate 009 : FAILED, can not compile this program.\n");
  return 1;
}
