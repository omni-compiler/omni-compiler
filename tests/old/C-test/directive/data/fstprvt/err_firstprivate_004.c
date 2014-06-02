static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of firstprivate 004 :
 * threadprivate 宣言された変数が firstprivate 宣言された場合の動作確認
 */

#include <omp.h>


#define MAGICNO		100


int	errors = 0;
int	thds;


int	prvt = MAGICNO;
#pragma omp threadprivate (prvt)


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel firstprivate (prvt)
  {
    int	id = omp_get_thread_num ();

    prvt += id;
    #pragma omp barrier

    if (prvt != MAGICNO + id) {
      errors += 1;
    }
  }


  printf ("err_firstprivate 004 : FAILED, can not compile this program.\n");
  return 1;
}
