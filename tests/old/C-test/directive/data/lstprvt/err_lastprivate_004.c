static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of lastprivate 004 :
 * firstprivate 宣言した変数を、lastprivate 宣言することは出来ない
 */

#include <omp.h>


#define MAGICNO	100

int	errors = 0;
int	thds;


int	prvt = MAGICNO;


main ()
{
  int	i;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel firstprivate(prvt)
  {
    #pragma omp for lastprivate (prvt)
    for (i=0; i<thds; i++) {
      prvt = i;
    }
  }


  printf ("err_lastprivate 004 : FAILED, can not compile this program.\n");
  return 1;
}
