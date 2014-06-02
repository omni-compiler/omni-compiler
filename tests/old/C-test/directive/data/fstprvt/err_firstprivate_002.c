static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of firstprivate 002 :
 * firstprivate 宣言した変数を、再度 firstprivate 宣言することは出来ない
 */

#include <omp.h>


#define	MAGICNO		100

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


  prvt = MAGICNO
  #pragma omp parallel firstprivate(prvt)
  {
    int	id = omp_get_thread_num ();
    int	i;

    #pragma omp for firstprivate(prvt)
    for (i=0; i<thds; i++) {
      prvt += id;
      sleep (1);
      if (prvt != MAGICNO + id) {
	errors += 1;
      }
    }
  }
  printf ("err_firstprivate 002 : FAILED, can not compile this program.\n");
  return 1;
}
