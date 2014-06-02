static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of ordered 001:
 * for に ordered 宣言されていない場合
 */

#include <omp.h>


#define LOOPNUM		(thds*100)



int	errors = 0;
int	thds;
int	cnt;


void
clear ()
{
  cnt = 0;
}


void
func_ordered (int i)
{
  #pragma omp ordered
  {

    if (cnt != i) {
      errors ++;
    }
    cnt ++;
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


  clear ();
  #pragma omp parallel
  {
    int	i;

    #pragma omp for schedule(static)
    for (i=0;  i<LOOPNUM;  i++) {
      func_ordered (i);
    }
  }


  printf ("this is wrong program. and, result is unexpected.\n");
  if (errors == 0) {
    printf ("err_ordered 001 : SUCCESS.\n");
    return 0;
  } else {
    printf ("err_ordered 001 : SUCCESS, (errors = %d).\n", errors);
    return 0;
  }
}
