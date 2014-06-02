static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* ordered 005:
 * strideを変更した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#ifdef __OMNI_SCASH__
#define LOOPNUM		(thds*10)
#else
#define LOOPNUM		(thds*100)
#endif



int	errors = 0;
int	thds;
int	cnt;


void
clear ()
{
  cnt = 0;
}


void
func_ordered (int i, int s)
{
  #pragma omp ordered
  {
    if (cnt != i) {
      #pragma omp critical
      errors ++;
    }
    cnt += s;
  }
}


main ()
{
  int	i,stride;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  for (stride=1; stride<10; stride++) {
    clear ();
    #pragma omp parallel for ordered
    for (i=0;  i<LOOPNUM;  i+=stride) {
      func_ordered (i, stride);
    }
  }


  if (errors == 0) {
    printf ("ordered 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("ordered 005 : FAILED\n");
    return 1;
  }
}
