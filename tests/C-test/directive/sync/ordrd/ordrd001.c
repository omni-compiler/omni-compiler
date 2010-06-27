static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* ordered 001:
 * for 内での orderedの動作確認
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
func_ordered (int i)
{
  #pragma omp ordered
  {

    if (cnt != i) {
      #pragma omp critical
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

    #pragma omp for schedule(static) ordered
    for (i=0;  i<LOOPNUM;  i++) {
      #pragma omp ordered
      {
	if (cnt != i) {
          #pragma omp critical
	  errors ++;
	}
	cnt ++;
      }
    }
  }

  clear ();
  #pragma omp parallel
  {
    int	i;

    #pragma omp for schedule(static) ordered
    for (i=0;  i<LOOPNUM;  i++) {
      func_ordered (i);
    }
  }


  clear ();
  {
    int	i;

    #pragma omp for schedule(static) ordered
    for (i=0;  i<LOOPNUM;  i++) {
      func_ordered (i);
    }
  }


  if (errors == 0) {
    printf ("ordered 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("ordered 001 : FAILED\n");
    return 1;
  }
}
