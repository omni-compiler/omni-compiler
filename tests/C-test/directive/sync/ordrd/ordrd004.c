static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* ordered 004:
 * static,dynamic,guidedスケジューリングにchunkサイズを
 * 指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#if defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__)
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
  int	chunk = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  for (chunk=1; chunk<=10; chunk+=1) {
    clear ();
    #pragma omp parallel
    {
      int	i;

      #pragma omp for schedule(static,chunk) ordered
      for (i=0;  i<LOOPNUM;  i++) {
	func_ordered (i);
      }
    }

    clear ();
    #pragma omp parallel
    {
      int	i;

      #pragma omp for schedule(dynamic,chunk) ordered
      for (i=0;  i<LOOPNUM;  i++) {
	func_ordered (i);
      }
    }


    clear ();
    #pragma omp parallel
    {
      int	i;

      #pragma omp for schedule(guided,chunk) ordered
      for (i=0;  i<LOOPNUM;  i++) {
	func_ordered (i);
      }
    }
  }


  if (errors == 0) {
    printf ("ordered 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("ordered 004 : FAILED\n");
    return 1;
  }
}
