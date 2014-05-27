static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* attribute 004 :
 * parallel region 内の heap 領域は shared になる事を確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


void
func ()
{
  static int	*heap;


  #pragma omp single
  {
    heap = (int *) malloc (sizeof(int));
    *heap = 0;
  }

  #pragma omp critical
  *heap += 1;

  #pragma omp barrier
  if (*heap != thds) {
    #pragma omp critical
    errors += 1;
  }
}


int
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
    static int	*heap;

    #pragma omp single
    {
      heap = (int *) malloc (sizeof (int));	
      if (heap == NULL) {
	printf ("can not allocate memory.\n");
      }
      *heap = 0;
    }

    #pragma omp critical
    *heap += 1;

    #pragma omp barrier
    if (*heap != thds) {
      #pragma omp critical
      errors += 1;
    }
  }

  #pragma omp parallel
  func();

  if (errors == 0) {
    printf ("attribute 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("attribute 004 : FAILED\n");
    return 1;
  }
}
