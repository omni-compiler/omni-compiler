static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 006 :
 * parallel section 間でthreadprivateの値が保証されている事を確認。
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

int	i;
#pragma omp threadprivate (i)


void
func_init ()
{
  i = omp_get_thread_num ();
}


void
func_check ()
{
  if(i != omp_get_thread_num ()) {
    #pragma omp critical
    errors += 1;
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

  #pragma omp parallel
  {
    i = omp_get_thread_num ();
  }
  i = omp_get_thread_num ();
  
  #pragma omp parallel
  {
    if(i != omp_get_thread_num ()) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel 
  func_init ();
  i = omp_get_thread_num ();
  #pragma omp parallel
  func_check ();


  func_init ();
  func_check ();


  if (errors == 0) {
    printf ("threadprivate 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 006 : FAILED\n");
    return 1;
  }
}
