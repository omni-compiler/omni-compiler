static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel sections 008:
 * parallel sections if が成り立たない場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	thds, ids[2];
int	errors = 0;


int
sameas(int n)
{
  return n;
}


void
check_parallel ()
{
  if (omp_in_parallel () != 0) {
    #pragma omp critical
    errors += 1;
  }
  if (omp_get_num_threads () != 1) {
    #pragma omp critical
    errors += 1;
  }
}


void
check ()
{
  if (ids[0] != ids[1]) {
    errors += 1;
  }
}


main ()
{
  int	 false = 0;
  double dfalse = 0.0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  #pragma omp parallel sections if (0)
  {
    #pragma omp section
    {	
      check_parallel ();
      ids[0] = omp_get_thread_num ();
      waittime(1);
    }
    #pragma omp section
    {	
      check_parallel ();
      ids[1] = omp_get_thread_num ();
      waittime(1);
    }
  }
  check();

  #pragma omp parallel sections if (false)
  {
    #pragma omp section
    {	
      check_parallel ();
      ids[0] = omp_get_thread_num ();
      waittime(1);
    }
    #pragma omp section
    {	
      check_parallel ();
      ids[1] = omp_get_thread_num ();
      waittime(1);
    }
  }
  check();

  #pragma omp parallel sections if (dfalse)
  {
    #pragma omp section
    {	
      check_parallel ();
      ids[0] = omp_get_thread_num ();
      waittime(1);
    }
    #pragma omp section
    {	
      check_parallel ();
      ids[1] = omp_get_thread_num ();
      waittime(1);
    }
  }
  check();

  #pragma omp parallel sections if (false == 2)
  {
    #pragma omp section
    {	
      check_parallel ();
      ids[0] = omp_get_thread_num ();
      waittime(1);
    }
    #pragma omp section
    {	
      check_parallel ();
      ids[1] = omp_get_thread_num ();
      waittime(1);
    }
  }
  check();

  #pragma omp parallel sections if (sameas(false))
  {
    #pragma omp section
    {	
      check_parallel ();
      ids[0] = omp_get_thread_num ();
      waittime(1);
    }
    #pragma omp section
    {	
      check_parallel ();
      ids[1] = omp_get_thread_num ();
      waittime(1);
    }
  }
  check();


  if (errors == 0) {
    printf ("parallel sections 008 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel sections 008 : FAILED\n");
    return 1;
  }
}
