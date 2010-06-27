static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* sections 006:
 * nowait を指定した場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	buf[3];


int	errors = 0;


void
clear ()
{
  int	i;

  for (i=0; i<3; i++) {
    buf[i] = -1;
  }
}


void
func_sections()
{
  int	id = omp_get_thread_num ();


  #pragma omp barrier

  #pragma omp sections nowait
  {
    #pragma omp section
    {
      buf[0] = omp_get_thread_num ();
      barrier (2);
    }

    #pragma omp section
    {
      barrier (2);
      waittime(1);
      buf[1] = omp_get_thread_num ();
    }
  }

  if (buf[1] != -1  &&  buf[1] != id) {
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


  clear ();
  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();
    #pragma omp barrier

    #pragma omp sections nowait
    {
      #pragma omp section
      {
	buf[0] = omp_get_thread_num ();
	barrier (2);
      }

      #pragma omp section
      {
	barrier (2);
	waittime(1);
	buf[1] = omp_get_thread_num ();
      }
    }

    if (buf[1] != -1  &&  buf[1] != id) {
      #pragma omp critical
      errors += 1;
    }
  }


  clear ();
  #pragma omp parallel
  {
    func_sections ();
  }


  if (errors == 0) {
    printf ("sections 006 : SUCCESS\n");
    return 0;
  } else {
    printf ("sections 006 : FAILED\n");
    return 1;
  }
}
