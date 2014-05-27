static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* for 008:
 * check implicit barrier at the end of omp for loop
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	*buf;


void
clear ()
{
  int lp;
  
  for (lp=0; lp<thds; lp++) {
    buf[lp] = -1;
  }
}


main ()
{
  int	lp, finish, id;

  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  buf = (int *) malloc (sizeof (int) * thds);
  if (buf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  omp_set_dynamic (0);

  finish = 0;
  clear ();
  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    #pragma omp for schedule (static,1)
    for (lp=0; lp<thds; lp++) {
      if (omp_get_thread_num () == 0) {
	finish = 1;
      } else {
	while (finish == 0) {
	  #pragma omp flush
	}
	waittime (1);
      }
      buf[id] = omp_get_thread_num ();
      #pragma omp flush
    }
    if (buf[(id+1)%thds] == -1) {
      #pragma omp critical
      {
	ERROR (errors);
      }
    }
  }


  finish = 0;
  clear ();
  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    #pragma omp for schedule (dynamic,1)
    for (lp=0; lp<thds; lp++) {
      barrier (thds);
      if (omp_get_thread_num () != 0) {
	waittime (1);
      }
      buf[id] = omp_get_thread_num ();
      #pragma omp flush
    }
    if (buf[(id+1)%thds] == -1) {
      #pragma omp critical
      {
	ERROR (errors);
      }
    }
  }


  id = -1;
  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided,1)
    for (lp=0; lp<thds*4; lp++) {
      if (lp == 0) {
	waittime (1);
	id = omp_get_thread_num ();
      }
    }
    if (id == -1) {
      #pragma omp critical
      {
	ERROR (errors);
      }
    }
  }


  if (errors == 0) {
    printf ("for 008 : SUCCESS\n");
    return 0;
  } else {
    printf ("for 008 : FAILED\n");
    return 1;
  }
}
