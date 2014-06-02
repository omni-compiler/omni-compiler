static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel sections 005:
 * check implicit barrier at parallel sections.
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


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);
  omp_set_num_threads (2);

  clear ();
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      int id = omp_get_thread_num ();
      if (id == 0) {
	buf[0] = id;
	barrier (2);
      } else {
	barrier (2);
	waittime(1);
	buf[1] = id;
      }
    }

    #pragma omp section
    {
      int id = omp_get_thread_num ();
      if (id == 0) {
	buf[0] = id;
	barrier (2);
      } else {
	barrier (2);
	waittime(1);
	buf[1] = id;
      }
    }
  }

  if (buf[0] == -1) {
    errors += 1;
  }
  if (buf[1] == -1) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("parallel sections 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel sections 005 : FAILED\n");
    return 1;
  }
}
