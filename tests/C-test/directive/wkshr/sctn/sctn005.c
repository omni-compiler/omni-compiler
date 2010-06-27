static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* sections 005:
 * sections directive の終りにある暗黙のバリアの動作を確認
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
  #pragma omp sections
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

  if (buf[0] == -1) {
    #pragma omp critical
    errors += 1;
  }
  if (buf[1] == -1) {
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
    #pragma omp sections
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

    if (buf[0] == -1) {
      #pragma omp critical
      errors += 1;
    }
    if (buf[1] == -1) {
      #pragma omp critical
      errors += 1;
    }
  }

  memset (buf, -1, sizeof (buf));
  #pragma omp parallel
  {
    func_sections ();
  }


  if (errors == 0) {
    printf ("sections 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("sections 005 : FAILED\n");
    return 1;
  }
}
