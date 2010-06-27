static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* sections 002:
 * sections directiveのfirst sectionを省略した場合の動作確認
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
    buf[i] = 0;
  }
}


void
check ()
{
  if (buf[0] != 1  ||  buf[1] != 2  ||  buf[2] != 3) {
    errors += 1;
  }
}


void
func_sections()
{
  #pragma omp sections
  {
    buf[0] += 1;

    #pragma omp section
    {
      buf[2] += 3;
    }

    #pragma omp section
    buf[1] += 2;
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
      buf[0] += 1;

      #pragma omp section
      buf[1] += 2;

      #pragma omp section
      {
	buf[2] += 3;
      }
    }
  }
  check ();


  clear ();
  #pragma omp parallel
  {
    func_sections ();
  }
  check ();


  clear ();
  func_sections ();
  check ();


  if (errors == 0) {
    printf ("sections 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("sections 002 : FAILED\n");
    return 1;
  }
}
