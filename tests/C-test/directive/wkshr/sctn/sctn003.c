static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* sections 003:
 * sections directive に section が現われないケース
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	buf[3];


void
func_sections()
{
  #pragma omp sections
  {
    {
      buf[0] += 1;
      buf[1] += 2; 
      buf[2] += 3;
    }
  }
}


main ()
{

  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  memset (buf, 0, sizeof (buf));
  #pragma omp parallel
  {
    #pragma omp sections
    {
      {
	buf[0] += 1;
	buf[1] += 2;
	buf[2] += 3;
      }
    }
  }

  if (buf[0] != 1  ||  buf[1] != 2  ||  buf[2] != 3) {
    errors += 1;
  }

  memset (buf, 0, sizeof (buf));
  #pragma omp parallel
  {
    func_sections ();
  }

  if (buf[0] != 1  ||  buf[1] != 2  ||  buf[2] != 3) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("sections 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("sections 003 : FAILED\n");
    return 1;
  }
}
