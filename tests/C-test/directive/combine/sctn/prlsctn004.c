static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* parallel sections 004:
 * parallel sections が並列で実行されていることを確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	buf[2];

int	errors = 0;


void
clear()
{
  int	i;

  for (i=0; i<2; i++) {
    buf[i] = -1;
  }
}


void
check ()
{
  int	i;

  if (buf[0] == buf[1]) {
    errors += 1;
  }
  for (i=0; i<2; i++) {
    if (buf[i] == -1) {
      errors += 1;
    }
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
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      buf[0] = omp_get_thread_num ();
      barrier (2);
    }

    #pragma omp section
    {
      buf[1] = omp_get_thread_num ();
      barrier (2);
    }
  }
  check ();


  if (errors == 0) {
    printf ("parallel sections 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("parallel sections 004 : FAILED\n");
    return 1;
  }
}
