static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* error case of sections 005:
 * section と sections が別の関数にある場合。
 */

#include <omp.h>


int	thds;
int	buf[3];


void
func_section ()
{
  #pragma omp section
  buf[0] += 1;
  buf[1] += 2;

  #pragma omp section
  buf[2] += 3;
}


main ()
{
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
    func_section();
  }

  printf ("err_sections 005 : FAILED, can not compile this program.\n");
  return 1;
}
