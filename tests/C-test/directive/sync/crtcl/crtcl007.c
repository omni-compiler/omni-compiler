static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* critical 007:
 * ラベル名付きcritical directive指定された処理が、
 * 異るチーム間でも排他処理が行われることを確認
 */

#include <omp.h>
#include "omni.h"


int	thds;
int	errors = 0;

int	data;


void
clear ()
{
  data = 0;
}


int
read_data ()
{
  return data;
}


void
write_data (int d)
{
  data = d;
}


void
check (int n)
{
  if (data != n) {
    errors += 1;
  }
}


void
func_critical ()
{
  #pragma omp critical (label)
  {
    int i;

    i = read_data ();
    waittime (1);
    write_data (i+1);
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
  omp_set_nested(0);


  clear ();
  #pragma omp parallel
  {
    #pragma omp barrier

    #pragma omp parallel
    {
      /* this nested parallel region is serialized. */
      #pragma omp critical (name)
      {
	int i;

	i = read_data ();
	waittime (1);
	write_data (i+1);
      }
    }
  }
  check (thds);

  clear ();
  #pragma omp parallel
  {
    #pragma omp barrier

    #pragma omp parallel
    {
      /* this nested parallel region is serialized. */
      func_critical ();
    }
  }
  check (thds);


  clear ();
  func_critical ();
  check (1);


  if (errors == 0) {
    printf ("critical 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("critical 007 : FAILED\n");
    return 1;
  }
}
