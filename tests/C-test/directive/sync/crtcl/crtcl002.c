static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* critical 002:
 * 異るcritical間での排他処理の確認
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
  if (4<n) {
    n = 4;
  }

  if (data != n) {
    errors += 1;
  }
}


void
func_critical ()
{
  int	id = omp_get_thread_num ();
  int	i;


  #pragma omp barrier

  switch (id) {
  case 0:
    #pragma omp critical
    {
      i = read_data ();
      waittime (1);
      write_data (i+1);
    }
    break;
  case 1:
    #pragma omp critical
    {
      i = read_data ();
      waittime (1);
      write_data (i+1);
    }
    break;
  case 2:
    #pragma omp critical
    {
      i = read_data ();
      waittime (1);
      write_data (i+1);
    }
    break;
  case 3:
    #pragma omp critical
    {
      i = read_data ();
      waittime (1);
      write_data (i+1);
    }
    break;
  default:
    break;
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
    int	i;

    #pragma omp barrier

    switch (id) {
    case 0:
      #pragma omp critical
      {
	i = read_data ();
	waittime (1);
	write_data (i+1);
      }
      break;
    case 1:
      #pragma omp critical
      {
	i = read_data ();
	waittime (1);
	write_data (i+1);
      }
      break;
    case 2:
      #pragma omp critical
      {
        i = read_data ();
        waittime (1);
        write_data (i+1);
      }
      break;
    case 3:
      #pragma omp critical
      {
        i = read_data ();
        waittime (1);
        write_data (i+1);
      }
      break;
    default:
      break;
    }
  }
  check (thds);

  clear ();
  #pragma omp parallel
  func_critical ();
  check (thds);


  clear ();
  func_critical ();
  check (1);


  if (errors == 0) {
    printf ("critical 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("critical 002 : FAILED\n");
    return 1;
  }
}
