static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* critical 004:
 * ラベルの異る critical 間では、排他処理されない事を確認。
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;
int	flag;


void
func_critical ()
{
  int	id = omp_get_thread_num ();

  switch(id) {
  case 0:
    #pragma omp critical (label1)
    {
      flag = 1;
      while (flag != 2) {
        #pragma omp flush
      }
    }
    break;
  case 1:
    #pragma omp critical (label2)
    {
      while (flag != 1) {
        #pragma omp flush
      }
      flag = 2;
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


  flag = 0;
  #pragma omp parallel
  {
    int	id = omp_get_thread_num ();

    switch(id) {
    case 0:
      #pragma omp critical (label1)
      {
	flag = 1;
	while (flag != 2) {
          #pragma omp flush
	}
      }
      break;
    case 1:
      #pragma omp critical (label2)
      {
	while (flag != 1) {
	  #pragma omp flush
	}
	flag = 2;
      }
      break;
    default:
      break;
    }
  }

  if (flag != 2) {
    errors += 1;
  }


  flag = 0;
  #pragma omp parallel
  {
    func_critical ();
  }
  if (flag != 2) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("critical 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("critical 004 : FAILED\n");
    return 1;
  }
}
