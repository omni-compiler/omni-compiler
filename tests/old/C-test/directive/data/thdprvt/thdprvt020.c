static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 020 :
 * threadprivateに構造体を指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;

struct x {
  int		i;
  double	d;
};

struct x i;
#pragma omp threadprivate (i)


void
func ()
{
  int	id = omp_get_thread_num ();


  if (i.i != -1) {
    #pragma omp critical
    errors += 1;
  }
  if (i.d != -2) {
    #pragma omp critical
    errors += 1;
  }

  i.i = id;
  #pragma omp barrier

  if (i.i != id) {
    #pragma omp critical
    errors += 1;
  }
  if (sizeof (i) != sizeof (struct x)) {
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

  i.i = -1;
  i.d = -2;
  #pragma omp parallel copyin(i)
  {
    int	id = omp_get_thread_num ();

    if (i.i != -1) {
      #pragma omp critical
      errors += 1;
    }
    if (i.d != -2) {
      #pragma omp critical
      errors += 1;
    }

    i.i = id;
    #pragma omp barrier

    if (i.i != id) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof (i) != sizeof (struct x)) {
      #pragma omp critical
      errors += 1;
    }
  }


  i.i = -1;
  i.d = -2;
  #pragma omp parallel copyin(i)
  func ();

  i.i = -1;
  i.d = -2;
  func ();


  if (errors == 0) {
    printf ("threadprivate 020 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 020 : FAILED\n");
    return 1;
  }
}
