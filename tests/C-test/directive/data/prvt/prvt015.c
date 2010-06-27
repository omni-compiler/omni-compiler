static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 015 :
 * enumに対してprivateを指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


enum x {
  ZERO = 0,
  ONE,
  TWO,
  THREE
};

enum x	prvt;


void
func1 (enum x *prvt)
{
  int	id = omp_get_thread_num ();

  *prvt = (enum x)id;
  #pragma omp barrier

  if (*prvt != (enum x)id) {
    #pragma omp critical
    errors += 1;
  }
  if (sizeof(*prvt) != sizeof(enum x)) {
    #pragma omp critical
    errors += 1;
  }
}


void
func2 ()
{
  static int	err;
  int		id = omp_get_thread_num ();

  prvt = (enum x)id;
  err  = 0;
  #pragma omp barrier

  if (prvt != (enum x)id) {
    #pragma omp critical
    err += 1;
  }

  #pragma omp barrier
  #pragma omp master
  if (err != thds - 1) {
    #pragma omp critical
    errors ++;
  }
  if (sizeof(prvt) != sizeof(enum x)) {
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


  #pragma omp parallel private(prvt)
  {
    int	id = omp_get_thread_num ();

    prvt = (enum x)id;
    #pragma omp barrier

    if (prvt != (enum x)id) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(prvt) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel private(prvt)
  func1 (&prvt);


  #pragma omp parallel private(prvt)
  func2 ();


  if (errors == 0) {
    printf ("private 015 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 015 : FAILED\n");
    return 1;
  }
}
