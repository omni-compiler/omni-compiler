static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 013 :
 * 構造体に対して firstprivate 宣言した場合の動作確認
 */

#include <omp.h>
#include "omni.h"

#define MAGICNO		100

int	errors = 0;
int	thds;


struct x {
  int		i;
  double	d;
};

struct x	prvt;


void
func1 (int magicno, struct x *prvt)
{
  int	id = omp_get_thread_num ();


  if (prvt->i != magicno) {
    #pragma omp critical
    errors += 1;
  }
  if (prvt->d != magicno+1) {
    #pragma omp critical
    errors += 1;
  }

  prvt->i = id;
  prvt->d = id-1;
  #pragma omp barrier

  if (prvt->i != id) {
    #pragma omp critical
    errors += 1;
  }
  if (prvt->d != id-1) {
    #pragma omp critical
    errors += 1;
  }
  if (sizeof(*prvt) != sizeof(struct x)) {
    #pragma omp critical
    errors += 1;
  }
}


void
func2 (int magicno)
{
  static int err;
  int	id = omp_get_thread_num ();


  if (prvt.i != magicno) {
    #pragma omp critical
    errors += 1;
  }
  if (prvt.d != magicno+1) {
    #pragma omp critical
    errors += 1;
  }

  #pragma omp barrier
  prvt.i = id;
  err  = 0;

  #pragma omp barrier
  if (prvt.i != id) {
    #pragma omp critical
    err += 1;
  }
  #pragma omp barrier
  #pragma omp master
  if (err != thds - 1) {
    #pragma omp critical
    errors ++;
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


  prvt.i = MAGICNO;
  prvt.d = MAGICNO+1;
  #pragma omp parallel firstprivate (prvt)
  {
    int	id = omp_get_thread_num ();

    if (prvt.i != MAGICNO) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt.d != MAGICNO+1) {
      #pragma omp critical
      errors += 1;
    }

    prvt.i = id;
    prvt.d = id-1;

    #pragma omp barrier
    if (prvt.i != id) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt.d != id-1) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(prvt) != sizeof(struct x)) {
      #pragma omp critical
      errors += 1;
    }
  }


  prvt.i = MAGICNO*2;
  prvt.d = MAGICNO*2+1;
  #pragma omp parallel firstprivate (prvt)
  func1 (MAGICNO*2, &prvt);


  prvt.i = MAGICNO*3;
  prvt.d = MAGICNO*3+1;
  #pragma omp parallel firstprivate (prvt)
  func2 (MAGICNO*3);


  if (errors == 0) {
    printf ("firstprivate 013 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 013 : FAILED\n");
    return 1;
  }
}
