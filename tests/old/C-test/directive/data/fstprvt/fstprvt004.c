static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 004 :
 * 複数の変数が宣言された場合の firstprivate 宣言の動作確認
 */

#include <omp.h>
#include "omni.h"


#define MAGICNO		100

int	errors = 0;
int	thds;

int	prvt1, prvt2, prvt3;


void
func1 (int magicno, int *prvt1, int *prvt2, int *prvt3)
{
  int	id = omp_get_thread_num ();

  if (*prvt1 != magicno+1) {
    #pragma omp critical
    errors += 1;
  }
  if (*prvt2 != magicno+2) {
    #pragma omp critical
    errors += 1;
  }
  if (*prvt3 != magicno+3) {
    #pragma omp critical
    errors += 1;
  }

  *prvt1 = id+1;
  *prvt2 = id+2;
  *prvt3 = id+3;

  #pragma omp barrier
  if (*prvt1 != id+1) {
    #pragma omp critical
    errors += 1;
  }
  if (*prvt2 != id+2) {
    #pragma omp critical
    errors += 1;
  }
  if (*prvt3 != id+3) {
    #pragma omp critical
    errors += 1;
  }
}


void
func2 (int magicno)
{
  static int err;
  int	id = omp_get_thread_num ();


  if (prvt1 != magicno+1) {
    #pragma omp critical
    errors += 1;
  }
  if (prvt2 != magicno+2) {
    #pragma omp critical
    errors += 1;
  }
  if (prvt3 != magicno+3) {
    #pragma omp critical
    errors += 1;
  }

  #pragma omp barrier
  prvt1 = id;
  err  = 0;

  #pragma omp barrier
  if (prvt1 != id) {
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


  prvt1 = MAGICNO+1;
  prvt2 = MAGICNO+2;
  prvt3 = MAGICNO+3;
  #pragma omp parallel firstprivate (prvt1) firstprivate (prvt2,prvt3)
  {
    int	id = omp_get_thread_num ();

    if (prvt1 != MAGICNO+1) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt2 != MAGICNO+2) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt3 != MAGICNO+3) {
      #pragma omp critical
      errors += 1;
    }

    prvt1 = id+1;
    prvt2 = id+2;
    prvt3 = id+3;

    #pragma omp barrier
    if (prvt1 != id+1) {
      #pragma omp critical
      errors += 1;
    }
    #pragma omp barrier
    if (prvt2 != id+2) {
      #pragma omp critical
      errors += 1;
    }
    #pragma omp barrier
    if (prvt3 != id+3) {
      #pragma omp critical
      errors += 1;
    }
  }

  prvt1 = MAGICNO*2+1;
  prvt2 = MAGICNO*2+2;
  prvt3 = MAGICNO*2+3;
  #pragma omp parallel firstprivate (prvt1,prvt2,prvt3)
  func1 (MAGICNO*2, &prvt1, &prvt2, &prvt3);

  prvt1 = MAGICNO*3+1;
  prvt2 = MAGICNO*3+2;
  prvt3 = MAGICNO*3+3;
  #pragma omp parallel firstprivate (prvt1)  firstprivate (prvt2,prvt3)
  func2 (MAGICNO*3);


  if (errors == 0) {
    printf ("firstprivate 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 004 : FAILED\n");
    return 1;
  }
}
