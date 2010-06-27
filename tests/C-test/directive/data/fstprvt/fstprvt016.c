static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 016 :
 * ポインタ変数に対して firstprivate 宣言した場合の動作確認
 */

#include <omp.h>
#include "omni.h"

#define MAGICNO		100

int	errors = 0;
int	thds;


void *	prvt;


void
func1 (int magicno, void * *prvt)
{
  int	id = omp_get_thread_num ();


  if (*prvt != (void *)magicno) {
    #pragma omp critical
    errors += 1;
  }

  *prvt = (void *)id;
  #pragma omp barrier

  if (*prvt != (void *)id) {
    #pragma omp critical
    errors += 1;
  }
  if (sizeof(*prvt) != sizeof(void *)) {
    #pragma omp critical
    errors += 1;
  }
}


void
func2 (int magicno)
{
  static int err;
  int	id = omp_get_thread_num ();


  if (prvt != (void *)magicno) {
    #pragma omp critical
    errors += 1;
  }

  #pragma omp barrier
  prvt = (void *)id;
  err  = 0;

  #pragma omp barrier
  if (prvt != (void *)id) {
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


  prvt = (void *)MAGICNO;
  #pragma omp parallel firstprivate (prvt)
  {
    int	id = omp_get_thread_num ();

    if (prvt != (void *)MAGICNO) {
      #pragma omp critical
      errors += 1;
    }

    prvt = (void *)id;

    #pragma omp barrier
    if (prvt != (void *)id) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(prvt) != sizeof(void *)) {
      #pragma omp critical
      errors += 1;
    }
  }


  prvt = (void *)(MAGICNO*2);
  #pragma omp parallel firstprivate (prvt)
  func1 (MAGICNO*2, &prvt);


  prvt = (void *)(MAGICNO*3);
  #pragma omp parallel firstprivate (prvt)
  func2 (MAGICNO*3);


  if (errors == 0) {
    printf ("firstprivate 016 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 016 : FAILED\n");
    return 1;
  }
}
