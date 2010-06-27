static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 017 :
 * 配列変数に対して firstprivate 宣言した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define MAGICNO		100
#define ARRAYSIZ	1024

int	errors = 0;
int	thds;


int	prvt[ARRAYSIZ];


void
init (int magicno)
{
  int	i;

  for (i=0; i<ARRAYSIZ; i++) {
    prvt[i] = magicno + i;
  }
}


void
func1 (int magicno, int *prvt)
{
  int	id = omp_get_thread_num ();
  int	i;


  for (i=0; i<ARRAYSIZ; i++) {
    if (prvt[i] != magicno + i) {
      #pragma omp critical
      errors += 1;
    }
  }

  for (i=0; i<ARRAYSIZ; i++) {
    prvt[i] = id + i;
  }
  #pragma omp barrier

  for (i=0; i<ARRAYSIZ; i++) {
    if (prvt[i] != id+i) {
      #pragma omp critical
      errors += 1;
    }
  }
}


void
func2 (int magicno)
{
  static int err;
  int	id = omp_get_thread_num ();
  int	i;


  for (i=0; i<ARRAYSIZ; i++) {
    if (prvt[i] != magicno + i) {
      #pragma omp critical
      errors += 1;
    }
  }

  #pragma omp barrier
  prvt[0] = id;
  err  = 0;

  #pragma omp barrier
  if (prvt[0] != id) {
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


  init (MAGICNO);
  #pragma omp parallel firstprivate (prvt)
  {
    int	id = omp_get_thread_num ();
    int	i;

    for (i=0; i<ARRAYSIZ; i++) {
      if (prvt[i] != MAGICNO + i) {
        #pragma omp critical
	errors += 1;
      }
    }

    for (i=0; i<ARRAYSIZ; i++) {
      prvt[i] = id + i;
    }

    #pragma omp barrier
    for (i=0; i<ARRAYSIZ; i++) {
      if (prvt[i] != id + i) {
        #pragma omp critical
	errors += 1;
      }
    }
    if (sizeof(prvt) != sizeof(int) * ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
  }


  init (MAGICNO*2);
  #pragma omp parallel firstprivate (prvt)
  func1 (MAGICNO*2, prvt);


  init (MAGICNO*3);
  #pragma omp parallel firstprivate (prvt)
  func2 (MAGICNO*3);


  if (errors == 0) {
    printf ("firstprivate 017 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 017 : FAILED\n");
    return 1;
  }
}
