static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 017 :
 * 配列変数に対してprivateを指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	ARRAYSIZ	1024

int	errors = 0;
int	thds;


int	prvt[ARRAYSIZ];


void
func1 (int *prvt)
{
  int	id = omp_get_thread_num ();
  int	i;


  for (i=0; i<ARRAYSIZ; i++) {
    prvt[i] = id+i;
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
func2 ()
{
  static int	err;
  int		id = omp_get_thread_num ();
  int		i;


  for (i=0; i<ARRAYSIZ; i++) {
    prvt[i] = id+i;
  }
  err  = 0;
  #pragma omp barrier

  for (i=0; i<ARRAYSIZ; i++) {
    if (prvt[i] != id+i) {
      #pragma omp critical
      err += 1;
    }
  }
  #pragma omp barrier
  #pragma omp master
  if (err != (thds - 1) * ARRAYSIZ) {
    #pragma omp critical
    errors ++;
  }
  if (sizeof(prvt) != sizeof(int) * ARRAYSIZ) {
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
    int	i;

    for (i=0; i<ARRAYSIZ; i++) {
      prvt[i] = id+i;
    }
    #pragma omp barrier

    
    for (i=0; i<ARRAYSIZ; i++) {
      if (prvt[i] != id+i) {
	#pragma omp critical
	errors += 1;
      }
    }
    if (sizeof(prvt) != sizeof(int) * ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel private(prvt)
  func1 (prvt);


  #pragma omp parallel private(prvt)
  func2 ();


  if (errors == 0) {
    printf ("private 017 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 017 : FAILED\n");
    return 1;
  }
}
