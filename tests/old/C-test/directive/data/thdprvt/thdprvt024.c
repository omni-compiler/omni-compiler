static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* threadprivate 024 :
 * threadprivateに配列変数を指定した場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define ARRAYSIZ	1024


int	errors = 0;
int	thds;

int	i[ARRAYSIZ];
#pragma omp threadprivate (i)


void
clear ()
{
  int	j;

  for (j=0; j<ARRAYSIZ; j++) {
    i[j] = j + ARRAYSIZ;
  }
}

void
func ()
{
  int	j;
  int	id = omp_get_thread_num ();

  for (j=0; j<ARRAYSIZ; j++) {
    if (i[j] != j + ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
  }
  for (j=0; j<ARRAYSIZ; j++) {
    i[j] = id + j;
  }
  #pragma omp barrier
  
  for (j=0; j<ARRAYSIZ; j++) {
    if (i[j] != id + j) {
      #pragma omp critical
      errors += 1;
    }
  }
  if (sizeof(i) != sizeof(int) * ARRAYSIZ) {
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

  clear ();
  #pragma omp parallel copyin(i)
  {
    int j;
    int	id = omp_get_thread_num ();

    for (j=0; j<ARRAYSIZ; j++) {
      if (i[j] != j + ARRAYSIZ) {
        #pragma omp critical
	errors += 1;
      }
    }
    for (j=0; j<ARRAYSIZ; j++) {
      i[j] = id + j;
    }
    #pragma omp barrier
  
    for (j=0; j<ARRAYSIZ; j++) {
      if (i[j] != id + j) {
        #pragma omp critical
	errors += 1;
      }
    }
    if (sizeof(i) != sizeof(int) * ARRAYSIZ) {
      #pragma omp critical
      errors += 1;
    }
  }

  clear ();
  #pragma omp parallel copyin(i)
  func ();

  clear ();
  func ();


  if (errors == 0) {
    printf ("threadprivate 024 : SUCCESS\n");
    return 0;
  } else {
    printf ("threadprivate 024 : FAILED\n");
    return 1;
  }
}
