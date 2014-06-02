static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 004 :
 * 複数の変数が宣言された場合のprivate 宣言の動作確認
 */

#include <omp.h>
#include "omni.h"


int	errors = 0;
int	thds;


int	prvt1, prvt2, prvt3;


void
func1 (int *prvt1, int *prvt2, int *prvt3)
{
  int	id = omp_get_thread_num ();

  *prvt1 = id;
  *prvt2 = id;
  *prvt3 = id;
  #pragma omp barrier

  if (*prvt1 != id  ||  *prvt2 != id  ||  *prvt3 != id) {
    #pragma omp critical
    errors += 1;
  }
}


void
func2 ()
{
  static int	err;
  int		id = omp_get_thread_num ();

  prvt1 = id;
  prvt2 = id;
  prvt3 = id;
  err  = 0;
  #pragma omp barrier

  if (prvt1 != id) {
    #pragma omp critical
    err += 1;
  }
  if (prvt2 != id) {
    #pragma omp critical
    err += 1;
  }
  if (prvt3 != id) {
    #pragma omp critical
    err += 1;
  }
  #pragma omp barrier
  #pragma omp master
  if (err != (thds - 1)*3) {
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


  #pragma omp parallel private(prvt1) private(prvt2, prvt3) 
  {
    int	id = omp_get_thread_num ();

    prvt1 = id;
    prvt2 = id;
    prvt3 = id;
    #pragma omp barrier

    if (prvt1 != id  ||  prvt2 != id  ||  prvt3 != id) {
      #pragma omp critical
      errors += 1;
    }
  }


  #pragma omp parallel private(prvt1) private(prvt2, prvt3)
  func1 (&prvt1, &prvt2, &prvt3);


  #pragma omp parallel private(prvt1) private(prvt2, prvt3)
  func2 ();


  if (errors == 0) {
    printf ("private 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 004 : FAILED\n");
  }
}
