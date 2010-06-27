static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* firstprivate 002 :
 * ローカル変数に対する firstprivate 宣言の動作確認
 */

#include <omp.h>
#include "omni.h"


#define MAGICNO		100

int	errors = 0;
int	thds;



void
func1 (int magicno, int *prvt, int *prvt2)
{
  int	id = omp_get_thread_num ();

  if (*prvt != magicno) {
    #pragma omp critical
    errors += 1;
  }
  if (*prvt2 != magicno+1) {
    #pragma omp critical
    errors += 1;
  }

  *prvt  = id;
  *prvt2 = id+magicno;

  #pragma omp barrier
  if (*prvt != id) {
    #pragma omp critical
    errors += 1;
  }
  #pragma omp barrier
  if (*prvt2 != id+magicno) {
    #pragma omp critical
    errors += 1;
  }
}


main ()
{
  int		prvt;
  static int	prvt2;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  prvt  = MAGICNO;
  prvt2 = MAGICNO + 1;
  #pragma omp parallel firstprivate (prvt) firstprivate(prvt2)
  {
    int	id = omp_get_thread_num ();

    if (prvt != MAGICNO) {
      #pragma omp critical
      errors += 1;
    }
    if (prvt2 != MAGICNO+1) {
      #pragma omp critical
      errors += 1;
    }

    prvt = id;
    prvt2 = id+MAGICNO;

    #pragma omp barrier
    if (prvt != id) {
      #pragma omp critical
      errors += 1;
    }
    #pragma omp barrier
    if (prvt2 != id+MAGICNO) {
      #pragma omp critical
      errors += 1;
    }
  }


  prvt  = MAGICNO*2;
  prvt2 = MAGICNO*2+1;
  #pragma omp parallel firstprivate (prvt) firstprivate(prvt2)
  func1 (MAGICNO*2, &prvt, &prvt2);


  if (errors == 0) {
    printf ("firstprivate 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("firstprivate 002 : FAILED\n");
    return 1;
  }
}
