static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 008 :
 * reduction(&&:...) の動作を確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

int	rdct_and, rdct_and2, rdct_and3, rdct_and4;
int	false = 0;


void
func_reduction (int loop)
{
  int	i;

  #pragma omp for reduction(&&:rdct_and,rdct_and2,rdct_and3,rdct_and4)
  for (i=1; i<loop; i++) {
    rdct_and  = rdct_and && i;
    rdct_and2 = i && rdct_and2;
    rdct_and3 = rdct_and3 && false;
    rdct_and4 = false && rdct_and4;
  }
}


void
check (int init, int loop)
{
  int	rst_and, rst_and2, rst_and3, rst_and4;
  int	i;


  rst_and = rst_and2 = init;
  for (i=1;  i<loop;  i++) {
    rst_and  = rst_and && i;
    rst_and2 = i && rst_and2;
    rst_and3 = rst_and3 && false;
    rst_and4 = false && rst_and4;
  }

  if (rst_and != rdct_and) {
    errors += 1;
  }

  if (rst_and2 != rdct_and2) {
    errors += 1;
  }

  if (rst_and3 != rdct_and3) {
    errors += 1;
  }

  if (rst_and4 != rdct_and4) {
    errors += 1;
  }
}


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  rdct_and = rdct_and2 = 0;
  #pragma omp parallel
  {
    #pragma omp for reduction(&&:rdct_and,rdct_and2,rdct_and3,rdct_and4)
    for (i=1; i<LOOPNUM; i++) {
      rdct_and  = rdct_and && i;
      rdct_and2 = i && rdct_and2;
      rdct_and3 = rdct_and3 && false;
      rdct_and4 = false && rdct_and4;
    }
  }
  check (0, LOOPNUM);

  rdct_and = rdct_and2 = 1;
  #pragma omp parallel
  {
    #pragma omp for reduction(&&:rdct_and,rdct_and2,rdct_and3,rdct_and4)
    for (i=1; i<LOOPNUM; i++) {
      rdct_and  = rdct_and && i;
      rdct_and2 = i && rdct_and2;
      rdct_and3 = rdct_and3 && false;
      rdct_and4 = false && rdct_and4;
    }
  }
  check (1, LOOPNUM);

  rdct_and = rdct_and2 = 0;
  #pragma omp parallel
  func_reduction (LOOPNUM);
  check (0, LOOPNUM);

  rdct_and = rdct_and2 = 1;
  #pragma omp parallel
  func_reduction (LOOPNUM);
  check (1, LOOPNUM);

  rdct_and = rdct_and2 = 0;
  func_reduction (LOOPNUM);
  check (0, LOOPNUM);

  rdct_and = rdct_and2 = 1;
  func_reduction (LOOPNUM);
  check (1, LOOPNUM);


  if (errors == 0) {
    printf ("reduction 008 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 008 : FAILED\n");
    return 1;
  }
}
