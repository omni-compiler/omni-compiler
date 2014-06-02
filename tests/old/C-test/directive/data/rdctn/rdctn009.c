static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 009 :
 * reduction(||:...) の動作を確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

int	rdct_or, rdct_or2, rdct_or3, rdct_or4;
int	true = 1;
int	false = 0;


void
func_reduction (int loop)
{
  int	i;

  #pragma omp for reduction(||:rdct_or,rdct_or2,rdct_or3,rdct_or4)
  for (i=0; i<loop; i++) {
    rdct_or  = rdct_or || i;
    rdct_or2 = i || rdct_or2;
    rdct_or3 = rdct_or3 || false;
    rdct_or4 = false || rdct_or4;
  }
}


void
check (int init, int loop)
{
  int	rst_or, rst_or2, rst_or3, rst_or4;
  int	i;


  rst_or = rst_or2 = rst_or3 = rst_or4 = init;
  for (i=0;  i<loop;  i++) {
    rst_or  = rst_or || i;
    rst_or2 = i || rst_or2;
    rst_or3 = rst_or3 || false;
    rst_or4 = false || rst_or4;
  }


  if (rst_or != rdct_or) {
    errors += 1;
  }

  if (rst_or2 != rdct_or2) {
    errors += 1;
  }

  if (rst_or3 != rdct_or3) {
    errors += 1;
  }

  if (rst_or4 != rdct_or4) {
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


  rdct_or = rdct_or2 = rdct_or3 = rdct_or4 = 0;
  #pragma omp parallel for reduction(||:rdct_or,rdct_or2,rdct_or3,rdct_or4)
  for (i=0; i<LOOPNUM; i++) {
    rdct_or  = rdct_or || i;
    rdct_or2 = i || rdct_or2;
    rdct_or3 = rdct_or3 || false;
    rdct_or4 = false || rdct_or4;
  }
  check (0, LOOPNUM);

  rdct_or = rdct_or2 = rdct_or3 = rdct_or4 = 1;
  #pragma omp parallel for reduction(||:rdct_or,rdct_or2,rdct_or3,rdct_or4)
  for (i=0; i<LOOPNUM; i++) {
    rdct_or  = rdct_or || i;
    rdct_or2 = i || rdct_or2;
    rdct_or3 = rdct_or3 || false;
    rdct_or4 = false || rdct_or4;
  }
  check (1, LOOPNUM);

  rdct_or = rdct_or2 = rdct_or3 = rdct_or4 = 0;
  #pragma omp parallel
  func_reduction (LOOPNUM);
  check (0, LOOPNUM);

  rdct_or = rdct_or2 = rdct_or3 = rdct_or4 = 1;
  #pragma omp parallel
  func_reduction (LOOPNUM);
  check (1, LOOPNUM);

  rdct_or = rdct_or2 = rdct_or3 = rdct_or4 = 0;
  func_reduction (LOOPNUM);
  check (0, LOOPNUM);

  rdct_or = rdct_or2 = rdct_or3 = rdct_or4 = 1;
  func_reduction (LOOPNUM);
  check (1, LOOPNUM);


  if (errors == 0) {
    printf ("reduction 009 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 009 : FAILED\n");
    return 1;
  }
}
