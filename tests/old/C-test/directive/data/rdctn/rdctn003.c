static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 003 :
 * reduction(-:...) の動作を確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

int	rdct_dec, rdct_dec2, rdct_mns, rdct_mns2;


void
func_reduction (int loop)
{
  int	i;

  #pragma omp for reduction(-:rdct_dec,rdct_dec2,rdct_mns,rdct_mns2)
  for (i=0; i<loop; i++) {
    rdct_dec --;
    -- rdct_dec2;
    rdct_mns -= i*2;
    rdct_mns2 = rdct_mns2 - i*3;
  }
}


void
check (int init, int loop)
{
  int	rst_dec, rst_dec2, rst_mns, rst_mns2;
  int	i;


  rst_dec = rst_dec2 = rst_mns = rst_mns2 = init;
  for (i=0;  i<loop;  i++) {
    rst_dec --;
    -- rst_dec2;
    rst_mns -= i*2;
    rst_mns2 = rst_mns2 - i*3;
  }

  if (rst_dec != rdct_dec) {
    errors += 1;
  }

  if (rst_dec2 != rdct_dec2) {
    errors += 1;
  }

  if (rst_mns != rdct_mns) {
    errors += 1;
  }

  if (rst_mns2 != rdct_mns2) {
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


  rdct_dec = rdct_dec2 = rdct_mns = rdct_mns2 = 0;
  #pragma omp parallel
  {
    #pragma omp for reduction(-:rdct_dec,rdct_dec2,rdct_mns,rdct_mns2) 
    for (i=0; i<LOOPNUM; i++) {
      rdct_dec --;
      -- rdct_dec2;
      rdct_mns -= i*2;
      rdct_mns2 = rdct_mns2 - i*3;
    }
  }
  check (0, LOOPNUM);


  rdct_dec = rdct_dec2 = rdct_mns = rdct_mns2 = 100;
  #pragma omp parallel
  func_reduction (LOOPNUM);
  check (100, LOOPNUM);


  rdct_dec = rdct_dec2  = rdct_mns = rdct_mns2 = 200;
  func_reduction (LOOPNUM);
  check (200, LOOPNUM);


  if (errors == 0) {
    printf ("reduction 003 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 003 : FAILED\n");
    return 1;
  }
}
