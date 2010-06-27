static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 007 :
 * reduction(^:...) の動作を確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

int	rdct_xor, rdct_xor2, rdct_xor3;


void
func_reduction (int loop)
{
  int	i;

  #pragma omp for reduction(^:rdct_xor,rdct_xor2,rdct_xor3)
  for (i=0; i<loop; i++) {
    rdct_xor ^= (1<<i);
    rdct_xor2 = rdct_xor2 ^ ((1<<(i*2))| 2);
    rdct_xor3 = ((1<<(i*3)) | 3) ^ rdct_xor3;
  }
}


void
check (int init, int loop)
{
  int	rst_xor, rst_xor2, rst_xor3;
  int	i;


  rst_xor = rst_xor2 = rst_xor3 = init;
  for (i=0;  i<loop;  i++) {
    rst_xor ^= (1<<i);
    rst_xor2 = rst_xor2 ^ ((1<<(i*2))| 2);
    rst_xor3 = ((1<<(i*3)) | 3) ^ rst_xor3;
  }

  if (rst_xor != rdct_xor) {
    errors += 1;
  }

  if (rst_xor2 != rdct_xor2) {
    errors += 1;
  }

  if (rst_xor3 != rdct_xor3) {
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


  rdct_xor = rdct_xor2 = rdct_xor3 = 0;
  #pragma omp parallel
  {
    #pragma omp for reduction(^:rdct_xor,rdct_xor2,rdct_xor3)
    for (i=0; i<LOOPNUM; i++) {
      rdct_xor ^= (1<<i);
      rdct_xor2 = rdct_xor2 ^ ((1<<(i*2))| 2);
      rdct_xor3 = ((1<<(i*3)) | 3) ^ rdct_xor3;
    }
  }
  check (0, LOOPNUM);

  rdct_xor = rdct_xor2 = rdct_xor3 = 1<<8;
  #pragma omp parallel
  {
    #pragma omp for reduction(^:rdct_xor,rdct_xor2,rdct_xor3)
    for (i=0; i<LOOPNUM; i++) {
      rdct_xor ^= (1<<i);
      rdct_xor2 = rdct_xor2 ^ ((1<<(i*2))| 2);
      rdct_xor3 = ((1<<(i*3)) | 3) ^ rdct_xor3;
    }
  }
  check (1<<8, LOOPNUM);

  rdct_xor = rdct_xor2 = rdct_xor3 = 1<<16;
  #pragma omp parallel
  func_reduction (LOOPNUM);
  check (1<<16, LOOPNUM);

  rdct_xor = rdct_xor2 = rdct_xor3 = 1<<24;
  func_reduction (LOOPNUM);
  check (1<<24, LOOPNUM);


  if (errors == 0) {
    printf ("reduction 007 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 007 : FAILED\n");
    return 1;
  }
}
