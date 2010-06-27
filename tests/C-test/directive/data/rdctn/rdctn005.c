static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 005 :
 * reduction(&:...) の動作を確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

int	rdct_land, rdct_land2, rdct_land3;


void
func_reduction (int loop)
{
  int	i;

  #pragma omp for reduction(&:rdct_land,rdct_land2,rdct_land3)
  for (i=1; i<loop; i++) {
    rdct_land &= (~(1<<i));
    rdct_land2 = rdct_land2 & ((~(1<<i)) | 2);
    rdct_land3 = ((~(1<<i)) | 3) & rdct_land3;
  }
}


void
check (int init, int loop)
{
  int	rst_land, rst_land2, rst_land3;
  int	i;


  rst_land = rst_land2 = rst_land3 = init;
  for (i=1;  i<loop;  i++) {
    rst_land &= (~(1<<i));
    rst_land2 = rst_land2 & ((~(1<<i)) | 2);
    rst_land3 = ((~(1<<i)) | 3) & rst_land3;
  }

  if (rst_land != rdct_land) {
    errors += 1;
  }

  if (rst_land2 != rdct_land2) {
    errors += 1;
  }

  if (rst_land3 != rdct_land3) {
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


  rdct_land = rdct_land2 = rdct_land3 = -1;
  #pragma omp parallel
  {
    #pragma omp for reduction(&:rdct_land,rdct_land2,rdct_land3)
    for (i=1; i<LOOPNUM; i++) {
      rdct_land &= (~(1<<i));
      rdct_land2 = rdct_land2 & ((~(1<<i)) | 2);
      rdct_land3 = ((~(1<<i)) | 3) & rdct_land3;
    }
  }
  check (-1, LOOPNUM);

  rdct_land = rdct_land2 = rdct_land3 = -2;
  #pragma omp parallel
  {
    #pragma omp for reduction(&:rdct_land,rdct_land2,rdct_land3)
    for (i=1; i<LOOPNUM; i++) {
      rdct_land &= (~(1<<i));
      rdct_land2 = rdct_land2 & ((~(1<<i)) | 2);
      rdct_land3 = ((~(1<<i)) | 3) & rdct_land3;
    }
  }
  check (-2, LOOPNUM);

  rdct_land = rdct_land2 = rdct_land3 = -3;
  #pragma omp parallel
  func_reduction (LOOPNUM);
  check (-3, LOOPNUM);

  rdct_land = rdct_land2 = rdct_land3 = -4;
  func_reduction (LOOPNUM);
  check (-4, LOOPNUM);


  if (errors == 0) {
    printf ("reduction 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 005 : FAILED\n");
    return 1;
  }
}
