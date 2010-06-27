static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 001 :
 * reduction の確認(何も実行されないケース)
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

int	rdct_pls, rdct_mns;
int	rdct_mul, rdct_land;
int	rdct_lor, rdct_xor;
int	rdct_and, rdct_or;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  rdct_pls  = 1;
  rdct_mns  = 2;
  rdct_mul  = 3;
  rdct_land = 4;
  rdct_lor  = 5;
  rdct_xor  = 6;
  rdct_and  = 0;
  rdct_or   = 0;
  #pragma omp parallel for reduction(+:rdct_pls) reduction(-:rdct_mns) \
			   reduction(*:rdct_mul) reduction(&:rdct_land) \
			   reduction (|:rdct_lor) reduction (^:rdct_xor) \
			   reduction (&&:rdct_and) reduction (||:rdct_or)
  for (i=0; i<LOOPNUM; i++) {
  }

  if (rdct_pls != 1) { errors += 1; }
  if (rdct_mns != 2) { errors += 1; }
  if (rdct_mul != 3) { errors += 1; }
  if (rdct_land != 4) { errors += 1; }
  if (rdct_lor != 5) { errors += 1; }
  if (rdct_xor != 6) { errors += 1; }
  if (rdct_and != 0) { errors += 1; }
  if (rdct_or  != 0) { errors += 1; }

  rdct_and  = 1;
  rdct_or   = 1;
  #pragma omp parallel for reduction (&&:rdct_and) reduction (||:rdct_or)
  for (i=0; i<LOOPNUM; i++) {
  }
  if (rdct_and == 0) { errors += 1; }
  if (rdct_or  == 0) { errors += 1; }


  if (errors == 0) {
    printf ("reduction 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 001 : FAILED\n");
    return 1;
  }
}
