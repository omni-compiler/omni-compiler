static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* condition compilation 001:
 * 条件コンパイルの動作確認
 */

#include "omni.h"

main ()
{
  int	errors;


#ifdef _OPENMP
  errors = 0;
#else
  ERROR (errors);
#endif


  if (errors == 0) {
    printf ("condition complilation 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("condition complilation 001 : FAILED\n");
    return 1;
  }
}
