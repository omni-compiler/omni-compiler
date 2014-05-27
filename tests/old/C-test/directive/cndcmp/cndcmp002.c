static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* condition compilation 002:
 * define パラメータの値の確認
 */

#include "omni.h"

main ()
{
  int	errors = 0;


  if (_OPENMP != 199810  &&		/* version 1.0 */
      _OPENMP != 200203			/* version 2.0 */
      ) {
    ERROR (errors);
  }

  if (errors == 0) {
    printf ("condition complilation 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("condition complilation 002 : FAILED\n");
    return 1;
  }
}
