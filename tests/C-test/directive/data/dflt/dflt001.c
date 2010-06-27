static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* default 001 :
 * default(shared) が宣言された場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	MAGICNO	100


int	errors = 0;
int	thds;

int		shrd;
const int	cnst = MAGICNO;


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  shrd = 0;
  #pragma omp parallel default(shared)
  {				      /* all variable is shared */
    #pragma omp critical
    {
      shrd  += 1;
    }
  }
  if (shrd != thds) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("default 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("default 001 : FAILED\n");
    return 1;
  }
}
