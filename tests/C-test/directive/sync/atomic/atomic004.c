static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* atomic 004:
 * char変数に対する、atomic の動作確認
 */

#include <omp.h>
#include "omni.h"


#define LOOPNUM		(thds * 2)

int		errors = 0;
int		thds;
char		atom_incr, atom_incr2, atom_decr, atom_decr2;
char		atom_plus, atom_mul, atom_minus, atom_div;
char		atom_and, atom_or, atom_xor;
unsigned char	atom_lshift, atom_rshift;


int
sameas (int v)
{
  return v;
}


void
clear ()
{
  atom_incr = 0;
  atom_incr2 = 0;
  atom_decr = 0;
  atom_decr2 = 0;

  atom_plus = 0;
  atom_mul = 1;
  atom_minus = 0;
  atom_div = 1<<(CHAR_BIT - 2);

  atom_and = -1;
  atom_or = 0;
  atom_xor = -1;
  atom_lshift = 1;
  atom_rshift = 1<<(CHAR_BIT - 2);
}


int
check ()
{
  int	i;
  char	tmp;

  int	err = 0;


  if (atom_incr != LOOPNUM) {
    err ++;
  }
  if (atom_incr2 != LOOPNUM) {
    err ++;
  }
  if (atom_decr != -LOOPNUM) {
    err ++;
  }
  if (atom_decr2 != -LOOPNUM) {
    err ++;
  }

  if (atom_plus != LOOPNUM) {
    err ++;
  }
  if (atom_minus != -LOOPNUM) {
    err ++;
  }

  if (atom_mul != 1) {
    err ++;
  }
  if (atom_div != (1<<(CHAR_BIT - 2))) {
    err ++;
  }
  for (i=0,tmp=-1; i<LOOPNUM; i++) {
    tmp &= ~(1<<(i%(CHAR_BIT-1)));
  }
  if (atom_and != tmp) {
    err ++;
  }
  for (i=tmp=0; i<LOOPNUM; i++) {
    tmp |= (1<<(i%(CHAR_BIT-1)));
  }
  if (atom_or != tmp) {
    err ++;
  }
  for (i=0,tmp=-1; i<LOOPNUM; i++) {
    tmp ^= (1<<(i%(CHAR_BIT-1)));
  }
  if (atom_xor != tmp) {
    err ++;
  }
  if (atom_lshift != 1) {
    err ++;
  }
  if (atom_rshift != (1<<(CHAR_BIT - 2))) {
    err ++;
  }

  return err;
}


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  if (2 < thds) {
    thds = 2;
    omp_set_num_threads (2);
  }

  omp_set_dynamic (0);


  clear ();
  #pragma omp parallel
  {
    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      atom_incr ++;
    }

    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      ++ atom_incr2;
    }

    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      atom_decr --;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      -- atom_decr2;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_plus += sameas(2) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_minus -= sameas(2) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_mul *= sameas(3) - 1;
      #pragma omp atomic	
      atom_mul /= 1 - sameas(3);
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_div /= 4 + sameas(-2);
      #pragma omp atomic	
      atom_div *= 4 + sameas(-2);
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_and &= ~(1<<(i%(CHAR_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_or |= (1<<(i%(CHAR_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_xor ^= (1<<(i%(CHAR_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_lshift <<= 1;

      #pragma omp atomic	
      atom_lshift >>= 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_rshift >>= 1;

      #pragma omp atomic	
      atom_rshift <<= 1;
    }
  }
  errors += check ();

  if (errors == 0) {
    printf ("atomic 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("atomic 004 : FAILED\n");
    return 1;
  }
}
