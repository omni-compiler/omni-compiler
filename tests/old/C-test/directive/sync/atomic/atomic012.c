static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* atomic 012:
 * 共用体のメンバに対する、atomic の動作確認
 */

#include <omp.h>
#include "omni.h"


#define LOOPNUM		(thds * 2)
#define SHORT_BIT	(sizeof(short) * CHAR_BIT)
#define LONG_BIT	(sizeof(long) * CHAR_BIT)
#define LONGLONG_BIT	(sizeof(long long) * CHAR_BIT)
#define INT_BIT		(sizeof(int) * CHAR_BIT)

union x {
  int		i;
  double	d;
};



int		errors = 0;
int		thds;
union x		atom_incr, atom_incr2, atom_decr, atom_decr2;
union x		atom_plus, atom_mul, atom_minus, atom_div;
union x		atom_and, atom_or, atom_xor;
union x		atom_lshift, atom_rshift;


int
sameas (int v)
{
  return v;
}


void
clear ()
{
  atom_incr.d = 0;
  atom_incr2.d = 0;
  atom_decr.d = 0;
  atom_decr2.d = 0;

  atom_plus.d = 0;
  atom_minus.d = 0;
  atom_mul.d = 1;
  atom_div.d = 1;

  atom_and.i = -1;
  atom_or.i = 0;
  atom_xor.i = -1;
  atom_lshift.i = 1;
  atom_rshift.i = 1<<(INT_BIT-2);
}


int
check ()
{
  int	 i;
  int	 tmp;
  double dtmp;

  int	err = 0;


  if (atom_incr.d != LOOPNUM) {
    err ++;
  }
  if (atom_incr2.d != LOOPNUM) {
    err ++;
  }
  if (atom_decr.d != -LOOPNUM) {
    err ++;
  }
  if (atom_decr2.d != -LOOPNUM) {
    err ++;
  }

  if (atom_plus.d != LOOPNUM) {
    err ++;
  }
  if (atom_minus.d != -LOOPNUM) {
    err ++;
  }

  for (i=0,dtmp=1; i<LOOPNUM; i++) {
    dtmp = dtmp * 2;
  }
  if (atom_mul.d != dtmp) {
    err ++;
  }
  for (i=0,dtmp=1; i<LOOPNUM; i++) {
    dtmp = dtmp / 2;
  }
  if (atom_div.d != dtmp) {
    err ++;
  }
  for (i=0,tmp=-1; i<LOOPNUM; i++) {
    tmp &= ~(1<<(i%(INT_BIT-1)));
  }
  if (atom_and.i != tmp) {
    err ++;
  }
  for (i=tmp=0; i<LOOPNUM; i++) {
    tmp |= (1<<(i%(INT_BIT-1)));
  }
  if (atom_or.i != tmp) {
    err ++;
  }
  for (i=0,tmp=-1; i<LOOPNUM; i++) {
    tmp ^= (1<<(i%(INT_BIT-1)));
  }
  if (atom_xor.i != tmp) {
    err ++;
  }
  if (atom_lshift.i != 1) {
    err ++;
  }
  if (atom_rshift.i != (1<<(INT_BIT-2))) {
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
  if (4 < thds) {
    thds = 4;
    omp_set_num_threads (4);
  }

  omp_set_dynamic (0);


  clear ();
  #pragma omp parallel
  {
    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      atom_incr.d ++;
    }

    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      ++ atom_incr2.d;
    }

    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      atom_decr.d --;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      -- atom_decr2.d;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_plus.d += sameas(2) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_minus.d -= sameas(2) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_mul.d *= sameas(3) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_div.d /= 4 + sameas(-2);
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_and.i &= ~(1<<(i%(INT_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_or.i |= (1<<(i%(INT_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_xor.i ^= (1<<(i%(INT_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_lshift.i <<= 1;

      #pragma omp atomic	
      atom_lshift.i >>= 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_rshift.i >>= 1;

      #pragma omp atomic	
      atom_rshift.i <<= 1;
    }
  }
  errors += check ();

  if (errors == 0) {
    printf ("atomic 012 : SUCCESS\n");
    return 0;
  } else {
    printf ("atomic 012 : FAILED\n");
    return 1;
  }
}
