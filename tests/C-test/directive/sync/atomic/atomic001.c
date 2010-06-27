static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* atomic 001:
 * atomic の動作確認
 */

#include <omp.h>
#include "omni.h"


#ifdef __OMNI_SCASH__
#define LOOPNUM		(thds * 5)
#else
#define LOOPNUM		(thds * 1000)
#endif
#define INT_BIT		(sizeof(int) * 8)

int		errors = 0;
int		thds;
int		atom_incr, atom_incr2, atom_decr, atom_decr2;
int		atom_plus, atom_mul, atom_minus, atom_div;
int		atom_and, atom_or, atom_xor;
unsigned int	atom_lshift, atom_rshift;


int
sameas (int v)
{
  return v;
}


void
clear ()
{
  int	i;

  atom_incr = 0;
  atom_incr2 = 0;
  atom_decr = 0;
  atom_decr2 = 0;

  atom_plus = 0;
  atom_mul = 1;
  atom_minus = 0;
  atom_div = 1 << (INT_BIT - 2);

  for (i=atom_and=0; i<INT_BIT; i++)
    atom_and |= (1<<i);
  for (i=atom_xor=0; i<INT_BIT; i++)
    atom_xor |= (1<<i);
  atom_or = 0;
  atom_lshift = 1;
  atom_rshift = 1 << (INT_BIT - 2);
}


int
check ()
{
  int	i, tmp;

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

  if (atom_plus != LOOPNUM*2) {
    err ++;
  }
  if (atom_minus != -LOOPNUM*3) {
    err ++;
  }

  if (atom_mul != 1) {
    err ++;
  }
  if (atom_div != (1 << (INT_BIT - 2))) {
    err ++;
  }
  for (i=tmp=0; i<INT_BIT; i++) {
    tmp |= (1<<i);
  }
  for (i=0; i<LOOPNUM; i++) {
    tmp &= ~(1<<(i%(INT_BIT-1)));
  }
  if (atom_and != tmp) {
    err ++;
  }
  for (i=tmp=0; i<INT_BIT; i++)
    tmp |= (1<<i);
  for (i=0; i<LOOPNUM; i++) {
    tmp ^= (1<<(i%(INT_BIT-1)));
  }
  if (atom_xor != tmp) {
    err ++;
  }
  for(i=tmp=0; i<LOOPNUM; i++)
    tmp |= (1<<(i%(INT_BIT-1)));
  if (atom_or != tmp) {
    err ++;
  }
  if (atom_lshift != 1) {
    err ++;
  }
  if (atom_rshift != (1 << (INT_BIT - 2))) {
    err ++;
  }

  return err;
}


void
func_atomic ()
{
  int i;

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
    atom_plus += sameas(1) + 1;
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    #pragma omp atomic	
    atom_minus -= sameas(2) + 1;
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    if (omp_get_thread_num () < (INT_BIT - 2)) {
      #pragma omp atomic	
      atom_mul *= sameas(3) - 1;
      #pragma omp atomic	
      atom_mul /= 1 - sameas(3);
    }
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    if (omp_get_thread_num () < (INT_BIT - 2)) {
      #pragma omp atomic	
      atom_div /= 1 + sameas(1);
      #pragma omp atomic	
      atom_div *= 1 + sameas(1);
    }
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    #pragma omp atomic	
    atom_and &= ~(1<<(i%(INT_BIT-1)));
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    #pragma omp atomic	
    atom_xor ^= (1<<(i%(INT_BIT-1)));
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    #pragma omp atomic	
    atom_or |= (1<<(i%(INT_BIT-1)));
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    if (omp_get_thread_num () < (INT_BIT - 2)) {
      #pragma omp atomic	
      atom_lshift <<= 1;

      #pragma omp atomic	
      atom_lshift >>= 1;
    }
  }

  #pragma omp for
  for (i=0; i<LOOPNUM; i++) {
    if (omp_get_thread_num () < (INT_BIT - 2)) {
      #pragma omp atomic	
      atom_rshift >>= 1;

      #pragma omp atomic	
      atom_rshift <<= 1;
    }
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
      atom_plus += sameas(1) + 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_minus -= sameas(2) + 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      if (omp_get_thread_num () < (INT_BIT - 2)) {
        #pragma omp atomic	
	atom_mul *= sameas(3) - 1;
        #pragma omp atomic	
	atom_mul /= 1 - sameas(3);
      }
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      if (omp_get_thread_num () < (INT_BIT - 2)) {
        #pragma omp atomic	
        atom_div /= 1 + sameas(1);
        #pragma omp atomic	
        atom_div *= 1 + sameas(1);
      }
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_and &= ~(1<<(i%(INT_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_xor ^= (1<<(i%(INT_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_or |= (1<<(i%(INT_BIT-1)));
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      if (omp_get_thread_num () < (INT_BIT - 2)) {
        #pragma omp atomic	
        atom_lshift <<= 1;

        #pragma omp atomic	
        atom_lshift >>= 1;
      }
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      if (omp_get_thread_num () < (INT_BIT - 2)) {
        #pragma omp atomic	
        atom_rshift >>= 1;

        #pragma omp atomic	
        atom_rshift <<= 1;
      }
    }
  }
  errors += check ();

  clear ();
  #pragma omp parallel
  func_atomic ();
  errors += check ();

  clear ();
  func_atomic ();
  errors += check ();

  if (errors == 0) {
    printf ("atomic 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("atomic 001 : FAILED\n");
    return 1;
  }
}
