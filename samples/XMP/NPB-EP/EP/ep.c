/*
  NAS Parallel Benchmarks 2.3 Serial Versions in C - EP
*/
/*
c   This is the serial version of the APP Benchmark 1,
c   the "embarassingly parallel" benchmark.
c
c   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
c   numbers.  MK is the Log_2 of the size of each batch of uniform random
c   numbers.  MK can be set for convenience on a given system, since it does
c   not affect the results.
*/
#include <xmp.h>
#include "../npb-C.h"
#include "npbparams.h"

/* parameters (compile-time constants) */
#define	MK		16
#define	MM		(M - MK)
#define	NN		(1 << MM)
#define	NK		(1 << MK)
#define	NQ		10
#define EPSILON		1.0e-8
#define	A		1220703125.0
#define	S		271828183.0
#define	TIMERS_ENABLED	FALSE

/* global variables */
/* common /storage/ */
static double x[(2*NK)+1];	/* x[1:2*NK] */
static double q[NQ];		/* q[0:NQ-1] */
static double qq[10000+1];	/* qq[1:10000] */

#pragma xmp nodes p(*)
#pragma xmp template t(16384)
#pragma xmp distribute t(block) onto p

/* program EMBAR */
int main(void) {

    double Mops, t1, t2, t3, t4, x1, x2, sx, sy, tm, an, tt, gc;
    double dum[3+1] = { 0.0, 1.0, 1.0, 1.0 };	/* dum[1:3] */
    int np, ierr, node, no_nodes, i, ik, kk, l, k, nit, ierrcode,
	no_large_nodes, np_add, k_offset, j, rank;
    boolean verified;
    char size[13+1];	/* character*13 */

/*
c   Because the size of the problem is too large to store in a 32-bit
c   integer for some classes, we put it into a string (for printing).
c   Have to strip off the decimal point put in there by the floating
c   point print statement (internal file)
*/
    rank = xmpc_node_num();

    if (rank == 0) printf("NAS Parallel Benchmarks 2.3-serial version - EP Benchmark\n");
    sprintf(size, "%12.0f", pow(2.0, M+1));
    for (j = 13; j >= 1; j--) {
	if (size[j] == '.') size[j] = ' ';
    }
    if (rank == 0) printf("Number of random numbers generated: %13s\n", size);
/*    printf("Number of active processes:         %12d\n", nprocs); */

    verified = FALSE;

/*
c   Compute the number of "batches" of random number pairs generated 
c   per processor. Adjust if the number of processors does not evenly 
c   divide the total number
*/
    np = NN;

/*
c   Call the random number generator functions and initialize
c   the x-array to reduce the effects of paging on the timings.
c   Also, call all mathematical functions that are used. Make
c   sure these initializations cannot be eliminated as dead code.
*/
    vranlc(0, &(dum[1]), dum[2], &(dum[3])); /* dum[] are dummy parameters */
    dum[1] = randlc(&(dum[2]), dum[3]);
    for (i = 1; i <= 2*NK; i++) x[i] = -1.0e99;
    Mops = log(sqrt(fabs(max(1.0, 1.0))));

    timer_clear(1);
    timer_clear(2);
    timer_clear(3);
    timer_start(1);

    vranlc(0, &t1, A, x);

/*   Compute AN = A ^ (2 * NK) (mod 2^46). */

    t1 = A;

    for ( i = 1; i <= MK+1; i++) {
	t2 = randlc(&t1, t1);
    }

    an = t1;
    tt = S;
    gc = 0.0;
    sx = 0.0;
    sy = 0.0;

    for ( i = 0; i <= NQ - 1; i++) {
	q[i] = 0.0;
    }
      
/*
c   Each instance of this loop may be performed independently. We compute
c   the k offsets separately to take into account the fact that some nodes
c   have more numbers to generate than others
*/
    k_offset = -1;

#pragma xmp loop on t(k)
    for (k = 1; k <= np; k++) {
	kk = k_offset + k;
	t1 = S;
	t2 = an;

/*      Find starting seed t1 for this kk. */

	for (i = 1; i <= 100; i++) {
            ik = kk / 2;
            if (2 * ik != kk) t3 = randlc(&t1, t2);
            if (ik == 0) break;
            t3 = randlc(&t2, t2);
            kk = ik;
	}

/*      Compute uniform pseudorandom numbers. */

	if (TIMERS_ENABLED == TRUE) timer_start(3);
	vranlc(2*NK, &t1, A, x);
	if (TIMERS_ENABLED == TRUE) timer_stop(3);

/*
c       Compute Gaussian deviates by acceptance-rejection method and 
c       tally counts in concentric square annuli.  This loop is not 
c       vectorizable.
*/
	if (TIMERS_ENABLED == TRUE) timer_start(2);

	for ( i = 1; i <= NK; i++) {
            x1 = 2.0 * x[2*i-1] - 1.0;
            x2 = 2.0 * x[2*i] - 1.0;
            t1 = x1 * x1 + x2 * x2;
            if (t1 <= 1.0) {
		t2 = sqrt(-2.0 * log(t1) / t1);
		t3 = (x1 * t2);				/* Xi */
		t4 = (x2 * t2);				/* Yi */
		l = max(abs(t3), abs(t4));
		q[l] += 1.0;				/* counts */
		sx = sx + t3;				/* sum of Xi */
		sy = sy + t4;				/* sum of Yi */
            }
	}
	if (TIMERS_ENABLED == TRUE) timer_stop(2);
    }
/* end of parallel region */    

#pragma xmp reduction(+:sx, sy)

    for (i = 0; i <= NQ-1; i++) {
        gc = gc + q[i];
    }

    timer_stop(1);
    tm = timer_read(1);

    nit = 0;
    if (M == 24) {
	if((fabs((sx- (-3.247834652034740e3))/sx) <= EPSILON) &&
	   (fabs((sy- (-6.958407078382297e3))/sy) <= EPSILON)) {
	    verified = TRUE;
	}
    } else if (M == 25) {
	if ((fabs((sx- (-2.863319731645753e3))/sx) <= EPSILON) &&
	    (fabs((sy- (-6.320053679109499e3))/sy) <= EPSILON)) {
	    verified = TRUE;
	}
    } else if (M == 28) {
	if ((fabs((sx- (-4.295875165629892e3))/sx) <= EPSILON) &&
	    (fabs((sy- (-1.580732573678431e4))/sy) <= EPSILON)) {
	    verified = TRUE;
	}
    } else if (M == 30) {
	if ((fabs((sx- (4.033815542441498e4))/sx) <= EPSILON) &&
	    (fabs((sy- (-2.660669192809235e4))/sy) <= EPSILON)) {
	    verified = TRUE;
	}
    } else if (M == 32) {
	if ((fabs((sx- (4.764367927995374e4))/sx) <= EPSILON) &&
	    (fabs((sy- (-8.084072988043731e4))/sy) <= EPSILON)) {
	    verified = TRUE;
	}
    }

    Mops = pow(2.0, M+1)/tm/1000000.0;
 
    if (rank == 0) {
      printf("EP Benchmark Results: \n"
             "CPU Time = %10.4f\n"
             "N = 2^%5d\n"
	     "No. Gaussian Pairs = %15.0f\n"
	     "Sums = %25.15e %25.15e\n"
	     "Counts:\n",
	     tm, M, gc, sx, sy);
      for (i = 0; i  <= NQ-1; i++) {
        printf("%3d %15.0f\n", i, q[i]);
      }
    }

    if (rank == 0) {
      c_print_results("EP", CLASS, M+1, 0, 0, nit,
		      tm, Mops, 	
		      "Random numbers generated",
		      verified, NPBVERSION, COMPILETIME,
		      CS1, CS2, CS3, CS4, CS5, CS6, CS7);
    }

//    if (TIMERS_ENABLED == TRUE) {
//      printf("Total time:     %f", timer_read(1));
//      printf("Gaussian pairs: %f", timer_read(2));
//      printf("Random numbers: %f", timer_read(3));
//    }
}
