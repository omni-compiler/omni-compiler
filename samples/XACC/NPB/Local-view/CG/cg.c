#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpinpb.h"
#include "timing.h"
#include "npbparams.h"
#include "../common/c_timers.h"
#include "../common/c_print_results.h"
#include "../common/c_rand.h"

typedef int bool;
#define true 1
#define false 0

#define ROW_ARRAY_LEN (na/num_proc_rows+2)

#define USE_GET

#include <xmp.h>
#pragma xmp nodes proc(num_proc_cols, num_proc_rows)
#pragma xmp nodes subproc(num_proc_cols) = proc(:,*)
double w[restrict na/num_proc_rows+2]:[*];
double q[restrict na/num_proc_rows+2]:[*];
double r[restrict na/num_proc_rows+2]:[*];
#pragma acc declare create(w,q,r)
double norm_temp1[2]:[*], norm_temp2[2]:[*], tmp:[*];

static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
static int icnvrt(double x, int ipwr2);
static void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[],
                   double aelt[], int firstrow, int lastrow, double x[], bool mark[],
                   int nzloc[], int nnza);
static void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], bool mark[]);
static void setup_proc_info(int num_procs_, int num_proc_rows_, int num_proc_cols_);
static void setup_submatrix_info(int *l2npcols,
                                 int reduce_exch_proc[],
                                 int reduce_send_starts[],
                                 int reduce_send_lengths[],
                                 int reduce_recv_starts[],
                                 int reduce_recv_lengths[]);
static void makea(int n, int nz, double a[], int colidx[], int rowstr[], int nonzer,
                  int firstrow, int lastrow, int firstcol, int lastcol,
                  double rcond, int arow[], int acol[], double aelt[], double v[], int iv[],
                  double shift);
static void conj_grad(int const colidx[restrict],
                      int const rowstr[restrict],
                      double const x[restrict],
                      double z[restrict],
                      double const a[restrict],
                      double p[restrict],
                      double q[restrict],
                      double r[restrict],
                      double w[restrict],
                      double * restrict rnorm,
                      int l2npcols,
                      int reduce_exch_proc[restrict],
                      int reduce_send_starts[restrict],
                      int reduce_send_lengths[restrict],
                      int reduce_recv_starts[restrict],
                      int reduce_recv_lengths[restrict]);

static void initialize_mpi();
static void *alloc_mem(size_t elmtsize, int length);
static void free_mem(void *);

double amult, tran;
int naa, nzz,
  npcols, nprows,
  proc_col, proc_row,
  firstrow,
  lastrow,
  firstcol,
  lastcol,
  exch_proc,
  exch_recv_length,
  send_start,
  send_len;
int nz;

static inline
int max(int a, int b){
  return (a > b)? a : b;
}

int main(int argc, char** argv){
//--------------------------------------------------------------------
//  num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows.
//  num_proc_cols and num_proc_cols are to be found in npbparams.h.
//  When num_procs is not square, then num_proc_cols must be = 2*num_proc_rows.
//--------------------------------------------------------------------
  const int num_procs = num_proc_cols * num_proc_rows;

//--------------------------------------------------------------------
//  Class specific parameters:
//  It appears here for reference only.
//  These are their values, however, this info is imported in the npbparams.h
//  include file, which is written by the sys/setparams.c program.
//--------------------------------------------------------------------

//---------
//  Class S:
//----------
//       parameter( na=1400,
//      >           nonzer=7,
//      >           shift=10.,
//      >           niter=15,
//      >           rcond=1.0d-1 )
//----------
//  Class W:
//----------
//       parameter( na=7000,
//      >           nonzer=8,
//      >           shift=12.,
//      >           niter=15,
//      >           rcond=1.0d-1 )
//----------
//  Class A:
//----------
//       parameter( na=14000,
//      >           nonzer=11,
//      >           shift=20.,
//      >           niter=15,
//      >           rcond=1.0d-1 )
//----------
//  Class B:
//----------
//       parameter( na=75000,
//      >           nonzer=13,
//      >           shift=60.,
//      >           niter=75,
//      >           rcond=1.0d-1 )
//----------
//  Class C:
//----------
//       parameter( na=150000,
//      >           nonzer=15,
//      >           shift=110.,
//      >           niter=75,
//      >           rcond=1.0d-1 )
//----------
//  Class D:
//----------
//       parameter( na=1500000,
//      >           nonzer=21,
//      >           shift=500.,
//      >           niter=100,
//      >           rcond=1.0d-1 )
//----------
//  Class E:
//----------
//       parameter( na=9000000,
//      >           nonzer=26,
//      >           shift=1500.,
//      >           niter=100,
//      >           rcond=1.0d-1 )


  nz = na*(nonzer+1)/num_procs*(nonzer+1)+nonzer
    + na*(nonzer+2+num_procs/256)/num_proc_cols;

  int * restrict colidx, * restrict rowstr, * restrict iv, *restrict arow, *restrict acol;

  colidx = alloc_mem(sizeof(int), nz);
  rowstr = alloc_mem(sizeof(int), na+1);
  iv = alloc_mem(sizeof(int), 2*na+1);
  arow = alloc_mem(sizeof(int), nz);
  acol = alloc_mem(sizeof(int), nz);

  double *restrict v, *restrict aelt, *restrict a, *restrict x, *restrict z, *restrict p;
  v = alloc_mem(sizeof(double), na+1);
  aelt = alloc_mem(sizeof(double), nz);
  a = alloc_mem(sizeof(double), nz);
  x = alloc_mem(sizeof(double), na/num_proc_rows+2);
  z = alloc_mem(sizeof(double), na/num_proc_rows+2);
  p = alloc_mem(sizeof(double), na/num_proc_rows+2);

  int l2npcols;
  int reduce_exch_proc[num_proc_cols];
  int reduce_send_starts[num_proc_cols];
  int reduce_send_lengths[num_proc_cols];
  int reduce_recv_starts[num_proc_cols];
  int reduce_recv_lengths[num_proc_cols];

  double zeta;
  double rnorm;

  double t, tmax, mflops;
  char class;
  bool verified;
  double zeta_verify_value, epsilon, err;

  double tsum[t_last+2], t1[t_last+2], tming[t_last+2], tmaxg[t_last+2];
  char t_recs[6][9] = {"total", "conjg", "rcomm", "ncomm", " totcomp", " totcomm"}; //6 = t_last+2


//--------------------------------------------------------------------
//  Set up mpi initialization and number of proc testing
//--------------------------------------------------------------------
  initialize_mpi();


  if( na == 1400 &&
      nonzer == 7 &&
      niter == 15 &&
      shift == 10. ){
    class = 'S';
    zeta_verify_value = 8.5971775078648;
  }else if( na == 7000 &&
            nonzer == 8 &&
            niter == 15 &&
            shift == 12. ){
    class = 'W';
    zeta_verify_value = 10.362595087124;
  }else if( na == 14000 &&
            nonzer == 11 &&
            niter == 15 &&
            shift == 20. ){
    class = 'A';
    zeta_verify_value = 17.130235054029;
  }else if( na == 75000 &&
            nonzer == 13 &&
            niter == 75 &&
            shift == 60. ){
    class = 'B';
    zeta_verify_value = 22.712745482631;
  }else if( na == 150000 &&
            nonzer == 15 &&
            niter == 75 &&
            shift == 110. ){
    class = 'C';
    zeta_verify_value = 28.973605592845;
  }else if( na == 1500000 &&
            nonzer == 21 &&
            niter == 100 &&
            shift == 500. ){
    class = 'D';
    zeta_verify_value = 52.514532105794;
  }else if( na == 9000000 &&
            nonzer == 26 &&
            niter == 100 &&
            shift == 1.5e3 ){
    class = 'E';
    zeta_verify_value = 77.522164599383;
  }else{
    class = 'U';
  }

  if( me == root ){
    printf("\n\n NAS Parallel Benchmarks 3.3 -- CG Benchmark\n\n");
    printf(" Size: %10d\n", na );
    printf(" Iterations: %5d\n", niter );
    printf(" Number of active processes: %5d\n", nprocs);
    printf(" Number of nonzeroes per row: %8d\n", nonzer);
    printf(" Eigenvalue shift: %8.3e\n", shift);
  }


  naa = na;
  nzz = nz;

//--------------------------------------------------------------------
//  Set up processor info, such as whether sq num of procs, etc
//--------------------------------------------------------------------
  setup_proc_info( num_procs,
                   num_proc_rows,
                   num_proc_cols );


//--------------------------------------------------------------------
//  Set up partition's submatrix info: firstcol, lastcol, firstrow, lastrow
//--------------------------------------------------------------------
  setup_submatrix_info(&l2npcols,
                       reduce_exch_proc,
                       reduce_send_starts,
                       reduce_send_lengths,
                       reduce_recv_starts,
                       reduce_recv_lengths);


  for(int i = 0; i < t_last; i++){
    timer_clear(i);
  }

//--------------------------------------------------------------------
//  Inialize random number generator
//--------------------------------------------------------------------
  tran    = 314159265.0;
  amult   = 1220703125.0;
  zeta    = randlc(&tran, &amult);

//--------------------------------------------------------------------
//  Set up partition's sparse random matrix for given class size
//--------------------------------------------------------------------
  makea(naa, nzz, a, colidx, rowstr, nonzer,
        firstrow, lastrow, firstcol, lastcol,
        rcond, arow, acol, aelt, v, iv, shift);

//--------------------------------------------------------------------
//  Note: as a result of the above call to makea:
//        values of j used in indexing rowstr go from 0 --> lastrow-firstrow+1
//        values of colidx which are col indexes go from firstcol --> lastcol
//        So:
//        Shift the col index vals from actual (firstcol --> lastcol )
//        to local, i.e., (0 --> lastcol-firstcol+1)
//--------------------------------------------------------------------
  for(int j = 0; j < lastrow - firstrow + 1; j++){
    for(int k = rowstr[j]; k < rowstr[j + 1]; k++){
      colidx[k] = colidx[k] - firstcol;
    }
  }

//--------------------------------------------------------------------
//  set starting vector to (1, 1, .... 1)
//--------------------------------------------------------------------
  for(int i = 0; i < na / num_proc_rows + 1; i++){
    x[i] = 1.0;
  }

  zeta  = 0.0;

#pragma acc data pcopyin(colidx[0:nz], rowstr[0:na+1], x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN], a[0:nz], p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN], w[0:ROW_ARRAY_LEN]) pcopyin(reduce_recv_lengths[0:num_proc_cols])
  {
//--------------------------------------------------------------------
//---->
//  Do one iteration untimed to init all code and data page tables
//---->                    (then reinit, start timing, to niter its)
//--------------------------------------------------------------------
  for(int it = 1; it <= 1; it++){
//--------------------------------------------------------------------
//  The call to the conjugate gradient routine:
//--------------------------------------------------------------------
    conj_grad(colidx,
              rowstr,
              x,
              z,
              a,
              p,
              q,
              r,
              w,
              &rnorm,
              l2npcols,
              reduce_exch_proc,
              reduce_send_starts,
              reduce_send_lengths,
              reduce_recv_starts,
              reduce_recv_lengths);

//--------------------------------------------------------------------
//  zeta = shift + 1/(x.z)
//  So, first: (x.z)
//  Also, find norm of z
//  So, first: (z.z)
//--------------------------------------------------------------------
    double norm1 = 0.0, norm2 = 0.0;
#pragma acc parallel loop reduction(+:norm1,norm2) pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      norm1 = norm1 + x[j] * z[j];
      norm2 = norm2 + z[j] * z[j];
    }
    norm_temp1[0] = norm1;
    norm_temp1[1] = norm2;

#if USE_REDUCTION_DIRECTIVE
    if (timeron) timer_start(t_ncomm);
#pragma xmp reduction(+:norm_temp1) on subproc
    if (timeron) timer_stop(t_ncomm);
#else
    for(int i = 0; i < l2npcols; i++){
      if (timeron) timer_start(t_ncomm);
      xmp_sync_image(reduce_exch_proc[i], NULL);
      norm_temp2[:]:[reduce_exch_proc[i]] = norm_temp1[:];
      xmp_sync_image(reduce_exch_proc[i], NULL);
      if (timeron) timer_stop(t_ncomm);

      norm_temp1[0] = norm_temp1[0] + norm_temp2[0];
      norm_temp1[1] = norm_temp1[1] + norm_temp2[1];
    }
#endif
    norm2 = norm_temp1[1];

    norm2 = 1.0 / sqrt(norm2);

//--------------------------------------------------------------------
//  Normalize z to obtain x
//--------------------------------------------------------------------
#pragma acc parallel loop pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      x[j] = norm2 * z[j];
    }
  } // end of do one iteration untimed

//--------------------------------------------------------------------
//  set starting vector to (1, 1, .... 1)
//--------------------------------------------------------------------
//
//  NOTE: a questionable limit on size:  should this be na/num_proc_cols+1 ?
//
#pragma acc parallel loop pcopy(x[0:ROW_ARRAY_LEN])
  for(int i = 0; i < na / num_proc_rows + 1; i++){
    x[i] = 1.0;
  }

  zeta = 0.0;

//--------------------------------------------------------------------
//  Synchronize and start timing
//--------------------------------------------------------------------
  for(int i = 0; i < t_last; i++){
    timer_clear(i);
  }

#pragma xmp barrier

  timer_clear(t_total);
  timer_start(t_total);
//--------------------------------------------------------------------
//---->
//  Main Iteration for inverse power method
//---->
//--------------------------------------------------------------------
  for(int it = 1; it <= niter; it++){
//--------------------------------------------------------------------
//  The call to the conjugate gradient routine:
//--------------------------------------------------------------------
    conj_grad(colidx,
              rowstr,
              x,
              z,
              a,
              p,
              q,
              r,
              w,
              &rnorm,
              l2npcols,
              reduce_exch_proc,
              reduce_send_starts,
              reduce_send_lengths,
              reduce_recv_starts,
              reduce_recv_lengths);

//--------------------------------------------------------------------
//  zeta = shift + 1/(x.z)
//  So, first: (x.z)
//  Also, find norm of z
//  So, first: (z.z)
//--------------------------------------------------------------------
    double norm1 = 0.0, norm2 = 0.0;
#pragma acc parallel loop reduction(+:norm1, norm2) pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      norm1 = norm1 + x[j] * z[j];
      norm2 = norm2 + z[j] * z[j];
    }

    norm_temp1[0] = norm1;
    norm_temp1[1] = norm2;

#ifdef USE_REDUCTION_DIRECTIVE
    if (timeron) timer_start(t_ncomm);
#pragma xmp reduction(+:norm_temp1) on subproc
    if (timeron) timer_stop(t_ncomm);
#else
    for(int i = 0; i < l2npcols; i++){
      if (timeron) timer_start(t_ncomm);
      xmp_sync_image(reduce_exch_proc[i], NULL);
      norm_temp2[:]:[reduce_exch_proc[i]] = norm_temp1[:];
      xmp_sync_image(reduce_exch_proc[i], NULL);
      if (timeron) timer_stop(t_ncomm);

      norm_temp1[0] = norm_temp1[0] + norm_temp2[0];
      norm_temp1[1] = norm_temp1[1] + norm_temp2[1];
    }
#endif

    norm1 = norm_temp1[0];
    norm2 = norm_temp1[1];

    norm2 = 1.0 / sqrt(norm2);

    if(me == root){
      zeta = shift + 1.0 / norm1;
      if(it == 1){
        printf("\n   iteration           ||r||                 zeta\n");
      }
      printf("    %5d       %20.14e%20.13f\n", it, rnorm, zeta);
    }

//--------------------------------------------------------------------
//  Normalize z to obtain x
//--------------------------------------------------------------------
#pragma acc parallel loop pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      x[j] = norm2 * z[j];
    }
  } // end of main iter inv pow meth

  timer_stop(t_total);

  }//end of acc data

//--------------------------------------------------------------------
//  End of timed section
//--------------------------------------------------------------------
  tmax = t = timer_read(t_total);

#pragma xmp reduction(max:tmax)

  if(me == root){
    printf(" Benchmark completed \n");

    epsilon = 1.e-10;
    if (class != 'U'){

      err = fabs(zeta - zeta_verify_value)/zeta_verify_value;
      if(err <= epsilon){
        verified = true;
        printf(" VERIFICATION SUCCESSFUL \n");
        printf(" Zeta is    %20.13e\n", zeta);
        printf(" Error is   %20.13e\n", err);
      }else{
        verified = false;
        printf(" VERIFICATION FAILED\n");
        printf(" Zeta                %20.13e\n", zeta);
        printf(" The correct zeta is %20.13e\n", zeta_verify_value);
      }
    }else{
      verified = false;
      printf(" Problem size unknown\n");
      printf(" NO VERIFICATION PERFORMED\n");
    }


    if( tmax != 0. ){
      mflops = (double)( 2*niter*na )
        * ( 3.+(double)( nonzer*(nonzer+1) )
            + 25.*(5.+(double)( nonzer*(nonzer+1) ))
            + 3. ) / tmax / 1000000.0;
    }else{
      mflops = 0.0;
    }
    c_print_results("CG", class, na, 0, 0,
                    niter, nnodes_compiled, (nprocs), tmax,
                    mflops, "          floating point",
                    verified, NPBVERSION, COMPILETIME,
                    MPICC, CLINK, CMPI_LIB, CMPI_INC, CFLAGS, CLINKFLAGS);

  }

  if (! timeron) goto continue999;

  for(int i = 0; i < t_last; i++){
    t1[i] = timer_read(i);
  }
  t1[t_conjg] = t1[t_conjg] - t1[t_rcomm];
  t1[t_last+1] = t1[t_rcomm] + t1[t_ncomm];
  t1[t_last] = t1[t_total] - t1[t_last+1];

  for(int i = 0; i < t_last+2; i++){
    tsum[i] = tming[i] = tmaxg[i] = t1[i];
  }
#pragma xmp reduction(+:tsum)
#pragma xmp reduction(min:tming)
#pragma xmp reduction(max:tmaxg)

  if (me == 0){
    printf(" nprocs =%6d           minimum     maximum     average\n", nprocs);
    for(int i = 0; i < t_last+2; i++){
      tsum[i] = tsum[i] / nprocs;
      printf(" timer %2d(%8s) :  %10.4f  %10.4f  %10.4f\n", i, t_recs[i], tming[i], tmaxg[i], tsum[i]);
    }
  }
 continue999:
  free_mem(colidx);
  free_mem(rowstr);
  free_mem(iv);
  free_mem(arow);
  free_mem(acol);
  free_mem(v);
  free_mem(aelt);
  free_mem(a);
  free_mem(x);
  free_mem(z);
  free_mem(p);

  return 0;
} // end main


void initialize_mpi()
{
  me = xmp_node_num() - 1;
  nprocs = xmp_num_nodes();
  root = 0;

  if (me == root){
    FILE *fp = fopen("timer.flag", "r");
    timeron = false;
    if (fp != NULL){
      timeron = true;
      fclose(fp);
    }
  }

#pragma xmp bcast(timeron)

  return;
}


void setup_proc_info(int num_procs,
                     int num_proc_rows_,
                     int num_proc_cols_)
{
  int i, ierr;
  int log2nprocs;
//--------------------------------------------------------------------
//  num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows
//  When num_procs is not square, then num_proc_cols = 2*num_proc_rows
//--------------------------------------------------------------------
//  First, number of procs must be power of two.
//--------------------------------------------------------------------
  if( nprocs != num_procs ){
    if( me == root ){
      printf("Error: num of procs allocated (%d) is not equal to \
compiled number of procs (%d)", nprocs, num_procs);
    }
    exit(1); //   stop
  }


  i = num_proc_cols;
 continue100:
  if( i != 1 && i/2*2 != i ){
    if ( me == root ){
      printf("Error: num_proc_cols is %d which is not a power of two\n", num_proc_cols);
    }
    exit(1);
  }
  i = i / 2;
  if( i != 0 ){
    goto continue100;
  }

  i = num_proc_rows;
 continue200:
  if( i != 1 && i/2*2 != i ){
    if ( me == root ){
      printf("Error: num_proc_rows is %d which is not a power of two\n", num_proc_rows);
    }
    exit(1);
  }
  i = i / 2;
  if( i != 0 ){
    goto continue200;
  }

  log2nprocs = 0;
  i = nprocs;
 continue300:
  if( i != 1 && i/2*2 != i ){
    printf("Error: nprocs is %d which is not a power of two\n", nprocs);
    exit(1);
  }
  i = i / 2;
  if( i != 0 ){
    log2nprocs = log2nprocs + 1;
    goto continue300;
  }

  npcols = num_proc_cols;
  nprows = num_proc_rows;

  return;
}


void setup_submatrix_info( int *l2npcols,
                           int reduce_exch_proc[],
                           int reduce_send_starts[],
                           int reduce_send_lengths[],
                           int reduce_recv_starts[],
                           int reduce_recv_lengths[] )
{
  int col_size, row_size;
  int i, j;
  int div_factor;

  proc_row = me / npcols;
  proc_col = me - proc_row*npcols;


//--------------------------------------------------------------------
//  If naa evenly divisible by npcols, then it is evenly divisible
//  by nprows
//--------------------------------------------------------------------

  if( naa/npcols*npcols == naa ){
    col_size = naa/npcols;
    firstcol = proc_col*col_size;
    lastcol  = firstcol - 1 + col_size;
    row_size = naa/nprows;
    firstrow = proc_row*row_size;
    lastrow  = firstrow - 1 + row_size;
//--------------------------------------------------------------------
//  If naa not evenly divisible by npcols, then first subdivide for nprows
//  and then, if npcols not equal to nprows (i.e., not a sq number of procs),
//  get col subdivisions by dividing by 2 each row subdivision.
//--------------------------------------------------------------------
  } else {
    if( proc_row < naa - naa/nprows*nprows){
      row_size = naa/nprows+ 1;
      firstrow = proc_row*row_size;
      lastrow  = firstrow - 1 + row_size;
    } else {
      row_size = naa/nprows;
      firstrow = (naa - naa/nprows*nprows)*(row_size+1)
        + (proc_row-(naa-naa/nprows*nprows))
        * row_size;
      lastrow  = firstrow - 1 + row_size;
    }
    if( npcols ==  nprows ){
      if( proc_col < naa - naa/npcols*npcols ){
        col_size = naa/npcols+ 1;
        firstcol = proc_col*col_size;
        lastcol  = firstcol - 1 + col_size;
      }else{
        col_size = naa/npcols;
        firstcol = (naa - naa/npcols*npcols)*(col_size+1)
          + (proc_col-(naa-naa/npcols*npcols))
          * col_size;
        lastcol  = firstcol - 1 + col_size;
      }
    } else {
      if( (proc_col/2) <
          naa - naa/(npcols/2)*(npcols/2) ) {
        col_size = naa/(npcols/2) + 1;
        firstcol = (proc_col/2)*col_size;
        lastcol  = firstcol - 1 + col_size;
      }else{
        col_size = naa/(npcols/2);
        firstcol = (naa - naa/(npcols/2)*(npcols/2))
          * (col_size+1)
          + ((proc_col/2)-(naa-naa/(npcols/2)*(npcols/2)))
          * col_size;
        lastcol  = firstcol - 1 + col_size;
      }
      if( (me % 2) == 0 ){
        lastcol  = firstcol - 1 + (col_size-1)/2 + 1;
      }else{
        firstcol = firstcol + (col_size-1)/2 + 1;
        lastcol  = firstcol - 1 + col_size/2;
      }
    }
  }



  if( npcols == nprows ) {
    send_start = 0;
    send_len   = lastrow - firstrow + 1;
  } else {
    if( (me % 2) == 0 ){
      send_start = 0;
      send_len   = (1 + lastrow-firstrow+1)/2;
    } else{
      send_start = (1 + lastrow-firstrow+1)/2;
      send_len   = (lastrow-firstrow+1)/2;
    }
  }



//--------------------------------------------------------------------
//  Transpose exchange processor
//--------------------------------------------------------------------

  if( npcols == nprows ){
    exch_proc = (me % nprows )*nprows + me/nprows;
  } else {
    exch_proc = 2*(( me/2 % nprows )*nprows + me/2/nprows)
      + ( me % 2 );
  }

  i = npcols / 2;
  *l2npcols = 0;
  while( i > 0 ){
    *l2npcols = *l2npcols + 1;
    i = i / 2;
  }


//--------------------------------------------------------------------
//  Set up the reduce phase schedules...
//--------------------------------------------------------------------

  div_factor = npcols;
  for(int i = 0; i < *l2npcols; i++){

    j = ( (proc_col+div_factor/2) % div_factor )
      + proc_col / div_factor * div_factor;
    reduce_exch_proc[i] = proc_row*npcols + j;

    div_factor = div_factor / 2;

  }

  for(int i = *l2npcols - 1; i >= 0; i--){

    if( nprows == npcols ){
      reduce_send_starts[i]  = send_start;
      reduce_send_lengths[i] = send_len;
      reduce_recv_lengths[i] = lastrow - firstrow + 1;
    } else {
      reduce_recv_lengths[i] = send_len;
      if( i == *l2npcols - 1){
        reduce_send_lengths[i] = lastrow-firstrow+1 - send_len;
        if( me/2*2 == me ){
          reduce_send_starts[i] = send_start + send_len;
        } else {
          reduce_send_starts[i] = 0;
        }
      } else {
        reduce_send_lengths[i] = send_len;
        reduce_send_starts[i]  = send_start;
      }
    }
    reduce_recv_starts[i] = send_start;

  }

  exch_recv_length = lastcol - firstcol + 1;

  return;
}


void conj_grad(int const colidx[restrict],
	       int const rowstr[restrict],
	       double const x[restrict],
	       double       z[restrict],
	       double const a[restrict],
	       double       p[restrict],
	       double       q_[restrict],
	       double       r_[restrict],
	       double       w_[restrict],
	       double * restrict rnorm,
	       int l2npcols,
	       int reduce_exch_proc[restrict],
	       int reduce_send_starts[restrict],
	       int reduce_send_lengths[restrict],
	       int reduce_recv_starts[restrict],
	       int reduce_recv_lengths[restrict])
{
//--------------------------------------------------------------------
//  Floaging point arrays here are named as in NPB1 spec discussion of
//  CG algorithm
//--------------------------------------------------------------------
  int const cgitmax = 25;
  double d, sum, rho, rho0, alpha, beta;

  if (timeron) timer_start(t_conjg);
#pragma acc data pcopy(colidx[0:nz], rowstr[0:na+1], x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN], a[0:nz], p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN], w[0:ROW_ARRAY_LEN]) pcopy(reduce_recv_lengths[0:l2npcols])
#pragma acc host_data use_device(q, w, r)
  {
//--------------------------------------------------------------------
//  Initialize the CG algorithm:
//--------------------------------------------------------------------
#pragma acc parallel loop pcopy(p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN], w[0:ROW_ARRAY_LEN], x[0:ROW_ARRAY_LEN])
  for(int j = 0; j < naa / nprows + 1; j++){
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
    w[j] = 0.0;
  }

//--------------------------------------------------------------------
//  rho = r.r
//  Now, obtain the norm of r: First, sum squares of r elements locally...
//--------------------------------------------------------------------
  rho = 0.0;
#pragma acc parallel loop reduction(+:rho) pcopy(r[0:ROW_ARRAY_LEN])
  for(int j = 0; j < lastcol - firstcol + 1; j++){
    rho = rho + r[j] * r[j];
  }

//--------------------------------------------------------------------
//  Exchange and sum with procs identified in reduce_exch_proc
//  (This is equivalent to mpi_allreduce.)
//  Sum the partial sums of rho, leaving rho on all processors
//--------------------------------------------------------------------
#ifdef USE_REDUCTION_DIRECTIVE
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:rho) on subproc
  if (timeron) timer_stop(t_rcomm);
#else
  for(int i = 0; i < l2npcols; i++){
    if (timeron) timer_start(t_rcomm);
    xmp_sync_image(reduce_exch_proc[i], NULL);
    tmp:[reduce_exch_proc[i]] = rho;
    xmp_sync_image(reduce_exch_proc[i], NULL);
    if (timeron) timer_stop(t_rcomm);

    rho = rho + tmp;
  }
#endif

//--------------------------------------------------------------------
//---->
//  The conj grad iteration loop
//---->
//--------------------------------------------------------------------
  for(int cgit = 0; cgit < cgitmax; cgit++){
//--------------------------------------------------------------------
//  q = A.p
//  The partition submatrix-vector multiply: use workspace w
//--------------------------------------------------------------------
#pragma acc parallel loop gang pcopy(a[0:nz], p[0:ROW_ARRAY_LEN], colidx[0:nz], rowstr[0:na+1], w[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastrow - firstrow + 1; j++){
      double sum = 0.0;
      int rowstr_j = rowstr[j];
      int rowstr_j1 = rowstr[j+1];
#pragma acc loop vector reduction(+:sum)
      for(int k = rowstr_j; k < rowstr_j1; k++){
        sum = sum + a[k] * p[colidx[k]];
      }
      w[j] = sum;
    }

//--------------------------------------------------------------------
//  Sum the partition submatrix-vec A.p's across rows
//  Exchange and sum piece of w with procs identified in reduce_exch_proc
//--------------------------------------------------------------------
    for(int i = l2npcols - 1; i >= 0; i--){
      if (timeron) timer_start(t_rcomm);
      int reduce_proc = reduce_exch_proc[i];
      xmp_sync_image(reduce_proc, NULL);
#ifdef USE_GET
      int reduce_start = reduce_recv_starts[i];
      int reduce_length = reduce_recv_lengths[i];
      q[reduce_start : reduce_length] = w[reduce_start : reduce_length]:[reduce_proc];
#else
      int reduce_start = reduce_send_starts[i];
      int reduce_length = reduce_send_lengths[i];
      q[reduce_start : reduce_length]:[reduce_proc] = w[reduce_start : reduce_length];
#endif
      xmp_sync_image(reduce_proc, NULL);
      if (timeron) timer_stop(t_rcomm);
#pragma acc parallel loop pcopy(w[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN], reduce_recv_lengths[0:l2npcols])
      for(int j = send_start; j < send_start + reduce_recv_lengths[i]; j++){
        w[j] = w[j] + q[j];
      }
    }


//--------------------------------------------------------------------
//  Exchange piece of q with transpose processor:
//--------------------------------------------------------------------
    if( l2npcols != 0 ) {
      if (timeron) timer_start(t_rcomm);
      xmp_sync_image(exch_proc, NULL);
#ifdef USE_GET
      q[0 : send_len] = w[send_start : send_len]:[exch_proc];
#else
      q[0 : send_len]:[exch_proc] = w[send_start : send_len];
#endif
      xmp_sync_image(exch_proc, NULL);

      if (timeron) timer_stop(t_rcomm);
    }else{
#pragma acc parallel loop pcopy(q[0:ROW_ARRAY_LEN], w[0:ROW_ARRAY_LEN])
      for(int j = 0; j < exch_recv_length; j++){
        q[j] = w[j];
      }
    }


//--------------------------------------------------------------------
//  Clear w for reuse...
//--------------------------------------------------------------------
    int j_iter_max = max(lastrow - firstrow + 1, lastcol - firstcol + 1);
#pragma acc parallel loop pcopy(w[0:ROW_ARRAY_LEN])
    for(int j = 0; j < j_iter_max; j++){
      w[j] = 0.0;
    }


//--------------------------------------------------------------------
//  Obtain p.q
//--------------------------------------------------------------------
    d = 0.0;
#pragma acc parallel loop reduction(+:d) pcopy(p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      d = d + p[j] * q[j];
    }

//--------------------------------------------------------------------
//  Obtain d with a sum-reduce
//--------------------------------------------------------------------
#ifdef USE_REDUCTION_DIRECTIVE
    if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:d) on subproc
    if (timeron) timer_stop(t_rcomm);
#else
    for(int i = 0; i < l2npcols; i++){
      if (timeron) timer_start(t_rcomm);
      xmp_sync_image(reduce_exch_proc[i], NULL);
      tmp:[reduce_exch_proc[i]] = d;
      xmp_sync_image(reduce_exch_proc[i], NULL);
      if (timeron) timer_stop(t_rcomm);

      d = d + tmp;
    }
#endif


//--------------------------------------------------------------------
//  Obtain alpha = rho / (p.q)
//--------------------------------------------------------------------
    alpha = rho / d;

//--------------------------------------------------------------------
//  Save a temporary of rho
//--------------------------------------------------------------------
    rho0 = rho;

//--------------------------------------------------------------------
//  Obtain z = z + alpha*p
//  and    r = r - alpha*q
//--------------------------------------------------------------------
#pragma acc parallel loop pcopy(z[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN], p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      z[j] = z[j] + alpha * p[j];
      r[j] = r[j] - alpha * q[j];
    }

//--------------------------------------------------------------------
//  rho = r.r
//  Now, obtain the norm of r: First, sum squares of r elements locally...
//--------------------------------------------------------------------
    rho = 0.0;
#pragma acc parallel loop reduction(+:rho) pcopy(r[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      rho = rho + r[j] * r[j];
    }

//--------------------------------------------------------------------
//  Obtain rho with a sum-reduce
//--------------------------------------------------------------------
#ifdef USE_REDUCTION_DIRECTIVE
    if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:rho) on subproc
    if (timeron) timer_stop(t_rcomm);
#else
    for(int i = 0; i < l2npcols; i++){
      if (timeron) timer_start(t_rcomm);
      xmp_sync_image(reduce_exch_proc[i], NULL);
      tmp:[reduce_exch_proc[i]] = rho;
      xmp_sync_image(reduce_exch_proc[i], NULL);
      if (timeron) timer_stop(t_rcomm);

      rho = rho + tmp;
    }
#endif

//--------------------------------------------------------------------
//  Obtain beta:
//--------------------------------------------------------------------
    beta = rho / rho0;

//--------------------------------------------------------------------
//  p = r + beta*p
//--------------------------------------------------------------------
#pragma acc parallel loop pcopy(p[0:ROW_ARRAY_LEN],r[0:ROW_ARRAY_LEN])
    for(int j = 0; j < lastcol - firstcol + 1; j++){
      p[j] = r[j] + beta * p[j];
    }
  } // end of do cgit=1,cgitmax

//--------------------------------------------------------------------
//  Compute residual norm explicitly:  ||r|| = ||x - A.z||
//  First, form A.z
//  The partition submatrix-vector multiply
//--------------------------------------------------------------------
#pragma acc parallel loop gang pcopy(a[0:nz], z[0:ROW_ARRAY_LEN], colidx[0:nz], rowstr[0:na+1], w[0:ROW_ARRAY_LEN])
  for(int j = 0; j < lastrow - firstrow + 1; j++){
    double sum = 0.0;
    int rowstr_j = rowstr[j];
    int rowstr_j1 = rowstr[j+1];
#pragma acc loop vector reduction(+:sum)
    for(int k=rowstr_j; k < rowstr_j1; k++){
      sum = sum + a[k]*z[colidx[k]];
    }
    w[j] = sum;
  }



//--------------------------------------------------------------------
//  Sum the partition submatrix-vec A.z's across rows
//--------------------------------------------------------------------
  for(int i = l2npcols - 1; i >= 0; i--){
    if (timeron) timer_start(t_rcomm);
    int reduce_proc = reduce_exch_proc[i];
    xmp_sync_image(reduce_proc, NULL);
#ifdef USE_GET
    int reduce_start = reduce_recv_starts[i];
    int reduce_length = reduce_recv_lengths[i];
    r[reduce_start : reduce_length] = w[reduce_start : reduce_length]:[reduce_proc];
#else
    int reduce_start = reduce_send_starts[i];
    int reduce_length = reduce_send_lengths[i];
    r[reduce_start : reduce_length]:[reduce_proc] = w[reduce_start : reduce_length];
#endif
    xmp_sync_image(reduce_proc, NULL);
    if (timeron) timer_stop(t_rcomm);

#pragma acc parallel loop pcopy(w[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN], reduce_recv_lengths[0:l2npcols])
    for(int j = send_start; j < send_start + reduce_recv_lengths[i]; j++){
      w[j] = w[j] + r[j];
    }
  }


//--------------------------------------------------------------------
//  Exchange piece of q with transpose processor:
//--------------------------------------------------------------------
  if( l2npcols != 0 ){
    if (timeron) timer_start(t_rcomm);
    xmp_sync_image(exch_proc, NULL);
#ifdef USE_GET
    r[0 : send_len] = w[send_start : send_len]:[exch_proc];
#else
    r[0 : send_len]:[exch_proc] = w[send_start : send_len];
#endif
    xmp_sync_image(exch_proc, NULL);
    if (timeron) timer_stop(t_rcomm);
  }else{
#pragma acc parallel loop pcopy(r[0:ROW_ARRAY_LEN], w[0:ROW_ARRAY_LEN])
    for(int j = 0; j < exch_recv_length; j++){
      r[j] = w[j];
    }
  }

//--------------------------------------------------------------------
//  At this point, r contains A.z
//--------------------------------------------------------------------
  sum = 0.0;
#pragma acc parallel loop private(d) reduction(+:sum) pcopy(x[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN])
  for(int j = 0; j < lastcol - firstcol + 1; j++){
    double d = x[j] - r[j];
    sum = sum + d * d;
  }

//--------------------------------------------------------------------
//  Obtain d with a sum-reduce
//--------------------------------------------------------------------
#ifdef USE_REDUCTION_DIRECTIVE
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:sum) on subproc
  if (timeron) timer_stop(t_rcomm);
#else
  for(int i = 0; i < l2npcols; i++){
    if (timeron) timer_start(t_rcomm);
    xmp_sync_image(reduce_exch_proc[i], NULL);
    tmp:[reduce_exch_proc[i]] = sum;
    xmp_sync_image(reduce_exch_proc[i], NULL);
    if (timeron) timer_stop(t_rcomm);

    sum = sum + tmp;
  }
#endif


  if( me == root ) *rnorm = sqrt(sum);
  } //end of acc data

  if (timeron) timer_stop(t_conjg);

} // end of routine conj_grad


void makea(int n, int nz, double a[], int colidx[], int rowstr[], int nonzer,
           int firstrow, int lastrow, int firstcol, int lastcol,
           double rcond, int arow[], int acol[], double aelt[], double v[], int iv[],
           double shift)
{
//--------------------------------------------------------------------
//      nonzer is approximately  (int(sqrt(nnza /n)));
//--------------------------------------------------------------------
  int nnza = 0, iouter;
  double size = 1.0;
  double ratio = pow(rcond, 1.0 / (double)(n));

//--------------------------------------------------------------------
//  Initialize iv(n+1 .. 2n) to zero.
//  Used by sprnvc to mark nonzero positions
//--------------------------------------------------------------------
  for(int i = 0; i < n; i++){
    iv[n + i] = 0;
  }
  for(iouter = 0; iouter < n; iouter++){
    int nzv = nonzer;
    sprnvc(n, nzv, v, colidx, &iv[0], &iv[n]);
    vecset(n, v, colidx, &nzv, iouter, 0.5);
    for(int ivelt = 0; ivelt < nzv; ivelt++){
      int jcol = colidx[ivelt];
      if (jcol >= firstcol && jcol <= lastcol){
        double scale = size * v[ivelt];
        for(int ivelt1 = 0; ivelt1 < nzv; ivelt1++){
          int irow = colidx[ivelt1];
          if (irow >= firstrow && irow <= lastrow){
            if (nnza >= nz) goto continue9999;
            acol[nnza] = jcol;
            arow[nnza] = irow;
            aelt[nnza] = v[ivelt1] * scale;
            nnza = nnza + 1;
          }
        }
      }
    }
    size = size * ratio;
  }

//--------------------------------------------------------------------
//       ... add the identity * rcond to the generated matrix to bound
//           the smallest eigenvalue from below by rcond
//--------------------------------------------------------------------
  for(int i = firstrow; i <= lastrow; i++){
    if (i >= firstcol && i <= lastcol){
      iouter = n + i;
      if (nnza >= nz) goto continue9999;
      acol[nnza] = i;
      arow[nnza] = i;
      aelt[nnza] = rcond - shift;
      nnza = nnza + 1;
    }
  }

//--------------------------------------------------------------------
//       ... make the sparse matrix from list of elements with duplicates
//           (v and iv are used as  workspace)
//--------------------------------------------------------------------
  sparse(a, colidx, rowstr, n, arow, acol, aelt,
         firstrow, lastrow,
         v, &iv[0], &iv[n], nnza );
  return;

 continue9999:
  printf("Space for matrix elements exceeded in makea\n");
  int nzmax = -1;
  printf("nnza, nzmax = %d %d\n", nnza, nzmax);
  printf(" iouter = %d\n", iouter + 1);
}


void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[],
            double aelt[], int firstrow, int lastrow, double x[], bool mark[],
            int nzloc[], int nnza)
{
//--------------------------------------------------------------------
//       rows range from firstrow to lastrow
//       the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
//--------------------------------------------------------------------
//--------------------------------------------------
//       generate a sparse matrix from a list of
//       [col, row, element] tri
//--------------------------------------------------
//--------------------------------------------------------------------
//    how many rows of result
//--------------------------------------------------------------------
  int nrows = lastrow - firstrow + 1;

//--------------------------------------------------------------------
//     ...count the number of triples in each row
//--------------------------------------------------------------------
  for(int j = 0; j < n; j++){
    rowstr[j] = 0;
    mark[j] = false;
  }
  rowstr[n] = 0;

  for(int nza = 0; nza < nnza; nza++){
    int j = arow[nza] - firstrow + 1;
    rowstr[j] = rowstr[j] + 1;
  }

  rowstr[0] = 0;
  for(int j = 0; j < nrows; j++){
    rowstr[j+1] = rowstr[j+1] + rowstr[j];
  }

//--------------------------------------------------------------------
//     ... rowstr(j) now is the location of the first nonzero
//           of row j of a
//--------------------------------------------------------------------


//--------------------------------------------------------------------
//     ... do a bucket sort of the triples on the row index
//--------------------------------------------------------------------
  for(int nza = 0; nza < nnza; nza++){
    int j = arow[nza] - firstrow;
    int k = rowstr[j];
    a[k] = aelt[nza];
    colidx[k] = acol[nza];
    rowstr[j] = rowstr[j] + 1;
  }

//--------------------------------------------------------------------
//       ... rowstr(j) now points to the first element of row j+1
//--------------------------------------------------------------------
  for(int j = nrows; j > 0; j--){
    rowstr[j] = rowstr[j - 1];
  }
  rowstr[0] = 0;

//--------------------------------------------------------------------
//       ... generate the actual output rows by adding elements
//--------------------------------------------------------------------
  int nza = 0;
  for(int i = 0; i < n; i++){
    x[i] = 0.0;
    mark[i] = false;
  }

  int jajp1 = rowstr[0];
  for(int j = 0; j < nrows; j++){
    int nzrow = 0;

//--------------------------------------------------------------------
//          ...loop over the jth row of a
//--------------------------------------------------------------------
    for(int k = jajp1; k < rowstr[j + 1]; k++){
      int i = colidx[k];
      x[i] = x[i] + a[k];
      if((! mark[i]) && (x[i] != 0.0)){
        mark[i] = true;
        nzloc[nzrow] = i;
        nzrow = nzrow + 1;
      }
    }

//--------------------------------------------------------------------
//          ... extract the nonzeros of this row
//--------------------------------------------------------------------
    for(int k = 0; k < nzrow; k++){
      int i = nzloc[k];
      mark[i] = false;
      double xi = x[i];
      x[i] = 0.0;
      if (xi != 0.0) {
        a[nza] = xi;
        colidx[nza] = i;
        nza = nza + 1;
      }
    }
    jajp1 = rowstr[j + 1];
    rowstr[j + 1] = nza + rowstr[0];
  }
}


void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], bool mark[])
{
//--------------------------------------------------------------------
//       generate a sparse n-vector (v, iv)
//       having nzv nonzeros
//
//       mark(i) is set to 1 if position i is nonzero.
//       mark is all zero on entry and is reset to all zero before exit
//       this corrects a performance bug found by John G. Lewis, caused by
//       reinitialization of mark on every one of the n calls to sprnvc
//--------------------------------------------------------------------
  int nzv = 0;
  int nzrow = 0;
  int nn1 = 1;
  do{
    nn1 = 2 * nn1;
  }while(nn1 < n);

//--------------------------------------------------------------------
//    nn1 is the smallest power of two not less than n
//--------------------------------------------------------------------
  while(nzv < nz){
    double vecelt = randlc(&tran, &amult);

//--------------------------------------------------------------------
//   generate an integer between 1 and n in a portable manner
//--------------------------------------------------------------------
    double vecloc = randlc(&tran, &amult);
    int i = icnvrt(vecloc, nn1);
    if(i >= n) continue;

//--------------------------------------------------------------------
//  was this integer generated already?
//--------------------------------------------------------------------
    if(! mark[i]){
      mark[i] = true;
      nzloc[nzrow] = i;
      nzrow = nzrow + 1;
      v[nzv] = vecelt;
      iv[nzv] = i;
      nzv = nzv + 1;
    }
  }

  for(int ii = 0; ii < nzrow; ii++){
    int i = nzloc[ii];
    mark[i] = false;
  }
}


int icnvrt(double x, int ipwr2)
{
//--------------------------------------------------------------------
//    scale a double precision number x in (0,1) by a power of 2 and chop it
//--------------------------------------------------------------------
  return (int)(ipwr2 * x);
}


void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
//--------------------------------------------------------------------
//       set ith element of sparse vector (v, iv) with
//       nzv nonzeros to val
//--------------------------------------------------------------------
  bool set = false;

  for(int k = 0; k < *nzv; k++){
    if(iv[k] == i){
      v[k] = val;
      set = true;
    }
  }

  if(! set){
    v[*nzv] = val;
    iv[*nzv] = i;
    *nzv = *nzv + 1;
  }
}


void *alloc_mem(size_t elmtsize, int length)
{
  void *p = malloc(elmtsize * length);
  if(p != NULL) return p;

  printf("Error: cannot allocate memory\n");
  exit(1);
}

void free_mem(void *p)
{
  free(p);
}
