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

#define XACC_COMM acc

#include <xmp.h>
#pragma xmp nodes proc(num_proc_cols, num_proc_rows)
#pragma xmp nodes subproc(num_proc_cols) = proc(:,*)
#pragma xmp template t(0:na-1,0:na-1)
#pragma xmp distribute t(block, block) onto proc

static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
static int icnvrt(double x, int ipwr2);
static void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[],
                   double aelt[], int firstrow, int lastrow, double x[], bool mark[],
                   int nzloc[], int nnza);
static void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], bool mark[]);
static void setup_proc_info( int num_procs_, int num_proc_rows_, int num_proc_cols_ );
static void setup_submatrix_info(double [na], double [na]);
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
                      double * restrict rnorm);

static void initialize_mpi();
static void *alloc_mem(size_t elmtsize, int length);
static void free_mem(void *);

double amult, tran;
int firstrow, lastrow, firstcol, lastcol;
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

  double * restrict v, * restrict aelt, * restrict a;
  v = alloc_mem(sizeof(double), na+1);
  aelt = alloc_mem(sizeof(double), nz);
  a = alloc_mem(sizeof(double), nz);
  double p[na], x[na], z[na],w[na], q[na], r[na];
#pragma xmp align [i] with t(*,i) :: w
#pragma xmp align [i] with t(i,*) :: p, x, z, q, r

  int i, j, k, it;

  double zeta;
  double rnorm;
  double norm_temp1[2], norm_temp2[2];

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

#pragma xmp task on proc(1,1) nocomm
  {
    printf("\n\n NAS Parallel Benchmarks 3.3 -- CG Benchmark\n\n");
    printf(" Size: %10d\n", na );
    printf(" Iterations: %5d\n", niter );
    printf(" Number of active processes: %5d\n", nprocs);
    printf(" Number of nonzeroes per row: %8d\n", nonzer);
    printf(" Eigenvalue shift: %8.3e\n", shift);
  }


  int naa = na;
  int nzz = nz;

//--------------------------------------------------------------------
//  Set up processor info, such as whether sq num of procs, etc
//--------------------------------------------------------------------
  setup_proc_info( num_procs,
                   num_proc_rows,
                   num_proc_cols );


//--------------------------------------------------------------------
//  Set up partition's submatrix info: firstcol, lastcol, firstrow, lastrow
//--------------------------------------------------------------------
  setup_submatrix_info(x, w);


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
//        values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1
//        values of colidx which are col indexes go from firstcol --> lastcol
//        So:
//        Shift the col index vals from actual (firstcol --> lastcol )
//        to local, i.e., (1 --> lastcol-firstcol+1)
//--------------------------------------------------------------------
   //shift for xmp loop index. rowstr[firstrow-1:lastrow-firstrow+2] <= rowstr[0:lastrow-firstrow+2]
   for(j = lastrow - firstrow + 1; j >= 0; j--){
     rowstr[j + firstrow] = rowstr[j];
   }

#pragma xmp loop on t(*,j)
   for(int j = 0; j < na; j++){
     for(int k = rowstr[j]; k < rowstr[j + 1]; k++){
       colidx[k] = colidx[k] - firstcol;
     }
   }

//--------------------------------------------------------------------
//  set starting vector to (1, 1, .... 1)
//--------------------------------------------------------------------
#pragma xmp loop on t(i,*)
  for(int i = 0; i < na; i++){
    x[i] = 1.0;
  }

  zeta  = 0.0;

#pragma acc data pcopyin(colidx[0:nz], rowstr[0:na+1], a[0:nz], w, q, r, p, x, z)
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
              &rnorm);

//--------------------------------------------------------------------
//  zeta = shift + 1/(x.z)
//  So, first: (x.z)
//  Also, find norm of z
//  So, first: (z.z)
//--------------------------------------------------------------------
    double norm1 = 0.0, norm2 = 0.0;
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:norm1,norm2) pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      norm1 = norm1 + x[j] * z[j];
      norm2 = norm2 + z[j] * z[j];
    }
    norm_temp1[0] = norm1;
    norm_temp1[1] = norm2;

    if (timeron) timer_start(t_ncomm);
#pragma xmp reduction(+:norm_temp1) on subproc
    if (timeron) timer_stop(t_ncomm);

    norm2 = norm_temp1[1];

    norm2 = 1.0 / sqrt(norm2);

//--------------------------------------------------------------------
//  Normalize z to obtain x
//--------------------------------------------------------------------
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      x[j] = norm2 * z[j];
    }
  } // end of do one iteration untimed

//--------------------------------------------------------------------
//  set starting vector to (1, 1, .... 1)
//--------------------------------------------------------------------
//
//  NOTE: a questionable limit on size:  should this be na/num_proc_cols+1 ?
//
#pragma xmp loop on t(i,*)
#pragma acc parallel loop pcopy(x[0:ROW_ARRAY_LEN])
  for(int i = 0; i < na; i++){
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
              &rnorm);

//--------------------------------------------------------------------
//  zeta = shift + 1/(x.z)
//  So, first: (x.z)
//  Also, find norm of z
//  So, first: (z.z)
//--------------------------------------------------------------------
    double norm1 = 0.0, norm2 = 0.0;
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:norm1, norm2) pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      norm1 = norm1 + x[j] * z[j];
      norm2 = norm2 + z[j] * z[j];
    }

    norm_temp1[0] = norm1;
    norm_temp1[1] = norm2;

    if (timeron) timer_start(t_ncomm);
#pragma xmp reduction(+:norm_temp1) on subproc
    if (timeron) timer_stop(t_ncomm);

    norm1 = norm_temp1[0];
    norm2 = norm_temp1[1];

    norm2 = 1.0 / sqrt(norm2);

#pragma xmp task on proc(1,1) nocomm
    {
      zeta = shift + 1.0 / norm1;
      if(it == 1){
        printf("\n   iteration           ||r||                 zeta\n");
      }
      printf("    %5d       %20.14e%20.13f\n", it, rnorm, zeta);
    }

//--------------------------------------------------------------------
//  Normalize z to obtain x
//--------------------------------------------------------------------
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(x[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
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

#pragma xmp task on proc(1,1) nocomm
   {
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

#pragma xmp task on proc(1,1) nocomm
  {
    printf(" nprocs =%6d           minimum     maximum     average\n", nprocs);
    for(int i = 0; i < t_last+2; i++){
      tsum[i] = tsum[i] / nprocs;
      printf(" timer %2d(%8s) :  %10.4f  %10.4f  %10.4f\n", i, t_recs[i], tming[i], tmaxg[i], tsum[i]);
    }
  }
 continue999:
  return 0;
} // end main


void initialize_mpi()
{
  nprocs = xmp_num_nodes();

#pragma xmp task on proc(1,1) nocomm
  {
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
  int i;
  int log2nprocs;
//--------------------------------------------------------------------
//  num_procs must be a power of 2, and num_procs=num_proc_cols*num_proc_rows
//  When num_procs is not square, then num_proc_cols = 2*num_proc_rows
//--------------------------------------------------------------------
//  First, number of procs must be power of two.
//--------------------------------------------------------------------
  if( nprocs != num_procs ){
#pragma xmp task on proc(1,1) nocomm
    {
      printf("Error: num of procs allocated (%d) is not equal to \
compiled number of procs (%d)", nprocs, num_procs);
    }
    exit(1); //   stop
  }


  i = num_proc_cols;
 continue100:
  if( i != 1 && i/2*2 != i ){
#pragma xmp task on proc(1,1) nocomm
    {
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
#pragma xmp task on proc(1,1) nocomm
    {
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

//  npcols = num_proc_cols;
//  nprows = num_proc_rows;

  return;
}


void setup_submatrix_info(double x[na], double w[na])
{
#pragma xmp align x[i] with t(i,*)
#pragma xmp align w[i] with t(*,i)

  firstcol = xmp_array_gcllbound(xmp_desc_of(x), 1);
  lastcol = xmp_array_gclubound(xmp_desc_of(x), 1);
  firstrow = xmp_array_gcllbound(xmp_desc_of(w), 1);
  lastrow = xmp_array_gclubound(xmp_desc_of(w), 1);

//--------------------------------------------------------------------
//  If naa evenly divisible by npcols, then it is evenly divisible
//  by nprows
//--------------------------------------------------------------------

//--------------------------------------------------------------------
//  If naa not evenly divisible by npcols, then first subdivide for nprows
//  and then, if npcols not equal to nprows (i.e., not a sq number of procs),
//  get col subdivisions by dividing by 2 each row subdivision.
//--------------------------------------------------------------------

//--------------------------------------------------------------------
//  Transpose exchange processor
//--------------------------------------------------------------------

//--------------------------------------------------------------------
//  Set up the reduce phase schedules...
//--------------------------------------------------------------------

  return;
}


void conj_grad(int const colidx[restrict],
               int const rowstr[restrict],
               double       x[restrict na],
               double       z[restrict na],
               double const a[restrict],
               double       p[restrict na],
               double       q[restrict na],
               double       r[restrict na],
               double       w[restrict na],
               double * restrict rnorm)
{
#pragma xmp align [i] with t(*,i) :: w
#pragma xmp align [i] with t(i,*) :: p, x, z, q, r
#pragma xmp static_desc :: w, p, x, z, q, r
//--------------------------------------------------------------------
//  Floaging point arrays here are named as in NPB1 spec discussion of
//  CG algorithm
//--------------------------------------------------------------------
  int const cgitmax = 25;
  double d, sum, rho, rho0, alpha, beta;

  if (timeron) timer_start(t_conjg);
#pragma acc data pcopy(colidx[0:nz], rowstr[0:na+1], a[0:nz]) pcopy(q, r, w, p, x, z)
  {
//--------------------------------------------------------------------
//  Initialize the CG algorithm:
//--------------------------------------------------------------------
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN], z[0:ROW_ARRAY_LEN], w[0:ROW_ARRAY_LEN], x[0:ROW_ARRAY_LEN])
  for(int j = 0; j < na; j++){
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
  }
#pragma xmp loop on t(*,j)
#pragma acc parallel loop pcopy(w[0:ROW_ARRAY_LEN])
  for(int j = 0; j < na; j++){
    w[j] = 0.0;
  }

//--------------------------------------------------------------------
//  rho = r.r
//  Now, obtain the norm of r: First, sum squares of r elements locally...
//--------------------------------------------------------------------
  rho = 0.0;
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:rho) pcopy(r[0:ROW_ARRAY_LEN])
  for(int j = 0; j < na; j++){
    rho = rho + r[j] * r[j];
  }

//--------------------------------------------------------------------
//  Exchange and sum with procs identified in reduce_exch_proc
//  (This is equivalent to mpi_allreduce.)
//  Sum the partial sums of rho, leaving rho on all processors
//--------------------------------------------------------------------
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:rho) on subproc
  if (timeron) timer_stop(t_rcomm);

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
#pragma xmp loop on t(*,j)
#pragma acc parallel loop gang pcopy(a[0:nz], p[0:ROW_ARRAY_LEN], colidx[0:nz], rowstr[0:na+1], w[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
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
#pragma xmp reduction(+:w) on subproc(:) XACC_COMM


//--------------------------------------------------------------------
//  Exchange piece of q with transpose processor:
//--------------------------------------------------------------------
    if (timeron) timer_start(t_rcomm);
#pragma xmp gmove XACC_COMM
    q[:] = w[:];
    if (timeron) timer_stop(t_rcomm);


//--------------------------------------------------------------------
//  Clear w for reuse...
//--------------------------------------------------------------------
#pragma xmp loop on t(*,j)
#pragma acc parallel loop pcopy(w[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      w[j] = 0.0;
    }


//--------------------------------------------------------------------
//  Obtain p.q
//--------------------------------------------------------------------
    d = 0.0;
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:d) pcopy(p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      d = d + p[j] * q[j];
    }

//--------------------------------------------------------------------
//  Obtain d with a sum-reduce
//--------------------------------------------------------------------
    if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:d) on subproc
    if (timeron) timer_stop(t_rcomm);


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
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(z[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN], p[0:ROW_ARRAY_LEN], q[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      z[j] = z[j] + alpha * p[j];
      r[j] = r[j] - alpha * q[j];
    }

//--------------------------------------------------------------------
//  rho = r.r
//  Now, obtain the norm of r: First, sum squares of r elements locally...
//--------------------------------------------------------------------
    rho = 0.0;
#pragma xmp loop on t(j,*)
#pragma acc parallel loop reduction(+:rho) pcopy(r[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      rho = rho + r[j]*r[j];
    }

//--------------------------------------------------------------------
//  Obtain rho with a sum-reduce
//--------------------------------------------------------------------
    if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:rho) on subproc
    if (timeron) timer_stop(t_rcomm);

//--------------------------------------------------------------------
//  Obtain beta:
//--------------------------------------------------------------------
    beta = rho / rho0;

//--------------------------------------------------------------------
//  p = r + beta*p
//--------------------------------------------------------------------
#pragma xmp loop on t(j,*)
#pragma acc parallel loop pcopy(p[0:ROW_ARRAY_LEN],r[0:ROW_ARRAY_LEN])
    for(int j = 0; j < na; j++){
      p[j] = r[j] + beta * p[j];
    }
  } // end of do cgit=1,cgitmax

//--------------------------------------------------------------------
//  Compute residual norm explicitly:  ||r|| = ||x - A.z||
//  First, form A.z
//  The partition submatrix-vector multiply
//--------------------------------------------------------------------
#pragma xmp loop on t(*,j)
#pragma acc parallel loop gang pcopy(a[0:nz], z[0:ROW_ARRAY_LEN], colidx[0:nz], rowstr[0:na+1], w[0:ROW_ARRAY_LEN])
  for(int j = 0; j < na; j++){
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
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:w) on subproc(:) XACC_COMM
  if (timeron) timer_stop(t_rcomm);


//--------------------------------------------------------------------
//  Exchange piece of q with transpose processor:
//--------------------------------------------------------------------
  if (timeron) timer_start(t_rcomm);
#pragma xmp gmove XACC_COMM
  r[:] = w[:];
  if (timeron) timer_stop(t_rcomm);


//--------------------------------------------------------------------
//  At this point, r contains A.z
//--------------------------------------------------------------------
  sum = 0.0;
#pragma xmp loop on t(j,*)
#pragma acc parallel loop private(d) reduction(+:sum) pcopy(x[0:ROW_ARRAY_LEN], r[0:ROW_ARRAY_LEN])
  for(int j = 0; j < na; j++){
    double d = x[j] - r[j];
    sum = sum + d * d;
  }

//--------------------------------------------------------------------
//  Obtain d with a sum-reduce
//--------------------------------------------------------------------
  if (timeron) timer_start(t_rcomm);
#pragma xmp reduction(+:sum) on subproc
  if (timeron) timer_stop(t_rcomm);


#pragma xmp task on proc(1,1) nocomm
  *rnorm = sqrt(sum);
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
