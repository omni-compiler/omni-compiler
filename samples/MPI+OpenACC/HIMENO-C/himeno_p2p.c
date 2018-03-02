/********************************************************************

 This benchmark test program is measuring a cpu performance
 of floating point operation by a Poisson equation solver.

 If you have any question, please ask me via email.
 written by Ryutaro HIMENO, November 26, 2001.
 Version 3.0
 ----------------------------------------------
 Ryutaro Himeno, Dr. of Eng.
 Head of Computer Information Division,
 RIKEN (The Institute of Pysical and Chemical Research)
 Email : himeno@postman.riken.go.jp
 ---------------------------------------------------------------
 You can adjust the size of this benchmark code to fit your target
 computer. In that case, please chose following sets of
 (mimax,mjmax,mkmax):
 small : 33,33,65
 small : 65,65,129
 midium: 129,129,257
 large : 257,257,513
 ext.large: 513,513,1025
 This program is to measure a computer performance in MFLOPS
 by using a kernel which appears in a linear solver of pressure
 Poisson eq. which appears in an incompressible Navier-Stokes solver.
 A point-Jacobi method is employed in this solver as this method can 
 be easyly vectrized and be parallelized.
 ------------------
 Finite-difference method, curvilinear coodinate system
 Vectorizable and parallelizable on each grid point
 No. of grid points : imax x jmax x kmax including boundaries
 ------------------
 A,B,C:coefficient matrix, wrk1: source term of Poisson equation
 wrk2 : working area, OMEGA : relaxation parameter
 BND:control variable for boundaries and objects ( = 0 or 1)
 P: pressure
********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "param.h"

float jacobi(int);
int initmax(int,int,int);
void initmt(int,int);
void initcomm(int,int,int);
void sendp(int,int,int);
void sendp1(void);
void sendp2(void);
void sendp3(void);

double fflop(int,int,int);
double mflops(int,double,double);
double gettime(void);

static float  p[MIMAX][MJMAX][MKMAX];
static float  a[4][MIMAX][MJMAX][MKMAX],
              b[3][MIMAX][MJMAX][MKMAX],
              c[3][MIMAX][MJMAX][MKMAX];
static float  bnd[MIMAX][MJMAX][MKMAX];
static float  wrk1[MIMAX][MJMAX][MKMAX],
              wrk2[MIMAX][MJMAX][MKMAX];
static float omega;
static int npe,id;

static int ndx,ndy,ndz;
static int imax,jmax,kmax;
static int ndims=3,iop[3];
static int npx[2],npy[2],npz[2];
MPI_Comm     mpi_comm_cart;
MPI_Datatype ijvec,ikvec,jkvec;

static float *sendp2_lo_sendbuf;
static float *sendp2_lo_recvbuf;
static float *sendp2_hi_sendbuf;
static float *sendp2_hi_recvbuf;
double halo_time = 0, halo_time0, ave_halo_time, max_halo_time;
#define LOOP_TIMES 100

int
main(int argc,char *argv[])
{
  int    i,j,k,nn;
  int    mx,my,mz,it;
  float  gosa;
  double cpu,cpu0,cpu1,flop,target;

  target= 60.0;
  omega= 0.8;
  mx= MX0-1;
  my= MY0-1;
  mz= MZ0-1;
  ndx= NDX0;
  ndy= NDY0;
  ndz= NDZ0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &npe);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  int    namelen;
  char   processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name,&namelen);
  fprintf(stderr, "[%d] %s\n", id, processor_name);

  initcomm(ndx,ndy,ndz);
  it= initmax(mx,my,mz);

  /*
   *    Initializing matrixes
   */
  initmt(mx,it);

  float *sendp2_buf = (float*)malloc(MIMAX*MKMAX*sizeof(float)*4);
  sendp2_lo_sendbuf = &sendp2_buf[MIMAX*MKMAX*0];
  sendp2_lo_recvbuf = &sendp2_buf[MIMAX*MKMAX*1];
  sendp2_hi_sendbuf = &sendp2_buf[MIMAX*MKMAX*2];
  sendp2_hi_recvbuf = &sendp2_buf[MIMAX*MKMAX*3];
#pragma acc enter data create(sendp2_buf[0:MIMAX*MKMAX*4])

  if(id==0){
    printf("Sequential version array size\n");
    printf(" mimax = %d mjmax = %d mkmax = %d\n",MX0,MY0,MZ0);
    printf("Parallel version array size\n");
    printf(" mimax = %d mjmax = %d mkmax = %d\n",MIMAX,MJMAX,MKMAX);
    printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);
    printf("I-decomp = %d J-decomp = %d K-decomp =%d\n",ndx,ndy,ndz);
  }

  nn= 3;
  if(id==0){
    printf(" Start rehearsal measurement process.\n");
    printf(" Measure the performance in %d times.\n\n",nn);
  }

#pragma acc data copyin(p, bnd, wrk1, wrk2, a, b, c) present(sendp2_buf[0:MIMAX*MKMAX*4])
  {
  MPI_Barrier(MPI_COMM_WORLD);
  cpu0= gettime();
  gosa= jacobi(nn);
  cpu1= gettime();
  cpu = cpu1 - cpu0;

  MPI_Allreduce(MPI_IN_PLACE,
                &cpu,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD);

  flop= fflop(mz,my,mx);

  if(id == 0){
    printf(" MFLOPS: %f time(s): %f %e\n\n",
           mflops(nn,cpu,flop),cpu,gosa);
  }

  nn= (int)(target/(cpu/3.0));
  nn= LOOP_TIMES;
  halo_time = 0.0;
  if(id == 0){
    printf(" Now, start the actual measurement process.\n");
    printf(" The loop will be excuted in %d times\n",nn);
    printf(" This will take about one minute.\n");
    printf(" Wait for a while\n\n");
  }

  /*
   *    Start measuring
   */
  MPI_Barrier(MPI_COMM_WORLD);
  cpu0= gettime();
  gosa= jacobi(nn);
  cpu1= gettime();
  cpu = cpu1 - cpu0;

  MPI_Allreduce(MPI_IN_PLACE,
                &cpu,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD);

  MPI_Allreduce(&halo_time,
                &max_halo_time,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                MPI_COMM_WORLD);

  MPI_Allreduce(&halo_time,
                &ave_halo_time,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD);
  ave_halo_time /= npe;
  }//end of acc data

  if(id == 0){
    printf("cpu : %f sec. halo(AVE.) %f sec. halo(MAX) %f sec.\n", cpu, ave_halo_time, max_halo_time);
    printf("Loop executed for %d times\n",nn);
    printf("Gosa : %e \n",gosa);
    printf("MFLOPS measured : %f\n",mflops(nn,cpu,flop));
    printf("Score based on Pentium III 600MHz : %f\n",
           mflops(nn,cpu,flop)/82.84);
  }

  free(sendp2_buf);

  MPI_Finalize();

  return (0);
}

double
fflop(int mx,int my, int mz)
{
  return((double)(mz-2)*(double)(my-2)*(double)(mx-2)*34.0);
}

double
mflops(int nn,double cpu,double flop)
{
  return(flop/cpu*1.e-6*(double)nn);
}

void
initmt(int mx,int it)
{
  int i,j,k;

  for(i=0 ; i<MIMAX ; ++i)
    for(j=0 ; j<MJMAX ; ++j)
      for(k=0 ; k<MKMAX ; ++k){
        a[0][i][j][k]=0.0;
        a[1][i][j][k]=0.0;
        a[2][i][j][k]=0.0;
        a[3][i][j][k]=0.0;
        b[0][i][j][k]=0.0;
        b[1][i][j][k]=0.0;
        b[2][i][j][k]=0.0;
        c[0][i][j][k]=0.0;
        c[1][i][j][k]=0.0;
        c[2][i][j][k]=0.0;
        p[i][j][k]=0.0;
        wrk1[i][j][k]=0.0;
        wrk2[i][j][k]=0.0;
        bnd[i][j][k]=0.0;
      }

  for(i=0 ; i<imax ; ++i)
    for(j=0 ; j<jmax ; ++j)
      for(k=0 ; k<kmax ; ++k){
        a[0][i][j][k]=1.0;
        a[1][i][j][k]=1.0;
        a[2][i][j][k]=1.0;
        a[3][i][j][k]=1.0/6.0;
        b[0][i][j][k]=0.0;
        b[1][i][j][k]=0.0;
        b[2][i][j][k]=0.0;
        c[0][i][j][k]=1.0;
        c[1][i][j][k]=1.0;
        c[2][i][j][k]=1.0;
        p[i][j][k]=(float)((i+it)*(i+it))/(float)((mx-1)*(mx-1));
        wrk1[i][j][k]=0.0;
        wrk2[i][j][k]=0.0;
        bnd[i][j][k]=1.0;
      }
}

float
jacobi(int nn)
{
  int i,j,k,n;
  float gosa,s0,ss;

#pragma acc data present(p, bnd, wrk1, wrk2, a, b, c) create(gosa)
  for(n=0 ; n<nn ; ++n){
    gosa = 0.0;
#pragma acc update device(gosa)

#pragma acc parallel loop firstprivate(omega, imax, jmax, kmax) reduction(+:gosa) collapse(2) gang vector_length(64) async
    for(i=1 ; i<imax-1 ; ++i){
      for(j=1 ; j<jmax-1 ; ++j){
#pragma acc loop vector reduction(+:gosa) private(s0, ss)
        for(k=1 ; k<kmax-1 ; ++k){
          s0 = a[0][i][j][k] * p[i+1][j  ][k  ]
             + a[1][i][j][k] * p[i  ][j+1][k  ]
             + a[2][i][j][k] * p[i  ][j  ][k+1]
             + b[0][i][j][k] * ( p[i+1][j+1][k  ] - p[i+1][j-1][k  ]
                               - p[i-1][j+1][k  ] + p[i-1][j-1][k  ] )
             + b[1][i][j][k] * ( p[i  ][j+1][k+1] - p[i  ][j-1][k+1]
                               - p[i  ][j+1][k-1] + p[i  ][j-1][k-1] )
             + b[2][i][j][k] * ( p[i+1][j  ][k+1] - p[i-1][j  ][k+1]
                               - p[i+1][j  ][k-1] + p[i-1][j  ][k-1] )
             + c[0][i][j][k] * p[i-1][j  ][k  ]
             + c[1][i][j][k] * p[i  ][j-1][k  ]
             + c[2][i][j][k] * p[i  ][j  ][k-1]
             + wrk1[i][j][k];
          ss = ( s0 * a[3][i][j][k] - p[i][j][k] ) * bnd[i][j][k];
          gosa += ss*ss;

          wrk2[i][j][k] = p[i][j][k] + omega * ss;
        }
      }
    }

#pragma acc parallel loop firstprivate(imax, jmax, kmax) collapse(2) gang vector_length(64) async
    for(i=1 ; i<imax-1 ; ++i){
      for(j=1 ; j<jmax-1 ; ++j){
#pragma acc loop vector
        for(k=1 ; k<kmax-1 ; ++k){
          p[i][j][k] = wrk2[i][j][k];
        }
      }
    }

#pragma acc wait

    halo_time0 = gettime();
    sendp(ndx,ndy,ndz);
    halo_time += gettime() - halo_time0;

#pragma acc update host(gosa)
    MPI_Allreduce(MPI_IN_PLACE,
                  &gosa,
                  1,
                  MPI_FLOAT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
  } /* end n loop */

  return(gosa);
}

void
initcomm(int ndx,int ndy,int ndz)
{
  int  i,j,k,tmp;
  int  ipd[3],idm[3],ir;
  MPI_Comm  icomm;

  if(ndx*ndy*ndz != npe){
    if(id==0){
      printf("Invalid number of PE\n");
      printf("Please check partitioning pattern or number of PE\n");
    }
    MPI_Finalize();
    exit(0);
  }

  icomm= MPI_COMM_WORLD;

  idm[0]= ndx;
  idm[1]= ndy;
  idm[2]= ndz;

  ipd[0]= 0;
  ipd[1]= 0;
  ipd[2]= 0;
  ir= 0;


  MPI_Cart_create(icomm,
                  ndims,
                  idm,
                  ipd,
                  ir,
                  &mpi_comm_cart);
  MPI_Cart_get(mpi_comm_cart,
               ndims,
               idm,
               ipd,
               iop);

  if(ndz > 1){
    MPI_Cart_shift(mpi_comm_cart,
                   2,
                   1,
                   &npz[0],
                   &npz[1]);
  }
  if(ndy > 1){
    MPI_Cart_shift(mpi_comm_cart,
                   1,
                   1,
                   &npy[0],
                   &npy[1]);
  }
  if(ndx > 1){
    MPI_Cart_shift(mpi_comm_cart,
                   0,
                   1,
                   &npx[0],
                   &npx[1]);
  }

}

int
initmax(int mx,int my,int mz)
{
  int  i,tmp,it;
  int  mx1[NDX0+1],my1[NDY0+1],mz1[NDZ0+1];
  int  mx2[NDX0+1],my2[NDY0+1],mz2[NDZ0+1];

  tmp= mx/ndx;
  mx1[0]= 0;
  for(i=1;i<=ndx;i++){
    if(i <= mx%ndx)
      mx1[i]= mx1[i-1] + tmp + 1;
    else
      mx1[i]= mx1[i-1] + tmp;
  }
  tmp= my/ndy;
  my1[0]= 0;
  for(i=1;i<=ndy;i++){
    if(i <= my%ndy)
      my1[i]= my1[i-1] + tmp + 1;
    else
      my1[i]= my1[i-1] + tmp;
  }
  tmp= mz/ndz;
  mz1[0]= 0;
  for(i=1;i<=ndz;i++){
    if(i <= mz%ndz)
      mz1[i]= mz1[i-1] + tmp + 1;
    else
      mz1[i]= mz1[i-1] + tmp;
  }

  for(i=0 ; i<ndx ; i++){
    mx2[i] = mx1[i+1] - mx1[i];
    if(i != 0)     mx2[i] = mx2[i] + 1;
    if(i != ndx-1) mx2[i] = mx2[i] + 1;
  }
  for(i=0 ; i<ndy ; i++){
    my2[i] = my1[i+1] - my1[i];
    if(i != 0)     my2[i] = my2[i] + 1;
    if(i != ndy-1) my2[i] = my2[i] + 1;
  }
  for(i=0 ; i<ndz ; i++){
    mz2[i] = mz1[i+1] - mz1[i];
    if(i != 0)     mz2[i] = mz2[i] + 1;
    if(i != ndz-1) mz2[i] = mz2[i] + 1;
  }

  imax = mx2[iop[0]];
  jmax = my2[iop[1]];
  kmax = mz2[iop[2]];

  if(iop[0] == 0)
    it= mx1[iop[0]];
  else
    it= mx1[iop[0]] - 1;

  if(ndx > 1){
    MPI_Type_contiguous(jmax*MKMAX, MPI_FLOAT, &jkvec);
    MPI_Type_commit(&jkvec);
  }
  if(ndy > 1){
    MPI_Type_contiguous(imax*kmax, MPI_FLOAT, &ikvec);
    MPI_Type_commit(&ikvec);
  }
  if(ndz > 1){
    MPI_Type_vector(imax*jmax,
                    1,
                    MKMAX,
                    MPI_FLOAT,
                    &ijvec);
    MPI_Type_commit(&ijvec);
  }

  return(it);
}

void
sendp(int ndx,int ndy,int ndz)
{
  if(ndx > 1)
    sendp1();

  if(ndy > 1)
    sendp2();

  if(ndz > 1)
    sendp3();

}

void
sendp3()
{
  printf("Z-dim halo exchange is not implemented\n");
  MPI_Finalize();
  exit(0);
}

void
sendp2_pack()
{
  int i,k;

#pragma acc data present(sendp2_lo_sendbuf[0:imax*kmax], sendp2_hi_sendbuf[0:imax*kmax])
  {
  if(npy[1] != MPI_PROC_NULL){
#pragma acc parallel loop vector_length(128) async
    for(i = 0; i < imax; i++){
#pragma acc loop
      for(k = 0; k < kmax; k++){
        sendp2_lo_sendbuf[i*kmax + k] = p[i][jmax-2][k];
      }
    }
  }

  if(npy[0] != MPI_PROC_NULL){
#pragma acc parallel loop vector_length(128) async
    for(i = 0; i < imax; i++){
#pragma acc loop
      for(k = 0; k < kmax; k++){
        sendp2_hi_sendbuf[i*kmax + k] = p[i][1][k];
      }
    }
  }
#pragma acc wait
  }
}

void
sendp2_unpack()
{
  int i,k;

#pragma acc data present(sendp2_lo_recvbuf[0:imax*kmax], sendp2_hi_recvbuf[0:imax*kmax])
  {
  if(npy[0] != MPI_PROC_NULL){
#pragma acc parallel loop vector_length(128) async
    for(i = 0; i < imax; i++){
#pragma acc loop
      for(k = 0; k < kmax; k++){
        p[i][0][k] = sendp2_lo_recvbuf[i*kmax + k];
      }
    }
  }
  if(npy[1] != MPI_PROC_NULL){
#pragma acc parallel loop vector_length(128) async
    for(i = 0; i < imax; i++){
#pragma acc loop
      for(k = 0; k < kmax; k++){
        p[i][jmax-1][k] = sendp2_hi_recvbuf[i*kmax + k];
      }
    }
  }
#pragma acc wait
  }
}

void
sendp2()
{
  static MPI_Request req[4];
  static int flag = 0;

  if(! flag){
#pragma acc data present(sendp2_lo_recvbuf[0:imax*kmax], sendp2_lo_sendbuf[0:imax*kmax], sendp2_hi_sendbuf[0:imax*kmax], sendp2_hi_recvbuf[0:imax*kmax])
#pragma acc host_data use_device(sendp2_lo_sendbuf, sendp2_lo_recvbuf, sendp2_hi_sendbuf, sendp2_hi_recvbuf)
  {
  MPI_Recv_init(sendp2_hi_recvbuf,
            1,
            ikvec,
            npy[1],
            1,
            mpi_comm_cart,
            req+2);
  MPI_Recv_init(sendp2_lo_recvbuf,
            1,
            ikvec,
            npy[0],
            2,
            mpi_comm_cart,
            req+0);
  MPI_Send_init(sendp2_hi_sendbuf,
            1,
            ikvec,
            npy[0],
            1,
            mpi_comm_cart,
            req+3);
  MPI_Send_init(sendp2_lo_sendbuf,
            1,
            ikvec,
            npy[1],
            2,
            mpi_comm_cart,
            req+1);
  flag = 1;
  }
  }

  sendp2_pack();
  MPI_Startall(4, req);
  MPI_Waitall(4,
              req,
              MPI_STATUSES_IGNORE);
  sendp2_unpack();
}

void
sendp1()
{
  static MPI_Request req[4];
  static int flag = 0;

  if(! flag){
#pragma acc data present(p)
#pragma acc host_data use_device(p)
  {
  MPI_Recv_init(&p[imax-1][0][0],
            1,
            jkvec,
            npx[1],
            1,
            mpi_comm_cart,
            req+2);
  MPI_Recv_init(&p[0][0][0],
            1,
            jkvec,
            npx[0],
            2,
            mpi_comm_cart,
            req+0);
  MPI_Send_init(&p[1][0][0],
            1,
            jkvec,
            npx[0],
            1,
            mpi_comm_cart,
            req+3);
  MPI_Send_init(&p[imax-2][0][0],
            1,
            jkvec,
            npx[1],
            2,
            mpi_comm_cart,
            req+1);
  flag = 1;
  }
  }

  MPI_Startall(4, req);
  MPI_Waitall(4,
              req,
              MPI_STATUSES_IGNORE);
}

double gettime()
{
  return MPI_Wtime();
}
