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
#include <sys/time.h>
#include <mpi.h>
#include "param.h"
#include <xmp.h>
#include <xmp_api.h> //--- 2020 Fujitsu ---


float jacobi(int);
int initmax(int,int,int);
void initmt();
void sendp(int,int,int);
void sendp1();
void sendp2();
void sendp3();

double fflop(int,int,int);
double mflops(int,double,double);
double time();

static float  p[MIMAX][MJMAX][MKMAX];
static float  a[4][MIMAX][MJMAX][MKMAX],
              b[3][MIMAX][MJMAX][MKMAX],
              c[3][MIMAX][MJMAX][MKMAX];
static float  bnd[MIMAX][MJMAX][MKMAX];
static float  wrk1[MIMAX][MJMAX][MKMAX],
              wrk2[MIMAX][MJMAX][MKMAX];
static float omega;

static int ndx,ndy,ndz;
static int imax,jmax,kmax;

//--- 2020 Fujitsu
//#pragma xmp template t(0:MKMAX-1, 0:MJMAX-1, 0:MIMAX-1)
//#pragma xmp nodes n(NDY, NDX)
//#pragma xmp distribute t(*, block, block) onto n
//#pragma xmp align p[k][j][i] with t(i, j, k)
//#pragma xmp align bnd[k][j][i] with t(i, j, k)
//#pragma xmp align wrk1[k][j][i] with t(i, j, k)
//#pragma xmp align wrk2[k][j][i] with t(i, j, k)
//#pragma xmp align a[*][k][j][i] with t(i, j, k)
//#pragma xmp align b[*][k][j][i] with t(i, j, k)
//#pragma xmp align c[*][k][j][i] with t(i, j, k)
//#pragma xmp shadow p[1][1][0]
//--- 2020 Fujitsu end
double reflect_time = 0, reflect_time0, ave_reflect_time, max_reflect_time;
#define LOOP_TIMES 100

int
main(int argc,char **argv)
{
  //--- 2020 Fujitsu
  //  int    namelen;
  //  char   processor_name[MPI_MAX_PROCESSOR_NAME];
  //#ifdef _XCALABLEMP
  //  MPI_Get_processor_name(processor_name,&namelen);
  //  fprintf(stderr, "[%d] %s\n", xmpc_node_num(), processor_name);
  //#endif
  //--- 2020 Fujitsu end

  int    i,j,k,nn;
  int    mx,my,mz,it;
  float  gosa;
  double cpu,cpu0,cpu1,flop,target;

  //--- 2020 Fujitsu
  int node_dims[2];
  xmp_desc_t n_desc, t_desc;
  xmp_desc_t p_desc, bnd_desc, wrk1_desc, wrk2_desc, a_desc, b_desc, c_desc;
  double *p_p, *bnd_p, *wrk1_p, *wrk2_p, *a_p, *b_p, *c_p;

  xmp_api_init(argc, argv);

  // for xmp nodes n(NDY, NDX)
  node_dims[1] = NDY; node_dims[0] = NDX;
  n_desc = xmp_global_nodes(2, node_dims, TRUE);

  // for template t(0:MKMAX-1, 0:MJMAX-1, 0:MIMAX-1)
  // and distribute t(*, block, block) onto n
  t_desc = xmpc_new_template(p_desc, 3, (long long)MKMAX, (long long)MJMAX, (long long)MIMAX);
  xmp_dist_template_BLOCK(t_desc, 0, 0);
  xmp_dist_template_BLOCK(t_desc, 1, 1);
  xmp_dist_template_BLOCK(t_desc, 2, 2);

  // for xmp align p[k][j][i] with t(i, j, k)
  // and xmp shadow p[1][1][0]
  p_desc = xmpc_new_array(t_desc, XMP_FLOAT, 3, (long long)MKMAX, (long long)MJMAX, (long long)MIMAX);
  xmp_align_array(p_desc, 0, 2, 0);
  xmp_align_array(p_desc, 1, 1, 0);
  xmp_align_array(p_desc, 2, 0, 0);
  xmp_set_shadow(p_desc, 0, 1, 1);
  xmp_set_shadow(p_desc, 1, 1, 1);
  xmp_set_shadow(p_desc, 1, 0, 0);
  xmp_allocate_array(p_desc,(void **)&p_p);

  // xmp align bnd[k][j][i] with t(i, j, k)
  bnd_desc = xmpc_new_array(t_desc, XMP_FLOAT, 3, (long long)MKMAX, (long long)MJMAX, (long long)MIMAX);
  xmp_align_array(bnd_desc, 0, 2, 0);
  xmp_align_array(bnd_desc, 1, 1, 0);
  xmp_align_array(bnd_desc, 2, 0, 0);
  xmp_allocate_array(bnd_desc,(void **)&bnd_p);

  // xmp align wrk1[k][j][i] with t(i, j, k)
  wrk1_desc = xmpc_new_array(t_desc, XMP_FLOAT, 3, (long long)MKMAX, (long long)MJMAX, (long long)MIMAX);
  xmp_align_array(wrk1_desc, 0, 2, 0);
  xmp_align_array(wrk1_desc, 1, 1, 0);
  xmp_align_array(wrk1_desc, 2, 0, 0);
  xmp_allocate_array(wrk1_desc,(void **)&wrk1_p);

  // xmp align wrk2[k][j][i] with t(i, j, k)
  wrk2_desc = xmpc_new_array(t_desc, XMP_FLOAT, 3, (long long)MKMAX, (long long)MJMAX, (long long)MIMAX);
  xmp_align_array(wrk2_desc, 0, 2, 0);
  xmp_align_array(wrk2_desc, 1, 1, 0);
  xmp_align_array(wrk2_desc, 2, 0, 0);
  xmp_allocate_array(wrk2_desc,(void **)&wrk2_p);


  //--- 2020 Fujitsu end


  target= 10.0;
  omega= 0.8;
  mx= MX0-1;
  my= MY0-1;
  mz= MZ0-1;
  ndx= NDX0;
  ndy= NDY0;
  ndz= NDZ0;

  imax= mx;
  jmax= my;
  kmax= mz;

  /*
   *    Initializing matrixes
   */
  initmt();

#pragma xmp task on t(0,0,0)
  {
  printf("Sequential version array size\n");
  printf(" mimax = %d mjmax = %d mkmax = %d\n",MX0,MY0,MZ0);
  printf("Parallel version array size\n");
  printf(" mimax = %d mjmax = %d mkmax = %d\n",MIMAX,MJMAX,MKMAX);
  printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);
  printf("I-decomp = %d J-decomp = %d K-decomp =%d\n",ndx,ndy,ndz);
  }

  nn= 3;
  
#pragma xmp task on t(0,0,0)
  {
  printf(" Start rehearsal measurement process.\n");
  printf(" Measure the performance in %d times.\n\n",nn);
  }

#pragma xmp barrier
#ifdef _XCALABLEMP
  cpu0= xmp_wtime();
#else
  cpu0= time();
#endif
  gosa= jacobi(nn);
#ifdef _XCALABLEMP
  cpu1= xmp_wtime();
#else
  cpu1= time();
#endif
  cpu = cpu1 - cpu0;
#pragma xmp reduction(max: cpu)

  flop= fflop(mz,my,mx);

#pragma xmp task on t(0,0,0)
  printf(" MFLOPS: %f time(s): %f %e\n\n",
	 mflops(nn,cpu,flop),cpu,gosa);

  nn= (int)(target/(cpu/3.0));
  nn = LOOP_TIMES;
  reflect_time = 0.0;
#pragma xmp task on t(0,0,0)
  {
  printf(" Now, start the actual measurement process.\n");
  printf(" The loop will be excuted in %d times\n",nn);
  printf(" This will take about one minute.\n");
  printf(" Wait for a while\n\n");
  }

  /*
   *    Start measuring
   */

#pragma xmp barrier
#ifdef _XCALABLEMP
  cpu0= xmp_wtime();
#else
  cpu0= time();
#endif
  gosa= jacobi(nn);
#ifdef _XCALABLEMP
  cpu1= xmp_wtime();
#else
  cpu1= time();
#endif
  cpu = cpu1 - cpu0;
#pragma xmp reduction(max:cpu)
  max_reflect_time = reflect_time;
  ave_reflect_time = reflect_time;
#pragma xmp reduction(max:max_reflect_time)
#pragma xmp reduction(+:ave_reflect_time)
#ifdef _XCALABLEMP
  ave_reflect_time /= xmp_num_nodes();
#endif

#pragma xmp task on t(0,0,0)
  {
    printf("cpu : %f sec. reflect(AVE.) %f sec. reflect(MAX) %f sec.\n", cpu, ave_reflect_time, max_reflect_time);
  printf("Loop executed for %d times\n",nn);
  printf("Gosa : %e \n",gosa);
  printf("MFLOPS measured : %f\n",mflops(nn,cpu,flop));
  printf("Score based on Pentium III 600MHz : %f\n",
	 mflops(nn,cpu,flop)/82.84);
  }

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
initmt()
{
  int i,j,k;

#pragma xmp loop (k,j,i) on t(k,j,i)
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

#pragma xmp loop (k,j,i) on t(k,j,i)
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
	p[i][j][k]=(float)((i)*(i))/(float)((imax-1)*(imax-1));
        wrk1[i][j][k]=0.0;
        wrk2[i][j][k]=0.0;
        bnd[i][j][k]=1.0;
      }
#pragma xmp reflect(p)
}

float
jacobi(int nn)
{
  int i,j,k,n;
  float gosa,s0,ss;

  for(n=0 ; n<nn ; ++n){
    gosa = 0.0;

#pragma xmp loop (k,j,i) on t(k,j,i) reduction(+:gosa)
    for(i=1 ; i<imax-1 ; ++i)
      for(j=1 ; j<jmax-1 ; ++j)
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
      
#pragma xmp loop (k,j,i) on t(k,j,i)
    for(i=1 ; i<imax-1 ; ++i)
      for(j=1 ; j<jmax-1 ; ++j)
        for(k=1 ; k<kmax-1 ; ++k)
          p[i][j][k] = wrk2[i][j][k];

#ifdef _XCALABLEMP
    reflect_time0 = xmp_wtime();
#endif
#pragma xmp reflect (p)
#ifdef _XCALABLEMP
    reflect_time += xmp_wtime() - reflect_time0;
#endif

#pragma xmp reduction(+:gosa)
  } /* end n loop */
  

  return(gosa);
}

double time()
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
