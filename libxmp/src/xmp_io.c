/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* #define ORIGINAL */
/* #define DEBUG */

/* #define MPI_TYPE_CREATE_RESIZED1  MPI_Type_create_resized  /\* ORIGINAL *\/ */
#define MPI_TYPE_CREATE_RESIZED1  MPI_Type_create_resized1

#ifdef ORIGINAL
#else /* RIST */
#include "xmp.h"
#endif
#include "xmp_constant.h"
#include "xmp_data_struct.h"
#include "xmp_io.h"

/* ------------------------------------------------------------------ */
#ifdef ORIGINAL
#else /* RIST */
extern void _XMP_fatal(char *msg);

static int MPI_Type_create_resized1(MPI_Datatype oldtype,
			     MPI_Aint     lb,
			     MPI_Aint     extent,
			     MPI_Datatype *newtype);
#endif
/* ------------------------------------------------------------------ */
#ifdef ORIGINAL
#else /* RIST */
/* ================================================================== */
/* beginning of inc_xmp_io.c */
#define MIN(a,b)  ( (a)<(b) ? (a) : (b) )
#define MAX(a,b)  ( (a)>(b) ? (a) : (b) )
#define func_m(p, q)  ((q) >= 0 ? -(q)/(p) : ((p) >= 0 ? (-(q)+(p)-1)/(p) : (-(q)-(p)-1)/(p) ))
/* ------------------------------------------------------------------ */
static int _xmp_io_block_cyclic_0
(
 int par_lower /* in */, int par_upper /* in */, int bw /* in */, int cycle /* in */,
 int rp_lb /* in */, int rp_ub /* in */, int step /* in */,
 MPI_Datatype dataType0 /* in */,
 MPI_Datatype *_dataType1 /* out: data type for file view */
)
{
  MPI_Datatype dataType_tmp;
  int continuous_size, space_size, total_size;
  int mpiRet;

  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // get extent of data type
  MPI_Aint tmp1, tmp2; int type_size;
  mpiRet =MPI_Type_get_extent(dataType0, &tmp1, &tmp2);
  if (mpiRet !=  MPI_SUCCESS) { return -1113; }  
  type_size = (int)tmp2;

  printf("_xmp_io_block_cyclic_0: rmyank = %d:  par_lower = %d  par_upper = %d  bw = %d  cycle = %d\n",
	 myrank,par_lower,par_upper,bw,cycle);

  if (bw <= 0){ _XMP_fatal("_xmp_io_block_cyclic_0: block width must be pisitive."); }
  if (cycle == 0){ _XMP_fatal("_xmp_io_block_cyclic_0: cycle must be non-zero."); }
  /* ++++++++++++++++++++++++++++++++++++++++ */
  if (step > 0){
    if (rp_lb > rp_ub){
      return 1;

    }else if(par_upper < rp_lb || rp_ub < par_lower){
      continuous_size = space_size = 0;
      total_size = ((rp_ub-rp_lb)/step + 1) * type_size;

      mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
      if (mpiRet != MPI_SUCCESS) { return 1; }

    }else{
      int lb_tmp = MAX(par_lower, rp_lb);
      int ub_tmp = MIN(par_upper, rp_ub);
      int a = cycle, b = step;
      int ib;
      int z_l = MAX(par_upper,rp_ub) + 1; int ib_l = bw; int x_l; int y_l;
      int z_u = MIN(par_lower,rp_lb) - 1; int ib_u = -1; int x_u; int y_u;
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0, y0;
	{
	  int x, y, z, w, w1; int q, r, tmp; int bb = -b;
	  if(a == 0 || bb == 0){ return 1; }
	  x = a; y = bb;
	  if(x < 0) x = -x;
	  if(y < 0) y = -y;
	  z = 1;
	  w = 0;
	  while( 1 ){
	    q = x/y;
	    r = x - q*y;
	    if( r == 0 ) break;
	    x = y;
	    y = r;
	    tmp = z;
	    z = w;
	    w = tmp - q * w;
	  }
	  w = w - (w/bb)*bb;
	  if (w < 0) w = w + bb;
	  w1 = (y - w * a) / bb;
	  d = y; x0 = w; y0 = w1;
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (a*b1), (a*k1*x0+par_lower+ib-lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int y_l_ib = a1*m_l_ib + k1*y0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib < z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; y_l=y_l_ib; }

	int m_u_ib = func_m( (- a*b1), (- a*k1*x0 - par_lower - ib + ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int y_u_ib = a1*m_u_ib + k1*y0;
	int z_u_ib = a*x_u_ib + par_lower + ib;
	if (z_u_ib > z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; y_u=y_u_ib; }
      } /* ib */

      if (ib_l == bw || ib_u == -1){ /* set is empty */
	continuous_size = space_size = 0;
	total_size = ((rp_ub-rp_lb)/step + 1) * type_size;

        mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
        if (mpiRet != MPI_SUCCESS) { return 1; }

      }else{
	int mcnt=4;
	mcnt=MAX(mcnt, abs(a1)+2);
	mcnt=MAX(mcnt, abs(bw*b1)+2);
	int b[mcnt]; MPI_Aint d[mcnt]; MPI_Datatype t[mcnt];
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u +1;
	int y_sta = func_m( step, 0 );
	int y_end = func_m( (-step), (- rp_lb + rp_ub) );
#ifdef DEBUG
	printf("y_sta=%d  y_end=%d\n", y_sta, y_end);
#endif /* DEBUG */
	int y_base1 = y_sta;
	int i_base1 = ista;
	MPI_Datatype newtype2a; int byte_newtype2a; MPI_Aint lb_newtype2a, extent_newtype2a;
	MPI_Datatype newtype2b; int byte_newtype2b; MPI_Aint lb_newtype2b, extent_newtype2b;
	MPI_Datatype newtype2c; int byte_newtype2c; MPI_Aint lb_newtype2c, extent_newtype2c;
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  int first=1;
	  int i;
	  for (i=ista; i<iend-(iend-ista) %(bw*b1); i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base1 = y; i_base1 = i; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		b[cnt]=1; d[cnt]=(y - y_base1)*type_size; t[cnt]=dataType0; cnt++;
#ifdef DEBUG
		printf("y - y_base1=%d\n",y - y_base1);
#endif /* DEBUG */
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	  b[cnt]=1; d[cnt]=( a1 )*type_size; t[cnt]=MPI_UB;  cnt++;
#ifdef DEBUG
	  printf("UB1: %d\n",a1);
#endif /* DEBUG */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2a);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
/* 	  MPI_Type_commit(&newtype2a); */
	  MPI_Type_size(newtype2a, &byte_newtype2a);
	  MPI_Type_get_extent(newtype2a, &lb_newtype2a, &extent_newtype2a);
#ifdef DEBUG
	  printf("newtype2a: byte_newtype2a=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype2a, (int)lb_newtype2a, (int)extent_newtype2a);
#endif /* DEBUG */
	}
	int y_base2 = y_sta;
	int i_base2 = ista;
	int pos = 0;
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  int first=1;
	  int i;
	  for (i=iend-(iend-ista) %(bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base2 = y; i_base2 = i; first=0; }
	      b[cnt]=1; d[cnt]=(y - y_base2)*type_size;  t[cnt]=dataType0; cnt++;
	      pos = y - y_base2;
#ifdef DEBUG
	      printf("y - y_base2=%d\n",y - y_base2);
#endif /* DEBUG */
	    }else{
	    }
	  }/* i */
	  b[cnt]=1; d[cnt]=(pos+1)*type_size; t[cnt]=MPI_UB;  cnt++;
#ifdef DEBUG
	  printf("UB2: %d\n",pos+1);
#endif /* DEBUG */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2b);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
/* 	  MPI_Type_commit(&newtype2b); */
	  MPI_Type_size(newtype2b, &byte_newtype2b);
	  MPI_Type_get_extent(newtype2b, &lb_newtype2b, &extent_newtype2b);
#ifdef DEBUG
	  printf("newtype2b: byte_newtype2b=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype2b, (int)lb_newtype2b, (int)extent_newtype2b);
#endif /* DEBUG */
	}
#ifdef DEBUG
	printf("y_base1=%d  y_base2=%d\n", y_base1, y_base2);
	printf("i_base1=%d  i_base2=%d\n", i_base1, i_base2);
#endif /* DEBUG */
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  if (byte_newtype2a > 0){
/* 	    b[cnt]=abs( (iend-ista) / (bw*b1) ) ; d[cnt]=(y_base1-y_sta)*type_size; t[cnt]=newtype2a; cnt++; */
	    b[cnt]=abs( (iend-ista) / (bw*b1) ) ; d[cnt]=(y_base1-y_base1)*type_size; t[cnt]=newtype2a; cnt++;
	  }
	  if (byte_newtype2b > 0){
/* 	    b[cnt]=1; d[cnt]=(y_base2-y_sta)*type_size; t[cnt]=newtype2b; cnt++; */
	    b[cnt]=1; d[cnt]=(y_base2-y_base1)*type_size; t[cnt]=newtype2b; cnt++;
	  }
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2c);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
/* 	  MPI_Type_commit(&newtype2c); */
	  MPI_Type_size(newtype2c, &byte_newtype2c);
	  MPI_Type_get_extent(newtype2c, &lb_newtype2c, &extent_newtype2c);
#ifdef DEBUG
	  printf("newtype2c: byte_newtype2c=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype2c, (int)lb_newtype2c, (int)extent_newtype2c);
#endif /* DEBUG */
	}
	{
	  MPI_Type_free(&newtype2a);
	  MPI_Type_free(&newtype2b);
	  dataType_tmp = newtype2c;
	  continuous_size = byte_newtype2c / type_size;
	  /* space_size = y_l * type_size; */
	  space_size = (y_l - y_sta) * type_size;
	  total_size = ((rp_ub-rp_lb)/step + 1) * type_size;
	}

      } /* ib_l */ /* ib_u */

    } /* if (rp_lb > rp_ub) */
    {
#ifdef DEBUG
      printf("space_size=%d  total_size=%d\n", space_size, total_size);
#endif /* DEBUG */
      int b[3]; MPI_Aint d[3]; MPI_Datatype t[3];
      b[0]=1; d[0]=0;          t[0]=MPI_LB;
      b[1]=1; d[1]=space_size; t[1]=dataType_tmp;
      b[2]=1; d[2]=total_size; t[2]=MPI_UB;
      mpiRet = MPI_Type_create_struct(3, b, d, t, _dataType1);
      if (mpiRet != MPI_SUCCESS) { return 1; }
#ifdef DEBUG
      int byte_dataType1; MPI_Aint lb_dataType1, extent_dataType1;
      MPI_Type_size(*_dataType1, &byte_dataType1);
      MPI_Type_get_extent(*_dataType1, &lb_dataType1, &extent_dataType1);
      printf("dataType1: byte_dataType1=%d bytes  lb=%d bytes  extent=%d bytes\n",
	     byte_dataType1, (int)lb_dataType1, (int)extent_dataType1);
#endif /* DEBUG */
      MPI_Type_free(&dataType_tmp);
    }

  } /* if (step > 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else if (step < 0){
    if (rp_lb < rp_ub){
      return 1;

    }else if (par_upper < rp_ub || rp_lb < par_lower){
      continuous_size = space_size = 0;
      total_size = ((rp_ub-rp_lb)/step + 1) * type_size;

      mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
      if (mpiRet != MPI_SUCCESS) { return 1; }

    }else{
      int lb_tmp = MIN( par_upper, rp_lb );
      int ub_tmp = MAX( par_lower, rp_ub );
      int a = cycle, b = step;
      int ib;
      int z_l = MIN(par_lower, rp_ub)-1; int ib_l = -1; int x_l; int y_l;
      int z_u = MAX(par_upper, rp_lb)+1; int ib_u = bw; int x_u; int y_u;
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0, y0;
	{
	  int x, y, z, w, w1; int q, r, tmp; int bb = -b;
	  if(a == 0 || bb == 0){ return 1; }
	  x = a; y = bb;
	  if(x < 0) x = -x;
	  if(y < 0) y = -y;
	  z = 1;
	  w = 0;
	  while( 1 ){
	    q = x/y;
	    r = x - q*y;
	    if( r == 0 ) break;
	    x = y;
	    y = r;
	    tmp = z;
	    z = w;
	    w = tmp - q * w;
	  }
	  w = w - (w/bb)*bb;
	  if (w < 0) w = w + bb;
	  w1 = (y - w * a) / bb;
	  d = y; x0 = w; y0 = w1;
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (-a*b1), (- a*k1*x0 - par_lower - ib + lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int y_l_ib = a1*m_l_ib + k1*y0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib > z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; y_l=y_l_ib; }

	int m_u_ib = func_m( (a*b1), (a*k1 * x0 + par_lower + ib - ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int y_u_ib = a1*m_u_ib + k1*y0;
	int z_u_ib = a * x_u_ib + par_lower + ib;
	if (z_u_ib < z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; y_u=y_u_ib; }
      } /* ib */

      if (ib_l == -1 || ib_u == bw){ /* set is empty */
	continuous_size = space_size = 0;
	total_size = ((rp_ub-rp_lb)/step + 1) * type_size;

        mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
        if (mpiRet != MPI_SUCCESS) { return 1; }

      }else{ /* ib_l */ /* ib_u */
	int mcnt=4;
	mcnt=MAX(mcnt, abs(a1)+2);
	mcnt=MAX(mcnt, abs(bw*b1)+2);
	int b[mcnt]; MPI_Aint d[mcnt]; MPI_Datatype t[mcnt];
	int iend=bw*x_l+ib_l +1;
	int ista=bw*x_u+ib_u;
	int y_sta = func_m( -step, 0 );
	int y_end = func_m( step, (rp_lb - rp_ub) );
#ifdef DEBUG
	printf("ista=%d  iend=%d  iend-ista=%d  (iend-ista) / (bw*b1)=%d  (iend-ista) %% (bw*b1)=%d\n",
	       ista, iend, iend-ista, (iend-ista) / (bw*b1), (iend-ista) % (bw*b1));
	printf("iend-(iend-ista) %% (bw*b1)=%d\n", iend-(iend-ista) % (bw*b1));
	printf("y_sta=%d  y_end=%d\n", y_sta, y_end);
	printf("y_l=%d  y_u=%d\n", y_l, y_u);
#endif /* DEBUG */
	MPI_Datatype newtype2a; int byte_newtype2a; MPI_Aint lb_newtype2a, extent_newtype2a;
	MPI_Datatype newtype2b; int byte_newtype2b; MPI_Aint lb_newtype2b, extent_newtype2b;
	MPI_Datatype newtype2c; int byte_newtype2c; MPI_Aint lb_newtype2c, extent_newtype2c;
	int i;
	int y_base1 = y_end;
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  int first=1;
	  for(i=ista; i<iend-(iend-ista) % (bw*b1); i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base1 = y; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		b[cnt]=1; d[cnt]=(y_base1 - y)*type_size; t[cnt]=dataType0; cnt++;
#ifdef DEBUG
		printf("y_base1 - y=%d\n",y_base1 - y);
#endif /* DEBUG */
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	  b[cnt]=1; d[cnt]=( a1 )*type_size; t[cnt]=MPI_UB;  cnt++;
#ifdef DEBUG
	  printf("UB1: %d\n",a1);
#endif /* DEBUG */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2a);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  /* MPI_Type_commit(&newtype2a); */
	  MPI_Type_size(newtype2a, &byte_newtype2a);
	  MPI_Type_get_extent(newtype2a, &lb_newtype2a, &extent_newtype2a);
#ifdef DEBUG
	  printf("newtype2a: byte_newtype2a=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype2a, (int)lb_newtype2a, (int)extent_newtype2a);
#endif /* DEBUG */
	}
	int y_base2 = y_end;
	int pos = 0;
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  int first=1;
	  for(i=iend-(iend-ista) % (bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base2 = y; first=0; }
	      b[cnt]=1; d[cnt]=(y_base2 - y)*type_size; t[cnt]=dataType0; cnt++;
	      pos = y_base2 - y;
#ifdef DEBUG
	      printf("y_base2 - y=%d\n",y_base2 - y);
#endif /* DEBUG */
	    }else{
	    }
	  }/* i */
	  b[cnt]=1; d[cnt]=(pos+1)*type_size; t[cnt]=MPI_UB;  cnt++;
#ifdef DEBUG
	  printf("UB2: %d\n",pos+1);
#endif /* DEBUG */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2b);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  /* MPI_Type_commit(&newtype2b); */
	  MPI_Type_size(newtype2b, &byte_newtype2b);
	  MPI_Type_get_extent(newtype2b, &lb_newtype2b, &extent_newtype2b);
#ifdef DEBUG
	  printf("newtype2b: byte_newtype2b=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype2b, (int)lb_newtype2b, (int)extent_newtype2b);
#endif /* DEBUG */
	}
#ifdef DEBUG
	printf("y_base1=%d  y_base2=%d\n", y_base1, y_base2);
#endif /* DEBUG */
	{
	  /* int b[10]; MPI_Aint d[10]; MPI_Datatype t[10]; */
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  if (byte_newtype2a > 0){
	    /* b[cnt]=abs( (iend-ista) / (bw*b1) ) ; d[cnt]=(y_end-y_base1)*type_size; t[cnt]=newtype2a; cnt++; */
	    b[cnt]=abs( (iend-ista) / (bw*b1) ) ; d[cnt]=(y_base1-y_base1)*type_size; t[cnt]=newtype2a; cnt++;
#ifdef DEBUG
	    printf("** y_base1-y_base1=%d\n",y_base1-y_base1);
#endif /* DEBUG */
	  }
	  if (byte_newtype2b > 0){
	    /* b[cnt]=1; d[cnt]=(y_end-y_base2)*type_size; t[cnt]=newtype2b; cnt++; */
	    b[cnt]=1; d[cnt]=(y_base1-y_base2)*type_size; t[cnt]=newtype2b; cnt++;
#ifdef DEBUG
	    printf("** y_base1-y_base2=%d\n",y_base1-y_base2);
#endif /* DEBUG */
	  }
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2c);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  /* MPI_Type_commit(&newtype2c); */
	  MPI_Type_size(newtype2c, &byte_newtype2c);
	  MPI_Type_get_extent(newtype2c, &lb_newtype2c, &extent_newtype2c);
#ifdef DEBUG
	  printf("newtype2c: byte_newtype2c=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype2c, (int)lb_newtype2c, (int)extent_newtype2c);
#endif /* DEBUG */
	}
	{
	  MPI_Type_free(&newtype2a);
	  MPI_Type_free(&newtype2b);
	  dataType_tmp = newtype2c;
	  continuous_size = byte_newtype2c / type_size;
	  space_size = (y_end - y_base1) * type_size;
	  total_size = ((rp_ub-rp_lb)/step + 1) * type_size;
	}

      } /* ib_l */ /* ib_u */

    } /* if (rp_lb < rp_ub) */
    {
#ifdef DEBUG
      printf("space_size=%d  total_size=%d\n", space_size, total_size);
#endif /* DEBUG */
      int b[3]; MPI_Aint d[3]; MPI_Datatype t[3];
      b[0]=1; d[0]=0;          t[0]=MPI_LB;
      b[1]=1; d[1]=space_size; t[1]=dataType_tmp;
      b[2]=1; d[2]=total_size; t[2]=MPI_UB;
      mpiRet = MPI_Type_create_struct(3, b, d, t, _dataType1);
      if (mpiRet != MPI_SUCCESS) { return 1; }
#ifdef DEBUG
      int byte_dataType1; MPI_Aint lb_dataType1, extent_dataType1;
      MPI_Type_size(*_dataType1, &byte_dataType1);
      MPI_Type_get_extent(*_dataType1, &lb_dataType1, &extent_dataType1);
      printf("dataType1: byte_dataType1=%d bytes  lb=%d bytes  extent=%d bytes\n",
	     byte_dataType1, (int)lb_dataType1, (int)extent_dataType1);
#endif /* DEBUG */
      MPI_Type_free(&dataType_tmp);
    }

  } /* if (step < 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else{ return 1; /* dummy */
  }
  /* ++++++++++++++++++++++++++++++++++++++++ */
/*   printf("------------------------------ _xmp_io_block_cyclic_0: NORMAL END\n"); */
  return MPI_SUCCESS;
}
/* ------------------------------------------------------------------ */
static int _xmp_io_block_cyclic_1
(
 int par_lower /* in */, int par_upper /* in */, int bw /* in */, int cycle /* in */,
 int rp_lb /* in */, int rp_ub /* in */, int step /* in */,
 int local_lower /* in */,
 int alloc_size /* in */,
 MPI_Datatype dataType0 /* in */,
 MPI_Datatype *_dataType1 /* out */
)
{
  MPI_Datatype dataType_tmp;
  int continuous_size, space_size, total_size;
  int mpiRet;

  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // get extent of data type
  MPI_Aint tmp1, tmp2; int type_size;
  mpiRet = MPI_Type_get_extent(dataType0, &tmp1, &tmp2);
  if (mpiRet !=  MPI_SUCCESS) { return -1113; }  
  type_size = (int)tmp2;

  printf("_xmp_io_block_cyclic_1: rmyank = %d:  par_lower = %d  par_upper = %d  bw = %d  cycle = %d  alloc_size = %d  type_size = %d\n",
	 myrank,par_lower,par_upper,bw,cycle, alloc_size,type_size);

  if (bw <= 0){ _XMP_fatal("_xmp_io_block_cyclic_1: block width must be pisitive."); }
  if (cycle == 0){ _XMP_fatal("_xmp_io_block_cyclic_1: cycle must be non-zero."); }
  /* ++++++++++++++++++++++++++++++++++++++++ */
  if (step > 0){
    if (rp_lb > rp_ub){
      return 1;

    }else if(par_upper < rp_lb || rp_ub < par_lower){
      continuous_size = space_size = 0;
      total_size = alloc_size * type_size;

      mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
      if (mpiRet != MPI_SUCCESS) { return 1; }

    }else{
      int lb_tmp = MAX(par_lower, rp_lb);
      int ub_tmp = MIN(par_upper, rp_ub);
      int a = cycle, b = step;
      int ib;
      int z_l = MAX(par_upper,rp_ub) + 1; int ib_l = bw; int x_l; int y_l;
      int z_u = MIN(par_lower,rp_lb) - 1; int ib_u = -1; int x_u; int y_u;
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0, y0;
	{
	  int x, y, z, w, w1; int q, r, tmp; int bb = -b;
	  if(a == 0 || bb == 0){ return 1; }
	  x = a; y = bb;
	  if(x < 0) x = -x;
	  if(y < 0) y = -y;
	  z = 1;
	  w = 0;
	  while( 1 ){
	    q = x/y;
	    r = x - q*y;
	    if( r == 0 ) break;
	    x = y;
	    y = r;
	    tmp = z;
	    z = w;
	    w = tmp - q * w;
	  }
	  w = w - (w/bb)*bb;
	  if (w < 0) w = w + bb;
	  w1 = (y - w * a) / bb;
	  d = y; x0 = w; y0 = w1;
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (a*b1), (a*k1*x0+par_lower+ib-lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int y_l_ib = a1*m_l_ib + k1*y0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib < z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; y_l=y_l_ib; }

	int m_u_ib = func_m( (- a*b1), (- a*k1*x0 - par_lower - ib + ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int y_u_ib = a1*m_u_ib + k1*y0;
	int z_u_ib = a*x_u_ib + par_lower + ib;
	if (z_u_ib > z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; y_u=y_u_ib; }
      } /* ib */
#ifdef DEBUG
      printf("bw = %d  x_l = %d  x_u = %d  ib_l = %d  ib_u = %d\n",bw,x_l,x_u,ib_l,ib_u);
#endif /* DEBUG */
      if (ib_l == bw || ib_u == -1){ /* set is empty */
	continuous_size = space_size = 0;
	total_size = alloc_size * type_size;

        mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
        if (mpiRet != MPI_SUCCESS) { return 1; }

      }else{ /* ib_l */ /* ib_u */
	int mcnt=4;
	mcnt=MAX(mcnt, abs(a1)+2);
	mcnt=MAX(mcnt, abs(bw*b1)+2);
	int b[mcnt]; MPI_Aint d[mcnt]; MPI_Datatype t[mcnt];
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u +1;
	int y_sta = func_m( step, 0 );
	int y_end = func_m( (-step), (- rp_lb + rp_ub) );
#ifdef DEBUG
	printf("y_sta=%d  y_end=%d\n", y_sta, y_end);
#endif /* DEBUG */
	int y_base1 = y_sta;
	int i_base1 = ista;
	MPI_Datatype newtype3a; int byte_newtype3a; MPI_Aint lb_newtype3a, extent_newtype3a;
	MPI_Datatype newtype3b; int byte_newtype3b; MPI_Aint lb_newtype3b, extent_newtype3b;
	MPI_Datatype newtype3c; int byte_newtype3c; MPI_Aint lb_newtype3c, extent_newtype3c;
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  int first=1;
	  int i;
#ifdef DEBUG
	  printf("ista = %d  iend = %d  (iend-ista) %(bw*b1) = %d  (bw*b1)\n",
		 ista,iend,(iend-ista) %(bw*b1),(bw*b1));
#endif /* DEBUG */
	  for (i=ista; i<iend-(iend-ista) %(bw*b1); i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base1 = y; i_base1 = i; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		b[cnt]=1; d[cnt]=(i - i_base1)*type_size; t[cnt]=dataType0; cnt++;
#ifdef DEBUG
		printf("i - i_base1=%d\n", i - i_base1);
#endif /* DEBUG */
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	  b[cnt]=1; d[cnt]=( abs(bw*b1) )*type_size; t[cnt]=MPI_UB;  cnt++;
#ifdef DEBUG
	  printf("UB1: %d\n", abs(bw*b1));
#endif /* DEBUG */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype3a);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
/* 	  MPI_Type_commit(&newtype3a); */
	  MPI_Type_size(newtype3a, &byte_newtype3a);
	  MPI_Type_get_extent(newtype3a, &lb_newtype3a, &extent_newtype3a);
#ifdef DEBUG
	  printf("newtype3a: byte_newtype3a=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype3a, (int)lb_newtype3a, (int)extent_newtype3a);
#endif /* DEBUG */
	}
	int y_base2 = y_sta;
	int i_base2 = ista;
	{
	  int pos=0;
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  int first=1;
	  int i;


	  for (i=iend-(iend-ista) %(bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base2 = y; i_base2 = i; first=0; }
	      b[cnt]=1; d[cnt]=(i - i_base2)*type_size;  t[cnt]=dataType0; cnt++;
	      pos = i - i_base2;
#ifdef DEBUG
	      printf("i - i_base2=%d\n", i - i_base2);
#endif /* DEBUG */
	    }else{
	    }
	  }/* i */
	  b[cnt]=1; d[cnt]=( pos+1 )*type_size; t[cnt]=MPI_UB;  cnt++;
#ifdef DEBUG
	  printf("UB2: %d\n", pos+1);
#endif /* DEBUG */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype3b);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
/* 	  MPI_Type_commit(&newtype3b); */
	  MPI_Type_size(newtype3b, &byte_newtype3b);
	  MPI_Type_get_extent(newtype3b, &lb_newtype3b, &extent_newtype3b);
#ifdef DEBUG
	  printf("newtype3b: byte_newtype3b=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype3b, (int)lb_newtype3b, (int)extent_newtype3b);
#endif /* DEBUG */
	}
#ifdef DEBUG
	printf("y_base1=%d  y_base2=%d\n", y_base1, y_base2);
	printf("i_base1=%d  i_base2=%d\n", i_base1, i_base2);
#endif /* DEBUG */
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  if (byte_newtype3a > 0){
/* 	    b[cnt]=abs( (iend-ista) / (bw*b1) ) ; d[cnt]=(i_base1)*type_size; t[cnt]=newtype3a; cnt++; */
	    b[cnt]=abs( (iend-ista) / (bw*b1) ) ; d[cnt]=(i_base1 - i_base1)*type_size; t[cnt]=newtype3a; cnt++;
	  }
	  if (byte_newtype3b > 0){
/* 	    b[cnt]=1; d[cnt]=(i_base2)*type_size; t[cnt]=newtype3b; cnt++; */
	    b[cnt]=1; d[cnt]=(i_base2 - i_base1)*type_size; t[cnt]=newtype3b; cnt++;
	  }
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype3c);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
/* 	  MPI_Type_commit(&newtype3c); */
	  MPI_Type_size(newtype3c, &byte_newtype3c);
	  MPI_Type_get_extent(newtype3c, &lb_newtype3c, &extent_newtype3c);
#ifdef DEBUG
	  printf("newtype3c: byte_newtype3c=%d bytes  lb=%d bytes  extent=%d bytes\n",
		 byte_newtype3c, (int)lb_newtype3c, (int)extent_newtype3c);
#endif /* DEBUG */
	}
	{
#ifdef DEBUG
	  printf("alloc_size=%d  type_size=%d\n", alloc_size, type_size);
#endif /* DEBUG */
	  MPI_Type_free(&newtype3a);
	  MPI_Type_free(&newtype3b);
	  dataType_tmp = newtype3c;
	  continuous_size = byte_newtype3c / type_size;
	  space_size = (ista + local_lower) * type_size;
	  total_size = alloc_size * type_size;
	}

      } /* ib_l */ /* ib_u */

    } /* if (rp_lb > rp_ub) */
    {
#ifdef DEBUG
      printf("space_size=%d  total_size=%d\n", space_size, total_size);
#endif /* DEBUG */
      int b[3]; MPI_Aint d[3]; MPI_Datatype t[3];
      b[0]=1; d[0]=0;          t[0]=MPI_LB;
      b[1]=1; d[1]=space_size; t[1]=dataType_tmp;
      b[2]=1; d[2]=total_size; t[2]=MPI_UB;
      mpiRet = MPI_Type_create_struct(3, b, d, t, _dataType1);
      if (mpiRet != MPI_SUCCESS) { return 1; }
#ifdef DEBUG
      int byte_dataType1; MPI_Aint lb_dataType1, extent_dataType1;
      MPI_Type_size(*_dataType1, &byte_dataType1);
      MPI_Type_get_extent(*_dataType1, &lb_dataType1, &extent_dataType1);
      printf("dataType1: byte_dataType1=%d bytes  lb=%d bytes  extent=%d bytes\n",
	     byte_dataType1, (int)lb_dataType1, (int)extent_dataType1);
#endif /* DEBUG */
      MPI_Type_free(&dataType_tmp);
    }

  } /* if (step > 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else if (step < 0){
    if (rp_lb < rp_ub){ return 1;
    }else if (par_upper < rp_ub || rp_lb < par_lower){ return 1; /* dummy */
    }else{ return 1; /* dummy */
    } /* if (rp_lb < rp_ub) */
  } /* if (step < 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else{ return 1; /* dummy */
  }
  /* ++++++++++++++++++++++++++++++++++++++++ */
/*   printf("-------------------- _xmp_io_block_cyclic_1: NORMAL END\n"); */
  return MPI_SUCCESS;
}
/* ------------------------------------------------------------------ */
static int _xmp_io_block_cyclic_2
(
 int par_lower /* in */, int par_upper /* in */, int bw /* in */, int cycle /* in */,
 int rp_lb /* in */, int rp_ub /* in */, int step /* in */,
 int *_cnt /* out */, int **_bc2_result /* out */
)
{
  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  printf("_xmp_io_block_cyclic_2: rmyank = %d:  par_lower = %d  par_upper = %d  bw = %d  cycle = %d\n",
	 myrank,par_lower,par_upper,bw,cycle);

  if (bw <= 0){ _XMP_fatal("_xmp_io_block_cyclic_2: block width must be pisitive."); }
  if (cycle == 0){ _XMP_fatal("_xmp_io_block_cyclic_2: cycle must be non-zero."); }
  /* ++++++++++++++++++++++++++++++++++++++++ */
  if (step > 0){
    if (rp_lb > rp_ub){
      return 1;

    }else if(par_upper < rp_lb || rp_ub < par_lower){
      *_cnt = 0;
      *_bc2_result = NULL;
    }else{
      int lb_tmp = MAX(par_lower, rp_lb);
      int ub_tmp = MIN(par_upper, rp_ub);
      int a = cycle, b = step;
      int ib;
      int z_l = MAX(par_upper,rp_ub) + 1; int ib_l = bw; int x_l; int y_l;
      int z_u = MIN(par_lower,rp_lb) - 1; int ib_u = -1; int x_u; int y_u;
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0, y0;
	{
	  int x, y, z, w, w1; int q, r, tmp; int bb = -b;
	  if(a == 0 || bb == 0){ return 1; }
	  x = a; y = bb;
	  if(x < 0) x = -x;
	  if(y < 0) y = -y;
	  z = 1;
	  w = 0;
	  while( 1 ){
	    q = x/y;
	    r = x - q*y;
	    if( r == 0 ) break;
	    x = y;
	    y = r;
	    tmp = z;
	    z = w;
	    w = tmp - q * w;
	  }
	  w = w - (w/bb)*bb;
	  if (w < 0) w = w + bb;
	  w1 = (y - w * a) / bb;
	  d = y; x0 = w; y0 = w1;
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (a*b1), (a*k1*x0+par_lower+ib-lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int y_l_ib = a1*m_l_ib + k1*y0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib < z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; y_l=y_l_ib; }

	int m_u_ib = func_m( (- a*b1), (- a*k1*x0 - par_lower - ib + ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int y_u_ib = a1*m_u_ib + k1*y0;
	int z_u_ib = a*x_u_ib + par_lower + ib;
	if (z_u_ib > z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; y_u=y_u_ib; }
      } /* ib */

      if (ib_l == bw || ib_u == -1){ /* set is empty */
	*_cnt = 0;
	*_bc2_result = NULL;
      }else{ /* ib_l */ /* ib_u */
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u +1;
	int y_sta = func_m( step, 0 );
	int y_end = func_m( (-step), (- rp_lb + rp_ub) );
#ifdef DEBUG
	printf("y_sta=%d  y_end=%d\n", y_sta, y_end);
#endif /* DEBUG */
	int y_base1 = y_sta;
	int i_base1 = ista;
	*_bc2_result = (int *)malloc(sizeof(int)*(6+abs(bw*b1*2)));
	int ncnt1=0, ncnt2=0;
	/* int di1[abs(bw*b1*2)]; */
	int *di1 = *_bc2_result + 6;
	{
	  int first=1;
	  int i;
	  for (i=ista; i<iend-(iend-ista) %(bw*b1); i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base1 = y; i_base1 = i; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		di1[ncnt1++] = i-i_base1;
#ifdef DEBUG
		printf("di1[%d]=%d\n", ncnt1-1, di1[ncnt1-1]);
#endif /* DEBUG */
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	}
	int y_base2 = y_sta;
	int i_base2 = ista;
	{
	  int first=1;
	  int i;
	  for (i=iend-(iend-ista) %(bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base2 = y; i_base2 = i; first=0; }
	      di1[ncnt1+ncnt2++] = i-i_base2;
#ifdef DEBUG
	      printf("di1[%d]=%d\n", ncnt1+ncnt2-1, di1[ncnt1+ncnt2-1]);
#endif /* DEBUG */
	    }else{
	    }
	  }/* i */
	}
#ifdef DEBUG
	printf("y_base1=%d  y_base2=%d\n", y_base1, y_base2);
	printf("i_base1=%d  i_base2=%d\n", i_base1, i_base2);
	printf("ncnt1=%d  ncnt2=%d  ((iend-ista) / (bw*b1))*ncnt1 + ncnt2=%d\n",
	       ncnt1, ncnt2, ((iend-ista) / (bw*b1))*ncnt1 + ncnt2);
#endif /* DEBUG */

	{
	  int pp = (iend-ista) / (bw*b1);
	  int *bc2_res = *_bc2_result;
	  bc2_res[0] = 6 + abs(bw*b1*2); /* alloc size */
	  bc2_res[1] = ncnt1;
	  bc2_res[2] = ncnt2;
	  bc2_res[3] = pp*ncnt1; /* pp_ncnt1 */
	  bc2_res[4] = bw*b1; /* bw_b1 */
	  bc2_res[5] = i_base1;
	  *_cnt = pp*ncnt1 + ncnt2;
	}
      } /* ib_l */ /* ib_u */

    } /* if (rp_lb > rp_ub) */
  } /* if (step > 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else if (step < 0){
    if (rp_lb < rp_ub){ return 1;
    }else if (par_upper < rp_ub || rp_lb < par_lower){
      *_cnt = 0;
      *_bc2_result = NULL;
    }else{
      int lb_tmp = MIN( par_upper, rp_lb );
      int ub_tmp = MAX( par_lower, rp_ub );
      int a = cycle, b = step;
      int ib;
      int z_l = MIN(par_lower, rp_ub)-1; int ib_l = -1; int x_l; int y_l;
      int z_u = MAX(par_upper, rp_lb)+1; int ib_u = bw; int x_u; int y_u;
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0, y0;
	{
	  int x, y, z, w, w1; int q, r, tmp; int bb = -b;
	  if(a == 0 || bb == 0){ return 1; }
	  x = a; y = bb;
	  if(x < 0) x = -x;
	  if(y < 0) y = -y;
	  z = 1;
	  w = 0;
	  while( 1 ){
	    q = x/y;
	    r = x - q*y;
	    if( r == 0 ) break;
	    x = y;
	    y = r;
	    tmp = z;
	    z = w;
	    w = tmp - q * w;
	  }
	  w = w - (w/bb)*bb;
	  if (w < 0) w = w + bb;
	  w1 = (y - w * a) / bb;
	  d = y; x0 = w; y0 = w1;
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (-a*b1), (- a*k1*x0 - par_lower - ib + lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int y_l_ib = a1*m_l_ib + k1*y0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib > z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; y_l=y_l_ib; }

	int m_u_ib = func_m( (a*b1), (a*k1 * x0 + par_lower + ib - ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int y_u_ib = a1*m_u_ib + k1*y0;
	int z_u_ib = a * x_u_ib + par_lower + ib;
	if (z_u_ib < z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; y_u=y_u_ib; }
      } /* ib */

      if (ib_l == -1 || ib_u == bw){ /* set is empty */
	*_cnt = 0;
	*_bc2_result = NULL;

      }else{ /* ib_l */ /* ib_u */
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u -1;
	int y_sta = func_m( -step, 0 );
	int y_end = func_m( step, (rp_lb - rp_ub) );
#ifdef DEBUG
	printf("y_sta=%d  y_end=%d\n", y_sta, y_end);
#endif /* DEBUG */
	int y_base1 = y_sta;
	int i_base1 = ista;
	int y_base2 = y_sta;
	int i_base2 = ista;
	*_bc2_result = (int *)malloc(sizeof(int)*(6+abs(bw*b1*2)));
	int ncnt1=0, ncnt2=0;
	/* int di1[abs(bw*b1*2)]; */
	int *di1 = *_bc2_result + 6;
	int i;
	{
	  int first=1;
	  for (i=ista; i>iend-(iend-ista) % (bw*b1); i--){ /* decrement */
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base1 = y; i_base1 = i; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		di1[ncnt1++] = i-i_base1;
#ifdef DEBUG
		printf("di1[%d]=%d\n", ncnt1-1, di1[ncnt1-1]);
#endif /* DEBUG */
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	}
	{
	  int first=1;
	  for (i=iend-(iend-ista) % (bw*b1); i>iend; i--){ /* decrement */
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base2 = y; i_base2 = i; first=0; /* FALSE */ }
	      di1[ncnt1+ncnt2++] = i-i_base2;
#ifdef DEBUG
	      printf("di1[%d]=%d\n", ncnt1+ncnt2-1, di1[ncnt1+ncnt2-1]);
#endif /* DEBUG */
	    }else{
	    }
	  }/* i */
	}
#ifdef DEBUG
	printf("y_base1=%d  y_base2=%d\n", y_base1, y_base2);
	printf("i_base1=%d  i_base2=%d\n", i_base1, i_base2);
	printf("ncnt1=%d  ncnt2=%d  ((iend-ista) / (bw*b1))*ncnt1 + ncnt2=%d\n",
	       ncnt1, ncnt2, ((iend-ista) / (bw*b1))*ncnt1 + ncnt2);
#endif /* DEBUG */
	{
	  int pp = (iend-ista) / (bw*b1);
	  int *bc2_res = *_bc2_result;
	  bc2_res[0] = 6 + abs(bw*b1*2); /* alloc size */
	  bc2_res[1] = ncnt1;
	  bc2_res[2] = ncnt2;
	  bc2_res[3] = pp*ncnt1; /* pp_ncnt1 */
	  bc2_res[4] = bw*b1; /* bw_b1 */
	  bc2_res[5] = i_base1;
	  *_cnt = pp*ncnt1 + ncnt2;
	}
      } /* ib_l */ /* ib_u */

    } /* if (rp_lb < rp_ub) */
  } /* if (step < 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else{ return 1; /* dummy */
  }
  /* ++++++++++++++++++++++++++++++++++++++++ */
  return MPI_SUCCESS;
}
/* ------------------------------------------------------------------ */
static int _xmp_io_block_cyclic_3
(
 int j /* in */, int *bc2_result /* in */,
 int *_local_index /* out */
)
{
  if ( bc2_result == NULL){
    return 1;
  }else{
    int ncnt1    = bc2_result[1];
    int pp_ncnt1 = bc2_result[3];
    int bw_b1    = bc2_result[4];
    int i_base1  = bc2_result[5];
    int *di1    = &bc2_result[6];
    int p = (ncnt1>0 ? j/ncnt1: 0);
    int q = (j<pp_ncnt1 ? j % ncnt1 : j-pp_ncnt1);
    int i = p * (bw_b1) + di1[q] + i_base1;
#ifdef DEBUG
    printf("j=%d  p=%d  q=%d  i=%d\n", j, p, q, i);
#endif /* DEBUG */
  }
  return MPI_SUCCESS;
}
/* ================================================================== */
/*****************************************************************************/
/*  FUNCTION NAME : xmp_allocate_range                                       */
/*  DESCRIPTION   : xmp_allocate_range is used to allocate memory.           */
/*  ARGUMENT      : n_dim[IN] : the number of dimensions.                    */
/*  RETURN VALUES : Upon successful completion, return the descriptor of     */
/*                  array section. NULL is returned when a program abend.    */
/*                                                                           */
/*****************************************************************************/
xmp_range_t *xmp_allocate_range(int n_dim)
{
  xmp_range_t *rp = NULL;
  if (n_dim <= 0){ return rp; }
  rp = (xmp_range_t *)malloc(sizeof(xmp_range_t));
  rp->dims = n_dim;
  rp->lb = (int*)malloc(sizeof(int)*rp->dims);
  rp->ub = (int*)malloc(sizeof(int)*rp->dims);
  rp->step = (int*)malloc(sizeof(int)*rp->dims);
  if(!rp->lb || !rp->ub || !rp->step){ return rp; }
  return rp;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_set_range                                            */
/*  DESCRIPTION   : xmp_set_range is used to set ranges of an array section. */
/*  ARGUMENT      : rp[IN] : descriptor of array section.                    */
/*               i_dim[IN] : target dimension.                               */
/*                  lb[IN] : lower bound of array section in the dimension   */
/*                           i_dim.                                          */
/*              length[IN] : length of array section in the dimension i_dim. */
/*                step[IN] : stride of array section in the dimension i_dim. */
/*  RETURN VALUES : None.                                                    */
/*                                                                           */
/*****************************************************************************/
void xmp_set_range(xmp_range_t *rp, int i_dim, int lb, int length, int step)
{

  if (rp == NULL){ _XMP_fatal("xmp_set_range: descriptor is NULL"); }
  if (step == 0){ _XMP_fatal("xmp_set_range: step == 0"); }
  if (i_dim-1 < 0 || i_dim-1 >= rp->dims){ _XMP_fatal("xmp_set_range: i_dim is out of range"); }
  if(!rp->lb || !rp->ub || !rp->step){ _XMP_fatal("xmp_set_range: null pointer"); }
  if (step != 0 && length == 0){ _XMP_fatal("xmp_set_range: invalid combination of length and step\n"); }
  rp->lb[i_dim-1] = lb;
  /* length = ub - lb + 1 */
  /* ub = lb + length - 1 */
  rp->ub[i_dim-1] = lb + length - 1;
  rp->step[i_dim-1] = step;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_free_range                                           */
/*  DESCRIPTION   : xmp_free_range release the memory for the descriptor.    */
/*  ARGUMENT      : rp[IN] : descriptor of array section.                    */
/*  RETURN VALUES : None.                                                    */
/*                                                                           */
/*****************************************************************************/
void xmp_free_range(xmp_range_t *rp)
{
  if (rp == NULL){
    return;
  }else{
    if (rp->lb){ free(rp->lb); }
    if (rp->ub){ free(rp->ub); }
    if (rp->step){ free(rp->step); }
    free(rp);
  }
}
/* ------------------------------------------------------------------ */
static int _xmp_range_get_dims(xmp_range_t *rp)
{
  if (rp == NULL){ return -1; }
  return rp->dims;
}
/* ------------------------------------------------------------------ */
static int *_xmp_range_get_lb_addr(xmp_range_t *rp)
{
  if (rp == NULL){ return NULL; }
  return rp->lb;
}
/* ------------------------------------------------------------------ */
static int *_xmp_range_get_ub_addr(xmp_range_t *rp)
{
  if (rp == NULL){ return NULL; }
  return rp->ub;
}
/* ------------------------------------------------------------------ */
static int *_xmp_range_get_step_addr(xmp_range_t *rp)
{
  if (rp == NULL){ return NULL; }
  return rp->step;
}
/* ------------------------------------------------------------------ */
/* end of inc_xmp_io.c */
/* ================================================================== */
#endif

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fopen_all                                            */
/*  DESCRIPTION   : xmp_fopen_all opens a global I/O file.                   */
/*                  Collective (global) execution.                           */
/*  ARGUMENT      : fname[IN] : file name.                                   */
/*                  amode[IN] : equivalent to fopen of POSIX.                */
/*                              combination of "rwa+".                       */
/*  RETURN VALUES : xmp_file_t* : file structure. NULL is returned when a    */
/*                                program abend.                             */
/*                                                                           */
/*****************************************************************************/
xmp_file_t *xmp_fopen_all(const char *fname, const char *amode)
{
  xmp_file_t *pstXmp_file = NULL;
  int         iMode = 0;
  size_t      modelen = 0;

  // allocate
  pstXmp_file = malloc(sizeof(xmp_file_t));
  if (pstXmp_file == NULL) { return NULL; } 
  memset(pstXmp_file, 0x00, sizeof(xmp_file_t));
  
  ///
  /// mode analysis
  ///
  modelen = strlen(amode);
  // mode has single character
  if (modelen == 1)
  {
    if (strncmp(amode, "r", modelen) == 0)
    {
      iMode = MPI_MODE_RDONLY;
    }
    else if (strncmp(amode, "w", modelen) == 0)
    {
      iMode = (MPI_MODE_WRONLY | MPI_MODE_CREATE);
    }
    else if (strncmp(amode, "a", modelen) == 0)
    {
      iMode = (MPI_MODE_RDWR | MPI_MODE_CREATE | MPI_MODE_APPEND);
      pstXmp_file->is_append = 0x01;
    }
    else
    {
      goto ErrorExit;
    }
  }
  // mode has two characters
  else if (modelen == 2)
  {
    if (strncmp(amode, "r+", modelen) == 0)
    {
      iMode = MPI_MODE_RDWR;
    }
    else if (strncmp(amode, "w+", modelen) == 0)
    {
      iMode = (MPI_MODE_RDWR | MPI_MODE_CREATE);
    }
    else if (strncmp(amode, "a+", modelen) == 0 ||
             strncmp(amode, "ra", modelen) == 0 ||
             strncmp(amode, "ar", modelen) == 0)
    {
      iMode = (MPI_MODE_RDWR | MPI_MODE_CREATE);
      pstXmp_file->is_append = 0x01;
    }
    else if (strncmp(amode, "rw", modelen) == 0 ||
             strncmp(amode, "wr", modelen) == 0)
    {
        goto ErrorExit;
    }
    else
    {
      goto ErrorExit;
    }
  }
  // mode has more than two characters
  else
  {
    goto ErrorExit;
  }

  // file open
  if (MPI_File_open(MPI_COMM_WORLD,
                    (char*)fname,
                    iMode,
                    MPI_INFO_NULL,
                    &(pstXmp_file->fh)) != MPI_SUCCESS)
  {
    goto ErrorExit;
  }

  // if "W" or "W+", then set file size to zero
  if ((iMode == (MPI_MODE_WRONLY | MPI_MODE_CREATE)  ||
       iMode == (MPI_MODE_RDWR   | MPI_MODE_CREATE)) &&
       pstXmp_file->is_append == 0x00)
  {
    if (MPI_File_set_size(pstXmp_file->fh, 0) != MPI_SUCCESS)
    {
      goto ErrorExit;
    }
  }

  // normal return
  return pstXmp_file;

// on error
ErrorExit:
  if (pstXmp_file != NULL)
  {
    free(pstXmp_file);
  }
  return NULL;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fclose_all                                           */
/*  DESCRIPTION   : xmp_fclose_all closes a global I/O file. Collective      */
/*                  (global) execution.                                      */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*  RETURN VALUES : 0: normal termination.                                   */
/*                  1: abnormal termination. pstXmp_file is NULL.            */
/*                  2: abnormal termination. error in MPI_File_close.        */
/*                                                                           */
/*****************************************************************************/
int xmp_fclose_all(xmp_file_t *pstXmp_file)
{
  // check argument
  if (pstXmp_file == NULL)     { return 1; }

  // file close
  if (MPI_File_close(&(pstXmp_file->fh)) != MPI_SUCCESS)
  {
    free(pstXmp_file);
#ifdef ORIGINAL
    return 1;
#else /* RIST */
    return 2;
#endif
  }
  free(pstXmp_file);
  return 0;
}


/*****************************************************************************/
/*  FUNCTION NAME : xmp_fseek                                                */
/*  DESCRIPTION   : xmp_fseek sets the indivisual file pointer in the file   */
/*                  structure. Local execution.                              */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  offset[IN] : displacement of current file view from      */
/*                               position of whence.                         */
/*                  whence[IN] : choose file position                        */
/*  RETURN VALUES : 0: normal termination.                                   */
/*                  an integer other than 0: abnormal termination.           */
/*                                                                           */
/*****************************************************************************/
int xmp_fseek(xmp_file_t *pstXmp_file, long long offset, int whence)
{
  int iMpiWhence;

  // checkk argument
  if (pstXmp_file == NULL) { return 1; }

  // convert offset to MPI_Offset
  switch (whence)
  {
    case SEEK_SET:
      iMpiWhence = MPI_SEEK_SET;
      break;
    case SEEK_CUR:
      iMpiWhence = MPI_SEEK_CUR;
      break;
    case SEEK_END:
      iMpiWhence = MPI_SEEK_END;
      break;
    default:
      return 1;
  }

  // file seek
  if (MPI_File_seek(pstXmp_file->fh, (MPI_Offset)offset, iMpiWhence) != MPI_SUCCESS)
  {
    return 1;
  }

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fseek_shared                                         */
/*  DESCRIPTION   : xmp_fseek_shared sets the shared file pointer in the     */
/*                  file structure. Local execution.                         */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  offset[IN] : displacement of current view from position  */
/*                               of whence.                                  */
/*                  whence[IN] : choose file position.                       */
/*  RETURN VALUES : 0: normal termination.                                   */
/*                  an integer other than 0: abnormal termination.           */
/*                                                                           */
/*****************************************************************************/
int xmp_fseek_shared(xmp_file_t *pstXmp_file, long long offset, int whence)
{
  int iMpiWhence;

  // check argument
  if (pstXmp_file == NULL) { return 1; }

  // convert offset to MPI_Offset
  switch (whence)
  {
    case SEEK_SET:
      iMpiWhence = MPI_SEEK_SET;
      break;
    case SEEK_CUR:
      iMpiWhence = MPI_SEEK_CUR;
      break;
    case SEEK_END:
      iMpiWhence = MPI_SEEK_END;
      break;
    default:
      return 1;
  }

  // file seek
  if (MPI_File_seek_shared(pstXmp_file->fh, (MPI_Offset)offset, iMpiWhence) != MPI_SUCCESS)
  {
    return 1;
  }

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_ftell                                                */
/*  DESCRIPTION   : xmp_ftell inquires the position of the indivisual file   */
/*                  pointer in the file structure. Local execution.          */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*  RETURN VALUES : Upon successful completion, the function shall open the  */
/*                  file and return a non-negative integer representing the  */
/*                  lowest numbered unused file descriptor.                  */
/*                  Otherwise, negative number shall be returned.            */
/*                                                                           */
/*****************************************************************************/
long long xmp_ftell(xmp_file_t *pstXmp_file)
{
  MPI_Offset offset;
  MPI_Offset disp;

  // check argument
  if (pstXmp_file == NULL) { return -1; }

  if (MPI_File_get_position(pstXmp_file->fh, &offset) != MPI_SUCCESS)
  {
    return -1;
  }

  if (MPI_File_get_byte_offset(pstXmp_file->fh, offset, &disp) != MPI_SUCCESS)
  {
    return -1;
  }

  return (long long)disp;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_ftell_shared                                         */
/*  DESCRIPTION   : xmp_ftell_shared inquires the position of shaed file     */
/*                  pointer in the file structure. Local execution.          */
/*  ARGUMENT      : pstXmp_file[IN] ファイル構造体                             */
/*  RETURN VALUES : Upon successful completion, the function shall open the  */
/*                  file and return a non-negative integer representing the  */
/*                  lowest numbered unused file descriptor.                  */
/*                  Otherwise, negative number shall be returned.            */
/*                                                                           */
/*****************************************************************************/
long long xmp_ftell_shared(xmp_file_t *pstXmp_file)
{
  MPI_Offset offset;
  MPI_Offset disp;

  // check argument
  if (pstXmp_file == NULL) { return -1; }

  if (MPI_File_get_position_shared(pstXmp_file->fh, &offset) != MPI_SUCCESS)
  {
    return -1;
  }

  if (MPI_File_get_byte_offset(pstXmp_file->fh, offset, &disp) != MPI_SUCCESS)
  {
    return -1;
  }

  return (long long)disp;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_file_sync_all                                        */
/*  DESCRIPTION   : xmp_file_sync_all guarantees completion of access to the */
/*                  file from nodes sharing the file.Collective (global)     */
/*                  execution.                                               */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*  RETURN VALUES : 0: normal termination                                    */
/*                  an integer other than 0: abnormal termination.           */
/*                                                                           */
/*****************************************************************************/
long long xmp_file_sync_all(xmp_file_t *pstXmp_file)
{
  // check argument
  if (pstXmp_file == NULL) { return 1; }

  // sync
  if (MPI_File_sync(pstXmp_file->fh) != MPI_SUCCESS)
  {
    return 1;
  }

  // barrier
  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_all                                            */
/*  DESCRIPTION   : xmp_fread_allreads data from the position of the shared  */
/*                  file pointer onto the all executing nodes. Collective    */
/*                  (global) execution.                                      */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  buffer[OUT] : beginning address of loading variables.    */
/*                  size[IN] : the byte size of a loading element of data.   */
/*                  count[IN] : the number of loading data element           */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  loading data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fread_all(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int readCount;

  // check argument
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // read
  if (MPI_File_read_all(pstXmp_file->fh,
                        buffer, size * count,
                        MPI_BYTE,
                        &status) != MPI_SUCCESS)
  {
    return -1;
  }
  
  // number of bytes read
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return readCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_all                                           */
/*  DESCRIPTION   : xmp_fwrite_all writes indivisual data on the all         */
/*                  executing nodes to the position of the shared file       */
/*                  pointer. Collective (global) execution.                  */
/*                  It is assumed that the file view is set previously.      */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  buffer[IN] : begginning address of storing variables.    */
/*                  size[IN] : the byte size of a storing element of data.   */
/*                  count[IN] : the number of storing data element.          */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  storing data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fwrite_all(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int writeCount;

  // check argument
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // if file open is "r+", then move pointer to end
  if (pstXmp_file->is_append)
  {
    if (MPI_File_seek(pstXmp_file->fh,
                      (MPI_Offset)0,
                      MPI_SEEK_END) != MPI_SUCCESS)
    {
      return -1;
    }

    pstXmp_file->is_append = 0x00;
  }

  // write
  if (MPI_File_write_all(pstXmp_file->fh, buffer, size * count, MPI_BYTE, &status) != MPI_SUCCESS)
  {
    return -1;
  }

  // number of bytes written
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return writeCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_darray_unpack                                  */
/*  DESCRIPTION   : xmp_fread_darray_unpack reads data cooperatively to the  */
/*                  global array from the position of the shared file        */
/*                  pointer. Data is read from the file to distributed apd   */
/*                  limited to range rp.                                     */
/*  ARGUMENT      : fp[IN] : file structure.                                 */
/*                  apd[IN/OUT] : distributed array descriptor.              */
/*                  rp[IN] : range descriptor.                               */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  loading data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
#ifdef ORIGINAL
int xmp_fread_darray_unpack(fp, ap, rp)
     xmp_file_t *fp;
     xmp_array_t ap;
     xmp_range_t *rp;
#else /* RIST */
int xmp_fread_darray_unpack(fp, apd, rp)
     xmp_file_t *fp;
     xmp_desc_t apd;
     xmp_range_t *rp;
#endif
{
  MPI_Status    status;
  char         *array_addr;
  char         *buf=NULL;
  char         *cp;
  int          *lb=NULL;
  int          *ub=NULL;
  int          *step=NULL;
  int          *cnt=NULL;
  int           buf_size;
  int           ret=0;
  int           disp;
  int           size;
  int           array_size;
  int           i, j;
#ifdef ORIGINAL
  _XMP_array_t *array_t;
  size_t array_type_size;
#else /* RIST */
  xmp_desc_t tempd = NULL;
  int **bc2_result = NULL;
  size_t array_type_size;
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
  int array_ndim;
#endif

  // check argument
  if (fp == NULL){ ret = -1; goto FunctionExit; }
#ifdef ORIGINAL
  if (ap == NULL){ ret = -1; goto FunctionExit; }
#else /* RIST */
  if (apd == NULL){ ret = -1; goto FunctionExit; }
#endif
  if (rp == NULL){ ret = -1; goto FunctionExit; }

#ifdef ORIGINAL
  array_t = (_XMP_array_t*)ap;
#else /* RIST */
  tempd = xmp_align_template(apd);
  if (tempd == NULL){ ret = -1; goto FunctionExit; }
  array_ndim = xmp_array_ndim(apd);
#endif

#ifdef ORIGINAL
#define RP_DIMS     (rp->dims)
#define RP_LB(i)    (rp->lb[(i)])
#define RP_UB(i)    (rp->ub[(i)])
#define RP_STEP(i)  (rp->step[(i)])
#else /* RIST */
  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ ret = -1; goto FunctionExit; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])
#endif

#ifdef ORIGINAL
#else /* RIST */
  // check number of dimensions
   if (array_ndim != RP_DIMS){ ret = -1; goto FunctionExit; }
#endif

   /* allocate arrays for the number of rotations */
   lb = (int*)malloc(sizeof(int)*RP_DIMS);
   ub = (int*)malloc(sizeof(int)*RP_DIMS);
   step = (int*)malloc(sizeof(int)*RP_DIMS);
   cnt = (int*)malloc(sizeof(int)*RP_DIMS);
   if(!lb || !ub || !step || !cnt){
      ret = -1;
      goto FunctionExit;
   }
#ifdef ORIGINAL
#else /* RIST */
   bc2_result = (int**)malloc(sizeof(int*)*RP_DIMS);
   if(!bc2_result){
      ret = -1;
      goto FunctionExit;
   }
   for(i=0; i<RP_DIMS; i++){ bc2_result[i]=NULL; }
#endif
  
   /* calculate the number of rotations */
   buf_size = 1;
   for(i=0; i<RP_DIMS; i++){
#ifdef ORIGINAL
/*      int par_lower_i = array_t->info[i].par_lower; */
/*      int par_upper_i = array_t->info[i].par_upper; */
/*      int align_manner_i = array_t->info[i].align_manner; */

     _XMP_array_info_t *ai = &(array_t->info[i]);
     int par_lower_i = ai->par_lower;
     int par_upper_i = ai->par_upper;
     int align_manner_i = ai->align_manner;
#else /* RIST */
     int par_lower_i = xmp_array_gcllbound(apd, i+1);
     int par_upper_i = xmp_array_gclubound(apd, i+1);
     int align_manner_i = xmp_align_format(apd, i+1);
#endif
      /* error check */
      if(RP_STEP(i) > 0 && RP_LB(i) > RP_UB(i)){
         ret = -1;
         goto FunctionExit;
      }
      if(RP_STEP(i) < 0 && RP_LB(i) < RP_UB(i)){
         ret = -1;
         goto FunctionExit;
      }
      if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
          align_manner_i == _XMP_N_ALIGN_DUPLICATION) {
         lb[i] = RP_LB(i);
         ub[i] = RP_UB(i);
         step[i] = RP_STEP(i);
#ifdef ORIGINAL
#else /* RIST */
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
#endif
      } else if(align_manner_i == _XMP_N_ALIGN_BLOCK){
         if(RP_STEP(i) > 0){
            if(par_upper_i < RP_LB(i) ||
               par_lower_i > RP_UB(i)){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (par_lower_i > RP_LB(i))?
                  RP_LB(i)+((par_lower_i-1-RP_LB(i))/RP_STEP(i)+1)*RP_STEP(i):
                  RP_LB(i);
               ub[i] = (par_upper_i < RP_UB(i)) ?
                  par_upper_i:
                  RP_UB(i);
               step[i] = RP_STEP(i);
            }
         } else {
            if(par_upper_i < RP_UB(i) ||
               par_lower_i > RP_LB(i)){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (par_upper_i < RP_LB(i))?
                  RP_LB(i)-((RP_LB(i)-par_upper_i-1)/RP_STEP(i)-1)*RP_STEP(i):
                  RP_LB(i);
               ub[i] = (par_lower_i > RP_UB(i))?
                  par_lower_i:
                  RP_UB(i);
               step[i] = RP_STEP(i);
            }
         }
#ifdef ORIGINAL
#else /* RIST */
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
#endif
#ifdef ORIGINAL
#else /* RIST */
      } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC ||
		align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	int bw_i = xmp_dist_size(tempd, i+1);
	if (bw_i <= 0){
	  _XMP_fatal("xmp_fread_darray_unpack: invalid block width");
	  ret = -1; goto FunctionExit; 
	}else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	  _XMP_fatal("xmp_fread_darray_unpack: invalid block width for cyclic distribution");
	  ret = -1; goto FunctionExit; 
	}
	int cycle_i = xmp_dist_stride(tempd, i+1);
	int ierr = _xmp_io_block_cyclic_2(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
					  RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
					  (int *)(&cnt[i]) /* out */, (int **)(&bc2_result[i]) /* out */);
	if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
#endif
      } else {
         ret = -1;
         goto FunctionExit;
      }
#ifdef ORIGINAL
      cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];
#else /* RIST */
/*       cnt[i] = (ub[i]-lb[i]+step[i])/step[i]; */
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];
#endif
   }
  
#ifdef ORIGINAL
   array_type_size = array_t->type_size;
#else /* RIST */
   array_type_size = xmp_array_type_size(apd);
#endif
   /* allocate buffer */
   if(buf_size == 0){
      buf = (char*)malloc(array_type_size);
   } else {
      buf = (char*)malloc(buf_size * array_type_size);
   }
   if(!buf){
      ret = -1;
      goto FunctionExit;
   }

   // write
   if(buf_size > 0){
      if (MPI_File_read(fp->fh, buf, buf_size * array_type_size, MPI_BYTE, &status) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
      
      // number of bytes written
      if (MPI_Get_count(&status, MPI_BYTE, &ret) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
   } else {
      ret = 0;
   }

   /* unpack data */
   cp = buf;
#ifdef ORIGINAL
   array_addr = (char*)(*array_t->array_addr_p);
#else /* RIST */
   array_addr = (char*)(*(xmp_array_laddr(apd)));
#endif
   for(j=0; j<buf_size; j++){
     disp = 0;
     size = 1;
     array_size = 1;
     for(i=RP_DIMS-1; i>=0; i--){
#ifdef ORIGINAL
/*        int par_lower_i = array_t->info[i].par_lower; */
/*        int align_manner_i = array_t->info[i].align_manner; */
/*        int ser_size_i = array_t->info[i].ser_size; */
/*        int alloc_size_i = array_t->info[i].alloc_size; */
/*        int local_lower_i = array_t->info[i].local_lower; */

       _XMP_array_info_t *ai = &(array_t->info[i]);
       int par_lower_i = ai->par_lower;
       int align_manner_i = ai->align_manner;
       int ser_size_i = ai->ser_size;
       int alloc_size_i = ai->alloc_size;
       int local_lower_i = ai->local_lower;
#else /* RIST */
       int par_lower_i = xmp_array_gcllbound(apd, i+1);
       int align_manner_i = xmp_align_format(apd, i+1);
       int ser_size_i = xmp_array_gsize(apd, i+1);
       int alloc_size_i = xmp_array_lsize(apd, i+1);
       int local_lower_i = xmp_array_lcllbound(apd, i+1);
#endif
       ub[i] = (j/size)%cnt[i];
       if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
	   align_manner_i == _XMP_N_ALIGN_DUPLICATION) {
	 disp += (lb[i]+ub[i]*step[i])*array_size;
	 array_size *= ser_size_i;

       } else if(align_manner_i == _XMP_N_ALIGN_BLOCK){
	 disp += (lb[i] + ub[i]*step[i] + local_lower_i - par_lower_i)*array_size;
	 array_size *= alloc_size_i;

#ifdef ORIGINAL
#else /* RIST */
       } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC ||
		 align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	 int local_index;
	 int ierr = _xmp_io_block_cyclic_3(ub[i] /* in */, bc2_result[i] /* in */,
				       &local_index /* out */);
	 if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
	 disp += (local_index + local_lower_i) * array_size;
	 array_size *= alloc_size_i;
#endif
       } /* align_manner_i */
       size *= cnt[i];
     } /* i */
     disp *= array_type_size;
     memcpy(array_addr+disp, cp, array_type_size);
     cp += array_type_size;
   } /* j */

 FunctionExit:
   if(buf) free(buf);
   if(lb) free(lb);
   if(ub) free(ub);
   if(step) free(step);
   if(cnt) free(cnt);
#ifdef ORIGINAL
#else /* RIST */
   if(bc2_result){
     for(i=0; i<RP_DIMS; i++){ if(bc2_result[i]){ free(bc2_result[i]); } }
   }
#endif

   return ret;
#undef RP_DIMS
#undef RP_LB
#undef RP_UB
#undef RP_STEP
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_darray_all                                     */
/*  DESCRIPTION   : xmp_fread_darray_all reads data cooperatively to the     */
/*                  global array from the position of the shared file        */
/*                  pointer. Data is read from the file to distributed apd   */
/*                  limited to range rp.                                     */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  apd[IN/OUT] : distributed array descriptor.              */
/*                  rp[IN] : range descriptor.                               */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  loading data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
#ifdef ORIGINAL
size_t xmp_fread_darray_all(xmp_file_t  *pstXmp_file,
                            xmp_array_t  ap,
                            xmp_range_t *rp)
#else /* RIST */
size_t xmp_fread_darray_all(xmp_file_t  *pstXmp_file,
                            xmp_desc_t  apd,
                            xmp_range_t *rp)
#endif
{
  MPI_Status status;        // MPI status
  int readCount;            // read bytes
  int mpiRet;               // return value of MPI functions
  int lower;                // lower bound accessed by this node
  int upper;                // upper bound accessed by this node
  int continuous_size;      // continuous size
  int space_size;           // space size
  int total_size;           // total size
  int type_size;
  MPI_Aint tmp1, tmp2;
  MPI_Datatype dataType[2];
  int i = 0;
#ifdef ORIGINAL
  _XMP_array_t *XMP_array_t;
#else /* RIST */
  xmp_desc_t tempd;
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
#endif
  int array_ndim;
  size_t array_type_size;

/* #ifdef DEBUG */
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
/* #endif */

  // check argument
  if (pstXmp_file == NULL) { return -1; }
#ifdef ORIGINAL
  if (ap == NULL)          { return -1; }
#else /* RIST */
  if (apd == NULL)         { return -1; }
#endif
  if (rp == NULL)          { return -1; }

#ifdef ORIGINAL
  XMP_array_t = (_XMP_array_t*)ap; 
  array_ndim = XMP_array_t->dim;
  array_type_size = XMP_array_t->type_size;
#else /* RIST */
  tempd = xmp_align_template(apd);
  if (tempd == NULL){ return -1; }
  array_ndim = xmp_array_ndim(apd);
  array_type_size = xmp_array_type_size(apd);
#endif

#ifdef ORIGINAL
#define RP_DIMS     (rp->dims)
#define RP_LB(i)    (rp->lb[(i)])
#define RP_UB(i)    (rp->ub[(i)])
#define RP_STEP(i)  (rp->step[(i)])
#else /* RIST */
  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ return -1; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])
#endif

  // check number of dimensions
  if (array_ndim != RP_DIMS) { return -1; }

  /* case unpack is required */
  for (i = RP_DIMS - 1; i >= 0; i--){
     if(RP_STEP(i) < 0){
#ifdef ORIGINAL
        int ret = xmp_fread_darray_unpack(pstXmp_file, ap, rp);
#else /* RIST */
        int ret = xmp_fread_darray_unpack(pstXmp_file, apd, rp);
#endif
        return ret;
     }
  }

#ifdef DEBUG
printf("READ(%d/%d) dims=%d\n", rank, nproc, RP_DIMS);
#endif

  // create basic data type
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &dataType[0]);

  // loop for each dimension
  for (i = RP_DIMS - 1; i >= 0; i--)
  {
#ifdef ORIGINAL
/*     int par_lower_i = XMP_array_t->info[i].par_lower; */
/*     int par_upper_i = XMP_array_t->info[i].par_upper; */
/*     int align_manner_i = XMP_array_t->info[i].align_manner; */
/*     int local_lower_i = XMP_array_t->info[i].local_lower; */
/*     int alloc_size_i = XMP_array_t->info[i].alloc_size; */

    _XMP_array_info_t *ai = &(XMP_array_t->info[i]);
    int par_lower_i = ai->par_lower;
    int par_upper_i = ai->par_upper;
    int align_manner_i = ai->align_manner;
    int local_lower_i = ai->local_lower;
    int alloc_size_i = ai->alloc_size;
#else /* RIST */
    int par_lower_i = xmp_array_gcllbound(apd, i+1);
    int par_upper_i = xmp_array_gclubound(apd, i+1);
    int align_manner_i = xmp_align_format(apd, i+1);
    int local_lower_i = xmp_array_lcllbound(apd, i+1);
    int alloc_size_i = xmp_array_lsize(apd, i+1);
#endif
#ifdef DEBUG
printf("READ(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
       rank, nproc, RP_LB(i),  RP_UB(i), RP_STEP(i));
printf("READ(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
       rank, nproc, par_lower_i, par_upper_i);
#endif
    // no distribution
    if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
        align_manner_i == _XMP_N_ALIGN_DUPLICATION)
    {
      // upper after distribution < lower
      if (par_upper_i < RP_LB(i)) { return -1; }
      // lower after distribution > upper
      if (par_lower_i > RP_UB(i)) { return -1; }

      // incremnet is negative
      if ( RP_STEP(i) < 0)
      {
      }
      // incremnet is positive
      else
      {
        // continuous size
        continuous_size = (RP_UB(i) - RP_LB(i)) / RP_STEP(i) + 1;

        // get extent of data type
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  
        type_size = (int)tmp2;

        // create basic data type
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * RP_STEP(i),
                                         dataType[0],
                                         &dataType[1]);

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[0]);

        // on error in MPI_Type_create_hvector
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // total size
        total_size
          = (par_upper_i
          -  par_lower_i + 1)
          *  type_size;

        // space size
        space_size
          = (RP_LB(i) - par_lower_i)
          * type_size;

        // create new file type
        mpiRet = MPI_TYPE_CREATE_RESIZED1(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // on error in MPI_Type_create_resized1
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("READ(%d/%d) NOT_ALIGNED\n", rank, nproc);
printf("READ(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
printf("READ(%d/%d) space_size=%d\n", rank, nproc, space_size);
printf("READ(%d/%d) total_size=%d\n", rank, nproc, total_size);
#endif
      }
    }
     // block distribution
    else if (align_manner_i == _XMP_N_ALIGN_BLOCK)
    {
      // increment is negative
      if ( RP_STEP(i) < 0)
      {
      }
      // increment is positive
      else
      {
        // upper after distribution < lower
        if (par_upper_i < RP_LB(i))
        {
          continuous_size = 0;
        }
        // lower after distribution > upper
        else if (par_lower_i > RP_UB(i))
        {
          continuous_size = 0;
        }
        // other
        else
        {
          // lower in this node
          lower
            = (par_lower_i > RP_LB(i)) ?
              RP_LB(i) + ((par_lower_i - 1 - RP_LB(i)) / RP_STEP(i) + 1) * RP_STEP(i)
            : RP_LB(i);

          // upper in this node
          upper
            = (par_upper_i < RP_UB(i)) ?
               par_upper_i : RP_UB(i);

          // continuous size
          continuous_size = (upper - lower + RP_STEP(i)) / RP_STEP(i);

/* 	  printf("fread_darray_all: rank = %d: lower = %d  upper = %d  continuous_size = %d\n", */
/* 		 rank, lower, upper, continuous_size); */
        }

        // get extent of data type
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  
        type_size = (int)tmp2;

/* 	printf("fread_darray_all: rank = %d: type_size = %d   RP_STEP(i) = %d\n", */
/* 	       rank,type_size, RP_STEP(i)); */

        // create basic data type
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * RP_STEP(i),
                                         dataType[0],
                                         &dataType[1]);

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[0]);

        // on error in MPI_Type_create_hvector
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // space size
        space_size
          = (local_lower_i 
          + (lower - par_lower_i))
          * type_size;

        // total size
        total_size = (alloc_size_i)* type_size;

        // create new file type
        mpiRet = MPI_TYPE_CREATE_RESIZED1(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // on error in MPI_Type_create_resized1
        if (mpiRet != MPI_SUCCESS) { return -1; }

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[1]);

/* 	printf("fread_darray_all: rank = %d:  space_size = %d  total_size = %d\n", */
/* 	       rank,space_size,total_size); */
#ifdef DEBUG
printf("READ(%d/%d) ALIGN_BLOCK\n", rank, nproc);
printf("READ(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
printf("READ(%d/%d) space_size=%d\n", rank, nproc, space_size);
printf("READ(%d/%d) total_size=%d\n", rank, nproc, total_size);
printf("READ(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc, lower, upper);
#endif
      }
    }
#ifdef ORIGINAL
    // cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC)
    {
      return -1;
    }
#else /* RIST */
    // cyclic or block-cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC ||
	     align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC)
    {
      int bw_i = xmp_dist_size(tempd, i+1);
      if (bw_i <= 0){
	_XMP_fatal("xmp_fread_darray_all: invalid block width");
	return -1;
      }else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	_XMP_fatal("xmp_fread_darray_all: invalid block width for cyclic distribution");
	return -1;
      }
      int cycle_i = xmp_dist_stride(tempd, i+1);
      int ierr = _xmp_io_block_cyclic_1(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
					RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
					local_lower_i /* in */,
					alloc_size_i /* in */,
					dataType[0] /* in */,
					&dataType[1] /* out */);
      if (ierr != MPI_SUCCESS) { return -1; }
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];
    }
#endif
    // other
    else
    {
      _XMP_fatal("xmp_fread_darray_all: invalid align manner");
      return -1;
    } /* align_manner_i */
  }

  // commit
  mpiRet = MPI_Type_commit(&dataType[0]);

  // on erro in commit
  if (mpiRet != MPI_SUCCESS) { return 1; }
  
#ifdef ORIGINAL
  char *array_addr = (char*)(*XMP_array_t->array_addr_p);
#else /* RIST */
  char *array_addr = (char*)(*(xmp_array_laddr(apd)));
#endif

  // read
  MPI_Type_size(dataType[0], &type_size);
/*   printf("fread_darray_all: type_size = %d\n",type_size); */

  if(type_size > 0){
     if (MPI_File_read(pstXmp_file->fh,
                       array_addr,
                       1,
                       dataType[0],
                       &status)
         != MPI_SUCCESS)
        {
           return -1;
        }
  } else {
     return 0;
  }
  
  // free MPI_Datatype out of use
  MPI_Type_free(&dataType[0]);

  // number of bytes read
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }
  return readCount;
#undef RP_DIMS
#undef RP_LB
#undef RP_UB
#undef RP_STEP
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_darray_pack                                   */
/*  DESCRIPTION   : xmp_fwrite_darray_pack writes data cooperatively from    */
/*                  the global array to the position of the shared file      */
/*                  pointer. Data is written from distributed apd limited to */
/*                  range rp to the file.                                    */
/*  ARGUMENT      : fp[IN] : file structure.                                 */
/*                  apd[IN] : distributed array descriptor.                  */
/*                  rp[IN] : range descriptor.                               */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  storing data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
#ifdef ORIGINAL
int xmp_fwrite_darray_pack(fp, ap, rp)
     xmp_file_t *fp;
     xmp_array_t ap;
     xmp_range_t *rp;
#else /* RIST */
int xmp_fwrite_darray_pack(fp, apd, rp)
     xmp_file_t *fp;
     xmp_desc_t  apd;
     xmp_range_t *rp;
#endif
{
   MPI_Status    status;
   char         *array_addr;
   char         *buf=NULL;
   char         *cp;
   int          *lb=NULL;
   int          *ub=NULL;
   int          *step=NULL;
   int          *cnt=NULL;
   int           buf_size;
   int           ret=0;
   int           disp;
   int           size;
   int           array_size;
   int           i, j;
   size_t array_type_size;
#ifdef ORIGINAL
   _XMP_array_t *array_t;
#else /* RIST */
   xmp_desc_t tempd = NULL;
   int **bc2_result = NULL;
   int rp_dims;
   int *rp_lb_addr = NULL;
   int *rp_ub_addr = NULL;
   int *rp_step_addr = NULL;
   int array_ndim;
#endif

#ifdef ORIGINAL
   array_t = (_XMP_array_t*)ap;
   array_type_size = array_t->type_size;
#else /* RIST */
   tempd = xmp_align_template(apd);
   if (tempd == NULL){ ret = -1; goto FunctionExit; }
   array_type_size = xmp_array_type_size(apd);
   array_ndim = xmp_array_ndim(apd);
#endif

#ifdef ORIGINAL
#define RP_DIMS     (rp->dims)
#define RP_LB(i)    (rp->lb[(i)])
#define RP_UB(i)    (rp->ub[(i)])
#define RP_STEP(i)  (rp->step[(i)])
#else /* RIST */
   rp_dims = _xmp_range_get_dims(rp);
   rp_lb_addr = _xmp_range_get_lb_addr(rp);
   rp_ub_addr = _xmp_range_get_ub_addr(rp);
   rp_step_addr = _xmp_range_get_step_addr(rp);
   if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ ret = -1; goto FunctionExit; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])
#endif

#ifdef ORIGINAL
#else /* RIST */
  // check number of dimensions
   if (array_ndim != RP_DIMS){ ret = -1; goto FunctionExit; }
#endif

   /* allocate arrays for the number of rotations */
   lb = (int*)malloc(sizeof(int)*RP_DIMS);
   ub = (int*)malloc(sizeof(int)*RP_DIMS);
   step = (int*)malloc(sizeof(int)*RP_DIMS);
   cnt = (int*)malloc(sizeof(int)*RP_DIMS);
   if(!lb || !ub || !step || !cnt){
      ret = -1;
      goto FunctionExit;
   }
#ifdef ORIGINAL
#else /* RIST */
   bc2_result = (int**)malloc(sizeof(int*)*RP_DIMS);
   if(!bc2_result){
      ret = -1;
      goto FunctionExit;
   }
   for(i=0; i<RP_DIMS; i++){ bc2_result[i]=NULL; }
#endif
  
   /* calculate the number of rotaions */
   buf_size = 1;
   for(i=0; i<RP_DIMS; i++){
#ifdef ORIGINAL
/*      int par_lower_i = array_t->info[i].par_lower; */
/*      int par_upper_i = array_t->info[i].par_upper; */
/*      int align_manner_i = array_t->info[i].align_manner; */

     _XMP_array_info_t *ai = &(array_t->info[i]);
     int par_lower_i = ai->par_lower;
     int par_upper_i = ai->par_upper;
     int align_manner_i = ai->align_manner;
#else /* RIST */
     int par_lower_i = xmp_array_gcllbound(apd, i+1);
     int par_upper_i = xmp_array_gclubound(apd, i+1);
     int align_manner_i = xmp_align_format(apd, i+1);
#endif
      /* error check */
      if(RP_STEP(i) > 0 && RP_LB(i) > RP_UB(i)){
         ret = -1;
         goto FunctionExit;
      }
      if(RP_STEP(i) < 0 && RP_LB(i) < RP_UB(i)){
         ret = -1;
         goto FunctionExit;
      }
      if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
          align_manner_i == _XMP_N_ALIGN_DUPLICATION) {
         lb[i] = RP_LB(i);
         ub[i] = RP_UB(i);
         step[i] = RP_STEP(i);
#ifdef ORIGINAL
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
#else /* RIST */
#endif
  
      } else if(align_manner_i == _XMP_N_ALIGN_BLOCK){
         if(RP_STEP(i) > 0){
            if(par_upper_i < RP_LB(i) ||
               par_lower_i > RP_UB(i)){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (par_lower_i > RP_LB(i))?
                  RP_LB(i)+((par_lower_i-1-RP_LB(i))/RP_STEP(i)+1)*RP_STEP(i):
                  RP_LB(i);
               ub[i] = (par_upper_i < RP_UB(i)) ?
                  par_upper_i:
                  RP_UB(i);
               step[i] = RP_STEP(i);
            }
         } else {
            if(par_upper_i < RP_UB(i) ||
               par_lower_i > RP_LB(i)){
               lb[i] = 1;
               ub[i] = 0;
               step[i] = 1;
            } else {
               lb[i] = (par_upper_i < RP_LB(i))?
                  RP_LB(i)-((RP_LB(i)-par_upper_i-1)/RP_STEP(i)-1)*RP_STEP(i):
                  RP_LB(i);
               ub[i] = (par_lower_i > RP_UB(i))?
                  par_lower_i:
                  RP_UB(i);
               step[i] = RP_STEP(i);
            }
         }
#ifdef ORIGINAL
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
#else /* RIST */
#endif

#ifdef ORIGINAL
#else /* RIST */
      } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC ||
		align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	int bw_i = xmp_dist_size(tempd, i+1);
	if (bw_i <= 0){
	  _XMP_fatal("xmp_fwrite_darray_pack: invalid block width");
	  ret = -1; goto FunctionExit;
	}else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	  _XMP_fatal("xmp_fwrite_darray_pack: invalid block width for cyclic distribution");
	  ret = -1; goto FunctionExit;
	}
	int cycle_i = xmp_dist_stride(tempd, i+1);
	int ierr = _xmp_io_block_cyclic_2(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
					  RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
					  (int *)&(cnt[i]) /* out */, (int **)(&bc2_result[i]) /* out */);
	if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
#endif
      } else {
         ret = -1;
         goto FunctionExit;
      }
#ifdef ORIGINAL
      cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];
#else /* RIST */
/*       cnt[i] = (ub[i]-lb[i]+step[i])/step[i]; */
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];
#endif

#ifdef DEBUG
      fprintf(stderr, "dim = %d: (%d: %d: %d) %d\n", i, lb[i], ub[i], step[i], buf_size);
#endif
   }
  
   /* allocate buffer */
   if(buf_size == 0){
      buf = (char*)malloc(array_type_size);
      fprintf(stderr, "size = 0\n");
   } else {
      buf = (char*)malloc(buf_size * array_type_size);
   }
   if(!buf){
      ret = -1;
      goto FunctionExit;
   }

   /* pack data */
   cp = buf;
#ifdef ORIGINAL
   array_addr = (char*)(*array_t->array_addr_p);
#else /* RIST */
   array_addr = (char*)(*(xmp_array_laddr(apd)));
#endif
   for(j=0; j<buf_size; j++){
     disp = 0;
     size = 1;
     array_size = 1;
     for(i=RP_DIMS-1; i>=0; i--){
#ifdef ORIGINAL
/*        int par_lower_i = array_t->info[i].par_lower; */
/*        int align_manner_i = array_t->info[i].align_manner; */
/*        int local_lower_i = array_t->info[i].local_lower; */
/*        int ser_size_i = array_t->info[i].ser_size; */
/*        int alloc_size_i = array_t->info[i].alloc_size; */

       _XMP_array_info_t *ai = &(array_t->info[i]);
       int par_lower_i = ai->par_lower;
       int align_manner_i = ai->align_manner;
       int local_lower_i = ai->local_lower;
       int ser_size_i = ai->ser_size;
       int alloc_size_i = ai->alloc_size;
#else /* RIST */
       int par_lower_i = xmp_array_gcllbound(apd, i+1);
       int align_manner_i = xmp_align_format(apd, i+1);
       int local_lower_i = xmp_array_lcllbound(apd, i+1);
       int ser_size_i = xmp_array_gsize(apd, i+1);
       int alloc_size_i = xmp_array_lsize(apd, i+1);
#endif
       ub[i] = (j/size)%cnt[i];
       if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
	   align_manner_i == _XMP_N_ALIGN_DUPLICATION) {
	 disp += (lb[i]+ub[i]*step[i])*array_size;
	 array_size *= ser_size_i;

       } else if(align_manner_i == _XMP_N_ALIGN_BLOCK){
	 disp += (lb[i]+ub[i]*step[i] + local_lower_i - par_lower_i)*array_size;
	 array_size *= alloc_size_i;

#ifdef ORIGINAL
#else /* RIST */
       } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC ||
		 align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	 int local_index;
	 int ierr = _xmp_io_block_cyclic_3(ub[i] /* in */, bc2_result[i] /* in */,
				       &local_index /* out */);
	 if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
	 disp += (local_index + local_lower_i) * array_size;
	 array_size *= alloc_size_i;
#endif
       } /* align_manner_i */
       size *= cnt[i];
     } /* i */
     disp *= array_type_size;
     memcpy(cp, array_addr+disp, array_type_size);
     cp += array_type_size;
   } /* j */

  // write
   if(buf_size > 0){
      if (MPI_File_write(fp->fh, buf, buf_size * array_type_size, MPI_BYTE, &status) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
      
      // number of bytes written
      if (MPI_Get_count(&status, MPI_BYTE, &ret) != MPI_SUCCESS) {
         ret = -1;
         goto FunctionExit;
      }
   } else {
      ret = 0;
   }
  
 FunctionExit:
   if(buf) free(buf);
   if(lb) free(lb);
   if(ub) free(ub);
   if(step) free(step);
   if(cnt) free(cnt);
#ifdef ORIGINAL
#else /* RIST */
   if(bc2_result){
     for(i=0; i<RP_DIMS; i++){ if(bc2_result[i]){ free(bc2_result[i]); } }
   }
#endif

   return ret;
#undef RP_DIMS
#undef RP_LB
#undef RP_UB
#undef RP_STEP
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_darray_all                                    */
/*  DESCRIPTION   : xmp_fwrite_darray_all writes data cooperatively from     */
/*                  the global array to the position of the shared file      */
/*                  pointer. Data is written from distributed apd limited to */
/*                  range rp to the file.                                    */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  apd[IN] : distributed array descriptor.                  */
/*                  rp[IN] : range descriptor.                               */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  storing data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
#ifdef ORIGINAL
size_t xmp_fwrite_darray_all(xmp_file_t *pstXmp_file,
                             xmp_array_t ap,
                             xmp_range_t *rp)
#else /* RIST */
size_t xmp_fwrite_darray_all(xmp_file_t *pstXmp_file,
                             xmp_desc_t apd,
                             xmp_range_t *rp)
#endif
{
  MPI_Status status;        // MPI status
  int writeCount;           // write btye
  int mpiRet;               // return value of MPI functions
  int lower;                // lower bound accessed by this node
  int upper;                // upper bound accessed by this node
  int continuous_size;      // continuous size
  int space_size;           // space size
  int total_size;           // total size
  int type_size;
  MPI_Aint tmp1, tmp2;
  MPI_Datatype dataType[2];
  int i = 0;
#ifdef ORIGINAL
  _XMP_array_t *XMP_array_t;
#else /* RIST */
  xmp_desc_t tempd;
#endif
#ifdef ORIGINAL
#else /* RIST */
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
#endif
  int array_ndim;
  size_t array_type_size;

/* #ifdef DEBUG */
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
/* #endif */

  // check argument
  if (pstXmp_file == NULL) { return -1101; }
#ifdef ORIGINAL
  if (ap == NULL)          { return -1102; }
#else /* RIST */
  if (apd == NULL)         { return -1103; }
#endif
  if (rp == NULL)          { return -1104; }

#ifdef ORIGINAL
  XMP_array_t = (_XMP_array_t*)ap;
  array_ndim = XMP_array_t->dim;
  array_type_size = XMP_array_t->type_size;
#else /* RIST */
  tempd = xmp_align_template(apd);
  if (tempd == NULL){ return -1105; }
  array_ndim = xmp_array_ndim(apd);
  array_type_size = xmp_array_type_size(apd);
#endif

#ifdef ORIGINAL
#define RP_DIMS     (rp->dims)
#define RP_LB(i)    (rp->lb[(i)])
#define RP_UB(i)    (rp->ub[(i)])
#define RP_STEP(i)  (rp->step[(i)])
#else /* RIST */
  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ return -1106; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])
#endif

  // check number of dimensions
  if (array_ndim != RP_DIMS) { return -1107; }

#ifdef DEBUG
printf("WRITE(%d/%d) dims=%d\n",rank, nproc, RP_DIMS);
#endif

  /* case pack is required */
  for (i = RP_DIMS - 1; i >= 0; i--){
     if(RP_STEP(i) < 0){
#ifdef ORIGINAL
        int ret = xmp_fwrite_darray_pack(pstXmp_file, ap, rp);
#else /* RIST */
        int ret = xmp_fwrite_darray_pack(pstXmp_file, apd, rp);
#endif
        return ret;
     }
  }

  // create basic data type
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &dataType[0]);

  // loop for each dimension
  for (i = RP_DIMS - 1; i >= 0; i--)
  {
#ifdef ORIGINAL
/*     int par_lower_i = XMP_array_t->info[i].par_lower; */
/*     int par_upper_i = XMP_array_t->info[i].par_upper; */
/*     int align_manner_i = XMP_array_t->info[i].align_manner; */
/*     int alloc_size_i = XMP_array_t->info[i].alloc_size; */
/*     int ser_lower_i = XMP_array_t->info[i].ser_lower; */
/*     int ser_upper_i = XMP_array_t->info[i].ser_upper; */
/*     int local_lower_i = XMP_array_t->info[i].local_lower; */
/*     int local_upper_i = XMP_array_t->info[i].local_upper; */
/*     int shadow_size_lo_i = XMP_array_t->info[i].shadow_size_lo; */
/*     int shadow_size_hi_i = XMP_array_t->info[i].shadow_size_hi; */

    _XMP_array_info_t *ai = &(XMP_array_t->info[i]);
    int par_lower_i = ai->par_lower;
    int par_upper_i = ai->par_upper;
    int align_manner_i = ai->align_manner;
    int alloc_size_i = ai->alloc_size;
    int ser_lower_i = ai->ser_lower;
    int ser_upper_i = ai->ser_upper;
    int local_lower_i = ai->local_lower;
    int local_upper_i = ai->local_upper;
    int shadow_size_lo_i = ai->shadow_size_lo;
    int shadow_size_hi_i = ai->shadow_size_hi;
#else /* RIST */
    int par_lower_i = xmp_array_gcllbound(apd, i+1);
    int par_upper_i = xmp_array_gclubound(apd, i+1);
    int align_manner_i = xmp_align_format(apd, i+1);
    int alloc_size_i = xmp_array_lsize(apd, i+1);
    int ser_lower_i = xmp_array_gcglbound(apd, i+1);
    int ser_upper_i = xmp_array_gcgubound(apd, i+1);
    int local_lower_i = xmp_array_lcllbound(apd, i+1);
    int local_upper_i = xmp_array_lclubound(apd, i+1);
    int shadow_size_lo_i = xmp_array_lshadow(apd, i+1);
    int shadow_size_hi_i = xmp_array_ushadow(apd, i+1);
#endif
#ifdef DEBUG
printf("WRITE(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
       rank, nproc, RP_LB(i),  RP_UB(i), RP_STEP(i));
printf("WRITE(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
       rank, nproc, par_lower_i, par_upper_i);
printf("WRITE(%d/%d) (local_lower,local_upper,alloc_size)=(%d,%d,%d)\n",
       rank, nproc, local_lower_i, local_upper_i, alloc_size_i);
printf("WRITE(%d/%d) (shadow_size_lo,shadow_size_hi)=(%d,%d)\n",
       rank, nproc, shadow_size_lo_i, shadow_size_hi_i);
#endif

    // no distribution
    if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
        align_manner_i == _XMP_N_ALIGN_DUPLICATION)
    {
      // upper after distribution < lower
      if (par_upper_i < RP_LB(i)) { return -1108; }
      // lower after distribution > upper
      if (par_lower_i > RP_UB(i)) { return -1109; }

      // incremnet is negative
      if ( RP_STEP(i) < 0)
      {
      }
      // incremnet is positive
      else
      {
        // continuous size
        continuous_size = (RP_UB(i) - RP_LB(i)) / RP_STEP(i) + 1;

        // get extent of data type
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1110; }  
        type_size = (int)tmp2;

        // create basic data type
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * RP_STEP(i),
                                         dataType[0],
                                         &dataType[1]);

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[0]);

        // on error in MPI_Type_contiguous
        if (mpiRet != MPI_SUCCESS) { return -1111; }

        // total size
        total_size
          = (ser_upper_i 
          -  ser_lower_i + 1)
          *  type_size;

        // space size
        space_size
          = (RP_LB(i) - par_lower_i)
          * type_size;

        // create new file type
        mpiRet = MPI_TYPE_CREATE_RESIZED1(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // on error in MPI_Type_create_resized1
        if (mpiRet != MPI_SUCCESS) { return -1112; }

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("WRITE(%d/%d) NOT_ALIGNED\n",rank, nproc);
printf("WRITE(%d/%d) type_size=%d\n",rank, nproc, type_size);
printf("WRITE(%d/%d) continuous_size=%d\n",rank, nproc, continuous_size);
printf("WRITE(%d/%d) space_size=%d\n",rank, nproc, space_size);
printf("WRITE(%d/%d) total_size=%d\n",rank, nproc, total_size);
#endif
      }
    }
     // block distribution
    else if (align_manner_i == _XMP_N_ALIGN_BLOCK)
    {
      // increment is negative
      if ( RP_STEP(i) < 0)
      {
      }
      // increment is positive
      else
      {
        // upper after distribution < lower
        if (par_upper_i < RP_LB(i))
        {
          continuous_size = 0;
        }
        // lower after distribution > upper
        else if (par_lower_i > RP_UB(i))
        {
          continuous_size = 0;
        }
        // other
        else
        {
          // lower in this node
          lower
            = (par_lower_i > RP_LB(i)) ?
              RP_LB(i) + ((par_lower_i - 1 - RP_LB(i)) / RP_STEP(i) + 1) * RP_STEP(i)
            : RP_LB(i);

          // upper in this node
          upper
            = (par_upper_i < RP_UB(i)) ?
               par_upper_i : RP_UB(i);

          // continuous size
          continuous_size = (upper - lower + RP_STEP(i)) / RP_STEP(i);
        }

        // get extent of data type
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
        if (mpiRet !=  MPI_SUCCESS) { return -1113; }  
        type_size = (int)tmp2;
        if(lower > upper){
           type_size = 0;
        }

        // create basic data type
        mpiRet = MPI_Type_create_hvector(continuous_size,
                                         1,
                                         type_size * RP_STEP(i),
                                         dataType[0],
                                         &dataType[1]);

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[0]);

        // on error in MPI_Type_create_hvector
        if (mpiRet != MPI_SUCCESS) { return -1114; }

        // space size
        space_size
          = (local_lower_i
          + (lower - par_lower_i))
          * type_size;

        // total size
        total_size = (alloc_size_i)* type_size;

        // create new file type
        mpiRet = MPI_TYPE_CREATE_RESIZED1(dataType[1],
                                         (MPI_Aint)space_size,
                                         (MPI_Aint)total_size,
                                         &dataType[0]);

        // on error in MPI_Type_create_resized1
        if (mpiRet != MPI_SUCCESS) { return -1115; }

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("WRITE(%d/%d) ALIGN_BLOCK\n",rank, nproc);
printf("WRITE(%d/%d) type_size=%d\n",rank, nproc, type_size);
printf("WRITE(%d/%d) continuous_size=%d\n",rank, nproc, continuous_size);
printf("WRITE(%d/%d) space_size=%d\n",rank, nproc, space_size);
printf("WRITE(%d/%d) total_size=%d\n",rank, nproc, total_size);
printf("WRITE(%d/%d) (lower,upper)=(%d,%d)\n",rank, nproc, lower, upper);
#endif
      }
    }
#ifdef ORIGINAL
    // cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC)
    {
      return -1116;
    }
#else /* RIST */
    // cyclic or block-cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC ||
	     align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC)
    {
      int bw_i = xmp_dist_size(tempd, i+1);
      if (bw_i <= 0){
	_XMP_fatal("xmp_fwrite_darray_all: invalid block width");
	return -1122;
      }else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	_XMP_fatal("xmp_fwrite_darray_all: invalid block width for cyclic distribution");
	return -1122;
      }
      int cycle_i = xmp_dist_stride(tempd, i+1);
      int ierr = _xmp_io_block_cyclic_1(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
					RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
					local_lower_i /* in */,
					alloc_size_i /* in */,
					dataType[0] /* in */,
					&dataType[1] /* out */);
      if (ierr != MPI_SUCCESS) { return -1117; }
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];
    }
#endif
    // other
    else
    {
      _XMP_fatal("xmp_fwrite_darray_all: invalid align manner");
      return -1118;
    } /* align_manner_i */
  }

  // commit
  mpiRet = MPI_Type_commit(&dataType[0]);

  // on error in commit
  if (mpiRet != MPI_SUCCESS) { return 1119; }
 
#ifdef ORIGINAL
  char *array_addr = (*XMP_array_t->array_addr_p);
#else /* RIST */
  char *array_addr = (char*)(*(xmp_array_laddr(apd)));
#endif

  // write
  MPI_Type_size(dataType[0], &type_size);
/*   printf("fwrite_darray_all: type_size = %d\n",type_size); */

  if(type_size > 0){
     if (MPI_File_write(pstXmp_file->fh,
                        array_addr,
                        1,
                        dataType[0],
                        &status)
         != MPI_SUCCESS)
        {
           return -1120;
        }
  } else {
     return 0;
  }
 
  // free MPI_Datatype out of use
  MPI_Type_free(&dataType[0]);

  // number of btyes written
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1121;
  }
/*   printf("------------------------------ fwrite_darray_all: NORMAL END\n"); */
  return writeCount;
#undef RP_DIMS
#undef RP_LB
#undef RP_UB
#undef RP_STEP
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread_shared                                         */
/*  DESCRIPTION   : xmp_fread_shared exclusively reads local data from the   */
/*                  position of the shared file pointer and moves the        */
/*                  position by the length of the data. Local execution.     */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  buffer[OUT] : beginning address of loading variables.    */
/*                  size[IN] : the byte size of a loading elemnent of data.  */
/*                  count[IN] : the number of loading data element.          */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  loading data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fread_shared(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int readCount;

  // check argument
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // read
  if (MPI_File_read_shared(pstXmp_file->fh, buffer, size * count, MPI_BYTE, &status) != MPI_SUCCESS)
  {
    return -1;
  }
  
  // number of bytes read
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return readCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite_shared                                        */
/*  DESCRIPTION   : xmp_fwrite_shared exclusively writes local data to the   */
/*                  position of the shared file pointer and moves the        */
/*                  position by the length of the data. Local execution.     */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  buffer[IN] : beginning address of storing variables.     */
/*                  size[IN] : the byte size of a storing element of data.   */
/*                  count[IN] : the number of storing data element.          */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  storing data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fwrite_shared(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int writeCount;

  // check argument
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // if file open is "r+", then move pointer to end
  if (pstXmp_file->is_append)
  {
    if (MPI_File_seek_shared(pstXmp_file->fh,
                             (MPI_Offset)0,
                             MPI_SEEK_END) != MPI_SUCCESS)
    {
      return -1;
    }

    pstXmp_file->is_append = 0x00;
  }

  // write
  if (MPI_File_write_shared(pstXmp_file->fh,
                            buffer,
                            size * count,
                            MPI_BYTE,
                            &status) != MPI_SUCCESS)
  {
    return -1;
  }

  // number of bytes written
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return writeCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fread                                                */
/*  DESCRIPTION   : xmp_fread reads data from the position of the indivisual */
/*                  file pointer and moves the position by the length of the */
/*                  data. Local execution.                                   */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  buffer[OUT] : beginning address of loading variables.    */
/*                  size[IN] : the byte size of loading element of data.     */
/*                  count[IN] : the number of loading data element.          */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  loading data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fread(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int readCount;

  // check argument
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // read
  if (MPI_File_read(pstXmp_file->fh, buffer, size * count, MPI_BYTE, &status) != MPI_SUCCESS)
  {
    return -1;
  }
  
  // number of bytes read
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return readCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_fwrite                                               */
/*  DESCRIPTION   : xmp_fwrite writes data to the position of the indivisual */
/*                  file pointer and moves the position by the length of the */
/*                  data. Local execution.                                   */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  buffer[IN] : begining address of storing variables.      */
/*                  size[IN] : the byte size of a storing element of data.   */
/*                  count[IN] : the number of storing data element.          */
/*  RETURN VALUES : Upon successful completion, return the byte size of      */
/*                  storing data. Otherwise, negative number shall be        */
/*                  returned.                                                */
/*                                                                           */
/*****************************************************************************/
size_t xmp_fwrite(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int writeCount;

  // check argument
  if (pstXmp_file == NULL) { return -1; }
  if (buffer      == NULL) { return -1; }
  if (size  < 1) { return -1; }
  if (count < 1) { return -1; }

  // if file open is "r+", then move pointer to end
  if (pstXmp_file->is_append)
  {
    if (MPI_File_seek(pstXmp_file->fh,
                      (MPI_Offset)0,
                      MPI_SEEK_END) != MPI_SUCCESS)
    {
      return -1;
    }

    pstXmp_file->is_append = 0x00;
  }

  // write
  if (MPI_File_write(pstXmp_file->fh,
                     buffer,
                     size * count,
                     MPI_BYTE,
                     &status) != MPI_SUCCESS)
  {
    return -1;
  }

  // number of bytes written
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
    return -1;
  }

  return writeCount;
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_file_set_view_all                                    */
/*  DESCRIPTION   : xmp_file_set_view_all sets a file view to the file.      */
/*                  Collective (global) execution. The file view of          */
/*                  distributed apd limited to range rp is set into file     */
/*                  structure.                                               */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  disp[IN] : displacement in byte from the beginning of    */
/*                             the file.                                     */
/*                  apd[IN] : distributed array descriptor.                  */
/*                  rp[IN] : range descriptor.                               */
/*  RETURN VALUES : 0: normal termination.                                   */
/*                  an integer other than 0: abnormal termination.           */
/*                                                                           */
/*****************************************************************************/
#ifdef ORIGINAL
int xmp_file_set_view_all(xmp_file_t  *pstXmp_file,
                          long long    disp,
                          xmp_array_t  ap,
                          xmp_range_t *rp)
#else /* RIST */
int xmp_file_set_view_all(xmp_file_t  *pstXmp_file,
                          long long    disp,
                          xmp_desc_t   apd,
                          xmp_range_t *rp)
#endif
{
  int i = 0;
  int mpiRet;               // return value of MPI functions
  int lower;                // lower bound accessed by this node
  int upper;                // upper bound accessed by this node
  int continuous_size;      // continuous size
  MPI_Datatype dataType[2];
  int type_size;
  MPI_Aint tmp1, tmp2;
#ifdef ORIGINAL
  _XMP_array_t *XMP_array_t;
#else /* RIST */
  xmp_desc_t tempd;
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
#endif
  int array_ndim;
  size_t array_type_size;

/* #ifdef DEBUG */
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
/* #endif */

  // check argument
  if (pstXmp_file == NULL) { return 1001; }
#ifdef ORIGINAL
  if (ap == NULL)          { return 1002; }
#else /* RIST */
  if (apd == NULL)         { return 1002; }
#endif
  if (rp == NULL)          { return 1004; }
  if (disp  < 0)           { return 1005; }

#ifdef ORIGINAL
  array_ndim = XMP_array_t->dim;
  array_type_size = XMP_array_t->type_size;
#else /* RIST */
  tempd = xmp_align_template(apd);
  if (tempd == NULL){ return 1006; }
  array_ndim = xmp_array_ndim(apd);
  array_type_size = xmp_array_type_size(apd);
#endif

#ifdef ORIGINAL
#define RP_DIMS     (rp->dims)
#define RP_LB(i)    (rp->lb[(i)])
#define RP_UB(i)    (rp->ub[(i)])
#define RP_STEP(i)  (rp->step[(i)])
#else /* RIST */
  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ return 1007; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])
#endif

  // check number of dimensions
  if (array_ndim != RP_DIMS) { return 1008; }

#ifdef DEBUG
printf("VIEW(%d/%d) dims=%d\n", rank, nproc, RP_DIMS);
#endif

  // create basic data type
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &dataType[0]);

  // loop for each dimension
  for (i = RP_DIMS - 1; i >= 0; i--)
  {
#ifdef ORIGINAL
/*     int par_lower_i = XMP_array_t->info[i].par_lower; */
/*     int par_upper_i = XMP_array_t->info[i].par_upper; */
/*     int align_manner_i = XMP_array_t->info[i].align_manner; */

    _XMP_array_t *XMP_array_t = (_XMP_array_t*)ap; 
    _XMP_array_info_t *ai = &(XMP_array_t->info[i]);
    int par_lower_i = ai->par_lower;
    int par_upper_i = ai->par_upper;
    int align_manner_i = ai->align_manner;
#else /* RIST */
    int par_lower_i = xmp_array_gcllbound(apd, i+1);
    int par_upper_i = xmp_array_gclubound(apd, i+1);
    int align_manner_i = xmp_align_format(apd, i+1);
#endif
    // get extent of data type
    mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &tmp2);
    if (mpiRet !=  MPI_SUCCESS) { return -1009; }
    type_size = (int)tmp2;

#ifdef DEBUG
printf("VIEW(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
        rank, nproc, RP_LB(i),  RP_UB(i), RP_STEP(i));
printf("VIEW(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
        rank, nproc, par_lower_i, par_upper_i);
#endif
    // no distribution
    if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
        align_manner_i == _XMP_N_ALIGN_DUPLICATION)
    {
      // continuous size
      continuous_size = (RP_UB(i) - RP_LB(i)) / RP_STEP(i) + 1;

      // create basic data type
      mpiRet = MPI_Type_contiguous(continuous_size, dataType[0], &dataType[1]);

      // free MPI_Datatype out of use
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];

      // on error in MPI_Type_contiguous
      if (mpiRet != MPI_SUCCESS) { return 1010; }

#ifdef DEBUG
printf("VIEW(%d/%d) NOT_ALIGNED\n", rank, nproc);
printf("VIEW(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
#endif
    }
    // block distribution
    else if (align_manner_i == _XMP_N_ALIGN_BLOCK)
    {
      int space_size;
      int total_size;

      // increment is positive
      if (RP_STEP(i) >= 0)
      {
        // lower > upper
        if (RP_LB(i) > RP_UB(i))
        {
          return 1011; /* return 1;  *//* MODIFIED */
        }
        // upper after distribution < lower
        else if (par_upper_i < RP_LB(i))
        {
          continuous_size = space_size = 0;
        }
        // lower after distribution > upper
        else if (par_lower_i > RP_UB(i))
        {
          continuous_size = space_size = 0;
        }
        // other
        else
        {
          // lower in this node
          lower
            = (par_lower_i > RP_LB(i)) ?
              RP_LB(i) + ((par_lower_i - 1 - RP_LB(i)) / RP_STEP(i) + 1) * RP_STEP(i)
            : RP_LB(i);

          // upper in this node
          upper
            = (par_upper_i < RP_UB(i)) ?
               par_upper_i : RP_UB(i);

          // continuous size
          continuous_size = (upper - lower) / RP_STEP(i) + 1;

          // space size
          space_size
            = ((lower - RP_LB(i)) / RP_STEP(i)) * type_size;

/* 	  printf("set_view_all: rank = %d: lower = %d  upper = %d  continuous_size = %d  space_size = %d\n", */
/* 		 rank,lower, upper, continuous_size, space_size); */

        }

        // total size
        total_size
          = ((RP_UB(i) - RP_LB(i)) / RP_STEP(i) + 1) * type_size;

        // create basic data type
        mpiRet = MPI_Type_contiguous(continuous_size, dataType[0], &dataType[1]);

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[0]);

        // on error in MPI_Type_contiguous
        if (mpiRet != MPI_SUCCESS) { return 1012; }

        // create new file type
        mpiRet = MPI_TYPE_CREATE_RESIZED1(dataType[1],
                                         space_size,
                                         total_size,
                                         &dataType[0]);

        // on error in MPI_Type_create_resized1
        if (mpiRet != MPI_SUCCESS) { return 1013; }


        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[1]);

/* 	printf("set_view_all: rank = %d: total_size = %d\n",rank,total_size); */

#ifdef DEBUG
printf("VIEW(%d/%d) ALIGN_BLOCK\n", rank, nproc );
printf("VIEW(%d/%d) type_size=%d\n", rank, nproc , type_size);
printf("VIEW(%d/%d) continuous_size=%d\n", rank, nproc , continuous_size);
printf("VIEW(%d/%d) space_size=%d\n", rank, nproc , space_size);
printf("VIEW(%d/%d) total_size=%d\n", rank, nproc , total_size);
printf("VIEW(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc , lower, upper);
printf("\n");
#endif
      }
      // incremnet is negative
      else if (RP_STEP(i) < 0)
      {
        // lower < upper
        if (RP_LB(i) < RP_UB(i))
        {
          return 1014;
        }
        // lower after distribution < upper
        else if (par_lower_i < RP_UB(i))
        {
          continuous_size = space_size = 0;
        }
        // upper after distribution > lower
        else if (par_upper_i > RP_LB(i))
        {
          continuous_size = space_size = 0;
        }
        // other
        else
        {
          // lower in this node
          lower
            = (par_upper_i <  RP_LB(i)) ?
              RP_LB(i) - (( RP_LB(i) - par_upper_i - 1) / RP_STEP(i) - 1) * RP_STEP(i)
            : RP_LB(i);

          // upper in this node
          upper
            = (par_lower_i > RP_UB(i)) ?
               par_lower_i : RP_UB(i);

          // continuous size
          continuous_size = (upper - lower) / RP_STEP(i) + 1;

          // space size
          space_size
            = ((lower - RP_LB(i)) / RP_STEP(i)) * type_size;
        }

        // create basic data type
        mpiRet = MPI_Type_contiguous(continuous_size, dataType[0], &dataType[1]);

        // total size
        total_size
          = ((RP_UB(i) - RP_LB(i)) / RP_STEP(i) + 1) * type_size;

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[0]);

	// on error in MPI_Type_contiguous
        if (mpiRet != MPI_SUCCESS) { return 1015; }

        // create new file type
        mpiRet = MPI_TYPE_CREATE_RESIZED1(dataType[1],
                                         space_size,
                                         total_size,
                                         &dataType[0]);

        // on error in MPI_Type_create_resized1
        if (mpiRet != MPI_SUCCESS) { return 1016; }

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[1]);

#ifdef DEBUG
printf("VIEW(%d/%d) ALIGN_BLOCK\n", rank, nproc);
printf("VIEW(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
printf("VIEW(%d/%d) space_size=%d\n", rank, nproc, space_size);
printf("VIEW(%d/%d) total_size=%d\n", rank, nproc, total_size);
printf("VIEW(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc, lower, upper);
#endif
      }
    }
#ifdef ORIGINAL
    // cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC)
    {
      return 1;
    }
#else /* RIST */
    // cyclic or block-cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC ||
	     align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC)
    {
      int bw_i = xmp_dist_size(tempd, i+1);
      if (bw_i <= 0){
	_XMP_fatal("xmp_file_set_view_all: invalid block width");
	return 1021;
      }else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	_XMP_fatal("xmp_file_set_view_all: invalid block width for cyclic distribution");
	return 1021;
      }
      int cycle_i = xmp_dist_stride(tempd, i+1);
      int ierr = _xmp_io_block_cyclic_0(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
					RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
					dataType[0] /* in */,
					&dataType[1] /* out */);
      if (ierr != MPI_SUCCESS) { return -1017; }
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];
    }
#endif
    // other
    else
    {
/*       printf("set_view_all: align_manner_i = %d\n",align_manner_i); */
      _XMP_fatal("xmp_file_set_view_all: invalid align manner");
      return 1018;
    } /* align_manner_i */
  }

  // commit
  mpiRet = MPI_Type_commit(&dataType[0]);

  // on erro in commit
  if (mpiRet != MPI_SUCCESS) { return 1019; }
  
  // set view
  mpiRet = MPI_File_set_view(pstXmp_file->fh,
                             (MPI_Offset)disp,
                             MPI_BYTE,
                             dataType[0],
                             "native",
                             MPI_INFO_NULL);


  // free MPI_Datatype out of use
  //MPI_Type_free(&dataType[0]);

  // on erro in set view
  if (mpiRet != MPI_SUCCESS) { return 1020; }
/*   printf("------------------------------ xmp_file_set_view_all: NORMAL END\n"); */
  return 0;
#undef RP_DIMS
#undef RP_LB
#undef RP_UB
#undef RP_STEP
}

/*****************************************************************************/
/*  FUNCTION NAME : xmp_file_clear_view_all                                  */
/*  DESCRIPTION   : xmp_file_clear_view_all clears the file view. Collective */
/*                  (global) execution. The positions of the shared and      */
/*                  indivisual file pointers are set to disp and the         */
/*                  elemental data type and the file type are set to         */
/*                  MPI_BYTE.                                                */
/*  ARGUMENT      : pstXmp_file[IN] : file structure.                        */
/*                  disp[IN] : displacement in byte from the beginning of    */
/*                             the file.                                     */
/*  RETURN VALUES : 0: normal termination.                                   */
/*                  an integer other than 0: abnormal termination.           */
/*                                                                           */
/*****************************************************************************/
int xmp_file_clear_view_all(xmp_file_t  *pstXmp_file, long long disp)
{
  // check argument
  if (pstXmp_file == NULL) { return 1; }
  if (disp  < 0)           { return 1; }

  // initialize view
  if (MPI_File_set_view(pstXmp_file->fh,
                        disp,
                        MPI_BYTE,
                        MPI_BYTE,
                        "native",
                        MPI_INFO_NULL) != MPI_SUCCESS)
  {
    return 1;
  }

  return 0;
}

/*****************************************************************************/
/*  FUNCTION NAME : MPI_Type_create_resized1                                 */
/*                                                                           */
/*****************************************************************************/
#ifdef ORIGINAL
int MPI_Type_create_resized1(MPI_Datatype oldtype,
			     MPI_Aint     lb,
			     MPI_Aint     extent,
			     MPI_Datatype *newtype)
#else /* RIST */
static int MPI_Type_create_resized1(MPI_Datatype oldtype,
			     MPI_Aint     lb,
			     MPI_Aint     extent,
			     MPI_Datatype *newtype)
#endif
{
        int          mpiRet;
        int          b[3];
        MPI_Aint     d[3];
        MPI_Datatype t[3];

        b[0] = b[1] = b[2] = 1;
        d[0] = 0;
        d[1] = lb;
        d[2] = extent;
        t[0] = MPI_LB;
        t[1] = oldtype;
        t[2] = MPI_UB;

        mpiRet = MPI_Type_create_struct(3, b, d, t, newtype);

        return mpiRet;
}
