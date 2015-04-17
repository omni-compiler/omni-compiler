#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
//#define DEBUG
//#define CHECK_POINT

#define MPI_TYPE_CREATE_RESIZED1  MPI_Type_create_resized1

#include "xmp.h"
#include "xmp_constant.h"
#include "xmp_data_struct.h"
#include "xmp_io_sys.h"
#include "xmp_internal.h"
/* ------------------------------------------------------------------ */
extern void _XMP_fatal(char *msg);

static int MPI_Type_create_resized1(MPI_Datatype oldtype,
				    MPI_Aint     lb,
				    MPI_Aint     extent,
				    MPI_Datatype *newtype);
/* ------------------------------------------------------------------ */
static int xmp_array_gclubound_tmp(xmp_desc_t d, int dim)
{
#ifdef CHECK_POINT
    fprintf(stderr, "IO:START(xmp_array_gclubound_tmp)\n");
#endif /* CHECK_POINT */
  int par_upper = xmp_array_gclubound(d, dim);
  int align_manner = xmp_align_format(d, dim);
  if (align_manner == _XMP_N_ALIGN_BLOCK_CYCLIC){
    int bw = xmp_align_size(d, dim);
    if (bw > 1){ par_upper = par_upper + (bw - 1); }
#ifdef CHECK_POINT
    fprintf(stderr, "IO:END  (xmp_array_gclubound_tmp)\n");
#endif /* CHECK_POINT */
    return par_upper;
  }else{
#ifdef CHECK_POINT
    fprintf(stderr, "IO:END  (xmp_array_gclubound_tmp)\n");
#endif /* CHECK_POINT */
    return par_upper;
  }
}
/* ================================================================== */
#define func_m(p, q)  ((q) >= 0 ? -(q)/(p) : ((p) >= 0 ? (-(q)+(p)-1)/(p) : (-(q)-(p)-1)/(p) ))
/* ------------------------------------------------------------------ */
/*****************************************************************************/
/*  FUNCTION NAME : _xmp_io_set_view_block_cyclic                            */
/*  DESCRIPTION   : This function is used to create data type for file view  */
/*                  for cyclic and block-cyclic distribution. This function  */
/*                  is only for internal use in this file.                   */
/*  ARGUMENT      : par_lower[IN] : global lower bound of array.             */
/*                  par_upper[IN] : global upper bound of array.             */
/*                  bw[IN] : block width.                                    */
/*                  cycle[IN] : cycle width.                                 */
/*                  rp_lb[IN] : global lower bound of array section.         */
/*                  rp_ub[IN] : global upper bound of array section.         */
/*                  step[IN] : stride of array section.                      */
/*                  dataType0[IN] : basic data type on input.                */
/*                  _dataType1[OUT] : data type for file view.               */
/*  RETURN VALUES : MPI_SUCCESS: normal termination.                         */
/*                  an integer other than MPI_SUCCESS: abnormal termination. */
/*                                                                           */
/*****************************************************************************/
static int _xmp_io_set_view_block_cyclic
(
 int par_lower /* in */, int par_upper /* in */, int bw /* in */, int cycle /* in */,
 int rp_lb /* in */, int rp_ub /* in */, int step /* in */,
 MPI_Datatype dataType0 /* in */,
 MPI_Datatype *_dataType1 /* out: data type for file view */
)
{
  MPI_Datatype dataType_tmp;
  long continuous_size, space_size, total_size;
  int mpiRet;

  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(_xmp_io_set_view_block_cyclic): rank=%d\n", myrank);
#endif /* CHECK_POINT */
  // get extent of data type
  MPI_Aint tmp1, type_size;
  mpiRet =MPI_Type_get_extent(dataType0, &tmp1, &type_size);
  if (mpiRet !=  MPI_SUCCESS) { return -1113; }  

  int byte_dataType0;
  MPI_Type_size(dataType0, &byte_dataType0);
#ifdef DEBUG
  fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: byte_dataType0=%d  lb=%ld  extent_dataType0=%ld\n",
	 myrank, byte_dataType0, (long)tmp1, (long)type_size);
  fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: par_lower=%d  par_upper=%d  bw=%d  cycle=%d\n",
	 myrank, par_lower, par_upper, bw, cycle);
#endif /* DEBUG */
  if (bw <= 0){ _XMP_fatal("_xmp_io_set_view_block_cyclic: block width must be pisitive."); }
  if (cycle == 0){ _XMP_fatal("_xmp_io_set_view_block_cyclic: cycle must be non-zero."); }
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
      int z_l = MAX(par_upper,rp_ub) + 1; int ib_l = bw; int x_l = 0; /* dummy */ int y_l = 0; /* dummy */
      int z_u = MIN(par_lower,rp_lb) - 1; int ib_u = -1; int x_u = 0; /* dummy */
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0;
	{
	  int x, y, z, w; int q, r, tmp; int bb = -b;
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
	  d = y; x0 = w; 
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (a*b1), (a*k1*x0+par_lower+ib-lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib < z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; }

	int m_u_ib = func_m( (- a*b1), (- a*k1*x0 - par_lower - ib + ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int z_u_ib = a*x_u_ib + par_lower + ib;
	if (z_u_ib > z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; }
      } /* ib */

      if (ib_l == bw || ib_u == -1){ /* set is empty */
	continuous_size = space_size = 0;
	total_size = ((rp_ub-rp_lb)/step + 1) * type_size;

        mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
        if (mpiRet != MPI_SUCCESS) { return 1; }
#ifdef DEBUG
	int byte_dataType_tmp; MPI_Aint lb_dataType_tmp, extent_dataType_tmp;
	MPI_Type_size(dataType_tmp, &byte_dataType_tmp);
	MPI_Type_get_extent(dataType_tmp, &lb_dataType_tmp, &extent_dataType_tmp);
	fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: set is empty: total_size=%ld"
	       "  byte_dataType_tmp=%d  lb=%ld  extent=%ld\n",
	       myrank, total_size,
	       byte_dataType_tmp, (long)lb_dataType_tmp, (long)extent_dataType_tmp);
#endif /* DEBUG */
      }else{
	int mcnt=4;
	mcnt=MAX(mcnt, abs(a1)+2);
	mcnt=MAX(mcnt, abs(bw*b1)+2);
	int b[mcnt]; MPI_Aint d[mcnt]; MPI_Datatype t[mcnt];
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u +1;
	int y_sta = func_m( step, 0 );
#ifdef DEBUG
	int y_end = func_m( (-step), (- rp_lb + rp_ub) );
 	printf("y_sta=%d  y_end=%d\n", y_sta, y_end);
	fprintf(stderr, "---------- myrank=%d: x_l=%d  ib_l=%d  x_u=%d  ib_u=%d ; ista=%d  iend=%d\n",
	       myrank, x_l, ib_l, x_u, ib_u, ista, iend);
#endif /* DEBUG */
	int y_base1 = y_sta;
	MPI_Datatype newtype2a; int byte_newtype2a; MPI_Aint lb_newtype2a, extent_newtype2a;
	MPI_Datatype newtype2aa;int byte_newtype2aa;MPI_Aint lb_newtype2aa,extent_newtype2aa;
	MPI_Datatype newtype2b; int byte_newtype2b; MPI_Aint lb_newtype2b, extent_newtype2b;
	MPI_Datatype newtype2c; int byte_newtype2c; MPI_Aint lb_newtype2c, extent_newtype2c;
	{
	  int cnt=0;
	  int first=1;
	  int i;
	  for (i=ista; i<iend-(iend-ista) %(bw*b1); i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base1 = y; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		b[cnt]=1; d[cnt]=(y - y_base1)*type_size; t[cnt]=dataType0; cnt++;
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2a);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype2a, &byte_newtype2a);
	  MPI_Type_get_extent(newtype2a, &lb_newtype2a, &extent_newtype2a);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype2a: byte_newtype2a=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype2a, (long)lb_newtype2a, (long)extent_newtype2a);
#endif /* DEBUG */
	  if (byte_newtype2a > 0){
	    int count = abs( (iend-ista) / (bw*b1) );
	    int blocklength = 1;
	    MPI_Aint stride = ( a1 )*type_size;
	    mpiRet =  MPI_Type_create_hvector(count,
					      blocklength,
					      stride,
					      newtype2a,
					      &newtype2aa);
	    if (mpiRet != MPI_SUCCESS) { return 1; }
	    MPI_Type_size(newtype2aa, &byte_newtype2aa);
	    MPI_Type_get_extent(newtype2aa, &lb_newtype2aa, &extent_newtype2aa);
#ifdef DEBUG
	    fprintf(stderr, "myrank=%d: newtype2aa: byte_newtype2aa=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		   myrank, byte_newtype2aa, (long)lb_newtype2aa, (long)extent_newtype2aa);
#endif /* DEBUG */
	  }
	}
	int y_base2 = y_sta;
	{
	  int cnt=0;
	  int first=1;
	  int i;
	  for (i=iend-(iend-ista) %(bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base2 = y; first=0; }
	      b[cnt]=1; d[cnt]=(y - y_base2)*type_size;  t[cnt]=dataType0; cnt++;
	    }else{
	    }
	  }/* i */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2b);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype2b, &byte_newtype2b);
	  MPI_Type_get_extent(newtype2b, &lb_newtype2b, &extent_newtype2b);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype2b: byte_newtype2b=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype2b, (long)lb_newtype2b, (long)extent_newtype2b);
#endif /* DEBUG */
	}
#ifdef DEBUG
	fprintf(stderr, "y_base1=%d  y_base2=%d\n", y_base1, y_base2);
#endif /* DEBUG */
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  if (byte_newtype2a > 0){
	    b[cnt]=1; d[cnt]=(y_base1-y_base1)*type_size; t[cnt]=newtype2aa; cnt++;
	    if (byte_newtype2b > 0){
	      b[cnt]=1; d[cnt]=(y_base2-y_base1)*type_size; t[cnt]=newtype2b; cnt++;
	    }
	  }else{
	    if (byte_newtype2b > 0){
	      b[cnt]=1; d[cnt]=(y_base2-y_base2)*type_size; t[cnt]=newtype2b; cnt++;
	    }
	  }
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2c);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype2c, &byte_newtype2c);
	  MPI_Type_get_extent(newtype2c, &lb_newtype2c, &extent_newtype2c);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype2c: byte_newtype2c=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype2c, (long)lb_newtype2c, (long)extent_newtype2c);
#endif /* DEBUG */
	}
	{
	  MPI_Type_free(&newtype2a);
	  if (byte_newtype2a > 0){ MPI_Type_free(&newtype2aa); }
	  MPI_Type_free(&newtype2b);
	  dataType_tmp = newtype2c;
	  continuous_size = byte_newtype2c / type_size;
	  space_size = (y_l - y_sta) * type_size;
	  total_size = ((rp_ub-rp_lb)/step + 1) * type_size;

	  if (extent_newtype2c + space_size > total_size){
#ifdef DEBUG
	    fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: "
		   "extent_newtype2c + space_size > total_size: %ld + %ld > %ld\n",
		   myrank, extent_newtype2c, space_size, total_size);
#endif /* DEBUG */
	    _XMP_fatal("_xmp_io_set_view_block_cyclic: data type is incorrect");
	  }
	}

      } /* ib_l */ /* ib_u */

    } /* if (rp_lb > rp_ub) */
    {
#ifdef DEBUG
      fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: space_size=%ld  total_size=%ld\n",
	     myrank, space_size, total_size);
#endif /* DEBUG */
      int b[3]; MPI_Aint d[3]; MPI_Datatype t[3];
      b[0]=1; d[0]=0;          t[0]=MPI_LB;
      b[1]=1; d[1]=space_size; t[1]=dataType_tmp;
      b[2]=1; d[2]=total_size; t[2]=MPI_UB;
      mpiRet = MPI_Type_create_struct(3, b, d, t, _dataType1);
      if (mpiRet != MPI_SUCCESS) { return 1; }
      int byte_dataType1; MPI_Aint lb_dataType1, extent_dataType1;
      MPI_Type_size(*_dataType1, &byte_dataType1);
      MPI_Type_get_extent(*_dataType1, &lb_dataType1, &extent_dataType1);

#ifdef DEBUG
      fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: space_size=%ld  total_size=%ld:  "
	     "byte_dataType1=%d  lb=%ld  extent=%ld\n",
	     myrank, space_size, total_size,
	     byte_dataType1, (long)lb_dataType1, (long)extent_dataType1);
#endif /* DEBUG */

      if (total_size != extent_dataType1){
#ifdef DEBUG
	fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: total_size != extent_dataType1: %ld  %ld\n",
	       myrank, (long)total_size, (long)extent_dataType1);
#endif /* DEBUG */
	_XMP_fatal("_xmp_io_set_view_block_cyclic: extent is wrong");
      }
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
      int z_l = MIN(par_lower, rp_ub)-1; int ib_l = -1; int x_l = 0; /* dummy */
      int z_u = MAX(par_upper, rp_lb)+1; int ib_u = bw; int x_u = 0; /* dummy */ 
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0;
	{
	  int x, y, z, w; int q, r, tmp; int bb = -b;
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
	  d = y; x0 = w; 
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (-a*b1), (- a*k1*x0 - par_lower - ib + lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib > z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; }

	int m_u_ib = func_m( (a*b1), (a*k1 * x0 + par_lower + ib - ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int z_u_ib = a * x_u_ib + par_lower + ib;
	if (z_u_ib < z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib;}
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
	//int y_sta = func_m( -step, 0 );
	int y_end = func_m( step, (rp_lb - rp_ub) );
#ifdef DEBUG
	fprintf(stderr, "ista=%d  iend=%d  iend-ista=%d  (iend-ista) / (bw*b1)=%d  (iend-ista) %% (bw*b1)=%d\n",
	       ista, iend, iend-ista, (iend-ista) / (bw*b1), (iend-ista) % (bw*b1));
	fprintf(stderr, "iend-(iend-ista) %% (bw*b1)=%d\n", iend-(iend-ista) % (bw*b1));
	fprintf(stderr, "y_sta=%d  y_end=%d\n", y_sta, y_end);
	fprintf(stderr, "y_l=%d  y_u=%d\n", y_l, y_u);
#endif /* DEBUG */
	MPI_Datatype newtype2a; int byte_newtype2a; MPI_Aint lb_newtype2a, extent_newtype2a;
	MPI_Datatype newtype2aa;int byte_newtype2aa;MPI_Aint lb_newtype2aa,extent_newtype2aa;
	MPI_Datatype newtype2b; int byte_newtype2b; MPI_Aint lb_newtype2b, extent_newtype2b;
	MPI_Datatype newtype2c; int byte_newtype2c; MPI_Aint lb_newtype2c, extent_newtype2c;
	int i;
	int y_base1 = y_end;
	{
	  int cnt=0;
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
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2a);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype2a, &byte_newtype2a);
	  MPI_Type_get_extent(newtype2a, &lb_newtype2a, &extent_newtype2a);
#ifdef DEBUG
	  fprintf(stderr, "newtype2a: byte_newtype2a=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 byte_newtype2a, (long)lb_newtype2a, (long)extent_newtype2a);
#endif /* DEBUG */
	  if (byte_newtype2a > 0){
	    int count = abs( (iend-ista) / (bw*b1) );
	    int blocklength = 1;
	    MPI_Aint stride = ( a1 )*type_size;
	    mpiRet =  MPI_Type_create_hvector(count,
					      blocklength,
					      stride,
					      newtype2a,
					      &newtype2aa);
	    if (mpiRet != MPI_SUCCESS) { return 1; }
	    MPI_Type_size(newtype2aa, &byte_newtype2aa);
	    MPI_Type_get_extent(newtype2aa, &lb_newtype2aa, &extent_newtype2aa);
#ifdef DEBUG
	    fprintf(stderr, "myrank=%d: newtype2aa: byte_newtype2aa=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		   myrank, byte_newtype2aa, (long)lb_newtype2aa, (long)extent_newtype2aa);
#endif /* DEBUG */
	  }
	}
	int y_base2 = y_end;
	{
	  int cnt=0;
	  int first=1;
	  for(i=iend-(iend-ista) % (bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
	      int y = (z-rp_lb) / step;
	      if (first){ y_base2 = y; first=0; }
	      b[cnt]=1; d[cnt]=(y_base2 - y)*type_size; t[cnt]=dataType0; cnt++;
	    }else{
	    }
	  }/* i */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2b);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  /* MPI_Type_commit(&newtype2b); */
	  MPI_Type_size(newtype2b, &byte_newtype2b);
	  MPI_Type_get_extent(newtype2b, &lb_newtype2b, &extent_newtype2b);
#ifdef DEBUG
	  fprintf(stderr, "newtype2b: byte_newtype2b=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 byte_newtype2b, (long)lb_newtype2b, (long)extent_newtype2b);
#endif /* DEBUG */
	}
#ifdef DEBUG
	fprintf(stderr, "y_base1=%d  y_base2=%d\n", y_base1, y_base2);
#endif /* DEBUG */
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  if (byte_newtype2a > 0){
	    b[cnt]=1 ; d[cnt]=(y_base1-y_base1)*type_size; t[cnt]=newtype2aa; cnt++;
	    if (byte_newtype2b > 0){
	      b[cnt]=1; d[cnt]=(y_base1-y_base2)*type_size; t[cnt]=newtype2b; cnt++;
	    }
	  }else{
	    if (byte_newtype2b > 0){
	      b[cnt]=1; d[cnt]=(y_base2-y_base2)*type_size; t[cnt]=newtype2b; cnt++;
	    }
	  }
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype2c);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype2c, &byte_newtype2c);
	  MPI_Type_get_extent(newtype2c, &lb_newtype2c, &extent_newtype2c);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype2c: byte_newtype2c=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype2c, (long)lb_newtype2c, (long)extent_newtype2c);
#endif /* DEBUG */
	}
	{
	  MPI_Type_free(&newtype2a);
	  if (byte_newtype2a > 0){ MPI_Type_free(&newtype2aa); }
	  MPI_Type_free(&newtype2b);
	  dataType_tmp = newtype2c;
	  continuous_size = byte_newtype2c / type_size;
	  space_size = (y_end - y_base1) * type_size;
	  total_size = ((rp_ub-rp_lb)/step + 1) * type_size;
	  if (extent_newtype2c + space_size > total_size){
	    _XMP_fatal("_xmp_io_set_view_block_cyclic: data type is incorrect");
	  }
	}

      } /* ib_l */ /* ib_u */

    } /* if (rp_lb < rp_ub) */
    {
#ifdef DEBUG
      fprintf(stderr, "_xmp_io_set_view_block_cyclic: myrank=%d: space_size=%ld  total_size=%ld\n",
	     myrank, space_size, total_size);
#endif /* DEBUG */
      int b[3]; MPI_Aint d[3]; MPI_Datatype t[3];
      b[0]=1; d[0]=0;          t[0]=MPI_LB;
      b[1]=1; d[1]=space_size; t[1]=dataType_tmp;
      b[2]=1; d[2]=total_size; t[2]=MPI_UB;
      mpiRet = MPI_Type_create_struct(3, b, d, t, _dataType1);
      if (mpiRet != MPI_SUCCESS) { return 1; }
      int byte_dataType1; MPI_Aint lb_dataType1, extent_dataType1;
      MPI_Type_size(*_dataType1, &byte_dataType1);
      MPI_Type_get_extent(*_dataType1, &lb_dataType1, &extent_dataType1);
#ifdef DEBUG
      fprintf(stderr, "_xmp_io_set_view_block_cyclic: dataType1: byte_dataType1=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
	     byte_dataType1, (long)lb_dataType1, (long)extent_dataType1);
#endif /* DEBUG */
      if (total_size != extent_dataType1){
	_XMP_fatal("_xmp_io_set_view_block_cyclic: extent is wrong");
      }
      MPI_Type_free(&dataType_tmp);
    }

  } /* if (step < 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else{ return 1; /* dummy */
  }
  /* ++++++++++++++++++++++++++++++++++++++++ */
#ifdef DEBUG
  fprintf(stderr, "------------------------------ _xmp_io_set_view_block_cyclic: NORMAL END: myrank=%d\n",myrank);
#endif /* DEBUG */
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (_xmp_io_set_view_block_cyclic): rank=%d\n", myrank);
#endif /* CHECK_POINT */
  return MPI_SUCCESS;
}

/*****************************************************************************/
/*  FUNCTION NAME : _xmp_io_write_read_block_cyclic                          */
/*  DESCRIPTION   : This function is used to create data type for local array*/
/*                  in memory for cyclic and block-cyclic distribution. This */
/*                  function is only for internal use in this file.          */
/*  ARGUMENT      : par_lower[IN] : global lower bound of array.             */
/*                  par_upper[IN] : global upper bound of array.             */
/*                  bw[IN] : block width.                                    */
/*                  cycle[IN] : cycle width.                                 */
/*                  rp_lb[IN] : global lower bound of array section.         */
/*                  rp_ub[IN] : global upper bound of array section.         */
/*                  step[IN] : stride of array section.                      */
/*                  local_lower[IN] : lower bound of local array.            */
/*                  alloc_size[IN] : size in element of local array.         */
/*                  dataType0[IN] : basic data type on input.                */
/*                  _dataType1[OUT] : data type for local array in memory.   */
/*  RETURN VALUES : MPI_SUCCESS: normal termination.                         */
/*                  an integer other than MPI_SUCCESS: abnormal termination. */
/*                                                                           */
/*****************************************************************************/
static int _xmp_io_write_read_block_cyclic
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
  long continuous_size, space_size, total_size;
  int mpiRet;

  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(_xmp_io_write_read_block_cyclic): rank=%d\n", myrank);
#endif /* CHECK_POINT */
  // get extent of data type
  MPI_Aint tmp1, type_size;
  mpiRet = MPI_Type_get_extent(dataType0, &tmp1, &type_size);
  if (mpiRet !=  MPI_SUCCESS) { return -1113; }  
#ifdef DEBUG
  fprintf(stderr, "_xmp_io_write_read_block_cyclic: myrank=%d:  par_lower=%d  par_upper=%d"
	 "  bw=%d  cycle=%d  alloc_size=%d  type_size=%ld\n",
	 myrank, par_lower, par_upper, bw, cycle, alloc_size, (long)type_size);
#endif /* DEBUG */

  if (bw <= 0){ _XMP_fatal("_xmp_io_write_read_block_cyclic: block width must be pisitive."); }
  if (cycle == 0){ _XMP_fatal("_xmp_io_write_read_block_cyclic: cycle must be non-zero."); }
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
      int z_l = MAX(par_upper,rp_ub) + 1; int ib_l = bw; int x_l = 0; /* dummy */
      int z_u = MIN(par_lower,rp_lb) - 1; int ib_u = -1; int x_u = 0; /* dummy */
      int a1, b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0;
	{
	  int x, y, z, w; int q, r, tmp; int bb = -b;
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
	  d = y; x0 = w;
	}
	a1 = a / d;  b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (a*b1), (a*k1*x0+par_lower+ib-lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib < z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; }

	int m_u_ib = func_m( (- a*b1), (- a*k1*x0 - par_lower - ib + ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int z_u_ib = a*x_u_ib + par_lower + ib;
	if (z_u_ib > z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; }
      } /* ib */
#ifdef DEBUG
      fprintf(stderr, "bw = %d  x_l = %d  x_u = %d  ib_l = %d  ib_u = %d\n",bw,x_l,x_u,ib_l,ib_u);
#endif /* DEBUG */
      if (ib_l == bw || ib_u == -1){ /* set is empty */
	continuous_size = space_size = 0;
	total_size = alloc_size * type_size;

        mpiRet = MPI_Type_contiguous(continuous_size, dataType0, &dataType_tmp);
        if (mpiRet != MPI_SUCCESS) { return 1; }
#ifdef DEBUG
	fprintf(stderr, "_xmp_io_write_read_block_cyclic: myrank=%d: set is empty: \n",
	       myrank);
#endif /* DEBUG */
      }else{ /* ib_l */ /* ib_u */
	int mcnt=4;
	mcnt=MAX(mcnt, abs(a1)+2);
	mcnt=MAX(mcnt, abs(bw*b1)+2);
	int b[mcnt]; MPI_Aint d[mcnt]; MPI_Datatype t[mcnt];
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u +1;
	//int y_sta = func_m( step, 0 );
	//int y_end = func_m( (-step), (- rp_lb + rp_ub) );
#ifdef DEBUG
	fprintf(stderr, "y_sta=%d  y_end=%d\n", y_sta, y_end);
#endif /* DEBUG */
	int i_base1 = ista;
	MPI_Datatype newtype3a; int byte_newtype3a; MPI_Aint lb_newtype3a, extent_newtype3a;
	MPI_Datatype newtype3aa;int byte_newtype3aa;MPI_Aint lb_newtype3aa,extent_newtype3aa;
	MPI_Datatype newtype3b; int byte_newtype3b; MPI_Aint lb_newtype3b, extent_newtype3b;
	MPI_Datatype newtype3c; int byte_newtype3c; MPI_Aint lb_newtype3c, extent_newtype3c;
	{
	  int cnt=0;
	  int first=1;
	  int i;
#ifdef DEBUG
	  fprintf(stderr, "ista = %d  iend = %d  (iend-ista) %(bw*b1) = %d  (bw*b1)\n",
		 ista,iend,(iend-ista) %(bw*b1),(bw*b1));
#endif /* DEBUG */
	  for (i=ista; i<iend-(iend-ista) %(bw*b1); i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
//	      int y = (z-rp_lb) / step;
	      if (first){ i_base1 = i; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		b[cnt]=1; d[cnt]=(i - i_base1)*type_size; t[cnt]=dataType0; cnt++;
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype3a);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype3a, &byte_newtype3a);
	  MPI_Type_get_extent(newtype3a, &lb_newtype3a, &extent_newtype3a);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype3a: byte_newtype3a=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype3a, (long)lb_newtype3a, (long)extent_newtype3a);
#endif /* DEBUG */
	  if (byte_newtype3a > 0){
	    int count = abs( (iend-ista) / (bw*b1) ) ;
	    int blocklength = 1;
	    MPI_Aint stride = ( abs(bw*b1) )*type_size;
	    mpiRet =  MPI_Type_create_hvector(count,
					      blocklength,
					      stride,
					      newtype3a,
					      &newtype3aa);
	    if (mpiRet != MPI_SUCCESS) { return 1; }
	    MPI_Type_size(newtype3aa, &byte_newtype3aa);
	    MPI_Type_get_extent(newtype3aa, &lb_newtype3aa, &extent_newtype3aa);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype3aa: byte_newtype3aa=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype3aa, (long)lb_newtype3aa, (long)extent_newtype3aa);
#endif /* DEBUG */
	  }
	}
	int i_base2 = ista;
	{
	  int cnt=0;
	  int first=1;
	  int i;
	  for (i=iend-(iend-ista) %(bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
//	      int y = (z-rp_lb) / step;
	      if (first){ i_base2 = i; first=0; }
	      b[cnt]=1; d[cnt]=(i - i_base2)*type_size;  t[cnt]=dataType0; cnt++;
	    }else{
	    }
	  }/* i */
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype3b);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype3b, &byte_newtype3b);
	  MPI_Type_get_extent(newtype3b, &lb_newtype3b, &extent_newtype3b);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype3b: byte_newtype3b=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype3b, (long)lb_newtype3b, (long)extent_newtype3b);
#endif /* DEBUG */
	}
#ifdef DEBUG
	fprintf(stderr, "i_base1=%d  i_base2=%d\n", i_base1, i_base2);
#endif /* DEBUG */
	{
	  int cnt=0;
	  b[cnt]=1; d[cnt]=0;      t[cnt]=MPI_LB;  cnt++;
	  if (byte_newtype3a > 0){
	    b[cnt]=1 ; d[cnt]=(i_base1 - i_base1)*type_size; t[cnt]=newtype3aa; cnt++;
	    if (byte_newtype3b > 0){
	      b[cnt]=1; d[cnt]=(i_base2 - i_base1)*type_size; t[cnt]=newtype3b; cnt++;
	    }
	  }else{
	    if (byte_newtype3b > 0){
	      b[cnt]=1; d[cnt]=(i_base2 - i_base2)*type_size; t[cnt]=newtype3b; cnt++;
	    }
	  }
	  mpiRet = MPI_Type_create_struct(cnt, b, d, t, &newtype3c);
	  if (mpiRet != MPI_SUCCESS) { return 1; }
	  MPI_Type_size(newtype3c, &byte_newtype3c);
	  MPI_Type_get_extent(newtype3c, &lb_newtype3c, &extent_newtype3c);
#ifdef DEBUG
	  fprintf(stderr, "myrank=%d: newtype3c: byte_newtype3c=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
		 myrank, byte_newtype3c, (long)lb_newtype3c, (long)extent_newtype3c);
#endif /* DEBUG */
	}
	{
#ifdef DEBUG
	  fprintf(stderr, "alloc_size=%d  type_size=%ld\n", alloc_size, (long)type_size);
#endif /* DEBUG */
	  MPI_Type_free(&newtype3a);
	  if (byte_newtype3a > 0){ MPI_Type_free(&newtype3aa); }
	  MPI_Type_free(&newtype3b);
	  dataType_tmp = newtype3c;
	  continuous_size = byte_newtype3c / type_size;
	  space_size = (ista + local_lower) * type_size;
	  total_size = alloc_size * type_size;
	  if (extent_newtype3c + space_size > total_size){
	    _XMP_fatal("_xmp_io_write_read_block_cyclic: data type is incorrect");
	  }
	}

      } /* ib_l */ /* ib_u */

    } /* if (rp_lb > rp_ub) */
    {
#ifdef DEBUG
      fprintf(stderr, "_xmp_io_write_read_block_cyclic: myrank=%d: space_size=%ld  total_size=%ld\n",
	     myrank, space_size, total_size);
#endif /* DEBUG */
      int b[3]; MPI_Aint d[3]; MPI_Datatype t[3];
      b[0]=1; d[0]=0;          t[0]=MPI_LB;
      b[1]=1; d[1]=space_size; t[1]=dataType_tmp;
      b[2]=1; d[2]=total_size; t[2]=MPI_UB;
      mpiRet = MPI_Type_create_struct(3, b, d, t, _dataType1);
      if (mpiRet != MPI_SUCCESS) { return 1; }
      int byte_dataType1; MPI_Aint lb_dataType1, extent_dataType1;
      MPI_Type_size(*_dataType1, &byte_dataType1);
      MPI_Type_get_extent(*_dataType1, &lb_dataType1, &extent_dataType1);
#ifdef DEBUG
      fprintf(stderr, "_xmp_io_write_read_block_cyclic: myrank=%d: space_size=%ld  total_size=%ld: "
	     "dataType1: byte_dataType1=%d bytes  lb=%ld bytes  extent=%ld bytes\n",
	     myrank,
	     space_size, total_size,
	     byte_dataType1, (long)lb_dataType1, (long)extent_dataType1);
#endif /* DEBUG */
      if (total_size != extent_dataType1){
	_XMP_fatal("_xmp_io_write_read_block_cyclic: extent is wrong");
      }
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
#ifdef DEBUG
  fprintf(stderr, "-------------------- _xmp_io_write_read_block_cyclic: NORMAL END: myrank=%d\n",myrank);
#endif /* DEBUG */
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (_xmp_io_write_read_block_cyclic): rank=%d\n", myrank);
#endif /* CHECK_POINT */
  return MPI_SUCCESS;
}

/*****************************************************************************/
/*  FUNCTION NAME : _xmp_io_pack_unpack_block_cyclic_aux1                    */
/*  DESCRIPTION   : This function is used to obtain data structure which hold*/
/*                  information on calculating index to local array for the  */
/*                  case of cyclic and block-cyclic distribution.  This      */
/*                  function is only for internal use in this file.          */
/*  ARGUMENT      : par_lower[IN] : global lower bound of array.             */
/*                  par_upper[IN] : global upper bound of array.             */
/*                  bw[IN] : block width.                                    */
/*                  cycle[IN] : cycle width.                                 */
/*                  rp_lb[IN] : global lower bound of array section.         */
/*                  rp_ub[IN] : global upper bound of array section.         */
/*                  step[IN] : stride of array section.                      */
/*                  _cnt[OUT] : number of indices in the range of array      */
/*                              section.                                     */
/*                  _bc2_result[OUT] : data structure which hold information */
/*                             on calculating index to local array for the   */
/*                             case of cyclic and block-cyclic distribution. */
/*  RETURN VALUES : MPI_SUCCESS: normal termination.                         */
/*                  an integer other than MPI_SUCCESS: abnormal termination. */
/*                                                                           */
/*****************************************************************************/
static int _xmp_io_pack_unpack_block_cyclic_aux1
(
 int par_lower /* in */, int par_upper /* in */, int bw /* in */, int cycle /* in */,
 int rp_lb /* in */, int rp_ub /* in */, int step /* in */,
 int *_cnt /* out */, int **_bc2_result /* out */
)
{
  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(_xmp_io_pack_unpack_block_cyclic_aux1): rank=%d\n", myrank);
#endif /* CHECK_POINT */
#ifdef DEBUG
  fprintf(stderr, "_xmp_io_pack_unpack_block_cyclic_aux1: rmyank = %d:  par_lower = %d  par_upper = %d  bw = %d  cycle = %d\n",
	 myrank,par_lower,par_upper,bw,cycle);
#endif /* DEBUG */

  if (bw <= 0){ _XMP_fatal("_xmp_io_pack_unpack_block_cyclic_aux1: block width must be pisitive."); }
  if (cycle == 0){ _XMP_fatal("_xmp_io_pack_unpack_block_cyclic_aux1: cycle must be non-zero."); }
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
      int z_l = MAX(par_upper,rp_ub) + 1; int ib_l = bw; int x_l = 0; /* dummy */
      int z_u = MIN(par_lower,rp_lb) - 1; int ib_u = -1; int x_u = 0; /* dummy */
      int b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0;
	{
	  int x, y, z, w; int q, r, tmp; int bb = -b;
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
	  d = y; x0 = w;
	}
	b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (a*b1), (a*k1*x0+par_lower+ib-lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib < z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; }

	int m_u_ib = func_m( (- a*b1), (- a*k1*x0 - par_lower - ib + ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int z_u_ib = a*x_u_ib + par_lower + ib;
	if (z_u_ib > z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; }
      } /* ib */

      if (ib_l == bw || ib_u == -1){ /* set is empty */
	*_cnt = 0;
	*_bc2_result = NULL;
      }else{ /* ib_l */ /* ib_u */
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u +1;
#ifdef DEBUG
	fprintf(stderr, "y_sta=%d  y_end=%d\n", y_sta, y_end);
#endif /* DEBUG */
	int i_base1 = ista;
	*_bc2_result = (int *)malloc(sizeof(int)*(7+abs(bw*b1*2)));
	int ncnt1=0, ncnt2=0;
	int *di1 = *_bc2_result + 7;
	{
	  int first=1;
	  int i;
	  for (i=ista; i<iend-(iend-ista) %(bw*b1); i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
//	      int y = (z-rp_lb) / step;
	      if (first){ i_base1 = i; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		di1[ncnt1++] = i-i_base1;
#ifdef DEBUG
		fprintf(stderr, "di1[%d]=%d\n", ncnt1-1, di1[ncnt1-1]);
#endif /* DEBUG */
	      }else{
		break;
	      }
	    }else{
	    }
	  }/* i */
	}
	int i_base2 = ista;
	{
	  int first=1;
	  int i;
	  for (i=iend-(iend-ista) %(bw*b1); i<iend; i++){
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
//	      int y = (z-rp_lb) / step;
	      if (first){ i_base2 = i; first=0; }
	      di1[ncnt1+ncnt2++] = i-i_base2;
#ifdef DEBUG
	      fprintf(stderr, "di1[%d]=%d\n", ncnt1+ncnt2-1, di1[ncnt1+ncnt2-1]);
#endif /* DEBUG */
	    }else{
	    }
	  }/* i */
	}
#ifdef DEBUG
	fprintf(stderr, "i_base1=%d  i_base2=%d\n", i_base1, i_base2);
	fprintf(stderr, "ncnt1=%d  ncnt2=%d  ((iend-ista) / (bw*b1))*ncnt1 + ncnt2=%d\n",
	       ncnt1, ncnt2, ((iend-ista) / (bw*b1))*ncnt1 + ncnt2);
#endif /* DEBUG */
	{
	  int pp = (iend-ista) / (bw*b1);
	  int *bc2_res = *_bc2_result;
	  bc2_res[0] = 7 + abs(bw*b1*2); /* alloc size */
	  bc2_res[1] = ncnt1;
	  bc2_res[2] = ncnt2;
	  bc2_res[3] = pp*ncnt1; /* pp_ncnt1 */
	  bc2_res[4] = bw*b1; /* bw_b1 */
	  bc2_res[5] = i_base1;
	  bc2_res[6] = i_base2;
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
      int z_l = MIN(par_lower, rp_ub)-1; int ib_l = -1; int x_l = 0; /* dummy */
      int z_u = MAX(par_upper, rp_lb)+1; int ib_u = bw; int x_u = 0; /* dummy */
      int b1;
      for (ib=0; ib<bw; ib++){
	int k = rp_lb - par_lower - ib;
	int d, x0;
	{
	  int x, y, z, w; int q, r, tmp; int bb = -b;
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
	  d = y; x0 = w;
	}
	b1 = b / d; int k1 = k / d;
	if (k % d != 0){ continue; }

	int m_l_ib = func_m( (-a*b1), (- a*k1*x0 - par_lower - ib + lb_tmp) );
	int x_l_ib = b1*m_l_ib + k1*x0;
	int z_l_ib = a * x_l_ib + par_lower + ib;
	if (z_l_ib > z_l){ z_l=z_l_ib; ib_l=ib; x_l=x_l_ib; }

	int m_u_ib = func_m( (a*b1), (a*k1 * x0 + par_lower + ib - ub_tmp) );
	int x_u_ib = b1*m_u_ib + k1*x0;
	int z_u_ib = a * x_u_ib + par_lower + ib;
	if (z_u_ib < z_u){ z_u=z_u_ib; ib_u=ib; x_u=x_u_ib; }
      } /* ib */

      if (ib_l == -1 || ib_u == bw){ /* set is empty */
	*_cnt = 0;
	*_bc2_result = NULL;

      }else{ /* ib_l */ /* ib_u */
	int ista=bw*x_l+ib_l;
	int iend=bw*x_u+ib_u -1;
#ifdef DEBUG
	fprintf(stderr, "y_sta=%d  y_end=%d\n", y_sta, y_end);
#endif /* DEBUG */
	int i_base1 = ista;
	int i_base2 = ista;
	*_bc2_result = (int *)malloc(sizeof(int)*(7+abs(bw*b1*2)));
	int ncnt1=0, ncnt2=0;
	int *di1 = *_bc2_result + 7;
	int i;
	{
	  int first=1;
	  for (i=ista; i>iend-(iend-ista) % (bw*b1); i--){ /* decrement */
	    int x = i / bw;
	    int ib = i - bw * x;
	    int z=a*x+par_lower+ib;
	    if ( (z-rp_lb) % step == 0 ){
//	      int y = (z-rp_lb) / step;
	      if (first){ i_base1 = i; first=0; }
	      if ((i-ista)/(bw*b1) == 0){
		di1[ncnt1++] = i-i_base1;
#ifdef DEBUG
		fprintf(stderr, "di1[%d]=%d\n", ncnt1-1, di1[ncnt1-1]);
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
//	      int y = (z-rp_lb) / step;
	      if (first){ i_base2 = i; first=0; /* FALSE */ }
	      di1[ncnt1+ncnt2++] = i-i_base2;
#ifdef DEBUG
	      fprintf(stderr, "di1[%d]=%d\n", ncnt1+ncnt2-1, di1[ncnt1+ncnt2-1]);
#endif /* DEBUG */
	    }else{
	    }
	  }/* i */
	}
#ifdef DEBUG
	fprintf(stderr, "i_base1=%d  i_base2=%d\n", i_base1, i_base2);
	fprintf(stderr, "ncnt1=%d  ncnt2=%d  ((iend-ista) / (bw*b1))*ncnt1 + ncnt2=%d\n",
	       ncnt1, ncnt2, ((iend-ista) / (bw*b1))*ncnt1 + ncnt2);
#endif /* DEBUG */
	{
	  int pp = (iend-ista) / (bw*b1);
	  int *bc2_res = *_bc2_result;
	  bc2_res[0] = 7 + abs(bw*b1*2); /* alloc size */
	  bc2_res[1] = ncnt1;
	  bc2_res[2] = ncnt2;
	  bc2_res[3] = pp*ncnt1; /* pp_ncnt1 */
	  bc2_res[4] = bw*b1; /* bw_b1 */
	  bc2_res[5] = i_base1;
	  bc2_res[6] = i_base2;
	  *_cnt = pp*ncnt1 + ncnt2;
	}
      } /* ib_l */ /* ib_u */

    } /* if (rp_lb < rp_ub) */
  } /* if (step < 0) */
  /* ++++++++++++++++++++++++++++++++++++++++ */
  else{ return 1; /* dummy */
  }
  /* ++++++++++++++++++++++++++++++++++++++++ */
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (_xmp_io_pack_unpack_block_cyclic_aux1): rank=%d\n", myrank);
#endif /* CHECK_POINT */
  return MPI_SUCCESS;
}

/*****************************************************************************/
/*  FUNCTION NAME : _xmp_io_pack_unpack_block_cyclic_aux2                    */
/*  DESCRIPTION   : This function is used to obtain index to local array     */
/*                  corresponding to index in target dimention corresponding */
/*                  to index of read/write buffer. This function is only for */
/*                  internal use in this file.                               */
/*  ARGUMENT      : j[IN] : index in target dimention corresponding to index */
/*                          of read/write buffer.                            */
/*                  bc2_result[IN] : data structure which hold information   */
/*                          on calculating index to local array for the      */
/*                          case of cyclic and block-cyclic distribution.    */
/*                  _local_index[OUT] : index to local array corresponding   */
/*                          to index j.                                      */
/*  RETURN VALUES : MPI_SUCCESS: normal termination.                         */
/*                  an integer other than MPI_SUCCESS: abnormal termination. */
/*                                                                           */
/*****************************************************************************/
static int _xmp_io_pack_unpack_block_cyclic_aux2
(
 int j /* in */, int *bc2_result /* in */,
 int *_local_index /* out */
)
{
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(_xmp_io_pack_unpack_block_cyclic_aux2)\n");
#endif /* CHECK_POINT */
  if ( bc2_result == NULL){
    return 1;
  }else{
    int ncnt1    = bc2_result[1];
    int pp_ncnt1 = bc2_result[3];
    int bw_b1    = bc2_result[4];
    int i_base1  = bc2_result[5];
    int i_base2  = bc2_result[6];
    int *di1     = bc2_result + 7;
    int p, q, i;
    if (j < pp_ncnt1){
      p = (ncnt1>0 ? j/ncnt1: 0);
      q =  j % ncnt1;
      i = p * (bw_b1) + di1[q] + i_base1;
    }else{
      p = 0;
      q = j-pp_ncnt1;
      i = di1[q] + i_base2;
    }
    *_local_index = i;
  }
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (_xmp_io_pack_unpack_block_cyclic_aux2)\n");
#endif /* CHECK_POINT */
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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_allocate_range)\n");
#endif /* CHECK_POINT */
  xmp_range_t *rp = NULL;
  if (n_dim <= 0){ return rp; }
  rp = (xmp_range_t *)malloc(sizeof(xmp_range_t));
  rp->dims = n_dim;
  rp->lb = (int*)malloc(sizeof(int)*rp->dims);
  rp->ub = (int*)malloc(sizeof(int)*rp->dims);
  rp->step = (int*)malloc(sizeof(int)*rp->dims);
  if(!rp->lb || !rp->ub || !rp->step){ return rp; }
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_allocate_range)\n");
#endif /* CHECK_POINT */
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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_set_range)\n");
#endif /* CHECK_POINT */
  if (rp == NULL){ _XMP_fatal("xmp_set_range: descriptor is NULL"); }
  if (step == 0){ _XMP_fatal("xmp_set_range: step must be non-zero"); }
  if (i_dim-1 < 0 || i_dim-1 >= rp->dims){ _XMP_fatal("xmp_set_range: i_dim is out of range"); }
  if(!rp->lb || !rp->ub || !rp->step){ _XMP_fatal("xmp_set_range: null pointer"); }
  if (step != 0 && length == 0){ _XMP_fatal("xmp_set_range: invalid combination of length and step\n"); }
  if (length < 0){ _XMP_fatal("xmp_set_range: length must be >= 0\n"); }
  rp->lb[i_dim-1] = lb;
  rp->step[i_dim-1] = step;
  if(step > 0){
    rp->ub[i_dim-1] = lb + length - 1;
  }else if (step < 0){
    rp->ub[i_dim-1] = lb - length + 1;
  }else{
    _XMP_fatal("xmp_set_range: step must be non-zero");
  }
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_set_range)\n");
#endif /* CHECK_POINT */
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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_free_range)\n");
#endif /* CHECK_POINT */
  if (rp == NULL){
    return;
  }else{
    if (rp->lb){ free(rp->lb); }
    if (rp->ub){ free(rp->ub); }
    if (rp->step){ free(rp->step); }
    free(rp);
  }
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_free_range)\n");
#endif /* CHECK_POINT */
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
/* ================================================================== */

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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fopen_all)\n");
#endif /* CHECK_POINT */
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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fopen_all)\n");
#endif /* CHECK_POINT */
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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fclose_all)\n");
#endif /* CHECK_POINT */
  // check argument
  if (pstXmp_file == NULL)     { return 1; }

  // file close
  if (MPI_File_close(&(pstXmp_file->fh)) != MPI_SUCCESS)
  {
    free(pstXmp_file);
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fclose_all)\n");
#endif /* CHECK_POINT */
    return 2;
  }
  free(pstXmp_file);
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fclose_all)\n");
#endif /* CHECK_POINT */
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
ssize_t xmp_fread_all(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int readCount;
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fread_all)\n");
#endif /* CHECK_POINT */
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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fread_all)\n");
#endif /* CHECK_POINT */
    return -1;
  }
  
  // number of bytes read
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fread_all)\n");
#endif /* CHECK_POINT */
    return -1;
  }

#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fread_all)\n");
#endif /* CHECK_POINT */
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
ssize_t xmp_fwrite_all(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
{
  MPI_Status status;
  int writeCount;
#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fwrite_all)\n");
#endif /* CHECK_POINT */
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
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fwrite_all)\n");
#endif /* CHECK_POINT */
    return -1;
  }

  // number of bytes written
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fwrite_all)\n");
#endif /* CHECK_POINT */
    return -1;
  }
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fwrite_all)\n");
#endif /* CHECK_POINT */
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
int xmp_fread_darray_unpack(fp, apd, rp)
     xmp_file_t *fp;
     xmp_desc_t apd;
     xmp_range_t *rp;
{
  MPI_Status    status;
  char         *array_addr=NULL;
  char         *buf=NULL;
  char         *cp;
  int          *lb=NULL;
  int          *ub=NULL;
  int          *step=NULL;
  int          *cnt=NULL;
  long           buf_size, j;
  int           ret=0;
  long           disp;
  long           size;
  long           array_size;
  int           i;
  xmp_desc_t tempd = NULL;
  int **bc2_result = NULL;
  size_t array_type_size;
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
  int array_ndims;
  //int ierr;

#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fread_darray_unpack)\n");
#endif /* CHECK_POINT */

  // check argument
  if (fp == NULL){ ret = -1; goto FunctionExit; }
  if (apd == NULL){ ret = -1; goto FunctionExit; }
  if (rp == NULL){ ret = -1; goto FunctionExit; }

  /*ierr = */xmp_align_template(apd, &tempd);
  if (tempd == NULL){ ret = -1; goto FunctionExit; }
  /*ierr =*/ xmp_array_ndims(apd, &array_ndims);

  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ ret = -1; goto FunctionExit; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])

  // check number of dimensions
   if (array_ndims != RP_DIMS){ ret = -1; goto FunctionExit; }

   /* allocate arrays for the number of rotations */
   lb = (int*)malloc(sizeof(int)*RP_DIMS);
   ub = (int*)malloc(sizeof(int)*RP_DIMS);
   step = (int*)malloc(sizeof(int)*RP_DIMS);
   cnt = (int*)malloc(sizeof(int)*RP_DIMS);
   if(!lb || !ub || !step || !cnt){
      ret = -1;
      goto FunctionExit;
   }
   bc2_result = (int**)malloc(sizeof(int*)*RP_DIMS);
   if(!bc2_result){
      ret = -1;
      goto FunctionExit;
   }
   for(i=0; i<RP_DIMS; i++){ bc2_result[i]=NULL; }
  
   /* calculate the number of rotations */
   buf_size = 1;
   for(i=0; i<RP_DIMS; i++){
     int par_lower_i = xmp_array_gcllbound(apd, i+1);
     int par_upper_i = xmp_array_gclubound_tmp(apd, i+1);
     int align_manner_i = xmp_align_format(apd, i+1);
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
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
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
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];

      } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC||
		align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	int bw_i = xmp_align_size(apd, i+1);
	if (bw_i <= 0){
	  _XMP_fatal("xmp_fread_darray_unpack: invalid block width");
	  ret = -1; goto FunctionExit; 
	}else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	  _XMP_fatal("xmp_fread_darray_unpack: invalid block width for cyclic distribution");
	  ret = -1; goto FunctionExit; 
	}
	int cycle_i = xmp_dist_stride(tempd, i+1);
	int zzcnt; int *zzptr;
	int ierr;
	ierr = _xmp_io_pack_unpack_block_cyclic_aux1(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
					  RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
					  &zzcnt /* out */, &zzptr /* out */);
	if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
	cnt[i] = zzcnt;
	bc2_result[i] = zzptr;

      } else {
         ret = -1;
         goto FunctionExit;
      }
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];
   }
  
   array_type_size = xmp_array_type_size(apd);

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

   // read
   if(buf_size > 0){
     if (MPI_File_read_all(fp->fh, buf, buf_size * array_type_size, MPI_BYTE, &status) != MPI_SUCCESS){
       ret = -1;
       goto FunctionExit;
     }
   }else{
     if (MPI_File_read_all(fp->fh, buf, 0, MPI_BYTE, &status) != MPI_SUCCESS){
       ret = -1;
       goto FunctionExit;
     }
   }
   // number of bytes written
   if (MPI_Get_count(&status, MPI_BYTE, &ret) != MPI_SUCCESS){
     ret = -1;
     goto FunctionExit;
   }

   /* unpack data */
   cp = buf;
   for(j=0; j<buf_size; j++){
     disp = 0;
     size = 1;
     array_size = 1;
     for(i=RP_DIMS-1; i>=0; i--){
       int par_lower_i = xmp_array_gcllbound(apd, i+1);
       int align_manner_i = xmp_align_format(apd, i+1);
       int ser_size_i = xmp_array_gsize(apd, i+1);
       int local_lower_i = xmp_array_lcllbound(apd, i+1);
       ub[i] = (j/size)%cnt[i];
       if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
	   align_manner_i == _XMP_N_ALIGN_DUPLICATION) {
	 disp += (lb[i]+ub[i]*step[i])*array_size;
	 array_size *= ser_size_i;

       } else if(align_manner_i == _XMP_N_ALIGN_BLOCK){
	 disp += (lb[i] + ub[i]*step[i] + local_lower_i - par_lower_i)*array_size;

       } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC ||
		 align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	 int local_index;
	 int ierr = _xmp_io_pack_unpack_block_cyclic_aux2(ub[i], bc2_result[i],
				       &local_index);
	 if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
	 disp += (local_index + local_lower_i) * array_size;
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
   if(bc2_result){
     for(i=0; i<RP_DIMS; i++){ if(bc2_result[i]){ free(bc2_result[i]); } }
   }

#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fread_darray_unpack)\n");
#endif /* CHECK_POINT */
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
ssize_t xmp_fread_darray_all(xmp_file_t  *pstXmp_file,
			     xmp_desc_t  apd,
			     xmp_range_t *rp)
{
  MPI_Status status;        // MPI status
  int readCount;            // read bytes
  int mpiRet;               // return value of MPI functions
  long continuous_size;      // continuous size
  long space_size;           // space size
  long total_size;           // total size
  MPI_Aint tmp1, type_size;
  MPI_Datatype dataType[2];
  int i = 0;
  xmp_desc_t tempd;
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
  int array_ndims;
  size_t array_type_size;
  int typesize_int;
  //int ierr;

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fread_darray_all): rank=%d\n", rank);
#endif /* CHECK_POINT */

  // check argument
  if (pstXmp_file == NULL) { return -1; }
  if (apd == NULL)         { return -1; }
  if (rp == NULL)          { return -1; }

  /*ierr =*/ xmp_align_template(apd, &tempd);
  if (tempd == NULL){ return -1; }
  /*ierr =*/ xmp_array_ndims(apd, &array_ndims);
  array_type_size = xmp_array_type_size(apd);

  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ return -1; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])

  // check number of dimensions
  if (array_ndims != RP_DIMS) { return -1; }

  /* case unpack is required */
  for (i = RP_DIMS - 1; i >= 0; i--){
     if(RP_STEP(i) < 0){
        int ret = xmp_fread_darray_unpack(pstXmp_file, apd, rp);
        return ret;
     }
  }

#ifdef DEBUG
fprintf(stderr, "READ(%d/%d) dims=%d\n", rank, nproc, RP_DIMS);
#endif

  // create basic data type
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &dataType[0]);

  // loop for each dimension
  for (i = RP_DIMS - 1; i >= 0; i--)
  {
    int par_lower_i = xmp_array_gcllbound(apd, i+1);
    int par_upper_i = xmp_array_gclubound_tmp(apd, i+1);
    int align_manner_i = xmp_align_format(apd, i+1);
    int local_lower_i = xmp_array_lcllbound(apd, i+1);
    int alloc_size_i;

    /*ierr =*/ xmp_array_lsize(apd, i+1, &alloc_size_i);
#ifdef DEBUG
fprintf(stderr, "READ(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
       rank, nproc, RP_LB(i),  RP_UB(i), RP_STEP(i));
fprintf(stderr, "READ(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
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
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &type_size);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  

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
fprintf(stderr, "READ(%d/%d) NOT_ALIGNED\n", rank, nproc);
fprintf(stderr, "READ(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
fprintf(stderr, "READ(%d/%d) space_size=%d\n", rank, nproc, space_size);
fprintf(stderr, "READ(%d/%d) total_size=%d\n", rank, nproc, total_size);
#endif
      }
    }
     // block distribution
    else if (align_manner_i == _XMP_N_ALIGN_BLOCK) {
      // increment is negative
      if ( RP_STEP(i) < 0) { }
      // increment is positive
      else {
	int lower, upper;
        // get extent of data type
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &type_size);
        if (mpiRet !=  MPI_SUCCESS) { return -1; }  

        // upper after distribution < lower
        if (par_upper_i < RP_LB(i)) {
          continuous_size = space_size = 0;
        }
        // lower after distribution > upper
        else if (par_lower_i > RP_UB(i)) {
          continuous_size = space_size = 0;
        }
        // other
        else {
          // lower in this node
          lower = (par_lower_i > RP_LB(i)) ?
                  RP_LB(i) + ((par_lower_i - 1 - RP_LB(i)) / RP_STEP(i) + 1) * RP_STEP(i)
	          : RP_LB(i);

          // upper in this node
          upper = (par_upper_i < RP_UB(i)) ?
                  par_upper_i
	          : RP_UB(i);

          // continuous size
          continuous_size = (upper - lower + RP_STEP(i)) / RP_STEP(i);

	  // space size
	  space_size = (local_lower_i + (lower - par_lower_i)) * type_size;
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
        if (mpiRet != MPI_SUCCESS) { return -1; }

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

#ifdef DEBUG
 	fprintf(stderr, "fread_darray_all: rank = %d:  space_size = %d  total_size = %d\n",
 	       rank,space_size,total_size);
fprintf(stderr, "READ(%d/%d) ALIGN_BLOCK\n", rank, nproc);
fprintf(stderr, "READ(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
fprintf(stderr, "READ(%d/%d) space_size=%d\n", rank, nproc, space_size);
fprintf(stderr, "READ(%d/%d) total_size=%d\n", rank, nproc, total_size);
fprintf(stderr, "READ(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc, lower, upper);
#endif
      }
    }
    // cyclic or block-cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC ||
	     align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC) {
      int bw_i = xmp_align_size(apd, i+1);
      if (bw_i <= 0) {
	_XMP_fatal("xmp_fread_darray_all: invalid block width");
	return -1;
      } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1) {
	_XMP_fatal("xmp_fread_darray_all: invalid block width for cyclic distribution");
	return -1;
      }
      int cycle_i = xmp_dist_stride(tempd, i+1);
      int ierr = _xmp_io_write_read_block_cyclic(par_lower_i   /* in */,
						 par_upper_i   /* in */,
						 bw_i          /* in */,
						 cycle_i       /* in */,
						 RP_LB(i)      /* in */,
						 RP_UB(i)      /* in */,
						 RP_STEP(i)    /* in */,
						 local_lower_i /* in */,
						 alloc_size_i  /* in */,
						 dataType[0]   /* in */,
						 &dataType[1]  /* out */);
      if (ierr != MPI_SUCCESS) { return -1; }
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];
    }
    // other
    else {
      _XMP_fatal("xmp_fread_darray_all: invalid align manner");
      return -1;
    } /* align_manner_i */
  }

  // commit
  mpiRet = MPI_Type_commit(&dataType[0]);

  // on erro in commit
  if (mpiRet != MPI_SUCCESS) { return 1; }
  
  char *array_addr;
  xmp_array_laddr(apd, (void **)&array_addr);

  // read
  MPI_Type_size(dataType[0], &typesize_int);
#ifdef DEBUG
  fprintf(stderr, "fread_darray_all: rank=%d: typesize_int = %d\n",rank,typesize_int);
#endif /* DEBUG */

  if(typesize_int > 0){
    if (MPI_File_read_all(pstXmp_file->fh,
			  array_addr,
			  1,
			  dataType[0],
			  &status)
	!= MPI_SUCCESS){ return -1; }
  }else{
    if (MPI_File_read_all(pstXmp_file->fh,
			  array_addr,
			  0, /* dummy */
			  MPI_BYTE, /* dummy */
			  &status)
	!= MPI_SUCCESS){ return -1; }
  }
  
#ifdef DEBUG
	fprintf(stderr, "CP(aft fread_darray_all) [%d/%d]\n", rank, nproc);
#endif
  // free MPI_Datatype out of use
  MPI_Type_free(&dataType[0]);

  // number of bytes read
  if (MPI_Get_count(&status, MPI_BYTE, &readCount) != MPI_SUCCESS)
  {
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fread_darray_all): rank=%d\n", rank);
#endif /* CHECK_POINT */
    return -1;
  }
#ifdef DEBUG
	fprintf(stderr, "CP(finish: xmp_fread_darray_all) [%d/%d]\n", rank, nproc);
#endif
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fread_darray_all): rank=%d\n", rank);
#endif /* CHECK_POINT */
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
int xmp_fwrite_darray_pack(fp, apd, rp)
     xmp_file_t *fp;
     xmp_desc_t  apd;
     xmp_range_t *rp;
{
   MPI_Status    status;
   char         *array_addr;
   char         *buf=NULL;
   char         *cp;
   int          *lb=NULL;
   int          *ub=NULL;
   int          *step=NULL;
   int          *cnt=NULL;
   long           buf_size, j;
   int           ret=0;
   long           disp;
   long           size;
   long           array_size;
   int           i;
   size_t array_type_size;
   xmp_desc_t tempd = NULL;
   int **bc2_result = NULL;
   int rp_dims;
   int *rp_lb_addr = NULL;
   int *rp_ub_addr = NULL;
   int *rp_step_addr = NULL;
   int array_ndims;
   //int ierr;

   int myrank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fwrite_darray_pack): rank=%d\n", myrank);
#endif /* CHECK_POINT */

   /*ierr =*/ xmp_align_template(apd, &tempd);
   if (tempd == NULL){ ret = -1; goto FunctionExit; }
   array_type_size = xmp_array_type_size(apd);
   /*ierr =*/ xmp_array_ndims(apd, &array_ndims);

   rp_dims = _xmp_range_get_dims(rp);
   rp_lb_addr = _xmp_range_get_lb_addr(rp);
   rp_ub_addr = _xmp_range_get_ub_addr(rp);
   rp_step_addr = _xmp_range_get_step_addr(rp);
   if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ ret = -1; goto FunctionExit; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])

  // check number of dimensions
   if (array_ndims != RP_DIMS){ ret = -1; goto FunctionExit; }

   /* allocate arrays for the number of rotations */
   lb = (int*)malloc(sizeof(int)*RP_DIMS);
   ub = (int*)malloc(sizeof(int)*RP_DIMS);
   step = (int*)malloc(sizeof(int)*RP_DIMS);
   cnt = (int*)malloc(sizeof(int)*RP_DIMS);
   if(!lb || !ub || !step || !cnt){
      ret = -1;
      goto FunctionExit;
   }
   bc2_result = (int**)malloc(sizeof(int*)*RP_DIMS);
   if(!bc2_result){
      ret = -1;
      goto FunctionExit;
   }
   for(i=0; i<RP_DIMS; i++){ bc2_result[i]=NULL; }
  
   /* calculate the number of rotaions */
   buf_size = 1;
   for(i=0; i<RP_DIMS; i++){
     int par_lower_i = xmp_array_gcllbound(apd, i+1);
     int par_upper_i = xmp_array_gclubound_tmp(apd, i+1);
     int align_manner_i = xmp_align_format(apd, i+1);

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
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];
  
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
	 cnt[i] = (ub[i]-lb[i]+step[i])/step[i];

      } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC ||
		align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	int bw_i = xmp_align_size(apd, i+1);
	if (bw_i <= 0){
	  _XMP_fatal("xmp_fwrite_darray_pack: invalid block width");
	  ret = -1; goto FunctionExit;
	}else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	  _XMP_fatal("xmp_fwrite_darray_pack: invalid block width for cyclic distribution");
	  ret = -1; goto FunctionExit;
	}
	int cycle_i = xmp_dist_stride(tempd, i+1);
	int zzcnt; int *zzptr;
	int ierr;
	ierr = _xmp_io_pack_unpack_block_cyclic_aux1(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
							 RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
							 &zzcnt /* out */, &zzptr /* out */);
	if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
	cnt[i] = zzcnt;
	bc2_result[i] = zzptr;

      } else {
         ret = -1;
         goto FunctionExit;
      }
      cnt[i] = (cnt[i]>0)? cnt[i]: 0;
      buf_size *= cnt[i];
   }
  
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

   /* pack data */
   cp = buf;
   xmp_array_laddr(apd, (void **)&array_addr);
   for(j=0; j<buf_size; j++){
     disp = 0;
     size = 1;
     array_size = 1;
     for(i=RP_DIMS-1; i>=0; i--){
       int par_lower_i = xmp_array_gcllbound(apd, i+1);
       int align_manner_i = xmp_align_format(apd, i+1);
       int local_lower_i = xmp_array_lcllbound(apd, i+1);
       int ser_size_i = xmp_array_gsize(apd, i+1);
       int alloc_size_i;
       //int ierr;
       /*ierr =*/ xmp_array_lsize(apd, i+1, &alloc_size_i);
       ub[i] = (j/size)%cnt[i];
       if (align_manner_i == _XMP_N_ALIGN_NOT_ALIGNED ||
	   align_manner_i == _XMP_N_ALIGN_DUPLICATION) {
	 disp += (lb[i]+ub[i]*step[i])*array_size;
	 array_size *= ser_size_i;

       } else if(align_manner_i == _XMP_N_ALIGN_BLOCK){
	 disp += (lb[i]+ub[i]*step[i] + local_lower_i - par_lower_i)*array_size;
	 array_size *= alloc_size_i;

       } else if(align_manner_i == _XMP_N_ALIGN_CYCLIC ||
		 align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC){
	 int local_index;
	 int ierr = _xmp_io_pack_unpack_block_cyclic_aux2(ub[i] /* in */, bc2_result[i] /* in */,
							  &local_index /* out */);
	 if (ierr != MPI_SUCCESS){ ret = -1; goto FunctionExit; }
	 disp += (local_index + local_lower_i) * array_size;
	 array_size *= alloc_size_i;
       } /* align_manner_i */
       size *= cnt[i];
     } /* i */
     disp *= array_type_size;
     memcpy(cp, array_addr+disp, array_type_size);
     cp += array_type_size;
   } /* j */

  // write
   if(buf_size > 0){
     if (MPI_File_write_all(fp->fh, buf, buf_size * array_type_size, MPI_BYTE, &status) != MPI_SUCCESS){
       ret = -1;
       goto FunctionExit;
     }
   }else{
     if (MPI_File_write_all(fp->fh, buf, 0, MPI_BYTE, &status) != MPI_SUCCESS){
       ret = -1;
       goto FunctionExit;
     }
   }
   // number of bytes written
   if (MPI_Get_count(&status, MPI_BYTE, &ret) != MPI_SUCCESS) {
     ret = -1;
     goto FunctionExit;
   }
  
 FunctionExit:
   if(buf) free(buf);
   if(lb) free(lb);
   if(ub) free(ub);
   if(step) free(step);
   if(cnt) free(cnt);
   if(bc2_result){
     for(i=0; i<RP_DIMS; i++){ if(bc2_result[i]){ free(bc2_result[i]); } }
   }
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fwrite_darray_pack): rank=%d\n", myrank);
#endif /* CHECK_POINT */
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
ssize_t xmp_fwrite_darray_all(xmp_file_t *pstXmp_file,
			      xmp_desc_t apd,
			      xmp_range_t *rp)
{
  MPI_Status status;        // MPI status
  int writeCount;           // write btye
  int mpiRet;               // return value of MPI functions
  long continuous_size;      // continuous size
  long space_size;           // space size
  long total_size;           // total size
  MPI_Aint tmp1, type_size;
  MPI_Datatype dataType[2];
  int i = 0;
  xmp_desc_t tempd;
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
  int array_ndims;
  size_t array_type_size;
  int typesize_int;
  //int ierr;

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_fwrite_darray_all): rank=%d\n", rank);
#endif /* CHECK_POINT */

  // check argument
  if (pstXmp_file == NULL) { return -1101; }
  if (apd == NULL)         { return -1103; }
  if (rp == NULL)          { return -1104; }

  /*ierr =*/ xmp_align_template(apd, &tempd);
  if (tempd == NULL){ return -1105; }
  /*ierr =*/ xmp_array_ndims(apd, &array_ndims);
  array_type_size = xmp_array_type_size(apd);

  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ return -1106; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])

  // check number of dimensions
  if (array_ndims != RP_DIMS) { return -1107; }

#ifdef DEBUG
fprintf(stderr, "WRITE(%d/%d) dims=%d\n",rank, nproc, RP_DIMS);
#endif

  /* case pack is required */
  for (i = RP_DIMS - 1; i >= 0; i--){
     if(RP_STEP(i) < 0){
        int ret = xmp_fwrite_darray_pack(pstXmp_file, apd, rp);
        return ret;
     }
  }

  // create basic data type
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &dataType[0]);

  // loop for each dimension
  for (i = RP_DIMS - 1; i >= 0; i--)
  {
    int par_lower_i = xmp_array_gcllbound(apd, i+1);
    int par_upper_i = xmp_array_gclubound_tmp(apd, i+1);
    int align_manner_i = xmp_align_format(apd, i+1);
    int ser_lower_i = xmp_array_gcglbound(apd, i+1);
    int ser_upper_i = xmp_array_gcgubound(apd, i+1);
    int local_lower_i = xmp_array_lcllbound(apd, i+1);
    int alloc_size_i;
    //int ierr;
    /*ierr =*/ xmp_array_lsize(apd, i+1, &alloc_size_i);
/*     int local_upper_i = xmp_array_lclubound(apd, i+1); */
/*     int shadow_size_lo_i = xmp_array_lshadow(apd, i+1); */
/*     int shadow_size_hi_i = xmp_array_ushadow(apd, i+1); */
#ifdef DEBUG
fprintf(stderr, "WRITE(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
       rank, nproc, RP_LB(i),  RP_UB(i), RP_STEP(i));
fprintf(stderr, "WRITE(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
       rank, nproc, par_lower_i, par_upper_i);
/* fprintf(stderr, "WRITE(%d/%d) (local_lower,local_upper,alloc_size)=(%d,%d,%d)\n", */
/*        rank, nproc, local_lower_i, local_upper_i, alloc_size_i); */
/* fprintf(stderr, "WRITE(%d/%d) (shadow_size_lo,shadow_size_hi)=(%d,%d)\n", */
/*        rank, nproc, shadow_size_lo_i, shadow_size_hi_i); */
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
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &type_size);
        if (mpiRet !=  MPI_SUCCESS) { return -1110; }  

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
fprintf(stderr, "WRITE(%d/%d) NOT_ALIGNED\n",rank, nproc);
fprintf(stderr, "WRITE(%d/%d) type_size=%ld\n",rank, nproc, (long)type_size);
fprintf(stderr, "WRITE(%d/%d) continuous_size=%ld\n",rank, nproc, continuous_size);
fprintf(stderr, "WRITE(%d/%d) space_size=%ld\n",rank, nproc, space_size);
fprintf(stderr, "WRITE(%d/%d) total_size=%ld\n",rank, nproc, total_size);
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
        int lower, upper;
        // get extent of data type
        mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &type_size);
        if (mpiRet !=  MPI_SUCCESS) { return -1113; }  

        // upper after distribution < lower
        if (par_upper_i < RP_LB(i))
        {
          continuous_size = space_size = 0;
        }
        // lower after distribution > upper
        else if (par_lower_i > RP_UB(i)) {
          continuous_size = space_size = 0;
        }
        // other
        else {
          // lower in this node
          lower = (par_lower_i > RP_LB(i)) ?
                  RP_LB(i) + ((par_lower_i - 1 - RP_LB(i)) / RP_STEP(i) + 1) * RP_STEP(i)
	          : RP_LB(i);

          // upper in this node
          upper = (par_upper_i < RP_UB(i)) ?
                  par_upper_i
	          : RP_UB(i);

          // continuous size
          continuous_size = (upper - lower + RP_STEP(i)) / RP_STEP(i);

	  if(lower > upper){ type_size = 0; }
	  // space size
	  space_size = (local_lower_i + (lower - par_lower_i)) * type_size;
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
fprintf(stderr, "WRITE(%d/%d) ALIGN_BLOCK\n",rank, nproc);
fprintf(stderr, "WRITE(%d/%d) type_size=%ld\n",rank, nproc, (long)type_size);
fprintf(stderr, "WRITE(%d/%d) continuous_size=%ld\n",rank, nproc, continuous_size);
fprintf(stderr, "WRITE(%d/%d) space_size=%ld\n",rank, nproc, space_size);
fprintf(stderr, "WRITE(%d/%d) total_size=%ld\n",rank, nproc, total_size);
fprintf(stderr, "WRITE(%d/%d) (lower,upper)=(%d,%d)\n",rank, nproc, lower, upper);
#endif
      }
    }
    // cyclic or block-cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC ||
	     align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC)
    {
      int bw_i = xmp_align_size(apd, i+1);
      if (bw_i <= 0){
	_XMP_fatal("xmp_fwrite_darray_all: invalid block width");
	return -1122;
      }else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	_XMP_fatal("xmp_fwrite_darray_all: invalid block width for cyclic distribution");
	return -1122;
      }
      int cycle_i = xmp_dist_stride(tempd, i+1);
      int ierr = _xmp_io_write_read_block_cyclic(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
						 RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
						 local_lower_i /* in */,
						 alloc_size_i /* in */,
						 dataType[0] /* in */,
						 &dataType[1] /* out */);
      if (ierr != MPI_SUCCESS) { return -1117; }
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];
    }
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
 
  char *array_addr;
  xmp_array_laddr(apd, (void **)&array_addr);

  // write
  MPI_Type_size(dataType[0], &typesize_int);
#ifdef DEBUG
  fprintf(stderr, "fwrite_darray_all: rank=%d: typesize_int = %d\n",rank,typesize_int);
#endif /* DEBUG */
  {
    if(typesize_int > 0){
      if ((MPI_File_write_all(pstXmp_file->fh,
				     array_addr,
				     1,
				     dataType[0],
				     &status))
	  != MPI_SUCCESS){ return -1120; }
    }else{
      if ((MPI_File_write_all(pstXmp_file->fh,
				     array_addr,
				     0, /* dummy */
				     MPI_BYTE, /* dummy */
				     &status))
	  != MPI_SUCCESS){ return -1120; }
    }
  }
  // free MPI_Datatype out of use
  MPI_Type_free(&dataType[0]);

  // number of btyes written
  if (MPI_Get_count(&status, MPI_BYTE, &writeCount) != MPI_SUCCESS)
  {
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fwrite_darray_all): rank=%d\n", rank);
#endif /* CHECK_POINT */
    return -1121;
  }
#ifdef DEBUG
  if(rank==0){fprintf(stderr, "-------------------- fwrite_darray_all: NORMAL END\n");}
#endif /* DEBUG */
#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_fwrite_darray_all): rank=%d\n", rank);
#endif /* CHECK_POINT */
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
ssize_t xmp_fread_shared(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
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
ssize_t xmp_fwrite_shared(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
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
ssize_t xmp_fread(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
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
ssize_t xmp_fwrite(xmp_file_t *pstXmp_file, void *buffer, size_t size, size_t count)
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
int xmp_file_set_view_all(xmp_file_t  *pstXmp_file,
			  long long    disp,
			  xmp_desc_t   apd,
			  xmp_range_t *rp)
{
  int i = 0;
  int mpiRet;               // return value of MPI functions
  int lower;                // lower bound accessed by this node
  int upper;                // upper bound accessed by this node
  long continuous_size;      // continuous size
  MPI_Datatype dataType[2];
  MPI_Aint tmp1, type_size;
  xmp_desc_t tempd;
  int rp_dims;
  int *rp_lb_addr = NULL;
  int *rp_ub_addr = NULL;
  int *rp_step_addr = NULL;
  int array_ndims;
  size_t array_type_size;
  //int ierr;

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

#ifdef CHECK_POINT
  fprintf(stderr, "IO:START(xmp_file_set_view_all): rank=%d\n", rank);
#endif /* CHECK_POINT */

  // check argument
  if (pstXmp_file == NULL) { return 1001; }
  if (apd == NULL)         { return 1002; }
  if (rp == NULL)          { return 1004; }
  if (disp  < 0)           { return 1005; }

  /*ierr =*/ xmp_align_template(apd, &tempd);
  if (tempd == NULL){ return 1006; }
  /*ierr =*/ xmp_array_ndims(apd, &array_ndims);
  array_type_size = xmp_array_type_size(apd);

  rp_dims = _xmp_range_get_dims(rp);
  rp_lb_addr = _xmp_range_get_lb_addr(rp);
  rp_ub_addr = _xmp_range_get_ub_addr(rp);
  rp_step_addr = _xmp_range_get_step_addr(rp);
  if (!rp_lb_addr || !rp_ub_addr || !rp_step_addr){ return 1007; }
#define RP_DIMS     (rp_dims)
#define RP_LB(i)    (rp_lb_addr[(i)])
#define RP_UB(i)    (rp_ub_addr[(i)])
#define RP_STEP(i)  (rp_step_addr[(i)])

  // check number of dimensions
  if (array_ndims != RP_DIMS) { return 1008; }

#ifdef DEBUG
fprintf(stderr, "VIEW(%d/%d) dims=%d\n", rank, nproc, RP_DIMS);
#endif

  // create basic data type
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &dataType[0]);

  // loop for each dimension
  for (i = RP_DIMS - 1; i >= 0; i--)
  {
    int par_lower_i = xmp_array_gcllbound(apd, i+1);
    int par_upper_i = xmp_array_gclubound_tmp(apd, i+1);
    int align_manner_i = xmp_align_format(apd, i+1);
#ifdef DEBUG
    fprintf(stderr, "xmp_file_set_view_all: myrank=%d: i=%d: "
	   "align_manner_i=%d  bw_i=%d  par_lower_i=%d  par_upper_i=%d\n",
	   rank, i,
	   xmp_align_format(apd, i+1),
	   xmp_align_size(apd, i+1),
	   xmp_array_gcllbound(apd, i+1),
	   xmp_array_gclubound_tmp(apd, i+1));
#endif /* DEBUG */

    // get extent of data type
    mpiRet =MPI_Type_get_extent(dataType[0], &tmp1, &type_size);
    if (mpiRet !=  MPI_SUCCESS) { return -1009; }

    int byte_dataType0; MPI_Type_size(dataType[0], &byte_dataType0);
#ifdef DEBUG
    fprintf(stderr, "xmp_file_set_view_all: rank=%d: i=%d  align_manner_i=%d  type_size=%ld  byte_dataType0=%d\n",
	   rank, i, align_manner_i, (long)type_size, byte_dataType0);
#endif /* DEBUG */

#ifdef DEBUG
fprintf(stderr, "VIEW(%d/%d) (lb,ub,step)=(%d,%d,%d)\n",
        rank, nproc, RP_LB(i),  RP_UB(i), RP_STEP(i));
fprintf(stderr, "VIEW(%d/%d) (par_lower,par_upper)=(%d,%d)\n",
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
fprintf(stderr, "VIEW(%d/%d) NOT_ALIGNED\n", rank, nproc);
fprintf(stderr, "VIEW(%d/%d) continuous_size=%d\n", rank, nproc, continuous_size);
#endif
    }
    // block distribution
    else if (align_manner_i == _XMP_N_ALIGN_BLOCK)
    {
      long space_size;
      long total_size;

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

/* 	  fprintf(stderr, "set_view_all: rank = %d: lower = %d  upper = %d  continuous_size = %d  space_size = %d\n", */
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

	{
	  int byte_datatype1; MPI_Aint lb_datatype1, extent_datatype1;
	  MPI_Type_size(dataType[1], &byte_datatype1);
	  MPI_Type_get_extent(dataType[1], &lb_datatype1, &extent_datatype1);
	  if (extent_datatype1 + space_size > total_size){
	    _XMP_fatal("xmp_file_set_view_all (block): data type is incorrect");
	  }
	}

        // create new file type
        mpiRet = MPI_TYPE_CREATE_RESIZED1(dataType[1],
                                         space_size,
                                         total_size,
                                         &dataType[0]);

        // on error in MPI_Type_create_resized1
        if (mpiRet != MPI_SUCCESS) { return 1013; }

        // free MPI_Datatype out of use
        MPI_Type_free(&dataType[1]);

	int byte_dataType0; MPI_Aint lb_dataType0, extent_dataType0;
	MPI_Type_size(dataType[0], &byte_dataType0);
	MPI_Type_get_extent(dataType[0], &lb_dataType0, &extent_dataType0);
#ifdef DEBUG
	fprintf(stderr, "set_view_all: after block: myrank=%d: byte_dataType0=%d  lb=%ld  extent=%ld ; space_size=%ld  total_size=%ld\n",
	       rank, byte_dataType0, (long)lb_dataType0, (long)extent_dataType0,
	       space_size, total_size);
#endif /* DEBUG */
#ifdef DEBUG
fprintf(stderr, "VIEW(%d/%d) ALIGN_BLOCK\n", rank, nproc );
fprintf(stderr, "VIEW(%d/%d) type_size=%ld\n", rank, nproc , (long)type_size);
fprintf(stderr, "VIEW(%d/%d) continuous_size=%ld\n", rank, nproc , continuous_size);
fprintf(stderr, "VIEW(%d/%d) space_size=%ld\n", rank, nproc , space_size);
fprintf(stderr, "VIEW(%d/%d) total_size=%ld\n", rank, nproc , total_size);
fprintf(stderr, "VIEW(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc , lower, upper);
fprintf(stderr, "\n");
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
/*           space_size */
/*             = ((lower - RP_LB(i)) / RP_STEP(i)) * type_size; */
          space_size
            = ( - (upper - RP_UB(i)) / RP_STEP(i)) * type_size;
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

	{
	  int byte_datatype1; MPI_Aint lb_datatype1, extent_datatype1;
	  MPI_Type_size(dataType[1], &byte_datatype1);
	  MPI_Type_get_extent(dataType[1], &lb_datatype1, &extent_datatype1);
	  if (extent_datatype1 + space_size > total_size){
	    _XMP_fatal("xmp_file_set_view_all (block): data type is incorrect");
	  }
	}

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
fprintf(stderr, "VIEW(%d/%d) ALIGN_BLOCK\n", rank, nproc);
fprintf(stderr, "VIEW(%d/%d) continuous_size=%ld\n", rank, nproc, continuous_size);
fprintf(stderr, "VIEW(%d/%d) space_size=%ld\n", rank, nproc, space_size);
fprintf(stderr, "VIEW(%d/%d) total_size=%ld\n", rank, nproc, total_size);
fprintf(stderr, "VIEW(%d/%d) (lower,upper)=(%d,%d)\n", rank, nproc, lower, upper);
#endif
      }
    }
    // cyclic or block-cyclic distribution
    else if (align_manner_i == _XMP_N_ALIGN_CYCLIC ||
	     align_manner_i == _XMP_N_ALIGN_BLOCK_CYCLIC)
    {
      int bw_i = xmp_align_size(apd, i+1);
      if (bw_i <= 0){
	_XMP_fatal("xmp_file_set_view_all: invalid block width");
	return 1021;
      }else if(align_manner_i == _XMP_N_ALIGN_CYCLIC && bw_i != 1){
	_XMP_fatal("xmp_file_set_view_all: invalid block width for cyclic distribution");
	return 1021;
      }
      int cycle_i = xmp_dist_stride(tempd, i+1);
      int ierr = _xmp_io_set_view_block_cyclic(par_lower_i /* in */, par_upper_i /* in */, bw_i /* in */, cycle_i /* in */,
					       RP_LB(i) /* in */, RP_UB(i) /* in */, RP_STEP(i) /* in */,
					       dataType[0] /* in */,
					       &dataType[1] /* out */);
      if (ierr != MPI_SUCCESS) { return -1017; }
      MPI_Type_free(&dataType[0]);
      dataType[0] = dataType[1];
    }
    // other
    else
    {
      _XMP_fatal("xmp_file_set_view_all: invalid align manner");
      return 1018;
    } /* align_manner_i */
  }

  // commit
  mpiRet = MPI_Type_commit(&dataType[0]);

  // on erro in commit
  if (mpiRet != MPI_SUCCESS) { return 1019; }
  
  // set view
  {
    int byte_dataType0;
    MPI_Type_size(dataType[0], &byte_dataType0);
#ifdef DEBUG
    fprintf(stderr, "set_view_all: myrank=%d: byte_dataType0=%d\n", rank, byte_dataType0);
#endif /* DEBUG */
    if (byte_dataType0 > 0){
      mpiRet = MPI_File_set_view(pstXmp_file->fh,
				 (MPI_Offset)disp,
				 MPI_BYTE,
				 dataType[0],
				 "native",
				 MPI_INFO_NULL);
    }else{
      mpiRet = MPI_File_set_view(pstXmp_file->fh,
				 (MPI_Offset)disp,
				 MPI_BYTE,
				 MPI_BYTE, /* dummy */
				 "native",
				 MPI_INFO_NULL);
    }
  }
  // free MPI_Datatype out of use
  MPI_Type_free(&dataType[0]);

  // on erro in set view
  if (mpiRet != MPI_SUCCESS) { return 1020; }

#ifdef CHECK_POINT
  fprintf(stderr, "IO:END  (xmp_file_set_view_all): rank=%d\n", rank);
#endif /* CHECK_POINT */

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
static int MPI_Type_create_resized1(MPI_Datatype oldtype,
				    MPI_Aint     lb,
				    MPI_Aint     extent,
				    MPI_Datatype *newtype)
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
