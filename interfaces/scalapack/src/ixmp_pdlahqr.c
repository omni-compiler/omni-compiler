#include "stdlib.h"
#include "xmp.h"

void ixmp_pdlahqr(_Bool *wantt,_Bool *wantz,int *n,int *ilo,int *ihi,double *a,xmp_desc_t da,double *wr,double *wi,int *iloz,int *ihiz,double *z,xmp_desc_t dz,double *work,int *lwork,double *iwork,int *ilwork,int *contxt,int *info){


  int desca[9],descz[9];
  int andims, zndims, *awk, *zwk, ierr;
  xmp_desc_t dta, dtz;
  int lbound_a1, ubound_a1, lbound_a2, ubound_a2;
  int axis_a1, axis_a2, blocksize_a1, blocksize_a2;
  int lbound_z1, ubound_z1, lbound_z2, ubound_z2;
  int axis_z1, axis_z2, blocksize_z1, blocksize_z2;

  ierr=xmp_array_ndims(da, &andims);
  awk = (int *)malloc(sizeof(int)* andims);
  ierr=xmp_array_ndims(dz, &zndims);
  zwk = (int *)malloc(sizeof(int)* zndims);

  ierr = xmp_array_lbound(da, 1, &lbound_a1);
  ierr = xmp_array_ubound(da, 1, &ubound_a1);
  ierr = xmp_array_lbound(da, 2, &lbound_a2);
  ierr = xmp_array_ubound(da, 2, &ubound_a2);
  ierr = xmp_align_template(da, &dta);
  ierr = xmp_align_axis(da, 1, &axis_a1);
  ierr = xmp_align_axis(da, 2, &axis_a2);
  ierr = xmp_dist_blocksize(dta, axis_a1, &blocksize_a1);
  ierr = xmp_dist_blocksize(dta, axis_a2, &blocksize_a2);

  desca[0]=1;
  desca[1]=*contxt;

  desca[2]=ubound_a2-lbound_a2+1;
  desca[3]=ubound_a1-lbound_a1+1;

  desca[4]=blocksize_a2;
  desca[5]=blocksize_a1;

  awk[0]=0;
  awk[1]=0;
  desca[6]=xmp_array_owner(da, andims, awk, 2)-1;
  desca[7]=xmp_array_owner(da, andims, awk, 1)-1;

  ierr = xmp_array_lead_dim(da, awk);
  desca[8]=awk[andims-1];

  ierr = xmp_array_lbound(dz, 1, &lbound_z1);
  ierr = xmp_array_ubound(dz, 1, &ubound_z1);
  ierr = xmp_array_lbound(dz, 2, &lbound_z2);
  ierr = xmp_array_ubound(dz, 2, &ubound_z2);
  ierr = xmp_align_template(dz, &dtz);
  ierr = xmp_align_axis(dz, 1, &axis_z1);
  ierr = xmp_align_axis(dz, 2, &axis_z2);
  ierr = xmp_dist_blocksize(dtz, axis_z1, &blocksize_z1);
  ierr = xmp_dist_blocksize(dtz, axis_z2, &blocksize_z2);

  descz[0]=1;
  descz[1]=*contxt;

  descz[2]=ubound_z2-lbound_z2+1;
  descz[3]=ubound_z1-lbound_z1+1;

  descz[4]=blocksize_z2;
  descz[5]=blocksize_z1;

  zwk[0]=0;
  zwk[1]=0;
  descz[6]=xmp_array_owner(dz, zndims, zwk, 2)-1;
  descz[7]=xmp_array_owner(dz, zndims, zwk, 1)-1;

  ierr = xmp_array_lead_dim(dz, zwk);
  descz[8]=zwk[zndims-1];

  pdlahqr_(wantt,wantz,n,ilo,ihi,a,desca,wr,wi,iloz,ihiz,z,descz,work,lwork,iwork,ilwork,info);

  free(awk);
  free(zwk);

}
