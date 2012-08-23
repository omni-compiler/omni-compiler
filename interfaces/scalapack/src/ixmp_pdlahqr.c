#include "stdlib.h"
#include "xmp.h"

void ixmp_pdlahqr(_Bool *wantt,_Bool *wantz,int *n,int *ilo,int *ihi,double *a,xmp_desc_t da,double *wr,double *wi,int *iloz,int *ihiz,double *z,xmp_desc_t dz,double *work,int *lwork,double *iwork,int *ilwork,int *contxt,int *info){


  int desca[9],descz[9];
  int andim, zndim, *aidx, *zidx;

  andim=xmp_array_ndim(da);
  aidx = (int *)malloc(sizeof(int)* andim);
  zndim=xmp_array_ndim(dz);
  zidx = (int *)malloc(sizeof(int)* zndim);

  desca[0]=1;
  desca[1]=*contxt;

  desca[2]=xmp_array_gsize(da,2);
  desca[3]=xmp_array_gsize(da,1);

  desca[4]=xmp_align_size(da,2);
  desca[5]=xmp_align_size(da,1);

  aidx[0]=0;
  aidx[0]=0;
  desca[6]=xmp_array_owner(da, andim, aidx, 2);
  desca[7]=xmp_array_owner(da, andim, aidx, 1);

  desca[8]=xmp_array_lead_dim(da);


  descz[0]=1;
  descz[1]=*contxt;

  descz[2]=xmp_array_gsize(dz,2);
  descz[3]=xmp_array_gsize(dz,1);

  descz[4]=xmp_align_size(dz,2);
  descz[5]=xmp_align_size(dz,1);

  zidx[0]=0;
  zidx[0]=0;
  descz[6]=xmp_array_owner(dz, zndim, zidx, 2);
  descz[7]=xmp_array_owner(dz, zndim, zidx, 1);

  descz[8]=xmp_array_lead_dim(dz);

  pdlahqr_(wantt,wantz,n,ilo,ihi,a,desca,wr,wi,iloz,ihiz,z,descz,work,lwork,iwork,ilwork,info);

  free(aidx);
  free(zidx);

}
