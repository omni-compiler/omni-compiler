#include "stdbool.h"
#include "stdlib.h"
#include "xmp.h"

void ixmp_pdlahqr(bool *wantt,bool *wantz,int *n,int *ilo,int *ihi,double *a,xmp_desc_t da,double *wr,double *wi,int *iloz,int *ihiz,double *z,xmp_desc_t dz,double *work,int *lwork,double *iwork,int *ilwork,int *contxt,int *info){


  int desca[9],descz[9];
  int andim, zndim, *asize, *zsize, *aidx, *zidx, alead_dim, zlead_dim;
  xmp_desc_t dat,dzt;

  xmp_array_ndim(da, &andim);
  asize = (int *)malloc(sizeof(int)* andim);
  aidx = (int *)malloc(sizeof(int)* andim);
  xmp_array_ndim(dz, &zndim);
  zsize = (int *)malloc(sizeof(int)* zndim);
  zidx = (int *)malloc(sizeof(int)* zndim);

  desca[0]=1;
  desca[1]=*contxt;

  xmp_array_gsize(da, asize);
  desca[2]=asize[1];
  desca[3]=asize[0];

  dat = xmp_align_template(da);
  xmp_dist_size(dat, asize);
  desca[4]=asize[1];
  desca[5]=asize[0];

  xmp_array_first_idx_node_index(da, aidx);
  desca[6]=aidx[1];
  desca[7]=aidx[0];

  xmp_array_lead_dim(da, &alead_dim);
  desca[8]=alead_dim;


  descz[0]=1;
  descz[1]=*contxt;

  xmp_array_gsize(dz, zsize);
  descz[2]=zsize[1];
  descz[3]=zsize[0];

  dzt = xmp_align_template(dz);
  xmp_dist_size(dzt, zsize);
  descz[4]=zsize[1];
  descz[5]=zsize[0];

  xmp_array_first_idx_node_index(dz, zidx);
  descz[6]=zidx[1];
  descz[7]=zidx[0];

  xmp_array_lead_dim(da, &zlead_dim);
  descz[8]=zlead_dim;

  pdlahqr_(wantt,wantz,n,ilo,ihi,a,desca,wr,wi,iloz,ihiz,z,descz,work,lwork,iwork,ilwork,info);

  free(asize);
  free(aidx);
  free(zsize);
  free(zidx);

}
