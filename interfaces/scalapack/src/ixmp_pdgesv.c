#include "stdlib.h"
#include "xmp.h"

void ixmp_pdgesv(int *n,int *nrhs,double *a,int *ia,int *ja,xmp_desc_t da,int *ipiv,double *b,int *ib,int *jb,xmp_desc_t db,int *contxt,int *info){


  int nbrhs,desca[9],descb[9];
  int andim, bndim, *asize, *bsize, *aidx, *bidx, alead_dim, blead_dim;
  xmp_desc_t dat,dbt;

  xmp_array_ndim(da, &andim);
  asize = (int *)malloc(sizeof(int)* andim);
  aidx = (int *)malloc(sizeof(int)* andim);
  xmp_array_ndim(db, &bndim);
  bsize = (int *)malloc(sizeof(int)* bndim);
  bidx = (int *)malloc(sizeof(int)* bndim);

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


  descb[0]=1;
  descb[1]=*contxt;

  xmp_array_gsize(db, bsize);
  if (bndim == 1) {
     descb[2]=bsize[0];
     descb[3]=1;
  }else if (bndim == 2) {
     descb[2]=bsize[1];
     descb[3]=bsize[0];
  }

  dbt = xmp_align_template(db);
  xmp_dist_size(dbt, bsize);
  if (bndim == 1) {
     descb[4]=bsize[0];
     descb[5]=1;
  }else if (bndim == 2) {
     descb[4]=bsize[1];
     descb[5]=bsize[0];
  }

  xmp_array_first_idx_node_index(db, bidx);
  if (bndim == 1) {
     descb[6]=bidx[0];
     descb[7]=1;
  }else if (bndim == 2) {
     descb[6]=bidx[1];
     descb[7]=bidx[0];
  }

  xmp_array_lead_dim(db, &blead_dim);
  descb[8]=blead_dim;

  pdgesv_(n,nrhs,a,ia,ja,desca,ipiv,b,ib,jb,descb,info);

  free(asize);
  free(aidx);
  free(bsize);
  free(bidx);

}
