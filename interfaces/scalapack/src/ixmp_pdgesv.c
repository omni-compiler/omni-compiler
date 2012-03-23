#include "stdlib.h"
#include "xmp.h"

void ixmp_pdgesv(int *n,int *nrhs,double *a,int *ia,int *ja,xmp_desc_t da,int *ipiv,double *b,int *ib,int *jb,xmp_desc_t db,int *contxt,int *info){


  int desca[9],descb[9];
  int andim, bndim, *aidx, *bidx;

  andim=xmp_array_ndim(da);
  aidx = (int *)malloc(sizeof(int)* andim);
  bndim=xmp_array_ndim(db);
  bidx = (int *)malloc(sizeof(int)* bndim);

  desca[0]=1;
  desca[1]=*contxt;

  desca[2]=xmp_array_gsize(da,2);
  desca[3]=xmp_array_gsize(da,1);

  desca[4]=xmp_align_size(da,2);
  desca[5]=xmp_align_size(da,1);

  xmp_array_first_idx_node_index(da, aidx);
  desca[6]=aidx[1];
  desca[7]=aidx[0];

  desca[8]=xmp_array_lead_dim(da);


  descb[0]=1;
  descb[1]=*contxt;

  if (bndim == 1) {
     descb[2]=xmp_array_gsize(db,1);
     descb[3]=1;
  }else if (bndim == 2) {
     descb[2]=xmp_array_gsize(db,2);
     descb[3]=xmp_array_gsize(db,1);
  }

  if (bndim == 1) {
     descb[4]=xmp_align_size(db,1);
     descb[5]=1;
  }else if (bndim == 2) {
     descb[4]=xmp_align_size(db,2);
     descb[5]=xmp_align_size(db,1);
  }

  xmp_array_first_idx_node_index(db, bidx);
  if (bndim == 1) {
     descb[6]=bidx[0];
     descb[7]=1;
  }else if (bndim == 2) {
     descb[6]=bidx[1];
     descb[7]=bidx[0];
  }

  descb[8]=xmp_array_lead_dim(db);

  pdgesv_(n,nrhs,a,ia,ja,desca,ipiv,b,ib,jb,descb,info);

  free(aidx);
  free(bidx);

}
