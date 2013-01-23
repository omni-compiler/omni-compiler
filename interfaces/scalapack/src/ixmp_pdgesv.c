#include "stdlib.h"
#include "xmp.h"

void ixmp_pdgesv(int *n,int *nrhs,double *a,int *ia,int *ja,xmp_desc_t da,int *ipiv,double *b,int *ib,int *jb,xmp_desc_t db,int *contxt,int *info){


  int desca[9],descb[9];
  int andims, bndims, *awk, *bwk, ierr;
  xmp_desc_t dta, dtb;
  int lbound_a1, ubound_a1, lbound_a2, ubound_a2;
  int axis_a1, axis_a2, blocksize_a1, blocksize_a2;
  int lbound_b1, ubound_b1, lbound_b2, ubound_b2;
  int axis_b1, axis_b2, blocksize_b1, blocksize_b2;

  ierr=xmp_array_ndims(da, &andims);
  awk = (int *)malloc(sizeof(int)* andims);
  ierr=xmp_array_ndims(db, &bndims);
  bwk = (int *)malloc(sizeof(int)* bndims);

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


  ierr = xmp_align_template(db, &dtb);

  descb[0]=1;
  descb[1]=*contxt;

  if (bndims == 1) {
     ierr = xmp_array_lbound(db, 1, &lbound_b1);
     ierr = xmp_array_ubound(db, 1, &ubound_b1);
     descb[2]=ubound_b1-lbound_b1+1;
     descb[3]=1;
  }else if (bndims == 2) {
     ierr = xmp_array_lbound(db, 1, &lbound_b1);
     ierr = xmp_array_ubound(db, 1, &ubound_b1);
     ierr = xmp_array_lbound(db, 2, &lbound_b2);
     ierr = xmp_array_ubound(db, 2, &ubound_b2);
     descb[2]=ubound_b2-lbound_b2+1;
     descb[3]=ubound_b1-lbound_b1+1;
  }

  if (bndims == 1) {
     ierr = xmp_dist_blocksize(dtb, 1, &blocksize_b1);
     descb[4]=blocksize_b1;
     descb[5]=1;
  }else if (bndims == 2) {
     ierr = xmp_align_axis(db, 1, &axis_b1);
     ierr = xmp_align_axis(db, 2, &axis_b2);
     ierr = xmp_dist_blocksize(dtb, axis_b1, &blocksize_b1);
     ierr = xmp_dist_blocksize(dtb, axis_b2, &blocksize_b2);
     descb[4]=blocksize_b2;
     descb[5]=blocksize_b1;
  }

  if (bndims == 1) {
     bwk[0]=0;
     descb[6]=xmp_array_owner(db, bndims, bwk, 1)-1;
     descb[7]=0;
  }else if (bndims == 2) {
     bwk[0]=0;
     bwk[1]=0;
     descb[6]=xmp_array_owner(db, bndims, bwk, 2)-1;
     descb[7]=xmp_array_owner(db, bndims, bwk, 1)-1;
  }

  ierr = xmp_array_lead_dim(db, bwk);
  descb[8]=bwk[bndims-1];

  pdgesv_(n,nrhs,a,ia,ja,desca,ipiv,b,ib,jb,descb,info);

  free(awk);
  free(bwk);

}
