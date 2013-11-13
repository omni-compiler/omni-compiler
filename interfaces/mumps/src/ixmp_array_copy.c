#include <complex.h>
#include "xmp.h"

void ixmp_array_icopy_(xmp_desc_t **d, int *a){

  int lsize;
  int ierr = xmp_array_lsize(*d,1,&lsize);
  int *la;
  ierr=xmp_array_laddr(*d, &la);

  for(int i=0; i<lsize; i++){
    a[i]=la[i];
  }

}

void ixmp_array_scopy_(xmp_desc_t **d, float *a){

  int lsize;
  int ierr = xmp_array_lsize(*d,1,&lsize);
  float *la;
  ierr=xmp_array_laddr(*d, &la);

  for(int i=0; i<lsize; i++){
    a[i]=la[i];
  }

}

void ixmp_array_dcopy_(xmp_desc_t **d, double *a){

  int lsize;
  int ierr = xmp_array_lsize(*d,1,&lsize);
  double *la;
  ierr=xmp_array_laddr(*d, &la);

  for(int i=0; i<lsize; i++){
    a[i]=la[i];
  }

}

void ixmp_array_ccopy_(xmp_desc_t **d, float _Complex *a){

  int lsize;
  int ierr = xmp_array_lsize(*d,1,&lsize);
  float _Complex *la;
  ierr=xmp_array_laddr(*d, &la);

  for(int i=0; i<lsize; i++){
    a[i]=la[i];
  }

}

void ixmp_array_zcopy_(xmp_desc_t **d, double _Complex *a){

  int lsize;
  int ierr = xmp_array_lsize(*d,1,&lsize);
  double _Complex *la;
  ierr=xmp_array_laddr(*d, &la);

  for(int i=0; i<lsize; i++){
    a[i]=la[i];
  }

}
