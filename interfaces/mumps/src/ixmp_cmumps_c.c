#include "stdlib.h"
#include "cmumps_c.h"
#include "xmp.h"

void ixmp_cmumps_c(CMUMPS_STRUC_C *id, xmp_desc_t dirn, xmp_desc_t djcn, xmp_desc_t da){

  int irn_size;
  int jcn_size;
  int a_size;
  void *jcn_laddr;
  void *irn_laddr;
  void *a_laddr;
  int ierr;

  if(id->job==1 || id->job==2 || id->job==4 || id->job==5 || id->job==6){
    ierr = xmp_array_lsize(dirn,1,&irn_size);
    ierr = xmp_array_lsize(djcn,1,&jcn_size);
    ierr = xmp_array_lsize(da,1,&a_size);
    if (irn_size == jcn_size && jcn_size == a_size) {
      id->nz_loc=irn_size;
    }else{
       exit(1);
    }

    ierr=xmp_array_laddr(dirn, &irn_laddr);
    ierr=xmp_array_laddr(djcn, &jcn_laddr);
    ierr=xmp_array_laddr(da, &a_laddr);
    id->irn_loc = (void *)irn_laddr;
    id->jcn_loc = (void *)jcn_laddr;
    id->a_loc = (void *)a_laddr;
  }

  cmumps_c(id);

}
