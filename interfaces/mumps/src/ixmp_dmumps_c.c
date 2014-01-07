#include "stdlib.h"
#include "ixmp_dmumps_c.h"

void ixmp_dmumps_c(ixmp_dmumps_struc_c *id){

  int irn_size;
  int jcn_size;
  int a_size;
  void *jcn_laddr;
  void *irn_laddr;
  void *a_laddr;
  int ierr;

  if((id->mumps_par.icntl[17]==2 || id->mumps_par.icntl[17]==3) && (id->mumps_par.job==1 || id->mumps_par.job==2 || id->mumps_par.job==4 || id->mumps_par.job==5 || id->mumps_par.job==6)){
    ierr = xmp_array_lsize(id->idesc,1,&irn_size);
    ierr = xmp_array_lsize(id->jdesc,1,&jcn_size);
    ierr = xmp_array_lsize(id->adesc,1,&a_size);
    if (irn_size == jcn_size && jcn_size == a_size) {
      id->mumps_par.nz_loc=irn_size;
    }else{
       exit(1);
    }

    ierr=xmp_array_laddr(id->idesc, &irn_laddr);
    ierr=xmp_array_laddr(id->jdesc, &jcn_laddr);
    ierr=xmp_array_laddr(id->adesc, &a_laddr);
    id->mumps_par.irn_loc = (void *)irn_laddr;
    id->mumps_par.jcn_loc = (void *)jcn_laddr;
    id->mumps_par.a_loc = (void *)a_laddr;
  }

  dmumps_c(&(id->mumps_par));

}
