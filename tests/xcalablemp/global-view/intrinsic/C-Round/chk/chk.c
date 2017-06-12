#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

#pragma xmp nodes p(*)

int chk_int(int ierr){

#pragma xmp reduction (max:ierr)
#pragma xmp task on p(1)
{
  if ( ierr == 0 ) {
     printf("PASS\n");
  }else{
     printf("ERROR\n");
     exit(1);
  }
}
  return 0;

}

int chk_int2(int ierr){

#pragma xmp reduction (max:ierr)
  int irank=xmp_node_num();

  if(irank==1){
    if ( ierr == 0 ) {
       printf("PASS\n");
    }else{
       printf("ERROR\n");
       exit(1);
    }
  }
  return 0;
}
