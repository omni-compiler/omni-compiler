#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

int ixmp_sub() {

#pragma xmp nodes p(4)

#pragma xmp task on p(2)
{

  int irank=xmp_node_num();
  if(irank == 1){
    printf("PASS\n");
  }
  else{
    fprintf(stderr, "ERROR rank=%d\n",irank);
    exit(1);
  }
}

  return 0;
}
