#include <stdio.h>
#include "xmp.h"
#include <stdlib.h>     

#pragma xmp nodes p1(2,4)  
#pragma xmp nodes p2(*,4)  
#pragma xmp nodes p3(4,*)  
#pragma xmp nodes p4(*,*,*)
#pragma xmp nodes p5(4,2,*)

int main(){

  int size1, size2, size3;
  int result = 0;

#pragma xmp task on p1(1,1)
  {
    xmp_nodes_size(xmp_desc_of(p1), 1, &size1);
    xmp_nodes_size(xmp_desc_of(p1), 2, &size2);

    if (size1 != 2 || size2 != 4) result = 1;

    xmp_nodes_size(xmp_desc_of(p2), 1, &size1);
    xmp_nodes_size(xmp_desc_of(p2), 2, &size2);

    if (size1 != 2 || size2 != 4) result = 1;

    xmp_nodes_size(xmp_desc_of(p3), 1, &size1);
    xmp_nodes_size(xmp_desc_of(p3), 2, &size2);

    if (size1 != 4 || size2 != 2) result = 1;

    xmp_nodes_size(xmp_desc_of(p4), 1, &size1);
    xmp_nodes_size(xmp_desc_of(p4), 2, &size2);
    xmp_nodes_size(xmp_desc_of(p4), 3, &size3);

    if (size1 != 2 || size2 != 2 || size3 != 2) result = 1;

    xmp_nodes_size(xmp_desc_of(p5), 1, &size1);
    xmp_nodes_size(xmp_desc_of(p5), 2, &size2);
    xmp_nodes_size(xmp_desc_of(p5), 3, &size3);

    if (size1 != 4 || size2 != 2 || size3 != 1) result = 1;
  }

#pragma xmp task on p1(1,1)
  {
    if (result == 0){
      printf("PASS\n");
    }
    else{
      printf("ERROR\n");
      exit(1);
    }
  }

  return 0;

}
