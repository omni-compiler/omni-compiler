#include <stdio.h>
#include <stdlib.h>

int chk_int(int ierr){

#pragma xmp nodes p(*)

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
