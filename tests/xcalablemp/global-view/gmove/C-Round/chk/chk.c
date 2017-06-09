#include <stdio.h>
#include <stdlib.h>

int chk_int(char *name, int ierr){

#pragma xmp nodes p(*)

#pragma xmp task on p(1)
{
  if ( ierr == 0 ) {
     printf("PASS %s\n",name) ;
  }else{
     printf("ERROR\n");
     exit(1);
  }
}
  return 0;

}
