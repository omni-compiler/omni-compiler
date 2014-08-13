#include <xmp.h>
#include <stdlib.h>
#include <stdio.h>   
#pragma xmp nodes p(*)

int main(void)
{
  if(xmp_num_nodes() <  4){
    printf("You have to run this program by more than 3 nodes.\n");
    exit(1);
  }

#pragma xmp task on p(1:3)
   {
#pragma xmp barrier
   }

#pragma xmp barrier
    
#pragma xmp task on p(1)
   {
     printf("PASS\n");
   }
   
   return 0;
}
