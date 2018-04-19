#include <stdio.h>

int main(){
#pragma xmp nodes p(2,2)
#pragma xmp template t(0:7,0:7,0:7)
#pragma xmp distribute t(*,block,block) onto p

  float a[8][8][8];
#pragma xmp align a[*][j][k] with t(*,j,k)

#pragma xmp array on t(:,:,:)
  a[:][:][:] = 0;

#pragma xmp task on p(1,1)
  {
    printf("PASS\n");
  }
  
  return 0;

}
		    
