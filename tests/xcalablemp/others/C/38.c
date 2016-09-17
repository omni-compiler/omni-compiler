#include <stdio.h>
#pragma xmp nodes p(2)
#pragma xmp template t(0:9)
#pragma xmp distribute t(block) onto p
int a[10];
#pragma xmp align a[i] with t(i)

int main(){
#pragma xmp loop on t(i)
  for(int i=0;i<10;i++){
    for(int j=0;j<2;j++){
      int i2 = i;
      if(i2 != i){
	printf("ERROR\n");
	return 1;
      }
    }
  }
#pragma xmp task on p(1)
  printf("PASS\n");
  return 0;
}
