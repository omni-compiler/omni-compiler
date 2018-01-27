#include <stdio.h>
#include <stdlib.h>     

#pragma xmp nodes p(4)
#pragma xmp template t(0:99)
int m[4] = { 10, 20, 30, 40 };
#pragma xmp distribute t(gblock(m)) onto p
int a[100];
#pragma xmp align a[i] with t(i)
#pragma xmp shadow a[1:2]

int main(){

#pragma xmp loop on t(i)
  for (int i = 0; i < 100; i++){
    a[i] = i;
  }

#pragma xmp task on t(20)
  {
    if (a[20] == 20){
      printf("PASS\n");
    }
    else{
      printf("ERROR\n");
      exit(1);
    }
  }

  return 0;
}
