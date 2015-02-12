#include <stdio.h>
#include <stdlib.h>

#pragma xmp nodes p(*)
#pragma xmp template t(0:9)
#pragma xmp distribute t(block) onto p 

int a[10];

int main(){

  for (int i = 0; i < 10; i++){
    a[i] = i + 1;
  }

  int result = 0;

#pragma xmp loop on t(i) reduction(+:result)
  for (int i = 0; i < 10; i++){
    int n = a[i];
    if (n != a[i]) result = -1;
  }

#pragma xmp task on p(1)
  {
    if(result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;
}
