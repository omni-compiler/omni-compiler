#include <stdio.h>
#include <stdlib.h>

int main(){

#pragma xmp nodes p(*)

#pragma xmp template t(0:63)
#pragma xmp distribute t(block) onto p

  int a[64];
#pragma xmp align a[i] with t(i)
#pragma xmp shadow a[1]

  int b[66];

  int result = 0;

#pragma xmp loop on t(i)
  for (int i = 0; i < 64; i++){
    a[i] = i;
  }

  b[0] = 63;
  for (int i = 1; i < 65; i++){
    b[i] = i - 1;
  }
  b[65] = 0;

#pragma xmp reflect (a) width(/periodic/1:1)

#pragma xmp loop on t(i) reduction(+:result)
  for (int i = 0; i < 64; i++){

    if (a[i-1] != b[i]){
      result = 1;
    }

    if (a[i] != b[i+1]){
      result = 1;
    }

    if (a[i+1] != b[i+2]){
      result = 1;
    }

  }

#pragma xmp task on p(1)
  {
    if (result == 0){
      printf("PASS\n");
    }
    else {
      printf("ERROR\n");
      exit(1);
    }
  }

  return 0;
}
