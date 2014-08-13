#include <math.h>
#include <stdio.h> 
#include <stdlib.h>     

static const double PI = 3.14159265359;

#pragma xmp nodes p(*)
#pragma xmp template t(0:99)
#pragma xmp distribute t(block) onto p

int main(){

  double a0[100], b0[100], c0[100];
#pragma xmp align a0[i] with t(i)
#pragma xmp align b0[i] with t(i)
#pragma xmp align c0[i] with t(i)

  double a[100], b[100], c[100];
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)
#pragma xmp align c[i] with t(i)

  int result = 0;

#pragma xmp loop on t(i)
  for (int i = 0; i < 100; i++){
    a0[i] = i * (2 * PI / 100);
    b0[i] = i * (2 * PI / 100);
    a[i]  = i * (2 * PI / 100);
    b[i]  = i * (2 * PI / 100);
  }

#pragma xmp loop on t(i)
  for (int i = 0; i < 100; i++){
    c0[i] = sin(a0[i] + b0[i]);
  }

#pragma xmp array on t(:)
  c[:] = sin(b[:] + b[:]);

#pragma xmp loop on t(i)
  for (int i = 0; i < 100; i++){
    if (c0[i] != c[i]) result = -1;
  }

#pragma xmp reduction(+:result)

#pragma xmp task on p(1)
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
