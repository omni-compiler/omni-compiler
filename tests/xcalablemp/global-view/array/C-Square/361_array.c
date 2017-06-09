#include <stdio.h> 
#include <stdlib.h>

int main(){

#pragma xmp nodes p[4]

#pragma xmp template t[100]
#pragma xmp distribute t[block] onto p

  int x[100][100], y[100][100];
#pragma xmp align x[i][*] with t[i]

  int result = 0;

#pragma xmp array on t[:]
  x[:][:] = 0;

#pragma xmp array on t[:99]
  x[0:99][0:99] = 1;

  for (int i = 0; i < 100; i++){
    y[i][:] = 0;
  }

  y[0:99][0:99] = 1;

#pragma xmp loop on t[i]
  for (int i = 0; i < 100; i++){
    for (int j = 0; j < 100; j++){
      if (x[i][j] != y[i][j]) result = -1;
    }
  }

#pragma xmp reduction(+:result)

#pragma xmp task on p[0]
  {
    if (result != 0){
      printf("ERROR\n");
      exit(1);
    }
    else {
      printf("PASS\n");
    }
  }

  return 0;
}
