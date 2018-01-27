#include <stdio.h>

int n = 100;
double a[n][n];
#pragma xmp nodes p(2)
#pragma xmp template t(0:n-1)
#pragma xmp distribute t(block) onto p
#pragma xmp align a[*][j] with t(j)
 
int main(){

  for (int i = 0; i < n; i++)
#pragma xmp loop on t(j)
    for (int j = 0; j < n; j++)
      a[i][j] = i + j;
 
#pragma xmp task on p(1)
  {
    printf("PASS\n");
  }

  return 0;

}
