#include <stdio.h>
#include <stdlib.h>

int main()
{
  int n = 10;
  int i;
  int *a;
  a = (int*)malloc(sizeof(int)*n);

#pragma acc data copy(a[0:n], n)
  {
#pragma acc parallel loop
    for(i = 0; i < n; i++){
      a[i] = i * i;
    }
  }

  for(i = 0; i < n; i++){
    if(a[i] != i * i){
      return 1;
    }
  }

  free(a);

  printf("PASS\n");
  return 0;
}
