// At parallel construct, variable which is not appeared in enclosing data clause should be treated as firstprivate
#include <stdio.h>
#include <stdlib.h>

int main()
{
  int n = 100;
  int *a;
  int i,j;

  a = (int *)malloc(sizeof(int) * n * n);

#pragma acc kernels loop copyout(a[0:n*n])
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i*n+j] = i * 137 + j;
    }
  }

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      if(a[i*n+j] != i * 137 + j) return 1;
    }
  }
  
  free(a);

  printf("PASS\n");
  return 0;
}
