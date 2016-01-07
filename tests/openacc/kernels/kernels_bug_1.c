#include <stdio.h>

int main()
{

  int n = 50;
  int a[50], b[50];
  int i, j;

  for(i=0;i<n;i++){
    a[i] = 4 * i + 1;
    b[i] = -1;
  }

#pragma acc data copyin(a[0:n]), copy(b[0:n])
  {
#pragma acc kernels
#pragma acc loop independent
    for(i=1;i<n-1;i++){
      b[i] = a[i];
    }
  }

  for(i=1;i<n-1;i++){
    if(b[i] != 4 * i + 1) return 1;
  }
  if(b[0] != -1 || b[n-1] != -1) return 2;

  printf("PASS\n");
  return 0;
}
