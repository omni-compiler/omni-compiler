#include <stdlib.h>

int main()
{
  int *a;
  int b[1];
  a = (int*)malloc(sizeof(int));

  *a = 123;
  
#pragma acc parallel copy(b)
  b[0] = *a;

  if(b[0] != 123){
    return 1;
  }

  free(a);

  return 0;
}
