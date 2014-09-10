#include <stdio.h>

int main(){
  int i, a = 0;

#pragma omp parallel for reduction(+:a)
  for(i=0;i<10;i++)
    a++;

  printf("PASS\n");
  return 0;
}

