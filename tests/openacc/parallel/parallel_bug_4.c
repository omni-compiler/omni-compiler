#include <stdio.h>

typedef struct st_1{
  int x;
  int y;
} st_1;


typedef struct st_2{
  st_1 a;
  st_1 b;
} st_2;

int main()
{
  st_1 A = {2,3};
  st_1 B = {4,5};
  st_2 C = {A, B};
  st_2 *C_p = &C;

  int i;
  int data[100];

#pragma acc parallel loop copyin(C_p[0:1])
  for(i=0;i<100;i++){
    data[i] = C_p->a.y;
  }

  for(i=0;i<100;i++){
    if(data[i] != 3){
      return 1;
    }
  }

  printf("PASS\n");

  return 0;
}
