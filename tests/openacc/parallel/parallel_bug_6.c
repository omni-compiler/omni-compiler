//test for MEMBER_ARRAY_REF type
#include <stdio.h>
#define N 10

typedef struct mytype {
  double v[N];
} mytype_t;

int main()
{
  mytype_t v2L;

  for(int i = 0; i < N; i++){
    v2L.v[i] = 1.0;
  }

#pragma acc data copy(v2L)
  {
#pragma acc parallel loop
    for(int it = 0; it < N; it++){
      v2L.v[it] = 0;
    }
  }

  for(int i = 0; i < N; i++){
    if(v2L.v[i] != 0){
      return 1;
    }
  }

  printf("PASS\n");
  return 0;
}
