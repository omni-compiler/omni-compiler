#include <stdio.h>
#define N 100

int a[N];
#pragma xmp nodes p[*]
#pragma xmp template t[N]
#pragma xmp distribute t[block] onto p
#pragma xmp align a[i] with t[i]

// same parameter name for global aligned-array
int pow2(int a)
{
  return a * a;
}


void mult(int b[N], int factor)
{
#pragma xmp align b[i] with t[i]
  int i;
#pragma xmp loop (i) on t[i]
  for(i=0;i<N;i++){
    b[i] *= factor;
  }

  {
    // same local name for parameter aligned-array
    int b = 3;
    b *= 2;
  }
}

int func()
{
  int i;
  int c[N];
#pragma xmp align c[i] with t[i]
#pragma xmp loop (i) on t[i]
  for(i=0;i<N;i++){
    c[i] = 0;
  }
  {
    // same local name for local aligned-array
    int c = 4;
    c *= 2;
    return c;
  }
}


int main()
{
  int i, v;
#pragma xmp loop (i) on t[i]
  for(i=0;i<N;i++){
    a[i] = i;
  }

  mult(a, 3);

  long long sum = 0;
#pragma xmp loop (i) on t[i] reduction(+:sum)
  for(i=0;i<N;i++){
    sum += a[i];
  }

  {
    // same local name for global aligned-array
    int a;
    a = pow2(2);
    v = a * a;
  }

#pragma xmp task on p[0]
  {
    if(v != 16){
      printf("Invalid result\n");
      return 1;
    }

    if(sum != N*(N-1)/2*3){
      printf("Invalid result\n");
      return 2;
    }

    printf("OK\n");
  }

  return 0;
}
