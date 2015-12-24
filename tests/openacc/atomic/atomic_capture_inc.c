#include <stdio.h>
#define N 8192

int a[N];

int test_int()
{
  int i;
  int idx = 0;

#pragma acc parallel loop copyin(idx)
  for(i=0;i<N;i++){
    int my_idx;
#pragma acc atomic capture
    my_idx = idx++;
    a[my_idx] = 1;
  }

  for(i=0;i<N;i++){
    if(a[i] != 1){
      return 1;
    }
  }

  return 0;
}

int test_unsignedint()
{
  int i;
  unsigned int idx = 0;

#pragma acc parallel loop copyin(idx)
  for(i=0;i<N;i++){
    unsigned int my_idx;
#pragma acc atomic capture
    my_idx = idx++;
    a[my_idx] = 2;
  }

  for(i=0;i<N;i++){
    if(a[i] != 2){
      return 1;
    }
  }

  return 0;
}

int test_long()
{
  int i;
  long idx = 0;

#pragma acc parallel loop copyin(idx)
  for(i=0;i<N;i++){
    long my_idx;
#pragma acc atomic capture
    my_idx = idx++;
    a[my_idx] = 3;
  }

  for(i=0;i<N;i++){
    if(a[i] != 3){
      return 1;
    }
  }

  return 0;
}

int test_unsignedlong()
{
  int i;
  unsigned long idx = 0;

#pragma acc parallel loop copyin(idx)
  for(i=0;i<N;i++){
    unsigned long my_idx;
#pragma acc atomic capture
    my_idx = idx++;
    a[my_idx] = 4;
  }

  for(i=0;i<N;i++){
    if(a[i] != 4){
      return 1;
    }
  }

  return 0;
}

int test_longlong()
{
  int i;
  long long idx = 0;

#pragma acc parallel loop copyin(idx)
  for(i=0;i<N;i++){
    long long my_idx;
#pragma acc atomic capture
    my_idx = idx++;
    a[my_idx] = 5;
  }

  for(i=0;i<N;i++){
    if(a[i] != 5){
      return 1;
    }
  }

  return 0;
}

int test_unsignedlonglong()
{
  int i;
  unsigned long long idx = 0;

#pragma acc parallel loop copyin(idx)
  for(i=0;i<N;i++){
    unsigned long long my_idx;
#pragma acc atomic capture
    my_idx = idx++;
    a[my_idx] = 6;
  }

  for(i=0;i<N;i++){
    if(a[i] != 6){
      return 1;
    }
  }

  return 0;
}


int main()
{
  int r;

  if( (r = test_int()) ){
    printf("failed at int(%d)\n", r);
    return 1;
  }
  if( (r = test_unsignedint()) ){
    printf("failed at unsigned int(%d)\n", r);
    return 1;
  }
  if( (r = test_long()) ){
    printf("failed at long(%d)\n", r);
    return 1;
  }
  if( (r = test_unsignedlong()) ){
    printf("failed at unsigned long(%d)\n", r);
    return 1;
  }
  if( (r = test_longlong()) ){
    printf("failed at long long(%d)\n", r);
    return 1;
  }
  if( (r = test_unsignedlonglong()) ){
    printf("failed at unsigned long long(%d)\n", r);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
