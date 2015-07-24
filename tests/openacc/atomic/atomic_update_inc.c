#include <stdio.h>
#define N 8192

static int test_basic()
{
  int i;
  int sum_0 = 0;
  int sum_1 = 0;

  //basic pattern
#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sum_0++;
#pragma acc atomic update
    ++sum_1;
  }

  //check result
  if(sum_0 != (N)){
    return 1;
  }
  if(sum_1 != (N)){
    return 2;
  }
  return 0;
}

static int test_int()
{
  int i;
  int sums[2] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sums[0]++;
#pragma acc atomic
    ++sums[1];
  }
  
  for(i = 0; i < 2; i++){
    if(sums[i] != (N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedint()
{
  int i;
  unsigned int sums[2] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sums[0]++;
#pragma acc atomic
    ++sums[1];
  }
  
  for(i = 0; i < 2; i++){
    if(sums[i] != (N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_long()
{
  int i;
  long sums[2] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sums[0]++;
#pragma acc atomic
    ++sums[1];
  }
  
  for(i = 0; i < 2; i++){
    if(sums[i] != (N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedlong()
{
  int i;
  unsigned long sums[2] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sums[0]++;
#pragma acc atomic
    ++sums[1];
  }
  
  for(i = 0; i < 2; i++){
    if(sums[i] != (N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_longlong()
{
  int i;
  long long sums[2] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sums[0]++;
#pragma acc atomic
    ++sums[1];
  }
  
  for(i = 0; i < 2; i++){
    if(sums[i] != (N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedlonglong()
{
  int i;
  unsigned long long sums[2] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sums[0]++;
#pragma acc atomic
    ++sums[1];
  }
  
  for(i = 0; i < 2; i++){
    if(sums[i] != (N)){
      return i + 1;
    }
  }

  return 0;
}

int main()
{
  int r;
  if( (r = test_basic()) ){
    printf("failed at basic(%d)\n", r);
    return 1;
  }
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
