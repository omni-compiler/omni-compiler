#include <stdio.h>
#define N 4096

static int test_basic()
{
  int array[N];
  int i;
  int sum_0 = 0;
  int sum_1 = 0;
  int sum_2 = 0;
  long long sum_3 = 0;
  int sum_4 = 0;

  for(i = 0; i < N; i++){
    array[i] = i + 1;
  }

  //basic pattern
#pragma acc parallel loop
  for(i = 0; i < N; i++){
#pragma acc atomic
    sum_0 += 2;
#pragma acc atomic update
    sum_1 += i;
#pragma acc atomic update
    sum_2 += array[i];
#pragma acc atomic
    sum_3 += i;
#pragma acc atomic
    sum_4 += (long long)i;
  }

  //check result
  if(sum_0 != (2*N)){
    return 1;
  }
  if(sum_1 != (N*(N-1)/2)){
    return 2;
  }
  if(sum_2 != (N*(N+1)/2)){
    return 3;
  }
  if(sum_3 != (N*(N-1)/2)){
    return 4;
  }
  if(sum_4 != (N*(N-1)/2)){
    return 5;
  }
  return 0;
}

static int test_int()
{
  int i;
  int sums[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    int v;
    v = i - 10;
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (N*(N-1)/2 - 10*N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedint()
{
  int i;
  unsigned int sums[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    unsigned int v;
    v = (unsigned int)i + 10;
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (N*(N-1)/2 + 10*N)){
      return i + 1;
    }
  }
  
  return 0;
}

static int test_long()
{
  int i;
  long sums[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    long v;
    v = (long)i - 10;
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (N*(N-1)/2 - 10*N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedlong()
{
  int i;
  unsigned long sums[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    unsigned long v;
    v = (unsigned long)i;
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (N*(N-1)/2)){
      return i + 1;
    }
  }

  return 0;
}

static int test_longlong()
{
  int i;
  long long sums[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    long long v;
    v = (long long)(-i);
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (- N*(N-1)/2)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedlonglong()
{
  int i;
  unsigned long long sums[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    unsigned long long v;
    v = (unsigned long long)(i + 10);
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (N*(N-1)/2) + 10*N){
      return i + 1;
    }
  }

  return 0;
}

static int test_float()
{
  int i;
  float sums[3] = {0.0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    float v;
    v = (float)(i + 10);
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (N*(N-1)/2) + 10*N){
      return i + 1;
    }
  }

  return 0;
}

static int test_double()
{
  int i;
  float sums[3] = {0.0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    double v;
    v = (double)(-i);
#pragma acc atomic
    sums[0] += v;
#pragma acc atomic
    sums[1] = sums[1] + v;
#pragma acc atomic
    sums[2] = v + sums[2];
  }
  
  for(i = 0; i < 3; i++){
    if(sums[i] != (- N*(N-1)/2)){
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
  if( (r = test_float()) ){
    printf("failed at float(%d)\n", r);
    return 1;
  }
  if( (r = test_double()) ){
    printf("failed at double(%d)\n", r);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
