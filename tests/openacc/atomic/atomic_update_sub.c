#include <stdio.h>
#define N 4096

static int test_basic()
{
  int array[N];
  int i;
  int red_0 = 0;
  int red_1 = 0;
  int red_2 = 0;
  long long red_3 = 0;
  int red_4 = 0;

  for(i = 0; i < N; i++){
    array[i] = i + 1;
  }

  //basic pattern
#pragma acc parallel loop copy(red_0,red_1,red_2,red_3,red_4)
  for(i = 0; i < N; i++){
#pragma acc atomic
    red_0 -= 2;
#pragma acc atomic update
    red_1 -= i;
#pragma acc atomic update
    red_2 -= array[i];
#pragma acc atomic
    red_3 -= i;
#pragma acc atomic
    red_4 -= (long long)i;
  }

  //check result
  if(red_0 != (-2*N)){
    return 1;
  }
  if(red_1 != (-N*(N-1)/2)){
    return 2;
  }
  if(red_2 != (-N*(N+1)/2)){
    return 3;
  }
  if(red_3 != (-N*(N-1)/2)){
    return 4;
  }
  if(red_4 != (-N*(N-1)/2)){
    return 5;
  }
  return 0;
}

static int test_int()
{
  int i;
  int reds[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    int v;
    v = i - 10;
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != -(N*(N-1)/2 - 10*N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedint()
{
  int i;
  unsigned int reds[3] = {N*(N-1)/2, N*(N-1)/2};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    unsigned int v;
    v = (unsigned int)i;
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != 0){
      printf("%d, %d\n", reds[i], 0);
      return i + 1;
    }
  }
  
  return 0;
}

static int test_long()
{
  int i;
  long reds[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    long v;
    v = (long)i - 10;
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != -(N*(N-1)/2 - 10*N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedlong()
{
  int i;
  unsigned long reds[3] = {(N*(N-1)/2), (N*(N-1)/2)};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    unsigned long v;
    v = (unsigned long)i;
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != 0){
      return i + 1;
    }
  }

  return 0;
}

static int test_longlong()
{
  int i;
  long long reds[3] = {0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    long long v;
    v = (long long)(i);
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != -(N*(N-1)/2)){
      return i + 1;
    }
  }

  return 0;
}

static int test_unsignedlonglong()
{
  int i;
  unsigned long long reds[3] = {(N*(N-1)/2), (N*(N-1)/2)};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    unsigned long long v;
    v = (unsigned long long)(i);
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != 0){
      return i + 1;
    }
  }

  return 0;
}

static int test_float()
{
  int i;
  float reds[3] = {0.0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    float v;
    v = (float)(i + 10);
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != -(N*(N-1)/2 + 10*N)){
      return i + 1;
    }
  }

  return 0;
}

static int test_double()
{
  int i;
  float reds[3] = {0.0};

#pragma acc parallel loop
  for(i = 0; i < N; i++){
    double v;
    v = (double)(-i);
#pragma acc atomic
    reds[0] -= v;
#pragma acc atomic
    reds[1] = reds[1] - v;
//#pragma acc atomic
//    reds[2] = v - reds[2];
  }
  
  for(i = 0; i < 2; i++){
    if(reds[i] != (N*(N-1)/2)){
      return i;
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
