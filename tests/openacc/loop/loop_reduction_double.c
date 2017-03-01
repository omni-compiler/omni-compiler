#include <stdio.h>

int main()
{
  int i;
  double double_red;

  //sum
  double_red = 3.0;
#pragma acc parallel
#pragma acc loop reduction(+:double_red)
  for(i=0;i<1000000;i++){
    double_red += 1.0;
  }
  //verify
  if(double_red != 1000003){
    return 1;
  }

  //mul
  double_red = 2.0;
#pragma acc parallel
#pragma acc loop reduction(*:double_red)
  for(i=0;i<128;i++){
    if(i%8 == 0){
      double_red *= 2.0;
    }else{
      double_red *= 1.0;
    }
  }
  if(double_red != 131072.0){
    return 2;
  }

  //max
  double_red = 123.0;
#pragma acc parallel
#pragma acc loop reduction(max:double_red)
  for(i=0;i<1000000;i++){
    if(double_red < i){
      double_red = i;
    }
  }
  if(double_red != 999999){
    return 3;
  }

  //min
  double_red = 33333;
#pragma acc parallel
#pragma acc loop reduction(min:double_red)
  for(i=0;i<1000000;i++){
    if(double_red > i-2){
      double_red = i-2.0;
    }
  }
  if(double_red != -2.0){
    return 4;
  }

  //max (result is negative value)
  double_red = -100;
#pragma acc parallel
#pragma acc loop reduction(max:double_red)
  for(i=-1;i>-10000;i--){
    if(double_red < i){
      double_red = i;
    }
  }
  if(double_red != -1){
    return 5;
  }

  printf("PASS\n");

  return 0;
}
