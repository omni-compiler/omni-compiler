int main()
{
  int i;
  double double_red;

  //float sum
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

  //int mul
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

  //int max
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

  //float min
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

  return 0;
}
