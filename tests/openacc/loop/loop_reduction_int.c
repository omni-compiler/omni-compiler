int main()
{
  int i;
  int int_red;

  //int sum
  int_red = 100;
#pragma acc parallel
#pragma acc loop reduction(+:int_red)
  for(i=0;i<1000000;i++){
    int_red += 1;
  }
  //verify
  if(int_red != 1000100){
    return 1;
  }

  //int mul
  int_red = 2;
#pragma acc parallel
#pragma acc loop reduction(*:int_red)
  for(i=0;i<128;i++){
    if(i%8 == 0){
      int_red *= 2;
    }else{
      int_red *= 1;
    }
  }
  if(int_red != 131072){
    return 2;
  }

  //int max
  int_red = 123;
#pragma acc parallel
#pragma acc loop reduction(max:int_red)
  for(i=0;i<1000000;i++){
    if(int_red < i){
      int_red = i;
    }
  }
  if(int_red != 999999){
    return 3;
  }

  //int min
  int_red = 33333;
#pragma acc parallel
#pragma acc loop reduction(min:int_red)
  for(i=0;i<1000000;i++){
    if(int_red > i-2){
      int_red = i-2;
    }
  }
  if(int_red != -2){
    return 4;
  }

  //int &
  int_red = 123456789;
#pragma acc parallel
#pragma acc loop reduction(&:int_red)
  for(i=0;i<1000000;i++){
    int_red &= ((i << 5) + 21);
  }
  if(int_red != 21){
    return 5;
  }

  //int |
  int_red = 1065536;
#pragma acc parallel
#pragma acc loop reduction(|:int_red)
  for(i=0;i<1000000;i++){
    int_red |= i;
  }
  if(int_red != 2097151){
    return 6;
  }

  //int ^
  int_red = 123456;
#pragma acc parallel
#pragma acc loop reduction(^:int_red)
  for(i=0;i<1000000;i++){
    int_red ^= i+1;
  }
  if(int_red != 958464){
    return 7;
  }

  //int &&
  int_red = 0;
#pragma acc parallel
#pragma acc loop reduction(&&:int_red)
  for(i=0;i<1000000;i++){
    int_red = int_red && 1;
  }
  if(int_red){
    return 8;
  }

  //int ||
  int_red = 1;
#pragma acc parallel
#pragma acc loop reduction(||:int_red)
  for(i=0;i<1000000;i++){
    int_red = int_red || 0;
  }
  if(! int_red){
    return 9;
  }


  return 0;
}
