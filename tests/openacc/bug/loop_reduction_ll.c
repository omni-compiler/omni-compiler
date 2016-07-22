int main()
{
  int i;
  long long red;

  // sum
  red = -10000;
#pragma acc parallel
#pragma acc loop reduction(+:red)
  for(i=0;i<20000;i++){
    red += 1;
  }
  //verify
  if(red != 10000){
    return 1;
  }

  // mul
  red = 2;
#pragma acc parallel
#pragma acc loop reduction(*:red)
  for(i=0;i<128;i++){
    if(i%8 == 0){
      red *= 2;
    }else{
      red *= 1;
    }
  }
  if(red != 131072){
    return 2;
  }

  // max
  red = 123;
#pragma acc parallel
#pragma acc loop reduction(max:red)
  for(i=0;i<1000000;i++){
    if(red < i){
      red = (long long)i;
    }
  }
  if(red != 999999){
    return 3;
  }

  // min
  red = 33333;
#pragma acc parallel
#pragma acc loop reduction(min:red)
  for(i=0;i<1000000;i++){
    if(red > (long long)i-2){
      red = (long long)i-2;
    }
  }
  if(red != -2){
    return 4;
  }

  // &
  red = 123456789;
#pragma acc parallel
#pragma acc loop reduction(&:red)
  for(i=0;i<1000000;i++){
    red &= (((long long)i << 5) + 21);
  }
  if(red != 21){
    return 5;
  }

  // |
  red = 1065536;
#pragma acc parallel
#pragma acc loop reduction(|:red)
  for(i=0;i<1000000;i++){
    red |= (long long)i;
  }
  if(red != 2097151){
    return 6;
  }

  // ^
  red = 123456;
#pragma acc parallel
#pragma acc loop reduction(^:red)
  for(i=0;i<1000000;i++){
    red ^= (long long)i+1;
  }
  if(red != 958464){
    return 7;
  }

  // &&
  red = 0;
#pragma acc parallel
#pragma acc loop reduction(&&:red)
  for(i=0;i<1000000;i++){
    red = red && 1;
  }
  if(red){
    return 8;
  }

  // ||
  red = 1;
#pragma acc parallel
#pragma acc loop reduction(||:red)
  for(i=0;i<1000000;i++){
    red = red || 0;
  }
  if(! red){
    return 9;
  }


  return 0;
}
