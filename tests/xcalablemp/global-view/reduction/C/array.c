#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(3)
#define N 10

void init_array(int a[N]){
  for(int i=0;i<10;i++)
    a[i] = xmp_node_num()*100 + i;
}

int plus(int a[N]){
  init_array(a);
#pragma xmp reduction(+:a)
  for(int i=0;i<N;i++)
    if(a[i] != (100+i)+(200+i)+(300+i))
      return -1;

  return 0;
}

int mult(int a[N]){
  init_array(a);
#pragma xmp reduction(*:a)
  for(int i=0;i<N;i++)
    if(a[i] != (100+i)*(200+i)*(300+i))
      return -1;

  return 0;
}

int max(int a[N]){
  init_array(a);
#pragma xmp reduction(MAX:a)
  for(int i=0;i<N;i++)
    if(a[i] != (300+i))
      return -1;

  return 0;
}

int min(int a[N]){
  init_array(a);
#pragma xmp reduction(MIN:a)
  for(int i=0;i<N;i++)
    if(a[i] != (100+i))
      return -1;

  return 0;
}

int main(){
  int a[N], error = 0;

  error += plus(a);
  error += mult(a);
  error += max(a);
  error += min(a);

#pragma xmp reduction(+:error)
#pragma xmp task on p(1)
  {
    if(error == 0){
      printf("PASS\n");
    }
    else{
      printf("ERROR\n");
    }
  }

  return error;
}
