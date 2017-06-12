#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define N 6
#define _TRUE  1
#define _FALSE 0
#pragma xmp nodes p[2]
#pragma xmp template t[N]
#pragma xmp distribute t[block] onto p
int a[N];
#pragma xmp align a[i] with t[i]
#pragma xmp shadow a[1]

int main(){
  int flag = _TRUE;
  
  //  check 1
#pragma xmp loop on t[i]
  for(int i=0;i<N;i++)
    a[i] = i;

  if(xmp_node_num() == 1){
    if(a[0] != 0 || a[1] != 1 || a[2] != 2)
      flag = _FALSE;
  }
  else{
    if(a[3] != 3 || a[4] != 4 || a[5] != 5)
      flag = _FALSE;
  }

#pragma xmp reduction(MIN:flag)
  if(flag == _FALSE) exit(1);
  
  // check 2
  int n = 1;
#pragma xmp loop on t[i]
  for(int i=0;i<N;i++)
    a[n]= i;
  
  if(xmp_node_num() == 1){
    if(a[1] != 2)
      flag = _FALSE;
  }
  
#pragma xmp reduction(MIN:flag)
  if(flag == _FALSE) exit(1);

  // check 3
  n = 1;
#pragma xmp loop on t[i]
  for(int i=0;i<N;i++)
    a[i+n]= i;

  if(xmp_node_num() == 1){
    if(a[1] != 0 || a[2] != 1)
      flag = _FALSE;
  }
  else{
    if(a[4] != 3 || a[5] != 4)
      flag = _FALSE;
  }

#pragma xmp reduction(MIN:flag)
  if(flag == _FALSE) exit(1);

#pragma xmp task on p[0]
  printf("PASS\n");
  
  return 0;
}

