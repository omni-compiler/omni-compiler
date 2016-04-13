#include <xmp.h>
#include <stdio.h>
#pragma xmp nodes p(4)
#define TRUE  1
#define FALSE 0

int main()
{
  int me = xmp_node_num() - 1;
  int k[5][10], m;
  double n = (double)me;

  for(int i=0;i<5;i++)
    for(int j=0;j<10;j++)
      k[i][j] = me;

  switch (me){
  case 0: m = 2;
    break;
  case 1: m = 3;
    break;
  case 2: m = 4; // <- This is selected
    break;
  case 3: m = 4;
    break;
  }
  
#pragma xmp reduction(max:m/n,k[1][:]/)

  // check
  int flag = TRUE;
  
  if(n != (double)2)
    flag == FALSE;

  for(int i=0;i<5;i++){
    for(int j=0;j<10;j++){
      if(i==1){
	if(k[i][j] != 2)
	  flag = FALSE;
      }
      else{
	if(k[i][j] != me)
	  flag = FALSE;
      }
    }
  }

#pragma xmp reduction(min:flag)
  if(flag == FALSE){
#pragma xmp task on p(1)
    printf("ERROR\n");
    return 1;
  }
  else{
#pragma xmp task on p(1)
    printf("PASS\n");
    return 0;
  }
}

