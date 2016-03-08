#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define TRUE  1
#define FALSE 0
#pragma xmp nodes p(*)

void output(int flag)
{
  int me = xmp_node_num();
  
#pragma xmp reduction(min:flag)

  if(flag == TRUE){
    if(me == 1)  printf("PASS\n");
  }
  else{
    exit(1);
  }
}

int check1()
{
  int me = xmp_node_num();
  int a[10];
  
  for(int i=0;i<10;i++)
    a[i] = i + (me * 100);

#pragma xmp bcast (a[2:5])

  for(int i=0;i<10;i++)
    if( 2<=i && i<7 ){
      if(a[i] != i + (1 * 100))
	return FALSE;
    }
    else
      if(a[i] != i + (me * 100))
	return FALSE;
  
  return TRUE;
}

int check2()
{
  int me = xmp_node_num();
  int a[5][10];

  for(int i=0;i<5;i++)
    for(int j=0;j<10;j++)
      a[i][j] = i * 10 + j + (me * 100);

  int s = 1, len = 2;
#pragma xmp bcast (a[s:len][:]) from p(2)

  for(int i=0;i<5;i++){
    for(int j=0;j<10;j++){
      if( s<=i && i<(s+len) ){
	if(a[i][j] != (i * 10 + j + (2 * 100)))
	  return FALSE;
      }
      else
	if(a[i][j] != (i * 10 + j + (me * 100)))
	  return FALSE;
    }
  }

  return TRUE;
}

int main()
{
  output(check1());
  output(check2());
  
  return 0;
}

