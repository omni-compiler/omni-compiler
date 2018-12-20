#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define _FALSE 0
#define _TRUE  1
#pragma xmp nodes p[2][2]
#pragma xmp template t[4][4]
#pragma xmp distribute t[block][block] onto p
double a[4][4];
#pragma xmp align a[i][j] with t[i][j]
#pragma xmp shadow a[0][*]

int main(){
  int n = 4, me = xmpc_node_num(), flag = _TRUE;
  
#pragma xmp loop on t[i][j]
  for(int i=0;i<n;i++)
    for(int j=0;j<n;j++)
      a[i][j] = i*n+j+1;

#pragma xmp reflect (a)
  
#pragma xmp loop (i) on t[i][j]
  for(int i=0;i<n;i++)
    for(int j=0;j<n;j++){
      if(me == 0 || me == 1){
	if(i==0 && j==0)
	  if(a[0][0] != 1) flag = _FALSE;
	if(i==0 && j==1)
	  if(a[0][1] != 2) flag = _FALSE;
	if(i==0 && j==2)
	  if(a[0][2] != 3) flag = _FALSE;
	if(i==0 && j==3)
          if(a[0][3] != 4) flag = _FALSE;
	if(i==1 && j==0)
	  if(a[1][0] != 5) flag = _FALSE;
	if(i==1 && j==1)
          if(a[1][1] != 6) flag = _FALSE;
	if(i==1 && j==2)
          if(a[1][2] != 7) flag = _FALSE;
	if(i==1 && j==3)
          if(a[1][3] != 8) flag = _FALSE;
      }
      else if(me == 2 || me == 3){
	if(i==2 && j==0)
          if(a[2][0] != 9) flag = _FALSE;
        if(i==2 && j==1)
          if(a[2][1] != 10) flag = _FALSE;
        if(i==2 && j==2)
          if(a[2][2] != 11) flag = _FALSE;
        if(i==2 && j==3)
          if(a[2][3] != 12) flag = _FALSE;
        if(i==3 && j==0)
          if(a[3][0] != 13) flag = _FALSE;
        if(i==3 && j==1)
          if(a[3][1] != 14) flag = _FALSE;
        if(i==3 && j==2)
          if(a[3][2] != 15) flag = _FALSE;
        if(i==3 && j==3)
          if(a[3][3] != 16) flag = _FALSE;
      }
    }

#pragma xmp reduction (+:flag)
  if(flag == _FALSE){
    if(me == 0) printf("ERROR !\n");
    exit(1);
  }
  else
    if(me == 0) printf("OK\n");
  
  return 0;
}
