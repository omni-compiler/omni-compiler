#include <stdio.h>
int a[4][4], b[4][4];
int flag = 0;
#pragma xmp nodes p[2][2]
#pragma xmp template t[4][4]
#pragma xmp distribute t[block][block] onto p
#pragma xmp align a[i][j] with t[i][j]
#pragma xmp shadow a[1][1]

int main(int argc, char** argv)
{
#pragma xmp loop (i,j) on t[i][j]
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      b[i][j] = a[i][j] = i * 4 + j; 
  
#pragma xmp reflect (a) width (/periodic/1, /periodic/1)
    
#pragma xmp loop (i) on t[i][*]
  for(int i=0;i<4;i++){
#pragma xmp loop (j) on t[*][j]
    for(int j=0;j<4;j++)
      if(b[i][j] != a[i][j])
	flag = 1;
  }

#pragma xmp reduction(MAX:flag)
  if(flag == 1)
    return -1;

  return 0;
}
