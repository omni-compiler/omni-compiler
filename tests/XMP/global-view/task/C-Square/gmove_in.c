// #489
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <xmp.h>

int (*a)[3];	
//int a[3][3];
int main(int argc, char** argv)
{
  a = (int (*)[3])xmp_malloc(xmp_desc_of(a), 3, 3);
#pragma xmp nodes p[3][1]
#pragma xmp template t[3][3]
#pragma xmp distribute t[block][block] onto p
#pragma xmp align a[i][j] with t[i][j]
  
#pragma xmp loop(i,j) on t[i][j]
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      a[i][j] = j*2+i; 
	
  int myid = xmpc_node_num();
  int b[3], flag = 0;

#ifdef _MPI3
#pragma xmp barrier
  for(int i=0;i<3;i++){
#pragma xmp task on p[i][0]
    {
#pragma xmp gmove in
      b[0:3] = a[0:3][i];
    }
  }

  for(int i=0;i<3;i++)
    if(b[i] != xmpc_node_num()*2 + i)
      flag = 1;

#pragma xmp reduction (+:flag)
  if(flag != 0)
    return 1;
#endif
  return 0;
}
