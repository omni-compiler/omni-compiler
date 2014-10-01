#include <stdio.h>
#include <xmp.h>
int n = 4;
int a[n][n];
#pragma xmp nodes p(2)
#pragma xmp template t(0:n-1,0:n-1)
#pragma xmp distribute t(*,block) onto p
#pragma xmp align a[i][j] with t(i,j)

int main(){
#pragma xmp loop (i,j) on t(i,j)
  for(int j=0; j<n; j++)
    for(int i=0; i<n; i++)
      a[i][j]=i;

  if(xmp_node_num() == 1)
    printf("PASS\n");

  return 0;
}
