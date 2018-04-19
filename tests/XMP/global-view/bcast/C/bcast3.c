#include <xmp.h>
#include <stdio.h>
#pragma xmp nodes p[*]

int hoge(int n1, int n2)
{
  int a[n1][n2];

  for(int i=0;i<10;i++)
    for(int j=0;j<10;j++)
      a[i][j] = 0;

  for(int i=0;i<10;i++)
    for(int j=0;j<10;j++)
      a[i][j] = xmp_node_num();
      
#pragma xmp bcast (a)

  int sum = 0;
  for(int i=0;i<10;i++)
    for(int j=0;j<10;j++)
      sum += a[i][j];

  int flag = (sum == 100)? 0 : 1;
#pragma xmp reduction(max:flag)
  if(xmp_node_num() == 1){
    if(flag == 0){
      printf("PASS\n");
    }
    else{
      printf("Error!\n");
      xmp_exit(1);
    }
  }
}

int main()
{
  hoge(10, 10);

  return 0;
}
