#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>     
#pragma xmp nodes p[4][4]
static const int N=1000;
int g[4] = {50,50,50,850};
#pragma xmp template t[N][N]
#pragma xmp distribute t[block][gblock(g)] onto p
int a[N][N],sa=0,ans;
double b[N][N],sb=0.0;
float c[N][N],sc=0.0;
#pragma xmp align a[i][j] with t[i][j]
#pragma xmp align b[i][j] with t[i][j]
#pragma xmp align c[i][j] with t[i][j]
int i,j,result=0;

int main(void)
{
#pragma xmp loop on t[i][j]
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      a[i][j] = 1;
      b[i][j] = 2.0;
      c[i][j] = 3.0;
    }
  }

#pragma xmp loop on t[i][j]
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      sa += a[i][j];
      sb += b[i][j];
      sc += c[i][j];
    }
  }

  ans = (1000/4) * g[(xmp_node_num()-1)%4];
  if(sa != ans){
    result = -1;
  }
  if(abs(sb-2.0*(double)ans)>0.000000001){
    result = -1;
  }
  if(abs(sc-3*(float)ans)>0.000001){
    result = -1;
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p[0][0]
  {
    if(result == 0 ){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;
}    
