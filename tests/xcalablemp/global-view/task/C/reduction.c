#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>     
static const int N=1000;
#pragma xmp nodes p(4)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int procs,w,a[N][N],sa=0,ans,i,j,result=0;
double b[N][N],sb=0.0;
float c[N][N],sc=0.0;
#pragma xmp align a[*][i] with t(i)
#pragma xmp align b[*][i] with t(i)
#pragma xmp align c[*][i] with t(i)

int main(void)
{
  if(xmp_num_nodes() != 4){
    printf("You have to run this program by 4 nodes.\n");
    exit(1);
  }
  
  procs = xmp_num_nodes();
  if(N%procs==0)
    w = N/procs;
  else
    w = N/procs+1;
  
  for(i=0;i<N;i++){
#pragma xmp loop on t(j)
    for(j=0;j<N;j++){
      a[i][j] = 1;
      b[i][j] = 2.0;
      c[i][j] = 3.0;
    }
  }

#pragma xmp task on p(1)
  {
    for(i=0;i<N;i++){
#pragma xmp loop on t(j)
      for(j=0;j<w;j++){
	sa += a[i][j];
	sb += b[i][j];
	sc += c[i][j];
      }
    }
  }
  
#pragma xmp task on p(2:4)
  {
    if(procs == 4){
      for(i=0;i<N;i++){
#pragma xmp loop on t(j)
	for(j=w;j<N;j++){
	  sa += a[i][j];
	  sb += b[i][j];
	  sc += c[i][j];
	}
      }
    }

#pragma xmp reduction(+:sa,sb,sc)
  }

  result = 0;
  if(xmp_node_num() == 1){
    ans = w*1000;
    if(sa != ans)                           result = -1;
    if(abs(sb-2.0*(double)ans)>0.000000001) result = -1;
    if(abs(sc-3*(float)ans)>0.000001)       result = -1;
   }
  else{
    ans = (N-w)*1000;
    if(sa != ans)                           result = -1;
    if(abs(sb-2.0*(double)ans)>0.000000001) result = -1;
    if(abs(sc-3*(float)ans)>0.000001)       result = -1;
  }
  
#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
  {
    if(result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;
}    
