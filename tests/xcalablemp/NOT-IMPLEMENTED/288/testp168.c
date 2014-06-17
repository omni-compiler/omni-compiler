#include<xmp.h>
#include<stdio.h>  
#include<stdlib.h> 
static const int N=1000;
#pragma xmp nodes p(4,4,*)
#pragma xmp template t1(0:N-1,0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t1(cyclic,cyclic,cyclic) onto p
#pragma xmp distribute t2(cyclic,cyclic(7),cyclic(13)) onto p
#pragma xmp distribute t3(cyclic,cyclic,cyclic(3)) onto p
int a[N][N][N],sa;
double b[N][N][N],sb;
float c[N][N][N],sc;
int i,j,k,result=0;
#pragma xmp align a[i][j][k] with t1(k,j,i)
#pragma xmp align b[i][j][k] with t2(k,j,i)
#pragma xmp align c[i][j][k] with t3(k,j,i)

int main(void){
  if(xmp_num_nodes() < 16){
    printf("%s\n","You have to run this program by more than 16 nodes.");
    exit(1);
  }
  
  sa=0;
  sb=0.0;
  sc=0.0;
   
#pragma xmp loop (k,j,i) on t1(k,j,i) reduction(-:sa)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++){
	a[i][j][k]=1;
	sa = sa-a[i][j][k];
      }

#pragma xmp loop (k,j,i) on t2(k,j,i) reduction(-:sb)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++){
	b[i][j][k]=0.5;
	sb = sb-b[i][j][k];
      }
  
#pragma xmp loop (k,j,i) on t3(k,j,i) reduction(-:sc)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++){
	c[i][j][k]=0.25;
	sc = sc-c[i][j][k];
      }

  if((sa != N*N*N)||(abs(sb-(double)(N*N*N*0.5))) > 0.000001||(abs(sb-(float)(N*N*N*0.25))) > 0.0001){
    result = -1;
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p(1,1,1)
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
