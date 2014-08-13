#include <xmp.h>
#include <stdio.h>  
#include <stdlib.h> 
static const int N=1000;
#pragma xmp nodes p(4,*)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp distribute t1(block,block) onto p
#pragma xmp distribute t2(cyclic,block) onto p
#pragma xmp distribute t3(cyclic,cyclic(5)) onto p
int a[N][N],sa=1;
double b[N][N],sb=1.0;
float c[N][N],sc=1.0;
int i,j,result=0;
#pragma xmp align a[i][j] with t1(j,i)
#pragma xmp align b[i][j] with t2(j,i)
#pragma xmp align c[i][j] with t3(j,i)

int main(void)
{
#pragma xmp loop (j,i) on t1(j,i)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      if((i==j)&&(i%100==0)){
	a[i][j] = 2;
      }else{
	a[i][j] = 1;
      }

#pragma xmp loop (j,i) on t2(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if(i%2==0){
	if(j%2==0){
	  b[i][j] = 2.0;
	}else{
	  b[i][j] = 1.0;
	}
      }else{
	if((j%2) == 1){
	  b[i][j] = 0.5;
	}else{
	  b[i][j] = 1.0;
	}
      }
    }
  }

#pragma xmp loop (j,i) on t3(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if(i%2==0){
	if(j%4==0){
	  c[i][j] = 1.0;
	}else if((j%4)==1){
	  c[i][j] = 4.0;
	}else if((j%4)==2){
	  c[i][j] = 1.0;
	}else{
	  c[i][j] = 0.25;
	}
      }else{
	if(j%4==0){
	  c[i][j] = 0.25;
	}else if((j%4)==1){
	  c[i][j] = 1.0;
	}else if((j%4)==2){
	  c[i][j] = 4.0;
	}else{
	  c[i][j] = 1.0;
	}
      }
    }
  }
   
#pragma xmp loop (j,i) on t1(j,i) reduction(*:sa)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sa = sa*a[i][j];

#pragma xmp loop (j,i) on t2(j,i) reduction(*:sb)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sb = sb*b[i][j];

#pragma xmp loop (j,i) on t3(j,i) reduction(*:sc)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sc = sc*c[i][j];
 
   if((sa != 1024)||abs(sb-(double)1.0) >= 0.0000001 ||abs(sc-(float)1.0) >= 0.0001)
     result = -1;
   
#pragma xmp reduction(+:result)
#pragma xmp task on p(1,1)
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
