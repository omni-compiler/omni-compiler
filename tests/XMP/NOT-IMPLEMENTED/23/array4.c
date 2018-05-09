#include <xmp.h>
#include <stdio.h>
#define N 100
int a[N][N], result = 0;
double b[N][N];
float c[N][N];
#pragma xmp nodes p(4,*)
#pragma xmp template t(0:99,0:99)
#pragma xmp distribute t(cyclic(2),cyclic(3)) onto p
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)

int main(void)
{
#pragma xmp loop (j,i) on t(j,i)
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      a[i][j] = j*N+i;
      b[i][j] = (double)(j*N+i);
      c[i][j] = (float)(j*N+i);
    }
  }
  
#pragma xmp array on t(:,:)
  a[:][:] = a[:][:] + 1;
#pragma xmp array on t(:,:)
  b[:][:] = b[:][:] + 1;
#pragma xmp array on t(:,:)
  c[:][:] = c[:][:] + 1;

#pragma xmp loop (i,j) on t(j,i)
   for(int i=0;i<N;i++){
     for(int j=0;j<N;j++){
       if(a[i][j] != j*N+i) result = -1;
       if(b[i][j] != (double)(j*N+i)) result = -1;
       if(c[i][j] != (float)(j*N+i)) result = -1;      
     }
   }
   
   return 0;
}
      
         
      
   
         
