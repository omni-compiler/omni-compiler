/*testp012.c*/
/*array指示文のテスト：多次元分散+全体配列*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>    
static const int N=1000;
int a[N][N];
double b[N][N];
float c[N][N];
char *result;
#pragma xmp nodes p(4,*)
#pragma xmp template t(N,N)
#pragma xmp distribute t(cyclic(2),cyclic(3)) onto p
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)
int i,j;
int main(void){
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j]=j*N+i;
         b[i][j]=(double)(j*N+i);
         c[i][j]=(float)(j*N+i);
      }
   }

#pragma xmp array on t(:,:)
   a[:][:]=a[:][:]+1;
#pragma xmp array on t(:,:)
   b[:][:]=b[:][:]+1;
#pragma xmp array on t(:,:)
   c[:][:]=c[:][:]+1;

   result="OK";
#pragma xmp loop (i) on t(:,i)
   for(i=0;i<N;i++){
#pragma xmp loop (j) on t(j,i)
      for(j=0;j<N;j++){
         if(a[i][j] != j*N+i) result="NG";
         if(b[i][j] != (double)(j*N+i)) result="NG";
         if(c[i][j] != (float)(j*N+i)) result="NG";      
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp012.c",result); 
   return 0;
}
      
         
      
   
         
