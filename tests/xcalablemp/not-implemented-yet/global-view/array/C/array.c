/*testp009.c*/
/*array指示文のテスト：全体配列+ストライドは1*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>    
static const int N=1000;
int a[N][N];
double b[N][N];
float c[N][N];
char *result;
#pragma xmp nodes p(*)
#pragma xmp template t(N)
#pragma xmp distribute t(cyclic(2)) onto p
#pragma xmp align a[*][i] with t(i)
#pragma xmp align b[*][i] with t(i)
#pragma xmp align c[*][i] with t(i)
int i,j;
int main(void){
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j]=j*N+i;
         b[i][j]=(double)(j*N+i);
         c[i][j]=(float)(j*N+i);
      }
   }
#pragma xmp array on t(:)
   a[:][:] = 1;
#pragma xmp array on t(:)
   b[:][:] = 2.0;
#pragma xmp array on t(:)
   c[:][:] = 3.0;
 
   result = "OK";
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(a[i][j]!=1||b[i][j]!=2.0||c[i][j]!=3.0){
            result="NG";
         }
      }
   }

   printf("%d %s %s\n",xmp_all_node_num(),"testp009.c",result); 
   return 0;
}
      
         
      
   
