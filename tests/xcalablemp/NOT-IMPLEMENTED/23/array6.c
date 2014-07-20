/*testp014.c*/
/*array指示文のテスト：多次元分散+部分配列+ストライドあり*/
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
#pragma xmp distribute t(block,block) onto p
#pragma xmp align a[i][j] with t(i,j)
#pragma xmp align b[i][j] with t(i,j)
#pragma xmp align c[i][j] with t(i,j)
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
   a[222:556:2][333:112:3] = 0;
#pragma xmp array on t(:)
   b[111:667:3][444:112:4] = 0.0;
#pragma xmp array on t(:)
   c[222:667:4][555:112:5] = 0.0;

   result = "OK";

#pragma xmp loop (j,i) on t(j,i)
   for(i=333;i<445;i=i+3){
      for(j=222;j<778;j=j+2){
         if(a[i][j] != 0){
            result="NG";
         }else{
            a[i][j] = j*N*i;
         }
      }
   }

#pragma xmp loop (j,i) on t(j,i)
   for(i=444;i<556;i=i+4){
      for(j=111;j<778;j=j+3){
         if(b[i][j] != 0){
            result = "NG";
         }else{
            b[i][j] = (double)j*N*i;
         }
      }
   }
#pragma xmp loop (j,i) on t(j,i)
   for(i=555;i<667;i=i+5){
      for(j=222;j<889;j=j+4){
         if(c[i][j] != 0){
            result = "NG";
         }else{
            c[i][j] = (float)j*N*i;
         }
      }
   }

#pragma xmp loop (j,i) on t(j,i)
   for(j=0;j<N;j++){
      for(i=0;i<N;i++){
         if(a[i][j] != j*N+i) result="NG";
         if(b[i][j] != (double)(j*N+i)) result="NG";
         if(c[i][j] != (float)(j*N+i)) result="NG";      
      }
   }
   printf("%d %s %s\n",xmp_node_num(),"testp014.c",result); 
   return 0;
}
      
         
      
   
         
