/*testp010.c*/
/*array指示文のテスト:部分配列+ストライドは1*/
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
#pragma xmp align a[i][*] with t(i)
#pragma xmp align b[i][*] with t(i)
#pragma xmp align c[i][*] with t(i)
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
   a[222:556][333:112]=0;
#pragma xmp array on t(:)
   b[111:667][444:112]=0.0;
#pragma xmp array on t(:)
   c[222:667][555:112]=0.0;

   result="OK";
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if((i >= 222)&&(i <= 777)&&(j >= 333)&&(j <= 444)){
            if(a[i][j] != 0) result="NG";
         }else{
            if(a[i][j] != j*N+i) result="NG";
         }
         if((i >= 111)&&(i <= 777)&&(j >= 444)&&(j <= 555)){
            if(b[i][j] != 0.0) result="NG";
         }else{
            if(b[i][j] != (double)(j*N+i)) result="NG";
         }
         if((i >= 222)&&(i <= 888)&&(j >= 555)&&(j <= 666)){
            if(a[i][j] != 0) result="NG";
         }else{
            if(a[i][j] != (float)(j*N+i)) result="NG";
         }
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp010.c",result); 
   return 0;
}
      
         
      
   
