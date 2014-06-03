/*testp057.c*/
/*reduction指示文(max)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h> 
#include<stdlib.h>   
static const int N=1000;
int random_array[1000000],ans_val,val;
int a[N][N],sa;
double b[N][N],sb;
float c[N][N],sc;
#pragma xmp nodes p(4,*)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
char *result;
int i,k,j,l;
int main(void){

   result = "OK";
   for(k=114;k<10001;k=k+113){
      for(i=1;i<N*N;i++){
         random_array[0]=k;
         random_array[i]=(random_array[i-1]*random_array[i-1])%100000000;
         random_array[i]=(random_array[i]-((random_array[i]%100)/100))%10000;
      }
  
#pragma xmp loop (j,i) on t(j,i)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            l = j*N+i;
            a[i][j] = random_array[l];
            b[i][j] = (double)random_array[l];
            c[i][j] = (float)random_array[l];
         }
      }
      sa = 0;
      sb = 0.0;
      sc = 0.0;
#pragma xmp loop (j,i) on t(j,i)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            if(a[i][j]>sa){
               sa = a[i][j];
            }
            if(b[i][j]>sb){
               sb = b[i][j];
            }
            if(c[i][j]>sc){
               sc = c[i][j];
            }
         }
      }
#pragma xmp reduction(max:sa,sb,sc) async(1)
      ans_val = 0;
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            l = j*N+i;
            if(random_array[l]>ans_val){
               ans_val = random_array[l];
            }
         }
      }
#pragma xmp wait_async(1)

      if(sa != ans_val||sb != (double)ans_val||sc != (float)ans_val){
         result = "NG";
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp057.c",result); 
   return 0;
}
      
         
      
   
