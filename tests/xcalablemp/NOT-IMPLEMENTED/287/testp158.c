/*testp158.c */
/*loop指示文とreduction節(^)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>  
#include<stdlib.h> 
static const int N=1000;
int random_array[1000],ans_val,val;
int a[N],sa;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic) onto p
int i,k;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){
   result="OK";
   for(k=114;k<10001;k=k+113){
      random_array[0] = k;
      for(i=1;i<N;i++){
         random_array[i]=(random_array[i-1]*random_array[i-1])%100000000;
         random_array[i]=((random_array[i]-(random_array[i]%100))/100)%10000;
      }

#pragma xmp loop (i) on t(i)
      for(i=0;i<N;i++){
         a[i] = random_array[i];
      }
      ans_val = -1;
#pragma xmp loop (i) on t(i) reduction(^:ans_val)
      for(i=0;i<N;i++){
         ans_val = ans_val^a[i];
      }

      sa = -1;
#pragma xmp loop (i) on t(i) reduction(^:sa)
      for(i=0;i<N;i++)
         sa = sa^a[i];       
      if(sa != ans_val){
         result = "NG";
      }
   }
   printf("%d %s %s\n",xmp_node_num(),"testp158.c",result); 
   return 0;
}
      
         
      
   












