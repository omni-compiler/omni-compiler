/*testp066.c*/
/*task指示文とarray指示文の組み合わせ*/ 
#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>     
static const int N=1000;
#pragma xmp nodes p(4)
#pragma xmp template t(0:N-1,0:N-1,0:N-1)
int m[4] = {200,100,400,300};
#pragma xmp distribute t(*,*, gblock(m)) onto p
int a[N],sa,ansa;
double b[N],sb,ansb;
char result;
int i;
#pragma xmp align a[i] with t(*,*,i)
#pragma xmp align b[i] with t(*,*,i)
#pragma xmp shadow a[1:0]
#pragma xmp shadow b[0:1]
int main(void){

   sa = 0;
   sb = 0.0;

#pragma xmp loop on t(*,*,i)
   for(i=0;i<N;i++){
      a[i]=1;
      b[i]=1.0;
   }

#pragma xmp task on p(1)
   {
#pragma xmp array on t(:,:,:)
      a[0:200] = 2;
#pragma xmp array on t(:,:,:)
      b[0:200] = 1.5;
   }
#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      sa = sa + a[i];
      sb = sb + b[i];
   }
#pragma xmp reduction(+:sa)
#pragma xmp reduction(+:sb)

   ansa = 1200;
   ansb = 1000.0+0.5*200;

   result = "OK";
   if(sa != ansa){
      result = "NG";
   }
   if(abs(sb-ansb) > 0.0000000001){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp066.c",result);
   return 0;
}    
         
      
   
