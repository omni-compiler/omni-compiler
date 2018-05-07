/*testp029.c*/
/*reduction指示文(*)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t(*,*,block) onto p 
int a[N],sa,ansa;
double b[N],sb,ansb;
float c[N],sc,ansc;
char *result;
int i;
#pragma xmp align a[i] with t(*,*,i)
#pragma xmp align b[i] with t(*,*,i)
#pragma xmp align c[i] with t(*,*,i)
int main(void){

   sa=1;
   sb=1.0;
   sc=1.0;
#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      if(i%100 == 0){
         a[i]=2;
         b[i]=2.0;
         c[i]=2.0;
      }else{
         a[i]=1;
         b[i]=1.0;
         c[i]=1.0;
      }
   }
#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      sa = sa*a[i];
   } 
#pragma xmp reduction(*:sa) async(1)

#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      sb = sb*b[i];
   }
#pragma xmp wait_async(1)
   sa = sa*3;

#pragma xmp reduction(*:sb) async(1)

#pragma xmp wait_async(1)
   sb = sb*1.5;

#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      sc = sc*c[i];
   }

#pragma xmp reduction(*:sc) async(1)
#pragma xmp wait_async(1)
   sc = sc*0.5;

   ansa=1024*3;
   ansb=1024.0*1.5;
   ansc=1024.0*0.5;

   result = "OK";
   if(sa != ansa||abs(sb-ansb) > 0.000001||abs(sc-ansc) > 0.000001){ 
      result = "NG";
   }
 
   printf("%d %s %s\n",xmp_node_num(),"testp029.c",result);
   return 0;
}
      
         
      
   
