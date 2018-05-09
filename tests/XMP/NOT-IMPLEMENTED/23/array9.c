/*testp089*/
/*loop指示文とarray指示文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>      
#pragma xmp nodes p(*)
static const int N=1000;
#pragma xmp template t1(0:N-1,0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t1(*,*,gblock((/325,435,111,129/))) onto p
#pragma xmp distribute t2(*,gblock((/325,435,111,129/)),*) onto p
#pragma xmp distribute t3(gblock((/325,435,111,129/)),*,*) onto p
int a[N];
double b[N];
float c[N];
#pragma xmp align a[i] with t1(*,*,i)
#pragma xmp align b[i] with t2(*,i,*)
#pragma xmp align c[i] with t3(i,*,*)
int i;
char *result;
int main(void){
  
   result = "OK";
#pragma xmp loop on t1(:,:,i)
   for(i=0;i<N;i++){
#pragma xmp array on t1(:,:,i)
      a[i] = i;
   }
#pragma xmp loop on t2(:,:,i)
   for(i=0;i<N;i++){
#pragma xmp array on t2(:,:,i)
      b[i] = (double)i;
   }
#pragma xmp loop on t3(:,:,i)
   for(i=0;i<N;i++){
#pragma xmp array on t3(:,:,i)
      c[i] = (float)i;
   }

#pragma xmp loop on t1(:,:,i)
   for(i=0;i<N;i++){
      if(a[i] != i) result = "NG";
   }
#pragma xmp loop on t2(:,:,i)
   for(i=0;i<N;i++){
      if(b[i] != (double)i) result = "NG";
   }
#pragma xmp loop on t3(:,:,i)
   for(i=0;i<N;i++){
      if(a[i] != (float)i) result = "NG";
   }
   
   printf("%d %s %s\n",xmp_node_num(),"testp089.c",result);
   return 0;
}    
         
      
   
