/*testp078.c*/
/*task指示文およびarray指示文のテスト*/ 
#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>    
static const int N=1000;
#pragma xmp nodes p(4,4)
#pragma xmp template t(0:N-1,0:N-1,0:N-1)
int m[4] = {100,400,250,250};
#pragma xmp distribute t(gblock(m),gblock(m)) onto p
int a[N][N],sa,ansa;
double b[N][N],sb,ansb;
float c[N][N],sc,ansc;
char *result;
int i,j;
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)
#pragma xmp shadow a[1:1]
#pragma xmp shadow b[2:2]
#pragma xmp shadow c[3:3]
int main(void){

   sa = 0;
   sb = 0.0;
   sc = 0.0;

#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j] = 1;
         b[i][j] = 1.0;
         c[i][j] = 1.0;
      }
   }

#pragma xmp task on p(1,1)
   {
#pragma xmp array on t(:,:)
      a[0,0:99] = a[0,0:99]+1;
#pragma xmp array on t(:,:)
      b[0,0:99] = b[0,0:99]+1;
#pragma xmp array on t(:,:)
      c[0,0:99] = c[0,0:99]+2;
   }
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sa = sa + a[i][j];
         sb = sb + b[i][j];
         sc = sc + c[i][j];
      }
   }
#pragma xmp reduction(+:sa,sb,sc)
   result = "OK";
   ansa = 1000100;
   if(sa != ansa){
      result = "NG";
   }
 
   ansb = 1000100.0;
   if(abs(sb-ansb) > 0.0000001){
      result = "NG";
   }
  
   ansc = 1000200.0;
   if(abs(sc-ansc) > 0.00001){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp078.c",result);
   return 0;
}    
         
      
   
