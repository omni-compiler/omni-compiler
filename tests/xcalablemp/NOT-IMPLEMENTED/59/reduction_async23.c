/*testp205.c*/
/*xmp_desc_of()とxmp_gtol()のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
static const int N=1000;
#pragma xmp nodes p(4)
#pragma xmp template t1(N)
#pragma xmp template t2(N)
#pragma xmp template t3(N)
#pragma xmp distribute t1(block) onto p
#pragma xmp distribute t2(cyclic) onto p
#pragma xmp distribute t3(cyclic(3)) onto p
int a[N];
double b[N];
float c[N];
#pragma xmp align a[i] with t1(i)
#pragma xmp align a[i] with t2(i)
#pragma xmp align a[i] with t3(i)
xmp_desc_t da;
xmp_desc_t db;
xmp_desc_t dc;
int g_idx[1];
int l_idx[1];
char *result;     
int i;
int main(void){

   da = xmp_desc_of(a);
   db = xmp_desc_of(b);
   dc = xmp_desc_of(c);

   result = "OK";
  
#pragma xmp reduction(max: tick) async(1)
   for(i=0;i<N;i++){
      g_idx[0] = i;
      xmp_gtol(da,g_idx,l_idx);
      if((i%249) != l_idx[0]) result = "NG1";
       
      xmp_gtol(db,g_idx,l_idx);
      if(i/4 != l_idx[0]) result = "NG2";

      xmp_gtol(dc,g_idx,l_idx);
      if((i/12)*3+(i%3) != l_idx[0]) result = "NG3";
   }                   

   printf("%d %s %s\n",xmp_all_node_num(),"testp205.c",result); 
   return 0;
}
      
         
      
   
   
   
