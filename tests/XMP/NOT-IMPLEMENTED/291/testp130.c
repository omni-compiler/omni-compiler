/*testp130.c*/
/*bcast指示文のテスト:fromはtemplate-ref,onはtemplate-ref全体*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>      
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(1000,1000)
#pragma xmp distribute t(block,*) onto p
int i,j,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,ans,w;
char *result;
int main(void){

   procs = xmp_num_nodes();
   if(N%procs == 0){
      w = N/procs;
   }else{
      w = N/procs+1;
   }
 
   result = "OK";
   for(j=0;j<N;j++){
      a = xmp_node_num();
      b = (double)a;
      c = (float)a;
      for(i=0;i<N;i++){
         aa[i] = a+i-1;
         bb[i] = (double)(a+i-1);
         cc[i] = (float)(a+i-1);
      }
#pragma xmp bcast (a) from t(j,:) on t(:,:)
#pragma xmp bcast (b) from t(j,:) on t(:,:)
#pragma xmp bcast (c) from t(j,:) on t(:,:)
#pragma xmp bcast (aa) from t(j,:) on t(:,:)
#pragma xmp bcast (bb) from t(j,:) on t(:,:)
#pragma xmp bcast (cc) from t(j,:) on t(:,:)
      ans = j/w+1;
      if(a != ans) result = "NG";
      if(b != (double)ans) result = "NG";
      if(c != (float)ans) result = "NG";
      for(i=0;i<N;i++){
         if(aa[i] != ans+i-1) 
            result = "NG";
         if(bb[i] != (double)(ans+i-1))
            result = "NG";
         if(cc[i] != (float)(ans+i-1))
            result = "NG";
      }
   }       
   printf("%d %s %s\n",xmp_node_num(),"testp130.c",result);
   return 0;
}    
         
      
   
