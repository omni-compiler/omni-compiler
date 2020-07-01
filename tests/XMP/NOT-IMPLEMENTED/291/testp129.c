/*testp129.c*/
/*bcast指示文のテスト:fromはtemplate-ref,onはnode-refかつ部分ノード*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>      
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t(*,block,*) onto p
int i,j,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,ans,id,w;
char *result;
int main(void){

   id = xmp_node_num();
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
#pragma xmp bcast (a) from t(:,j,:) on t(2:procs-1)
#pragma xmp bcast (b) from t(:,j,:) on t(2:procs-1)
#pragma xmp bcast (c) from t(:,j,:) on t(2:procs-1)
#pragma xmp bcast (aa) from t(:,j,:) on t(2:procs-1)
#pragma xmp bcast (bb) from t(:,j,:) on t(2:procs-1)
#pragma xmp bcast (cc) from t(:,j,:) on t(2:procs-1)
      if((id >= 2)&&(id <= procs-1))
         {
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
         }else{
            if(a != xmp_node_num()) result = "NG";
            if(b != (double)a) result = "NG";
            if(c != (float)a) result = "NG";
            for(i=0;i<N;i++){
               if(aa[i] != a+i-1) 
                  result = "NG";
               if(bb[i] != (double)(a+i-1))
                  result = "NG";
               if(cc[i] != (float)(a+i-1))
                  result = "NG";
            }
         }
   }       
   printf("%d %s %s\n",xmp_node_num(),"testp129.c",result);
   return 0;
}    
         
      
   
