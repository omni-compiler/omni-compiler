#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>     
#pragma xmp nodes p(4,4)
#pragma xmp template t(0:999,:)
#pragma xmp distribute t(gblock(*),gblock(*)) onto p
static const int N=1000;
int *m1,*m2;
int i,j;
int s,remain;
int **a;
#pragma xmp align a[i][j] with t(j,i)
char *result;
int main(void){
   if(xmp_num_nodes()!=16){
      printf("%s\n","You have to run this program by 16 nodes.");
   }

   m1=(int *)malloc(sizeof(int) * 4);
   m2=(int *)malloc(sizeof(int) * 4);
   remain=N;
   for(i=0;i<3;i++){
      m1[i]=remain/2;
      remain =remain-m1[i];
   }
   m1[3]=remain;
   remain =N;
   for(i=0;i<3;i++){
      m2[i]=remain/3;
      remain=remain-m2[i];
   }
   m2[3]=remain;

#pragma xmp template_fix(gblock(m1),gblock(m2)) t(N,N)
   for(i=0;i<N;i++)
      a[i]=(int *)malloc(sizeof(int) * N);

#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j]=xmp_node_num();
      }
   }

   s=0;
#pragma xmp loop (j,i) on t(j,i) reduction(+:s)
   for(i=1;i<N+1;i++){
      for(j=1;j<N+1;j++){
         s+=s+a[i][j];
      }
   }

   result="OK";
   if(s != 45000000){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp008.c",result);
   for(i=0;i<N;i++)
      free(a[i]); 
     
   free(m1);
   free(m2);
   return 0;
}
      
         
      
   
   
