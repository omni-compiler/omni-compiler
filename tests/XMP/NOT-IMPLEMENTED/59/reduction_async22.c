/*testp106.c*/
/*loop指示文とreduction指示文(*)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>   
#include<stdlib.h>
#pragma xmp nodes p(4,*)
static const int N=1000;
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t1(block,block) onto p
#pragma xmp distribute t2(cyclic,block) onto p
#pragma xmp distribute t3(cyclic,cyclic(5)) onto p
int a[N][N],sa1,sa2;
double b[N][N],sb1,sb2;
float c[N][N],sc1,sc2;
#pragma xmp align a[i][j] with t1(j,i)
#pragma xmp align b[i][j] with t2(j,i)
#pragma xmp align c[i][j] with t3(j,i)
int i,j;
char *result;
int main(void){
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if((i == j)&&((i%100)==0)){
            a[i][j] = 2;
         }else{
            a[i][j] = 1;
         }
      }
   }

#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if((i%2)==0){
            if((j%2)==0){
               b[i][j] = 2.0;
            }else{
               b[i][j] = 1.0;
            }
         }else{
            if((j%2)==1){
               b[i][j] = 0.5;
            }else{
               b[i][j] = 1.0;
            }
         }
      }
   }

#pragma xmp loop (j,i) on t3(j,i)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            if((i%2)==0){
               if((j%4)==0){
                  c[i][j] = 1.0;
               }else if((j%4)==1){
                  c[i][j] = 4.0;
               }else if((j%4)==2){
                  c[i][j] = 1.0;
               }else{
                  c[i][j] = 0.25;
               }
            }else{
               if((j%4)==0){
                  c[i][j] = 0.25;
               }else if((j%4)==1){
                  c[i][j] = 1.0;
               }else if((j%4)==2){
                  c[i][j] = 4.0;
               }else{
                  c[i][j] = 1.0;
               }      
            }
         }
      }

      sa1 = 1;
      sb1 = 1.0;
      sc1 = 1.0;
#pragma xmp loop (i) on t1(:,i)
      for(i=0;i<N;i++){
         sa2 = 1;
#pragma xmp loop (j) on t1(j,i)
         for(j=0;j<N;j++){
            sa2 = sa2*a[i][j];
         }
#pragma xmp reduction(*:sa2) async(1)
#pragma xmp loop (j) on t1(j,i)
         for(j=0;j<N;j++){
            a[i][j] = 0;
         }
#pragma xmp wait_async(1)
         sa1 = sa1*sa2;
      }

#pragma xmp loop (i) on t2(:,i)
      for(i=0;i<N;i++){
         sb2 = 1.0;
#pragma xmp loop (j) on t2(j,i)
         for(j=0;j<N;j++){
            sb2 = sb2*b[i][j];
         }
#pragma xmp reduction(*:sb2) async(2)
#pragma xmp loop (j) on t2(j,i)
         for(j=0;j<N;j++){
            b[i][j] = 0;
         }
#pragma xmp wait_async(1)
         sb1 = sb1*sb2;
      }

#pragma xmp loop (i) on t3(:,i)
      for(i=0;i<N;i++){
         sc2 = 1;
#pragma xmp loop (j) on t3(j,i)
         for(j=0;j<N;j++){
            sc2 = sc2*c[i][j];
         }
#pragma xmp reduction(*:sc2) async(3)
#pragma xmp loop (j) on t3(j,i)
         for(j=0;j<N;j++){
            c[i][j] = 0;
         }
#pragma xmp wait_async(1)
         sc1 = sc1*sc2;
      }

      j = xmp_node_num()%4;
#pragma xmp reduction(*:sa1,sb1,sc1) on p(j,:)
 
      result = "OK";
      if(sa1 != 1024||abs(sb1-1.0)>0.000001||abs(sc1-1.0)>0.0001){
         result = "NG";
      }

#pragma xmp loop (j,i) on t1(j,i)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            if(a[i][j] != 0) result = "NG";
         }
      }

#pragma xmp loop (j,i) on t2(j,i)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            if(b[i][j] != 0.0) result = "NG";
         }
      }

#pragma xmp loop (j,i) on t3(j,i)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            if(c[i][j] != 0.0) result = "NG";
         }
      }

      printf("%d %s %s\n",xmp_node_num(),"testp106.c",result); 
      return 0;
   }
      
         
      
   
   
