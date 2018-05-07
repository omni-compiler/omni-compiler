/*testp113.c*/
/*reflect指示文,2次元分散,width,asyncあり*/
#include<xmp.h>
#include<stdio.h>   
#pragma xmp nodes p(4,*)
static const int N=1000;
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
int a[N][N];
double b[N][N];
float c[N][N];
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)
#pragma xmp shadow a[*][*]
#pragma xmp shadow b[1][1]
#pragma xmp shadow c[2][3]
int p2,w1,w2,pi,pj,procs;
char *result;
int i,j,ii,jj;
int main(void){
   result = "OK";
  
   procs = xmp_num_nodes();
   p2 = procs/4;
   w1 = 250;
   if(N%p2 == 0){
      w1 = N/p2;
   }else{
      w2 = N/p2+1;
   }
   pi = xmp_node_num()%4;
   pj = xmp_node_num()/4;
 
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j] = 1;
         b[i][j] = (double)1.0;
         c[i][j] = (float)1.0;
      }
   }
#pragma xmp reflect (a) async(1)
#pragma xmp reflect (b) async(2)
#pragma xmp reflect (c) async(3)

#pragma xmp wait_async(1)
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         for(jj=j-4;jj<j+5;jj++){
            for(ii=i-4;ii<i+5;ii++){
               if(ii < 1||ii > N||jj < 1||jj > N){ 
                  continue;
               }else{
                  if(a[ii][jj] != 1) result = "NG1";
               }
            }
         }
      }
   }
#pragma xmp wait_async(2)
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         for(jj=j-1;jj<j+2;jj++){
            for(ii=i-1;ii<i+2;ii++){
               if(ii < 1||ii > N||jj < 1||jj > N){ 
                  continue;
               }else{
                  if(b[ii][jj] != 1.0) result = "NG2";
               }
            }
         }
      }
   }
#pragma xmp wait_async(3)
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         for(jj=j-3;jj<j+4;jj++){
            for(ii=i-3;ii<i+4;ii++){
               if(ii < 1||ii > N||jj < 1||jj > N){ 
                  continue;
               }else{
                  if(c[ii][jj] != 1.0) result = "NG3";
               }
            }
         }
      }
   }

#pragma xmp loop (j,i) on t(j,i)
   for(j=1;j<N;j++){
      for(i=1;i<N;i++){
         a[i][j] = 2;
         b[i][j] = 2.0;
         c[i][j] = 2.0;
      }
   }

#pragma xmp reflect (a) width(1,1) async(1)
#pragma xmp reflect (b) width(1,1) async(2)
#pragma xmp reflect (c) width(1,2) async(3)

#pragma xmp wait_async(1)
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         for(jj=j-1;jj<j+2;jj++){
            for(ii=i-1;ii<i+2;ii++){
               if(ii < 1||ii > N||jj < 1||jj > N){ 
                  continue;   
               }else{
                  if(c[ii][jj] != 2) result = "NG4";
               }
            }
         }
      }
   }  
#pragma xmp wait_async(2)
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         for(jj=j-1;jj<j+2;jj++){
            for(ii=i-1;ii<i+2;ii++){
               if(ii < 1||ii > N||jj < 1||jj > N){ 
                  continue;
               }else{
                  if(c[ii][jj] != 2.0) result = "NG5";
               }
            }
         }
      }
   }
#pragma xmp wait_async(3)
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         for(jj=j-2;jj<j+3;jj++){
            for(ii=i-2;ii<i+3;ii++){
               if(ii < 1||ii > N||jj < 1||jj > N){ 
                  continue;
               }else{
                  if(c[ii][jj] != 2.0) result = "NG6";
               }
            }
            if(i!=1&&i==pi*w1+1){
               if(c[i-2][j] != 1.0) result = "NG7";
            }
            if(i!=N&&i==(pi+1)*w1){
               if(c[i+2][j] != 1.0) result = "NG8";
            }
            if(j!=1&&j==pj*w2+1){
               if(c[i][j-2] != 1.0) result = "NG9";
            }
            if(j!=1&&j==(pi+1)*w2){
               if(c[i][j+2] != 1.0) result = "NG";
            }
         }
      }
   }               
   printf("%d %s %s\n",xmp_node_num(),"testp113.c",result); 
   return 0;

}
