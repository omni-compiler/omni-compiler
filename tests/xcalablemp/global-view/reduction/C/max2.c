#include<xmp.h>
#include<stdio.h> 
#include<stdlib.h>   
static const int N=1000;
int random_array[1000000],ans_val,val,result=0;
int a[1000][1000],sa;
double b[1000][1000],sb;
float c[1000][1000],sc;
#pragma xmp nodes p(4,*)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
int i,k,j,l;
int main(void)
{
  for(k=114;k<10001;k=k+113){
    random_array[0]=k;
    for(i=1;i<N*N;i++){
      random_array[i]=(random_array[i-1]*random_array[i-1])%100000000;
      random_array[i]=(random_array[i]-((random_array[i]%100)/100))%10000;
    }
    
#pragma xmp loop (j,i) on t(j,i)
    for(i=0;i<N;i++){
      for(j=0;j<N;j++){
	l = j*N+i;
	a[i][j] = random_array[l];
	b[i][j] = (double)random_array[l];
	c[i][j] = (float)random_array[l];
      }
    }
    sa = 0;
    sb = 0.0;
    sc = 0.0;

#pragma xmp loop (j,i) on t(j,i)
    for(i=0;i<N;i++){
      for(j=0;j<N;j++){
	if(a[i][j]>sa){
	  sa = a[i][j];
	}
	if(b[i][j]>sb){
	  sb = b[i][j];
	}
	if(c[i][j]>sc){
	  sc = c[i][j];
	}
      }
    }

#pragma xmp reduction(max:sa,sb,sc) 
    ans_val = 0;
    for(i=0;i<N;i++){
      for(j=0;j<N;j++){
	l = j*N+i;
	if(random_array[l]>ans_val){
	  ans_val = random_array[l];
	}
      }
    }
    
    if(sa != ans_val||sb != (double)ans_val||sc != (float)ans_val){
      result = -1;  // NG
    }
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p(1,1)
  {
    if(result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;
}
