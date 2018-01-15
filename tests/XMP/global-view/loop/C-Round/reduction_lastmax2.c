#include<xmp.h>
#include<stdio.h>  
#include<stdlib.h> 
static const int N=1000;
int random_array[1000000];
#pragma xmp nodes p(4,*)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp distribute t1(cyclic,cyclic) onto p
#pragma xmp distribute t2(block,block) onto p
#pragma xmp distribute t3(cyclic,block) onto p
int a[N][N],sa=0;
double b[N][N],sb=0.0;
float c[N][N],sc=0.0;
int ia=0,ib=0,ic=0,ii=0;
int ja=0,jb=0,jc=0,jj=0;
int i,j,k,m,result=0,ans_val=0;
#pragma xmp align a[i][j] with t1(j,i)
#pragma xmp align b[i][j] with t2(j,i)
#pragma xmp align c[i][j] with t3(j,i)

int main(void)
{
  srand(0);
  for(i=0;i<N*N;i++)
    random_array[i] = rand();

#pragma xmp loop (j,i) on t1(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      m = j*N+i;
      a[i][j] = random_array[m];
    }
  }

#pragma xmp loop (j,i) on t2(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      m = j*N+i;
      b[i][j] = (double)random_array[m];
    }
  }
  
#pragma xmp loop (j,i) on t3(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      m = j*N+i;
      c[i][j] = (float)random_array[m];
    }
  }

  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      m = j*N+i;
      if(ans_val <= random_array[m]){
	ii = i;
	jj = j;
	ans_val = random_array[m];
      } 
    }
  }

#pragma xmp loop (j,i) on t1(j,i) reduction(lastmax:sa/ia,ja/)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if(sa <= a[i][j]){
	ia = i;
	ja = j;
	sa = a[i][j];
      }
    } 
  }

#pragma xmp loop (j,i) on t2(j,i) reduction(lastmax:sb/ib,jb/)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if(sb <= b[i][j]){
	ib = i;
	jb = j;
	sb = b[i][j];
      }
    } 
  }

#pragma xmp loop (j,i) on t3(j,i) reduction(lastmax:sc/ic,jc/)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if(sc <= c[i][j]){
	ic = i;
	jc = j;
	sc = c[i][j];
      }
    } 
  }

  if( (sa != ans_val) || (sb != (double)ans_val) || (sc != (float)ans_val) || (ia != ii) ||
      (ib != ii) || (ic != ii) || (ja != jj) || (jb != jj) || (jc != jj))
    result = -1;

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
