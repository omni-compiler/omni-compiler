#include <xmp.h>
#include <stdio.h>  
#include <stdlib.h> 
static const int N=1000;
int random_array[1000];
#pragma xmp nodes p(*)
#pragma xmp template t1(0:N-1,0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t1(block,*,*) onto p
#pragma xmp distribute t2(*,block,*) onto p
#pragma xmp distribute t3(*,*,block) onto p
int a[N],sa=0;
double b[N],sb=0.0;
float c[N],sc=0.0;
int i,k,result=0,ans_val=0;
#pragma xmp align a[i] with t1(i,*,*)
#pragma xmp align b[i] with t2(*,i,*)
#pragma xmp align c[i] with t3(*,*,i)

int main(void)
{
  srand(0);
  for(i=0;i<N;i++)
    random_array[i] = rand();
    
#pragma xmp loop on t1(i,:,:)
  for(i=0;i<N;i++)
    a[i] = random_array[i];

#pragma xmp loop on t2(:,i,:)
  for(i=0;i<N;i++)
    b[i] = random_array[i];
  
#pragma xmp loop on t3(:,:,i)
  for(i=0;i<N;i++)
    c[i] = random_array[i];

  for(i=0;i<N;i++)
    if(random_array[i]>ans_val)
      ans_val=random_array[i];

#pragma xmp loop (i) on t1(i,:,:) reduction(max:sa)
  for(i=0;i<N;i++)
    if(a[i]>sa)
      sa=a[i];
  
#pragma xmp loop (i) on t2(:,i,:) reduction(max:sb)
  for(i=0;i<N;i++)
    if(b[i]>sb)
      sb=b[i];
  
#pragma xmp loop (i) on t3(:,:,i) reduction(max:sc)
  for(i=0;i<N;i++)
    if(c[i]>sc)
      sc=c[i];

  if( (sa != ans_val) || (sb != (double)ans_val) || (sc != (float)ans_val) )
    result = -1;


#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
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
