#include<xmp.h>
#include<stdio.h>  
#include<stdlib.h> 
static const int N=1000;
int random_array[1000],ans_val;
#pragma xmp nodes p[*]
#pragma xmp template t1[N][N][N]
#pragma xmp template t2[N][N][N]
#pragma xmp template t3[N][N][N]
#pragma xmp distribute t1[*][*][block] onto p
#pragma xmp distribute t2[*][block][*] onto p
#pragma xmp distribute t3[block][*][*] onto p
int a[N],sa=RAND_MAX;
double b[N],sb=RAND_MAX;
float c[N],sc=RAND_MAX;
int ia=0,ib=0,ic=0,ii=0,i,k,result=0,ans_val=RAND_MAX;
#pragma xmp align a[i] with t1[*][*][i]
#pragma xmp align b[i] with t2[*][i][*]
#pragma xmp align c[i] with t3[i][*][*]

int main(void)
{
  srand(0);
  for(i=0;i<N;i++)
    random_array[i] = rand();
  
#pragma xmp loop on t1[:][:][i]
  for(i=0;i<N;i++)
    a[i] = random_array[i];

#pragma xmp loop on t2[:][i][:]
  for(i=0;i<N;i++)
    b[i] = (double)random_array[i];

#pragma xmp loop on t3[i][:][:]
  for(i=0;i<N;i++)
    c[i] = (float)random_array[i];

  for(i=0;i<N;i++){
    if(ans_val >= random_array[i]){
      ii = i;
      ans_val = random_array[i];
    } 
  }

#pragma xmp loop on t1[:][:][i] reduction(lastmin:sa/ia/)
  for(i=0;i<N;i++){
    if(sa >= a[i]){
      ia = i;
      sa = a[i];
    } 
  }

#pragma xmp loop on t2[:][i][:] reduction(lastmin:sb/ib/)
  for(i=0;i<N;i++){
    if(sb >= b[i]){
      ib = i;
      sb = b[i];
    } 
  }
  
#pragma xmp loop on t3[i][:][:] reduction(lastmin:sc/ic/)
  for(i=0;i<N;i++){
    if(sc >= c[i]){
      ic = i;
      sc = c[i];
    } 
  }
  
  if((sa != ans_val)||(sb != (double)ans_val)||(sc != (float)ans_val)||(ia != ii)||(ib != ii)||(ic != ii))
    result = -1;
  
#pragma xmp reduction(+:result)
#pragma xmp task on p[0]
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
