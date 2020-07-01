#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>   
static const int N=1000;
#pragma xmp nodes p[*][4]
#pragma xmp template t1[N][N]
#pragma xmp template t2[N][N]
#pragma xmp template t3[N][N]
#pragma xmp distribute t1[block][block] onto p
#pragma xmp distribute t2[cyclic][block] onto p
#pragma xmp distribute t3[cyclic][cyclic] onto p
int a[1000][1000],sa=0;
double b[1000][1000],sb=0.0;
float c[1000][1000],sc=0.0;
int i,j, result = 0;
#pragma xmp align a[i][j] with t1[i][j]
#pragma xmp align b[i][j] with t2[i][j]
#pragma xmp align c[i][j] with t3[i][j]

int main(void)
{
#pragma xmp loop on t1[i][j]
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      a[i][j] = 1;
   
#pragma xmp loop on t2[i][j]
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      b[i][j] = 0.5;

#pragma xmp loop on t3[i][j]
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      c[i][j] = 0.25;

#pragma xmp loop on t1[i][j]
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sa = sa+a[i][j];
#pragma xmp reduction (+:sa) async(100)

#pragma xmp loop on t2[i][j]
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sb = sb+b[i][j];
#pragma xmp reduction (+:sb) async(200)

#pragma xmp loop on t3[i][j]
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sc = sc+c[i][j];
#pragma xmp reduction (+:sc) async(300)

#pragma xmp wait_async(100, 200, 300)

  if(sa != N*N||abs(sb-((double)N*N*0.5))>0.000001||abs(sc-((float)N*N*0.25))>0.0001)
    result = -1; // ERROR

#pragma xmp reduction(+:result)
#pragma xmp task on p[0][0]
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
