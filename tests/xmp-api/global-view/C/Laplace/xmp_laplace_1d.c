#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <xmp.h>

#define N1 100
#define N2 200
double u[N2][N1], uu[N2][N1];

#pragma xmp nodes p[*]
#pragma xmp template t[N2][N1]
#pragma xmp distribute t[block][*] onto p
#pragma xmp align u[j][i] with t[j][i]
#pragma xmp align uu[j][i] with t[j][i]
#pragma xmp shadow uu[1:1][0:0]

int main(int argc, char **argv)
{
  int i, j, k, niter = 10;
  double value = 0.0;

#pragma xmp loop (j,i) on t[j][i]
  for(j = 0; j < N2; j++){
    for(i = 0; i < N1; i++){
      u[j][i] = 0.0;
      uu[j][i] = 0.0;
    }
  }

#pragma xmp bcast(value)

#pragma xmp loop (j,i) on t[j][i]
  for(j = 1; j < N2-1; j++)
    for(i = 1; i < N1-1; i++)
      u[j][i] = sin((double)i/N1*M_PI) + cos((double)j/N2*M_PI);
  
  for(k = 0; k < niter; k++){

    // if(xmpc_all_node_num() == 0) printf("iter =%d\n",k);

#pragma xmp loop (j,i) on t[j][i]
    for(j = 1; j < N2-1; j++)
      for(i = 1; i < N1-1; i++)
	uu[j][i] = u[j][i];
    
#pragma xmp reflect (uu)

#pragma xmp loop (j,i) on t[j][i]
    for(j = 1; j < N2-1; j++)
      for(i = 1; i < N1-1; i++)
	u[j][i] = (uu[j-1][i] + uu[j+1][i] + uu[j][i-1] + uu[j][i+1])/4.0;
  
    value = 0.0;
#pragma xmp loop (j,i) on t[j][i] reduction(+:value)
    for(j = 1; j < N2-1; j++)
      for(i = 1; i < N1-1; i++)
	value += fabs(uu[j][i] - u[j][i]);
    //#pragma xmp task on p[0][0]
    if(xmpc_all_node_num() == 0)
      printf("k=%d: Verification = %20.16f\n", k, value);
  }

  return 0;
}
