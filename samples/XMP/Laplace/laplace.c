#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#define N1 64
#define N2 64
double u[N2][N1], uu[N2][N1];

int main(int argc, char **argv)
{
  int j, i, k, niter = 100;
  double value = 0.0;

  for(j = 0; j < N2; j++){
    for(i = 0; i < N1; i++){
      u[j][i] = 0.0;
      uu[j][i] = 0.0;
    }
  }

  for(j = 1; j < N2-1; j++)
    for(i = 1; i < N1-1; i++)
      u[j][i] = sin((double)i/N1*M_PI) + cos((double)j/N2*M_PI);
  
  for(k = 0; k < niter; k++){
    for(j = 1; j < N2-1; j++)
      for(i = 1; i < N1-1; i++)
	uu[j][i] = u[j][i];
    
    for(j = 1; j < N2-1; j++)
      for(i = 1; i < N1-1; i++)
	u[j][i] = (uu[j-1][i] + uu[j+1][i] + uu[j][i-1] + uu[j][i+1])/4.0;
  }
  
  for(j = 1; j < N2-1; j++)
    for(i = 1; i < N1-1; i++)
      value += fabs(uu[j][i] - u[j][i]);

  printf("Verification = %20.16f\n", value);

  return 0;
}
