#include <stdio.h>
#include <math.h>

#define N 2000000000

int main() {
  double pi = 0.0;
  long long i;

#pragma acc parallel
#pragma acc loop reduction(+:pi)
  for (i = 0; i < N; i++) {
    double t = ((double)i + 0.5) / N;
    pi += 4.0 / (1.0 + t * t);
  }

  pi /= N;

  if( fabs(pi - M_PI) > 1e-10){
    printf("error. result = %lf\n", pi);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
