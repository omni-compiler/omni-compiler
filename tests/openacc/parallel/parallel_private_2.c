#include <stdio.h>
#include <stdlib.h>

int main()
{
  int n = 100;
  int i, j, j1, j2;
  double *data_in = (double *)malloc(n*n*sizeof(double));
  double *data_out = (double *)malloc(n*n*sizeof(double));
  double *data_host_out = (double *)malloc(n*n*sizeof(double));
  double tmp = 0;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      data_in[i*n+j] = 3 * i + j;
    }
  }

#pragma acc data copyin(data_in[0:n*n]),  copyout(data_out[0:n*n])
  {
#pragma acc parallel loop
    for (j1 = 0; j1 < n; j1++){
#pragma acc loop private(tmp)
      for (j2 = j1; j2 < n; j2++){
	tmp = 0;
#pragma acc loop reduction(+:tmp)
	for (i = 0; i < n; i++){
	  tmp += data_in[i*n+j1] * data_in[i*n+j2];
	}
	data_out[j1*n+j2] = tmp;
	data_out[j2*n+j1] = data_out[j1*n+j2];
      }
    }
  }

  for (j1 = 0; j1 < n; j1++){
    for (j2 = j1; j2 < n; j2++){
      tmp = 0;
      for (i = 0; i < n; i++){
	tmp += data_in[i*n+j1] * data_in[i*n+j2];
      }
      data_host_out[j1*n+j2] = tmp;
      data_host_out[j2*n+j1] = data_host_out[j1*n+j2];
    }
  }

  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(data_out[i*n+j] != data_host_out[i*n+j]){
	printf("%f, %f\n", data_out[i*n+j], data_host_out[i*n+j]);
	return 1;
      }
    }
  }

  printf("PASS\n");

  return 0;
}
