#define M 10
#define N 100


int main()
{
  
  int i,p;
  double a[N];
  double b[N];

  for(i=0;i<N;i++) a[i] = i;

#pragma acc parallel
  {
#pragma acc loop private(p)
    for(i=0;i<N;i++){
      p = a[i];
      b[i] = p * p;
    }
  }
  for(i=0;i<N;i++){
    if(b[i] != i*i){
      return 1;
    }
  }

  double tmp[N];
  double in[M][N];
  double out[M][N];
  int j;

  for(i=0;i<M;i++){
    for(j=0;j<N;j++){
      in[i][j] = i * 3 + j;
    }
  }

#pragma acc parallel
  {
#pragma acc loop private(tmp) private(j)
    for(i=0;i<M;i++){
#pragma acc loop seq
      for(j=0;j<N;j++){
	tmp[j] = in[i][j];
      }
#pragma acc loop seq
      for(j=0;j<N;j++){
	out[i][j] = tmp[j] * 2;
      }
    }
  }
  for(i=0;i<M;i++){
    for(j=0;j<N;j++){
      if(out[i][j] != i*6+j*2){
	return 2;
      }
    }
  }

  return 0;
}
  
