#define N 100

int main()
{
  int i;
  int a[N];

  for(i = 0; i < N; i++) a[i] = i + 1;
  
#pragma acc data copy(a)
  {
#pragma acc kernels if(0)
    for(i=0; i < N; i++){
      a[i] = 1;
    }
  }

  for(i = 0; i < N; i++)
    if(a[i] != i + 1)
      return 1;

#pragma acc data copy(a)
  {
#pragma acc kernels
#pragma acc loop
    for(i=0; i < N; i++){
      a[i] = i;
    }
  }

  for(i = 0; i < N; i++)
    if(a[i] != i)
      return 2;

  return 0;
}
