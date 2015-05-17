#define N 10

void func(int a[])
{
  int i;
#pragma acc data copyout(a[0:N])
  {
#pragma acc parallel loop
    for(i=0;i<N;i++){
      a[i] = 100;
    }
  }
}

int main()
{
  int a[N];
  int i;

  func(a);
  
  for(i=0;i<N;i++){
    if(a[i] != 100){
      return 1;
    }
  }

  return 0;
}
