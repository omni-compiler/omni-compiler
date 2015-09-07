int a[10];
double b[100];

#pragma acc declare create(a) copyin(b)

int main()
{
  int i;

#pragma acc parallel loop present(a)
  for(i=0;i<10;i++){
    a[i] = i;
  }

#pragma acc data present(b)
  {
#pragma acc parallel loop
    for(i=0;i<100;i++){
      b[i] = i + 1.0;
    }
  }

  return 0;
}
