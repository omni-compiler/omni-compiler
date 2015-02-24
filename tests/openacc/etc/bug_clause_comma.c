int main()
{
  int a[10], b[10], c[10];
  int sum = 0;
  int i;
  
  for(i = 0; i < 10; i++){
    a[i] = i;
    b[i] = 2;
  }
  
#pragma acc data copyin(a,b) , copyout(c)
  {
#pragma acc parallel loop private(i),reduction(+:sum), present(a,b,c)
    for(i=0; i < 10; i++){
      c[i] = a[i] + b[i];
      sum += c[i];
    }
  }

  if(sum != 65 || c[3] != 5){
    return 1;
  }

  return 0;
}
