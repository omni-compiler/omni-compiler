int main()
{
  int a = 1;
  int array[10];
  int i;
  
#pragma acc parallel num_gangs(1)
  {
    a += 2;
#pragma acc loop vector private(a)
    for(i=0;i<10;i++){
      a = array[i];
      a = a*a;
      array[i] = a;
    }
  }

  if(a != 3){
    //printf("a=%d\n",a);
    return 1;
  }
  return 0;
}
