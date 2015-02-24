int main()
{
  int a[100]; int b[200];
  int i;

  //test 1
#pragma acc enter data create(a)
#pragma acc parallel loop present(a)
  for(i=0;i<100;i++){
    a[i] = i*2;
  }
#pragma acc exit data copyout(a)

  //verify
  for(i=0;i<100;i++){
    if(a[i] != i*2) return 1;
  }

  
  //test 2
  for(i=0;i<200;i++){
    b[i] = i;
  }
  
#pragma acc enter data copyin(b)
#pragma acc parallel loop present(b)
  for(i=0;i<200;i++){
    b[i] += i;
  }
#pragma acc exit data copyout(b)

  for(i=0;i<200;i++){
    if(b[i] != i*2) return 2;
  }

  return 0;
}
