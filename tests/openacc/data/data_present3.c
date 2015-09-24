static int test1()
{
  int array[100];
  int i;

  for(i=0;i<100;i++) array[i] = i;
  
#pragma acc enter data copyin(array[10:80])

#pragma acc kernels loop present(array[10:80])
  for(i=10;i<90;i++){
    array[i]++;
  }

#pragma acc exit data copyout(array[10:80])

  for(i=0;i<100;i++){
    if(i>=10 && i < 90){
      if(array[i] != i+1) return 1;
    }else{
      if(array[i] != i) return 2;
    }
  }

  return 0;
}

static void test2_sub(int *a){
  int i;
#pragma acc data present(a[20:25])  /* array[50:25] actually*/
  {
#pragma acc kernels loop
    for(i=20;i<45;i++){
      a[i]++;
    }
  }
}

static int test2()
{
  int array[100];
  int i;

  for(i=0;i<100;i++) array[i] = i;
  
#pragma acc data copy(array[10:80])
  {
  
#pragma acc kernels loop
  for(i=10;i<90;i++) array[i]++;
  
  test2_sub(array + 30);

  }

  for(i=0;i<100;i++){
    if(i < 10){
      if(array[i] != i) return 3;
    }else if(i < 50){
      if(array[i] != i+1) return 4;
    }else if(i < 75){
      if(array[i] != i+2) return 5;
    }else if(i < 90){
      if(array[i] != i+1) return 6;
    }else{
      if(array[i] != i) return 7;
    }
  }

  return 0;
}

int main(){
  int r;
  r = test1();
  if(r) return r;

  r = test2();
  if(r) return r;
}
