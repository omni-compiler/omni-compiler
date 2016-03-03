#include <stdio.h>
#pragma xmp nodes p(4)
#pragma xmp template t(0:19)
#pragma xmp distribute t(block) onto p
int a[20], b;
#pragma xmp align a[i] with t(i)
#pragma xmp shadow a[1]

void test_loop()
{
#pragma xmp loop on t(i) profile
  for(int i=0;i<20;i++)
    a[i] = i;
}

void test_task()
{
#pragma xmp task on p(1:2) profile
  {
  }
}

void test_reduction()
{
#pragma xmp reduction(+:b) profile
#if (MPI_VERSION >= 3)
#pragma xmp reduction(+:b) profile async(1)
#pragma xmp reduction(+:b) async(1) profile
#endif
}

void test_gmove()
{
#pragma xmp gmove profile
  a[0] = a[19];

#if (MPI_VERSION >= 3)
#pragma xmp gmove profile async(1)
  a[0] = a[19];

#pragma xmp gmove async(1) profile
  a[0] = a[19];
#endif
}

void test_bcast()
{
#pragma xmp bcast(b) profile
#if (MPI_VERSION >= 3)
#pragma xmp bcast(b) profile async(1)
#pragma xmp bcast(b) async(1) profile
#endif
}

void test_reflect()
{
#pragma xmp reflect(a) profile
#if (MPI_VERSION >= 3)
#pragma xmp reflect(a) profile async(1)
#pragma xmp reflect(a) async(1) profile
#endif
}

void test_barrier()
{
#pragma xmp barrier profile
#if (MPI_VERSION >= 3)
#pragma xmp barrier profile async(1)
#pragma xmp barrier async(1) profile
#endif
}

int main(){
  test_loop();
  test_task();
  test_reduction();
  test_gmove();
  test_bcast();
  test_reflect();
  test_barrier();
  
#pragma xmp task on t(1)
  printf("PASS\n");
 
 return 0;
}
