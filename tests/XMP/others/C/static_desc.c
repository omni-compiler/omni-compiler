#include "stdio.h"

double MPI_Wtime();

void func1(float a[100], float b[100], float c[100]){

#pragma xmp nodes p(4)
#pragma xmp template t(0:99)
#pragma xmp distribute t(block) onto p

#pragma xmp align [i] with t(i) :: a, b, c

#pragma xmp array on t(:)
  a[:] = 0;
#pragma xmp array on t(:)
  b[:] = 0;
#pragma xmp array on t(:)
  c[:] = 0;

#pragma xmp loop on t(i)
  for (int i = 0; i < 100; i++){
    a[i] = a[i] + b[i] * c[i];
  }

}

void func2(float a[100], float b[100], float c[100]){

#pragma xmp nodes p(4)
#pragma xmp static_desc p

#pragma xmp template t(0:99)
#pragma xmp distribute t(block) onto p
#pragma xmp static_desc t

#pragma xmp align [i] with t(i) :: a, b, c
#pragma xmp static_desc :: a, b, c

#pragma xmp array on t(:)
  a[:] = 0;
#pragma xmp array on t(:)
  b[:] = 0;
#pragma xmp array on t(:)
  c[:] = 0;

#pragma xmp loop on t(i)
  for (int i = 0; i < 100; i++){
    a[i] = a[i] + b[i] * c[i];
  }

}

int main(void){

#pragma xmp nodes p(4)

#pragma xmp template t(0:99)
#pragma xmp distribute t(block) onto p

  float a[100], b[100], c[100];
#pragma xmp align [i] with t(i) :: a, b, c

  double t0, t1;

  t0 = MPI_Wtime();
  for (int i = 0; i < 10000; i++){
    func1(a, b, c);
  }
  t1 = MPI_Wtime();
#pragma xmp task on p(1) nocomm
  {
    printf("no static_desc: %f\n", t1 - t0);
  }

  t0 = MPI_Wtime();
  for (int i = 0; i < 10000; i++){
    func2(a, b, c);
  }
  t1 = MPI_Wtime();
#pragma xmp task on p(1) nocomm
  {
    printf("static_desc: %f\n", t1 - t0);
  }

#pragma xmp task on p(1) nocomm
  {
    printf("PASS\n");
  }

  return 0;

}
