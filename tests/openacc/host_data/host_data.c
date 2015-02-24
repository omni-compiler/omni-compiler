#include<stdlib.h>

int main()
{
  double a;
  double b[10];
  double c[20][30];
  double *d;
  double *p, *q, *r, *s, *t;

  d = (double*)malloc(sizeof(double)*40);

#pragma acc data create(a,b,c, d[0:40])
  {
  
#pragma acc host_data use_device(a)
    p = &a;

#pragma acc host_data use_device(b,c,d)
    {
      q = b;
      r = (double*)c;
      s = &c[2][3];
      s = &b[3];
      t = d;
    }
  }

  return 0;
}

