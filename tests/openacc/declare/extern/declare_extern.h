#ifndef _DECLARE_EXTERN_H
#define _DECLARE_EXTERN_H
#define N 100

#include <stdio.h>

extern int a[N];
extern double b[N];
extern float c;
extern long long d[N];

#pragma acc declare create(a,b) copyin(c,d) //copyout, copy is not allowed

void func();

#endif //_DECLARE_EXTERN_H
