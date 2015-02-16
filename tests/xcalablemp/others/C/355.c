#include<stdio.h>

int main()
{
double A[22], B[22];
#pragma xmp nodes P(4)
#pragma xmp template TA(0:21)
#pragma xmp template TB(0:21)
int a[4] = {6,8,4,4};
#pragma xmp distribute TA(gblock(a)) onto P
int b[4] = {3,8,5,6};
#pragma xmp distribute TB(gblock(b)) onto P
#pragma xmp align A[i] with TA(i)
#pragma xmp align B[i] with TB(i)

#pragma xmp gmove
B[:] = A[:];
printf("PASS\n");
return 0;
}
