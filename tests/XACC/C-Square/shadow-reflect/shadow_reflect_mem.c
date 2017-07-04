typedef struct QCDSpinor {
    double v[4][3][2];
} QCDSpinor_t;

#include <xmp.h>
#include <stdio.h>
#define N 10
#pragma xmp nodes p[2][2]
#pragma xmp template t[N][N]
#pragma xmp distribute t[block][block] onto p

void foo()
{
QCDSpinor_t a[N][N][N][N];
#pragma xmp align a[i][j][*][*] with t[i][j]
#pragma xmp shadow a[1][1][0][0]

#pragma acc data copy(a)
{
#pragma xmp reflect (a) width(/periodic/1:1,/periodic/1:1,0,0) acc
}
}

int main()
{
    for(int i=0;i<10000;i++){
	if(xmp_node_num() == 1) fprintf(stderr, "%d\n", i);
	foo();
    }

    if(xmp_node_num() == 1) printf("PASS\n");
    return 0;
}
