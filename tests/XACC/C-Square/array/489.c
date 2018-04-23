#include <stdio.h>
#include <xmp.h>

#define SIZE 100

#pragma xmp nodes p[2][2]
#pragma xmp template t[SIZE][SIZE]
#pragma xmp distribute t[block][block] onto p

int (*g_2d)[SIZE];
int (*g_3d)[SIZE][SIZE];
#pragma xmp align g_2d[i][j] with t[i][j]
#pragma xmp align g_3d[i][j][*] with t[i][j]

int main(){
    int (*l_2d)[SIZE];
    int (*l_3d)[SIZE][SIZE];
#pragma xmp align l_2d[i][j] with t[i][j]
#pragma xmp align l_3d[i][j][*] with t[i][j]

    g_2d = (int (*)[SIZE])xmp_malloc(xmp_desc_of(g_2d), SIZE, SIZE);
    g_3d = (int (*)[SIZE][SIZE])xmp_malloc(xmp_desc_of(g_3d), SIZE, SIZE, SIZE);
    l_2d = (int (*)[SIZE])xmp_malloc(xmp_desc_of(l_2d), SIZE, SIZE);
    l_3d = (int (*)[SIZE][SIZE])xmp_malloc(xmp_desc_of(l_3d), SIZE, SIZE, SIZE);


#pragma acc data copy(g_2d, g_3d, l_2d, l_3d)
    {
#pragma xmp loop [i][j] on t[i][j]
#pragma acc parallel loop collapse(2)
	for(int i = 0; i < SIZE; i++){
	    for(int j = 0; j < SIZE; j++){
		g_2d[i][j] = i * 100 + j + 1;
		g_3d[i][j][0] = i * 100 + j + 2;
		l_2d[i][j] = i * 100 + j + 3;
		l_3d[i][j][0] = i * 100 + j + 4;
	    }
	}
    }

    int err = 0;
#pragma xmp loop [i][j] on t[i][j] reduction(+:err)
    for(int i = 0; i < SIZE; i++){
	for(int j = 0; j < SIZE; j++){
	    if(g_2d[i][j] != i * 100 + j + 1) err++;
	    if(g_3d[i][j][0] != i * 100 + j + 2) err++;
	    if(l_2d[i][j] != i * 100 + j + 3) err++;
	    if(l_3d[i][j][0] != i * 100 + j + 4) err++;
	}
    }

#pragma xmp task on p[0][0]
    if(err != 0){
	printf("# of errors is %d\n", err);
	return 1;
    }else{
	printf("PASS\n");
    }

    return 0;
}
