#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>

#pragma xmp nodes p[2][2]

static void print_array(int *a, int size1, int size2)
{
    for(int k = 0; k < xmp_num_nodes(); k++){
	if(k == xmp_node_num() - 1){
	    fprintf(stderr, "--- node %d ---\n", xmp_node_num());
	    for(int i = 0; i < size1; i++){
		for(int j = 0; j < size2; j++){
		    fprintf(stderr, "%3d, ", a[i*size2+j]);
		}
		fprintf(stderr, "\n");
	    }
	}
	#pragma xmp barrier
    }
}

int main(void)
{
    int const n = 16;

#pragma xmp template t[n][n]
#pragma xmp distribute t[block][block] onto p

    int a[n][n];
#pragma xmp align a[j][i] with t[j][i]
#pragma xmp shadow a[1][1]

    int *laddr_a, lsize1_a, lsize2_a;
    xmp_array_laddr(xmp_desc_of(a), (void **)&laddr_a);
    xmp_array_lsize(xmp_desc_of(a), 1, &lsize1_a); //includes shadow
    xmp_array_lsize(xmp_desc_of(a), 2, &lsize2_a); //includes shadow

    for(int i = 0; i < lsize1_a; i++){
	for(int j = 0; j < lsize2_a; j++){
	    laddr_a[i * lsize2_a + j] = -1;
	}
    }

#pragma xmp loop (j, i) on t[j][i]
    for(int i = 0; i < n; i++){
	for(int j = 0; j < n; j++){
	    a[i][j] = 1;
	}
    }

#pragma xmp reflect(a)

    int err = 0;

#pragma xmp task on p[0][0]
    {
	for(int i = 0; i < lsize1_a; i++){
	    for(int j = 0; j < lsize2_a; j++){
		int v = laddr_a[i * lsize2_a + j];
		if(i >= 1 && j >= 1){
		    if(v != 1) err++;
		}else{
		    if(v != -1) err++;
		}
	    }
	}
    }
#pragma xmp task on p[0][1]
    {
	for(int i = 0; i < lsize1_a; i++){
	    for(int j = 0; j < lsize2_a; j++){
		int v = laddr_a[i * lsize2_a + j];
		if(i >= 1 && j <= lsize2_a-2){
		    if(v != 1) err++;
		}else{
		    if(v != -1) err++;
		}
	    }
	}
    }
#pragma xmp task on p[1][0]
    {
	for(int i = 0; i < lsize1_a; i++){
	    for(int j = 0; j < lsize2_a; j++){
		int v = laddr_a[i * lsize2_a + j];
		if(i <= lsize1_a-2 && j >= 1){
		    if(v != 1) err++;
		}else{
		    if(v != -1) err++;
		}
	    }
	}
    }
#pragma xmp task on p[1][1]
    {
	for(int i = 0; i < lsize1_a; i++){
	    for(int j = 0; j < lsize2_a; j++){
		int v = laddr_a[i * lsize2_a + j];
		if(i <= lsize1_a-2 && j <= lsize2_a-2){
		    if(v != 1) err++;
		}else{
		    if(v != -1) err++;
		}
	    }
	}
    }

#pragma xmp reduction(+:err)

    if(err > 0) return 1;

#pragma xmp task on p[0][0]
    printf("PASS\n");

    return 0;
}
