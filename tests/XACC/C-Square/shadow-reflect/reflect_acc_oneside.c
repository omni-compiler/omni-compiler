#include <stdio.h>

#define ARRAY_SIZE 16

#pragma xmp nodes p[4]
#pragma xmp template t[ARRAY_SIZE]
#pragma xmp distribute t[block] onto p

int array_l[ARRAY_SIZE];
int array_u[ARRAY_SIZE];

#pragma xmp align array_l[i] with t[i]
#pragma xmp align array_u[i] with t[i]
#pragma xmp shadow array_l[1:0]
#pragma xmp shadow array_u[0:1]


int main(void)
{
    int i;

#pragma xmp loop (i) on t[i]
    for(i = 0; i < ARRAY_SIZE; i++){
	array_l[i] = array_u[i] = i * 3 + 5;
    }

#pragma acc data copy(array_l, array_u)
    {
#pragma xmp reflect (array_l, array_u) acc
    }

    char err = 0;

#pragma xmp loop (i) on t[i] reduction(max:err)
    for(i = 1; i < ARRAY_SIZE; i++){
	if(array_l[i-1] != (i-1) * 3 + 5){
	    err = 1;
	}
    }

    if(err){
	return 1;
    }

#pragma xmp loop (i) on t[i] reduction(max:err)
    for(i = 0; i < ARRAY_SIZE-1; i++){
	if(array_u[i+1] != (i+1) * 3 + 5){
	    err = 1;
	}
    }

    if(err){
	return 2;
    }


#pragma xmp task on p[0]
    printf("PASS\n");

    return 0;
}
