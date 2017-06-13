#include <stdio.h>

#define lx 6
#define ly 6
#define lz 6
#define lt 6

#pragma xmp nodes px[1][1][2]
#pragma xmp nodes py[1][2][1]
#pragma xmp nodes pz[2][1][1]

#pragma xmp template tx[lz][ly][lx]
#pragma xmp template ty[lz][ly][lx]
#pragma xmp template tz[lz][ly][lx]

#pragma xmp distribute tx[block][block][block] onto px
#pragma xmp distribute ty[block][block][block] onto py
#pragma xmp distribute tz[block][block][block] onto pz

double array_x[lz][ly][lx][lt];
double array_y[lz][ly][lx][lt];
double array_z[lz][ly][lx][lt];

#pragma xmp align array_x[k][j][i][*] with tx[k][j][i]
#pragma xmp align array_y[k][j][i][*] with ty[k][j][i]
#pragma xmp align array_z[k][j][i][*] with tz[k][j][i]

#pragma xmp shadow array_x[0:1][0:1][0:1][0]
#pragma xmp shadow array_y[0:1][0:1][0:1][0]
#pragma xmp shadow array_z[0:1][0:1][0:1][0]

int main(void)
{
    int it,ix,iy,iz;

#pragma xmp task on px[0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < 4; ix++){ //0,1,2|3
		for(it = 0; it < lt; it++){
		    array_x[iz][iy][ix][it] = 1.0;
		}
	    }
	}
    }
#pragma xmp task on py[0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < 4; iy++){ //0,1,2|3
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_y[iz][iy][ix][it] = 1.0;
		}
	    }
	}
    }
#pragma xmp task on pz[0][0][0]
    for(iz = 0; iz < 4; iz++){ //0,1,2|3
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_z[iz][iy][ix][it] = 1.0;
		}
	    }
	}
    }

#pragma xmp task on px[0][0][1]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 3; ix < 6; ix++){ //3,4,5|6
		for(it = 0; it < lt; it++){
		    array_x[iz][iy][ix][it] = 2.0;
		}
	    }
	}
    }

#pragma xmp task on py[0][1][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 3; iy < 6; iy++){ //3,4,5|6
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_y[iz][iy][ix][it] = 2.0;
		}
	    }
	}
    }
#pragma xmp task on pz[1][0][0]
    for(iz = 3; iz < 6; iz++){ //3,4,5|6
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_z[iz][iy][ix][it] = 2.0;
		}
	    }
	}
    }

#pragma acc data copy(array_x)
#pragma acc data copy(array_y)
#pragma acc data copy(array_z)
    {
#pragma xmp reflect(array_x) width(0,0,0:1,0) acc
#pragma xmp reflect(array_y) width(0,0:1,0,0) acc
#pragma xmp reflect(array_z) width(0:1,0,0,0) acc
    }

    int err = 0;
#pragma xmp task on px[0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < 4; ix++){ //0,1,2|3
		for(it = 0; it < lt; it++){
		    if(0 <= ix && ix <= 2){
			if(array_x[iz][iy][ix][it] != 1.0) err++;
		    }else{
			if(array_x[iz][iy][ix][it] != 2.0) err++;
		    }
		}
	    }
	}
    }

#pragma xmp reduction(+:err)
    if(err != 0) return 1;

#pragma xmp task on py[0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < 4; iy++){ //0,1,2|3
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    if(0 <= iy && iy <= 2){
			if(array_y[iz][iy][ix][it] != 1.0) err++;
		    }else{
			if(array_y[iz][iy][ix][it] != 2.0) err++;
		    }
		}
	    }
	}
    }

#pragma xmp reduction(+:err)
    if(err != 0) return 2;

#pragma xmp task on pz[0][0][0]
    for(iz = 0; iz < 4; iz++){ //0,1,2|3
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    if(0 <= iz && iz <= 2){
			if(array_z[iz][iy][ix][it] != 1.0) err++;
		    }else{
			if(array_z[iz][iy][ix][it] != 2.0) err++;
		    }
		}
	    }
	}
    }

#pragma xmp reduction(+:err)
    if(err != 0) return 3;

#pragma xmp task on px[0][0][0]
    printf("PASS\n");

    return 0;
}
