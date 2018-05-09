#include <stdio.h>

#define lz 6
#define ly 8
#define lx 10
#define lt 12

#define SL_T 2
#define SU_T 2
#define SL_X 2
#define SU_X 2
#define SL_Y 2
#define SU_Y 2
#define SL_Z 2
#define SU_Z 2

#pragma xmp nodes pt[1][1][1][2]
#pragma xmp nodes px[1][1][2][1]
#pragma xmp nodes py[1][2][1][1]
#pragma xmp nodes pz[2][1][1][1]

#pragma xmp template tt[lz][ly][lx][lt]
#pragma xmp template tx[lz][ly][lx][lt]
#pragma xmp template ty[lz][ly][lx][lt]
#pragma xmp template tz[lz][ly][lx][lt]

#pragma xmp distribute tt[block][block][block][block] onto pt
#pragma xmp distribute tx[block][block][block][block] onto px
#pragma xmp distribute ty[block][block][block][block] onto py
#pragma xmp distribute tz[block][block][block][block] onto pz


int test_dist_t(double array_t[lz][ly][lx][lt], int lst, int ust, int lsx, int usx, int lsy, int usy, int lsz, int usz, int lwt, int uwt)
{
    int it,ix,iy,iz;
    int err = 0;
#pragma xmp align array_t[iz][iy][ix][it] with tt[iz][iy][ix][it]
#pragma xmp shadow array_t[lsz:usz][lsy:usy][lsx:usx][lst:ust]

#pragma xmp task on pt[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt/2 + uwt; it++){ //-1|0,1,2
		    array_t[iz][iy][ix][it] = 1.0;
		}
	    }
	}
    }
#pragma xmp task on pt[0][0][0][1]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = lt/2 - lwt; it < lt; it++){ //2|3,4,5
		    array_t[iz][iy][ix][it] = 2.0;
		}
	    }
	}
    }

#pragma acc data copy(array_t)
    {
#pragma xmp reflect(array_t) width(0      ,0      ,0      ,lwt:uwt) acc
    }

#pragma xmp task on pt[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix <lx; ix++){
		for(it = 0; it < lt/2 + uwt; it++){ //0,1,2|3
		    if(0 <= it && it < lt/2){
			if(array_t[iz][iy][ix][it] != 1.0) err++;
		    }else{
			if(array_t[iz][iy][ix][it] != 2.0) err++;
		    }
		}
	    }
	}
    }
#pragma xmp task on pt[0][0][0][1]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix <lx; ix++){
		for(it = lt/2 - lwt; it < lt; it++){ //2|3,4,5
		    if(lt/2 <= it && it < lt){
			if(array_t[iz][iy][ix][it] != 2.0) err++;
		    }else{
			if(array_t[iz][iy][ix][it] != 1.0) err++;
		    }
		}
	    }
	}
    }

#pragma xmp reduction(+:err)
    if(err == 0) return 0;

#pragma xmp task on pt[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix <lx; ix++){
		for(it = 0; it < lt/2 + uwt; it++){ //0,1,2|3
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 1,iz, iy, ix, it, array_t[iz][iy][ix][it]);
		}
	    }
	}
    }
#pragma xmp task on pt[0][0][0][1]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix <lx; ix++){
		for(it = lt/2 - lwt; it < lt; it++){ //2|3,4,5
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 2,iz, iy, ix, it, array_t[iz][iy][ix][it]);
		}
	    }
	}
    }
    
    return 1;
}

int test_dist_x(double array_x[lz][ly][lx][lt], int lst, int ust, int lsx, int usx, int lsy, int usy, int lsz, int usz, int lwx, int uwx)
{
    int it,ix,iy,iz;
    int err = 0;
#pragma xmp align array_x[iz][iy][ix][it] with tx[iz][iy][ix][it]
#pragma xmp shadow array_x[lsz:usz][lsy:usy][lsx:usx][lst:ust]

#pragma xmp task on px[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx/2 + uwx; ix++){ //-1|0,1,2
		for(it = 0; it < lt; it++){
		    array_x[iz][iy][ix][it] = 1.0;
		}
	    }
	}
    }
#pragma xmp task on px[0][0][1][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = lx/2 - lwx; ix < lx; ix++){ //2|3,4,5
		for(it = 0; it < lt; it++){
		    array_x[iz][iy][ix][it] = 2.0;
		}
	    }
	}
    }

#pragma acc data copy(array_x)
    {
#pragma xmp reflect(array_x) width(0      ,0      ,lwx:uwx,0      ) acc
    }

#pragma xmp task on px[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx/2 + uwx; ix++){ //0,1,2|3
		for(it = 0; it < lt; it++){
		    if(0 <= ix && ix < lx/2){
			if(array_x[iz][iy][ix][it] != 1.0) err++;
		    }else{
			if(array_x[iz][iy][ix][it] != 2.0) err++;
		    }
		}
	    }
	}
    }
#pragma xmp task on px[0][0][1][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = lx/2 - lwx; ix < lx; ix++){ //2|3,4,5
		for(it = 0; it < lt; it++){
		    if(lx/2 <= ix && ix < lx){
			if(array_x[iz][iy][ix][it] != 2.0) err++;
		    }else{
			if(array_x[iz][iy][ix][it] != 1.0) err++;
		    }
		}
	    }
	}
    }

#pragma xmp reduction(+:err)
    if(err == 0) return 0;
	
#pragma xmp task on px[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx/2 + uwx; ix++){ //0,1,2|3
		for(it = 0; it < lt; it++){
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 1,iz, iy, ix, it, array_x[iz][iy][ix][it]);
		}
	    }
	}
    }
#pragma xmp barrier
#pragma xmp task on px[0][0][1][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly; iy++){
	    for(ix = lx/2 - lwx; ix < lx; ix++){ //2|3,4,5
		for(it = 0; it < lt; it++){
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 2, iz, iy, ix, it, array_x[iz][iy][ix][it]);
		}
	    }
	}
    }
    
    return 1;
}

int test_dist_y(double array_y[lz][ly][lx][lt], int lst, int ust, int lsx, int usx, int lsy, int usy, int lsz, int usz, int lwy, int uwy)
{
    int it,ix,iy,iz;
    int err = 0;
#pragma xmp align array_y[iz][iy][ix][it] with ty[iz][iy][ix][it]
#pragma xmp shadow array_y[lsz:usz][lsy:usy][lsx:usx][lst:ust]

#pragma xmp task on py[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly/2 + uwy; iy++){ //-1|0,1,2
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_y[iz][iy][ix][it] = 1.0;
		}
	    }
	}
    }
#pragma xmp task on py[0][1][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = ly/2 - lwy; iy < ly; iy++){ //2|3,4,5
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_y[iz][iy][ix][it] = 2.0;
		}
	    }
	}
    }

#pragma acc data copy(array_y)
    {
#pragma xmp reflect(array_y) width(0      ,lwy:uwy,0      ,0      ) acc
    }

#pragma xmp task on py[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly/2 + uwy; iy++){ //0,1,2|3
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    if(0 <= iy && iy < ly/2){
			if(array_y[iz][iy][ix][it] != 1.0) err++;
		    }else{
			if(array_y[iz][iy][ix][it] != 2.0) err++;
		    }
		}
	    }
	}
    }
#pragma xmp task on py[0][1][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = ly/2 - lwy; iy < ly; iy++){ //2|3,4,5
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    if(ly/2 <= iy && iy < ly){
			if(array_y[iz][iy][ix][it] != 2.0) err++;
		    }else{
			if(array_y[iz][iy][ix][it] != 1.0) err++;
		    }
		}
	    }
	}
    }

#pragma xmp reduction(+:err)
    if(err == 0) return 0;

#pragma xmp task on py[0][0][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = 0; iy < ly/2 + uwy; iy++){ //0,1,2|3
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 1,iz, iy, ix, it, array_y[iz][iy][ix][it]);
		}
	    }
	}
    }
#pragma xmp barrier
#pragma xmp task on py[0][1][0][0]
    for(iz = 0; iz < lz; iz++){
	for(iy = ly/2 - lwy; iy < ly; iy++){ //2|3,4,5
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 2,iz, iy, ix, it, array_y[iz][iy][ix][it]);
		}
	    }
	}
    }

    return 1;
}

int test_dist_z(double array_z[lz][ly][lx][lt], int lst, int ust, int lsx, int usx, int lsy, int usy, int lsz, int usz, int lwz, int uwz)
{
    int it,ix,iy,iz;
    int err = 0;
#pragma xmp align array_z[iz][iy][ix][it] with tz[iz][iy][ix][it]
#pragma xmp shadow array_z[lsz:usz][lsy:usy][lsx:usx][lst:ust]

#pragma xmp task on pz[0][0][0][0]
    for(iz = 0; iz < lz/2 + uwz; iz++){ //-1|0,1,2
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_z[iz][iy][ix][it] = 1.0;
		}
	    }
	}
    }
#pragma xmp task on pz[1][0][0][0]
    for(iz = lz/2 - lwz; iz < lz; iz++){ //2|3,4,5
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    array_z[iz][iy][ix][it] = 2.0;
		}
	    }
	}
    }

#pragma acc data copy(array_z)
    {
#pragma xmp reflect(array_z) width(lwz:uwz,0      ,0      ,0      ) acc
    }

#pragma xmp task on pz[0][0][0][0]
    for(iz = 0; iz < lz/2 + uwz; iz++){ //0,1,2|3
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    if(0 <= iz && iz < lz/2){
			if(array_z[iz][iy][ix][it] != 1.0) err++;
		    }else{
			if(array_z[iz][iy][ix][it] != 2.0) err++;
		    }
		}
	    }
	}
    }
#pragma xmp task on pz[1][0][0][0]
    for(iz = lz/2 - lwz; iz < lz; iz++){ //2|3,4,5
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    if(lz/2 <= iz && iz < lz){
			if(array_z[iz][iy][ix][it] != 2.0) err++;
		    }else{
			if(array_z[iz][iy][ix][it] != 1.0) err++;
		    }
		}
	    }
	}
    }

#pragma xmp reduction(+:err)
    if(err == 0) return 0;

#pragma xmp task on pz[0][0][0][0]
    for(iz = 0; iz < lz/2 + uwz; iz++){ //0,1,2|3
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 1,iz, iy, ix, it, array_z[iz][iy][ix][it]);
		}
	    }
	}
    }
#pragma xmp barrier
#pragma xmp task on pz[1][0][0][0]
    for(iz = lz/2 - lwz; iz < lz; iz++){ //2|3,4,5
	for(iy = 0; iy < ly; iy++){
	    for(ix = 0; ix < lx; ix++){
		for(it = 0; it < lt; it++){
		    printf("node(%d) array[%2d][%2d][%2d][%2d] = %f\n", 2,iz, iy, ix, it, array_z[iz][iy][ix][it]);
		}
	    }
	}
    }

    return 1;
}

int main(void)
{
    int lst, ust, lsx, usx, lsy, usy, lsz, usz;
    int lw, uw;

    for(lst = 0; lst <= SL_T; lst+=2){
    for(ust = 0; ust <= SU_T; ust+=2){
    for(lsx = 0; lsx <= SL_X; lsx+=2){
    for(usx = 0; usx <= SU_X; usx+=2){
    for(lsy = 0; lsy <= SL_Y; lsy+=2){
    for(usy = 0; usy <= SU_Y; usy+=2){
    for(lsz = 0; lsz <= SL_Z; lsz+=2){
    for(usz = 0; usz <= SU_Z; usz+=2){
	double array_t[lz][ly][lx][lt];
	double array_x[lz][ly][lx][lt];
	double array_y[lz][ly][lx][lt];
	double array_z[lz][ly][lx][lt];

#pragma xmp align array_t[iz][iy][ix][it] with tt[iz][iy][ix][it]
#pragma xmp align array_x[iz][iy][ix][it] with tx[iz][iy][ix][it]
#pragma xmp align array_y[iz][iy][ix][it] with ty[iz][iy][ix][it]
#pragma xmp align array_z[iz][iy][ix][it] with tz[iz][iy][ix][it]
#pragma xmp shadow array_t[lsz:usz][lsy:usy][lsx:usx][lst:ust]
#pragma xmp shadow array_x[lsz:usz][lsy:usy][lsx:usx][lst:ust]
#pragma xmp shadow array_y[lsz:usz][lsy:usy][lsx:usx][lst:ust]
#pragma xmp shadow array_z[lsz:usz][lsy:usy][lsx:usx][lst:ust]

        for(lw = 0; lw <= lst; lw++){
	    for(uw = 0; uw <= ust; uw++){
	        if(test_dist_t(array_t, lst, ust, lsx, usx, lsy, usy, lsz, usz, lw, uw)){
#pragma xmp task on px[0][0][0][0] nocomm
		    printf("[%d:%d][%d:%d][%d:%d][%d:%d], width(%d:%d,%d:%d,%d:%d,%d:%d)\n", lsz,usz,lsy,usy,lsx,usx,lst,ust, 0,0,0,0,0,0,lw,uw);
		    return 1;
		}
	    }
        }
	for(lw = 0; lw <= lsx; lw++){
	    for(uw = 0; uw <= usx; uw++){
		if(test_dist_x(array_x, lst, ust, lsx, usx, lsy, usy, lsz, usz, lw, uw)){
#pragma xmp task on px[0][0][0][0] nocomm
		    printf("[%d:%d][%d:%d][%d:%d][%d:%d], width(%d:%d,%d:%d,%d:%d,%d:%d)\n", lsz,usz,lsy,usy,lsx,usx,lst,ust, 0,0,0,0,lw,uw,0,0);
		    return 2;
		}
	    }
	}
	for(lw = 0; lw <= lsy; lw++){
	    for(uw = 0; uw <= usy; uw++){
		if(test_dist_y(array_y, lst, ust, lsx, usx, lsy, usy, lsz, usz, lw, uw)){
#pragma xmp task on px[0][0][0][0] nocomm
		    printf("[%d:%d][%d:%d][%d:%d][%d:%d], width(%d:%d,%d:%d,%d:%d,%d:%d)\n", lsz,usz,lsy,usy,lsx,usx,lst,ust, 0,0,lw,uw,0,0,0,0);
		    return 3;
		}
	    }
	}
	for(lw = 0; lw <= lsz; lw++){
	    for(uw = 0; uw <= usz; uw++){
		if(test_dist_z(array_z, lst, ust, lsx, usx, lsy, usy, lsz, usz, lw, uw)){
#pragma xmp task on px[0][0][0][0] nocomm
		    printf("[%d:%d][%d:%d][%d:%d][%d:%d], width(%d:%d,%d:%d,%d:%d,%d:%d)\n", lsz,usz,lsy,usy,lsx,usx,lst,ust, lw,uw,0,0,0,0,0,0);
		    return 4;
		}
	    }
	}
    }
    }
    }
    }
    }
    }
    }
    }

#pragma xmp task on px[0][0][0][0]
    printf("PASS\n");

    return 0;
}
