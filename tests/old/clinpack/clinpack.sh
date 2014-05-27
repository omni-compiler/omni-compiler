#gcc -O3 -DUNIX -DDP -DROLL clinpack.c -o clinpack_dpr
#gcc -O3 -DUNIX -DDP -DUNROLL clinpack.c -o clinpack_dpu
#gcc -O3 -DUNIX -DSP -DROLL clinpack.c -o clinpack_spr
#gcc -O3 -DUNIX -DSP -DUNROLL clinpack.c -o clinpack_spu
echo start > clinpack.res
./clinpack_dpr >> clinpack.res
./clinpack_dpu >> clinpack.res
./clinpack_spr >> clinpack.res
./clinpack_spu >> clinpack.res
