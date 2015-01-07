# sort: xmp_sort.o sort.o
# 	xmpcc -o $@ xmp_sort.o sort.o

sort: sort.o
	xmpcc -o $@ sort.o

xmp_sort.o: xmp_sort.c
	mpicc -c -I ~/work/omni-compiler/libxmp/include/ $< -std=gnu99

sort.o: sort.c
	xmpcc -c sort.c

clean:
	rm -f sort *.o *~
