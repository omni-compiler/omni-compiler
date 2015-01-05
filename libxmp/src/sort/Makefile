sort: xmp_sort_up.o sort.o
	xmpcc -o $@ xmp_sort_up.o sort.o

xmp_sort_up.o: xmp_sort_up.c
	mpicc -c -I ~/work/omni-compiler/libxmp/include/ $< -std=gnu99

sort.o: sort.c
	xmpcc -c sort.c

clean:
	rm -f sort *.o *~
