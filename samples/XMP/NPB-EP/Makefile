is.xmp: EP/ep.c
	sync
	rm -f */*.o
	xmpcc -O3 EP/ep.c common/c_randdp.c common/c_print_results.c common/wtime.c common/c_timers.c -I./common -o bin/ep.xmp.B

clean:	
	rm -f */*.o
