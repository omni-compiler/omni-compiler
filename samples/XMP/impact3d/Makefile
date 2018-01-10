TARGET = serialh mpi1h mpi2h mpi3h xmp1h xmp2rh xmp3rh

build:
	@(cd share; ./configure)
	@./configure ${TARGET}

clean:
	@for dir in share ${TARGET}; do\
	   (cd $$dir; make clean);\
	done
	@/bin/rm -f lm/*

resclean:
	@/bin/rm -f ./results/*

